#include "graph_ir.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <variant>

#include "cpu_attention.h"
#include "gguf_structs.h"
#include "graph_ir_expr.h"
#include "graph_ir_mamba.h"
#include "graph_ir_ops.h"
#include "graph_ir_validate.h"
#include "logger.h"
#include "quantization.h"
#include "utils.h"
#include "weight_management.h"
#ifdef HAS_CUDA
#include "cuda_kernels.h"
#include "graph_ir_cuda.h"
#endif

using nlohmann::json;

namespace ember_ir {
namespace {

const std::set<std::string> kCpuOps = {"embedding", "pos_embedding", "rmsnorm", "layernorm", "linear", "split", "rope",
                                       "attention", "add", "mul", "silu", "gelu", "moe_ffn",
                                       "linear_attention", "ssm_scan",
                                       // Mamba-1 (graph_ir_mamba.h)
                                       "conv1d_causal", "selective_scan",
                                       // gated and attention-residual families (graph_ir_ops.h)
                                       "sub", "relu", "sigmoid", "affine", "mul_col", "rms_row", "list", "append", "depth_attend",
                                       // runtime-defined elementwise ops (graph_ir_expr.h) and the per-feature gain
                                       "expr", "scale",
                                       // Mixture-of-Recursions' control node (forward's mor_step runs it)
                                       "depth_route"};
const std::set<std::string> kRecurrentOps = {"linear_attention", "ssm_scan", "conv1d_causal", "selective_scan"};
// The ops run_nodes_device implements.
const std::set<std::string> kCudaOps = {"embedding", "pos_embedding", "rmsnorm", "layernorm", "linear", "split", "rope",
                                        "attention", "add", "mul", "silu", "gelu", "moe_ffn",
                                        "linear_attention", "ssm_scan", "conv1d_causal", "selective_scan",
                                        "sub", "relu", "sigmoid", "affine", "mul_col", "rms_row", "list", "append", "depth_attend",
                                        "expr", "scale"};
const std::set<std::string> kRowWiseOps = {"rmsnorm", "layernorm", "linear", "add", "mul", "silu", "gelu",
                                           "sub", "relu", "sigmoid", "affine", "mul_col", "rms_row", "list", "append", "depth_attend",
                                           "expr", "scale"};

std::vector<std::string> names(const json& j) {
  if (j.is_string()) return {j.get<std::string>()};
  std::vector<std::string> v;
  for (const auto& x : j) v.push_back(x.get<std::string>());
  return v;
}

std::vector<Node> parse_nodes(const json& sub, const char* where) {
  std::vector<Node> nodes;
  for (const auto& n : sub.at("nodes")) {
    Node node{n.at("op").get<std::string>(), names(n.at("in")), names(n.at("out")), n};
    if (!kCpuOps.count(node.op))
      throw std::runtime_error(std::string("IR op '") + node.op + "' (" + where +
                               ") has no CPU kernel in ember's interpreter");
    nodes.push_back(std::move(node));
  }
  return nodes;
}

bool has(const json& n, const char* key) { return n.contains(key) && !n.at(key).is_null(); }

// A per-layer attribute (paperkiln ir.hpp): {"per_layer": [v_0, ..., v_{L-1}]}, block i reading v_i.
bool is_per_layer(const json& v) { return v.is_object() && v.contains("per_layer"); }
const json& at_layer(const json& v, int layer) { return is_per_layer(v) ? v.at("per_layer").at(layer) : v; }

// An integer attribute that is non-zero in some block.
bool any_nonzero(const json& n, const char* key) {
  if (!has(n, key)) return false;
  const json& v = n.at(key);
  if (!is_per_layer(v)) return v.get<int>() != 0;
  for (const auto& e : v.at("per_layer"))
    if (e.get<int>() != 0) return true;
  return false;
}

// ssm_scan names its tensors by attributes a, b, c, d.
std::string ssm_weight(const json& n, const char* part) {
  if (!has(n, part)) throw std::runtime_error(std::string("IR ssm_scan names no '") + part + "' tensor");
  return n.at(part).get<std::string>();
}

// rope on q or k ([T, heads * hd]) for the forms apply_rope_batch_cpu does not
// cover: pairing "neox" (x[j], x[j + n_rot/2]) and partial rotary (the first
// n_rot dims of each head; the rest pass through). freqs: [max_len][n_rot/2]
// (cos, sin), as the interleaved table. Same per-pair arithmetic as
// apply_rope_batch_cpu.
void rope_rows_cpu(std::vector<float>& x, int T, int heads, int hd, int n_rot, bool neox, int start_pos,
                   const std::vector<std::pair<float, float>>& freqs) {
  const int half = n_rot / 2;
  for (int t = 0; t < T; ++t) {
    const auto* f = freqs.data() + static_cast<size_t>(start_pos + t) * half;
    for (int h = 0; h < heads; ++h) {
      float* r = x.data() + (static_cast<size_t>(t) * heads + h) * hd;
      for (int i = 0; i < half; ++i) {
        const size_t i0 = neox ? i : 2 * i, i1 = neox ? i + half : 2 * i + 1;
        const float v0 = r[i0], v1 = r[i1];
        r[i0] = v0 * f[i].first - v1 * f[i].second;
        r[i1] = v0 * f[i].second + v1 * f[i].first;
      }
    }
  }
}

// ---- Third-party IR builder (build_ir_from_gguf) ----

std::optional<double> meta_number(const GGUFData& g, const std::string& key) {
  auto it = g.metadata.find(key);
  if (it == g.metadata.end()) return std::nullopt;
  return std::visit(
      [](const auto& v) -> std::optional<double> {
        using V = std::decay_t<decltype(v)>;
        if constexpr (std::is_arithmetic_v<V> && !std::is_same_v<V, bool>)
          return static_cast<double>(v);
        else
          return std::nullopt;
      },
      it->second);
}

std::optional<std::string> meta_string(const GGUFData& g, const std::string& key) {
  auto it = g.metadata.find(key);
  if (it == g.metadata.end()) return std::nullopt;
  if (const auto* s = std::get_if<std::string>(&it->second)) return *s;
  return std::nullopt;
}

std::optional<bool> meta_bool(const GGUFData& g, const std::string& key) {
  auto it = g.metadata.find(key);
  if (it == g.metadata.end()) return std::nullopt;
  if (const auto* b = std::get_if<bool>(&it->second)) return *b;
  return std::nullopt;
}

// The value a float32 holds, as the shortest decimal that round-trips it,
// read back as a double (paperkiln's ir::f32 / gguf_ir.py f32 convention).
double f32_value(double v) {
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof(buf), static_cast<float>(v));
  return std::strtod(std::string(buf, r.ptr).c_str(), nullptr);
}

// elu(x) + 1 as KimiLinearAttention::feature_map: x + 1 for x > 0, else
// (exp(x) - 1) + 1, each step rounded to f32.
inline float elu_plus_one(float x) {
  float r = x;
  if (r <= 0.f) r = std::exp(r) - 1.f;
  r += 1.f;
  return r;
}

}  // namespace

void linear_attention_row_cpu(const float* q, const float* k, const float* v, float* num, float* den, float* out,
                              int width, float den_floor) {
  // num and den take token t before out[t] is formed (token t sees keys 0..t).
  for (int c = 0; c < width; ++c) {
    const float pk = elu_plus_one(k[c]);
    num[c] = num[c] + pk * v[c];
    den[c] = den[c] + pk;
    out[c] = elu_plus_one(q[c]) * num[c] / std::max(den[c], den_floor);
  }
}

void ssm_scan_row_cpu(const float* u, const float* A, const float* B, const float* C, float D, float* s,
                      float* scratch, float* y, int n) {
  // ops::ssm_scan: all of s_t from s_{t-1}, then y from s_t.
  for (int i = 0; i < n; ++i) {
    float acc = 0;
    for (int j = 0; j < n; ++j) acc += A[static_cast<size_t>(i) * n + j] * s[j];
    acc += B[i] * u[i];
    scratch[i] = acc;
  }
  std::memcpy(s, scratch, static_cast<size_t>(n) * sizeof(float));
  for (int i = 0; i < n; ++i) y[i] = C[i] * s[i] + D * u[i];
}

bool has_recurrent_state(const std::string& ir_json) {
  const json j = json::parse(ir_json);
  if (!j.contains("block")) return false;
  for (const auto& s : j.at("block").value("state", json::array()))
    if (s.value("kind", "") == "recurrent") return true;
  return false;
}

bool has_windowed_attention(const std::string& ir_json) {
  const json j = json::parse(ir_json);
  if (!j.contains("block")) return false;
  const json& blk = j.at("block");
  for (const auto& s : blk.value("state", json::array()))
    if (any_nonzero(s, "window")) return true;
  for (const auto& n : blk.value("nodes", json::array()))
    if (n.value("op", "") == "attention" && any_nonzero(n, "window")) return true;
  return false;
}

bool has_state_tokens(const std::string& ir_json) {
  const json j = json::parse(ir_json);
  for (const char* sub : {"embed", "block", "head"}) {
    if (!j.contains(sub)) continue;
    for (const auto& n : j.at(sub).value("nodes", json::array())) {
      const std::string op = n.value("op", "");
      if (op == "embedding" && has(n, "state_weight")) return true;
      if (op == "attention" && n.contains("mask") && n.at("mask") != "causal") return true;
    }
  }
  if (j.contains("block"))
    for (const auto& s : j.at("block").value("state", json::array()))
      if (has(s, "evict")) return true;
  return false;
}

bool device_supported(const std::string& ir_json) {
  // Per-turn state tokens have no device kernel yet: the CPU serves them.
  if (has_state_tokens(ir_json)) return false;
  const json j = json::parse(ir_json);
  // depth_attend reads at most every tensor ever put in a list (list inputs
  // and appends, the block's once per repeat) plus its own extra inputs.
  // Per-layer attributes (the ones the interpreter accepts: attention and
  // kv-state window and sinks, RoPE theta and freq_scale) run on either path.
  size_t list_tensors = 0, extra = 0;
  for (const char* sub : {"embed", "block", "head"}) {
    if (!j.contains(sub)) continue;
    const size_t times = std::string(sub) == "block" ? j.at(sub).value("repeat", 1) : 1;
    for (const auto& n : j.at(sub).value("nodes", json::array())) {
      const std::string op = n.value("op", "");
      if (!kCudaOps.count(op)) return false;
      if (op == "expr") {  // the CUDA form's limits (graph_ir_expr.h lower)
        try {
          expr::lower(expr::cached(n.at("expr").get<std::string>(), n.at("in").size()));
        } catch (const std::exception&) {
          return false;
        }
      }
      const size_t ins = n.contains("in") && n.at("in").is_array() ? n.at("in").size() : 1;
      if (op == "list") list_tensors += ins * times;
      if (op == "append") list_tensors += times;
      if (op == "depth_attend") extra = std::max(extra, ins - 1);
#ifdef HAS_CUDA
      if (op == "ssm_scan" && n.value("n", 0) > kIrSsmMax) return false;
#endif
    }
  }
#ifdef HAS_CUDA
  if (list_tensors + extra > static_cast<size_t>(kIrDepthSourcesMax)) return false;
#endif
  return true;
}

bool has_ir_builder(const std::string& arch) {
  static const std::set<std::string> kBuilt = {"llama", "mistral", "qwen2", "qwen3", "gpt2", "gptneox", "gemma3",
                                               "mamba"};
  return kBuilt.count(arch) > 0;
}

std::string find_ir(const GGUFData& gguf) {
  if (const char* path = std::getenv("EMBER_IR")) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("EMBER_IR: cannot read ") + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }
  auto it = gguf.metadata.find("paperkiln.ir");
  if (it != gguf.metadata.end()) {
    if (const auto* s = std::get_if<std::string>(&it->second)) return *s;
  }
  // No IR in the file: every built architecture gets one from its metadata,
  // except that llama and mistral without a sliding window keep the llama path
  // unless EMBER_IR_BUILD=1 (a window needs the IR: the llama path has none).
  const std::string arch = meta_string(gguf, "general.architecture").value_or("llama");
  if (arch == "llama" || arch == "mistral") {
    const char* b = std::getenv("EMBER_IR_BUILD");
    const bool windowed = meta_number(gguf, arch + ".attention.sliding_window").value_or(0) > 0;
    return windowed || (b && std::string(b) == "1") ? build_ir_from_gguf(gguf) : std::string();
  }
  if (has_ir_builder(arch)) return build_ir_from_gguf(gguf);
  throw std::runtime_error("ember cannot run general.architecture = '" + arch +
                           "': the GGUF has no paperkiln.ir and ember has no IR builder for that architecture "
                           "(built: llama, mistral, qwen2, qwen3, gpt2, gptneox, gemma3, mamba)");
}

std::string build_ir_from_gguf(const GGUFData& g) {
  const std::string arch = meta_string(g, "general.architecture").value_or("");
  if (!has_ir_builder(arch)) throw std::runtime_error("no IR builder for general.architecture = '" + arch + "'");
  const std::string p = arch + ".";
  auto get = [&](const std::string& key, std::optional<double> dflt = std::nullopt) -> double {
    auto v = meta_number(g, p + key);
    if (!v) v = dflt;
    if (!v) throw std::runtime_error("IR builder: " + arch + " file has no " + p + key);
    return *v;
  };
  auto shape_has = [&](const std::string& t) { return g.tensor_infos_map.count(t) > 0; };
  auto no_rope_scaling = [&]() {
    const auto scaling = meta_string(g, p + "rope.scaling.type");
    if ((scaling && *scaling != "none") || shape_has("rope_freqs.weight"))
      throw std::runtime_error("IR builder: " + arch + " with RoPE scaling is not built yet");
  };
  const int d = static_cast<int>(get("embedding_length"));
  const int n_layers = static_cast<int>(get("block_count"));
  int vocab = static_cast<int>(g.tokenizer_tokens.size());
  if (vocab == 0) {
    auto it = g.tensor_infos_map.find("token_embd.weight");
    if (it == g.tensor_infos_map.end() || it->second.shape.size() < 2)
      throw std::runtime_error("IR builder: no tokens and no token_embd.weight");
    vocab = static_cast<int>(it->second.shape[1]);
  }
  const std::string B = "blk.{i}.";

  auto linear = [](const std::string& x, const std::string& w, const std::string& out, const std::string& b = "",
                   const std::string& tied = "") {
    json n = {{"op", "linear"}, {"in", {x}}, {"weight", w}, {"bias", nullptr}, {"out", out}};
    if (!b.empty()) n["bias"] = b;
    if (!tied.empty()) n["tied_to"] = tied;
    return n;
  };
  auto rmsnorm = [](const std::string& x, const std::string& w, double eps, const std::string& out, int group = 0) {
    json n = {{"op", "rmsnorm"}, {"in", {x}}, {"weight", w}, {"eps", f32_value(eps)}, {"out", out}};
    if (group) n["group"] = group;
    return n;
  };
  auto layernorm = [](const std::string& x, const std::string& name, double eps, const std::string& out) {
    return json{{"op", "layernorm"}, {"in", {x}},           {"weight", name + ".weight"},
                {"bias", name + ".bias"}, {"eps", f32_value(eps)}, {"out", out}};
  };
  auto op = [](const std::string& name, std::vector<std::string> in, const json& out, json attrs = json::object()) {
    attrs["op"] = name;
    attrs["in"] = in;
    attrs["out"] = out;
    return attrs;
  };
  auto rope = [](int H, int KV, int hd, int n_rot, const json& theta, const std::string& pairing,
                 const json& freq_scale = nullptr) {
    json n = {{"op", "rope"},     {"in", {"q", "k", "pos"}}, {"heads", H},          {"kv_heads", KV},
              {"head_dim", hd},   {"n_rot", n_rot},          {"theta", theta},      {"rope_heads", "all"},
              {"pairing", pairing}, {"out", {"q", "k"}}};
    if (!freq_scale.is_null()) n["freq_scale"] = freq_scale;
    return n;
  };
  auto attention = [](int H, int KV, int hd, double scale, const json& window = 0) {
    return json{{"op", "attention"}, {"in", {"q", "k", "v"}}, {"state", "kv"},     {"heads", H},
                {"kv_heads", KV},    {"head_dim", hd},        {"causal", true},    {"scale", f32_value(scale)},
                {"window", window},  {"sinks", 0},            {"softmax", "max-shifted, masked entries exactly zero"},
                {"out", "a"}};
  };
  auto kv_state = [](int H, int KV, int hd, int ctx, const json& window = 0) {
    return json{{"name", "kv"},     {"kind", "kv"}, {"heads", H}, {"kv_heads", KV}, {"head_dim", hd},
                {"max_len", ctx},   {"window", window}, {"sinks", 0}, {"dtype", "f32"}};
  };
  const bool tied = !shape_has("output.weight");
  auto tied_head = [&](json norm_nodes) {
    norm_nodes.push_back(linear("h", "output.weight", "logits", "", tied ? "token_embd.weight" : ""));
    return norm_nodes;
  };
  json embed = json::array({{{"op", "embedding"}, {"in", {"tokens"}}, {"weight", "token_embd.weight"}, {"out", "x"}}});
  json state, nodes, head;
  int ctx = 0;
  std::string summary;

  if (arch == "llama" || arch == "mistral" || arch == "qwen2" || arch == "qwen3") {
    const int H = static_cast<int>(get("attention.head_count"));
    const int KV = static_cast<int>(get("attention.head_count_kv", H));
    const int hd = static_cast<int>(get("attention.key_length", d / H));
    const int n_rot = static_cast<int>(get("rope.dimension_count", hd));
    ctx = static_cast<int>(get("context_length"));
    const double eps = get("attention.layer_norm_rms_epsilon");
    const double theta = get("rope.freq_base", 10000.0);
    no_rope_scaling();
    // HF Mistral's window (key j visible from query p iff p - w < j <= p); qwen2's
    // GGUF does not say which layers a window would cover, so none.
    int window = 0;
    if (arch == "llama" || arch == "mistral")
      window = static_cast<int>(meta_number(g, p + "attention.sliding_window").value_or(0));
    const bool bias = shape_has("blk.0.attn_q.bias"), qk_norm = shape_has("blk.0.attn_q_norm.weight");
    const std::string pairing = arch == "qwen2" || arch == "qwen3" ? "neox" : "interleaved";
    auto proj = [&](const std::string& name, const std::string& out) {
      return linear("h", B + name + ".weight", out, bias ? B + name + ".bias" : "");
    };
    nodes = json::array({rmsnorm("x", B + "attn_norm.weight", eps, "h"), proj("attn_q", "q"), proj("attn_k", "k"),
                         proj("attn_v", "v")});
    if (qk_norm) {
      nodes.push_back(rmsnorm("q", B + "attn_q_norm.weight", eps, "q", hd));
      nodes.push_back(rmsnorm("k", B + "attn_k_norm.weight", eps, "k", hd));
    }
    for (json n : {rope(H, KV, hd, n_rot, f32_value(theta), pairing),
                   attention(H, KV, hd, 1.0 / std::sqrt(static_cast<double>(hd)), window),
                   linear("a", B + "attn_output.weight", "o"), op("add", {"x", "o"}, "x"),
                   rmsnorm("x", B + "ffn_norm.weight", eps, "h2"), linear("h2", B + "ffn_gate.weight", "g"),
                   linear("h2", B + "ffn_up.weight", "u"), op("silu", {"g"}, "g"), op("mul", {"g", "u"}, "gu"),
                   linear("gu", B + "ffn_down.weight", "f"), op("add", {"x", "f"}, "x")})
      nodes.push_back(n);
    head = tied_head(json::array({rmsnorm("x", "output_norm.weight", eps, "h")}));
    state = json::array({kv_state(H, KV, hd, ctx, window)});
    summary = "heads " + std::to_string(H) + "/" + std::to_string(KV) + " x " + std::to_string(hd) + ", n_rot " +
              std::to_string(n_rot) + ", " + pairing + " RoPE" + (window ? ", window " + std::to_string(window) : "") +
              (bias ? ", q/k/v biases" : "") + (qk_norm ? ", per-head q/k RMSNorm" : "");
  } else if (arch == "gpt2" || arch == "gptneox") {
    // LayerNorm blocks over a fused, biased qkv (q | k | v as llama.cpp's convert
    // writes it; for gptneox it has already de-interleaved HF's per-head [q k v] rows).
    const bool neox = arch == "gptneox";
    const int H = static_cast<int>(get("attention.head_count"));
    const int KV = static_cast<int>(get("attention.head_count_kv", H));
    const int hd = static_cast<int>(get("attention.key_length", d / H));
    ctx = static_cast<int>(get("context_length"));
    const double eps = get("attention.layer_norm_epsilon");
    if (!neox)
      embed.push_back({{"op", "pos_embedding"}, {"in", {"x", "pos"}}, {"weight", "position_embd.weight"}, {"out", "x"}});
    nodes = json::array({layernorm("x", B + "attn_norm", eps, "h"),
                         linear("h", B + "attn_qkv.weight", "qkv", B + "attn_qkv.bias"),
                         op("split", {"qkv"}, {"q", "k", "v"}, {{"sizes", {H * hd, KV * hd, KV * hd}}})});
    int n_rot = 0;
    if (neox) {
      no_rope_scaling();
      n_rot = static_cast<int>(get("rope.dimension_count", hd));
      nodes.push_back(rope(H, KV, hd, n_rot, f32_value(get("rope.freq_base", 10000.0)), "neox"));
    }
    nodes.push_back(attention(H, KV, hd, 1.0 / std::sqrt(static_cast<double>(hd))));
    nodes.push_back(linear("a", B + "attn_output.weight", "o", B + "attn_output.bias"));
    const json ffn = json::array({layernorm("x", B + "ffn_norm", eps, "h2"),
                                  linear("h2", B + "ffn_up.weight", "u", B + "ffn_up.bias"),
                                  op("gelu", {"u"}, "u", {{"approx", neox ? "none" : "tanh"}}),
                                  linear("u", B + "ffn_down.weight", "f", B + "ffn_down.bias")});
    const bool parallel = neox && meta_bool(g, p + "use_parallel_residual").value_or(true);
    if (parallel) {
      // llama.cpp: cur = ffn(ln2 x) + x, then + attn_out; HF sums mlp + attn + x (ROUNDING apart).
      for (const auto& n : ffn) nodes.push_back(n);
      nodes.push_back(op("add", {"f", "x"}, "x"));
      nodes.push_back(op("add", {"x", "o"}, "x"));
    } else {
      nodes.push_back(op("add", {"x", "o"}, "x"));
      for (const auto& n : ffn) nodes.push_back(n);
      nodes.push_back(op("add", {"x", "f"}, "x"));
    }
    head = json::array({layernorm("x", "output_norm", eps, "h")});
    if (neox)
      head.push_back(linear("h", "output.weight", "logits"));
    else
      head = tied_head(head);
    state = json::array({kv_state(H, KV, hd, ctx)});
    summary = "heads " + std::to_string(H) + "/" + std::to_string(KV) + " x " + std::to_string(hd) + ", LayerNorm" +
              (neox ? ", NEOX RoPE n_rot " + std::to_string(n_rot) + (parallel ? ", parallel residual" : "")
                    : std::string(", learned positions"));
  } else if (arch == "gemma3") {
    // llama.cpp b6000 hparams.set_swa_pattern(6): layer i is local (windowed,
    // theta 10000) iff i % 6 < 5; the rest are global (the file's theta, linear scaling).
    constexpr int kSwaPattern = 6;
    constexpr double kLocalTheta = 10000.0;
    const int H = static_cast<int>(get("attention.head_count"));
    const int KV = static_cast<int>(get("attention.head_count_kv", H));
    const int hd = static_cast<int>(get("attention.key_length", d / H));
    const int n_rot = static_cast<int>(get("rope.dimension_count", hd));
    ctx = static_cast<int>(get("context_length"));
    const double eps = get("attention.layer_norm_rms_epsilon");
    if (shape_has("rope_freqs.weight")) throw std::runtime_error("IR builder: gemma3 with rope_freqs is not built yet");
    const auto stype = meta_string(g, p + "rope.scaling.type");
    if (stype && *stype != "none" && *stype != "linear")
      throw std::runtime_error("IR builder: gemma3 RoPE scaling '" + *stype + "' is not built yet");
    double factor = 1.0;
    if (stype && *stype == "linear") {
      factor = meta_number(g, p + "rope.scaling.factor").value_or(1.0);
      if (factor == 0.0) factor = 1.0;
    }
    const int win = static_cast<int>(get("attention.sliding_window"));
    const double theta_global = get("rope.freq_base");
    json windows = json::array(), thetas = json::array(), scales = json::array();
    for (int i = 0; i < n_layers; ++i) {
      const bool local = i % kSwaPattern < kSwaPattern - 1;
      windows.push_back(local ? win : 0);
      thetas.push_back(f32_value(local ? kLocalTheta : theta_global));
      scales.push_back(local ? 1.0 : f32_value(1.0 / factor));
    }
    windows = {{"per_layer", windows}};
    thetas = {{"per_layer", thetas}};
    scales = {{"per_layer", scales}};
    // llama.cpp: f_attention_scale = 1/sqrt(n_embd_head_k), except 1/sqrt(n_embd / n_head) for the 27B.
    const double scale = n_layers == 62 ? 1.0 / std::sqrt(static_cast<double>(d) / H)
                                        : 1.0 / std::sqrt(static_cast<double>(hd));
    embed.push_back(op("affine", {"x"}, "x", {{"scale", f32_value(std::sqrt(static_cast<double>(d)))}, {"shift", 0.0}}));
    nodes = json::array({rmsnorm("x", B + "attn_norm.weight", eps, "h"), linear("h", B + "attn_q.weight", "q"),
                         linear("h", B + "attn_k.weight", "k"), linear("h", B + "attn_v.weight", "v"),
                         rmsnorm("q", B + "attn_q_norm.weight", eps, "q", hd),
                         rmsnorm("k", B + "attn_k_norm.weight", eps, "k", hd),
                         rope(H, KV, hd, n_rot, thetas, "neox", factor != 1.0 ? scales : json(nullptr)),
                         attention(H, KV, hd, scale, windows), linear("a", B + "attn_output.weight", "o"),
                         rmsnorm("o", B + "post_attention_norm.weight", eps, "o"), op("add", {"x", "o"}, "x"),
                         rmsnorm("x", B + "ffn_norm.weight", eps, "h2"), linear("h2", B + "ffn_gate.weight", "g"),
                         linear("h2", B + "ffn_up.weight", "u"), op("gelu", {"g"}, "g", {{"approx", "tanh"}}),
                         op("mul", {"g", "u"}, "gu"), linear("gu", B + "ffn_down.weight", "f"),
                         rmsnorm("f", B + "post_ffw_norm.weight", eps, "f"), op("add", {"x", "f"}, "x")});
    head = tied_head(json::array({rmsnorm("x", "output_norm.weight", eps, "h")}));
    state = json::array({kv_state(H, KV, hd, ctx, windows)});
    summary = "heads " + std::to_string(H) + "/" + std::to_string(KV) + " x " + std::to_string(hd) +
              ", sandwich RMSNorms, GeGLU, window " + std::to_string(win) + " on 5 of every 6 layers" +
              (factor != 1.0 ? ", global RoPE scaled 1/" + std::to_string(factor) : "");
  } else {  // mamba
    const int K = static_cast<int>(get("ssm.conv_kernel")), Di = static_cast<int>(get("ssm.inner_size"));
    const int N = static_cast<int>(get("ssm.state_size")), R = static_cast<int>(get("ssm.time_step_rank"));
    const double eps = get("attention.layer_norm_rms_epsilon");
    ctx = static_cast<int>(get("context_length", 0.0));
    if (ctx == 0) ctx = 1048576;  // llama.cpp writes 1M: a recurrence has no length limit
    if (!shape_has("blk.0.ssm_dt.weight") || !shape_has("blk.0.ssm_d"))
      throw std::runtime_error("IR builder: mamba file lacks ssm_dt.weight or ssm_d");
    if (meta_bool(g, p + "ssm.dt_b_c_rms").value_or(false) || shape_has("blk.0.ssm_dt_norm.weight"))
      throw std::runtime_error("IR builder: mamba with dt/B/C RMSNorms (Falcon-Mamba) is not built yet");
    nodes = json::array(
        {rmsnorm("x", B + "attn_norm.weight", eps, "h"), linear("h", B + "ssm_in.weight", "xz"),
         op("split", {"xz"}, {"xs", "z"}, {{"sizes", {Di, Di}}}),
         op("conv1d_causal", {"xs"}, "xs",
            {{"state", "conv"}, {"weight", B + "ssm_conv1d.weight"}, {"bias", B + "ssm_conv1d.bias"},
             {"channels", Di}, {"kernel", K}}),
         op("silu", {"xs"}, "xs"), linear("xs", B + "ssm_x.weight", "xdb"),
         op("split", {"xdb"}, {"dt", "bm", "cm"}, {{"sizes", {R, N, N}}}),
         linear("dt", B + "ssm_dt.weight", "dt", B + "ssm_dt.bias"),
         op("selective_scan", {"xs", "dt", "bm", "cm"}, "y",
            {{"state", "ssm"}, {"a", B + "ssm_a"}, {"d", B + "ssm_d"}, {"channels", Di}, {"d_state", N},
             {"dt_softplus", true}}),
         op("silu", {"z"}, "z"), op("mul", {"y", "z"}, "y"), linear("y", B + "ssm_out.weight", "o"),
         op("add", {"x", "o"}, "x")});
    state = json::array({{{"name", "conv"}, {"kind", "recurrent"}, {"op", "conv1d_causal"}, {"dtype", "f32"},
                          {"tensors", json::array({{{"name", "x"}, {"shape", {K - 1, Di}}, {"init", 0}}})}},
                         {{"name", "ssm"}, {"kind", "recurrent"}, {"op", "selective_scan"}, {"dtype", "f32"},
                          {"tensors", json::array({{{"name", "s"}, {"shape", {Di, N}}, {"init", 0}}})}}});
    head = tied_head(json::array({rmsnorm("x", "output_norm.weight", eps, "h")}));
    summary = "selective scan d_inner " + std::to_string(Di) + ", d_state " + std::to_string(N) + ", dt_rank " +
              std::to_string(R) + ", conv kernel " + std::to_string(K);
  }

  const auto eos = meta_number(g, "tokenizer.ggml.eos_token_id");
  const auto bos = meta_number(g, "tokenizer.ggml.bos_token_id");
  json doc;
  doc["_comment"] = "built by ember's build_ir_from_gguf (graph_ir.cpp; paperkiln tools/gguf_ir.py's rules) from "
                    "general.architecture = " + arch + " and its metadata";
  doc["ir_version"] = 0;
  doc["family_hint"] = arch;
  doc["model"] = {{"vocab", vocab}, {"d", d}, {"n_layers", n_layers}, {"ctx", ctx}, {"dtype", "f32"}};
  doc["tokenizer"] = {{"type", "gguf"},
                      {"source", "gguf"},
                      {"specials", eos ? json{{"eot", static_cast<int64_t>(*eos)}} : json::object()},
                      {"bos", bos ? json(static_cast<int64_t>(*bos)) : json(nullptr)}};
  doc["inputs"] = json::array({{{"name", "tokens"}, {"kind", "token_ids"}}, {{"name", "pos"}, {"kind", "positions"}}});
  if (arch == "mamba") doc["inputs"] = json::array({{{"name", "tokens"}, {"kind", "token_ids"}}});
  doc["embed"] = {{"nodes", embed}};
  doc["block"] = {{"repeat", n_layers}, {"state", state}, {"nodes", nodes}};
  doc["head"] = {{"nodes", head}};
  doc["outputs"] = json::array({{{"name", "logits"}, {"kind", "logits"}}});
  doc["modes"] = {{"prefill", "all nodes over T tokens"},
                  {"decode", "the same graph with T = 1, attention reads and appends to the kv state, a recurrent op "
                             "reads and replaces its recurrent state"}};
  Logger::info("[IR] built the IR for general.architecture = " + arch + " from its metadata: d " + std::to_string(d) +
               ", " + std::to_string(n_layers) + " layers, " + summary + (tied ? ", tied head" : ""));
  return doc.dump();
}

Interpreter::Interpreter(const std::string& ir_json, const GGUFData& gguf, bool use_gpu) {
  const json j = json::parse(ir_json);
  try {
    init(j, ir_json, gguf, use_gpu);
  } catch (const json::exception&) {
    // ember's own checks read the document as its schema shapes it: a document that is not is
    // refused naming the rules it breaks (IR_SPEC 10.1), not a JSON accessor's message.
    validate_ir(j, gguf);
    throw;
  }
}

void Interpreter::init(const json& j, const std::string& ir_json, const GGUFData& gguf, bool use_gpu) {
  const int version = j.at("ir_version").get<int>();
  // 1 is spec/IR_SPEC.md's first normative text of the same document as 0 (paperkiln, 25 Oct 2026)
  if (version != 0 && version != 1)
    throw std::runtime_error("IR version " + std::to_string(version) + " is not supported (ember reads 0 and 1)");
  family_hint_ = j.value("family_hint", "");
  embed_ = parse_nodes(j.at("embed"), "embed");
  block_ = parse_nodes(j.at("block"), "block");
  head_ = parse_nodes(j.at("head"), "head");
  for (const auto& n : head_)
    if (!kRowWiseOps.count(n.op)) throw std::runtime_error("IR head op '" + n.op + "' is not row-wise");
  n_layers_ = j.at("block").at("repeat").get<int>();
  if (n_layers_ < 1) throw std::runtime_error("IR_SPEC 3.3: block.repeat " + std::to_string(n_layers_) + " is not a positive integer");
  // Tied-block recursion (paperkiln ir.hpp weight_index): application i reads
  // stored layer m_i's tensors; its state slots, kv cache and per-layer
  // attributes stay its own (indexed by i). Stored layers are 0 .. max m_i, each
  // used, and their count is the file's <arch>.block_count.
  if (j.at("block").contains("weight_index")) {
    const json& wi = j.at("block").at("weight_index");
    if (!is_per_layer(wi) || !wi.at("per_layer").is_array())
      throw std::runtime_error("IR block weight_index must be {\"per_layer\": [m_0, ..]}");
    for (const auto& e : wi.at("per_layer")) {
      if (!e.is_number_integer() || e.get<int>() < 0)
        throw std::runtime_error("IR block weight_index entries must be integers >= 0");
      weight_map_.push_back(e.get<int>());
    }
    if (static_cast<int>(weight_map_.size()) != n_layers_)
      throw std::runtime_error("IR block weight_index has " + std::to_string(weight_map_.size()) + " entries for " +
                               std::to_string(n_layers_) + " block applications");
    const int stored = *std::max_element(weight_map_.begin(), weight_map_.end()) + 1;
    std::vector<bool> used(static_cast<size_t>(stored), false);
    for (int m : weight_map_) used[m] = true;
    for (int m = 0; m < stored; ++m)
      if (!used[m]) throw std::runtime_error("IR block weight_index never uses stored layer " + std::to_string(m));
    const std::string arch = meta_string(gguf, "general.architecture").value_or("");
    if (const auto bc = meta_number(gguf, arch + ".block_count"); bc && static_cast<int>(*bc) != stored)
      throw std::runtime_error("IR block weight_index uses " + std::to_string(stored) + " stored layers, the file's " +
                               arch + ".block_count is " + std::to_string(static_cast<int>(*bc)));
    std::string m;
    for (int v : weight_map_) m += (m.empty() ? "" : ", ") + std::to_string(v);
    Logger::info("[IR] tied-block recursion: " + std::to_string(n_layers_) + " applications of " +
                 std::to_string(stored) + " stored layers, weight_index [" + m + "]");
  }
  vocab_ = j.at("model").at("vocab").get<int>();
  max_len_ = j.at("model").value("ctx", 0);

  bool has_kv = false, kv_rows_active = false, evict_declared = false;
  for (const auto& s : j.at("block").value("state", json::array())) {
    const std::string kind = s.at("kind").get<std::string>();
    if (kind == "recurrent") {
      RecSlot slot{s.at("name").get<std::string>(), s.at("op").get<std::string>(), {}};
      if (!kRecurrentOps.count(slot.op))
        throw std::runtime_error("IR recurrent state '" + slot.name + "': op '" + slot.op + "' is not a recurrent op");
      if (s.value("dtype", "f32") != "f32")
        throw std::runtime_error("IR recurrent state '" + slot.name + "': dtype must be f32");
      if (!s.contains("tensors") || s.at("tensors").empty())
        throw std::runtime_error("IR recurrent state '" + slot.name + "' declares no tensors");
      for (const auto& t : s.at("tensors")) {
        RecTensor rt;
        rt.name = t.at("name").get<std::string>();
        rt.shape = t.at("shape").get<std::vector<int64_t>>();
        rt.floats = 1;
        for (int64_t e : rt.shape) {
          if (e <= 0) throw std::runtime_error("IR recurrent state '" + slot.name + "': bad shape for " + rt.name);
          rt.floats *= static_cast<size_t>(e);
        }
        rt.init = t.value("init", 0.0f);
        rt.offset = rec_floats_;
        rec_floats_ += rt.floats;
        slot.tensors.push_back(std::move(rt));
      }
      for (const auto& other : rec_slots_)
        if (other.name == slot.name) throw std::runtime_error("IR recurrent state '" + slot.name + "' is declared twice");
      rec_slots_.push_back(std::move(slot));
      continue;
    }
    if (kind != "kv") throw std::runtime_error("IR state kind '" + kind + "' has no CPU implementation in ember");
    has_kv = true;
    kv_slot_ = s.value("name", "kv");
    if (s.contains("kv_from")) throw std::runtime_error("IR kv state: kv_from (recursive KV sharing) is not implemented in ember");
    if (s.contains("kv_rows")) {
      if (s.at("kv_rows") != "active")
        throw std::runtime_error("IR kv state: kv_rows '" + s.at("kv_rows").dump() + "' is not implemented in ember");
      kv_rows_active = true;
    }
    if (has(s, "evict")) {
      if (s.at("evict") != "turn_state")
        throw std::runtime_error("IR kv state: evict " + s.at("evict").dump() + " is not implemented in ember (turn_state only)");
      evict_declared = true;
    }
    kv_heads_ = s.at("kv_heads").get<int>();
    head_dim_ = s.at("head_dim").get<int>();
    max_len_ = s.at("max_len").get<int>();
    // Sliding window + sinks (paperkiln swa_attention). The cache keeps every
    // position up to max_len, which holds the first `sinks` and the last
    // `window` rows the rule needs.
    if (has(s, "window") && is_per_layer(s.at("window"))) {
      window_layer_ = s.at("window").at("per_layer").get<std::vector<int>>();
      if (static_cast<int>(window_layer_.size()) != n_layers_)
        throw std::runtime_error("IR kv state: per-layer window has " + std::to_string(window_layer_.size()) +
                                 " entries for " + std::to_string(n_layers_) + " blocks");
      window_ = *std::max_element(window_layer_.begin(), window_layer_.end());
    } else {
      window_ = s.value("window", 0);
    }
    if (has(s, "sinks") && is_per_layer(s.at("sinks")))
      throw std::runtime_error("IR kv state: per-layer sinks are not implemented in ember");
    sinks_ = s.value("sinks", 0);
  }
  for (const auto& n : block_) {
    if (n.op != "attention") continue;
    if (!n.attrs.value("causal", true))
      throw std::runtime_error("IR attention with causal=false is not implemented in ember");
    for (int l = 0; l < n_layers_; ++l) {
      int w = 0, s = 0;
      attention_window(n, l, w, s);
      if (w < 0 || s < 0) throw std::runtime_error("IR attention window/sinks must be non-negative");
      // A windowed kv state declares what its cache holds; an attention node
      // reading it with another window or sink count is an exporter error.
      const int kvw = window_layer_.empty() ? window_ : window_layer_[l];
      if (kvw > 0 && (w != kvw || s != sinks_))
        throw std::runtime_error("IR attention window/sinks (" + std::to_string(w) + ", " + std::to_string(s) +
                                 ") in block " + std::to_string(l) + " differ from its kv state's (" +
                                 std::to_string(kvw) + ", " + std::to_string(sinks_) + ")");
      if (w > 0 && (l == 0 || window_layer_.size() > 0))
        Logger::info("[IR] sliding-window attention: block " + std::to_string(l) + " window " + std::to_string(w) +
                     ", sinks " + std::to_string(s));
    }
  }
  for (const auto* sub : {&embed_, &block_, &head_})
    for (const auto& n : *sub)
      if (n.op == "gelu" && n.attrs.value("approx", "none") != "tanh" && n.attrs.value("approx", "none") != "none")
        throw std::runtime_error("IR gelu approx '" + n.attrs.value("approx", "") + "' is not implemented in ember");
  // The gating ops' attributes and arities (paperkiln ir.hpp): checked here so
  // a malformed node fails at load, not mid-generation.
  for (const auto* sub : {&embed_, &block_, &head_})
    for (const auto& n : *sub) {
      auto arity = [&](size_t k, bool at_least) {
        if (at_least ? n.in.size() < k : n.in.size() != k)
          throw std::runtime_error("IR " + n.op + " needs " + (at_least ? "at least " : "") + std::to_string(k) +
                                   " input(s), has " + std::to_string(n.in.size()));
        if (n.out.size() != 1) throw std::runtime_error("IR " + n.op + " needs one output");
      };
      if (n.op == "sub" || n.op == "mul_col" || n.op == "append") arity(2, false);
      if (n.op == "relu" || n.op == "sigmoid" || n.op == "affine" || n.op == "rms_row" || n.op == "scale") arity(1, false);
      if (n.op == "scale" && !has(n.attrs, "weight")) throw std::runtime_error("IR scale needs its weight [d]");
      if (n.op == "list" || n.op == "depth_attend") arity(1, true);
      if (n.op == "affine" && (!has(n.attrs, "scale") || !has(n.attrs, "shift")))
        throw std::runtime_error("IR affine needs float attributes scale and shift");
      if (n.op == "rms_row" && !has(n.attrs, "eps")) throw std::runtime_error("IR rms_row needs its eps attribute");
      for (const auto& kv : n.attrs.items()) {  // per-layer attributes: block nodes, the attributes ember reads per block
        if (!is_per_layer(kv.value())) continue;
        const bool ok = sub == &block_ && ((n.op == "attention" && (kv.key() == "window" || kv.key() == "sinks")) ||
                                           (n.op == "rope" && (kv.key() == "theta" || kv.key() == "freq_scale")) ||
                                           (n.op == "affine" && (kv.key() == "scale" || kv.key() == "shift")) ||
                                           (n.op == "depth_route" && kv.key() == "step"));
        if (!ok)
          throw std::runtime_error("IR " + n.op + ": a per-layer '" + kv.key() + "' is not implemented in ember");
        if (static_cast<int>(kv.value().at("per_layer").size()) != n_layers_)
          throw std::runtime_error("IR " + n.op + ": per-layer '" + kv.key() + "' has " +
                                   std::to_string(kv.value().at("per_layer").size()) + " entries for " +
                                   std::to_string(n_layers_) + " blocks");
      }
      if (n.op == "expr") {  // a runtime-defined op: parsed here, so a bad string fails at load naming its column
        arity(1, true);
        if (!n.attrs.contains("expr") || !n.attrs.at("expr").is_string())
          throw std::runtime_error("IR expr needs its string attribute expr");
        try {
          expr::cached(n.attrs.at("expr").get<std::string>(), n.in.size());
        } catch (const std::invalid_argument& e) {
          throw std::runtime_error(std::string("IR ") + e.what());
        }
      }
    }
#ifdef HAS_CUDA
  if (use_gpu && !device_supported(ir_json))
    throw std::runtime_error("IR uses an op or a shape that has no CUDA kernel (EMBER_IR_DEVICE=cpu runs it)");
#endif
  for (const auto& n : block_) {
    if (n.op == "rope") {
      const std::string pairing = n.attrs.value("pairing", "interleaved");
      if (pairing != "interleaved" && pairing != "neox")
        throw std::runtime_error("IR rope pairing '" + pairing + "' is not implemented in ember");
      if (n.attrs.value("rope_heads", "all") != "all")
        throw std::runtime_error("IR rope_heads '" + n.attrs.value("rope_heads", "") + "' is not implemented in ember");
      if (kv_heads_ > 0 && has(n.attrs, "kv_heads") && n.attrs.at("kv_heads").get<int>() != kv_heads_)
        throw std::runtime_error("IR rope kv_heads differs from its kv state's");
      // The table is the kv state's head_dim wide, as before, unless the node
      // names its own n_rot (partial rotary) or NEOX pairing: then n_rot
      // (default head_dim) must fit the kv state's heads.
      const int hd = n.attrs.at("head_dim").get<int>();
      int n_rot = head_dim_;
      if (has(n.attrs, "n_rot") || pairing == "neox") {
        n_rot = has(n.attrs, "n_rot") ? n.attrs.at("n_rot").get<int>() : hd;
        if (hd != head_dim_ || n_rot <= 0 || n_rot % 2 != 0 || n_rot > hd)
          throw std::runtime_error("IR rope: n_rot " + std::to_string(n_rot) + " / head_dim " + std::to_string(hd) +
                                   " (kv state head_dim " + std::to_string(head_dim_) + ") is not usable");
      }
      std::vector<float> thetas(static_cast<size_t>(n_layers_));
      std::vector<float> scales(static_cast<size_t>(n_layers_), 1.0f);
      for (int l = 0; l < n_layers_; ++l) {
        thetas[l] = at_layer(n.attrs.at("theta"), l).get<float>();
        if (has(n.attrs, "freq_scale")) scales[l] = at_layer(n.attrs.at("freq_scale"), l).get<float>();
        if (!(scales[l] > 0.0f)) throw std::runtime_error("IR rope freq_scale must be positive");
      }
      if (rope_dim_ != 0 && (rope_theta_layer_ != thetas || rope_scale_layer_ != scales || rope_dim_ != n_rot ||
                             rope_neox_ != (pairing == "neox")))
        throw std::runtime_error(
            "IR rope nodes with different theta, freq_scale, n_rot or pairing are not implemented in ember");
      rope_theta_layer_ = thetas;
      rope_scale_layer_ = scales;
      rope_theta_ = thetas[0];
      rope_freq_scale_ = scales[0];
      rope_dim_ = n_rot;
      rope_neox_ = pairing == "neox";
    }
  }
  // The rmsnorm groups (per-head q/k norm) and the attention head counts.
  for (const auto& n : block_) {
    if (n.op == "rmsnorm" && has(n.attrs, "group") && n.attrs.at("group").get<int>() <= 0)
      throw std::runtime_error("IR rmsnorm group must be positive");
    if (n.op == "attention") {
      const int hq = n.attrs.at("heads").get<int>(), hkv = n.attrs.at("kv_heads").get<int>();
      if (hkv <= 0 || hq % hkv != 0)
        throw std::runtime_error("IR attention: heads " + std::to_string(hq) + " is not a multiple of kv_heads " +
                                 std::to_string(hkv));
    }
  }
  // Mixture-of-Recursions (ir.hpp depth_route, llama_mor): one depth_route node, the block's
  // first, outside it nowhere; its steps one consecutive run of equal-length steps 0 .. R-1; causal
  // token choice; a kv state with kv_rows "active"; no recurrent state or windowed attention.
  for (const auto* sub : {&embed_, &head_})
    for (const auto& n : *sub)
      if (n.op == "depth_route") throw std::runtime_error("IR depth_route outside the block");
  for (size_t k = 0; k < block_.size(); ++k) {
    const Node& n = block_[k];
    if (n.op != "depth_route") continue;
    if (k != 0) throw std::runtime_error("IR depth_route must be the block's first node");
    if (n.in.size() != 1 || n.out.size() != 1) throw std::runtime_error("IR depth_route reads [x] and writes one output");
    if (n.attrs.value("route", "") != "token_choice")
      throw std::runtime_error("IR depth_route route '" + n.attrs.value("route", "") +
                               "' is not implemented in ember (causal token choice only)");
    const std::string gate = n.attrs.value("gate", "");
    if (gate != "score" && gate != "none") throw std::runtime_error("IR depth_route gate '" + gate + "' is not score or none");
    mor_R_ = n.attrs.at("n_rec").get<int>();
    if (mor_R_ < 1) throw std::runtime_error("IR depth_route n_rec must be at least 1");
    if (!is_per_layer(n.attrs.at("step"))) throw std::runtime_error("IR depth_route step must be {\"per_layer\": [...]}");
    mor_step_ = n.attrs.at("step").at("per_layer").get<std::vector<int>>();
    if (static_cast<int>(mor_step_.size()) != n_layers_)
      throw std::runtime_error("IR depth_route step has " + std::to_string(mor_step_.size()) + " entries for " +
                               std::to_string(n_layers_) + " block applications");
    // One run of applications with steps 0, 0, .., 1, 1, .., R-1, all steps the same length.
    int first = -1, last = -1;
    for (int i = 0; i < n_layers_; ++i)
      if (mor_step_[i] >= 0) {
        if (first < 0) first = i;
        last = i;
      }
    const int span = first < 0 ? 0 : last - first + 1;
    if (first < 0 || span % mor_R_ != 0) throw std::runtime_error("IR depth_route step: the recursion is not R equal steps");
    for (int i = 0; i < n_layers_; ++i) {
      const int want = i < first || i > last ? -1 : (i - first) / (span / mor_R_);
      if (mor_step_[i] != want)
        throw std::runtime_error("IR depth_route step: application " + std::to_string(i) + " has step " +
                                 std::to_string(mor_step_[i]) + ", expected " + std::to_string(want) +
                                 " (consecutive, ascending, equal-length steps)");
    }
    mor_gate_score_ = gate == "score";
    mor_x_ = n.in[0];
    mor_out_ = n.out[0];
    mor_router_ = n.attrs.at("weight").get<std::string>();
    if (mor_router_.find("{i}") != std::string::npos)
      throw std::runtime_error("IR depth_route weight names its tensor in full (no {i})");
  }
  if (!mor_step_.empty()) {
    if (!kv_rows_active) throw std::runtime_error("IR depth_route needs a kv state with kv_rows \"active\"");
    if (!rec_slots_.empty()) throw std::runtime_error("IR depth_route with recurrent state is not implemented in ember");
    if (window_ > 0 || !window_layer_.empty() || sinks_ > 0)
      throw std::runtime_error("IR depth_route with windowed attention is not implemented in ember");
    mor_rows_.assign(n_layers_, {});
  }
  // Per-turn state tokens (paperkiln IR_SPEC_DRAFT.md 1.1, 1.3, 1.4): the extended embedding
  // (state_weight, one embedding in the embed subgraph, its name in full), attention "mask"
  // "turn_state" (window 0 and sinks 0 only), kv "evict" "turn_state" (every attention node
  // turn_state). Not with depth_route (MoR's compact caches) or, for eviction, recurrent state.
  for (const auto* sub : {&embed_, &block_, &head_})
    for (const auto& n : *sub) {
      if (n.op == "embedding" && has(n.attrs, "state_weight")) {
        if (sub != &embed_) throw std::runtime_error("IR embedding state_weight outside the embed subgraph");
        if (!state_weight_.empty()) throw std::runtime_error("IR: more than one embedding with a state_weight");
        state_weight_ = n.attrs.at("state_weight").get<std::string>();
        if (state_weight_.find("{i}") != std::string::npos)
          throw std::runtime_error("IR embedding state_weight names its tensor in full (no {i})");
      }
      if (n.op == "attention" && n.attrs.contains("mask")) {
        const std::string mask = n.attrs.at("mask").is_string() ? n.attrs.at("mask").get<std::string>() : "";
        if (mask != "causal" && mask != "turn_state")
          throw std::runtime_error("IR attention mask " + n.attrs.at("mask").dump() +
                                   " is not implemented in ember (causal or turn_state)");
        if (mask == "turn_state") turn_state_ = true;
      }
    }
  if (turn_state_) {
    for (const auto& n : block_) {
      if (n.op != "attention" || n.attrs.value("mask", "causal") != "turn_state") continue;
      for (int l = 0; l < n_layers_; ++l) {
        int w = 0, s = 0;
        attention_window(n, l, w, s);
        if (w != 0 || s != 0)
          throw std::runtime_error("IR attention mask turn_state needs window 0 and sinks 0 (block " + std::to_string(l) +
                                   " has window " + std::to_string(w) + ", sinks " + std::to_string(s) + ")");
      }
    }
    if (window_ > 0 || !window_layer_.empty() || sinks_ > 0)
      throw std::runtime_error("IR attention mask turn_state with a windowed kv state is not allowed");
    if (!mor_step_.empty() || kv_rows_active)
      throw std::runtime_error("IR attention mask turn_state with depth_route / kv_rows active is not implemented in ember");
  }
  {
    bool all_turn_state = true;
    for (const auto& n : block_)
      if (n.op == "attention" && n.attrs.value("mask", "causal") != "turn_state") all_turn_state = false;
    if (evict_declared && (!turn_state_ || !all_turn_state))
      throw std::runtime_error("IR kv state evict turn_state needs every attention node to have mask turn_state");
    evict_ = evict_declared;
    evict_declared_ = evict_declared;
    if (const char* e = std::getenv("EMBER_EVICT")) {
      if (std::string(e) == "0") evict_ = false;
      if (std::string(e) == "1" && turn_state_) {
        if (!all_turn_state)
          throw std::runtime_error("EMBER_EVICT=1: eviction needs every attention node to have mask turn_state");
        evict_ = true;
      }
    }
    if (evict_ && !rec_slots_.empty())
      throw std::runtime_error("IR kv state evict turn_state with recurrent state is not implemented in ember");
  }
  // EMBER_IR_CTX=N: hold at most N positions (a third-party IR declares the
  // model's whole trained context, e.g. 32768, which is ~0.8 GB of f32 KV for
  // Qwen2.5-0.5B).
  if (const char* c = std::getenv("EMBER_IR_CTX")) {
    const int cap = std::atoi(c);
    if (cap > 0 && max_len_ > cap) {
      Logger::info("[IR] EMBER_IR_CTX: the kv state holds " + std::to_string(cap) + " of the IR's " +
                   std::to_string(max_len_) + " positions");
      max_len_ = cap;
    }
  }
  bool has_attention = false;
  for (const auto& n : block_) has_attention = has_attention || n.op == "attention";
  if ((has_kv || has_attention) && (kv_heads_ <= 0 || head_dim_ <= 0 || max_len_ <= 0))
    throw std::runtime_error("IR block declares no usable kv state (kv_heads, head_dim, max_len)");
  if (max_len_ <= 0) throw std::runtime_error("IR model.ctx is required when no kv state bounds the sequence");
  for (const auto* sub : {&embed_, &head_})
    for (const auto& n : *sub)
      if (kRecurrentOps.count(n.op)) throw std::runtime_error("IR op '" + n.op + "' outside the block has no state");
  // Every recurrent node names a slot of its op whose tensors fit the node.
  for (const auto& n : block_) {
    if (n.op == "linear_attention") {
      const RecSlot& slot = rec_slot(n, "linear_attention");
      const int heads = n.attrs.at("heads").get<int>(), hd = n.attrs.at("head_dim").get<int>();
      if (n.attrs.value("feature_map", "elu_plus_one") != "elu_plus_one")
        throw std::runtime_error("IR linear_attention feature_map '" + n.attrs.value("feature_map", "") +
                                 "' is not implemented in ember");
      if (!n.attrs.value("causal", true))
        throw std::runtime_error("IR linear_attention with causal=false is not implemented in ember");
      if (n.in.size() != 3 || n.out.size() != 1) throw std::runtime_error("IR linear_attention reads [q, k, v] and writes one output");
      for (const char* t : {"num", "den"}) {
        bool found = false;
        for (const auto& rt : slot.tensors)
          if (rt.name == t) {
            found = true;
            if (rt.shape != std::vector<int64_t>{heads, hd})
              throw std::runtime_error("IR recurrent state '" + slot.name + "': " + t + " is not [heads, head_dim]");
          }
        if (!found) throw std::runtime_error("IR recurrent state '" + slot.name + "' has no tensor '" + t + "'");
      }
    } else if (n.op == "ssm_scan") {
      const RecSlot& slot = rec_slot(n, "ssm_scan");
      const int64_t sn = n.attrs.at("n").get<int64_t>();
      if (n.in.size() != 1 || n.out.size() != 1) throw std::runtime_error("IR ssm_scan reads [u] and writes one output");
      if (slot.tensors.size() != 1 || slot.tensors[0].name != "s" || slot.tensors[0].shape != std::vector<int64_t>{sn})
        throw std::runtime_error("IR recurrent state '" + slot.name + "' must hold one tensor s [n]");
    } else if (n.op == "conv1d_causal") {
      const RecSlot& slot = rec_slot(n, "conv1d_causal");
      const int64_t C = n.attrs.at("channels").get<int64_t>(), K = n.attrs.at("kernel").get<int64_t>();
      if (n.in.size() != 1 || n.out.size() != 1) throw std::runtime_error("IR conv1d_causal reads [x] and writes one output");
      if (K < 2 || C < 1) throw std::runtime_error("IR conv1d_causal needs channels >= 1 and kernel >= 2");
      if (slot.tensors.size() != 1 || slot.tensors[0].name != "x" ||
          slot.tensors[0].shape != std::vector<int64_t>{K - 1, C})
        throw std::runtime_error("IR recurrent state '" + slot.name + "' must hold one tensor x [kernel - 1, channels]");
    } else if (n.op == "selective_scan") {
      const RecSlot& slot = rec_slot(n, "selective_scan");
      const int64_t D = n.attrs.at("channels").get<int64_t>(), N = n.attrs.at("d_state").get<int64_t>();
      if (n.in.size() != 4 || n.out.size() != 1)
        throw std::runtime_error("IR selective_scan reads [x, dt, B, C] and writes one output");
      if (slot.tensors.size() != 1 || slot.tensors[0].name != "s" || slot.tensors[0].shape != std::vector<int64_t>{D, N})
        throw std::runtime_error("IR recurrent state '" + slot.name + "' must hold one tensor s [channels, d_state]");
    }
  }
  if (rec_floats_ > 0) {
    rec_init_.assign(rec_floats_, 0.0f);
    for (const auto& slot : rec_slots_)
      for (const auto& rt : slot.tensors)
        std::fill(rec_init_.begin() + rt.offset, rec_init_.begin() + rt.offset + rt.floats, rt.init);
    rec_.resize(static_cast<size_t>(n_layers_) * rec_floats_);
    for (int l = 0; l < n_layers_; ++l) std::copy(rec_init_.begin(), rec_init_.end(), rec_.begin() + l * rec_floats_);
    // Snapshots are a whole state each; at most 64, and at most 256 MiB.
    const size_t budget = (256u << 20) / sizeof(float);
    rec_max_snaps_ = std::max<size_t>(2, std::min<size_t>(64, budget / rec_.size()));
  }

  // Quantised weights stay in their blocks for large models on the CPU (the
  // memory saving pays for the per-row dequantisation); small models, like
  // the llama path's dequantize_all_if_small, and the CUDA path (f32 kernels)
  // dequantise at load. EMBER_IR_QUANT=1 / EMBER_IR_DEQUANT=1 force either.
  {
    size_t params = 0;
    for (const auto& kv : gguf.tensor_infos_map) params += kv.second.num_elements;
#ifdef HAS_CUDA
    const bool device = use_gpu;
#else
    const bool device = false;
#endif
    dequant_all_ = device || params <= 64u * 1000 * 1000;
    const char* q = std::getenv("EMBER_IR_QUANT");
    if (q && std::string(q) == "1" && !device) dequant_all_ = false;
    if (const char* dq = std::getenv("EMBER_IR_DEQUANT")) dequant_all_ = dequant_all_ || std::string(dq) == "1";
  }
  bind(embed_, -1, gguf);
  for (int l = 0; l < n_layers_; ++l) bind(block_, l, gguf);
  bind(head_, -1, gguf);
  // The state rows: state_weight [k, d] (ggml ne0 = d, the token embedding's width); k from its
  // size, equal to the file's paperkiln.state_tokens.k when the file names one.
  {
    const auto meta_k = meta_number(gguf, "paperkiln.state_tokens.k");
    if (!state_weight_.empty()) {
      bind_tensor(state_weight_, gguf, true);
      int d = 0;
      for (const auto& n : embed_)
        if (n.op == "embedding" && has(n.attrs, "state_weight")) {
          std::string w = layer_name(n.attrs.at("weight").get<std::string>(), -1);
          d = static_cast<int>(shapes_.at(w).at(0));
        }
      const auto& sh = shapes_.at(state_weight_);
      const size_t numel = bound(state_weight_).numel;
      if (sh.empty() || sh[0] != d || d <= 0 || numel % static_cast<size_t>(d) != 0 || numel == 0)
        throw std::runtime_error("IR embedding state_weight '" + state_weight_ + "' is not [k, d] with d " +
                                 std::to_string(d) + " (the token embedding's width)");
      state_k_ = static_cast<int>(numel / static_cast<size_t>(d));
      if (meta_k && static_cast<int>(*meta_k) != state_k_)
        throw std::runtime_error("paperkiln.state_tokens.k is " + std::to_string(static_cast<int>(*meta_k)) +
                                 ", the IR's state_weight '" + state_weight_ + "' has " + std::to_string(state_k_) +
                                 " rows");
      Logger::info("[IR] per-turn state tokens: k " + std::to_string(state_k_) + " (ids " + std::to_string(vocab_) +
                   " .. " + std::to_string(vocab_ + state_k_ - 1) + " read '" + state_weight_ + "')" +
                   (turn_state_ ? std::string(", attention mask turn_state") : std::string()) +
                   (evict_ ? std::string(", kv eviction at turn_state") : std::string()));
    } else if (meta_k && *meta_k > 0) {
      throw std::runtime_error("paperkiln.state_tokens.k is " + std::to_string(static_cast<int>(*meta_k)) +
                               " but the IR's embedding has no state_weight");
    }
  }
  // paperkiln spec/IR_SPEC.md section 10.1 (graph_ir_validate.h): after ember's own checks above, so a
  // document they refuse keeps their message; then what ember does not implement among valid
  // documents (section 10.3).
  validate_ir(j, gguf);
  for (const auto* sub : {&embed_, &block_, &head_})
    for (const auto& n : *sub)
      if (n.attrs.contains("tensor_index"))
        throw std::runtime_error("IR " + n.op + ": tensor_index (IR_SPEC 7.2) is not implemented in ember yet");
  size_t n_quant = 0, bytes_quant = 0, bytes_f32 = 0, n_mapped = 0;
  for (const auto& kv : weights_) {
    n_quant += kv.second.quantised();
    n_mapped += kv.second.quantised() && kv.second.q.empty();
    bytes_quant += kv.second.qbytes;
    bytes_f32 += kv.second.f32.size() * sizeof(float);
  }

  // RoPE table, computed exactly as TinyLlamaModel::initialize_rope_freqs,
  // over rope_dim_ (n_rot; head_dim for full rotary).
  // One table per distinct (theta, freq_scale) (per-layer RoPE bases and scales); freq_scale 1 leaves the
  // angle pos * freq.
  for (size_t l = 0; l < rope_theta_layer_.size(); ++l) {
    const float theta = rope_theta_layer_[l], fs = rope_scale_layer_[l];
    if (!(theta > 0.0f) || rope_tables_.count({theta, fs})) continue;
    auto& tab = rope_tables_[{theta, fs}];
    tab.resize(static_cast<size_t>(max_len_) * rope_dim_ / 2);
    for (int pos = 0; pos < max_len_; ++pos) {
      for (int i = 0; i < rope_dim_; i += 2) {
        float freq = 1.0f / std::pow(theta, float(i) / rope_dim_);
        float val = (static_cast<float>(pos) * fs) * freq;
        tab[(static_cast<size_t>(pos) * rope_dim_ / 2) + (i / 2)] = {std::cos(val), std::sin(val)};
      }
    }
  }
  if (rope_theta_ > 0.0f) rope_freqs_ = rope_tables_.at({rope_theta_, rope_freq_scale_});
#ifdef HAS_CUDA
  if (use_gpu) {
    init_device();
    gpu_ = true;
  }
#else
  (void)use_gpu;
#endif
  if (!gpu_ && kv_heads_ > 0) {
    kv_.resize(n_layers_);
    for (auto& layer : kv_) {
      layer.k.assign(static_cast<size_t>(max_len_) * kv_heads_ * head_dim_, 0.0f);
      layer.v.assign(layer.k.size(), 0.0f);
    }
  }
  Logger::info("[IR] interpreter ready: family_hint=" + family_hint_ + ", " + std::to_string(n_layers_) + " blocks of " +
               std::to_string(block_.size()) + " ops, " + std::to_string(weights_.size()) + " weights bound (" +
               std::to_string(n_quant) + " kept quantised, " + std::to_string(bytes_quant >> 20) + " MiB" +
               (n_mapped ? " (" + std::to_string(n_mapped) + " read in place from the memory map)" : std::string()) +
               "; f32 " +
               std::to_string(bytes_f32 >> 20) + " MiB" + (dequant_all_ ? "; dequantised at load" : "") + "), device " +
               (gpu_ ? "cuda" : "cpu"));
  if (rec_floats_ > 0)
    Logger::info("[IR] recurrent state: " + std::to_string(rec_slots_.size()) + " slot(s), " +
                 std::to_string(rec_floats_) + " floats per block; up to " + std::to_string(rec_max_snaps_) +
                 " snapshots for prefix reuse and speculative rewinds");
}

Interpreter::~Interpreter() {
#ifdef HAS_CUDA
  free_device();
#endif
}

void Interpreter::attention_window(const Node& n, int layer, int& window, int& sinks) const {
  window = has(n.attrs, "window") ? at_layer(n.attrs.at("window"), layer).get<int>()
                                  : (window_layer_.empty() ? window_ : window_layer_.at(layer));
  sinks = has(n.attrs, "sinks") ? at_layer(n.attrs.at("sinks"), layer).get<int>() : sinks_;
}

const std::vector<std::pair<float, float>>& Interpreter::rope_table(int layer) const {
  if (rope_theta_layer_.empty()) return rope_freqs_;
  return rope_tables_.at({rope_theta_layer_.at(layer), rope_scale_layer_.at(layer)});
}

std::string Interpreter::layer_name(const std::string& pattern, int layer) const {
  std::string s = pattern;
  const auto p = s.find("{i}");
  if (p != std::string::npos) {
    if (layer < 0) throw std::runtime_error("IR weight '" + pattern + "' uses {i} outside the block");
    s.replace(p, 3, std::to_string(weight_map_.empty() ? layer : weight_map_.at(layer)));
  }
  return s;
}

void Interpreter::bind_tensor(const std::string& name, const GGUFData& gguf, bool keep_quantised) {
  if (auto bw = weights_.find(name); bw != weights_.end()) {
    Weight& w = bw->second;
    if (keep_quantised || !w.quantised()) return;
    // Bound quantised for a row-wise op, now needed whole (a norm, a bias): f32.
    w.f32.resize(w.numel);
    dequant_span_to_f32(w.type, w.qp, 0, w.numel, w.f32.data());
    std::vector<uint8_t>().swap(w.q);
    w.qp = nullptr;
    w.qbytes = 0;
    return;
  }
  auto it = gguf.tensor_infos_map.find(name);
  if (it == gguf.tensor_infos_map.end()) throw std::runtime_error("IR weight '" + name + "' is not in the GGUF");
  const GGUFTensorInfo& info = it->second;
  const uint8_t* base = gguf.mapped_tensor_data
                            ? static_cast<const uint8_t*>(gguf.mapped_tensor_data) + gguf.offset_diff_for_mmap
                            : gguf.tensor_data.data();
  const uint8_t* src = base + info.offset;
  std::vector<int64_t> shape(info.shape.begin(), info.shape.end());
  const GGMLType t = info.type;
  const bool block_type = t == GGMLType::GGML_TYPE_Q8_0 || t == GGMLType::GGML_TYPE_Q4_0 ||
                          t == GGMLType::GGML_TYPE_Q4_1 || t == GGMLType::GGML_TYPE_Q5_0 || t == GGMLType::GGML_TYPE_Q5_1;
  const bool half = t == GGMLType::GGML_TYPE_F16 || t == GGMLType::GGML_TYPE_BF16;
  if (keep_quantised && !dequant_all_ && !shape.empty() && (half || (block_type && shape[0] % 32 == 0))) {
    // Rows (ne0 elements) start on block boundaries, so a row is a whole
    // number of blocks and dequantises on its own.
    const size_t bytes = half ? info.num_elements * 2 : info.num_elements / 32 * ggml_type_size(t);
    if (info.size_in_bytes < bytes)
      throw std::runtime_error("IR weight '" + name + "' is shorter than its element count implies");
    Weight w;
    if (gguf.mapped_tensor_data) {
      w.qp = src;  // read in place: the mapping lives as long as the GGUFData
    } else {
      w.q.assign(src, src + bytes);
      w.qp = w.q.data();  // a vector's buffer survives the moves below
    }
    w.qbytes = bytes;
    w.type = t;
    w.numel = info.num_elements;
    shapes_[name] = shape;
    weights_[name] = std::move(w);
    return;
  }
  std::vector<float> out;
  if (info.type == GGMLType::GGML_TYPE_F32) {
    out.resize(info.num_elements);
    std::memcpy(out.data(), src, info.num_elements * sizeof(float));
  } else if (info.type == GGMLType::GGML_TYPE_Q8_0) {
    std::vector<block_q8_0> blocks(info.size_in_bytes / sizeof(block_q8_0));
    std::memcpy(blocks.data(), src, blocks.size() * sizeof(block_q8_0));
    dequantize_vector_q8_0_to_f32(blocks, out, info.num_elements);
  } else if (!dequant_legacy_to_f32(info, src, out)) {
    throw std::runtime_error("IR weight '" + name + "': tensor type " + ggml_type_name(info.type) +
                             " is not supported by ember's interpreter");
  }
  shapes_[name] = shape;
  Weight w;
  w.numel = out.size();
  w.f32 = std::move(out);
  weights_[name] = std::move(w);
}

void Interpreter::bind(const std::vector<Node>& nodes, int layer, const GGUFData& gguf) {
  for (const auto& n : nodes) {
    // depth_attend over one source never reads its pseudo-query (ir.hpp), so
    // a file may leave it out; run_nodes refuses a missing one for S >= 2.
    if (n.op == "depth_attend" && has(n.attrs, "weight") &&
        !gguf.tensor_infos_map.count(layer_name(n.attrs.at("weight").get<std::string>(), layer)))
      continue;
    if (has(n.attrs, "weight")) {
      std::string w = layer_name(n.attrs.at("weight").get<std::string>(), layer);
      if (!gguf.tensor_infos_map.count(w) && has(n.attrs, "tied_to"))
        w = layer_name(n.attrs.at("tied_to").get<std::string>(), layer);
      bind_tensor(w, gguf, n.op == "linear" || n.op == "embedding");
    }
    if (has(n.attrs, "bias")) bind_tensor(layer_name(n.attrs.at("bias").get<std::string>(), layer), gguf);
    // rmsnorm weight_offset c: the weight is read as w + c, rounded to f32 once at load (Gemma's (1 + w)
    // for a file holding HF's raw w; llama.cpp's convert already stores 1 + w).
    if (n.op == "rmsnorm" && has(n.attrs, "weight_offset") && n.attrs.at("weight_offset").get<float>() != 0.0f) {
      const std::string w = layer_name(n.attrs.at("weight").get<std::string>(), layer);
      if (offset_applied_.insert(w).second) {
        const float off = n.attrs.at("weight_offset").get<float>();
        for (float& x : weights_.at(w).f32) x = x + off;
      }
    }
    if (n.op == "moe_ffn") {
      const std::string norm = n.attrs.value("gate_norm", "renorm"), act = n.attrs.value("activation", "silu");
      if (norm != "renorm" && norm != "raw") throw std::runtime_error("IR moe_ffn gate_norm '" + norm + "' is not implemented");
      if (act != "silu") throw std::runtime_error("IR moe_ffn activation '" + act + "' is not implemented");
      bind_tensor(layer_name(n.attrs.at("router").get<std::string>(), layer), gguf);
      for (const char* part : {"gate", "up", "down"})
        bind_tensor(layer_name(n.attrs.at("experts").at(part).get<std::string>(), layer), gguf);
    }
    if (n.op == "ssm_scan") {
      const size_t sn = n.attrs.at("n").get<size_t>();
      const std::pair<const char*, size_t> parts[] = {{"a", sn * sn}, {"b", sn}, {"c", sn}, {"d", 1}};
      for (const auto& [part, numel] : parts) {
        const std::string w = layer_name(ssm_weight(n.attrs, part), layer);
        bind_tensor(w, gguf);
        if (bound(w).numel != numel)
          throw std::runtime_error("IR ssm_scan '" + w + "' has " + std::to_string(bound(w).numel) +
                                   " elements, expected " + std::to_string(numel));
      }
    }
    auto expect = [&](const std::string& w, size_t numel) {
      if (bound(w).numel != numel)
        throw std::runtime_error("IR " + n.op + " '" + w + "' has " + std::to_string(bound(w).numel) +
                                 " elements, expected " + std::to_string(numel));
    };
    if (n.op == "conv1d_causal") {  // weight [channels, kernel], bias [channels] (bound above)
      const size_t C = n.attrs.at("channels").get<size_t>(), K = n.attrs.at("kernel").get<size_t>();
      expect(layer_name(n.attrs.at("weight").get<std::string>(), layer), C * K);
      if (has(n.attrs, "bias")) expect(layer_name(n.attrs.at("bias").get<std::string>(), layer), C);
    }
    if (n.op == "selective_scan") {  // a [channels, d_state] as stored, d [channels] optional
      const size_t D = n.attrs.at("channels").get<size_t>(), N = n.attrs.at("d_state").get<size_t>();
      const std::string a = layer_name(n.attrs.at("a").get<std::string>(), layer);
      bind_tensor(a, gguf);
      expect(a, D * N);
      if (has(n.attrs, "d")) {
        const std::string d = layer_name(n.attrs.at("d").get<std::string>(), layer);
        bind_tensor(d, gguf);
        expect(d, D);
      }
    }
  }
}

const Interpreter::RecSlot& Interpreter::rec_slot(const Node& n, const char* op) const {
  if (!has(n.attrs, "state")) throw std::runtime_error(std::string("IR ") + op + " names no state slot");
  const std::string name = n.attrs.at("state").get<std::string>();
  for (const auto& slot : rec_slots_)
    if (slot.name == name) {
      if (slot.op != op)
        throw std::runtime_error("IR recurrent state '" + name + "' belongs to op '" + slot.op + "', not " + op);
      return slot;
    }
  throw std::runtime_error(std::string("IR ") + op + ": no recurrent state named '" + name + "'");
}

float* Interpreter::rec_tensor(const RecSlot& slot, const char* tensor, int layer, size_t floats) {
  for (const auto& rt : slot.tensors)
    if (rt.name == tensor) {
      if (rt.floats != floats)
        throw std::runtime_error("IR recurrent state '" + slot.name + "': " + tensor + " does not fit the op");
      return rec_.data() + static_cast<size_t>(layer) * rec_floats_ + rt.offset;
    }
  throw std::runtime_error("IR recurrent state '" + slot.name + "' has no tensor '" + tensor + "'");
}

int Interpreter::ts_seg_after(int p) const {
  if (p < 0) return 0;
  return ts_seg_.at(p) + (state_k_ > 0 && ts_sidx_.at(p) == state_k_ - 1 ? 1 : 0);
}

int Interpreter::ts_segment_start(int m) const {
  if (m <= 0) return 0;
  // The position after the last state id of block m - 1.
  for (int p = static_cast<int>(ts_seg_.size()) - 1; p >= 0; --p)
    if (ts_seg_[p] == m - 1 && ts_sidx_[p] == state_k_ - 1) return p + 1;
  return static_cast<int>(ts_seg_.size());
}

void Interpreter::ts_evict_below(int seg) {
  if (!evict_ || seg <= evicted_below_) return;
  // Keep every state row and the text rows of segments >= seg; move them down in every block's
  // cache, in order (row w < row r, so the copies never overlap).
  const size_t row = static_cast<size_t>(kv_heads_) * head_dim_;
  size_t w = 0;
  for (size_t r = 0; r < kv_pos_.size(); ++r) {
    const int p = kv_pos_[r];
    if (ts_sidx_.at(p) < 0 && ts_seg_.at(p) < seg) continue;
    if (w != r) {
      for (auto& layer : kv_) {
        std::memcpy(layer.k.data() + w * row, layer.k.data() + r * row, row * sizeof(float));
        std::memcpy(layer.v.data() + w * row, layer.v.data() + r * row, row * sizeof(float));
      }
      kv_pos_[w] = p;
    }
    ++w;
  }
  kv_pos_.resize(w);
  evicted_below_ = seg;
}

void Interpreter::ts_begin(const std::vector<int>& tokens, int start_pos) {
  const int n = static_cast<int>(ts_seg_.size()), T = static_cast<int>(tokens.size());
  if (start_pos < 0 || start_pos > n)
    throw std::runtime_error("IR state tokens: a forward at position " + std::to_string(start_pos) +
                             ", but the sequence holds " + std::to_string(n) + " positions");
  // The segment the first token lands in, and how much of a state block is open before it.
  const int m = ts_seg_after(start_pos - 1);
  int open = 0;
  if (start_pos > 0 && ts_sidx_[start_pos - 1] >= 0 && ts_sidx_[start_pos - 1] < state_k_ - 1)
    open = ts_sidx_[start_pos - 1] + 1;
  // Ids alone fix s(p) and c(p): a state id must be the next slot of an open block (S_0 opens
  // one), a text id must not interrupt one. Checked before anything changes.
  std::vector<int> segs(T), sidx(T);
  int seg = m;
  for (int t = 0; t < T; ++t) {
    const int id = tokens[t];
    if (id >= vocab_ && state_k_ > 0) {
      const int j = id - vocab_;
      if (j >= state_k_)
        throw std::runtime_error("IR state tokens: id " + std::to_string(id) + " is past the " + std::to_string(state_k_) +
                                 " state slots (vocab " + std::to_string(vocab_) + ")");
      if (j != open)
        throw std::runtime_error("IR state tokens: state id S_" + std::to_string(j) + " at position " +
                                 std::to_string(start_pos + t) + ", expected " +
                                 (open == 0 ? std::string("S_0 to open a block") : "S_" + std::to_string(open)));
      segs[t] = seg;
      sidx[t] = j;
      if (++open == state_k_) {
        open = 0;
        ++seg;
      }
    } else {
      if (open != 0)
        throw std::runtime_error("IR state tokens: text id " + std::to_string(id) + " at position " +
                                 std::to_string(start_pos + t) + " inside an incomplete state block (S_0 .. S_" +
                                 std::to_string(open - 1) + " of " + std::to_string(state_k_) + ")");
      segs[t] = seg;
      sidx[t] = -1;
    }
  }
  // A rewind into a segment whose text rows are gone can resume only at that segment's start
  // (resume_point); the rows of segments >= m all lie at or past it and are dropped below.
  if (turn_state_ && m < evicted_below_) {
    const int s0 = ts_segment_start(m);
    if (start_pos != s0)
      throw std::runtime_error("IR state tokens: position " + std::to_string(start_pos) + " is inside turn " +
                               std::to_string(m) + ", whose text rows were evicted; resume from its start, position " +
                               std::to_string(s0) + " (resume_point)");
    evicted_below_ = m;
  }
  ts_seg_.resize(start_pos);
  ts_sidx_.resize(start_pos);
  ts_seg_.insert(ts_seg_.end(), segs.begin(), segs.end());
  ts_sidx_.insert(ts_sidx_.end(), sidx.begin(), sidx.end());
  if (!turn_state_) return;
  while (!kv_pos_.empty() && kv_pos_.back() >= start_pos) kv_pos_.pop_back();
  // Rows no query of this call can see: earlier segments' text.
  if (evict_ && T > 0) ts_evict_below(segs[0]);
  kv_row0_ = static_cast<int>(kv_pos_.size());
  if (kv_row0_ + T > max_len_)
    throw std::runtime_error("IR attention: " + std::to_string(kv_row0_ + T) + " kv rows exceed the kv state (" +
                             std::to_string(max_len_) + ")");
  // Visible rows per query (section 1.3), ascending: held rows, then this call's rows up to t.
  ts_vis_.assign(T, {});
  for (int t = 0; t < T; ++t) {
    const int s = segs[t];
    auto visible = [&](int j) { return ts_seg_[j] == s || (ts_sidx_[j] >= 0 && ts_seg_[j] < s); };
    std::vector<int>& v = ts_vis_[t];
    for (int r = 0; r < kv_row0_; ++r)
      if (visible(kv_pos_[r])) v.push_back(r);
    for (int u = 0; u <= t; ++u)
      if (visible(start_pos + u)) v.push_back(kv_row0_ + u);
  }
}

int Interpreter::resume_point(int want) const {
  if (turn_state_ && evict_ && want > 0 && want <= static_cast<int>(ts_seg_.size())) {
    const int m = ts_seg_after(want - 1);
    if (m < evicted_below_) return ts_segment_start(m);
  }
  if (rec_floats_ == 0 || want <= 0) return std::max(want, 0);
  if (want == rec_pos_) return want;
  auto below = [want](const auto& snaps) {
    auto it = snaps.upper_bound(want);  // the first snapshot past want
    return it == snaps.begin() ? 0 : std::prev(it)->first;
  };
#ifdef HAS_CUDA
  if (gpu_) return below(dsnaps_);
#endif
  return below(rec_snaps_);
}

// The device path keeps the same bookkeeping (keys, kept rows, eviction) over
// device buffers: drec_ is the live state, each snapshot a device copy, and
// every copy is device to device on the interpreter's stream.
void Interpreter::rec_begin(int start_pos, int T, bool all_rows) {
#ifdef HAS_CUDA
  const size_t bytes = rec_.size() * sizeof(float);
#endif
  if (start_pos == 0) {  // a new sequence
#ifdef HAS_CUDA
    if (gpu_) {
      gpuErrchk(cudaMemcpyAsync(drec_, drec_init_, bytes, cudaMemcpyDeviceToDevice, stream_));
      while (!dsnaps_.empty()) dsnap_drop(dsnaps_.begin());
    } else
#endif
    {
      for (int l = 0; l < n_layers_; ++l) std::copy(rec_init_.begin(), rec_init_.end(), rec_.begin() + l * rec_floats_);
      rec_snaps_.clear();
    }
  } else if (start_pos != rec_pos_) {
    auto missing = [&] {
      return std::runtime_error("IR recurrent state holds " + std::to_string(rec_pos_) + " tokens and has no snapshot at " +
                                std::to_string(start_pos) + " (resume_point says where a prefix can resume)");
    };
#ifdef HAS_CUDA
    if (gpu_) {
      auto it = dsnaps_.find(start_pos);
      if (it == dsnaps_.end()) throw missing();
      gpuErrchk(cudaMemcpyAsync(drec_, it->second, bytes, cudaMemcpyDeviceToDevice, stream_));
    } else
#endif
    {
      auto it = rec_snaps_.find(start_pos);
      if (it == rec_snaps_.end()) throw missing();
      rec_ = it->second;
    }
  }
  // Snapshots past start_pos belong to the continuation being replaced.
#ifdef HAS_CUDA
  if (gpu_)
    while (!dsnaps_.empty() && dsnaps_.rbegin()->first > start_pos) dsnap_drop(std::prev(dsnaps_.end()));
  else
#endif
    rec_snaps_.erase(rec_snaps_.upper_bound(start_pos), rec_snaps_.end());
  rec_pos_ = -1;  // until this call completes
  // Rows whose state is kept: every row of an all_rows call (speculative
  // verification), else the second-to-last row (the same prompt again).
  rec_snap_rows_.assign(T, -1);
  int kept = 0;
  for (int t = 0; t + 1 < T; ++t)
    if (all_rows || t == T - 2) rec_snap_rows_[t] = kept++;
  // Each kept row starts as the current state (a slot no node updates keeps it).
#ifdef HAS_CUDA
  if (gpu_) {
    if (kept > 0) grow(drec_rows_, static_cast<size_t>(kept) * rec_.size());
    for (int k = 0; k < kept; ++k)
      gpuErrchk(cudaMemcpyAsync(drec_rows_.alloc + k * rec_.size(), drec_, bytes, cudaMemcpyDeviceToDevice, stream_));
    return;
  }
#endif
  rec_rows_.resize(static_cast<size_t>(kept) * rec_.size());
  for (int k = 0; k < kept; ++k) std::copy(rec_.begin(), rec_.end(), rec_rows_.begin() + k * rec_.size());
}

void Interpreter::rec_row_done(const RecSlot& slot, int layer, int t) {
  const int k = rec_snap_rows_[t];
  if (k < 0) return;
  // This slot only: another slot of the block may be at another row.
  const size_t off = static_cast<size_t>(layer) * rec_floats_ + slot.tensors.front().offset;
  const size_t floats = slot.tensors.back().offset + slot.tensors.back().floats - slot.tensors.front().offset;
#ifdef HAS_CUDA
  if (gpu_) {
    gpuErrchk(cudaMemcpyAsync(drec_rows_.alloc + k * rec_.size() + off, drec_ + off, floats * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream_));
    return;
  }
#endif
  std::copy(rec_.begin() + off, rec_.begin() + off + floats, rec_rows_.begin() + k * rec_.size() + off);
}

void Interpreter::rec_end(int start_pos, int T) {
#ifdef HAS_CUDA
  if (gpu_) {
    const size_t bytes = rec_.size() * sizeof(float);
    for (int t = 0; t + 1 < T; ++t) {
      const int k = rec_snap_rows_[t];
      if (k < 0) continue;
      gpuErrchk(cudaMemcpyAsync(dsnap_take(start_pos + t + 1), drec_rows_.alloc + k * rec_.size(), bytes,
                                cudaMemcpyDeviceToDevice, stream_));
    }
    gpuErrchk(cudaMemcpyAsync(dsnap_take(start_pos + T), drec_, bytes, cudaMemcpyDeviceToDevice, stream_));
    while (dsnaps_.size() > rec_max_snaps_) dsnap_drop(dsnaps_.begin());  // the oldest positions go first
    rec_pos_ = start_pos + T;
    return;
  }
#endif
  for (int t = 0; t + 1 < T; ++t) {
    const int k = rec_snap_rows_[t];
    if (k < 0) continue;
    rec_snaps_[start_pos + t + 1].assign(rec_rows_.begin() + k * rec_.size(), rec_rows_.begin() + (k + 1) * rec_.size());
  }
  rec_snaps_[start_pos + T] = rec_;
  while (rec_snaps_.size() > rec_max_snaps_) rec_snaps_.erase(rec_snaps_.begin());  // the oldest positions go first
  rec_pos_ = start_pos + T;
  std::vector<float>().swap(rec_rows_);
}

const Interpreter::Weight& Interpreter::bound(const std::string& name) const {
  auto it = weights_.find(name);
  if (it == weights_.end()) throw std::runtime_error("IR weight '" + name + "' was not bound");
  return it->second;
}

const std::vector<float>& Interpreter::weight(const std::string& name) const {
  const Weight& w = bound(name);
  if (w.quantised()) throw std::runtime_error("IR weight '" + name + "' is held quantised but read whole");
  return w.f32;
}

void Interpreter::run_nodes(const std::vector<Node>& nodes, int layer, Env& env, int T, int start_pos,
                            bool /*head_last_only*/) {
  auto value = [&](const Node& n, size_t k, bool want_list) -> Buf& {
    auto it = env.find(n.in.at(k));
    if (it == env.end()) throw std::runtime_error("IR op '" + n.op + "': input '" + n.in.at(k) + "' is not defined");
    if (it->second.is_list != want_list)
      throw std::runtime_error("IR op '" + n.op + "': input '" + n.in.at(k) + "' is " +
                               (want_list ? "a tensor, not a list" : "a list, not a tensor"));
    return it->second;
  };
  auto in = [&](const Node& n, size_t k) -> Buf& { return value(n, k, false); };
  auto list_in = [&](const Node& n, size_t k) -> Buf& { return value(n, k, true); };
  auto wname = [&](const Node& n) {
    std::string w = layer_name(n.attrs.at("weight").get<std::string>(), layer);
    if (!weights_.count(w) && has(n.attrs, "tied_to")) w = layer_name(n.attrs.at("tied_to").get<std::string>(), layer);
    return w;
  };
  for (const auto& n : nodes) {
    if (n.op == "embedding") {
      const auto& tokens = in(n, 0);  // token ids carried as floats (exact below 2^24)
      const std::string wn = wname(n);
      const Weight& W = bound(wn);
      const int d = static_cast<int>(shapes_.at(wn).at(0));
      Buf out{std::vector<float>(static_cast<size_t>(T) * d), d};
      // Extended embedding (state_weight): an id in [vocab, vocab + k) reads row id - vocab of the
      // state rows; without one, an id past the table is refused. EXACT.
      const bool ext = has(n.attrs, "state_weight");
      for (int t = 0; t < T; ++t) {
        const int id = static_cast<int>(tokens.v[t]);
        const Weight* src = &W;
        int row = id;
        if (ext && id >= vocab_) {
          if (id >= vocab_ + state_k_)
            throw std::runtime_error("IR embedding: token id " + std::to_string(id) + " is past the state slots (vocab " +
                                     std::to_string(vocab_) + ", k " + std::to_string(state_k_) + ")");
          src = &bound(state_weight_);
          row = id - vocab_;
        } else if (id < 0 || static_cast<size_t>(id + 1) * d > W.numel) {
          throw std::runtime_error("IR embedding: token id " + std::to_string(id) + " out of range");
        }
        float* dst = out.v.data() + static_cast<size_t>(t) * d;
        if (src->quantised())
          dequant_span_to_f32(src->type, src->qp, static_cast<size_t>(row) * d, d, dst);
        else
          std::memcpy(dst, src->f32.data() + static_cast<size_t>(row) * d, d * sizeof(float));
      }
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "pos_embedding") {  // x + position_embd[pos], pos absolute. EXACT.
      Buf x = in(n, 0);
      const std::string wn = wname(n);
      const auto& P = weight(wn);
      const int d = x.width;
      if (shapes_.at(wn).at(0) != d) throw std::runtime_error("IR pos_embedding '" + wn + "': width differs from x");
      if (static_cast<size_t>(start_pos + T) * d > P.size())
        throw std::runtime_error("IR pos_embedding: position " + std::to_string(start_pos + T - 1) + " exceeds '" + wn + "'");
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < d; ++i) x.v[static_cast<size_t>(t) * d + i] += P[static_cast<size_t>(start_pos + t) * d + i];
      env[n.out.at(0)] = std::move(x);
    } else if (n.op == "layernorm") {  // (x - mean) / sqrt(biased var + eps) * w + b. ROUNDING.
      const Buf& x = in(n, 0);
      const auto& w = weight(wname(n));
      const auto& b = weight(layer_name(n.attrs.at("bias").get<std::string>(), layer));
      const float eps = n.attrs.value("eps", 1e-5f);
      const int d = x.width;
      Buf out{std::vector<float>(x.v.size()), d};
      for (int t = 0; t < T; ++t) {
        const float* r = x.v.data() + static_cast<size_t>(t) * d;
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < d; ++i) mean += r[i];
        mean /= d;
        for (int i = 0; i < d; ++i) var += (r[i] - mean) * (r[i] - mean);
        var /= d;
        const float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
        for (int i = 0; i < d; ++i)
          out.v[static_cast<size_t>(t) * d + i] = (r[i] - static_cast<float>(mean)) * inv * w[i] + b[i];
      }
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "split") {  // columns of [T, sum(sizes)] into the outputs, in order. EXACT.
      const Buf& x = in(n, 0);
      const auto sizes = n.attrs.at("sizes").get<std::vector<int>>();
      int total = 0;
      for (int s : sizes) total += s;
      if (total != x.width || sizes.size() != n.out.size())
        throw std::runtime_error("IR split: sizes do not match the input width or the outputs");
      int col = 0;
      for (size_t o = 0; o < sizes.size(); ++o) {
        Buf part{std::vector<float>(static_cast<size_t>(T) * sizes[o]), sizes[o]};
        for (int t = 0; t < T; ++t)
          std::memcpy(part.v.data() + static_cast<size_t>(t) * sizes[o], x.v.data() + static_cast<size_t>(t) * x.width + col,
                      sizes[o] * sizeof(float));
        col += sizes[o];
        env[n.out[o]] = std::move(part);
      }
    } else if (n.op == "gelu") {  // tanh: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))); none: 0.5x(1 + erf(x/sqrt2)).
      const Buf& x = in(n, 0);
      const bool tanh_form = n.attrs.value("approx", "none") == "tanh";
      Buf out{std::vector<float>(x.v.size()), x.width};
      const float c = std::sqrt(2.0f / 3.14159265358979323846f);
      for (size_t i = 0; i < x.v.size(); ++i) {
        const float v = x.v[i];
        out.v[i] = tanh_form ? 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)))
                             : 0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f)));
      }
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "rmsnorm") {
      const Buf& x = in(n, 0);
      Buf out{{}, x.width};
      out.v.resize(x.v.size());
      // group g: each consecutive g columns normalised on their own, one
      // weight [g] shared (per-head q/k norm); a row of rows of width g.
      const int g = has(n.attrs, "group") ? n.attrs.at("group").get<int>() : x.width;
      const auto& w = weight(wname(n));
      if (g <= 0 || x.width % g != 0 || w.size() != static_cast<size_t>(g))
        throw std::runtime_error("IR rmsnorm '" + wname(n) + "': group " + std::to_string(g) + " does not fit width " +
                                 std::to_string(x.width) + " and weight [" + std::to_string(w.size()) + "]");
      rmsnorm_batch_cpu(x.v, w, out.v, T * (x.width / g), g, n.attrs.value("eps", 1e-5f));
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "linear") {
      const Buf& x = in(n, 0);
      const std::string wn = wname(n);
      const Weight& W = bound(wn);
      if (shapes_.at(wn).at(0) != x.width)
        throw std::runtime_error("IR linear '" + wn + "': input width " + std::to_string(x.width) +
                                 " does not match the weight's ne0 " + std::to_string(shapes_.at(wn).at(0)));
      const int in_dim = x.width, out_dim = static_cast<int>(W.numel / in_dim);
      Buf out{std::vector<float>(static_cast<size_t>(T) * out_dim), out_dim};
      if (W.quantised())  // one row of blocks at a time; equal to the f32 path (graph_ir.h)
        matmul_rows_f32_batch_cpu(
            [&](int64_t o, float* row) {
              dequant_span_to_f32(W.type, W.qp, static_cast<size_t>(o) * in_dim, in_dim, row);
            },
            x.v.data(), out.v.data(), T, out_dim, in_dim);
      else
        matmul_f32_f32_batch_cpu(W.f32, x.v, out.v, T, out_dim, in_dim);
      if (has(n.attrs, "bias")) {
        const auto& b = weight(layer_name(n.attrs.at("bias").get<std::string>(), layer));
        for (int t = 0; t < T; ++t)
          for (int o = 0; o < out_dim; ++o) out.v[static_cast<size_t>(t) * out_dim + o] += b[o];
      }
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "rope") {
      Buf q = in(n, 0), k = in(n, 1);
      const int hd = n.attrs.at("head_dim").get<int>();
      const auto& freqs = rope_table(layer);
      if (!rope_neox_ && rope_dim_ == head_dim_) {  // full interleaved: the llama path's kernel
        apply_rope_batch_cpu(q.v, k.v, T, q.width / hd, k.width / hd, hd, start_pos, freqs, max_len_,
                             /*use_adjacent_pairing=*/true);
      } else {  // NEOX pairing or partial rotary
        if (start_pos < 0 || start_pos + T > max_len_)
          throw std::runtime_error("IR rope: position " + std::to_string(start_pos + T - 1) + " exceeds the rope table (" +
                                   std::to_string(max_len_) + ")");
        if (q.width % hd != 0 || k.width % hd != 0) throw std::runtime_error("IR rope: q/k widths are not whole heads");
        rope_rows_cpu(q.v, T, q.width / hd, hd, rope_dim_, rope_neox_, start_pos, freqs);
        rope_rows_cpu(k.v, T, k.width / hd, hd, rope_dim_, rope_neox_, start_pos, freqs);
      }
      env[n.out.at(0)] = std::move(q);
      env[n.out.at(1)] = std::move(k);
    } else if (n.op == "attention") {
      const Buf &q = in(n, 0), &k = in(n, 1), &v = in(n, 2);
      const int hd = n.attrs.at("head_dim").get<int>(), hq = n.attrs.at("heads").get<int>();
      const int hkv = n.attrs.at("kv_heads").get<int>();
      if (hd != head_dim_ || hkv != kv_heads_) throw std::runtime_error("IR attention shape differs from its kv state");
      // Under a recursion step (mor_step) the rows go to the application's compact cache, at
      // attn_row_, and the query reads the rows before and at it; under turn_state they go to the
      // compact cache at kv_row0_ and each query reads its visible rows (ts_vis_); elsewhere rows
      // sit at their positions.
      const bool ts = n.attrs.value("mask", "causal") == "turn_state";
      const int kpos = ts ? kv_row0_ : attn_row_ >= 0 ? attn_row_ : start_pos;
      if (kpos + T > max_len_)
        throw std::runtime_error("IR attention: position " + std::to_string(kpos + T) + " exceeds the kv state (" +
                                 std::to_string(max_len_) + ")");
      KVCacheLayer& cache = kv_.at(layer);
      const size_t row = static_cast<size_t>(hkv) * hd;
      std::memcpy(cache.k.data() + kpos * row, k.v.data(), static_cast<size_t>(T) * row * sizeof(float));
      std::memcpy(cache.v.data() + kpos * row, v.v.data(), static_cast<size_t>(T) * row * sizeof(float));
      const float scale = n.attrs.contains("scale") ? n.attrs.at("scale").get<float>() : 1.0f / std::sqrt(float(hd));
      Buf out{{}, hq * hd};
      int window = 0, sinks = 0;
      attention_window(n, layer, window, sinks);
      if (ts)
        attention_rows_batch_cpu(q.v, cache, out.v, T, ts_vis_, hq, hkv, hd, scale);
      else if (window > 0)
        attention_window_batch_cpu(q.v, cache, out.v, T, start_pos, hq, hkv, hd, scale, window, sinks);
      else
        attention_batch_cpu(q.v, cache, out.v, T, kpos, hq, hkv, hd, scale);
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "add" || n.op == "mul") {
      const Buf &a = in(n, 0), &b = in(n, 1);
      if (a.v.size() != b.v.size()) throw std::runtime_error("IR " + n.op + ": operand sizes differ");
      Buf out{std::vector<float>(a.v.size()), a.width};
      if (n.op == "add")
        for (size_t i = 0; i < a.v.size(); ++i) out.v[i] = a.v[i] + b.v[i];
      else
        for (size_t i = 0; i < a.v.size(); ++i) out.v[i] = a.v[i] * b.v[i];
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "moe_ffn") {
      const Buf& x = in(n, 0);
      const auto& ex = n.attrs.at("experts");
      const std::string gn = layer_name(ex.at("gate").get<std::string>(), layer);
      const auto& R = weight(layer_name(n.attrs.at("router").get<std::string>(), layer));
      const auto& G = weight(gn);
      const auto& U = weight(layer_name(ex.at("up").get<std::string>(), layer));
      const auto& D = weight(layer_name(ex.at("down").get<std::string>(), layer));
      const int E = n.attrs.at("n_experts").get<int>(), k = n.attrs.at("top_k").get<int>();
      const auto& gs = shapes_.at(gn);  // ggml order {d, d_ff, E}
      if (gs.size() != 3 || gs[0] != x.width || gs[2] != E)
        throw std::runtime_error("IR moe_ffn '" + gn + "': expert stack shape does not match d and n_experts");
      const int d = x.width, d_ff = static_cast<int>(gs[1]);
      const bool renorm = n.attrs.value("gate_norm", "renorm") == "renorm";
      Buf out{std::vector<float>(static_cast<size_t>(T) * d), d};
#pragma omp parallel for schedule(static)
      for (int t = 0; t < T; ++t)
        moe_ffn_f32_cpu(x.v.data() + static_cast<size_t>(t) * d, d, d_ff, E, k, R.data(), G.data(), U.data(), D.data(),
                        out.v.data() + static_cast<size_t>(t) * d, renorm);
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "silu") {
      const Buf& x = in(n, 0);
      Buf out{std::vector<float>(x.v.size()), x.width};
      silu_cpu(x.v, out.v);
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "sub") {  // a - b. EXACT.
      const Buf &a = in(n, 0), &b = in(n, 1);
      if (a.v.size() != b.v.size() || a.width != b.width) throw std::runtime_error("IR sub: operand shapes differ");
      Buf out{std::vector<float>(a.v.size()), a.width};
      cpu::sub(a.v.data(), b.v.data(), out.v.data(), a.v.size());
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "relu" || n.op == "sigmoid") {  // relu EXACT; sigmoid ROUNDING.
      const Buf& x = in(n, 0);
      Buf out{std::vector<float>(x.v.size()), x.width};
      (n.op == "relu" ? cpu::relu : cpu::sigmoid)(x.v.data(), out.v.data(), x.v.size());
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "affine") {  // (x * scale) + shift. EXACT.
      const Buf& x = in(n, 0);
      Buf out{std::vector<float>(x.v.size()), x.width};
      cpu::affine(x.v.data(), out.v.data(), x.v.size(), at_layer(n.attrs.at("scale"), layer).get<float>(),
                  at_layer(n.attrs.at("shift"), layer).get<float>());
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "scale") {  // (x * g) + b per feature, g and b [d]. EXACT.
      const Buf& x = in(n, 0);
      const std::string wn = wname(n);
      const auto& g = weight(wn);
      const int d = x.width;
      if (d <= 0 || g.size() != static_cast<size_t>(d))
        throw std::runtime_error("IR scale '" + wn + "' has " + std::to_string(g.size()) + " values for width " +
                                 std::to_string(d));
      const float* b = nullptr;
      if (has(n.attrs, "bias")) {
        const std::string bn = layer_name(n.attrs.at("bias").get<std::string>(), layer);
        const auto& B = weight(bn);
        if (B.size() != static_cast<size_t>(d))
          throw std::runtime_error("IR scale bias '" + bn + "' has " + std::to_string(B.size()) + " values for width " +
                                   std::to_string(d));
        b = B.data();
      }
      Buf out{std::vector<float>(x.v.size()), d};
      cpu::scale(x.v.data(), g.data(), b, out.v.data(), static_cast<int>(x.v.size() / d), d);
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "mul_col") {  // x [T, w] * c [T, 1]. EXACT.
      const Buf &x = in(n, 0), &c = in(n, 1);
      if (c.width != 1 || c.v.size() != static_cast<size_t>(T) || x.v.size() != static_cast<size_t>(T) * x.width)
        throw std::runtime_error("IR mul_col: c must be [T, 1] beside x [T, w]");
      Buf out{std::vector<float>(x.v.size()), x.width};
      cpu::mul_col(x.v.data(), c.v.data(), out.v.data(), T, x.width);
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "rms_row") {  // [T, w] -> [T, 1]: sqrt(ss / w + eps). ROUNDING.
      const Buf& x = in(n, 0);
      if (x.width <= 0 || x.v.size() != static_cast<size_t>(T) * x.width)
        throw std::runtime_error("IR rms_row: input is not [T, w]");
      Buf out{std::vector<float>(static_cast<size_t>(T)), 1};
      cpu::rms_row(x.v.data(), out.v.data(), T, x.width, n.attrs.at("eps").get<float>());
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "expr") {  // runtime-defined elementwise op (graph_ir_expr.h). ROUNDING.
      std::vector<const float*> ptrs;
      const Buf& x0 = in(n, 0);
      for (size_t k = 0; k < n.in.size(); ++k) {
        const Buf& a = in(n, k);
        if (a.width != x0.width || a.v.size() != x0.v.size())
          throw std::runtime_error("IR expr: inputs must share one shape [T, w]");
        ptrs.push_back(a.v.data());
      }
      if (x0.width <= 0 || x0.v.size() % x0.width) throw std::runtime_error("IR expr: input is not [T, w]");
      Buf out{std::vector<float>(x0.v.size()), x0.width};
      expr::eval(expr::cached(n.attrs.at("expr").get<std::string>(), n.in.size()), ptrs,
                 static_cast<int>(x0.v.size() / x0.width), x0.width, out.v.data());
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "list") {  // the inputs, in order, as one list value. EXACT.
      Buf out;
      out.is_list = true;
      for (size_t k = 0; k < n.in.size(); ++k) out.items.push_back(in(n, k));
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "append") {  // the list with x added at the end. EXACT.
      const Buf& x = in(n, 1);
      // The usual form appends in place (out names the list): no copy.
      Buf out;
      if (n.out.at(0) == n.in.at(0))
        out = std::move(list_in(n, 0));
      else
        out = list_in(n, 0);
      out.items.push_back(x);
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "depth_attend") {  // attention over depth (AttnRes block form). ROUNDING.
      const Buf& L = list_in(n, 0);
      std::vector<const float*> src;
      int d = 0;
      auto add_source = [&](const Buf& s) {
        if (d == 0) d = s.width;
        if (s.width != d || d <= 0 || s.v.size() != static_cast<size_t>(T) * d)
          throw std::runtime_error("IR depth_attend: sources must all be [T, d]");
        src.push_back(s.v.data());
      };
      for (const Buf& s : L.items) add_source(s);
      for (size_t k = 1; k < n.in.size(); ++k) add_source(in(n, k));
      if (src.empty()) throw std::runtime_error("IR depth_attend: no sources");
      const float* w = nullptr;
      if (src.size() > 1) {  // one source: out = s_0 and the pseudo-query is not read
        const std::string wn = layer_name(n.attrs.at("weight").get<std::string>(), layer);
        if (!weights_.count(wn)) throw std::runtime_error("IR depth_attend: pseudo-query '" + wn + "' is not in the GGUF");
        const auto& W = weight(wn);
        if (W.size() != static_cast<size_t>(d))
          throw std::runtime_error("IR depth_attend: pseudo-query '" + wn + "' is not [d]");
        w = W.data();
      }
      Buf out{std::vector<float>(static_cast<size_t>(T) * d), d};
      cpu::depth_attend(src, w, out.v.data(), T, d, n.attrs.value("eps", 1e-5f));
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "linear_attention") {  // elu+1 linear attention over the slot's num, den. ROUNDING.
      const Buf &q = in(n, 0), &k = in(n, 1), &v = in(n, 2);
      const int w = n.attrs.at("heads").get<int>() * n.attrs.at("head_dim").get<int>();
      if (q.width != w || k.width != w || v.width != w)
        throw std::runtime_error("IR linear_attention: q/k/v widths do not match heads * head_dim");
      const RecSlot& slot = rec_slot(n, "linear_attention");
      float* num = rec_tensor(slot, "num", layer, w);
      float* den = rec_tensor(slot, "den", layer, w);
      const float floor = n.attrs.value("den_floor", 1e-8f);
      Buf out{std::vector<float>(static_cast<size_t>(T) * w), w};
      for (int t = 0; t < T; ++t) {  // the decode update for t = 0..T-1, in order
        const size_t r = static_cast<size_t>(t) * w;
        linear_attention_row_cpu(q.v.data() + r, k.v.data() + r, v.v.data() + r, num, den, out.v.data() + r, w, floor);
        rec_row_done(slot, layer, t);
      }
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "ssm_scan") {  // the sequential state-space scan over the slot's s. ROUNDING.
      const Buf& u = in(n, 0);
      const int sn = n.attrs.at("n").get<int>();
      if (u.width != sn) throw std::runtime_error("IR ssm_scan: input width differs from n");
      const RecSlot& slot = rec_slot(n, "ssm_scan");
      float* s = rec_tensor(slot, "s", layer, sn);
      const auto& A = weight(layer_name(ssm_weight(n.attrs, "a"), layer));
      const auto& B = weight(layer_name(ssm_weight(n.attrs, "b"), layer));
      const auto& C = weight(layer_name(ssm_weight(n.attrs, "c"), layer));
      const float D = weight(layer_name(ssm_weight(n.attrs, "d"), layer)).at(0);
      std::vector<float> scratch(sn);
      Buf out{std::vector<float>(static_cast<size_t>(T) * sn), sn};
      for (int t = 0; t < T; ++t) {
        const size_t r = static_cast<size_t>(t) * sn;
        ssm_scan_row_cpu(u.v.data() + r, A.data(), B.data(), C.data(), D, s, scratch.data(), out.v.data() + r, sn);
        rec_row_done(slot, layer, t);
      }
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "conv1d_causal") {  // depthwise causal conv over the slot's previous inputs. ROUNDING.
      const Buf& x = in(n, 0);
      const int C = n.attrs.at("channels").get<int>(), K = n.attrs.at("kernel").get<int>();
      if (x.width != C) throw std::runtime_error("IR conv1d_causal: input width differs from channels");
      const RecSlot& slot = rec_slot(n, "conv1d_causal");
      float* st = rec_tensor(slot, "x", layer, static_cast<size_t>(K - 1) * C);
      const auto& w = weight(layer_name(n.attrs.at("weight").get<std::string>(), layer));
      const float* bias =
          has(n.attrs, "bias") ? weight(layer_name(n.attrs.at("bias").get<std::string>(), layer)).data() : nullptr;
      Buf out{std::vector<float>(static_cast<size_t>(T) * C), C};
      for (int t = 0; t < T; ++t) {
        const size_t r = static_cast<size_t>(t) * C;
        cpu::conv1d_causal_row(x.v.data() + r, w.data(), bias, st, out.v.data() + r, C, K);
        rec_row_done(slot, layer, t);
      }
      env[n.out.at(0)] = std::move(out);
    } else if (n.op == "selective_scan") {  // Mamba-1's scan over the slot's s. ROUNDING.
      const Buf &x = in(n, 0), &dt = in(n, 1), &bm = in(n, 2), &cm = in(n, 3);
      const int D = n.attrs.at("channels").get<int>(), N = n.attrs.at("d_state").get<int>();
      if (x.width != D || dt.width != D || bm.width != N || cm.width != N)
        throw std::runtime_error("IR selective_scan: x/dt must be [T, channels] and B/C [T, d_state]");
      const RecSlot& slot = rec_slot(n, "selective_scan");
      float* s = rec_tensor(slot, "s", layer, static_cast<size_t>(D) * N);
      const auto& A = weight(layer_name(n.attrs.at("a").get<std::string>(), layer));
      const float* d = has(n.attrs, "d") ? weight(layer_name(n.attrs.at("d").get<std::string>(), layer)).data() : nullptr;
      const bool softplus = n.attrs.value("dt_softplus", true);
      Buf out{std::vector<float>(static_cast<size_t>(T) * D), D};
      for (int t = 0; t < T; ++t) {
        cpu::selective_scan_row(x.v.data() + static_cast<size_t>(t) * D, dt.v.data() + static_cast<size_t>(t) * D,
                                bm.v.data() + static_cast<size_t>(t) * N, cm.v.data() + static_cast<size_t>(t) * N,
                                A.data(), d, s, out.v.data() + static_cast<size_t>(t) * D, D, N, softplus);
        rec_row_done(slot, layer, t);
      }
      env[n.out.at(0)] = std::move(out);
    }
  }
}

// One application of a recursion step (ir.hpp depth_route (a)-(d)). The first application of step
// 0 routes every row; the first of step k keeps the step's input x_in; every application runs the
// block's other nodes on the tokens deeper than k, one at a time in position order (T = 1 at its
// absolute position, its k/v row appended to this application's compact cache); after the step's
// last application gate "score" blends each active row with its x_in. Inactive rows keep x_in.
void Interpreter::mor_step(int layer, Env& env, int T, int start_pos) {
  const int s = mor_step_[layer];
  const bool first = layer == 0 || mor_step_[layer - 1] != s;
  const bool last = layer == n_layers_ - 1 || mor_step_[layer + 1] != s;
  auto it = env.find(mor_x_);
  if (it == env.end() || it->second.is_list) throw std::runtime_error("IR depth_route: input '" + mor_x_ + "' is not a tensor");
  Buf& x = it->second;
  const int d = x.width;
  if (s == 0 && first) {
    const auto& W = weight(mor_router_);
    if (shapes_.at(mor_router_).at(0) != d || W.size() != static_cast<size_t>(mor_R_) * d)
      throw std::runtime_error("IR depth_route '" + mor_router_ + "' is not [n_rec, d]");
    std::vector<float> logits(static_cast<size_t>(T) * mor_R_);
    matmul_f32_f32_batch_cpu(W, x.v, logits, T, mor_R_, d);  // x W^T, as linear
    mor_depth_.assign(T, 0);
    mor_g_.assign(T, 0.0f);
    for (int t = 0; t < T; ++t) cpu::mor_route(logits.data() + static_cast<size_t>(t) * mor_R_, mor_R_, mor_depth_[t], mor_g_[t]);
    env[mor_out_] = Buf{std::vector<float>(mor_depth_.begin(), mor_depth_.end()), 1};
  }
  if (first) mor_xin_ = x.v;
  std::vector<int> active;
  for (int t = 0; t < T; ++t)
    if (mor_depth_.at(t) > s) active.push_back(t);
  for (int t : active) {
    Env row;
    for (const auto& kv : env)
      if (!kv.second.is_list && kv.second.width > 0 && kv.second.v.size() == static_cast<size_t>(T) * kv.second.width) {
        const auto b = kv.second.v.begin() + static_cast<size_t>(t) * kv.second.width;
        row[kv.first] = Buf{std::vector<float>(b, b + kv.second.width), kv.second.width};
      }
    attn_row_ = static_cast<int>(mor_rows_[layer].size());
    try {
      run_nodes(block_, layer, row, 1, start_pos + t, false);
    } catch (...) {
      attn_row_ = -1;
      throw;
    }
    attn_row_ = -1;
    mor_rows_[layer].push_back(start_pos + t);
    const Buf& y = row.at(mor_x_);
    if (y.width != d) throw std::runtime_error("IR depth_route: the block changes the width of '" + mor_x_ + "'");
    std::copy(y.v.begin(), y.v.end(), x.v.begin() + static_cast<size_t>(t) * d);
  }
  if (last && mor_gate_score_)
    for (int t : active)
      cpu::mor_gate(mor_xin_.data() + static_cast<size_t>(t) * d, x.v.data() + static_cast<size_t>(t) * d, d, mor_g_[t]);
}

std::vector<float> Interpreter::forward(const std::vector<int>& tokens, int start_pos, bool all_rows) {
  const int T = static_cast<int>(tokens.size());
  if (T == 0) return {};
  if (rec_floats_ > 0 && (start_pos < 0 || start_pos + T > max_len_))
    throw std::runtime_error("IR recurrent state: positions " + std::to_string(start_pos) + ".." +
                             std::to_string(start_pos + T - 1) + " exceed model.ctx (" + std::to_string(max_len_) + ")");
#ifdef HAS_CUDA
  if (gpu_) {
    std::vector<float> out = forward_device(tokens, start_pos, all_rows);
    seq_len_ = start_pos + T;
    return out;
  }
#endif
  if (state_k_ > 0 || turn_state_) ts_begin(tokens, start_pos);
  if (rec_floats_ > 0) rec_begin(start_pos, T, all_rows);
  Env env;
  env["tokens"] = Buf{std::vector<float>(tokens.begin(), tokens.end()), 1};
  run_nodes(embed_, -1, env, T, start_pos, false);
  if (!mor_step_.empty()) {
    // Resuming at start_pos (a new sequence, prefix reuse, a rewind): each compact cache keeps
    // the rows of positions below it.
    for (auto& rows : mor_rows_)
      while (!rows.empty() && rows.back() >= start_pos) rows.pop_back();
    mor_depth_.clear();
  }
  for (int l = 0; l < n_layers_; ++l) {
    if (!mor_step_.empty() && mor_step_[l] >= 0)
      mor_step(l, env, T, start_pos);
    else
      run_nodes(block_, l, env, T, start_pos, false);
  }
  if (turn_state_) {
    // Every block has written this call's rows at kv_row0_ ..: they now hold these positions.
    for (int t = 0; t < T; ++t) kv_pos_.push_back(start_pos + t);
    // The text rows of segments before the last token's are invisible to every later query.
    if (evict_) ts_evict_below(ts_seg_.back());
  }
  if (rec_floats_ > 0) rec_end(start_pos, T);
  int rows = T;
  if (!all_rows && T > 1) {  // head ops are row-wise: run them on the last row only
    auto last_row = [T](Buf& b) {
      if (b.width > 0 && b.v.size() == static_cast<size_t>(T) * b.width) b.v.erase(b.v.begin(), b.v.end() - b.width);
    };
    for (auto& kv : env) {
      last_row(kv.second);
      for (Buf& item : kv.second.items) last_row(item);  // a list's tensors
    }
    rows = 1;
  }
  run_nodes(head_, -1, env, rows, start_pos, !all_rows);
  auto it = env.find("logits");
  if (it == env.end()) throw std::runtime_error("IR head produced no 'logits'");
  seq_len_ = start_pos + T;
  return std::move(it->second.v);
}

// Conformance hooks (graph_ir.h). Each reads or sets the CPU state only.
static void cpu_only(bool gpu, const char* what) {
  if (gpu) throw std::runtime_error(std::string("IR ") + what + " is implemented for the CPU path only");
}

int Interpreter::kv_rows_in_use(int application) const {
  if (kv_.empty()) return 0;
  if (turn_state_) return static_cast<int>(kv_pos_.size());
  if (!mor_step_.empty() && mor_step_.at(application) >= 0) return static_cast<int>(mor_rows_.at(application).size());
  return seq_len_;
}

std::vector<int> Interpreter::kv_positions(int application) const {
  cpu_only(gpu_, "kv_positions");
  if (application < 0 || application >= n_layers_)
    throw std::runtime_error("IR kv_positions: application " + std::to_string(application) + " of " +
                             std::to_string(n_layers_));
  if (kv_.empty()) return {};
  if (turn_state_) return kv_pos_;
  if (!mor_step_.empty() && mor_step_[application] >= 0) return mor_rows_[application];
  std::vector<int> p(static_cast<size_t>(seq_len_));
  for (int i = 0; i < seq_len_; ++i) p[i] = i;
  return p;
}

std::vector<Interpreter::RecurrentTensorInfo> Interpreter::recurrent_tensors() const {
  std::vector<RecurrentTensorInfo> out;
  for (const auto& slot : rec_slots_)
    for (const auto& rt : slot.tensors) out.push_back({slot.name, rt.name, rt.shape});
  return out;
}

std::vector<float> Interpreter::recurrent_values(const std::string& slot, const std::string& tensor,
                                                 int application) const {
  cpu_only(gpu_, "recurrent_values");
  if (application < 0 || application >= n_layers_)
    throw std::runtime_error("IR recurrent_values: application " + std::to_string(application) + " of " +
                             std::to_string(n_layers_));
  for (const auto& s : rec_slots_)
    if (s.name == slot)
      for (const auto& rt : s.tensors)
        if (rt.name == tensor) {
          const auto b = rec_.begin() + static_cast<size_t>(application) * rec_floats_ + rt.offset;
          return std::vector<float>(b, b + rt.floats);
        }
  throw std::runtime_error("IR recurrent_values: no tensor '" + tensor + "' in a slot '" + slot + "'");
}

void Interpreter::truncate(int n) {
  cpu_only(gpu_, "truncate");
  if (n < 0 || n > seq_len_)
    throw std::runtime_error("IR truncate to " + std::to_string(n) + ": the state covers " + std::to_string(seq_len_) +
                             " positions");
  if (rec_floats_ > 0 && rec_pos_ < 0) throw std::runtime_error("IR truncate: the recurrent state is not usable (a call failed)");
  if (resume_point(n) != n)
    throw std::runtime_error("IR truncate to " + std::to_string(n) + ": the state can resume only at " +
                             std::to_string(resume_point(n)) + " (re-prefill from there)");
  if (n == seq_len_) return;
  if (state_k_ > 0 || turn_state_) {
    // n is its segment's start when that segment's text rows are gone (resume_point): every row
    // of segments >= m lies at or past n, so dropping positions >= n leaves segments < m only.
    const int m = n > 0 ? ts_seg_after(n - 1) : 0;
    if (turn_state_ && m < evicted_below_) evicted_below_ = m;
    ts_seg_.resize(n);
    ts_sidx_.resize(n);
    while (!kv_pos_.empty() && kv_pos_.back() >= n) kv_pos_.pop_back();
  }
  for (auto& rows : mor_rows_)
    while (!rows.empty() && rows.back() >= n) rows.pop_back();
  if (rec_floats_ > 0 && n != rec_pos_) {
    if (n == 0) {
      for (int l = 0; l < n_layers_; ++l) std::copy(rec_init_.begin(), rec_init_.end(), rec_.begin() + l * rec_floats_);
      rec_snaps_.clear();
    } else {
      rec_ = rec_snaps_.at(n);
      rec_snaps_.erase(rec_snaps_.upper_bound(n), rec_snaps_.end());
    }
    rec_pos_ = n;
  }
  seq_len_ = n;
}

Interpreter::State Interpreter::save_state() const {
  cpu_only(gpu_, "save_state");
  State s;
  s.seq_len = seq_len_;
  const size_t row = static_cast<size_t>(kv_heads_) * head_dim_;
  for (int l = 0; l < static_cast<int>(kv_.size()); ++l) {
    const size_t n = static_cast<size_t>(kv_rows_in_use(l)) * row;
    s.k.emplace_back(kv_[l].k.begin(), kv_[l].k.begin() + n);
    s.v.emplace_back(kv_[l].v.begin(), kv_[l].v.begin() + n);
  }
  s.ts_seg = ts_seg_;
  s.ts_sidx = ts_sidx_;
  s.kv_pos = kv_pos_;
  s.evicted_below = evicted_below_;
  s.mor_rows = mor_rows_;
  s.mor_depth = mor_depth_;
  s.rec = rec_;
  s.rec_pos = rec_pos_;
  s.rec_snaps = rec_snaps_;
  return s;
}

void Interpreter::load_state(const State& s) {
  cpu_only(gpu_, "load_state");
  if (s.k.size() != kv_.size() || s.v.size() != kv_.size() || s.rec.size() != rec_.size() ||
      s.mor_rows.size() != mor_rows_.size())
    throw std::runtime_error("IR load_state: the state is not this interpreter's");
  for (size_t l = 0; l < kv_.size(); ++l) {
    if (s.k[l].size() > kv_[l].k.size() || s.v[l].size() != s.k[l].size())
      throw std::runtime_error("IR load_state: the kv rows do not fit the cache");
    std::copy(s.k[l].begin(), s.k[l].end(), kv_[l].k.begin());
    std::copy(s.v[l].begin(), s.v[l].end(), kv_[l].v.begin());
  }
  seq_len_ = s.seq_len;
  ts_seg_ = s.ts_seg;
  ts_sidx_ = s.ts_sidx;
  kv_pos_ = s.kv_pos;
  evicted_below_ = s.evicted_below;
  mor_rows_ = s.mor_rows;
  mor_depth_ = s.mor_depth;
  rec_ = s.rec;
  rec_pos_ = s.rec_pos;
  rec_snaps_ = s.rec_snaps;
}

#ifdef HAS_CUDA
// ---------------------------------------------------------------------------
// Device path: the CPU run_nodes above, op for op, on device buffers.

void Interpreter::init_device() {
  gpuErrchk(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("IR CUDA: cublasCreate failed");
  cublasSetStream(cublas_, stream_);
  size_t floats = 0;
  for (auto& w : weights_) {  // the host copies are not used on this path: release them
    if (w.second.quantised())  // the device path binds every weight as f32 (dequant_all_)
      throw std::runtime_error("IR CUDA: weight '" + w.first + "' is still quantised");
    std::vector<float>& host = w.second.f32;
    float* d = nullptr;
    gpuErrchk(cudaMalloc(&d, host.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(d, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
    dweights_[w.first] = d;
    floats += host.size();
    std::vector<float>().swap(host);
  }
  const size_t kv_floats = static_cast<size_t>(max_len_) * kv_heads_ * head_dim_;
  dk_.assign(n_layers_, nullptr);
  dv_.assign(n_layers_, nullptr);
  for (int l = 0; l < n_layers_ && kv_floats > 0; ++l) {
    gpuErrchk(cudaMalloc(&dk_[l], kv_floats * sizeof(float)));
    gpuErrchk(cudaMalloc(&dv_[l], kv_floats * sizeof(float)));
    gpuErrchk(cudaMemset(dk_[l], 0, kv_floats * sizeof(float)));
    gpuErrchk(cudaMemset(dv_[l], 0, kv_floats * sizeof(float)));
  }
  for (const auto& tab : rope_tables_) {  // one per distinct (theta, freq_scale)
    std::vector<float> flat(tab.second.size() * 2);
    for (size_t i = 0; i < tab.second.size(); ++i) {
      flat[2 * i] = tab.second[i].first;
      flat[2 * i + 1] = tab.second[i].second;
    }
    float* d = nullptr;
    gpuErrchk(cudaMalloc(&d, flat.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(d, flat.data(), flat.size() * sizeof(float), cudaMemcpyHostToDevice));
    drope_tables_[tab.first] = d;
  }
  if (rope_theta_ > 0.0f) drope_ = drope_tables_.at({rope_theta_, rope_freq_scale_});
  if (rec_floats_ > 0) {  // rec_ holds every block's initial state here
    const size_t bytes = rec_.size() * sizeof(float);
    gpuErrchk(cudaMalloc(&drec_, bytes));
    gpuErrchk(cudaMalloc(&drec_init_, bytes));
    gpuErrchk(cudaMemcpy(drec_init_, rec_.data(), bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(drec_, rec_.data(), bytes, cudaMemcpyHostToDevice));
  }
  gpuErrchk(cudaMalloc(&dstep_, 2 * sizeof(int)));
  gpuErrchk(cudaHostAlloc(reinterpret_cast<void**>(&hstep_), 2 * sizeof(int), cudaHostAllocDefault));
  gpuErrchk(cudaHostAlloc(reinterpret_cast<void**>(&hlogits_), static_cast<size_t>(vocab_) * sizeof(float),
                          cudaHostAllocDefault));
  Logger::info("[IR] CUDA device path: " + std::to_string(dweights_.size()) + " weights (" +
               std::to_string(floats * sizeof(float) / (1024 * 1024)) + " MiB f32), " +
               (kv_floats > 0 ? std::to_string(n_layers_) + " kv slots of " + std::to_string(max_len_) + " positions"
                              : std::string("no kv slots")) +
               (rec_floats_ > 0 ? " and " + std::to_string(rec_floats_) + " recurrent floats per block" : "") +
               " on the device");
  compile_device();
}

int Interpreter::slot(const std::string& name) {
  auto it = slots_.find(name);
  if (it != slots_.end()) return it->second;
  const int s = static_cast<int>(slot_names_.size());
  slots_.emplace(name, s);
  slot_names_.push_back(name);
  return s;
}

std::vector<Interpreter::DStep> Interpreter::compile_nodes(const std::vector<Node>& nodes, int layer) {
  static const std::unordered_map<std::string, DOp> kOps = {
      {"embedding", DOp::Embedding}, {"pos_embedding", DOp::PosEmbedding}, {"layernorm", DOp::LayerNorm},
      {"split", DOp::Split},         {"gelu", DOp::Gelu},                  {"rmsnorm", DOp::RmsNorm},
      {"linear", DOp::Linear},       {"rope", DOp::Rope},                  {"attention", DOp::Attention},
      {"add", DOp::Add},             {"mul", DOp::Mul},                    {"moe_ffn", DOp::MoeFfn},
      {"silu", DOp::Silu},           {"sub", DOp::Sub},                    {"relu", DOp::Relu},
      {"sigmoid", DOp::Sigmoid},     {"affine", DOp::Affine},              {"mul_col", DOp::MulCol},
      {"rms_row", DOp::RmsRow},      {"list", DOp::List},                  {"append", DOp::Append},
      {"depth_attend", DOp::DepthAttend}, {"linear_attention", DOp::LinearAttention}, {"ssm_scan", DOp::SsmScan},
      {"expr", DOp::Expr},           {"conv1d_causal", DOp::Conv1dCausal}, {"selective_scan", DOp::SelectiveScan},
      {"scale", DOp::Scale}};
  auto wname = [&](const Node& n) {
    std::string w = layer_name(n.attrs.at("weight").get<std::string>(), layer);
    if (!weights_.count(w) && has(n.attrs, "tied_to")) w = layer_name(n.attrs.at("tied_to").get<std::string>(), layer);
    return w;
  };
  auto elems = [&](const std::string& wn) {
    size_t e = 1;
    for (int64_t s : shapes_.at(wn)) e *= static_cast<size_t>(s);
    return e;
  };
  std::vector<DStep> steps;
  for (const auto& n : nodes) {
    DStep s;
    auto op = kOps.find(n.op);
    if (op == kOps.end()) throw std::runtime_error("IR op '" + n.op + "' has no CUDA kernel in ember's interpreter");
    s.op = op->second;
    s.node = &n;
    for (const auto& name : n.in) s.in.push_back(slot(name));
    for (const auto& name : n.out) s.out.push_back(slot(name));
    switch (s.op) {
      case DOp::Embedding:
      case DOp::PosEmbedding:
      case DOp::RmsNorm:
      case DOp::LayerNorm:
      case DOp::Linear:
        s.wname = wname(n);
        s.w = dweight(s.wname);
        s.w_elems = elems(s.wname);
        s.w_ne0 = static_cast<int>(shapes_.at(s.wname).at(0));
        if (has(n.attrs, "bias") && (s.op == DOp::LayerNorm || s.op == DOp::Linear))
          s.b = dweight(layer_name(n.attrs.at("bias").get<std::string>(), layer));
        if (s.op == DOp::LayerNorm && !s.b) throw std::runtime_error("IR layernorm '" + s.wname + "' has no bias");
        s.eps = n.attrs.value("eps", 1e-5f);
        if (s.op == DOp::RmsNorm && has(n.attrs, "group")) s.n = n.attrs.at("group").get<int>();
        break;
      case DOp::Split:
        s.sizes = n.attrs.at("sizes").get<std::vector<int>>();
        if (s.sizes.size() != n.out.size()) throw std::runtime_error("IR split: sizes do not match the outputs");
        break;
      case DOp::Gelu:
        s.flag = n.attrs.value("approx", "none") == "tanh";
        break;
      case DOp::Rope:
        s.hd = n.attrs.at("head_dim").get<int>();
        s.n_rot = rope_dim_;
        s.neox = rope_neox_;
        s.rope = layer < 0 || rope_theta_layer_.empty()
                     ? drope_
                     : drope_tables_.at({rope_theta_layer_.at(layer), rope_scale_layer_.at(layer)});
        break;
      case DOp::Attention:
        s.hd = n.attrs.at("head_dim").get<int>();
        s.hq = n.attrs.at("heads").get<int>();
        s.hkv = n.attrs.at("kv_heads").get<int>();
        if (s.hd != head_dim_ || s.hkv != kv_heads_) throw std::runtime_error("IR attention shape differs from its kv state");
        s.scale = n.attrs.contains("scale") ? n.attrs.at("scale").get<float>() : 1.0f / std::sqrt(float(s.hd));
        attention_window(n, layer < 0 ? 0 : layer, s.window, s.sinks);
        if (s.window > 0 && layer <= 0)
          Logger::info("[IR] CUDA sliding-window attention (ir_window_attention_cuda): window " +
                       std::to_string(s.window) + ", sinks " + std::to_string(s.sinks) +
                       (window_layer_.empty() && !is_per_layer(n.attrs.value("window", json(0))) ? ""
                                                                                                 : " (block 0)"));
        break;
      case DOp::MoeFfn: {
        const auto& ex = n.attrs.at("experts");
        s.wname = layer_name(ex.at("gate").get<std::string>(), layer);
        s.w = dweight(layer_name(n.attrs.at("router").get<std::string>(), layer));
        s.gate = dweight(s.wname);
        s.up = dweight(layer_name(ex.at("up").get<std::string>(), layer));
        s.down = dweight(layer_name(ex.at("down").get<std::string>(), layer));
        s.E = n.attrs.at("n_experts").get<int>();
        s.top_k = n.attrs.at("top_k").get<int>();
        const auto& gs = shapes_.at(s.wname);  // ggml order {d, d_ff, E}
        if (gs.size() != 3 || gs[2] != s.E)
          throw std::runtime_error("IR moe_ffn '" + s.wname + "': expert stack shape does not match n_experts");
        s.w_ne0 = static_cast<int>(gs[0]);
        s.d_ff = static_cast<int>(gs[1]);
        s.flag = n.attrs.value("gate_norm", "renorm") == "renorm";
        break;
      }
      case DOp::Affine:
        s.scale = at_layer(n.attrs.at("scale"), layer).get<float>();
        s.shift = at_layer(n.attrs.at("shift"), layer).get<float>();
        break;
      case DOp::RmsRow:
        s.eps = n.attrs.at("eps").get<float>();
        break;
      case DOp::DepthAttend:
        // The pseudo-query may be absent (one source never reads it; a call
        // with more sources then fails, as on the CPU).
        s.eps = n.attrs.value("eps", 1e-5f);
        s.wname = layer_name(n.attrs.at("weight").get<std::string>(), layer);
        if (dweights_.count(s.wname)) {
          s.w = dweight(s.wname);
          s.w_elems = elems(s.wname);
        }
        break;
      case DOp::LinearAttention: {
        s.rec = &rec_slot(n, "linear_attention");
        s.hq = n.attrs.at("heads").get<int>();
        s.hd = n.attrs.at("head_dim").get<int>();
        const size_t w = static_cast<size_t>(s.hq) * s.hd;
        // rec_tensor checks the slot's tensors; the device state has the host layout.
        s.rec_a = drec_ + (rec_tensor(*s.rec, "num", layer, w) - rec_.data());
        s.rec_b = drec_ + (rec_tensor(*s.rec, "den", layer, w) - rec_.data());
        s.eps = n.attrs.value("den_floor", 1e-8f);
        break;
      }
      case DOp::SsmScan:
        s.rec = &rec_slot(n, "ssm_scan");
        s.n = n.attrs.at("n").get<int>();
        s.rec_a = drec_ + (rec_tensor(*s.rec, "s", layer, s.n) - rec_.data());
        s.w = dweight(layer_name(ssm_weight(n.attrs, "a"), layer));
        s.b = dweight(layer_name(ssm_weight(n.attrs, "b"), layer));
        s.ssm_c = dweight(layer_name(ssm_weight(n.attrs, "c"), layer));
        s.ssm_d = dweight(layer_name(ssm_weight(n.attrs, "d"), layer));
        break;
      case DOp::Scale:
        s.wname = wname(n);
        s.w = dweight(s.wname);
        s.w_elems = elems(s.wname);
        if (has(n.attrs, "bias")) s.b = dweight(layer_name(n.attrs.at("bias").get<std::string>(), layer));
        break;
      case DOp::Expr:
        s.expr = expr::lower(expr::cached(n.attrs.at("expr").get<std::string>(), n.in.size()));
        break;
      case DOp::Conv1dCausal:
        s.rec = &rec_slot(n, "conv1d_causal");
        s.n = n.attrs.at("channels").get<int>();
        s.kernel = n.attrs.at("kernel").get<int>();
        s.rec_a = drec_ + (rec_tensor(*s.rec, "x", layer, static_cast<size_t>(s.kernel - 1) * s.n) - rec_.data());
        s.wname = layer_name(n.attrs.at("weight").get<std::string>(), layer);
        s.w = dweight(s.wname);
        if (elems(s.wname) != static_cast<size_t>(s.n) * s.kernel)
          throw std::runtime_error("IR conv1d_causal '" + s.wname + "' is not [channels, kernel]");
        if (has(n.attrs, "bias")) s.b = dweight(layer_name(n.attrs.at("bias").get<std::string>(), layer));
        break;
      case DOp::SelectiveScan:
        s.rec = &rec_slot(n, "selective_scan");
        s.n = n.attrs.at("channels").get<int>();
        s.d_state = n.attrs.at("d_state").get<int>();
        s.rec_a = drec_ + (rec_tensor(*s.rec, "s", layer, static_cast<size_t>(s.n) * s.d_state) - rec_.data());
        s.wname = layer_name(n.attrs.at("a").get<std::string>(), layer);
        s.w = dweight(s.wname);
        if (elems(s.wname) != static_cast<size_t>(s.n) * s.d_state)
          throw std::runtime_error("IR selective_scan '" + s.wname + "' is not [channels, d_state]");
        if (has(n.attrs, "d")) s.ssm_d = dweight(layer_name(n.attrs.at("d").get<std::string>(), layer));
        s.flag = n.attrs.value("dt_softplus", true);
        break;
      case DOp::Add:
      case DOp::Mul:
      case DOp::Silu:
      case DOp::Sub:
      case DOp::Relu:
      case DOp::Sigmoid:
      case DOp::MulCol:
      case DOp::List:
      case DOp::Append:
        break;
    }
    if (s.op == DOp::Embedding || s.op == DOp::PosEmbedding) {  // the graph replay's range checks
      int& rows = s.op == DOp::Embedding ? embed_rows_ : pos_rows_;
      rows = static_cast<int>(std::min<size_t>(rows, s.w_elems / s.w_ne0));
    }
    steps.push_back(std::move(s));
  }
  return steps;
}

void Interpreter::compile_device() {
  embed_rows_ = pos_rows_ = std::numeric_limits<int>::max();
  plan_embed_ = compile_nodes(embed_, -1);
  plan_block_.resize(n_layers_);
  for (int l = 0; l < n_layers_; ++l) plan_block_[l] = compile_nodes(block_, l);
  plan_head_ = compile_nodes(head_, -1);
  logits_slot_ = slot("logits");
  dbufs_.assign(slot_names_.size(), DBuf{});  // sized once: steps hold slots, dout returns references
  dspare_.assign(slot_names_.size(), DBuf{});
  ditems_.assign(slot_names_.size(), {});
  Logger::info("[IR] CUDA device plan: " + std::to_string(plan_embed_.size()) + " + " + std::to_string(n_layers_) +
               " x " + std::to_string(block_.size()) + " + " + std::to_string(plan_head_.size()) + " steps over " +
               std::to_string(slot_names_.size()) + " buffer slots");
}

void Interpreter::free_device() {
  auto release = [](std::vector<DBuf>& bufs) {
    for (auto& b : bufs)
      if (b.alloc) cudaFree(b.alloc);
    bufs.clear();
  };
  if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
  graph_exec_ = nullptr;
  release(dbufs_);
  release(dspare_);
  for (auto& items : ditems_) release(items);
  ditems_.clear();
  for (auto& w : dweights_) cudaFree(w.second);
  dweights_.clear();
  for (float* p : dk_) if (p) cudaFree(p);
  for (float* p : dv_) if (p) cudaFree(p);
  dk_.clear();
  dv_.clear();
  for (auto& kv : dsnaps_) cudaFree(kv.second);
  for (float* p : dsnap_pool_) cudaFree(p);
  dsnaps_.clear();
  dsnap_pool_.clear();
  for (auto& tab : drope_tables_) cudaFree(tab.second);
  drope_tables_.clear();
  for (float* p : {drec_, drec_init_, drec_rows_.alloc, dmoe_.alloc, dexpr_.alloc})
    if (p) cudaFree(p);
  drope_ = drec_ = drec_init_ = nullptr;
  drec_rows_ = dmoe_ = dexpr_ = DBuf{};
  if (dtokens_) cudaFree(dtokens_);
  if (dstep_) cudaFree(dstep_);
  if (hstep_) cudaFreeHost(hstep_);
  if (hlogits_) cudaFreeHost(hlogits_);
  dtokens_ = dstep_ = hstep_ = nullptr;
  hlogits_ = nullptr;
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
  cublas_ = nullptr;
  stream_ = nullptr;
}

const float* Interpreter::dweight(const std::string& name) const {
  auto it = dweights_.find(name);
  if (it == dweights_.end()) throw std::runtime_error("IR weight '" + name + "' was not bound");
  return it->second;
}

void Interpreter::grow(DBuf& b, size_t need) {
  if (b.cap >= need) return;
  // Grows only on the first calls of a shape; cudaFree waits for the device.
  // A captured graph may hold the old pointer: it is captured again.
  if (capturing_) throw std::runtime_error("IR CUDA graph: a buffer must grow during capture");
  if (b.alloc) gpuErrchk(cudaFree(b.alloc));
  gpuErrchk(cudaMalloc(&b.alloc, need * sizeof(float)));
  b.cap = need;
  ++alloc_epoch_;
}

Interpreter::DBuf& Interpreter::dout(int slot, int rows, int width, std::initializer_list<const DBuf*> inputs) {
  return dout(slot, rows, width, std::vector<const DBuf*>(inputs));
}

Interpreter::DBuf& Interpreter::dout(int slot, int rows, int width, const std::vector<const DBuf*>& inputs) {
  DBuf* b = &dbufs_[slot];
  for (const DBuf* in : inputs)
    if (in && b->alloc && in->alloc == b->alloc) {
      b = &dspare_[slot];
      pending_.push_back(slot);
      break;
    }
  grow(*b, static_cast<size_t>(rows) * width);
  b->p = b->alloc;
  b->rows = rows;
  b->width = width;
  b->is_list = false;
  return *b;
}

void Interpreter::copy_item(DBuf& dst, const DBuf& x) {
  grow(dst, static_cast<size_t>(x.rows) * x.width);
  dst.p = dst.alloc;
  dst.rows = x.rows;
  dst.width = x.width;
  gpuErrchk(cudaMemcpyAsync(dst.p, x.p, static_cast<size_t>(x.rows) * x.width * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream_));
}

float* Interpreter::dsnap_take(int key) {
  auto it = dsnaps_.find(key);
  if (it != dsnaps_.end()) return it->second;
  float* p = nullptr;
  if (!dsnap_pool_.empty()) {
    p = dsnap_pool_.back();
    dsnap_pool_.pop_back();
  } else {
    gpuErrchk(cudaMalloc(&p, rec_.size() * sizeof(float)));
  }
  dsnaps_.emplace(key, p);
  return p;
}

void Interpreter::dsnap_drop(std::map<int, float*>::iterator it) {
  dsnap_pool_.push_back(it->second);
  dsnaps_.erase(it);
}

void Interpreter::commit() {
  for (int s : pending_) std::swap(dbufs_[s], dspare_[s]);
  pending_.clear();
}

void Interpreter::run_nodes_device(const std::vector<DStep>& steps, int layer, int T, int start_pos) {
  cudaStream_t st = stream_;
  auto value = [&](const DStep& s, size_t k, bool want_list) -> DBuf& {
    DBuf& b = dbufs_[s.in.at(k)];
    if (b.rows == 0)
      throw std::runtime_error("IR op '" + s.node->op + "': input '" + s.node->in.at(k) + "' is not defined");
    if (b.is_list != want_list)
      throw std::runtime_error("IR op '" + s.node->op + "': input '" + s.node->in.at(k) + "' is " +
                               (want_list ? "a tensor, not a list" : "a list, not a tensor"));
    return b;
  };
  auto in = [&](const DStep& s, size_t k) -> DBuf& { return value(s, k, false); };
  auto list_in = [&](const DStep& s, size_t k) -> DBuf& { return value(s, k, true); };
  // A recurrent op over rows [0, T): one launch per run of rows up to each
  // kept row, whose state rec_row_done then copies (the CPU's per-row
  // snapshot, rec_begin). A call without kept rows is one launch.
  auto rec_rows = [&](const DStep& s, auto&& launch) {
    for (int t0 = 0, t = 0; t < T; ++t) {
      const bool kept = t + 1 < T && rec_snap_rows_[t] >= 0;
      if (!kept && t + 1 < T) continue;
      launch(t0, t - t0 + 1);
      if (kept) rec_row_done(*s.rec, layer, t);
      t0 = t + 1;
    }
  };
  for (const DStep& s : steps) {
    switch (s.op) {
      case DOp::Embedding: {
        const int d = s.w_ne0;
        const size_t rows = s.w_elems / d;
        for (int t = 0; t < T; ++t)
          if (htokens_[t] < 0 || static_cast<size_t>(htokens_[t]) >= rows)
            throw std::runtime_error("IR embedding: token id " + std::to_string(htokens_[t]) + " out of range");
        DBuf& out = dout(s.out[0], T, d, {});
        ir_embedding_cuda(s.w, capturing_ ? dstep_ : dtokens_, out.p, T, d, st);  // dstep_[0]: the token
        break;
      }
      case DOp::PosEmbedding: {
        const DBuf& x = in(s, 0);
        const int d = x.width;
        if (s.w_ne0 != d) throw std::runtime_error("IR pos_embedding '" + s.wname + "': width differs from x");
        if (static_cast<size_t>(start_pos + T) * d > s.w_elems)
          throw std::runtime_error("IR pos_embedding: position " + std::to_string(start_pos + T - 1) + " exceeds '" +
                                   s.wname + "'");
        DBuf& out = dout(s.out[0], T, d, {&x});
        ir_pos_embedding_cuda(x.p, s.w, out.p, T, d, start_pos, st, dpos_);
        break;
      }
      case DOp::LayerNorm: {
        const DBuf& x = in(s, 0);
        DBuf& out = dout(s.out[0], T, x.width, {&x});
        ir_layernorm_cuda(x.p, s.w, s.b, out.p, T, x.width, s.eps, st);
        break;
      }
      case DOp::Split: {
        const DBuf& x = in(s, 0);
        int total = 0;
        for (int sz : s.sizes) total += sz;
        if (total != x.width) throw std::runtime_error("IR split: sizes do not match the input width or the outputs");
        int col = 0;
        for (size_t o = 0; o < s.sizes.size(); ++o) {
          DBuf& part = dout(s.out[o], T, s.sizes[o], {&x});
          gpuErrchk(cudaMemcpy2DAsync(part.p, s.sizes[o] * sizeof(float), x.p + col, x.width * sizeof(float),
                                      s.sizes[o] * sizeof(float), T, cudaMemcpyDeviceToDevice, st));
          col += s.sizes[o];
        }
        break;
      }
      case DOp::Gelu: {
        const DBuf& x = in(s, 0);
        DBuf& out = dout(s.out[0], T, x.width, {&x});
        ir_gelu_cuda(x.p, out.p, static_cast<size_t>(T) * x.width, s.flag, st);
        break;
      }
      case DOp::RmsNorm: {  // group g: rows of width g, one weight [g] (per-head q/k norm)
        const DBuf& x = in(s, 0);
        const int g = s.n ? s.n : x.width;
        if (g <= 0 || x.width % g != 0 || s.w_elems != static_cast<size_t>(g))
          throw std::runtime_error("IR rmsnorm '" + s.wname + "': group " + std::to_string(g) + " does not fit width " +
                                   std::to_string(x.width) + " and weight [" + std::to_string(s.w_elems) + "]");
        DBuf& out = dout(s.out[0], T, x.width, {&x});
        rmsnorm_batch_cuda(out.p, x.p, s.w, T * (x.width / g), g, s.eps, st);
        break;
      }
      case DOp::Linear: {
        const DBuf& x = in(s, 0);
        if (s.w_ne0 != x.width)
          throw std::runtime_error("IR linear '" + s.wname + "': input width " + std::to_string(x.width) +
                                   " does not match the weight's ne0 " + std::to_string(s.w_ne0));
        const int out_dim = static_cast<int>(s.w_elems / x.width);
        DBuf& out = dout(s.out[0], T, out_dim, {&x});
        gemm_rm_f32_cuda(cublas_, x.p, s.w, out.p, T, out_dim, x.width, st);
        if (s.b) ir_bias_add_cuda(out.p, s.b, T, out_dim, st);
        break;
      }
      case DOp::Rope: {
        if (start_pos + T > max_len_ || !s.rope)
          throw std::runtime_error("IR rope: position " + std::to_string(start_pos + T - 1) + " exceeds the rope table");
        DBuf* q = &in(s, 0);
        DBuf* k = &in(s, 1);
        // Rotated in place when the outputs reuse the input names (the usual
        // form); otherwise copied to the outputs first.
        for (size_t o = 0; o < 2; ++o) {
          DBuf*& src = o == 0 ? q : k;
          if (s.out.at(o) == s.in.at(o)) continue;
          DBuf& dst = dout(s.out.at(o), T, src->width, {src});
          gpuErrchk(cudaMemcpyAsync(dst.p, src->p, static_cast<size_t>(T) * src->width * sizeof(float),
                                    cudaMemcpyDeviceToDevice, st));
          src = &dst;
        }
        if (!s.neox && s.n_rot == s.hd) {  // full interleaved: the llama path's form
          ir_rope_cuda(q->p, k->p, T, q->width / s.hd, k->width / s.hd, s.hd, start_pos, s.rope, st, dpos_);
        } else {  // NEOX pairing or partial rotary (rope_rows_cpu)
          if (q->width % s.hd != 0 || k->width % s.hd != 0)
            throw std::runtime_error("IR rope: q/k widths are not whole heads");
          ir_rope_rows_cuda(q->p, T, q->width / s.hd, s.hd, s.n_rot, s.neox, start_pos, s.rope, st, dpos_);
          ir_rope_rows_cuda(k->p, T, k->width / s.hd, s.hd, s.n_rot, s.neox, start_pos, s.rope, st, dpos_);
        }
        break;
      }
      case DOp::Attention: {
        const DBuf &q = in(s, 0), &k = in(s, 1), &v = in(s, 2);
        const int hd = s.hd, hq = s.hq, hkv = s.hkv;
        if (k.width != hkv * hd || v.width != hkv * hd || q.width != hq * hd)
          throw std::runtime_error("IR attention: q/k/v widths do not match heads * head_dim");
        if (start_pos + T > max_len_)
          throw std::runtime_error("IR attention: position " + std::to_string(start_pos + T) + " exceeds the kv state (" +
                                   std::to_string(max_len_) + ")");
        const size_t row = static_cast<size_t>(hkv) * hd;
        float* K = dk_[layer];
        float* V = dv_[layer];
        ir_kv_write_cuda(K, V, k.p, v.p, T, static_cast<int>(row), start_pos, st, dpos_);
        DBuf& out = dout(s.out[0], T, hq * hd, {&q, &k, &v});
        if (s.window > 0)  // the cache keeps every position, so the window reads it in place
          ir_window_attention_cuda(q.p, K, V, out.p, T, start_pos, hq, hkv, hd, s.scale, s.window, s.sinks, st, dpos_);
        else if (T == 1 && dpos_)  // the same split kernel as below, its length read from dstep_[1]
          attention_devpos_cuda(q.p, K, V, out.p, hq, hd, s.scale, max_len_, hkv, dstep_, st);
        else if (T == 1)
          attention_decode_at_cuda(q.p, K, V, out.p, hq, start_pos + 1, hd, s.scale, max_len_, hkv, st);
        else
          attention_batch_prefill_cuda(q.p, nullptr, nullptr, K, V, out.p, T, start_pos, max_len_, hq, hkv, hd, s.scale,
                                       st, nullptr);
        break;
      }
      case DOp::Add:
      case DOp::Mul: {
        const DBuf &a = in(s, 0), &b = in(s, 1);
        if (static_cast<size_t>(a.rows) * a.width != static_cast<size_t>(b.rows) * b.width)
          throw std::runtime_error("IR " + s.node->op + ": operand sizes differ");
        DBuf& out = dout(s.out[0], a.rows, a.width, {&a, &b});
        ir_binary_cuda(a.p, b.p, out.p, static_cast<size_t>(a.rows) * a.width, s.op == DOp::Add ? 0 : 1, st);
        break;
      }
      case DOp::MoeFfn: {
        const DBuf& x = in(s, 0);
        if (s.w_ne0 != x.width)
          throw std::runtime_error("IR moe_ffn '" + s.wname + "': expert stack shape does not match d and n_experts");
        const int d = x.width;
        grow(dmoe_, moe_ffn_scratch_floats(T, d, s.d_ff, s.E, s.top_k));
        DBuf& out = dout(s.out[0], T, d, {&x});
        moe_ffn_cuda(cublas_, x.p, T, d, s.d_ff, s.E, s.top_k, s.flag, s.w, s.gate, s.up, s.down, dmoe_.alloc, out.p,
                     st);
        break;
      }
      case DOp::Silu: {
        const DBuf& x = in(s, 0);
        DBuf& out = dout(s.out[0], T, x.width, {&x});
        ir_silu_cuda(x.p, out.p, static_cast<size_t>(T) * x.width, st);
        break;
      }
      case DOp::Sub: {
        const DBuf &a = in(s, 0), &b = in(s, 1);
        if (a.rows != b.rows || a.width != b.width) throw std::runtime_error("IR sub: operand shapes differ");
        DBuf& out = dout(s.out[0], a.rows, a.width, {&a, &b});
        ir_binary_cuda(a.p, b.p, out.p, static_cast<size_t>(a.rows) * a.width, 2, st);
        break;
      }
      case DOp::Relu:
      case DOp::Sigmoid:
      case DOp::Affine: {
        const DBuf& x = in(s, 0);
        DBuf& out = dout(s.out[0], x.rows, x.width, {&x});
        const size_t n = static_cast<size_t>(x.rows) * x.width;
        if (s.op == DOp::Relu)
          ir_relu_cuda(x.p, out.p, n, st);
        else if (s.op == DOp::Sigmoid)
          ir_sigmoid_cuda(x.p, out.p, n, st);
        else
          ir_affine_cuda(x.p, out.p, n, s.scale, s.shift, st);
        break;
      }
      case DOp::Scale: {
        const DBuf& x = in(s, 0);
        if (s.w_elems != static_cast<size_t>(x.width))
          throw std::runtime_error("IR scale '" + s.wname + "' has " + std::to_string(s.w_elems) + " values for width " +
                                   std::to_string(x.width));
        DBuf& out = dout(s.out[0], x.rows, x.width, {&x});
        ir_scale_cuda(x.p, s.w, s.b, out.p, x.rows, x.width, st);
        break;
      }
      case DOp::MulCol: {
        const DBuf &x = in(s, 0), &c = in(s, 1);
        if (c.width != 1 || c.rows != T || x.rows != T) throw std::runtime_error("IR mul_col: c must be [T, 1] beside x [T, w]");
        DBuf& out = dout(s.out[0], T, x.width, {&x, &c});
        ir_mul_col_cuda(x.p, c.p, out.p, T, x.width, st);
        break;
      }
      case DOp::RmsRow: {
        const DBuf& x = in(s, 0);
        if (x.width <= 0 || x.rows != T) throw std::runtime_error("IR rms_row: input is not [T, w]");
        DBuf& out = dout(s.out[0], T, 1, {&x});
        ir_rms_row_cuda(x.p, out.p, T, x.width, s.eps, st);
        break;
      }
      case DOp::List: {  // the inputs, copied in order, as one list value
        std::vector<DBuf>& items = ditems_[s.out[0]];
        if (items.size() < s.in.size()) items.resize(s.in.size());
        for (size_t k = 0; k < s.in.size(); ++k) copy_item(items[k], in(s, k));
        DBuf& L = dbufs_[s.out[0]];
        L.is_list = true;
        L.rows = T;
        L.width = items[0].width;
        L.items = static_cast<int>(s.in.size());
        break;
      }
      case DOp::Append: {  // the list with a copy of x added at the end (in place when out names the list)
        const DBuf& x = in(s, 1);
        const DBuf& src = list_in(s, 0);
        const int o = s.out[0];
        if (o != s.in[0]) {
          std::vector<DBuf>& items = ditems_[o];
          if (items.size() < static_cast<size_t>(src.items)) items.resize(src.items);
          for (int k = 0; k < src.items; ++k) copy_item(items[k], ditems_[s.in[0]][k]);
          DBuf& L = dbufs_[o];
          L.is_list = true;
          L.rows = src.rows;
          L.width = src.width;
          L.items = src.items;
        }
        DBuf& L = dbufs_[o];
        std::vector<DBuf>& items = ditems_[o];
        if (items.size() < static_cast<size_t>(L.items) + 1) items.resize(L.items + 1);
        copy_item(items[L.items], x);
        ++L.items;
        break;
      }
      case DOp::DepthAttend: {  // attention over depth (AttnRes block form)
        const DBuf& L = list_in(s, 0);
        std::vector<const float*> src;
        std::vector<const DBuf*> tensors;  // the extra inputs (aliasing for dout)
        int d = 0;
        auto add_source = [&](const DBuf& b) {
          if (d == 0) d = b.width;
          if (b.width != d || d <= 0 || b.rows != T) throw std::runtime_error("IR depth_attend: sources must all be [T, d]");
          src.push_back(b.p);
        };
        for (int k = 0; k < L.items; ++k) add_source(ditems_[s.in[0]][k]);
        for (size_t k = 1; k < s.in.size(); ++k) {
          tensors.push_back(&in(s, k));
          add_source(*tensors.back());
        }
        if (src.empty()) throw std::runtime_error("IR depth_attend: no sources");
        DBuf& out = dout(s.out[0], T, d, tensors);
        if (src.size() == 1) {  // softmax over one source is the identity: s_0 bit for bit, w not read
          gpuErrchk(cudaMemcpyAsync(out.p, src[0], static_cast<size_t>(T) * d * sizeof(float),
                                    cudaMemcpyDeviceToDevice, st));
          break;
        }
        if (!s.w) throw std::runtime_error("IR depth_attend: pseudo-query '" + s.wname + "' is not in the GGUF");
        if (s.w_elems != static_cast<size_t>(d))
          throw std::runtime_error("IR depth_attend: pseudo-query '" + s.wname + "' is not [d]");
        ir_depth_attend_cuda(src.data(), static_cast<int>(src.size()), s.w, out.p, T, d, s.eps, st);
        break;
      }
      case DOp::LinearAttention: {  // over the slot's num, den (device state)
        const DBuf &q = in(s, 0), &k = in(s, 1), &v = in(s, 2);
        const int w = s.hq * s.hd;
        if (q.width != w || k.width != w || v.width != w || q.rows != T || k.rows != T || v.rows != T)
          throw std::runtime_error("IR linear_attention: q/k/v widths do not match heads * head_dim");
        DBuf& out = dout(s.out[0], T, w, {&q, &k, &v});
        rec_rows(s, [&](int t0, int n) {
          const size_t r = static_cast<size_t>(t0) * w;
          ir_linear_attention_cuda(q.p + r, k.p + r, v.p + r, s.rec_a, s.rec_b, out.p + r, n, w, s.eps, st);
        });
        break;
      }
      case DOp::Expr: {  // a runtime-defined elementwise op (graph_ir_expr.h)
        const DBuf& x0 = in(s, 0);
        std::vector<const float*> ins;
        std::vector<const DBuf*> bufs;
        for (size_t k = 0; k < s.in.size(); ++k) {
          const DBuf& a = in(s, k);
          if (a.width != x0.width || a.rows != x0.rows) throw std::runtime_error("IR expr: inputs must share one shape [T, w]");
          ins.push_back(a.p);
          bufs.push_back(&a);
        }
        if (x0.width <= 0 || x0.rows <= 0) throw std::runtime_error("IR expr: input is not [T, w]");
        grow(dexpr_, ir_expr_scratch_floats(s.expr, x0.rows, x0.width));
        DBuf& out = dout(s.out[0], x0.rows, x0.width, bufs);
        ir_expr_cuda(s.expr, ins.data(), static_cast<int>(ins.size()), dexpr_.alloc, out.p, x0.rows, x0.width, st);
        break;
      }
      case DOp::Conv1dCausal: {  // over the slot's x (device state)
        const DBuf& x = in(s, 0);
        if (x.width != s.n || x.rows != T) throw std::runtime_error("IR conv1d_causal: input width differs from channels");
        DBuf& out = dout(s.out[0], T, s.n, {&x});
        rec_rows(s, [&](int t0, int n) {
          const size_t r = static_cast<size_t>(t0) * s.n;
          ir_conv1d_causal_cuda(x.p + r, s.w, s.b, s.rec_a, out.p + r, n, s.n, s.kernel, st);
        });
        break;
      }
      case DOp::SelectiveScan: {  // Mamba-1's scan over the slot's s (device state)
        const DBuf &x = in(s, 0), &dt = in(s, 1), &bm = in(s, 2), &cm = in(s, 3);
        const int D = s.n, N = s.d_state;
        if (x.width != D || dt.width != D || bm.width != N || cm.width != N || x.rows != T || dt.rows != T ||
            bm.rows != T || cm.rows != T)
          throw std::runtime_error("IR selective_scan: x/dt must be [T, channels] and B/C [T, d_state]");
        DBuf& out = dout(s.out[0], T, D, {&x, &dt, &bm, &cm});
        rec_rows(s, [&](int t0, int n) {
          const size_t rd = static_cast<size_t>(t0) * D, rn = static_cast<size_t>(t0) * N;
          ir_selective_scan_cuda(x.p + rd, dt.p + rd, bm.p + rn, cm.p + rn, s.w, s.ssm_d, s.rec_a, out.p + rd, n, D, N,
                                 s.flag, st);
        });
        break;
      }
      case DOp::SsmScan: {  // over the slot's s (device state)
        const DBuf& u = in(s, 0);
        if (u.width != s.n || u.rows != T) throw std::runtime_error("IR ssm_scan: input width differs from n");
        DBuf& out = dout(s.out[0], T, s.n, {&u});
        rec_rows(s, [&](int t0, int n) {
          const size_t r = static_cast<size_t>(t0) * s.n;
          ir_ssm_scan_cuda(u.p + r, s.w, s.b, s.ssm_c, s.ssm_d, s.rec_a, out.p + r, n, s.n, st);
        });
        break;
      }
    }
    commit();
  }
}

void Interpreter::run_device(int T, int start_pos) {
  for (auto& b : dbufs_) b.rows = b.items = 0;  // allocations persist, definitions do not
  pending_.clear();
  run_nodes_device(plan_embed_, -1, T, start_pos);
  for (int l = 0; l < n_layers_; ++l) run_nodes_device(plan_block_[l], l, T, start_pos);
}

// Captures the T = 1 step (embed, blocks, head) on stream_ as a graph. The
// kernels are those of the per-launch step, with the token and position read
// from dstep_ (dpos_), so a replay computes exactly what the per-launch step
// would. Nothing runs while capturing; the caller replays the graph. False
// (and no graph) when the capture fails, e.g. a buffer that must still grow
// at this shape: the call then runs per-launch, which grows it, and the next
// T = 1 call captures again. Four failures in a row turn the graph off.
bool Interpreter::capture_decode(int start_pos) {
  if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
  graph_exec_ = nullptr;
  if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    cudaGetLastError();
    graph_off_ = true;
    Logger::warning("[IR] CUDA graph: stream capture unavailable; decode runs per-launch.");
    return false;
  }
  capturing_ = true;
  dpos_ = dstep_ + 1;
  std::string err;
  try {
    run_device(1, start_pos);
    run_nodes_device(plan_head_, -1, 1, start_pos);
    if (dbufs_[logits_slot_].rows != 1 || dbufs_[logits_slot_].width > vocab_)
      err = "the head produced no [1, vocab] 'logits'";
  } catch (const std::exception& e) {
    err = e.what();
  }
  capturing_ = false;
  dpos_ = nullptr;
  pending_.clear();
  cudaGraph_t graph = nullptr;
  cudaError_t e = cudaStreamEndCapture(stream_, &graph);
  if (err.empty() && e == cudaSuccess) e = cudaGraphInstantiate(&graph_exec_, graph, 0);
  if (graph) cudaGraphDestroy(graph);
  if (!err.empty() || e != cudaSuccess) {
    cudaGetLastError();
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
    if (++graph_failures_ >= 4) graph_off_ = true;
    Logger::info("[IR] CUDA graph: capture failed (" + (err.empty() ? std::string(cudaGetErrorString(e)) : err) + ")" +
                 (graph_off_ ? "; decode runs per-launch from now on." : "; this step runs per-launch."));
    return false;
  }
  graph_failures_ = 0;
  graph_epoch_ = alloc_epoch_;
  graph_logits_ = dbufs_[logits_slot_].p;
  graph_vocab_ = dbufs_[logits_slot_].width;
  ++graph_captures_;
  Logger::info("[IR] CUDA graph: captured the T = 1 step (" + std::to_string(n_layers_) + " blocks, capture " +
               std::to_string(graph_captures_) + "); decode replays it with the position in device memory");
  return true;
}

bool Interpreter::decode_graph(int token, int start_pos) {
  static const bool enabled = [] {
    const char* v = std::getenv("EMBER_IR_GRAPH");
    return !(v && std::string(v) == "0");
  }();
  // The range checks a replay cannot make (the per-launch step throws them).
  if (!enabled || graph_off_ || start_pos < 0 || start_pos + 1 > max_len_ || start_pos + 1 > pos_rows_ ||
      token < 0 || token >= embed_rows_)
    return false;
  hstep_[0] = token;
  hstep_[1] = start_pos;
  gpuErrchk(cudaMemcpyAsync(dstep_, hstep_, 2 * sizeof(int), cudaMemcpyHostToDevice, stream_));
  if (!graph_exec_ || graph_epoch_ != alloc_epoch_) {
    htokens_.assign(1, token);
    if (!capture_decode(start_pos)) return false;
  }
  gpuErrchk(cudaGraphLaunch(graph_exec_, stream_));
  gpuErrchk(cudaMemcpyAsync(hlogits_, graph_logits_, static_cast<size_t>(graph_vocab_) * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_));
  return true;
}

std::vector<float> Interpreter::forward_device(const std::vector<int>& tokens, int start_pos, bool all_rows) {
  const int T = static_cast<int>(tokens.size());
  if (rec_floats_ > 0) rec_begin(start_pos, T, all_rows);
  if (T == 1 && decode_graph(tokens[0], start_pos)) {
    if (rec_floats_ > 0) rec_end(start_pos, T);
    gpuErrchk(cudaStreamSynchronize(stream_));
    return std::vector<float>(hlogits_, hlogits_ + graph_vocab_);
  }
  htokens_ = tokens;
  if (dtokens_cap_ < T) {
    if (dtokens_) gpuErrchk(cudaFree(dtokens_));
    gpuErrchk(cudaMalloc(&dtokens_, static_cast<size_t>(T) * sizeof(int)));
    dtokens_cap_ = T;
  }
  gpuErrchk(cudaMemcpyAsync(dtokens_, tokens.data(), static_cast<size_t>(T) * sizeof(int), cudaMemcpyHostToDevice,
                            stream_));
  run_device(T, start_pos);
  if (rec_floats_ > 0) rec_end(start_pos, T);
  int rows = T;
  if (!all_rows && T > 1) {  // head ops are row-wise: run them on the last row only
    auto last_row = [T](DBuf& b) {
      if (b.rows == T && b.alloc) {
        b.p = b.alloc + static_cast<size_t>(T - 1) * b.width;
        b.rows = 1;
      }
    };
    for (size_t sl = 0; sl < dbufs_.size(); ++sl) {
      DBuf& b = dbufs_[sl];
      if (b.is_list && b.rows == T) {  // a list's tensors
        for (int k = 0; k < b.items; ++k) last_row(ditems_[sl][k]);
        b.rows = 1;
      } else {
        last_row(b);
      }
    }
    rows = 1;
  }
  run_nodes_device(plan_head_, -1, rows, start_pos);
  const DBuf& logits = dbufs_[logits_slot_];
  if (logits.rows == 0) throw std::runtime_error("IR head produced no 'logits'");
  std::vector<float> out(static_cast<size_t>(logits.rows) * logits.width);
  gpuErrchk(cudaMemcpyAsync(out.data(), logits.p, out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
  gpuErrchk(cudaStreamSynchronize(stream_));
  return out;
}
#endif  // HAS_CUDA

}  // namespace ember_ir
