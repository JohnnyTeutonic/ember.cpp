#include "graph_ir.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "cpu_attention.h"
#include "gguf_structs.h"
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
                                       "attention", "add", "mul", "silu", "gelu", "moe_ffn"};
const std::set<std::string> kRowWiseOps = {"rmsnorm", "layernorm", "linear", "add", "mul", "silu", "gelu"};

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

}  // namespace

bool has_windowed_attention(const std::string& ir_json) {
  const json j = json::parse(ir_json);
  if (!j.contains("block")) return false;
  const json& blk = j.at("block");
  for (const auto& s : blk.value("state", json::array()))
    if (s.value("window", 0) != 0) return true;
  for (const auto& n : blk.value("nodes", json::array()))
    if (n.value("op", "") == "attention" && n.value("window", 0) != 0) return true;
  return false;
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
  return {};
}

Interpreter::Interpreter(const std::string& ir_json, const GGUFData& gguf, bool use_gpu) {
  const json j = json::parse(ir_json);
  const int version = j.at("ir_version").get<int>();
  if (version != 0) throw std::runtime_error("IR version " + std::to_string(version) + " is not supported (ember reads 0)");
  family_hint_ = j.value("family_hint", "");
  embed_ = parse_nodes(j.at("embed"), "embed");
  block_ = parse_nodes(j.at("block"), "block");
  head_ = parse_nodes(j.at("head"), "head");
  for (const auto& n : head_)
    if (!kRowWiseOps.count(n.op)) throw std::runtime_error("IR head op '" + n.op + "' is not row-wise");
  n_layers_ = j.at("block").at("repeat").get<int>();
  vocab_ = j.at("model").at("vocab").get<int>();
  max_len_ = j.at("model").value("ctx", 0);

  for (const auto& s : j.at("block").value("state", json::array())) {
    const std::string kind = s.at("kind").get<std::string>();
    if (kind != "kv") throw std::runtime_error("IR state kind '" + kind + "' has no CPU implementation in ember");
    kv_heads_ = s.at("kv_heads").get<int>();
    head_dim_ = s.at("head_dim").get<int>();
    max_len_ = s.at("max_len").get<int>();
    // Sliding window + sinks (paperkiln swa_attention). The cache keeps every
    // position up to max_len, which holds the first `sinks` and the last
    // `window` rows the rule needs.
    window_ = s.value("window", 0);
    sinks_ = s.value("sinks", 0);
  }
  for (const auto& n : block_) {
    if (n.op != "attention") continue;
    if (!n.attrs.value("causal", true))
      throw std::runtime_error("IR attention with causal=false is not implemented in ember");
    int w = 0, s = 0;
    attention_window(n, w, s);
    if (w < 0 || s < 0) throw std::runtime_error("IR attention window/sinks must be non-negative");
    // A windowed kv state declares what its cache holds; an attention node
    // reading it with another window or sink count is an exporter error.
    if (window_ > 0 && (w != window_ || s != sinks_))
      throw std::runtime_error("IR attention window/sinks (" + std::to_string(w) + ", " + std::to_string(s) +
                               ") differ from its kv state's (" + std::to_string(window_) + ", " +
                               std::to_string(sinks_) + ")");
    if (w > 0)
      Logger::info("[IR] sliding-window attention: window " + std::to_string(w) + ", sinks " + std::to_string(s));
  }
  for (const auto* sub : {&embed_, &block_, &head_})
    for (const auto& n : *sub)
      if (n.op == "gelu" && n.attrs.value("approx", "none") != "tanh" && n.attrs.value("approx", "none") != "none")
        throw std::runtime_error("IR gelu approx '" + n.attrs.value("approx", "") + "' is not implemented in ember");
  for (const auto& n : block_) {
    if (n.op == "rope") {
      if (n.attrs.value("pairing", "interleaved") != "interleaved")
        throw std::runtime_error("IR rope pairing '" + n.attrs.value("pairing", "") + "' is not implemented in ember");
      if (n.attrs.value("rope_heads", "all") != "all")
        throw std::runtime_error("IR rope_heads '" + n.attrs.value("rope_heads", "") + "' is not implemented in ember");
      rope_theta_ = n.attrs.at("theta").get<float>();
    }
  }
  if (kv_heads_ <= 0 || head_dim_ <= 0 || max_len_ <= 0)
    throw std::runtime_error("IR block declares no usable kv state (kv_heads, head_dim, max_len)");

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
  size_t n_quant = 0, bytes_quant = 0, bytes_f32 = 0;
  for (const auto& kv : weights_) {
    n_quant += kv.second.quantised();
    bytes_quant += kv.second.q.size();
    bytes_f32 += kv.second.f32.size() * sizeof(float);
  }

  // RoPE table, computed exactly as TinyLlamaModel::initialize_rope_freqs.
  if (rope_theta_ > 0.0f) rope_freqs_.resize(static_cast<size_t>(max_len_) * head_dim_ / 2);
  for (int pos = 0; rope_theta_ > 0.0f && pos < max_len_; ++pos) {
    for (int i = 0; i < head_dim_; i += 2) {
      float freq = 1.0f / std::pow(rope_theta_, float(i) / head_dim_);
      float val = static_cast<float>(pos) * freq;
      rope_freqs_[(static_cast<size_t>(pos) * head_dim_ / 2) + (i / 2)] = {std::cos(val), std::sin(val)};
    }
  }
#ifdef HAS_CUDA
  if (use_gpu) {
    init_device();
    gpu_ = true;
  }
#else
  (void)use_gpu;
#endif
  if (!gpu_) {
    kv_.resize(n_layers_);
    for (auto& layer : kv_) {
      layer.k.assign(static_cast<size_t>(max_len_) * kv_heads_ * head_dim_, 0.0f);
      layer.v.assign(layer.k.size(), 0.0f);
    }
  }
  Logger::info("[IR] interpreter ready: family_hint=" + family_hint_ + ", " + std::to_string(n_layers_) + " blocks of " +
               std::to_string(block_.size()) + " ops, " + std::to_string(weights_.size()) + " weights bound (" +
               std::to_string(n_quant) + " kept quantised, " + std::to_string(bytes_quant >> 20) + " MiB; f32 " +
               std::to_string(bytes_f32 >> 20) + " MiB" + (dequant_all_ ? "; dequantised at load" : "") + "), device " +
               (gpu_ ? "cuda" : "cpu"));
}

Interpreter::~Interpreter() {
#ifdef HAS_CUDA
  free_device();
#endif
}

void Interpreter::attention_window(const Node& n, int& window, int& sinks) const {
  window = has(n.attrs, "window") ? n.attrs.at("window").get<int>() : window_;
  sinks = has(n.attrs, "sinks") ? n.attrs.at("sinks").get<int>() : sinks_;
}

std::string Interpreter::layer_name(const std::string& pattern, int layer) const {
  std::string s = pattern;
  const auto p = s.find("{i}");
  if (p != std::string::npos) {
    if (layer < 0) throw std::runtime_error("IR weight '" + pattern + "' uses {i} outside the block");
    s.replace(p, 3, std::to_string(layer));
  }
  return s;
}

void Interpreter::bind_tensor(const std::string& name, const GGUFData& gguf, bool keep_quantised) {
  if (auto bw = weights_.find(name); bw != weights_.end()) {
    Weight& w = bw->second;
    if (keep_quantised || !w.quantised()) return;
    // Bound quantised for a row-wise op, now needed whole (a norm, a bias): f32.
    w.f32.resize(w.numel);
    dequant_span_to_f32(w.type, w.q.data(), 0, w.numel, w.f32.data());
    std::vector<uint8_t>().swap(w.q);
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
  if (keep_quantised && !dequant_all_ && !shape.empty() && (t == GGMLType::GGML_TYPE_F16 || (block_type && shape[0] % 32 == 0))) {
    // Rows (ne0 elements) start on block boundaries, so a row is a whole
    // number of blocks and dequantises on its own.
    const size_t bytes = t == GGMLType::GGML_TYPE_F16 ? info.num_elements * 2 : info.num_elements / 32 * ggml_type_size(t);
    if (info.size_in_bytes < bytes)
      throw std::runtime_error("IR weight '" + name + "' is shorter than its element count implies");
    Weight w;
    w.q.assign(src, src + bytes);
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
    if (has(n.attrs, "weight")) {
      std::string w = layer_name(n.attrs.at("weight").get<std::string>(), layer);
      if (!gguf.tensor_infos_map.count(w) && has(n.attrs, "tied_to"))
        w = layer_name(n.attrs.at("tied_to").get<std::string>(), layer);
      bind_tensor(w, gguf, n.op == "linear" || n.op == "embedding");
    }
    if (has(n.attrs, "bias")) bind_tensor(layer_name(n.attrs.at("bias").get<std::string>(), layer), gguf);
    if (n.op == "moe_ffn") {
      const std::string norm = n.attrs.value("gate_norm", "renorm"), act = n.attrs.value("activation", "silu");
      if (norm != "renorm" && norm != "raw") throw std::runtime_error("IR moe_ffn gate_norm '" + norm + "' is not implemented");
      if (act != "silu") throw std::runtime_error("IR moe_ffn activation '" + act + "' is not implemented");
      bind_tensor(layer_name(n.attrs.at("router").get<std::string>(), layer), gguf);
      for (const char* part : {"gate", "up", "down"})
        bind_tensor(layer_name(n.attrs.at("experts").at(part).get<std::string>(), layer), gguf);
    }
  }
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
  auto in = [&](const Node& n, size_t k) -> Buf& {
    auto it = env.find(n.in.at(k));
    if (it == env.end()) throw std::runtime_error("IR op '" + n.op + "': input '" + n.in.at(k) + "' is not defined");
    return it->second;
  };
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
      for (int t = 0; t < T; ++t) {
        const int id = static_cast<int>(tokens.v[t]);
        if (id < 0 || static_cast<size_t>(id + 1) * d > W.numel)
          throw std::runtime_error("IR embedding: token id " + std::to_string(id) + " out of range");
        float* dst = out.v.data() + static_cast<size_t>(t) * d;
        if (W.quantised())
          dequant_span_to_f32(W.type, W.q.data(), static_cast<size_t>(id) * d, d, dst);
        else
          std::memcpy(dst, W.f32.data() + static_cast<size_t>(id) * d, d * sizeof(float));
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
      rmsnorm_batch_cpu(x.v, weight(wname(n)), out.v, T, x.width, n.attrs.value("eps", 1e-5f));
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
              dequant_span_to_f32(W.type, W.q.data(), static_cast<size_t>(o) * in_dim, in_dim, row);
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
      apply_rope_batch_cpu(q.v, k.v, T, q.width / hd, k.width / hd, hd, start_pos, rope_freqs_, max_len_,
                           /*use_adjacent_pairing=*/true);
      env[n.out.at(0)] = std::move(q);
      env[n.out.at(1)] = std::move(k);
    } else if (n.op == "attention") {
      const Buf &q = in(n, 0), &k = in(n, 1), &v = in(n, 2);
      const int hd = n.attrs.at("head_dim").get<int>(), hq = n.attrs.at("heads").get<int>();
      const int hkv = n.attrs.at("kv_heads").get<int>();
      if (hd != head_dim_ || hkv != kv_heads_) throw std::runtime_error("IR attention shape differs from its kv state");
      if (start_pos + T > max_len_)
        throw std::runtime_error("IR attention: position " + std::to_string(start_pos + T) + " exceeds the kv state (" +
                                 std::to_string(max_len_) + ")");
      KVCacheLayer& cache = kv_.at(layer);
      const size_t row = static_cast<size_t>(hkv) * hd;
      std::memcpy(cache.k.data() + start_pos * row, k.v.data(), static_cast<size_t>(T) * row * sizeof(float));
      std::memcpy(cache.v.data() + start_pos * row, v.v.data(), static_cast<size_t>(T) * row * sizeof(float));
      const float scale = n.attrs.contains("scale") ? n.attrs.at("scale").get<float>() : 1.0f / std::sqrt(float(hd));
      Buf out{{}, hq * hd};
      int window = 0, sinks = 0;
      attention_window(n, window, sinks);
      if (window > 0)
        attention_window_batch_cpu(q.v, cache, out.v, T, start_pos, hq, hkv, hd, scale, window, sinks);
      else
        attention_batch_cpu(q.v, cache, out.v, T, start_pos, hq, hkv, hd, scale);
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
    }
  }
}

std::vector<float> Interpreter::forward(const std::vector<int>& tokens, int start_pos, bool all_rows) {
  const int T = static_cast<int>(tokens.size());
  if (T == 0) return {};
#ifdef HAS_CUDA
  if (gpu_) return forward_device(tokens, start_pos, all_rows);
#endif
  Env env;
  env["tokens"] = Buf{std::vector<float>(tokens.begin(), tokens.end()), 1};
  run_nodes(embed_, -1, env, T, start_pos, false);
  for (int l = 0; l < n_layers_; ++l) run_nodes(block_, l, env, T, start_pos, false);
  int rows = T;
  if (!all_rows && T > 1) {  // head ops are row-wise: run them on the last row only
    for (auto& kv : env) {
      Buf& b = kv.second;
      if (b.width > 0 && b.v.size() == static_cast<size_t>(T) * b.width)
        b.v.erase(b.v.begin(), b.v.end() - b.width);
    }
    rows = 1;
  }
  run_nodes(head_, -1, env, rows, start_pos, !all_rows);
  auto it = env.find("logits");
  if (it == env.end()) throw std::runtime_error("IR head produced no 'logits'");
  return std::move(it->second.v);
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
  for (int l = 0; l < n_layers_; ++l) {
    gpuErrchk(cudaMalloc(&dk_[l], kv_floats * sizeof(float)));
    gpuErrchk(cudaMalloc(&dv_[l], kv_floats * sizeof(float)));
    gpuErrchk(cudaMemset(dk_[l], 0, kv_floats * sizeof(float)));
    gpuErrchk(cudaMemset(dv_[l], 0, kv_floats * sizeof(float)));
  }
  if (!rope_freqs_.empty()) {
    std::vector<float> flat(rope_freqs_.size() * 2);
    for (size_t i = 0; i < rope_freqs_.size(); ++i) {
      flat[2 * i] = rope_freqs_[i].first;
      flat[2 * i + 1] = rope_freqs_[i].second;
    }
    gpuErrchk(cudaMalloc(&drope_, flat.size() * sizeof(float)));
    gpuErrchk(cudaMemcpy(drope_, flat.data(), flat.size() * sizeof(float), cudaMemcpyHostToDevice));
  }
  Logger::info("[IR] CUDA device path: " + std::to_string(dweights_.size()) + " weights (" +
               std::to_string(floats * sizeof(float) / (1024 * 1024)) + " MiB f32) and " + std::to_string(n_layers_) +
               " kv slots of " + std::to_string(max_len_) + " positions on the device");
}

void Interpreter::free_device() {
  auto release = [](DEnv& env) {
    for (auto& b : env)
      if (b.second.alloc) cudaFree(b.second.alloc);
    env.clear();
  };
  release(denv_);
  release(dspare_);
  for (auto& w : dweights_) cudaFree(w.second);
  dweights_.clear();
  for (float* p : dk_) if (p) cudaFree(p);
  for (float* p : dv_) if (p) cudaFree(p);
  dk_.clear();
  dv_.clear();
  if (drope_) cudaFree(drope_);
  if (dtokens_) cudaFree(dtokens_);
  if (dmoe_) cudaFree(dmoe_);
  drope_ = dmoe_ = nullptr;
  dtokens_ = nullptr;
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

Interpreter::DBuf& Interpreter::dout(const std::string& name, int rows, int width,
                                     std::initializer_list<const DBuf*> inputs) {
  DBuf* b = &denv_[name];
  for (const DBuf* in : inputs)
    if (in && b->alloc && in->alloc == b->alloc) {
      b = &dspare_[name];
      pending_.push_back(name);
      break;
    }
  const size_t need = static_cast<size_t>(rows) * width;
  if (b->cap < need) {  // grows only on the first calls of a shape; cudaFree waits for the device
    if (b->alloc) gpuErrchk(cudaFree(b->alloc));
    gpuErrchk(cudaMalloc(&b->alloc, need * sizeof(float)));
    b->cap = need;
  }
  b->p = b->alloc;
  b->rows = rows;
  b->width = width;
  return *b;
}

void Interpreter::commit() {
  for (const auto& name : pending_) std::swap(denv_[name], dspare_[name]);
  pending_.clear();
}

void Interpreter::run_nodes_device(const std::vector<Node>& nodes, int layer, int T, int start_pos) {
  cudaStream_t st = stream_;
  auto in = [&](const Node& n, size_t k) -> DBuf& {
    auto it = denv_.find(n.in.at(k));
    if (it == denv_.end() || it->second.rows == 0)
      throw std::runtime_error("IR op '" + n.op + "': input '" + n.in.at(k) + "' is not defined");
    return it->second;
  };
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
  for (const auto& n : nodes) {
    if (n.op == "embedding") {
      const std::string wn = wname(n);
      const int d = static_cast<int>(shapes_.at(wn).at(0));
      const size_t rows = elems(wn) / d;
      for (int t = 0; t < T; ++t)
        if (htokens_[t] < 0 || static_cast<size_t>(htokens_[t]) >= rows)
          throw std::runtime_error("IR embedding: token id " + std::to_string(htokens_[t]) + " out of range");
      DBuf& out = dout(n.out.at(0), T, d, {});
      ir_embedding_cuda(dweight(wn), dtokens_, out.p, T, d, st);
    } else if (n.op == "pos_embedding") {
      const DBuf& x = in(n, 0);
      const std::string wn = wname(n);
      const int d = x.width;
      if (shapes_.at(wn).at(0) != d) throw std::runtime_error("IR pos_embedding '" + wn + "': width differs from x");
      if (static_cast<size_t>(start_pos + T) * d > elems(wn))
        throw std::runtime_error("IR pos_embedding: position " + std::to_string(start_pos + T - 1) + " exceeds '" + wn + "'");
      DBuf& out = dout(n.out.at(0), T, d, {&x});
      ir_pos_embedding_cuda(x.p, dweight(wn), out.p, T, d, start_pos, st);
    } else if (n.op == "layernorm") {
      const DBuf& x = in(n, 0);
      DBuf& out = dout(n.out.at(0), T, x.width, {&x});
      ir_layernorm_cuda(x.p, dweight(wname(n)), dweight(layer_name(n.attrs.at("bias").get<std::string>(), layer)), out.p,
                        T, x.width, n.attrs.value("eps", 1e-5f), st);
    } else if (n.op == "split") {
      const DBuf& x = in(n, 0);
      const auto sizes = n.attrs.at("sizes").get<std::vector<int>>();
      int total = 0;
      for (int s : sizes) total += s;
      if (total != x.width || sizes.size() != n.out.size())
        throw std::runtime_error("IR split: sizes do not match the input width or the outputs");
      int col = 0;
      for (size_t o = 0; o < sizes.size(); ++o) {
        DBuf& part = dout(n.out[o], T, sizes[o], {&x});
        gpuErrchk(cudaMemcpy2DAsync(part.p, sizes[o] * sizeof(float), x.p + col, x.width * sizeof(float),
                                    sizes[o] * sizeof(float), T, cudaMemcpyDeviceToDevice, st));
        col += sizes[o];
      }
    } else if (n.op == "gelu") {
      const DBuf& x = in(n, 0);
      DBuf& out = dout(n.out.at(0), T, x.width, {&x});
      ir_gelu_cuda(x.p, out.p, static_cast<size_t>(T) * x.width, n.attrs.value("approx", "none") == "tanh", st);
    } else if (n.op == "rmsnorm") {
      const DBuf& x = in(n, 0);
      DBuf& out = dout(n.out.at(0), T, x.width, {&x});
      rmsnorm_batch_cuda(out.p, x.p, dweight(wname(n)), T, x.width, n.attrs.value("eps", 1e-5f), st);
    } else if (n.op == "linear") {
      const DBuf& x = in(n, 0);
      const std::string wn = wname(n);
      if (shapes_.at(wn).at(0) != x.width)
        throw std::runtime_error("IR linear '" + wn + "': input width " + std::to_string(x.width) +
                                 " does not match the weight's ne0 " + std::to_string(shapes_.at(wn).at(0)));
      const int out_dim = static_cast<int>(elems(wn) / x.width);
      DBuf& out = dout(n.out.at(0), T, out_dim, {&x});
      gemm_rm_f32_cuda(cublas_, x.p, dweight(wn), out.p, T, out_dim, x.width, st);
      if (has(n.attrs, "bias"))
        ir_bias_add_cuda(out.p, dweight(layer_name(n.attrs.at("bias").get<std::string>(), layer)), T, out_dim, st);
    } else if (n.op == "rope") {
      const int hd = n.attrs.at("head_dim").get<int>();
      if (start_pos + T > max_len_ || !drope_)
        throw std::runtime_error("IR rope: position " + std::to_string(start_pos + T - 1) + " exceeds the rope table");
      DBuf* q = &in(n, 0);
      DBuf* k = &in(n, 1);
      // Rotated in place when the outputs reuse the input names (the usual
      // form); otherwise copied to the outputs first.
      for (size_t o = 0; o < 2; ++o) {
        DBuf*& src = o == 0 ? q : k;
        if (n.out.at(o) == n.in.at(o)) continue;
        DBuf& dst = dout(n.out.at(o), T, src->width, {src});
        gpuErrchk(cudaMemcpyAsync(dst.p, src->p, static_cast<size_t>(T) * src->width * sizeof(float),
                                  cudaMemcpyDeviceToDevice, st));
        src = &dst;
      }
      ir_rope_cuda(q->p, k->p, T, q->width / hd, k->width / hd, hd, start_pos, drope_, st);
    } else if (n.op == "attention") {
      const DBuf &q = in(n, 0), &k = in(n, 1), &v = in(n, 2);
      const int hd = n.attrs.at("head_dim").get<int>(), hq = n.attrs.at("heads").get<int>();
      const int hkv = n.attrs.at("kv_heads").get<int>();
      if (hd != head_dim_ || hkv != kv_heads_) throw std::runtime_error("IR attention shape differs from its kv state");
      {
        int window = 0, sinks = 0;
        attention_window(n, window, sinks);
        if (window > 0)
          throw std::runtime_error("IR sliding-window attention has no CUDA kernel yet (EMBER_IR_DEVICE=cpu runs it)");
      }
      if (k.width != hkv * hd || v.width != hkv * hd || q.width != hq * hd)
        throw std::runtime_error("IR attention: q/k/v widths do not match heads * head_dim");
      if (start_pos + T > max_len_)
        throw std::runtime_error("IR attention: position " + std::to_string(start_pos + T) + " exceeds the kv state (" +
                                 std::to_string(max_len_) + ")");
      const size_t row = static_cast<size_t>(hkv) * hd;
      float* K = dk_.at(layer);
      float* V = dv_.at(layer);
      gpuErrchk(cudaMemcpyAsync(K + start_pos * row, k.p, static_cast<size_t>(T) * row * sizeof(float),
                                cudaMemcpyDeviceToDevice, st));
      gpuErrchk(cudaMemcpyAsync(V + start_pos * row, v.p, static_cast<size_t>(T) * row * sizeof(float),
                                cudaMemcpyDeviceToDevice, st));
      const float scale = n.attrs.contains("scale") ? n.attrs.at("scale").get<float>() : 1.0f / std::sqrt(float(hd));
      DBuf& out = dout(n.out.at(0), T, hq * hd, {&q, &k, &v});
      if (T == 1)
        attention_decode_at_cuda(q.p, K, V, out.p, hq, start_pos + 1, hd, scale, max_len_, hkv, st);
      else
        attention_batch_prefill_cuda(q.p, nullptr, nullptr, K, V, out.p, T, start_pos, max_len_, hq, hkv, hd, scale, st,
                                     nullptr);
    } else if (n.op == "add" || n.op == "mul") {
      const DBuf &a = in(n, 0), &b = in(n, 1);
      if (static_cast<size_t>(a.rows) * a.width != static_cast<size_t>(b.rows) * b.width)
        throw std::runtime_error("IR " + n.op + ": operand sizes differ");
      DBuf& out = dout(n.out.at(0), a.rows, a.width, {&a, &b});
      ir_binary_cuda(a.p, b.p, out.p, static_cast<size_t>(a.rows) * a.width, n.op == "add" ? 0 : 1, st);
    } else if (n.op == "moe_ffn") {
      const DBuf& x = in(n, 0);
      const auto& ex = n.attrs.at("experts");
      const std::string gn = layer_name(ex.at("gate").get<std::string>(), layer);
      const int E = n.attrs.at("n_experts").get<int>(), k = n.attrs.at("top_k").get<int>();
      const auto& gs = shapes_.at(gn);  // ggml order {d, d_ff, E}
      if (gs.size() != 3 || gs[0] != x.width || gs[2] != E)
        throw std::runtime_error("IR moe_ffn '" + gn + "': expert stack shape does not match d and n_experts");
      const int d = x.width, d_ff = static_cast<int>(gs[1]);
      const size_t need = moe_ffn_scratch_floats(T, d, d_ff, E, k);
      if (need > dmoe_cap_) {
        if (dmoe_) gpuErrchk(cudaFree(dmoe_));
        gpuErrchk(cudaMalloc(&dmoe_, need * sizeof(float)));
        dmoe_cap_ = need;
      }
      DBuf& out = dout(n.out.at(0), T, d, {&x});
      moe_ffn_cuda(cublas_, x.p, T, d, d_ff, E, k, n.attrs.value("gate_norm", "renorm") == "renorm",
                   dweight(layer_name(n.attrs.at("router").get<std::string>(), layer)), dweight(gn),
                   dweight(layer_name(ex.at("up").get<std::string>(), layer)),
                   dweight(layer_name(ex.at("down").get<std::string>(), layer)), dmoe_, out.p, st);
    } else if (n.op == "silu") {
      const DBuf& x = in(n, 0);
      DBuf& out = dout(n.out.at(0), T, x.width, {&x});
      ir_silu_cuda(x.p, out.p, static_cast<size_t>(T) * x.width, st);
    }
    commit();
  }
}

std::vector<float> Interpreter::forward_device(const std::vector<int>& tokens, int start_pos, bool all_rows) {
  const int T = static_cast<int>(tokens.size());
  for (auto& b : denv_) b.second.rows = 0;  // allocations persist, definitions do not
  htokens_ = tokens;
  if (dtokens_cap_ < T) {
    if (dtokens_) gpuErrchk(cudaFree(dtokens_));
    gpuErrchk(cudaMalloc(&dtokens_, static_cast<size_t>(T) * sizeof(int)));
    dtokens_cap_ = T;
  }
  gpuErrchk(cudaMemcpyAsync(dtokens_, tokens.data(), static_cast<size_t>(T) * sizeof(int), cudaMemcpyHostToDevice,
                            stream_));
  run_nodes_device(embed_, -1, T, start_pos);
  for (int l = 0; l < n_layers_; ++l) run_nodes_device(block_, l, T, start_pos);
  int rows = T;
  if (!all_rows && T > 1) {  // head ops are row-wise: run them on the last row only
    for (auto& kv : denv_) {
      DBuf& b = kv.second;
      if (b.rows == T) {
        b.p = b.alloc + static_cast<size_t>(T - 1) * b.width;
        b.rows = 1;
      }
    }
    rows = 1;
  }
  run_nodes_device(head_, -1, rows, start_pos);
  auto it = denv_.find("logits");
  if (it == denv_.end() || it->second.rows == 0) throw std::runtime_error("IR head produced no 'logits'");
  std::vector<float> out(static_cast<size_t>(it->second.rows) * it->second.width);
  gpuErrchk(cudaMemcpyAsync(out.data(), it->second.p, out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
  gpuErrchk(cudaStreamSynchronize(stream_));
  return out;
}
#endif  // HAS_CUDA

}  // namespace ember_ir
