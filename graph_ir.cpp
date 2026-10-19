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

Interpreter::Interpreter(const std::string& ir_json, const GGUFData& gguf) {
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
    if (s.value("window", 0) != 0 || s.value("sinks", 0) != 0)
      throw std::runtime_error("IR kv state with window/sinks is not implemented in ember yet");
    kv_heads_ = s.at("kv_heads").get<int>();
    head_dim_ = s.at("head_dim").get<int>();
    max_len_ = s.at("max_len").get<int>();
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

  bind(embed_, -1, gguf);
  for (int l = 0; l < n_layers_; ++l) bind(block_, l, gguf);
  bind(head_, -1, gguf);

  // RoPE table, computed exactly as TinyLlamaModel::initialize_rope_freqs.
  if (rope_theta_ > 0.0f) rope_freqs_.resize(static_cast<size_t>(max_len_) * head_dim_ / 2);
  for (int pos = 0; rope_theta_ > 0.0f && pos < max_len_; ++pos) {
    for (int i = 0; i < head_dim_; i += 2) {
      float freq = 1.0f / std::pow(rope_theta_, float(i) / head_dim_);
      float val = static_cast<float>(pos) * freq;
      rope_freqs_[(static_cast<size_t>(pos) * head_dim_ / 2) + (i / 2)] = {std::cos(val), std::sin(val)};
    }
  }
  kv_.resize(n_layers_);
  for (auto& layer : kv_) {
    layer.k.assign(static_cast<size_t>(max_len_) * kv_heads_ * head_dim_, 0.0f);
    layer.v.assign(layer.k.size(), 0.0f);
  }
  Logger::info("[IR] interpreter ready: family_hint=" + family_hint_ + ", " + std::to_string(n_layers_) + " blocks of " +
               std::to_string(block_.size()) + " ops, " + std::to_string(weights_.size()) + " weights bound");
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

void Interpreter::bind_tensor(const std::string& name, const GGUFData& gguf) {
  if (weights_.count(name)) return;
  auto it = gguf.tensor_infos_map.find(name);
  if (it == gguf.tensor_infos_map.end()) throw std::runtime_error("IR weight '" + name + "' is not in the GGUF");
  const GGUFTensorInfo& info = it->second;
  const uint8_t* base = gguf.mapped_tensor_data
                            ? static_cast<const uint8_t*>(gguf.mapped_tensor_data) + gguf.offset_diff_for_mmap
                            : gguf.tensor_data.data();
  const uint8_t* src = base + info.offset;
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
  std::vector<int64_t> shape(info.shape.begin(), info.shape.end());
  shapes_[name] = shape;
  weights_[name] = std::move(out);
}

void Interpreter::bind(const std::vector<Node>& nodes, int layer, const GGUFData& gguf) {
  for (const auto& n : nodes) {
    if (has(n.attrs, "weight")) {
      std::string w = layer_name(n.attrs.at("weight").get<std::string>(), layer);
      if (!gguf.tensor_infos_map.count(w) && has(n.attrs, "tied_to"))
        w = layer_name(n.attrs.at("tied_to").get<std::string>(), layer);
      bind_tensor(w, gguf);
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

const std::vector<float>& Interpreter::weight(const std::string& name) const {
  auto it = weights_.find(name);
  if (it == weights_.end()) throw std::runtime_error("IR weight '" + name + "' was not bound");
  return it->second;
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
      const auto& W = weight(wn);
      const int d = static_cast<int>(shapes_.at(wn).at(0));
      Buf out{std::vector<float>(static_cast<size_t>(T) * d), d};
      for (int t = 0; t < T; ++t) {
        const int id = static_cast<int>(tokens.v[t]);
        if (id < 0 || static_cast<size_t>(id + 1) * d > W.size())
          throw std::runtime_error("IR embedding: token id " + std::to_string(id) + " out of range");
        std::memcpy(out.v.data() + static_cast<size_t>(t) * d, W.data() + static_cast<size_t>(id) * d, d * sizeof(float));
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
      const auto& W = weight(wn);
      if (shapes_.at(wn).at(0) != x.width)
        throw std::runtime_error("IR linear '" + wn + "': input width " + std::to_string(x.width) +
                                 " does not match the weight's ne0 " + std::to_string(shapes_.at(wn).at(0)));
      const int out_dim = static_cast<int>(W.size() / x.width);
      Buf out{std::vector<float>(static_cast<size_t>(T) * out_dim), out_dim};
      matmul_f32_f32_batch_cpu(W, x.v, out.v, T, out_dim, x.width);
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

}  // namespace ember_ir
