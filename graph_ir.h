#pragma once
// Interpreter for paperkiln's block-structured IR (LOCAL_LLM_STACK.md section 4).
//
// The IR (GGUF key `paperkiln.ir`, or a side file named by EMBER_IR) is an
// `embed` subgraph, a `block` subgraph repeated L times (weights `blk.{i}.*`)
// and a `head` subgraph. Tensors flow as [T, width] row matrices, T being the
// tokens in this call; prefill and decode run the same graph (decode is
// T = 1). Weights are GGUF tensors by name, [out, in] row-major, dequantised
// to f32 at load. State is owned here: one KV slot per block.
//
// CPU kernels, built from the same primitives as ember's llama CPU path so
// the arithmetic matches it: embedding, pos_embedding, rmsnorm, layernorm,
// linear (+ bias), split, rope (interleaved pairs), attention (causal, KV
// state), add, mul, silu, gelu (tanh or erf), moe_ffn (llama.cpp's expert
// layout). Any other op fails at load, naming the op and the backend.
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "model.h"

struct GGUFData;

namespace ember_ir {

struct Node {
  std::string op;
  std::vector<std::string> in, out;
  nlohmann::json attrs;  // the node's JSON object
};

class Interpreter {
 public:
  // Parses the IR JSON and binds every weight it names from the GGUF.
  // Throws std::runtime_error naming the op, tensor or field at fault.
  Interpreter(const std::string& ir_json, const GGUFData& gguf);

  // Runs tokens at absolute positions start_pos .. start_pos + T - 1 (the KV
  // state must already hold positions < start_pos). Returns logits for the
  // last token ([vocab]) or, with all_rows, for every token ([T][vocab]).
  std::vector<float> forward(const std::vector<int>& tokens, int start_pos, bool all_rows = false);

  int vocab() const { return vocab_; }
  int max_len() const { return max_len_; }
  int n_layers() const { return n_layers_; }
  const std::string& family_hint() const { return family_hint_; }

 private:
  struct Buf {
    std::vector<float> v;
    int width = 0;
  };
  using Env = std::unordered_map<std::string, Buf>;

  void run_nodes(const std::vector<Node>& nodes, int layer, Env& env, int T, int start_pos, bool head_last_only);
  const std::vector<float>& weight(const std::string& name) const;
  std::string layer_name(const std::string& pattern, int layer) const;
  void bind(const std::vector<Node>& nodes, int layer, const GGUFData& gguf);
  void bind_tensor(const std::string& name, const GGUFData& gguf);

  std::vector<Node> embed_, block_, head_;
  int n_layers_ = 0, vocab_ = 0, max_len_ = 0;
  std::string family_hint_;
  std::unordered_map<std::string, std::vector<float>> weights_;
  std::unordered_map<std::string, std::vector<int64_t>> shapes_;  // ggml order (ne0 first)
  std::vector<KVCacheLayer> kv_;  // per block: [max_len][kv_heads][head_dim]
  int kv_heads_ = 0, head_dim_ = 0;
  std::vector<std::pair<float, float>> rope_freqs_;
  float rope_theta_ = 0.0f;
};

// The IR for a GGUF: the EMBER_IR side file if set, else the `paperkiln.ir`
// key, else empty.
std::string find_ir(const GGUFData& gguf);

}  // namespace ember_ir
