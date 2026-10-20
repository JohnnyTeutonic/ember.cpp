#pragma once
// Interpreter for paperkiln's block-structured IR (LOCAL_LLM_STACK.md section 4).
//
// The IR (GGUF key `paperkiln.ir`, or a side file named by EMBER_IR) is an
// `embed` subgraph, a `block` subgraph repeated L times (weights `blk.{i}.*`)
// and a `head` subgraph. Tensors flow as [T, width] row matrices, T being the
// tokens in this call; prefill and decode run the same graph (decode is
// T = 1). Weights are GGUF tensors by name, [out, in] row-major. The weights
// of `linear` and `embedding` stay in their GGUF blocks when quantised (Q8_0,
// Q4_0, Q4_1, Q5_0, Q5_1, F16): each call dequantises one row at a time into
// f32 with the load-time routines and dots it in the f32 kernel's order, so
// results equal dequantising at load bit for bit. Every other weight (norms,
// biases, MoE stacks) is f32 at load. Kept quantised only for models over
// 64M parameters on the CPU; small models and the CUDA path dequantise every
// weight at load (EMBER_IR_QUANT=1 / EMBER_IR_DEQUANT=1 force either way). State is owned here: per block, at most
// one kv slot and any number of recurrent slots.
//
// Recurrent state (block.state kind "recurrent": linear_attention's num and
// den, ssm_scan's s). The interpreter tracks how many tokens it has absorbed
// (one sequence). forward() at start_pos 0 starts a sequence from the init
// state; at the absorbed count it continues; anywhere else it restores a
// snapshot of the state taken after start_pos tokens, or throws when it has
// none (a recurrence cannot drop tokens). Snapshots are taken at the end of
// every call, after the second-to-last row of a multi-row call (the same
// prompt again resumes there), and after every row of an all_rows call
// (speculative verification resumes after the accepted rows). A later call
// from position p drops the snapshots past p. resume_point() tells the caller
// where a prefix can actually be resumed.
//
// CPU kernels, built from the same primitives as ember's llama CPU path so
// the arithmetic matches it: embedding, pos_embedding, rmsnorm, layernorm,
// linear (+ bias), split, rope (interleaved pairs), attention (causal, KV
// state, optional sliding window + sinks as paperkiln's swa_attention), add, mul, silu, gelu (tanh or erf), moe_ffn (llama.cpp's expert
// layout), linear_attention (elu+1, ops::kimi_attention) and ssm_scan
// (ops::ssm_scan) over recurrent state, CPU only; and for the gated and
// attention-residual families (flex highway / plain, attnres, srd's router)
// sub, relu, sigmoid, affine, mul_col, rms_row, list, append and depth_attend
// (graph_ir_ops.h), CPU only: a list is a value like a tensor (named by "out",
// read by "in"), an ordered set of [T, d] tensors. Any other op fails at load,
// naming the op and the backend.
//
// CUDA builds add a device path, chosen at construction (use_gpu): weights
// are uploaded once, the KV state and the named row buffers live on the
// device, and the op sequence runs on one stream with no host sync until the
// logits are copied back. The ops are compiled at load into a per-block plan
// (enum ops, buffer slots, device weight pointers, numeric attributes), so a
// call does no JSON or string work. GEMMs are FP32 cuBLAS (default math); the
// other kernels follow the CPU forms (graph_ir_cuda.h), sliding-window
// attention included, so the two paths agree to rounding.
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "ggml_types.h"
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
  // use_gpu selects the device path (CUDA builds; ignored otherwise).
  Interpreter(const std::string& ir_json, const GGUFData& gguf, bool use_gpu = false);
  ~Interpreter();
  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;

  // Runs tokens at absolute positions start_pos .. start_pos + T - 1 (the KV
  // state must already hold positions < start_pos). Returns logits for the
  // last token ([vocab]) or, with all_rows, for every token ([T][vocab]).
  std::vector<float> forward(const std::vector<int>& tokens, int start_pos, bool all_rows = false);

  // True when a block declares recurrent state (see the header comment).
  bool has_recurrent() const { return rec_floats_ > 0; }
  // The largest position <= want from which forward() can resume the current
  // sequence: want itself without recurrent state; else want when the state
  // or a snapshot is there, otherwise the nearest snapshot below it (0 at
  // worst: a fresh start).
  int resume_point(int want) const;

  int vocab() const { return vocab_; }
  int max_len() const { return max_len_; }
  int n_layers() const { return n_layers_; }
  const std::string& family_hint() const { return family_hint_; }
  bool on_gpu() const { return gpu_; }

 private:
  struct Buf {
    std::vector<float> v;
    int width = 0;
    bool is_list = false;    // a list value (op list / append): its tensors are `items`, v is empty
    std::vector<Buf> items;  // in order; each a tensor
  };
  using Env = std::unordered_map<std::string, Buf>;
  // A bound weight: f32 values, or (quantised) the tensor's raw GGUF bytes.
  struct Weight {
    std::vector<float> f32;
    std::vector<uint8_t> q;
    GGMLType type = GGMLType::GGML_TYPE_F32;
    size_t numel = 0;
    bool quantised() const { return !q.empty(); }
  };

  void run_nodes(const std::vector<Node>& nodes, int layer, Env& env, int T, int start_pos, bool head_last_only);
  // The attention node's own window/sinks when it names them, else its kv state's.
  void attention_window(const Node& n, int& window, int& sinks) const;
  const std::vector<float>& weight(const std::string& name) const;  // f32 weights only
  const Weight& bound(const std::string& name) const;
  std::string layer_name(const std::string& pattern, int layer) const;
  void bind(const std::vector<Node>& nodes, int layer, const GGUFData& gguf);
  // keep_quantised: the tensor is only read row-wise (linear, embedding).
  void bind_tensor(const std::string& name, const GGUFData& gguf, bool keep_quantised = false);

  std::vector<Node> embed_, block_, head_;
  int n_layers_ = 0, vocab_ = 0, max_len_ = 0;
  std::string family_hint_;
  bool dequant_all_ = false;  // every weight f32 at load (see the header comment)
  std::unordered_map<std::string, Weight> weights_;
  std::unordered_map<std::string, std::vector<int64_t>> shapes_;  // ggml order (ne0 first)
  std::vector<KVCacheLayer> kv_;  // per block: [max_len][kv_heads][head_dim]
  int kv_heads_ = 0, head_dim_ = 0;
  int window_ = 0, sinks_ = 0;  // kv state's sliding window (0 = unbounded) and sinks
  std::vector<std::pair<float, float>> rope_freqs_;
  float rope_theta_ = 0.0f;
  bool gpu_ = false;

  // Recurrent slots: each block's state is rec_floats_ floats, slot after
  // slot, each slot's tensors in declaration order.
  struct RecTensor {
    std::string name;
    std::vector<int64_t> shape;
    size_t offset = 0, floats = 0;  // within the block's state
    float init = 0.0f;
  };
  struct RecSlot {
    std::string name, op;
    std::vector<RecTensor> tensors;
  };
  const RecSlot& rec_slot(const Node& n, const char* op) const;
  float* rec_tensor(const RecSlot& slot, const char* tensor, int layer, size_t floats);
  void rec_begin(int start_pos, int T, bool all_rows);
  void rec_row_done(const RecSlot& slot, int layer, int t);  // the slot's state after row t (kept rows only)
  void rec_end(int start_pos, int T);
  std::vector<RecSlot> rec_slots_;
  size_t rec_floats_ = 0;
  std::vector<float> rec_init_;  // one block's initial state
  std::vector<float> rec_;  // [n_layers][rec_floats_]
  int rec_pos_ = 0;  // tokens absorbed; -1 after a failed call
  std::map<int, std::vector<float>> rec_snaps_;  // state after `key` tokens
  size_t rec_max_snaps_ = 0;
  std::vector<int> rec_snap_rows_;  // this call: rows whose state is kept (-1 = not kept), indices into rec_rows_
  std::vector<float> rec_rows_;  // this call: [kept row][n_layers][rec_floats_]


#ifdef HAS_CUDA
  // Device path. A DBuf is a [rows, width] buffer, one per tensor name of the
  // IR (its slot); `p` points into the owned allocation (the head runs on the
  // last row of a prefill in place). Buffers persist across calls so
  // steady-state decode allocates nothing; an op whose output name is also
  // one of its inputs writes the slot's spare and swaps.
  struct DBuf {
    float* alloc = nullptr;
    float* p = nullptr;
    size_t cap = 0;  // floats
    int width = 0, rows = 0;  // rows == 0: not defined in this call
  };
  // One op of the device plan, resolved once at load (compile_device): the op
  // as an enum, its tensors as slots, its weights as device pointers and its
  // attributes as numbers, so a call does no JSON, string or map work. Only
  // the checks that depend on the call (widths, positions) remain per call.
  enum class DOp { Embedding, PosEmbedding, LayerNorm, Split, Gelu, RmsNorm, Linear, Rope, Attention, Add, Mul, MoeFfn, Silu };
  struct DStep {
    DOp op = DOp::Add;
    const Node* node = nullptr;  // names for error messages
    std::vector<int> in, out;    // slots
    std::string wname;           // the resolved weight (error messages)
    const float* w = nullptr;    // weight: table, norm scale, linear [out, in], moe router
    const float* b = nullptr;    // bias (layernorm, linear)
    const float *gate = nullptr, *up = nullptr, *down = nullptr;  // moe_ffn expert stacks
    size_t w_elems = 0;
    int w_ne0 = 0;  // the weight's ne0 (rows of width ne0)
    int hd = 0, hq = 0, hkv = 0, window = 0, sinks = 0;  // rope, attention
    int E = 0, top_k = 0, d_ff = 0;                      // moe_ffn
    float eps = 1e-5f, scale = 0.0f;
    bool flag = false;  // gelu: tanh form; moe_ffn: renorm gates
    std::vector<int> sizes;  // split
  };
  void init_device();
  void compile_device();
  std::vector<DStep> compile_nodes(const std::vector<Node>& nodes, int layer);
  int slot(const std::string& name);
  void free_device();
  const float* dweight(const std::string& name) const;
  DBuf& dout(int slot, int rows, int width, std::initializer_list<const DBuf*> inputs);
  void commit();
  void run_nodes_device(const std::vector<DStep>& steps, int layer, int T, int start_pos);
  std::vector<float> forward_device(const std::vector<int>& tokens, int start_pos, bool all_rows);

  cublasHandle_t cublas_ = nullptr;
  cudaStream_t stream_ = nullptr;
  std::unordered_map<std::string, float*> dweights_;
  std::unordered_map<std::string, int> slots_;  // tensor name -> slot (load time only)
  std::vector<std::string> slot_names_;
  std::vector<DBuf> dbufs_, dspare_;            // by slot
  std::vector<int> pending_;  // slots written to their spare by dout, swapped by commit
  std::vector<DStep> plan_embed_, plan_head_;
  std::vector<std::vector<DStep>> plan_block_;  // per block
  int logits_slot_ = -1;
  std::vector<float*> dk_, dv_;  // per block: [max_len][kv_heads][head_dim]
  float* drope_ = nullptr;
  int* dtokens_ = nullptr;
  int dtokens_cap_ = 0;
  std::vector<int> htokens_;  // this call's tokens (range checks on the host)
  float* dmoe_ = nullptr;
  size_t dmoe_cap_ = 0;
#endif
};

// The IR for a GGUF: the EMBER_IR side file if set, else the `paperkiln.ir`
// key, else empty.
std::string find_ir(const GGUFData& gguf);

// True when the IR declares sliding-window attention (a kv state or an
// attention node with window != 0). Only the interpreter implements it, so
// such a GGUF never runs on the llama path, whatever its family_hint.
bool has_windowed_attention(const std::string& ir_json);

// True when the IR declares recurrent state (a block.state of kind
// "recurrent"). It has no CUDA kernels, so such an IR runs on the CPU.
bool has_recurrent_state(const std::string& ir_json);

// True when the CUDA path has a kernel for every op of the IR and the IR has
// no recurrent state; otherwise the interpreter must run on the CPU.
bool device_supported(const std::string& ir_json);

// The recurrent ops for one token (row), in paperkiln's order of operations
// (graph_ir.cpp); the state is updated in place. Exposed for the op tests.
//   linear_attention: q, k, v, num, den, out each `width` = heads * head_dim.
void linear_attention_row_cpu(const float* q, const float* k, const float* v, float* num, float* den, float* out,
                              int width, float den_floor);
//   ssm_scan: u, s, y [n]; A [n, n] row-major; B, C [n]; scratch [n].
void ssm_scan_row_cpu(const float* u, const float* A, const float* B, const float* C, float D, float* s,
                      float* scratch, float* y, int n);

}  // namespace ember_ir
