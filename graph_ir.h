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
// the arithmetic matches it: embedding, pos_embedding, rmsnorm (optionally
// per group of `group` columns, one weight [group] shared: per-head q/k
// norm), layernorm, linear (+ bias; tied_to names the weight a missing one
// reads), split, rope (pairing "interleaved" (x[2j], x[2j+1]) or "neox"
// (x[j], x[j + n_rot/2]); n_rot <= head_dim rotates each head's first n_rot
// dims, angle pos * theta^(-2j/n_rot); k has kv_heads heads), attention
// (causal, grouped-query: query head h reads kv head h / (heads / kv_heads);
// KV state, optional sliding window + sinks as paperkiln's swa_attention), add, mul, silu, gelu (tanh or erf), moe_ffn (llama.cpp's expert
// layout), linear_attention (elu+1, ops::kimi_attention) and ssm_scan
// (ops::ssm_scan) over recurrent state, and recurrence (a recurrent op the
// document defines, IR_SPEC 6.10, graph_ir_recurrence.h; CPU only: a CUDA
// build serves such a document on the CPU); and for the gated and
// attention-residual families (flex highway / plain, attnres, srd's router)
// sub, relu, sigmoid, affine, mul_col, rms_row, list, append and depth_attend
// (graph_ir_ops.h): a list is a value like a tensor (named by "out", read by
// "in"), an ordered set of [T, d] tensors. Any other op fails at load,
// naming the op and the backend. Weights may be F32, F16, BF16, Q8_0 or the
// legacy 4/5-bit blocks.
//
// Third-party GGUFs (LOCAL_LLM_STACK.md section 7): a file with no
// `paperkiln.ir` whose general.architecture is qwen2 or qwen3 gets the IR
// that build_ir_from_gguf makes from its llama.cpp metadata, node for node
// what paperkiln's tools/gguf_ir.py builds. llama and mistral files keep the
// llama path (EMBER_IR_BUILD=1 builds their IR too); any other architecture
// fails at load, naming it. EMBER_IR_CTX=N caps the positions the KV state
// holds (a third-party IR declares the model's full trained context).
//
// CUDA builds add a device path, chosen at construction (use_gpu), for every
// op above (device_supported): weights are uploaded once, the KV state, the
// recurrent state with its snapshots and the named row buffers live on the
// device, and the op sequence runs on one stream with no host sync until the
// logits are copied back. The ops are compiled at load into a per-block plan
// (enum ops, buffer slots, device weight pointers, numeric attributes), so a
// call does no JSON or string work. GEMMs are FP32 cuBLAS (default math); the
// other kernels follow the CPU forms (graph_ir_cuda.h), sliding-window
// attention included, so the two paths agree to rounding. A T = 1 call
// (decode) replays a CUDA graph of the whole step, captured once, with the
// token and position in device memory (EMBER_IR_GRAPH=0: per-launch).
// Mixture-of-Recursions (depth_route, kv_rows "active") runs on the device
// too: routing on the host from the device's router logits, each recursion
// step batched over its active rows at their absolute positions into the
// application's compact kv cache (mor_step_device); its decode runs
// per-launch.
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <set>
#include <vector>

#include <nlohmann/json.hpp>

#include "ggml_types.h"
#include "graph_ir_expr.h"
#include "model.h"

struct GGUFData;

namespace ember_ir {
namespace rec {
class Recurrence;
}

struct Node {
  std::string op;
  std::vector<std::string> in, out;
  nlohmann::json attrs;  // the node's JSON object
};

class Interpreter {
 public:
  // Parses the IR JSON and binds every weight it names from the GGUF.
  // Throws std::runtime_error naming the op, tensor or field at fault, or the
  // IR_SPEC rule a document breaks (graph_ir_validate.h). A
  // weight kept quantised may point into the GGUF's memory map, so `gguf`
  // must outlive the interpreter.
  // use_gpu selects the device path (CUDA builds; ignored otherwise).
  Interpreter(const std::string& ir_json, const GGUFData& gguf, bool use_gpu = false);
  ~Interpreter();
  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;

  // Runs tokens at absolute positions start_pos .. start_pos + T - 1 (the KV
  // state must already hold positions < start_pos). Returns logits for the
  // last token ([vocab]) or, with all_rows, for every token ([T][vocab]).
  // tail_rows > 0 (all_rows false): the last min(tail_rows, T) tokens' logits, each row the one
  // all_rows gives (the head ops are row-wise); POST /score's continuation rows.
  std::vector<float> forward(const std::vector<int>& tokens, int start_pos, bool all_rows = false, int tail_rows = 0);

  // True when a block declares recurrent state (see the header comment).
  bool has_recurrent() const { return rec_floats_ > 0; }
  // The largest position <= want from which forward() can resume the current
  // sequence: want itself without recurrent state; else want when the state
  // or a snapshot is there, otherwise the nearest snapshot below it (0 at
  // worst: a fresh start). Under turn_state eviction: want, unless want lies
  // inside a segment m whose text rows were evicted; then the start of m.
  int resume_point(int want) const;

  int vocab() const { return vocab_; }
  int max_len() const { return max_len_; }
  int n_layers() const { return n_layers_; }
  const std::string& family_hint() const { return family_hint_; }
  // Mixture-of-Recursions (depth_route): the recursion depth of each row of the last forward
  // (1 .. n_rec), its routing trace; empty when the block has no depth_route.
  const std::vector<int>& last_depths() const { return mor_depth_; }
  // The absolute positions of the kv rows application `application` holds, ascending, on either path
  // (the bookkeeping is host-side on both): a recursion step's compact cache under depth_route
  // (kv_rows "active"), the compact cache under turn_state, else 0 .. seq_len() - 1; empty without a
  // kv slot. kv_positions() below is the CPU-only conformance hook.
  std::vector<int> held_positions(int application) const;
  bool on_gpu() const { return gpu_; }

  // Per-turn state tokens (paperkiln IR_SPEC_DRAFT.md section 1, tiny_chat_long_2026-10).
  // state_k(): the k state slots of the extended embedding ("state_weight", [k, d]); ids vocab ..
  // vocab + k - 1 read its rows; 0 without one. A state block is exactly those k ids in ascending
  // order; forward() refuses a state id out of order and a text id inside an open block.
  // Attention "mask": "turn_state": a key j is visible from the query at p iff s(j) == s(p), or j
  // is a state position with s(j) < s(p), s(p) being the number of complete state blocks before p.
  // evicts(): the kv slot's "evict": "turn_state" is in force (EMBER_EVICT=0 keeps every row;
  // EMBER_EVICT=1 evicts for any turn_state IR): once a later segment starts, the text rows of
  // the finished segments are dropped and their state rows kept, positions absolute. A forward
  // from a position inside a segment whose text rows are gone throws; resume_point() then names
  // that segment's start, where the caller re-prefills from.
  int state_k() const { return state_k_; }
  bool turn_state() const { return turn_state_; }
  bool evicts() const { return evict_; }
  // kv rows held per block (turn_state IRs; else the positions written so far).
  int kv_rows_held() const { return turn_state_ ? static_cast<int>(kv_pos_.size()) : static_cast<int>(ts_seg_.size()); }

  // Conformance hooks (tools/ir_adapter.cpp; paperkiln spec/IR_SPEC.md section 11.3). CPU path
  // only (each throws on the device path). No other caller uses them, and none changes what
  // forward() computes: the accessors read the state, truncate() and load_state() set it to what
  // a fresh run over the same ids would leave.
  // kv_slot(): the kv slot's name (empty without one). evict_declared(): the document's kv slot
  // says "evict": "turn_state" (evicts() is what is in force after EMBER_EVICT). seq_len(): the
  // positions the state covers (the end of the last forward, truncate or load_state).
  const std::string& kv_slot() const { return kv_slot_; }
  bool evict_declared() const { return evict_declared_; }
  int seq_len() const { return seq_len_; }
  bool has_depth_route() const { return !mor_step_.empty(); }
  // The absolute positions of the k/v rows application `application`'s kv cache holds, ascending:
  // under turn_state the compact cache's (after any eviction), under depth_route a recursion
  // step's compact cache, else 0 .. seq_len() - 1. Empty without a kv slot.
  std::vector<int> kv_positions(int application) const;
  // The recurrent slots' tensors in declaration order, and one tensor's values at an application.
  struct RecurrentTensorInfo {
    std::string slot, tensor;
    std::vector<int64_t> shape;
  };
  std::vector<RecurrentTensorInfo> recurrent_tensors() const;
  std::vector<float> recurrent_values(const std::string& slot, const std::string& tensor, int application) const;
  // Drops the state of positions >= n (IR_SPEC.md section 6.9), as a forward from n would before
  // running. n must be a point the state can resume from (resume_point(n) == n): the caller
  // re-prefills from resume_point(n) otherwise.
  void truncate(int n);
  // The whole sequence state (kv rows in use, state-token bookkeeping, MoR row sets, recurrent
  // tensors and their snapshots), copied out and back (the adapter's snapshot / restore).
  struct State {
    int seq_len = 0;
    std::vector<std::vector<float>> k, v;  // per application: its rows in use
    std::vector<int> ts_seg, ts_sidx, kv_pos;
    int evicted_below = 0;
    std::vector<std::vector<int>> mor_rows;
    std::vector<int> mor_depth;
    std::vector<float> rec;
    int rec_pos = 0;
    std::map<int, std::vector<float>> rec_snaps;
  };
  State save_state() const;
  void load_state(const State& s);

 private:
  void init(const nlohmann::json& j, const std::string& ir_json, const GGUFData& gguf, bool use_gpu);
  int kv_rows_in_use(int application) const;  // rows 0 .. n-1 of the application's cache are live
  std::string kv_slot_;
  bool evict_declared_ = false;
  int seq_len_ = 0;
  struct Buf {
    std::vector<float> v;
    int width = 0;
    bool is_list = false;    // a list value (op list / append): its tensors are `items`, v is empty
    std::vector<Buf> items;  // in order; each a tensor
  };
  using Env = std::unordered_map<std::string, Buf>;
  // A bound weight: f32 values, or (quantised) the tensor's raw GGUF bytes at qp: inside the
  // GGUF's memory map when it is mapped (no copy; the GGUFData outlives the interpreter), else
  // in q.
  struct Weight {
    std::vector<float> f32;
    std::vector<uint8_t> q;
    const uint8_t* qp = nullptr;
    size_t qbytes = 0;
    GGMLType type = GGMLType::GGML_TYPE_F32;
    size_t numel = 0;
    bool quantised() const { return qp != nullptr; }
  };

  void run_nodes(const std::vector<Node>& nodes, int layer, Env& env, int T, int start_pos, bool head_last_only);
  // The attention node's own window/sinks when it names them, else its kv state's.
  // Per-layer values ({"per_layer": [...]}) are read for `layer`.
  void attention_window(const Node& n, int layer, int& window, int& sinks) const;
  // The RoPE table of block `layer` (its theta; one table per distinct theta).
  const std::vector<std::pair<float, float>>& rope_table(int layer) const;
  const std::vector<float>& weight(const std::string& name) const;  // f32 weights only
  const Weight& bound(const std::string& name) const;
  std::string layer_name(const std::string& pattern, int layer) const;
  // The node being bound, run or compiled carries "tensor_index": "application" (IR_SPEC 7.2): its tensor
  // names' {i} is the application index, not the stored layer m_i (layer_name). Set per node by AppIndex.
  bool app_index_ = false;
  struct AppIndex {
    bool& flag;
    bool saved;
    AppIndex(bool& f, const Node& n) : flag(f), saved(f) { flag = n.attrs.contains("tensor_index"); }
    ~AppIndex() { flag = saved; }
  };
  void bind(const std::vector<Node>& nodes, int layer, const GGUFData& gguf);
  // keep_quantised: the tensor is only read row-wise (linear, embedding).
  void bind_tensor(const std::string& name, const GGUFData& gguf, bool keep_quantised = false);

  std::vector<Node> embed_, block_, head_;
  // depth_attend's source count S per node and application (-1 outside the block), fixed at load by the
  // lists' construction (IR_SPEC 5.18).
  std::map<std::pair<const Node*, int>, int> da_sources_;
  int n_layers_ = 0, vocab_ = 0, max_len_ = 0;
  // block.weight_index (paperkiln ir.hpp; tied-block recursion): application i
  // reads stored layer weight_map_[i]'s tensors. Empty: the identity.
  std::vector<int> weight_map_;
  // Mixture-of-Recursions (paperkiln ir.hpp depth_route, causal token choice). mor_step_[i]: the
  // recursion step of application i, -1 outside it (empty: no depth_route). A step's applications
  // run on its active tokens only (depth > step), one at a time in position order; each such
  // application keeps a compact kv cache holding just the rows written there, mor_rows_[i] their
  // absolute positions (kv_rows "active"), so attention's causal kernel reads exactly those rows
  // and RoPE keeps the absolute position. attn_row_ >= 0 places the next kv row (and the query's
  // causal limit) in that compact cache.
  void mor_step(int layer, Env& env, int T, int start_pos);
  std::vector<int> mor_step_;
  std::vector<std::vector<int>> mor_rows_;
  int mor_R_ = 0;
  bool mor_gate_score_ = false;
  std::string mor_x_, mor_out_, mor_router_;
  std::vector<int> mor_depth_;
  std::vector<float> mor_g_, mor_xin_;
  int attn_row_ = -1;
  // Per-turn state tokens (see state_k()). ts_seg_[p] = s(p), ts_sidx_[p] = j for the state id
  // vocab + j, -1 for text, over the positions of the current sequence. Under turn_state the kv
  // cache is compact: row r holds position kv_pos_[r] (ascending; the identity without eviction),
  // a call's rows start at kv_row0_, and ts_vis_[t] lists the rows its token t sees, ascending.
  // Text rows of segments below evicted_below_ are gone.
  void ts_begin(const std::vector<int>& tokens, int start_pos);
  void ts_evict_below(int seg);
  int ts_seg_after(int p) const;  // s of a token placed at position p + 1
  int ts_segment_start(int m) const;
  int state_k_ = 0;
  std::string state_weight_;
  bool turn_state_ = false, evict_ = false;
  std::vector<int> ts_seg_, ts_sidx_, kv_pos_;
  std::vector<std::vector<int>> ts_vis_;
  int kv_row0_ = 0, evicted_below_ = 0;
  std::string family_hint_;
  bool dequant_all_ = false;  // every weight f32 at load (see the header comment)
  std::unordered_map<std::string, Weight> weights_;
  std::unordered_map<std::string, std::vector<int64_t>> shapes_;  // ggml order (ne0 first)
  std::vector<KVCacheLayer> kv_;  // per block: [max_len][kv_heads][head_dim]
  int kv_heads_ = 0, head_dim_ = 0;
  int window_ = 0, sinks_ = 0;  // kv state's sliding window (0 = unbounded) and sinks
  std::vector<std::pair<float, float>> rope_freqs_;  // [max_len][rope_dim_ / 2]
  float rope_theta_ = 0.0f;
  int rope_dim_ = 0;         // n_rot (head_dim_ when the rope node names none)
  bool rope_neox_ = false;   // pairing "neox"
  float rope_freq_scale_ = 1.0f;  // linear RoPE scaling: the angle is (pos * freq_scale) * theta^(-2j/n_rot)
  // Per-layer attributes (paperkiln ir.hpp, {"per_layer": [v_0, ..., v_{L-1}]}): the kv state's window
  // per block (empty when one value serves every block), each block's RoPE theta and freq_scale, a
  // table per distinct (theta, freq_scale).
  std::vector<int> window_layer_;
  std::vector<float> rope_theta_layer_, rope_scale_layer_;
  std::map<std::pair<float, float>, std::vector<std::pair<float, float>>> rope_tables_;
  std::set<std::string> offset_applied_;  // rmsnorm weights already read as w + weight_offset
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
  std::vector<float> rec_init_;  // every application's initial state, [n_layers][rec_floats_]
  // The recurrence nodes of the block (IR_SPEC 6.10; graph_ir_recurrence.h), compiled at load.
  std::map<const Node*, std::shared_ptr<rec::Recurrence>> recs_;
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
  // A list value (op list / append) is a DBuf with is_list set and `items`
  // tensors, held as copies in the slot's ditems_ (a list is a value: a later
  // write of a tensor it was built from does not change it).
  struct DBuf {
    float* alloc = nullptr;
    float* p = nullptr;
    size_t cap = 0;  // floats
    int width = 0, rows = 0;  // rows == 0: not defined in this call
    bool is_list = false;
    int items = 0;
  };
  // One op of the device plan, resolved once at load (compile_device): the op
  // as an enum, its tensors as slots, its weights as device pointers and its
  // attributes as numbers, so a call does no JSON, string or map work. Only
  // the checks that depend on the call (widths, positions) remain per call.
  enum class DOp {
    Embedding, PosEmbedding, LayerNorm, Split, Gelu, RmsNorm, Linear, Rope, Attention, Add, Mul, MoeFfn, Silu,
    Sub, Relu, Sigmoid, Affine, MulCol, RmsRow, List, Append, DepthAttend, LinearAttention, SsmScan, Expr,
    Conv1dCausal, SelectiveScan, Scale
  };
  struct DStep {
    DOp op = DOp::Add;
    const Node* node = nullptr;  // names for error messages
    std::vector<int> in, out;    // slots
    std::string wname;           // the resolved weight (error messages)
    const float* w = nullptr;    // weight: table, norm scale, linear [out, in], moe router, depth_attend query, ssm A
    const float* b = nullptr;    // bias (layernorm, linear); ssm B
    const float *gate = nullptr, *up = nullptr, *down = nullptr;  // moe_ffn expert stacks
    const float *ssm_c = nullptr, *ssm_d = nullptr;               // ssm_scan C [n], D (a device scalar)
    float *rec_a = nullptr, *rec_b = nullptr;  // device state: linear_attention num, den; ssm_scan s
    const RecSlot* rec = nullptr;              // the recurrent op's state slot
    size_t w_elems = 0;
    int w_ne0 = 0;  // the weight's ne0 (rows of width ne0)
    int hd = 0, hq = 0, hkv = 0, window = 0, sinks = 0;  // rope, attention, linear_attention (hq heads)
    int E = 0, top_k = 0, d_ff = 0;                      // moe_ffn
    int n = 0;  // ssm_scan state size; rmsnorm group (0: the row); conv1d_causal, selective_scan channels
    int n_rot = 0, kernel = 0, d_state = 0;  // rope n_rot; conv1d_causal kernel; selective_scan d_state
    bool neox = false;                       // rope pairing
    const float* rope = nullptr;             // rope: this block's (cos, sin) table on the device
    std::vector<expr::Stage> expr;           // expr: the device program (graph_ir_expr.h lower)
    float eps = 1e-5f, scale = 0.0f, shift = 0.0f;       // eps: also linear_attention's den_floor
    bool flag = false;  // gelu: tanh form; moe_ffn: renorm gates
    std::vector<int> sizes;  // split
    const float* state_w = nullptr;  // embedding: the state rows (state_weight), ids vocab .. vocab + k - 1
    bool ts = false;                 // attention: mask turn_state
  };
  void init_device();
  void compile_device();
  std::vector<DStep> compile_nodes(const std::vector<Node>& nodes, int layer);
  int slot(const std::string& name);
  void free_device();
  const float* dweight(const std::string& name) const;
  // (Re)allocates b to hold `need` floats when it is smaller. A growth during
  // graph capture throws (the capture is abandoned; see capture_decode).
  void grow(DBuf& b, size_t need);
  DBuf& dout(int slot, int rows, int width, std::initializer_list<const DBuf*> inputs);
  DBuf& dout(int slot, int rows, int width, const std::vector<const DBuf*>& inputs);
  void copy_item(DBuf& dst, const DBuf& x);
  void commit();
  void run_nodes_device(const std::vector<DStep>& steps, int layer, int T, int start_pos);
  std::vector<float> forward_device(const std::vector<int>& tokens, int start_pos, bool all_rows, int tail_rows);
  void run_device(int T, int start_pos);  // embed, blocks; not the head
  // The captured T = 1 step: true when it ran (logits in hlogits_).
  bool decode_graph(int token, int start_pos);
  bool capture_decode(int start_pos);
  float* dsnap_take(int key);  // the device snapshot buffer for `key`, created or reused
  void dsnap_drop(std::map<int, float*>::iterator it);

  cublasHandle_t cublas_ = nullptr;
  cudaStream_t stream_ = nullptr;
  std::unordered_map<std::string, float*> dweights_;
  std::unordered_map<std::string, int> slots_;  // tensor name -> slot (load time only)
  std::vector<std::string> slot_names_;
  std::vector<DBuf> dbufs_, dspare_;            // by slot
  std::vector<std::vector<DBuf>> ditems_;       // by slot: a list value's tensors
  std::vector<int> pending_;  // slots written to their spare by dout, swapped by commit
  std::vector<DStep> plan_embed_, plan_head_;
  std::vector<std::vector<DStep>> plan_block_;  // per block
  int logits_slot_ = -1;
  std::vector<float*> dk_, dv_;  // per block: [max_len][kv_heads][head_dim]
  float* drope_ = nullptr;  // the first block's table (one of drope_tables_)
  std::map<std::pair<float, float>, float*> drope_tables_;  // rope_tables_ on the device
  DBuf dexpr_;  // expr scratch (a reduction's argument and its columns)
  int* dtokens_ = nullptr;
  int dtokens_cap_ = 0;
  std::vector<int> htokens_;  // this call's tokens (range checks on the host)
  DBuf dmoe_;  // moe_ffn scratch
  // Per-turn state tokens on the device: the host bookkeeping is the CPU path's (ts_begin,
  // ts_evict_below, kv_pos_); dts_code_ [max_len_] holds each kv row's code 2 s + c for the
  // turn_state kernel, uploaded before every call (ts_upload_codes), and eviction compacts the
  // device caches in place through the row list dkv_idx_ [max_len_]. A call's first kv row is
  // kv_row0_ (dstep_[2] in the captured decode step).
  void ts_upload_codes(int start_pos, int T);
  int* dts_code_ = nullptr;
  int* dkv_idx_ = nullptr;

  // Mixture-of-Recursions on the device (mor_step_device, the CPU's mor_step). Routing: the router
  // logits x W^T by cuBLAS, then each token's depth and gate on the host by cpu::mor_route (the CPU's
  // code; one sync per forward), so the depth is EXACT given the logits. A step holds the full value of
  // every slot the block reads or writes aside (mor_touch_, swapped into msave_), gathers the active
  // rows of x and of the block's other inputs (mor_livein_, re-gathered at each application, as the
  // CPU rebuilds each token's row from the full values) and runs the block's plan with T = the active
  // count: rope at each row's absolute position (drowpos_, absent when every row is active), attention
  // over the application's compact cache from row attn_row_ (the rows it holds, mor_rows_). After the
  // step's last application the gate blends the compact x with its input, the rows go back into the
  // full x (inactive rows keep x_in) and the held values return, as the CPU's env keeps them. A forward
  // through a depth_route never takes the captured decode graph (the routing reads the host).
  void mor_step_device(int layer, int T, int start_pos);
  int mor_xs_ = -1, mor_out_slot_ = -1;  // slots of depth_route's input and of its depth report
  std::vector<int> mor_touch_, mor_livein_;
  std::vector<DBuf> msave_;  // by slot: the full value held aside during a step
  DBuf dmor_logits_, dmor_xin_, dmor_g_;
  int* dmor_idx_ = nullptr;  // [2 cap]: the step's active rows, then their absolute positions
  int dmor_idx_cap_ = 0;
  const int* drowpos_ = nullptr;  // non-null inside a step whose rows are not all active
  std::vector<int> mor_active_;   // this step's active rows (host)
  int mor_A_ = 0;

  // Device recurrent state: drec_ [n_layers][rec_floats_] (fixed: a restore
  // copies into it, so a captured graph keeps reading the live state);
  // drec_init_ the initial state of every block; snapshots and this call's
  // kept rows as on the CPU path (rec_begin / rec_row_done / rec_end), by
  // device-to-device copies.
  float* drec_ = nullptr;
  float* drec_init_ = nullptr;
  std::map<int, float*> dsnaps_;  // state after `key` tokens
  std::vector<float*> dsnap_pool_;  // released snapshot buffers
  DBuf drec_rows_;  // this call: [kept row][n_layers][rec_floats_]

  // CUDA graph of the T = 1 step (EMBER_IR_GRAPH=0 disables): captured once,
  // replayed with the token and position in device memory (dstep_ = [token,
  // pos, kv row]; dpos_ points at the position while capturing). Any buffer growth
  // (alloc_epoch_) invalidates it; it is captured again at the next T = 1.
  cudaGraphExec_t graph_exec_ = nullptr;
  int* dstep_ = nullptr;
  int* hstep_ = nullptr;       // pinned
  float* hlogits_ = nullptr;   // pinned, [vocab]
  const int* dpos_ = nullptr;  // non-null only while capturing
  bool capturing_ = false, graph_off_ = false;
  int graph_failures_ = 0;     // consecutive failed captures
  int graph_captures_ = 0;
  unsigned long long alloc_epoch_ = 0, graph_epoch_ = 0;
  const float* graph_logits_ = nullptr;
  int graph_vocab_ = 0;
  int embed_rows_ = 0, pos_rows_ = 0;  // replay range checks: token ids, positions
#endif
};

// The IR for a GGUF: the EMBER_IR side file if set, else the `paperkiln.ir`
// key, else the IR build_ir_from_gguf makes for a qwen2 / qwen3 / gpt2 /
// gptneox / gemma3 / mamba file (and llama / mistral when the file names a
// sliding window, or with EMBER_IR_BUILD=1), else empty (the llama path).
// Throws for a file with no IR whose architecture ember cannot run, naming it.
std::string find_ir(const GGUFData& gguf);

// The IR of a third-party GGUF from its metadata and tensor names, node for
// node as paperkiln's tools/gguf_ir.py builds it:
//   llama, mistral, qwen2, qwen3: pre-RMSNorm blocks, q/k/v biases when the
//     file has blk.0.attn_q.bias, per-head q/k RMSNorm (group head_dim) when it
//     has blk.0.attn_q_norm.weight, RoPE pairing "neox" for qwen2/qwen3 and
//     "interleaved" for llama/mistral, grouped-query attention (llama/mistral's
//     attention.sliding_window as the window), SwiGLU;
//   gpt2, gptneox: LayerNorm blocks over a fused biased qkv, gelu (tanh for
//     gpt2, exact for gptneox); gpt2 learned positions, gptneox partial NEOX
//     RoPE and (use_parallel_residual) llama.cpp's parallel residual;
//   gemma3: sqrt(d)-scaled embeddings, sandwich RMSNorms, per-head q/k
//     RMSNorm, NEOX RoPE and GeGLU; window and theta 10000 on layers with
//     i % 6 < 5, the file's theta (and linear scaling) on the rest;
//   mamba: Mamba-1 blocks (conv1d_causal, selective_scan) with no positions.
// The head is tied to token_embd.weight when output.weight is absent (gptneox
// always has its own). Throws naming the architecture or the missing key.
std::string build_ir_from_gguf(const GGUFData& gguf);

// ember's tuned llama path (model.cpp, api.cpp) computes one graph, and a document may take it only when it
// is that graph (paperkiln spec/IR_SPEC.md 9.1: family_hint is never semantics; it is a pre-filter here):
// a valid document (graph_ir_validate.h) with family_hint "llama" whose nodes are, modulo value names and the
// operand order of add and mul,
//   embed  embedding(tokens; token_embd.weight)
//   block  rmsnorm(attn_norm) -> linear attn_q, attn_k, attn_v (no bias) -> rope (interleaved, every head,
//          n_rot = head_dim, freq_scale 1, the q / k heads) -> attention (the one kv slot: window 0, sinks 0,
//          causal mask, scale 1/sqrt(head_dim), no evict, kv_rows or weight_index) -> linear attn_output -> add
//          -> rmsnorm(ffn_norm) -> SwiGLU (ffn_gate, ffn_up, silu, mul, ffn_down) or moe_ffn (ffn_gate_inp and
//          the *_exps stacks, gate_norm renorm, silu) -> add
//   head   rmsnorm(output_norm) -> linear output.weight (tied to token_embd.weight or not)
// with the numbers the tuned path reads from the GGUF's metadata instead (parse_model_config_from_gguf: d,
// layers, heads, kv heads, head_dim = d / heads, every eps, theta, experts and experts used, hidden_act silu).
// Otherwise the interpreter runs the document. why (optional) gets the first difference.
bool tuned_llama_match(const std::string& ir_json, const GGUFData& gguf, std::string* why = nullptr);

// True when the IR declares sliding-window attention (a kv state or an
// attention node with window != 0). Only the interpreter implements it, so
// such a GGUF never runs on the llama path, whatever its family_hint.
bool has_windowed_attention(const std::string& ir_json);

// True when the IR declares recurrent state (a block.state of kind
// "recurrent").
bool has_recurrent_state(const std::string& ir_json);

// True when the CUDA path has a kernel for every op of the IR (recurrent
// state, per-turn state tokens: state_weight, mask "turn_state", evict, and
// Mixture-of-Recursions: depth_route, kv_rows "active", with no pos_embedding
// in its block) within its limits (depth_attend sources, ssm_scan n);
// otherwise the interpreter must run on the CPU.
bool device_supported(const std::string& ir_json);

// True when the IR uses per-turn state tokens: an embedding with
// "state_weight" or an attention node with "mask": "turn_state". Only the
// interpreter implements them, so such a GGUF never runs on the llama path.
bool has_state_tokens(const std::string& ir_json);

// The recurrent ops for one token (row), in paperkiln's order of operations
// (graph_ir.cpp); the state is updated in place. Exposed for the op tests.
//   linear_attention: q, k, v, num, den, out each `width` = heads * head_dim.
void linear_attention_row_cpu(const float* q, const float* k, const float* v, float* num, float* den, float* out,
                              int width, float den_floor);
//   ssm_scan: u, s, y [n]; A [n, n] row-major; B, C [n]; scratch [n].
void ssm_scan_row_cpu(const float* u, const float* A, const float* B, const float* C, float D, float* s,
                      float* scratch, float* y, int n);

}  // namespace ember_ir
