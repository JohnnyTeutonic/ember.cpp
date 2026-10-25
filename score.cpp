// POST /score's forward (api.h score_rows; the request handling and mtstudio's arithmetic are in score.h
// and server.cpp).
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "api.h"
#include "graph_ir.h"
#include "model.h"
#ifdef HAS_CUDA
#include "cuda_kernels.h"
#endif

namespace tinyllama {

namespace {
// EMBER_SCORE_FULL=1: every request a fresh prefill that computes every row's logits, the rows then
// picked (the reference the default path is checked against; mtstudio score's own computation).
bool score_full() {
  static const bool on = [] {
    const char* v = std::getenv("EMBER_SCORE_FULL");
    return v && std::string(v) == "1";
  }();
  return on;
}
}  // namespace

int TinyLlamaSession::score_vocab() const { return ir_ ? ir_->vocab() : config_.vocab_size; }

int TinyLlamaSession::score_context() const {
  int ctx = config_.max_position_embeddings;
  if (ir_ && ir_->max_len() > 0) ctx = std::min(ctx, ir_->max_len());
  return ctx;
}

const std::vector<std::string>& TinyLlamaSession::gguf_tokens() const {
  static const std::vector<std::string> none;
  const GGUFData* g = model_ ? model_->get_gguf_data() : nullptr;
  return g ? g->tokenizer_tokens : none;
}

bool TinyLlamaSession::score_reuse_ok() const {
  static const bool off = [] {
    const char* v = std::getenv("EMBER_SCORE_REUSE");
    return v && std::string(v) == "0";
  }();
  if (off || score_full()) return false;
  if (ir_)
    return !ir_->on_gpu() && !ir_->has_recurrent() && !ir_->evicts() && !ir_->has_depth_route() &&
           ir_->state_k() == 0 && !ir_->turn_state();
  return config_.num_cpu_offload_layers == config_.num_hidden_layers;  // every layer on the CPU
}

std::vector<float> TinyLlamaSession::score_rows(const std::vector<int>& ids, int first_row, bool reuse_prefix,
                                                int* reused) {
  const int n = static_cast<int>(ids.size());
  if (n == 0 || first_row < 0 || first_row >= n) throw std::invalid_argument("score_rows: first_row out of range");
  if (n > score_context())
    throw std::invalid_argument("score_rows: " + std::to_string(n) + " positions exceed the context of " +
                                std::to_string(score_context()));
  const int V = score_vocab();
  kv_token_history_.clear();  // generate() must not reuse what this leaves in the cache
  int reuse = 0;
  if (reuse_prefix && score_reuse_ok()) {
    size_t lcp = 0;
    while (lcp < score_ids_.size() && lcp < ids.size() && score_ids_[lcp] == ids[lcp]) ++lcp;
    reuse = static_cast<int>(std::min(lcp, static_cast<size_t>(first_row)));  // rows >= first_row are computed
  }
  score_ids_.clear();  // set again only when this forward completes
  if (reused) *reused = reuse;
  const int m = n - reuse;         // positions computed
  const int rows = n - first_row;  // rows returned: the last `rows` of them
  // The default path computes the head on the returned rows only, where that is row-wise by
  // construction (the CPU paths); EMBER_SCORE_FULL=1 and the device compute every row.
  std::vector<float> out;
  if (ir_) {
    const std::vector<int> tail(ids.begin() + reuse, ids.end());
    if (!score_full() && !ir_->on_gpu()) {
      out = ir_->forward(tail, reuse, /*all_rows=*/false, rows);
    } else {
      const std::vector<float> all = ir_->forward(tail, reuse, /*all_rows=*/true);
      if (all.size() != static_cast<size_t>(m) * V)
        throw std::runtime_error("score: the interpreter returned " + std::to_string(all.size()) + " logits");
      out.assign(all.end() - static_cast<long>(rows) * V, all.end());
    }
  } else {
    // As generate()'s prefill: positions [0, reuse) kept, every later one rewritten before it is read.
    if (reuse > 0) {
      kv_cache_.seq_len = reuse;
    } else {
      kv_cache_.clear_data();
      kv_cache_.seq_len = 0;
    }
    const int hs = config_.hidden_size;
    std::vector<float> h(static_cast<size_t>(m) * hs);
    for (int i = 0; i < m; ++i) {
      const std::vector<float> e = model_->lookup_embedding(ids[reuse + i]);
      if (e.size() != static_cast<size_t>(hs))
        throw std::runtime_error("score: no embedding for token id " + std::to_string(ids[reuse + i]));
      std::copy(e.begin(), e.end(), h.begin() + static_cast<long>(i) * hs);
    }
    const int L = config_.num_hidden_layers, cpu_layers = config_.num_cpu_offload_layers;
    if (cpu_layers > 0) h = model_->forward_cpu_batch(h, m, cpu_layers, reuse, &kv_cache_);
    if (h.size() != static_cast<size_t>(m) * hs) throw std::runtime_error("score: the CPU layers failed");
    if (cpu_layers == L) {
      // The final norm and the LM head are row-wise (rmsnorm_batch_cpu; matmul_f32_f32_batch_cpu's
      // one dot product per row and output), so the returned rows alone give the bits all rows give.
      std::vector<float> keep = h;
      if (!score_full()) keep.assign(h.end() - static_cast<long>(rows) * hs, h.end());
      const int k_rows = score_full() ? m : rows;
      std::vector<float> lg = model_->forward_cpu_logits_batch(keep, k_rows);
      if (lg.size() != static_cast<size_t>(k_rows) * V) throw std::runtime_error("score: the LM head failed");
      out.assign(lg.end() - static_cast<long>(rows) * V, lg.end());
    } else {
#ifdef HAS_CUDA
      float* d = nullptr;
      gpuErrchk(cudaMalloc(&d, h.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
      // FP32 GEMMs only (no BF16 tensor cores or TF32, which /chat's prefill may use: they move
      // log-probabilities by ~1e-3), and every row's logits (the same FP32 matvec per row as
      // decode's LM head).
      struct Fp32Only {
        TinyLlamaModel& model;
        explicit Fp32Only(TinyLlamaModel& mdl) : model(mdl) { model.set_gemm_fp32_only(true); }
        ~Fp32Only() { model.set_gemm_fp32_only(false); }
      } fp32_only(*model_);
      const std::vector<float> all =
          model_->forward_device_batch_prefill(d, m, reuse, &kv_cache_, 0, /*all_logits=*/true);
      gpuErrchk(cudaFree(d));
      if (all.size() != static_cast<size_t>(m) * V)
        throw std::runtime_error("score: the device prefill returned " + std::to_string(all.size()) + " logits");
      out.assign(all.end() - static_cast<long>(rows) * V, all.end());
#else
      throw std::runtime_error("score: GPU layers requested but CUDA not available");
#endif
    }
    kv_cache_.seq_len = n;
  }
  if (out.size() != static_cast<size_t>(rows) * V)
    throw std::runtime_error("score: " + std::to_string(out.size()) + " logits for " + std::to_string(rows) + " rows");
  score_ids_ = ids;
  return out;
}

}  // namespace tinyllama
