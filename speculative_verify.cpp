// Speculative verification on CUDA as a captured graph per row count (EMBER_SPEC=k; experiments/verify_2026-10-27).
//
// forward_device_verify (model.cpp) runs a pass of n = 1 + draft rows through cuBLAS GEMMs and one decode-attention
// launch per row, launch by launch, then copies every row's logits to pageable memory: about 2.5 decode steps. Here
// the same pass is one CUDA graph per n, captured once and replayed with the position and the ids in device memory
// (as forward_device_graph replays the decode step), every row goes through the decode step's own kernels
// (verify_kernels.h), and the greedy pick of every row is made on the device, so the host reads n small records
// instead of n logit rows. api.cpp reproduces its sampler from the record and fetches a row only when the record
// cannot decide (a near tie, a NaN, a repetition penalty).
#include "model.h"

#ifdef HAS_CUDA
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "logger.h"
#include "verify_kernels.h"

using ember_verify::kMaxRows;
using ember_verify::RowPick;
using ember_verify::VerifyLinear;

TinyLlamaModel::VerifyOptions& TinyLlamaModel::verify_options() {
  if (!verify_opts_read_) {
    verify_opts_read_ = true;
    const char* lin = std::getenv("EMBER_VERIFY_LINEAR");
    const std::string l = lin ? lin : "";
    verify_opts_.linear = l == "gemm"    ? static_cast<int>(VerifyLinear::kGemm)
                          : l == "gemvb" ? static_cast<int>(VerifyLinear::kGemvBatched)
                          : l == "rows"  ? static_cast<int>(VerifyLinear::kRows)
                                         : static_cast<int>(VerifyLinear::kSmallM);
    const char* g = std::getenv("EMBER_VERIFY_GRAPH");
    verify_opts_.graph = !(g && std::string(g) == "0");
    const char* a = std::getenv("EMBER_VERIFY_ATTN");
    verify_opts_.batched_attention = !(a && std::string(a) == "perrow");
    verify_opts_.timing = std::getenv("EMBER_VERIFY_TIMING") != nullptr;
  }
  return verify_opts_;
}

// The buffers forward_device_graph allocates on its first call (the decode stream, the step and logit staging and
// the FP32 embedding table, exactly as it fills them), for a pass that comes before any decode step. Whichever runs
// first allocates them; the other finds decode_stream_ set and uses them.
bool TinyLlamaModel::verify_ensure_decode_buffers() {
  if (decode_stream_) return decode_embed_f32_dev_ != nullptr;
  const int hs = config_.hidden_size, vs = config_.vocab_size;
  std::vector<float> table(static_cast<size_t>(vs) * hs);
  for (int t = 0; t < vs; ++t) {
    const std::vector<float> e = lookup_embedding(t);
    if (static_cast<int>(e.size()) != hs) return false;
    std::copy(e.begin(), e.end(), table.begin() + static_cast<size_t>(t) * hs);
  }
  gpuErrchk(cudaStreamCreate(&decode_stream_));
  gpuErrchk(cudaMalloc(&decode_step_dev_, 2 * sizeof(int)));
  gpuErrchk(cudaHostAlloc(reinterpret_cast<void**>(&decode_step_host_), 2 * sizeof(int), cudaHostAllocDefault));
  gpuErrchk(cudaHostAlloc(reinterpret_cast<void**>(&decode_logits_host_), static_cast<size_t>(vs) * sizeof(float),
                          cudaHostAllocDefault));
  gpuErrchk(cudaMalloc(&decode_embed_f32_dev_, table.size() * sizeof(float)));
  gpuErrchk(cudaMemcpy(decode_embed_f32_dev_, table.data(), table.size() * sizeof(float), cudaMemcpyHostToDevice));
  return true;
}

// The pass for n rows on stream st: forward_device_graph's sequence of operations with a row dimension.
void TinyLlamaModel::verify_run_ops(int n, KVCache* kv_cache, cudaStream_t st) {
  const VerifyOptions& o = verify_options();
  const VerifyLinear lin = static_cast<VerifyLinear>(o.linear);
  const int L = config_.num_hidden_layers, hs = config_.hidden_size, is = config_.intermediate_size;
  const int vs = config_.vocab_size, n_heads = config_.num_attention_heads, n_kv = config_.num_key_value_heads;
  const int head_dim = hs / n_heads, kv_row = n_kv * head_dim;
  const float eps = config_.rms_norm_eps;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const int max_len = kv_cache->allocated_max_seq_len;
  float* x = verify_rows_dev_;
  float* xn = x + (size_t)kMaxRows * hs;
  float* resid = xn + (size_t)kMaxRows * hs;
  float* q = resid + (size_t)kMaxRows * hs;
  float* attn = q + (size_t)kMaxRows * hs;
  float* proj = attn + (size_t)kMaxRows * hs;
  float* k = proj + (size_t)kMaxRows * hs;
  float* v = k + (size_t)kMaxRows * kv_row;
  float* gate = v + (size_t)kMaxRows * kv_row;
  float* up = gate + (size_t)kMaxRows * is;
  float* logits = verify_logits_dev_;
  const int* sd = verify_step_dev_;

  // EMBER_VERIFY_TIMING: each stage synchronised and charged its event time (graph off).
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  if (o.timing) {
    gpuErrchk(cudaEventCreate(&e0));
    gpuErrchk(cudaEventCreate(&e1));
  }
  auto stage = [&](const char* name, auto&& fn) {
    if (!o.timing) {
      fn();
      return;
    }
    gpuErrchk(cudaEventRecord(e0, st));
    fn();
    gpuErrchk(cudaEventRecord(e1, st));
    gpuErrchk(cudaEventSynchronize(e1));
    float ms = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&ms, e0, e1));
    verify_stage_ms_[name] += ms;
  };
  auto linear = [&](const float* X, const float* W, float* Y, int O, int I) {
    ember_verify::linear_rows(cublas_handle_, lin, X, W, Y, n, O, I, st);
  };

  stage("embed", [&] { ember_verify::embed_rows(decode_embed_f32_dev_, x, sd, n, hs, vs, st); });
  for (int l = 0; l < L; ++l) {
    KVCacheLayer& kvl = kv_cache->layers[l];
    stage("copy", [&] {
      gpuErrchk(cudaMemcpyAsync(resid, x, (size_t)n * hs * sizeof(float), cudaMemcpyDeviceToDevice, st));
    });
    stage("rmsnorm", [&] { ember_verify::rmsnorm_rows(x, layers[l].input_layernorm_dev, xn, n, hs, eps, st); });
    stage("qkv", [&] {
      linear(xn, w_q_f32_dev_ + (size_t)l * hs * hs, q, hs, hs);
      linear(xn, w_k_f32_dev_ + (size_t)l * kv_row * hs, k, kv_row, hs);
      linear(xn, w_v_f32_dev_ + (size_t)l * kv_row * hs, v, kv_row, hs);
    });
    stage("rope_kv", [&] {
      ember_verify::rope_rows(q, n, n_heads, head_dim, all_freqs_cis_dev, sd, config_.is_gguf_file_loaded, st);
      ember_verify::rope_rows(k, n, n_kv, head_dim, all_freqs_cis_dev, sd, config_.is_gguf_file_loaded, st);
      ember_verify::kv_write_rows(kvl.k_dev_fp32, kvl.v_dev_fp32, k, v, sd, n, kv_row, st);
    });
    stage("attention", [&] {
      if (o.batched_attention) {
        ember_verify::attention_rows(q, kvl.k_dev_fp32, kvl.v_dev_fp32, attn, n, 0, n_heads, head_dim, scale, max_len,
                                     n_kv, sd, verify_attn_scratch_dev_, st);
      } else {
        for (int r = 0; r < n; ++r)
          ember_verify::attention_rows(q + (size_t)r * hs, kvl.k_dev_fp32, kvl.v_dev_fp32, attn + (size_t)r * hs, 1, r,
                                       n_heads, head_dim, scale, max_len, n_kv, sd, verify_attn_scratch_dev_, st);
      }
    });
    stage("o_proj", [&] { linear(attn, w_o_f32_dev_ + (size_t)l * hs * hs, proj, hs, hs); });
    stage("residual", [&] {
      add_residual_cuda(proj, resid, x, n * hs, st);
      gpuErrchk(cudaMemcpyAsync(resid, x, (size_t)n * hs * sizeof(float), cudaMemcpyDeviceToDevice, st));
    });
    stage("rmsnorm", [&] {
      ember_verify::rmsnorm_rows(x, layers[l].post_attention_layernorm_dev, xn, n, hs, eps, st);
    });
    stage("gate_up", [&] {
      linear(xn, w_gate_f32_dev_ + (size_t)l * is * hs, gate, is, hs);
      linear(xn, w_up_f32_dev_ + (size_t)l * is * hs, up, is, hs);
    });
    stage("swiglu", [&] { swiglu_cuda(gate, up, gate, n * is, st); });
    stage("down", [&] { linear(gate, w_down_f32_dev_ + (size_t)l * hs * is, proj, hs, is); });
    stage("residual", [&] { add_residual_cuda(proj, resid, x, n * hs, st); });
  }
  stage("final_norm", [&] { ember_verify::rmsnorm_rows(x, final_norm_dev, xn, n, hs, eps, st); });
  stage("lm_head", [&] { linear(xn, lm_head_f32_dev_, logits, vs, hs); });
  stage("select", [&] { ember_verify::select_rows(logits, n, vs, verify_picks_dev_, st); });
  if (o.timing) {
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  }
}

bool TinyLlamaModel::forward_device_verify_graph(const std::vector<int>& tokens, int pos, KVCache* kv_cache,
                                                 std::vector<RowPick>* picks) {
  static const bool decode_graph_on = [] {  // forward_device_graph's switches: the rows follow its kernels
    const char* v = std::getenv("EMBER_DECODE_GRAPH");
    const char* o = std::getenv("EMBER_DECODE_OPT");
    return !(v && std::string(v) == "0") && !(o && std::string(o) == "0");
  }();
  const int n = static_cast<int>(tokens.size());
  if (!decode_graph_on || verify_graph_disabled_ || decode_graph_disabled_ || n < 1 || n > kMaxRows || !picks)
    return false;
  const int L = config_.num_hidden_layers, hs = config_.hidden_size, is = config_.intermediate_size;
  const int vs = config_.vocab_size, n_heads = config_.num_attention_heads, n_kv = config_.num_key_value_heads;
  if (n_heads == 0 || n_kv == 0) return false;
  const int head_dim = hs / n_heads, kv_row = n_kv * head_dim;
  if (config_.num_cpu_offload_layers != 0 || config_.use_kvcache_quantization || !config_.use_optimized_cuda_kernels ||
      config_.num_experts > 0) {
    verify_graph_disabled_ = true;
    return false;
  }
  if (!kv_cache || static_cast<int>(kv_cache->layers.size()) < L || kv_cache->allocated_num_kv_heads != n_kv ||
      kv_cache->allocated_head_dim != head_dim || pos < 0 || pos + n > kv_cache->allocated_max_seq_len)
    return false;
  ensure_f32_concatenated_weights_loaded();
  ensure_lm_head_dequantized();
  if (!w_q_f32_dev_ || !w_k_f32_dev_ || !w_v_f32_dev_ || !w_o_f32_dev_ || !w_gate_f32_dev_ || !w_up_f32_dev_ ||
      !w_down_f32_dev_ || !lm_head_f32_dev_ || !final_norm_dev || !all_freqs_cis_dev) {
    verify_graph_disabled_ = true;
    return false;
  }
  const size_t scratch =
      ember_verify::attention_rows_scratch_floats(kMaxRows, n_heads, head_dim, kv_cache->allocated_max_seq_len);
  if (scratch == 0 || !verify_ensure_decode_buffers()) {
    verify_graph_disabled_ = true;
    return false;
  }
  if (!verify_rows_dev_) {
    // Rows: x, xn, resid, q, attn, proj [hs]; k, v [kv_row]; gate, up [is]; then logits [vocab].
    const size_t act = (size_t)kMaxRows * (6 * (size_t)hs + 2 * (size_t)kv_row + 2 * (size_t)is);
    gpuErrchk(cudaMalloc(&verify_rows_dev_, (act + (size_t)kMaxRows * vs) * sizeof(float)));
    verify_logits_dev_ = verify_rows_dev_ + act;
    gpuErrchk(cudaMalloc(&verify_step_dev_, (kMaxRows + 1) * sizeof(int)));
    gpuErrchk(cudaHostAlloc(reinterpret_cast<void**>(&verify_step_host_), (kMaxRows + 1) * sizeof(int),
                            cudaHostAllocDefault));
    gpuErrchk(cudaMalloc(&verify_picks_dev_, kMaxRows * sizeof(RowPick)));
    gpuErrchk(cudaHostAlloc(reinterpret_cast<void**>(&verify_picks_host_), kMaxRows * sizeof(RowPick),
                            cudaHostAllocDefault));
  }
  if (scratch > verify_attn_scratch_floats_) {
    if (verify_attn_scratch_dev_) gpuErrchk(cudaFree(verify_attn_scratch_dev_));
    gpuErrchk(cudaMalloc(&verify_attn_scratch_dev_, scratch * sizeof(float)));
    verify_attn_scratch_floats_ = scratch;
    for (auto& g : verify_graphs_) cudaGraphExecDestroy(g.second);  // they hold the old scratch
    verify_graphs_.clear();
  }

  const VerifyOptions& o = verify_options();
  const int opts_key = o.linear * 8 + (o.graph ? 4 : 0) + (o.batched_attention ? 2 : 0) + (o.timing ? 1 : 0);
  const void* key = kv_cache->layers[0].k_dev_fp32;
  if (key != verify_graph_key_ || opts_key != verify_graph_opts_key_) {
    for (auto& g : verify_graphs_) cudaGraphExecDestroy(g.second);
    verify_graphs_.clear();
    verify_graph_key_ = key;
    verify_graph_opts_key_ = opts_key;
  }

  cudaStream_t st = decode_stream_;
  verify_step_host_[0] = pos;
  for (int i = 0; i < n; ++i) verify_step_host_[1 + i] = tokens[i];
  gpuErrchk(cudaMemcpyAsync(verify_step_dev_, verify_step_host_, (n + 1) * sizeof(int), cudaMemcpyHostToDevice, st));
  if (o.graph && !o.timing) {
    auto it = verify_graphs_.find(n);
    if (it == verify_graphs_.end()) {
      if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        cudaGetLastError();
        verify_graph_disabled_ = true;
        Logger::warning("[verify graph] stream capture unavailable; using the per-launch verification.");
        return false;
      }
      cudaError_t e = cudaSuccess;
      try {
        verify_run_ops(n, kv_cache, st);
      } catch (const std::exception& ex) {
        Logger::warning(std::string("[verify graph] capture: ") + ex.what());
        e = cudaErrorUnknown;
      }
      cudaGraph_t graph = nullptr;
      const cudaError_t e_end = cudaStreamEndCapture(st, &graph);
      if (e == cudaSuccess) e = e_end;
      cublasSetStream(cublas_handle_, 0);
      cudaGraphExec_t exec = nullptr;
      if (e == cudaSuccess) e = cudaGraphInstantiate(&exec, graph, 0);
      if (graph) cudaGraphDestroy(graph);
      if (e != cudaSuccess) {
        cudaGetLastError();
        verify_graph_disabled_ = true;
        Logger::warning(std::string("[verify graph] capture failed (") + cudaGetErrorString(e) +
                        "); using the per-launch verification.");
        return false;
      }
      it = verify_graphs_.emplace(n, exec).first;
      Logger::info("[verify graph] captured the verification pass for " + std::to_string(n) + " rows.");
    }
    gpuErrchk(cudaGraphLaunch(it->second, st));
  } else {
    verify_run_ops(n, kv_cache, st);
    cublasSetStream(cublas_handle_, 0);
  }
  gpuErrchk(cudaMemcpyAsync(verify_picks_host_, verify_picks_dev_, n * sizeof(RowPick), cudaMemcpyDeviceToHost, st));
  gpuErrchk(cudaStreamSynchronize(st));
  picks->assign(verify_picks_host_, verify_picks_host_ + n);
  verify_last_n_ = n;
  return true;
}

std::vector<float> TinyLlamaModel::verify_row_logits(int r) {
  if (r < 0 || r >= verify_last_n_ || !verify_logits_dev_) return {};
  const int vs = config_.vocab_size;
  std::vector<float> out(vs);
  gpuErrchk(cudaMemcpy(out.data(), verify_logits_dev_ + (size_t)r * vs, (size_t)vs * sizeof(float),
                       cudaMemcpyDeviceToHost));
  return out;
}
#endif  // HAS_CUDA
