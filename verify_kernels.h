// Kernels of the speculative verification pass (speculative_verify.cpp, EMBER_SPEC=k on CUDA).
//
// A verification pass runs n = 1 + draft rows (positions pos .. pos + n - 1) through every layer in one CUDA graph,
// built like the graph decode (model.cpp forward_device_graph). Every kernel here computes each row with the decode
// step's arithmetic: the same statements, the same reduction order, the same attention chunking. Only the linear
// layers have a choice (VerifyLinear): kRows is the decode step's own cuBLAS call per row, and kFixed is
// fixed_linear, the decode step's linear kernel when the decode runs on it (EMBER_DECODE_LINEAR=fixed, opt-in;
// experiments/spec_r1_2026-10-28). Apart from fixed_linear these kernels serve only
// the pass. pos lives in device memory (pos_dev[0]) and the n
// token ids after it (pos_dev[1 .. n]), so one captured graph per n serves every position.
#ifndef EMBER_VERIFY_KERNELS_H
#define EMBER_VERIFY_KERNELS_H

#ifdef HAS_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace ember_verify {

constexpr int kMaxRows = 16;  // rows per pass on this path (EMBER_SPEC <= 15)

// The linear layers Y[n][O] = X[n][I] W[O][I]^T, FP32.
enum class VerifyLinear {
  kSmallM = 0,  // one warp per output, all n rows at once: the weights are read once (not the decode's order)
  kGemm = 1,    // cublasSgemm with n columns (the batched prefill's call; not the decode's order)
  kGemvBatched = 2,  // cublasSgemvStridedBatched, the weight shared (stride 0)
  kRows = 3,    // the decode step's cublasSgemm(n = 1), once per row: the decode's arithmetic exactly
  kFixed = 4,   // fixed_linear: kSmallM's kernel with a scalar twin for any shape; each row's arithmetic is the
                // same for every n, so with the decode step on fixed_linear (n = 1) a row is the decode's bits
};

// Y[n][O] = X[n][I] W[O][I]^T, one warp per output and every row at once, each row's dot product a fixed fma chain
// (lane l takes the inputs 4 (l + 32 j) .. 4 (l + 32 j) + 3, j = 0, 1, ..., in order) and a fixed butterfly over
// the warp. A row's result depends only on its own X row and W, never on n, the other rows, the alignment or the
// stream: the float4 kernel (I % 4 == 0, X and W 16-byte aligned) and its scalar twin (any other shape) compute the
// same chain. 1 <= n <= kMaxRows.
void fixed_linear(const float* X, const float* W, float* Y, int n, int O, int I, cudaStream_t st);

// Per row of the last pass: the largest logit, the first and last index holding it, the largest logit below it
// (-inf if none), and whether the row holds a NaN. Enough to reproduce api.cpp's greedy sampler without the row.
struct RowPick {
  float max;
  float second;
  int first;
  int last;
  int nan;
};

// out[r][i] = table[ids[r]][i] (decode_embed_kernel's rule per row); ids = pos_dev + 1.
void embed_rows(const float* table, float* out, const int* pos_dev, int n, int hidden, int vocab, cudaStream_t st);
// rmsnorm_vector_cuda_optimized per row (its deterministic kernel under EMBER_DETERMINISTIC).
void rmsnorm_rows(const float* x, const float* w, float* out, int n, int hidden, float eps, cudaStream_t st);
// rope_devpos_cuda per row at position pos + r; x is [n][heads * head_dim].
void rope_rows(float* x, int n, int heads, int head_dim, const float* freqs, const int* pos_dev, bool adjacent,
               cudaStream_t st);
// kv_write_devpos_cuda per row at position pos + r; k, v are [n][row].
void kv_write_rows(float* k_cache, float* v_cache, const float* k, const float* v, const int* pos_dev, int n, int row,
                   cudaStream_t st);
// The graph decode's split-K attention per row (row r attends to pos + r + 1 keys), the kernel and chunking
// attention_devpos_cuda uses (the tiled kernel's block bodies, decode_attn_tiled.cuh, by default; EMBER_DECODE_ATTN=split
// the 256-key kernel for max_len positions). scratch: attention_rows_scratch_floats(...). False when the decode itself
// would not use a split-K kernel for this shape (EMBER_DECODE_ATTN=legacy, or past its limits): the caller then takes
// another path.
size_t attention_rows_scratch_floats(int n, int heads, int head_dim, int max_len);
// q, out: rows row0 .. row0 + n - 1 of the pass (row0 = 0 for all rows at once; the per-row form launches it with
// n = 1 per row, for measurement).
bool attention_rows(const float* q, const float* k_cache, const float* v_cache, float* out, int n, int row0,
                    int heads, int head_dim, float scale, int max_len, int kv_heads, const int* pos_dev,
                    float* scratch, cudaStream_t st);
// Y[n][O] = X[n][I] W[O][I]^T. kSmallM needs n <= kMaxRows, I % 4 == 0 and 16-byte aligned X, W (else it runs as
// kGemm). The handle's stream is set to st.
void linear_rows(cublasHandle_t h, VerifyLinear mode, const float* X, const float* W, float* Y, int n, int O, int I,
                 cudaStream_t st);
// picks[r] for logits [n][vocab].
void select_rows(const float* logits, int n, int vocab, RowPick* picks, cudaStream_t st);

}  // namespace ember_verify

#endif  // HAS_CUDA
#endif  // EMBER_VERIFY_KERNELS_H
