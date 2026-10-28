// Kernels of the speculative verification pass; see verify_kernels.h. Each kernel that has a decode-step
// counterpart in cuda_kernels.cu repeats that kernel's statements with a row index added, so a row here gives the
// decode step's values bit for bit for the same inputs (checked by experiments/verify_2026-10-27/vf_bench.cpp).
#include "verify_kernels.h"

#ifdef HAS_CUDA
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "cuda_kernels.h"
#include "decode_attn_tiled.cuh"
#include "determinism.h"

namespace ember_verify {
namespace {

// ---- embedding: decode_embed_kernel per row -------------------------------------------------------------------
__global__ void embed_rows_kernel(const float* __restrict__ table, float* __restrict__ out,
                                  const int* __restrict__ pos_dev, int hidden_size, int vocab_size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i >= hidden_size) return;
  const int tok = pos_dev[1 + r];
  out[(size_t)r * hidden_size + i] = (tok >= 0 && tok < vocab_size) ? table[(size_t)tok * hidden_size + i] : 0.0f;
}

// ---- rmsnorm: rmsnorm_optimized_kernel / rmsnorm_optimized_det_kernel per row ----------------------------------
// The decode wrappers launch one block of 256 threads, so their grid stride is blockDim.x and their global thread
// index is threadIdx.x; here blockIdx.x is the row.
__global__ void rmsnorm_rows_kernel(const float* __restrict__ x_all, const float* __restrict__ weight,
                                    float* __restrict__ out_all, int n, float eps) {
  const float* x = x_all + (size_t)blockIdx.x * n;
  float* out = out_all + (size_t)blockIdx.x * n;
  const int tid_global = threadIdx.x;
  const int tid_block = threadIdx.x;

  __shared__ float s_block_sum_sq;

  if (tid_block == 0) {
    s_block_sum_sq = 0.0f;
  }
  __syncthreads();

  float thread_partial_sum_sq = 0.0f;
  for (int i = tid_global; i < n; i += blockDim.x) {
    float val = x[i];
    thread_partial_sum_sq += val * val;
  }

  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    thread_partial_sum_sq += __shfl_down_sync(0xFFFFFFFF, thread_partial_sum_sq, offset);
  }

  if ((tid_block % warpSize) == 0) {
    atomicAdd(&s_block_sum_sq, thread_partial_sum_sq);
  }
  __syncthreads();

  float total_sum_sq = s_block_sum_sq;
  float rms_scale = rsqrtf(total_sum_sq / n + eps);

  for (int i = tid_global; i < n; i += blockDim.x) {
    out[i] = x[i] * weight[i] * rms_scale;
  }
}

__global__ void rmsnorm_rows_det_kernel(const float* __restrict__ x_all, const float* __restrict__ weight,
                                        float* __restrict__ out_all, int n, float eps) {
  const float* x = x_all + (size_t)blockIdx.x * n;
  float* out = out_all + (size_t)blockIdx.x * n;
  const int tid_global = threadIdx.x;
  const int tid_block = threadIdx.x;

  __shared__ float s_warp_sum_sq[32];
  __shared__ float s_block_sum_sq;

  float thread_partial_sum_sq = 0.0f;
  for (int i = tid_global; i < n; i += blockDim.x) {
    float val = x[i];
    thread_partial_sum_sq += val * val;
  }

  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    thread_partial_sum_sq += __shfl_down_sync(0xFFFFFFFF, thread_partial_sum_sq, offset);
  }

  if ((tid_block % warpSize) == 0) {
    s_warp_sum_sq[tid_block / warpSize] = thread_partial_sum_sq;
  }
  __syncthreads();
  if (tid_block == 0) {
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    float total = 0.0f;
    for (int w = 0; w < num_warps; ++w) total += s_warp_sum_sq[w];
    s_block_sum_sq = total;
  }
  __syncthreads();

  float total_sum_sq = s_block_sum_sq;
  float rms_scale = rsqrtf(total_sum_sq / n + eps);

  for (int i = tid_global; i < n; i += blockDim.x) {
    out[i] = x[i] * weight[i] * rms_scale;
  }
}

// ---- rope: rope_devpos_kernel per row --------------------------------------------------------------------------
__global__ void rope_rows_kernel(float* x_all, int num_heads, int head_dim, const float* all_freqs_cis_base,
                                 const int* pos_dev, bool use_adjacent_pairing) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_pairs = num_heads * (head_dim / 2);
  if (idx >= total_pairs) return;
  float* x = x_all + (size_t)blockIdx.y * num_heads * head_dim;
  const int pos = pos_dev[0] + blockIdx.y;
  const int head_idx = idx / (head_dim / 2);
  const int dim_pair_idx = idx % (head_dim / 2);
  const size_t freq_base_offset = ((size_t)pos * (head_dim / 2) + dim_pair_idx) * 2;
  int base_x0, base_x1;
  if (use_adjacent_pairing) {
    base_x0 = head_idx * head_dim + (dim_pair_idx * 2);
    base_x1 = head_idx * head_dim + (dim_pair_idx * 2 + 1);
  } else {
    base_x0 = head_idx * head_dim + dim_pair_idx;
    base_x1 = head_idx * head_dim + dim_pair_idx + (head_dim / 2);
  }
  const float x0 = x[base_x0];
  const float x1 = x[base_x1];
  const float cos_val = all_freqs_cis_base[freq_base_offset];
  const float sin_val = all_freqs_cis_base[freq_base_offset + 1];
  x[base_x0] = x0 * cos_val - x1 * sin_val;
  x[base_x1] = x0 * sin_val + x1 * cos_val;
}

// ---- KV write: kv_write_devpos_kernel per row ------------------------------------------------------------------
__global__ void kv_write_rows_kernel(float* k_cache, float* v_cache, const float* __restrict__ k,
                                     const float* __restrict__ v, const int* pos_dev, int row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i >= row) return;
  const size_t off = (size_t)(pos_dev[0] + r) * row + i;
  k_cache[off] = k[(size_t)r * row + i];
  v_cache[off] = v[(size_t)r * row + i];
}

// ---- attention: decode_attn_split_kernel + decode_attn_merge_kernel per row ------------------------------------
// The constants and the chunking are cuda_kernels.cu's (decode_attn_split); blockIdx.z is the row, whose length is
// pos + row + 1, and each row has its own scratch block instead of the static array.
constexpr int kDecChunkMax = 2048;
constexpr int kDecChunksMax = 64;
constexpr int kDecHeadsMax = 128;
constexpr int kDecHeadDimMax = 256;
constexpr int kDecThreads = 128;

__global__ void attn_split_rows_kernel(const float* __restrict__ Q_all, const float* __restrict__ K,
                                       const float* __restrict__ V, int head_dim, float scale, int num_kv_heads,
                                       int num_heads, int chunk, const int* pos_dev, int row0,
                                       float* __restrict__ scratch) {
  const int r = blockIdx.z;
  const int seq_len = pos_dev[0] + row0 + r + 1;
  const float* Q = Q_all + (size_t)r * num_heads * head_dim;
  const int h = blockIdx.x, c = blockIdx.y, nc = gridDim.y;
  float* part = scratch + (((size_t)r * num_heads + h) * nc + c) * (head_dim + 2);  // [m, l, acc[head_dim]]
  const int k0 = c * chunk;
  const int k1 = min(seq_len, k0 + chunk);
  if (k0 >= k1) {
    if (threadIdx.x == 0) { part[0] = -INFINITY; part[1] = 0.0f; }
    return;
  }
  const int kvh = h / (num_heads / num_kv_heads);
  extern __shared__ float sm[];
  float* q = sm;                   // [head_dim]
  float* s = sm + head_dim;        // [chunk]
  __shared__ float red[kDecThreads / 32];
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) q[d] = Q[h * head_dim + d];
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  for (int k = k0 + warp; k < k1; k += nwarps) {
    const float* kv = K + ((size_t)k * num_kv_heads + kvh) * head_dim;
    float dot = 0.0f;
    for (int d = lane; d < head_dim; d += 32) dot += q[d] * kv[d];
    for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
    if (lane == 0) s[k - k0] = dot * scale;
  }
  __syncthreads();
  const int n = k1 - k0;
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, s[i]);
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  mx = red[0];
  for (int w = 1; w < nwarps; ++w) mx = fmaxf(mx, red[w]);
  __syncthreads();
  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float e = expf(s[i] - mx);
    s[i] = e;
    sum += e;
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
  if (lane == 0) red[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float l = 0.0f;
    for (int w = 0; w < nwarps; ++w) l += red[w];
    part[0] = mx;
    part[1] = l;
  }
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += s[i] * V[((size_t)(k0 + i) * num_kv_heads + kvh) * head_dim + d];
    part[2 + d] = acc;
  }
}

__global__ void attn_merge_rows_kernel(float* __restrict__ out_all, int head_dim, int nc, int num_heads,
                                       const float* __restrict__ scratch) {
  const int h = blockIdx.x, r = blockIdx.y;
  float* out = out_all + (size_t)r * num_heads * head_dim;
  const float* base = scratch + ((size_t)r * num_heads + h) * nc * (head_dim + 2);
  float M = -INFINITY;
  for (int c = 0; c < nc; ++c) {
    const float* p = base + (size_t)c * (head_dim + 2);
    if (p[1] > 0.0f) M = fmaxf(M, p[0]);
  }
  float L = 0.0f;
  for (int c = 0; c < nc; ++c) {
    const float* p = base + (size_t)c * (head_dim + 2);
    if (p[1] > 0.0f) L += expf(p[0] - M) * p[1];
  }
  const float inv = L > 0.0f ? 1.0f / L : 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int c = 0; c < nc; ++c) {
      const float* p = base + (size_t)c * (head_dim + 2);
      if (p[1] > 0.0f) acc += expf(p[0] - M) * p[2 + d];
    }
    out[h * head_dim + d] = acc * inv;
  }
}

// ---- attention: decode_attn_tiled_kernel + decode_attn_tiled_merge_kernel per row ------------------------------
// The same block bodies (decode_attn_tiled.cuh) at row r's length pos + row0 + r + 1, each row with its own scratch.
__global__ void attn_tiled_rows_kernel(const float* __restrict__ Q_all, const float* __restrict__ K,
                                       const float* __restrict__ V, int head_dim, float scale, int num_kv_heads,
                                       int num_heads, int min_chunk, int max_chunks, const int* pos_dev, int row0,
                                       float* __restrict__ scratch) {
  const int r = blockIdx.z;
  const int L = pos_dev[0] + row0 + r + 1;
  const int h = blockIdx.x, c = blockIdx.y, nc = gridDim.y;
  const float* Q = Q_all + (size_t)r * num_heads * head_dim;
  ember_tiled::tiled_split_block(Q + (size_t)h * head_dim, K, V, L, c, h / (num_heads / num_kv_heads), num_kv_heads,
                                 head_dim, scale, min_chunk, max_chunks,
                                 scratch + (((size_t)r * num_heads + h) * nc + c) * (head_dim + 2));
}

__global__ void attn_tiled_merge_rows_kernel(float* __restrict__ out_all, int head_dim, int nc, int num_heads,
                                             int min_chunk, int max_chunks, const int* pos_dev, int row0,
                                             const float* __restrict__ scratch) {
  const int h = blockIdx.x, r = blockIdx.y;
  const int L = pos_dev[0] + row0 + r + 1;
  ember_tiled::tiled_merge_block(scratch + ((size_t)r * num_heads + h) * nc * (head_dim + 2),
                                 out_all + (size_t)r * num_heads * head_dim + (size_t)h * head_dim, L, head_dim,
                                 min_chunk, max_chunks);
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// decode_attn_tiled's grid for a cache of max_len positions (cuda_kernels.cu); false where it would refuse.
bool tiled_shape(int heads, int head_dim, int max_len, int kv_heads, int* nc_out, int* mn_out, int* mx_out) {
  if (decode_attention_kind() != 0) return false;
  int mn = 0, mx = 0;
  decode_attention_tiles(&mn, &mx);
  if (!ember_tiled::tiled_shape_ok(heads, head_dim, max_len, kDecHeadsMax, mn, mx) || kv_heads <= 0 ||
      heads % kv_heads != 0)
    return false;
  *nc_out = ember_tiled::tiled_grid_chunks(max_len, mn, mx);
  *mn_out = mn;
  *mx_out = mx;
  return true;
}

// decode_attn_split's chunking for a cache of max_len positions; false where it would refuse.
bool split_shape(int heads, int head_dim, int max_len, int* nc_out, int* chunk_out) {
  if (decode_attention_kind() == 2 || heads > kDecHeadsMax || head_dim > kDecHeadDimMax || max_len <= 0) return false;
  int nc = (max_len + 255) / 256;
  if (nc > kDecChunksMax) nc = kDecChunksMax;
  const int chunk = (max_len + nc - 1) / nc;
  if (chunk > kDecChunkMax) return false;
  *nc_out = nc;
  *chunk_out = chunk;
  return true;
}

// ---- linear layers: one warp per output row, every row of X at once --------------------------------------------
// Each lane takes float4 slices of the weight row (lane, lane + 32, ...) and accumulates M dot products with one
// fma chain per row; a butterfly over the warp then adds the 32 lane sums. The weight is read once per pass.
template <int M>
__global__ void __launch_bounds__(256) linear_smallm_kernel(const float* __restrict__ X, const float* __restrict__ W,
                                                            float* __restrict__ Y, int O, int I) {
  const int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (o >= O) return;
  const float4* w4 = reinterpret_cast<const float4*>(W + (size_t)o * I);
  const int I4 = I >> 2;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.0f;
  for (int i = lane; i < I4; i += 32) {
    const float4 a = __ldg(w4 + i);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float4 b = __ldg(reinterpret_cast<const float4*>(X + (size_t)m * I) + i);
      float t = acc[m];
      t = fmaf(a.x, b.x, t);
      t = fmaf(a.y, b.y, t);
      t = fmaf(a.z, b.z, t);
      t = fmaf(a.w, b.w, t);
      acc[m] = t;
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = acc[m];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    if (lane == 0) Y[(size_t)m * O + o] = v;
  }
}

template <int M>
void launch_smallm(const float* X, const float* W, float* Y, int O, int I, cudaStream_t st) {
  constexpr int kWarps = 8;
  linear_smallm_kernel<M><<<(O + kWarps - 1) / kWarps, kWarps * 32, 0, st>>>(X, W, Y, O, I);
}

bool smallm(const float* X, const float* W, float* Y, int n, int O, int I, cudaStream_t st) {
  if ((I & 3) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0 || (reinterpret_cast<uintptr_t>(W) & 15) != 0)
    return false;
  switch (n) {
#define EMBER_VERIFY_CASE(m) \
  case m:                    \
    launch_smallm<m>(X, W, Y, O, I, st); \
    return true;
    EMBER_VERIFY_CASE(1) EMBER_VERIFY_CASE(2) EMBER_VERIFY_CASE(3) EMBER_VERIFY_CASE(4)
    EMBER_VERIFY_CASE(5) EMBER_VERIFY_CASE(6) EMBER_VERIFY_CASE(7) EMBER_VERIFY_CASE(8)
    EMBER_VERIFY_CASE(9) EMBER_VERIFY_CASE(10) EMBER_VERIFY_CASE(11) EMBER_VERIFY_CASE(12)
    EMBER_VERIFY_CASE(13) EMBER_VERIFY_CASE(14) EMBER_VERIFY_CASE(15) EMBER_VERIFY_CASE(16)
#undef EMBER_VERIFY_CASE
    default:
      return false;
  }
}

// linear_smallm_kernel's arithmetic with scalar loads, for shapes its float4 loads refuse: lane l takes the same
// groups of four inputs, 4 (l + 32 j) .. +3 (the last group cut at I), in the same order, then the same butterfly.
template <int M>
__global__ void __launch_bounds__(256) linear_fixed_scalar_kernel(const float* __restrict__ X,
                                                                  const float* __restrict__ W, float* __restrict__ Y,
                                                                  int O, int I) {
  const int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (o >= O) return;
  const float* w = W + (size_t)o * I;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.0f;
  for (int i = 4 * lane; i < I; i += 128) {
    const int e = min(I, i + 4);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float* x = X + (size_t)m * I;
      float t = acc[m];
      for (int j = i; j < e; ++j) t = fmaf(__ldg(w + j), __ldg(x + j), t);
      acc[m] = t;
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = acc[m];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    if (lane == 0) Y[(size_t)m * O + o] = v;
  }
}

void fixed_scalar(const float* X, const float* W, float* Y, int n, int O, int I, cudaStream_t st) {
  constexpr int kWarps = 8;
  const int blocks = (O + kWarps - 1) / kWarps;
  switch (n) {
#define EMBER_VERIFY_CASE(m) \
  case m:                    \
    linear_fixed_scalar_kernel<m><<<blocks, kWarps * 32, 0, st>>>(X, W, Y, O, I); \
    return;
    EMBER_VERIFY_CASE(1) EMBER_VERIFY_CASE(2) EMBER_VERIFY_CASE(3) EMBER_VERIFY_CASE(4)
    EMBER_VERIFY_CASE(5) EMBER_VERIFY_CASE(6) EMBER_VERIFY_CASE(7) EMBER_VERIFY_CASE(8)
    EMBER_VERIFY_CASE(9) EMBER_VERIFY_CASE(10) EMBER_VERIFY_CASE(11) EMBER_VERIFY_CASE(12)
    EMBER_VERIFY_CASE(13) EMBER_VERIFY_CASE(14) EMBER_VERIFY_CASE(15) EMBER_VERIFY_CASE(16)
#undef EMBER_VERIFY_CASE
    default:
      throw std::runtime_error("fixed_linear: n out of range");
  }
}

// ---- greedy pick per row ---------------------------------------------------------------------------------------
constexpr int kSelThreads = 256;

__global__ void select_rows_kernel(const float* __restrict__ logits, int vocab, RowPick* __restrict__ picks) {
  const float* l = logits + (size_t)blockIdx.x * vocab;
  __shared__ float s_f[kSelThreads / 32];
  __shared__ int s_a[kSelThreads / 32];
  __shared__ int s_b[kSelThreads / 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  float mx = -INFINITY;
  int nan = 0;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float v = l[i];
    if (v != v) nan = 1;
    else mx = fmaxf(mx, v);
  }
  for (int o = 16; o > 0; o >>= 1) {
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    nan |= __shfl_xor_sync(0xffffffff, nan, o);
  }
  if (lane == 0) { s_f[warp] = mx; s_a[warp] = nan; }
  __syncthreads();
  mx = s_f[0];
  nan = s_a[0];
  for (int w = 1; w < nwarps; ++w) { mx = fmaxf(mx, s_f[w]); nan |= s_a[w]; }
  __syncthreads();
  int first = INT_MAX, last = -1;
  float second = -INFINITY;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float v = l[i];
    if (v == mx) {
      first = min(first, i);
      last = max(last, i);
    } else if (v < mx) {
      second = fmaxf(second, v);
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    first = min(first, __shfl_xor_sync(0xffffffff, first, o));
    last = max(last, __shfl_xor_sync(0xffffffff, last, o));
    second = fmaxf(second, __shfl_xor_sync(0xffffffff, second, o));
  }
  if (lane == 0) { s_f[warp] = second; s_a[warp] = first; s_b[warp] = last; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < nwarps; ++w) {  // warp 0's values are this thread's
      second = fmaxf(second, s_f[w]);
      first = min(first, s_a[w]);
      last = max(last, s_b[w]);
    }
    RowPick p;
    p.max = mx;
    p.second = second;
    p.first = first;
    p.last = last;
    p.nan = nan;
    picks[blockIdx.x] = p;
  }
}

void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("verify kernels: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

void embed_rows(const float* table, float* out, const int* pos_dev, int n, int hidden, int vocab, cudaStream_t st) {
  embed_rows_kernel<<<dim3((hidden + 255) / 256, n), 256, 0, st>>>(table, out, pos_dev, hidden, vocab);
  check_launch("embed_rows");
}

void rmsnorm_rows(const float* x, const float* w, float* out, int n, int hidden, float eps, cudaStream_t st) {
  if (ember::deterministic())
    rmsnorm_rows_det_kernel<<<n, 256, 0, st>>>(x, w, out, hidden, eps);
  else
    rmsnorm_rows_kernel<<<n, 256, sizeof(float), st>>>(x, w, out, hidden, eps);
  check_launch("rmsnorm_rows");
}

void rope_rows(float* x, int n, int heads, int head_dim, const float* freqs, const int* pos_dev, bool adjacent,
               cudaStream_t st) {
  const int total_pairs = heads * (head_dim / 2);
  rope_rows_kernel<<<dim3((total_pairs + 255) / 256, n), 256, 0, st>>>(x, heads, head_dim, freqs, pos_dev, adjacent);
  check_launch("rope_rows");
}

void kv_write_rows(float* k_cache, float* v_cache, const float* k, const float* v, const int* pos_dev, int n, int row,
                   cudaStream_t st) {
  kv_write_rows_kernel<<<dim3((row + 255) / 256, n), 256, 0, st>>>(k_cache, v_cache, k, v, pos_dev, row);
  check_launch("kv_write_rows");
}

size_t attention_rows_scratch_floats(int n, int heads, int head_dim, int max_len) {
  // Room for either kernel (the tiled one falls back to split for a misaligned pointer).
  int nc = 0, chunk = 0, tnc = 0, mn = 0, mx = 0;
  const bool split = split_shape(heads, head_dim, max_len, &nc, &chunk);
  const bool tiled = tiled_shape(heads, head_dim, max_len, heads, &tnc, &mn, &mx);
  if (!split && !tiled) return 0;
  const int chunks = std::max(split ? nc : 0, tiled ? tnc : 0);
  return (size_t)n * heads * chunks * (head_dim + 2);
}

bool attention_rows(const float* q, const float* k_cache, const float* v_cache, float* out, int n, int row0,
                    int heads, int head_dim, float scale, int max_len, int kv_heads, const int* pos_dev,
                    float* scratch, cudaStream_t st) {
  int tnc = 0, mn = 0, mx = 0;
  if (tiled_shape(heads, head_dim, max_len, kv_heads, &tnc, &mn, &mx) && aligned16(q) && aligned16(k_cache) &&
      aligned16(v_cache)) {
    const size_t shm = ember_tiled::tiled_shm_bytes(head_dim, max_len, mn, mx);
    attn_tiled_rows_kernel<<<dim3(heads, tnc, n), ember_tiled::kTiledThreads, shm, st>>>(
        q, k_cache, v_cache, head_dim, scale, kv_heads, heads, mn, mx, pos_dev, row0, scratch);
    attn_tiled_merge_rows_kernel<<<dim3(heads, n), ember_tiled::kTiledThreads, 0, st>>>(out, head_dim, tnc, heads, mn,
                                                                                        mx, pos_dev, row0, scratch);
    check_launch("attention_rows");
    return true;
  }
  int nc = 0, chunk = 0;
  if (!split_shape(heads, head_dim, max_len, &nc, &chunk)) return false;
  const size_t shm = (size_t)(head_dim + chunk) * sizeof(float);
  attn_split_rows_kernel<<<dim3(heads, nc, n), kDecThreads, shm, st>>>(q, k_cache, v_cache, head_dim, scale,
                                                                         kv_heads, heads, chunk, pos_dev, row0,
                                                                         scratch);
  attn_merge_rows_kernel<<<dim3(heads, n), kDecThreads, 0, st>>>(out, head_dim, nc, heads, scratch);
  check_launch("attention_rows");
  return true;
}

void fixed_linear(const float* X, const float* W, float* Y, int n, int O, int I, cudaStream_t st) {
  if (n < 1 || n > kMaxRows) throw std::runtime_error("fixed_linear: n out of range");
  if (!smallm(X, W, Y, n, O, I, st)) fixed_scalar(X, W, Y, n, O, I, st);
  check_launch("fixed_linear");
}

void linear_rows(cublasHandle_t h, VerifyLinear mode, const float* X, const float* W, float* Y, int n, int O, int I,
                 cudaStream_t st) {
  cublasSetStream(h, st);
  const float one = 1.0f, zero = 0.0f;
  switch (mode) {
    case VerifyLinear::kFixed:
      fixed_linear(X, W, Y, n, O, I, st);
      return;
    case VerifyLinear::kSmallM:
      if (n <= kMaxRows && smallm(X, W, Y, n, O, I, st)) {
        check_launch("linear_smallm");
        return;
      }
      break;
    case VerifyLinear::kGemvBatched: {
      const cublasStatus_t s = cublasSgemvStridedBatched(h, CUBLAS_OP_T, I, O, &one, W, I, 0, X, 1, I, &zero, Y, 1,
                                                         O, n);
      if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("verify: cublasSgemvStridedBatched failed");
      return;
    }
    case VerifyLinear::kRows:
      for (int r = 0; r < n; ++r)
        matvec_f32_f32_cuda(h, W, X + (size_t)r * I, Y + (size_t)r * O, O, I, st);
      return;
    case VerifyLinear::kGemm:
      break;
  }
  gemm_rm_f32_cuda(h, X, W, Y, n, O, I, st);
}

void select_rows(const float* logits, int n, int vocab, RowPick* picks, cudaStream_t st) {
  select_rows_kernel<<<n, kSelThreads, 0, st>>>(logits, vocab, picks);
  check_launch("select_rows");
}

}  // namespace ember_verify
#endif  // HAS_CUDA
