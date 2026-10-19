// CUDA kernels for the IR interpreter's device path (see graph_ir_cuda.h).
#include "graph_ir_cuda.h"

#include <stdexcept>
#include <string>

namespace ember_ir {
namespace {

constexpr int kThreads = 256;

void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("IR CUDA kernel ") + what + ": " + cudaGetErrorString(e));
}

unsigned blocks_for(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

__global__ void embedding_kernel(const float* __restrict__ table, const int* __restrict__ tokens,
                                 float* __restrict__ out, int d) {
  const int t = blockIdx.x;
  const float* src = table + (size_t)tokens[t] * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x) out[(size_t)t * d + i] = src[i];
}

__global__ void pos_embedding_kernel(const float* __restrict__ x, const float* __restrict__ P, float* __restrict__ out,
                                     int d, int start_pos) {
  const int t = blockIdx.x;
  const float* p = P + (size_t)(start_pos + t) * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x) out[(size_t)t * d + i] = __fadd_rn(x[(size_t)t * d + i], p[i]);
}

// One block per row; block-wide double sums in the CPU kernel's order of
// operations (mean first, then the centred squares).
__device__ double block_sum(double v, double* red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) red[warp] = v;
  __syncthreads();
  double s = 0.0;
  for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += red[w];
  __syncthreads();
  return s;
}

__global__ void layernorm_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                 float* __restrict__ out, int d, float eps) {
  __shared__ double red[kThreads / 32];
  const float* r = x + (size_t)blockIdx.x * d;
  float* o = out + (size_t)blockIdx.x * d;
  double s = 0.0;
  for (int i = threadIdx.x; i < d; i += blockDim.x) s += r[i];
  const double mean = block_sum(s, red) / d;
  double v = 0.0;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    const double c = r[i] - mean;
    v += c * c;
  }
  const double var = block_sum(v, red) / d;
  const float inv = static_cast<float>(1.0 / sqrt(var + eps));
  const float m = static_cast<float>(mean);
  for (int i = threadIdx.x; i < d; i += blockDim.x)
    o[i] = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(r[i], m), inv), w[i]), b[i]);
}

__global__ void bias_add_kernel(float* __restrict__ y, const float* __restrict__ b, size_t n, int O) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __fadd_rn(y[i], b[i % O]);
}

__global__ void binary_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out,
                              size_t n, int op) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = op == 0 ? __fadd_rn(a[i], b[i]) : __fmul_rn(a[i], b[i]);
}

__global__ void silu_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = x[i];
    out[i] = __fmul_rn(v, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-v))));
  }
}

__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n, int tanh_form) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  const float half_v = __fmul_rn(0.5f, v);
  if (tanh_form) {
    const float c = sqrtf(2.0f / 3.14159265358979323846f);
    const float cube = __fmul_rn(__fmul_rn(__fmul_rn(0.044715f, v), v), v);
    out[i] = __fmul_rn(half_v, __fadd_rn(1.0f, tanhf(__fmul_rn(c, __fadd_rn(v, cube)))));
  } else {
    out[i] = __fmul_rn(half_v, __fadd_rn(1.0f, erff(__fdiv_rn(v, sqrtf(2.0f)))));
  }
}

// One block per token, threads over (head, pair) of q then k.
__global__ void rope_kernel(float* __restrict__ q, float* __restrict__ k, int hq, int hk, int hd, int start_pos,
                            const float* __restrict__ freqs) {
  const int t = blockIdx.x, half = hd / 2;
  const float* f = freqs + (size_t)(start_pos + t) * half * 2;
  const int nq = hq * half, total = (hq + hk) * half;
  for (int j = threadIdx.x; j < total; j += blockDim.x) {
    const bool is_q = j < nq;
    const int jj = is_q ? j : j - nq;
    const int h = jj / half, i = jj % half;
    float* x = (is_q ? q + (size_t)t * hq * hd : k + (size_t)t * hk * hd) + (size_t)h * hd + 2 * i;
    const float c = f[2 * i], s = f[2 * i + 1];
    const float x0 = x[0], x1 = x[1];
    x[0] = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
    x[1] = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
  }
}

// Block-wide float max / sum (every thread gets the result).
__device__ float block_reduce(float v, bool is_max, float* red) {
  for (int o = 16; o > 0; o >>= 1) {
    const float u = __shfl_xor_sync(0xffffffff, v, o);
    v = is_max ? fmaxf(v, u) : v + u;
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float r = red[0];
  for (int w = 1; w < (int)(blockDim.x >> 5); ++w) r = is_max ? fmaxf(r, red[w]) : r + red[w];
  __syncthreads();
  return r;
}

constexpr int kWinChunk = 1024;   // visible keys scored per pass (shared memory)
constexpr int kWinDimsPerThread = 4;  // head_dim <= kThreads * 4

// Sliding-window attention with sinks, one block per (token, q head). The
// visible keys of query position p (cpu_attention.cpp swa_visible_ranges) are
// walked in ascending order, sinks then window, kWinChunk at a time: scores
// into shared memory (a warp per key), then an online softmax (running max,
// sum and output rescaled when a later chunk raises the max). With n <=
// kWinChunk visible keys this is the CPU form's single max / exp / sum pass.
// Masked keys are never read, so they contribute exactly zero.
__global__ void window_attention_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                        const float* __restrict__ V, float* __restrict__ out, int hq, int hkv,
                                        int hd, int start_pos, float scale, int window, int sinks) {
  const int t = blockIdx.x, h = blockIdx.y;
  const int pos = start_pos + t;
  const int win_lo = pos + 1 > window ? pos + 1 - window : 0;
  const int sink_hi = min(min(sinks, pos + 1), win_lo);
  const int n = sink_hi + (pos + 1 - win_lo);
  const int kvh = h / (hq / hkv);
  extern __shared__ float sm[];
  float* q = sm;       // [hd]
  float* s = sm + hd;  // [kWinChunk]
  __shared__ float red[kThreads / 32];
  for (int d = threadIdx.x; d < hd; d += blockDim.x) q[d] = Q[((size_t)t * hq + h) * hd + d];
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  float acc[kWinDimsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
  float M = -INFINITY, L = 0.0f;
  for (int c0 = 0; c0 < n; c0 += kWinChunk) {
    const int cn = min(kWinChunk, n - c0);
    for (int i = warp; i < cn; i += nwarps) {
      const int j = c0 + i;
      const int key = j < sink_hi ? j : win_lo + (j - sink_hi);
      const float* kr = K + ((size_t)key * hkv + kvh) * hd;
      float dot = 0.0f;
      for (int d = lane; d < hd; d += 32) dot += q[d] * kr[d];
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
      if (lane == 0) s[i] = dot * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < cn; i += blockDim.x) mx = fmaxf(mx, s[i]);
    const float newM = fmaxf(M, block_reduce(mx, true, red));
    const float rescale = expf(M - newM);  // 0 on the first chunk (M = -inf)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cn; i += blockDim.x) {
      const float e = expf(s[i] - newM);
      s[i] = e;
      sum += e;
    }
    L = L * rescale + block_reduce(sum, false, red);  // block_reduce syncs: s[] is complete
    for (int r = 0; r < kWinDimsPerThread; ++r) {
      const int d = threadIdx.x + r * blockDim.x;
      if (d >= hd) break;
      float a = 0.0f;
      for (int i = 0; i < cn; ++i) {
        const int j = c0 + i;
        const int key = j < sink_hi ? j : win_lo + (j - sink_hi);
        a += s[i] * V[((size_t)key * hkv + kvh) * hd + d];
      }
      acc[r] = acc[r] * rescale + a;
    }
    M = newM;
    __syncthreads();  // s[] is rewritten by the next chunk
  }
  const float inv = 1.0f / (L + 1e-9f);  // softmax_vector_cpu's normaliser
  for (int r = 0; r < kWinDimsPerThread; ++r) {
    const int d = threadIdx.x + r * blockDim.x;
    if (d >= hd) break;
    out[((size_t)t * hq + h) * hd + d] = acc[r] * inv;
  }
}

}  // namespace

void ir_window_attention_cuda(const float* q, const float* K, const float* V, float* out, int T, int start_pos,
                              int hq, int hkv, int hd, float scale, int window, int sinks, cudaStream_t stream) {
  if (window <= 0 || sinks < 0 || hkv <= 0 || hq % hkv != 0 || hd > kThreads * kWinDimsPerThread)
    throw std::runtime_error("IR CUDA window attention: unsupported shape (window " + std::to_string(window) +
                             ", head_dim " + std::to_string(hd) + ")");
  const size_t shm = (size_t)(hd + kWinChunk) * sizeof(float);
  window_attention_kernel<<<dim3(T, hq), kThreads, shm, stream>>>(q, K, V, out, hq, hkv, hd, start_pos, scale, window,
                                                                  sinks);
  check_launch("window attention");
}

void ir_embedding_cuda(const float* table, const int* tokens, float* out, int T, int d, cudaStream_t stream) {
  embedding_kernel<<<T, 128, 0, stream>>>(table, tokens, out, d);
  check_launch("embedding");
}

void ir_pos_embedding_cuda(const float* x, const float* P, float* out, int T, int d, int start_pos,
                           cudaStream_t stream) {
  pos_embedding_kernel<<<T, 128, 0, stream>>>(x, P, out, d, start_pos);
  check_launch("pos_embedding");
}

void ir_layernorm_cuda(const float* x, const float* w, const float* b, float* out, int T, int d, float eps,
                       cudaStream_t stream) {
  layernorm_kernel<<<T, kThreads, 0, stream>>>(x, w, b, out, d, eps);
  check_launch("layernorm");
}

void ir_bias_add_cuda(float* y, const float* b, int T, int O, cudaStream_t stream) {
  const size_t n = (size_t)T * O;
  bias_add_kernel<<<blocks_for(n), kThreads, 0, stream>>>(y, b, n, O);
  check_launch("bias");
}

void ir_binary_cuda(const float* a, const float* b, float* out, size_t n, int op, cudaStream_t stream) {
  binary_kernel<<<blocks_for(n), kThreads, 0, stream>>>(a, b, out, n, op);
  check_launch(op == 0 ? "add" : "mul");
}

void ir_silu_cuda(const float* x, float* out, size_t n, cudaStream_t stream) {
  silu_kernel<<<blocks_for(n), kThreads, 0, stream>>>(x, out, n);
  check_launch("silu");
}

void ir_gelu_cuda(const float* x, float* out, size_t n, bool tanh_form, cudaStream_t stream) {
  gelu_kernel<<<blocks_for(n), kThreads, 0, stream>>>(x, out, n, tanh_form ? 1 : 0);
  check_launch("gelu");
}

void ir_rope_cuda(float* q, float* k, int T, int hq, int hk, int hd, int start_pos, const float* freqs,
                  cudaStream_t stream) {
  rope_kernel<<<T, 128, 0, stream>>>(q, k, hq, hk, hd, start_pos, freqs);
  check_launch("rope");
}

}  // namespace ember_ir
