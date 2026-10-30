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
                                     int d, int start_pos, const int* __restrict__ dpos) {
  const int t = blockIdx.x;
  if (dpos) start_pos = *dpos;
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
  if (i < n) out[i] = op == 0 ? __fadd_rn(a[i], b[i]) : op == 1 ? __fmul_rn(a[i], b[i]) : __fsub_rn(a[i], b[i]);
}

// op 0 relu, 1 sigmoid, 2 affine.
__global__ void unary_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n, int op, float scale,
                             float shift) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  if (op == 0)
    out[i] = 0.0f < v ? v : 0.0f;  // std::max(0.0f, v)
  else if (op == 1)
    out[i] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-v)));
  else
    out[i] = __fadd_rn(__fmul_rn(v, scale), shift);
}

__global__ void mul_col_kernel(const float* __restrict__ x, const float* __restrict__ c, float* __restrict__ out,
                               size_t n, int w) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __fmul_rn(x[i], c[i / w]);
}

__global__ void scale_kernel(const float* __restrict__ x, const float* __restrict__ g, const float* __restrict__ b,
                             float* __restrict__ out, size_t n, int w) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int j = (int)(i % w);
  const float p = __fmul_rn(x[i], g[j]);
  out[i] = b ? __fadd_rn(p, b[j]) : p;
}

// One thread per row: the CPU's ascending sum.
__global__ void rms_row_kernel(const float* __restrict__ x, float* __restrict__ out, int T, int w, float eps) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* r = x + (size_t)t * w;
  float ss = 0.0f;
  for (int j = 0; j < w; ++j) ss = __fadd_rn(ss, __fmul_rn(r[j], r[j]));
  out[t] = __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)w), eps));
}

struct DepthSources {
  const float* p[kIrDepthSourcesMax];
};

// One block per row. Thread i forms source i's logit (unit-weight rmsnorm,
// then the dot with w, both in ascending j); thread 0 the softmax in
// ascending i; then the threads over j the mixture in ascending i.
__global__ void depth_attend_kernel(DepthSources src, int S, const float* __restrict__ w, float* __restrict__ out,
                                    int d, float eps) {
  __shared__ float a[kIrDepthSourcesMax];
  const size_t row = (size_t)blockIdx.x * d;
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    const float* s = src.p[i] + row;
    float ss = 0.0f;
    for (int j = 0; j < d; ++j) ss = __fadd_rn(ss, __fmul_rn(s[j], s[j]));
    ss = __fdiv_rn(ss, (float)d);
    const float r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, eps)));
    float l = 0.0f;
    for (int j = 0; j < d; ++j) l = __fadd_rn(l, __fmul_rn(__fmul_rn(s[j], r), w[j]));
    a[i] = l;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float mx = -INFINITY;
    for (int i = 0; i < S; ++i) mx = mx < a[i] ? a[i] : mx;  // std::max(mx, a[i])
    float z = 0.0f;
    for (int i = 0; i < S; ++i) {
      a[i] = expf(__fsub_rn(a[i], mx));
      z = __fadd_rn(z, a[i]);
    }
    for (int i = 0; i < S; ++i) a[i] = __fdiv_rn(a[i], z);
  }
  __syncthreads();
  for (int j = threadIdx.x; j < d; j += blockDim.x) {
    float o = __fmul_rn(src.p[0][row + j], a[0]);
    for (int i = 1; i < S; ++i) o = __fadd_rn(o, __fmul_rn(src.p[i][row + j], a[i]));
    out[row + j] = o;
  }
}

// elu(x) + 1 as graph_ir.cpp elu_plus_one: x + 1 for x > 0, else
// (exp(x) - 1) + 1, each step rounded.
__device__ float elu_plus_one(float x) {
  float r = x;
  if (r <= 0.f) r = __fsub_rn(expf(r), 1.f);
  return __fadd_rn(r, 1.f);
}

// One thread per channel, rows in order (linear_attention_row_cpu per row).
__global__ void linear_attention_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                        const float* __restrict__ v, float* __restrict__ num,
                                        float* __restrict__ den, float* __restrict__ out, int T, int w,
                                        float den_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= w) return;
  float nm = num[c], dn = den[c];
  for (int t = 0; t < T; ++t) {
    const size_t i = (size_t)t * w + c;
    const float pk = elu_plus_one(k[i]);
    nm = __fadd_rn(nm, __fmul_rn(pk, v[i]));
    dn = __fadd_rn(dn, pk);
    out[i] = __fdiv_rn(__fmul_rn(elu_plus_one(q[i]), nm), dn < den_floor ? den_floor : dn);  // std::max
  }
  num[c] = nm;
  den[c] = dn;
}

// One block; s and the next state in shared memory (ssm_scan_row_cpu per row).
__global__ void ssm_scan_kernel(const float* __restrict__ u, const float* __restrict__ A, const float* __restrict__ B,
                                const float* __restrict__ C, const float* __restrict__ D, float* __restrict__ s_io,
                                float* __restrict__ y, int T, int n) {
  extern __shared__ float sm[];
  float* s = sm;       // [n]
  float* ns = sm + n;  // [n]
  for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = s_io[i];
  const float dd = D[0];
  __syncthreads();
  for (int t = 0; t < T; ++t) {
    const float* ut = u + (size_t)t * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const float* a = A + (size_t)i * n;
      float acc = 0.0f;
      for (int j = 0; j < n; ++j) acc = __fadd_rn(acc, __fmul_rn(a[j], s[j]));
      ns[i] = __fadd_rn(acc, __fmul_rn(B[i], ut[i]));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      s[i] = ns[i];
      y[(size_t)t * n + i] = __fadd_rn(__fmul_rn(C[i], ns[i]), __fmul_rn(dd, ut[i]));
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < n; i += blockDim.x) s_io[i] = s[i];
}

__global__ void kv_write_kernel(float* __restrict__ K, float* __restrict__ V, const float* __restrict__ k,
                                const float* __restrict__ v, size_t n, int row, int start_pos,
                                const int* __restrict__ dpos) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const size_t off = (size_t)(dpos ? *dpos : start_pos) * row + i;
  K[off] = k[i];
  V[off] = v[i];
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
                            const float* __restrict__ freqs, const int* __restrict__ dpos) {
  const int t = blockIdx.x, half = hd / 2;
  if (dpos) start_pos = *dpos;
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

// One block per token, threads over (head, pair) (rope_rows_cpu).
__global__ void rope_rows_kernel(float* __restrict__ x, int heads, int hd, int n_rot, int neox, int start_pos,
                                 const float* __restrict__ freqs, const int* __restrict__ dpos) {
  const int t = blockIdx.x, half = n_rot / 2;
  if (dpos) start_pos = *dpos;
  const float* f = freqs + (size_t)(start_pos + t) * half * 2;
  for (int j = threadIdx.x; j < heads * half; j += blockDim.x) {
    const int h = j / half, i = j % half;
    float* r = x + ((size_t)t * heads + h) * hd;
    const int i0 = neox ? i : 2 * i, i1 = neox ? i + half : 2 * i + 1;
    const float c = f[2 * i], s = f[2 * i + 1];
    const float v0 = r[i0], v1 = r[i1];
    r[i0] = __fsub_rn(__fmul_rn(v0, c), __fmul_rn(v1, s));
    r[i1] = __fadd_rn(__fmul_rn(v0, s), __fmul_rn(v1, c));
  }
}

__global__ void conv1d_causal_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                     const float* __restrict__ bias, float* __restrict__ state, float* __restrict__ out,
                                     int T, int C, int K) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  for (int t = 0; t < T; ++t) {
    const float xc = x[(size_t)t * C + c];
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      const float v = k < K - 1 ? state[(size_t)k * C + c] : xc;
      acc = __fadd_rn(acc, __fmul_rn(w[(size_t)c * K + k], v));
    }
    out[(size_t)t * C + c] = bias ? __fadd_rn(acc, bias[c]) : acc;
    if (K > 1) {  // the channel's state becomes the last K - 1 entries of [state; x]
      for (int k = 0; k + 2 < K; ++k) state[(size_t)k * C + c] = state[(size_t)(k + 1) * C + c];
      state[(size_t)(K - 2) * C + c] = xc;
    }
  }
}

__global__ void selective_scan_kernel(const float* __restrict__ x, const float* __restrict__ dt,
                                      const float* __restrict__ B, const float* __restrict__ Cm,
                                      const float* __restrict__ A, const float* __restrict__ d, float* __restrict__ s,
                                      float* __restrict__ y, int T, int D, int N, int softplus) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= D) return;
  float* sc = s + (size_t)c * N;
  const float* ac = A + (size_t)c * N;
  for (int t = 0; t < T; ++t) {
    const float dtc = dt[(size_t)t * D + c], xc = x[(size_t)t * D + c];
    const float delta = softplus ? (dtc <= 20.0f ? log1pf(expf(dtc)) : dtc) : dtc;
    const float xd = __fmul_rn(xc, delta);
    const float* bt = B + (size_t)t * N;
    const float* ct = Cm + (size_t)t * N;
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
      const float decay = __fmul_rn(sc[n], expf(__fmul_rn(delta, ac[n])));
      const float v = __fadd_rn(decay, __fmul_rn(bt[n], xd));
      sc[n] = v;
      acc = __fadd_rn(acc, __fmul_rn(v, ct[n]));
    }
    y[(size_t)t * D + c] = d ? __fadd_rn(acc, __fmul_rn(d[c], xc)) : acc;
  }
}

struct ExprCode {
  int n;
  expr::Instr code[expr::kMaxCode];
};
struct ExprPtrs {
  const float* in[expr::Expr::kMaxInputs];
  const float* col[expr::kMaxColumns];
};

// One thread per element: the stage's stack program (graph_ir_expr.cpp
// unary_op / binary_op, each primitive rounded on its own).
__global__ void expr_kernel(ExprCode code, ExprPtrs p, float* __restrict__ out, size_t n, int w) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const size_t t = i / w;
  float st[expr::kMaxStack];
  int sp = 0;
  for (int k = 0; k < code.n; ++k) {
    const expr::Instr ins = code.code[k];
    switch (ins.op) {
      case expr::Op::CONSTANT: st[sp++] = ins.c; break;
      case expr::Op::INPUT: st[sp++] = p.in[ins.idx][i]; break;
      case expr::Op::COL: st[sp++] = p.col[ins.idx][t]; break;
      case expr::Op::WIDTH: st[sp++] = (float)w; break;
      case expr::Op::NEG: st[sp - 1] = -st[sp - 1]; break;
      case expr::Op::EXP: st[sp - 1] = expf(st[sp - 1]); break;
      case expr::Op::LOG: st[sp - 1] = logf(st[sp - 1]); break;
      case expr::Op::SQRT: st[sp - 1] = __fsqrt_rn(st[sp - 1]); break;
      case expr::Op::TANH: st[sp - 1] = tanhf(st[sp - 1]); break;
      case expr::Op::SIGMOID: st[sp - 1] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-st[sp - 1]))); break;
      case expr::Op::ERF: st[sp - 1] = erff(st[sp - 1]); break;
      case expr::Op::ABS: st[sp - 1] = fabsf(st[sp - 1]); break;
      case expr::Op::RELU: st[sp - 1] = 0.0f < st[sp - 1] ? st[sp - 1] : 0.0f; break;  // std::max(0.0f, a)
      case expr::Op::WHERE: {
        const float b = st[--sp], a = st[--sp];
        st[sp - 1] = st[sp - 1] > 0.0f ? a : b;
        break;
      }
      default: {
        const float b = st[--sp], a = st[sp - 1];
        float r;
        switch (ins.op) {
          case expr::Op::ADD: r = __fadd_rn(a, b); break;
          case expr::Op::SUB: r = __fsub_rn(a, b); break;
          case expr::Op::MUL: r = __fmul_rn(a, b); break;
          case expr::Op::DIV: r = __fdiv_rn(a, b); break;
          case expr::Op::POW: r = powf(a, b); break;
          case expr::Op::MAX: r = a >= b ? a : b; break;
          default: r = a <= b ? a : b; break;  // MIN
        }
        st[sp - 1] = r;
      }
    }
  }
  out[i] = st[0];
}

// One thread per row, ascending columns: 0 + r0 + r1 + ... (rowsum), or the
// first largest entry (rowmax).
__global__ void row_reduce_kernel(const float* __restrict__ x, float* __restrict__ out, int T, int w, int is_max) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* r = x + (size_t)t * w;
  float acc;
  if (is_max) {
    acc = r[0];
    for (int j = 1; j < w; ++j) acc = r[j] > acc ? r[j] : acc;
  } else {
    acc = 0.0f;
    for (int j = 0; j < w; ++j) acc = __fadd_rn(acc, r[j]);
  }
  out[t] = acc;
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
                                        int hd, int start_pos, float scale, int window, int sinks,
                                        const int* __restrict__ dpos, float softcap) {
  const int t = blockIdx.x, h = blockIdx.y;
  const int pos = (dpos ? *dpos : start_pos) + t;
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
      if (lane == 0) {
        float sc = dot * scale;
        if (softcap > 0.0f) sc = softcap * tanhf(sc / softcap);  // IR_SPEC 5.15 softcap (revision 1.5)
        s[i] = sc;
      }
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

// Attention mask turn_state (IR_SPEC 6.8.2) over a compact kv cache, one block per (token, q head).
// codes[r] = 2 s + c for the position row r holds (s its segment, c = 1 at a state position); the
// call's token t is at row row0 + t (row0 read from *drow when given: the captured decode step), and
// sees row j <= row0 + t iff s(j) == s(t) or c(j) == 1. The rows are walked in ascending order,
// kWinChunk at a time, as window_attention_kernel walks its keys: visible rows scored into shared
// memory (a warp per row), masked rows -inf (never read), an online softmax over the chunks. A chunk
// with no visible row is skipped; the query's own row is always visible, so the last chunk has one.
__global__ void turn_state_attention_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                            const float* __restrict__ V, float* __restrict__ out, int hq, int hkv,
                                            int hd, int row0, const int* __restrict__ codes, float scale,
                                            const int* __restrict__ drow) {
  const int t = blockIdx.x, h = blockIdx.y;
  if (drow) row0 = *drow;
  const int n = row0 + t + 1;
  const int qs = codes[row0 + t] >> 1;
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
      const int code = codes[c0 + i];
      if ((code >> 1) != qs && !(code & 1)) {  // masked (uniform over the warp)
        if (lane == 0) s[i] = -INFINITY;
        continue;
      }
      const float* kr = K + ((size_t)(c0 + i) * hkv + kvh) * hd;
      float dot = 0.0f;
      for (int d = lane; d < hd; d += 32) dot += q[d] * kr[d];
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
      if (lane == 0) s[i] = dot * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < cn; i += blockDim.x) mx = fmaxf(mx, s[i]);
    const float cm = block_reduce(mx, true, red);
    if (cm == -INFINITY) {  // every row of the chunk is masked (uniform over the block)
      __syncthreads();
      continue;
    }
    const float newM = fmaxf(M, cm);
    const float rescale = expf(M - newM);  // 0 on the first visible chunk (M = -inf)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cn; i += blockDim.x) {
      const float e = expf(s[i] - newM);  // exactly 0 for a masked row
      s[i] = e;
      sum += e;
    }
    L = L * rescale + block_reduce(sum, false, red);  // block_reduce syncs: s[] is complete
    for (int r = 0; r < kWinDimsPerThread; ++r) {
      const int d = threadIdx.x + r * blockDim.x;
      if (d >= hd) break;
      float a = 0.0f;
      for (int i = 0; i < cn; ++i)
        if (s[i] != 0.0f) a += s[i] * V[((size_t)(c0 + i) * hkv + kvh) * hd + d];
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

// Row w of K and V takes row idx[w], for w = w0 .. n-1, in ascending order, in place: idx ascends and
// idx[w] >= w, so a row is read before any later step overwrites it (one block, a barrier per row).
__global__ void kv_compact_kernel(float* __restrict__ K, float* __restrict__ V, const int* __restrict__ idx, int w0,
                                  int n, int row) {
  for (int w = w0; w < n; ++w) {
    const int r = idx[w];
    if (r != w)
      for (int i = threadIdx.x; i < row; i += blockDim.x) {
        K[(size_t)w * row + i] = K[(size_t)r * row + i];
        V[(size_t)w * row + i] = V[(size_t)r * row + i];
      }
    __syncthreads();
  }
}

// out[t] = table[id] for a text id, state[id - vocab] for a state id (IR_SPEC 6.8.1). EXACT (a copy).
__global__ void embedding_state_kernel(const float* __restrict__ table, const float* __restrict__ state, int vocab,
                                       const int* __restrict__ tokens, float* __restrict__ out, int d) {
  const int t = blockIdx.x;
  const int id = tokens[t];
  const float* src = id >= vocab ? state + (size_t)(id - vocab) * d : table + (size_t)id * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x) out[(size_t)t * d + i] = src[i];
}

// Mixture-of-Recursions (graph_ir_cuda.h): one block per compact row.
__global__ void rows_gather_kernel(const float* __restrict__ src, const int* __restrict__ idx, float* __restrict__ dst,
                                   int w) {
  const int a = blockIdx.x;
  const float* s = src + (size_t)idx[a] * w;
  for (int i = threadIdx.x; i < w; i += blockDim.x) dst[(size_t)a * w + i] = s[i];
}

__global__ void rows_scatter_kernel(const float* __restrict__ src, const int* __restrict__ idx, float* __restrict__ dst,
                                    int w) {
  const int a = blockIdx.x;
  float* d = dst + (size_t)idx[a] * w;
  for (int i = threadIdx.x; i < w; i += blockDim.x) d[i] = src[(size_t)a * w + i];
}

__global__ void mor_gate_kernel(const float* __restrict__ x_in, float* __restrict__ x, const float* __restrict__ g,
                                int w) {
  const int a = blockIdx.x;
  const float ga = g[a];
  for (int i = threadIdx.x; i < w; i += blockDim.x) {
    const size_t j = (size_t)a * w + i;
    const float diff = __fsub_rn(x[j], x_in[j]);
    const float p = __fmul_rn(ga, diff);
    x[j] = __fadd_rn(x_in[j], p);
  }
}

// rope_kernel and rope_rows_kernel with row t at position pos[t].
__global__ void rope_at_kernel(float* __restrict__ q, float* __restrict__ k, int hq, int hk, int hd,
                               const int* __restrict__ pos, const float* __restrict__ freqs) {
  const int t = blockIdx.x, half = hd / 2;
  const float* f = freqs + (size_t)pos[t] * half * 2;
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

__global__ void rope_rows_at_kernel(float* __restrict__ x, int heads, int hd, int n_rot, int neox,
                                    const int* __restrict__ pos, const float* __restrict__ freqs) {
  const int t = blockIdx.x, half = n_rot / 2;
  const float* f = freqs + (size_t)pos[t] * half * 2;
  for (int j = threadIdx.x; j < heads * half; j += blockDim.x) {
    const int h = j / half, i = j % half;
    float* r = x + ((size_t)t * heads + h) * hd;
    const int i0 = neox ? i : 2 * i, i1 = neox ? i + half : 2 * i + 1;
    const float c = f[2 * i], s = f[2 * i + 1];
    const float v0 = r[i0], v1 = r[i1];
    r[i0] = __fsub_rn(__fmul_rn(v0, c), __fmul_rn(v1, s));
    r[i1] = __fadd_rn(__fmul_rn(v0, s), __fmul_rn(v1, c));
  }
}

}  // namespace

void ir_rows_gather_cuda(const float* src, const int* idx, float* dst, int n, int w, cudaStream_t stream) {
  if (n <= 0) return;
  rows_gather_kernel<<<n, 128, 0, stream>>>(src, idx, dst, w);
  check_launch("rows gather");
}

void ir_rows_scatter_cuda(const float* src, const int* idx, float* dst, int n, int w, cudaStream_t stream) {
  if (n <= 0) return;
  rows_scatter_kernel<<<n, 128, 0, stream>>>(src, idx, dst, w);
  check_launch("rows scatter");
}

void ir_mor_gate_cuda(const float* x_in, float* x, const float* g, int n, int w, cudaStream_t stream) {
  if (n <= 0) return;
  mor_gate_kernel<<<n, 128, 0, stream>>>(x_in, x, g, w);
  check_launch("mor gate");
}

void ir_rope_at_cuda(float* q, float* k, int T, int hq, int hk, int hd, const int* pos, const float* freqs,
                     cudaStream_t stream) {
  rope_at_kernel<<<T, 128, 0, stream>>>(q, k, hq, hk, hd, pos, freqs);
  check_launch("rope (positions)");
}

void ir_rope_rows_at_cuda(float* x, int T, int heads, int hd, int n_rot, bool neox, const int* pos, const float* freqs,
                          cudaStream_t stream) {
  if (n_rot <= 0 || n_rot % 2 != 0 || n_rot > hd)
    throw std::runtime_error("IR CUDA rope: n_rot " + std::to_string(n_rot) + " / head_dim " + std::to_string(hd));
  rope_rows_at_kernel<<<T, 128, 0, stream>>>(x, heads, hd, n_rot, neox ? 1 : 0, pos, freqs);
  check_launch("rope (rows, positions)");
}

void ir_turn_state_attention_cuda(const float* q, const float* K, const float* V, float* out, int T, int row0,
                                  const int* codes, int hq, int hkv, int hd, float scale, cudaStream_t stream,
                                  const int* drow) {
  if (hkv <= 0 || hq % hkv != 0 || hd > kThreads * kWinDimsPerThread)
    throw std::runtime_error("IR CUDA turn_state attention: unsupported shape (head_dim " + std::to_string(hd) + ")");
  const size_t shm = (size_t)(hd + kWinChunk) * sizeof(float);
  turn_state_attention_kernel<<<dim3(T, hq), kThreads, shm, stream>>>(q, K, V, out, hq, hkv, hd, row0, codes, scale,
                                                                      drow);
  check_launch("turn_state attention");
}

void ir_kv_compact_cuda(float* K, float* V, const int* idx, int w0, int n, int row, cudaStream_t stream) {
  if (w0 >= n) return;
  kv_compact_kernel<<<1, kThreads, 0, stream>>>(K, V, idx, w0, n, row);
  check_launch("kv compact");
}

void ir_embedding_state_cuda(const float* table, const float* state, int vocab, const int* tokens, float* out, int T,
                             int d, cudaStream_t stream) {
  embedding_state_kernel<<<T, 128, 0, stream>>>(table, state, vocab, tokens, out, d);
  check_launch("embedding (state rows)");
}

void ir_window_attention_cuda(const float* q, const float* K, const float* V, float* out, int T, int start_pos,
                              int hq, int hkv, int hd, float scale, int window, int sinks, cudaStream_t stream,
                              const int* dpos, float softcap) {
  if (window <= 0 || sinks < 0 || hkv <= 0 || hq % hkv != 0 || hd > kThreads * kWinDimsPerThread)
    throw std::runtime_error("IR CUDA window attention: unsupported shape (window " + std::to_string(window) +
                             ", head_dim " + std::to_string(hd) + ")");
  const size_t shm = (size_t)(hd + kWinChunk) * sizeof(float);
  window_attention_kernel<<<dim3(T, hq), kThreads, shm, stream>>>(q, K, V, out, hq, hkv, hd, start_pos, scale, window,
                                                                  sinks, dpos, softcap);
  check_launch("window attention");
}

void ir_kv_write_cuda(float* K, float* V, const float* k, const float* v, int T, int row, int start_pos,
                      cudaStream_t stream, const int* dpos) {
  const size_t n = (size_t)T * row;
  kv_write_kernel<<<blocks_for(n), kThreads, 0, stream>>>(K, V, k, v, n, row, start_pos, dpos);
  check_launch("kv write");
}

void ir_relu_cuda(const float* x, float* out, size_t n, cudaStream_t stream) {
  unary_kernel<<<blocks_for(n), kThreads, 0, stream>>>(x, out, n, 0, 0.0f, 0.0f);
  check_launch("relu");
}

void ir_sigmoid_cuda(const float* x, float* out, size_t n, cudaStream_t stream) {
  unary_kernel<<<blocks_for(n), kThreads, 0, stream>>>(x, out, n, 1, 0.0f, 0.0f);
  check_launch("sigmoid");
}

void ir_affine_cuda(const float* x, float* out, size_t n, float scale, float shift, cudaStream_t stream) {
  unary_kernel<<<blocks_for(n), kThreads, 0, stream>>>(x, out, n, 2, scale, shift);
  check_launch("affine");
}

void ir_mul_col_cuda(const float* x, const float* c, float* out, int T, int w, cudaStream_t stream) {
  const size_t n = (size_t)T * w;
  mul_col_kernel<<<blocks_for(n), kThreads, 0, stream>>>(x, c, out, n, w);
  check_launch("mul_col");
}

void ir_scale_cuda(const float* x, const float* g, const float* b, float* out, int T, int w, cudaStream_t stream) {
  const size_t n = (size_t)T * w;
  scale_kernel<<<blocks_for(n), kThreads, 0, stream>>>(x, g, b, out, n, w);
  check_launch("scale");
}

void ir_rms_row_cuda(const float* x, float* out, int T, int w, float eps, cudaStream_t stream) {
  rms_row_kernel<<<(T + 63) / 64, 64, 0, stream>>>(x, out, T, w, eps);
  check_launch("rms_row");
}

void ir_depth_attend_cuda(const float* const* sources, int S, const float* w, float* out, int T, int d, float eps,
                          cudaStream_t stream) {
  if (S < 2 || S > kIrDepthSourcesMax)
    throw std::runtime_error("IR CUDA depth_attend: " + std::to_string(S) + " sources (2.." +
                             std::to_string(kIrDepthSourcesMax) + " supported)");
  DepthSources src{};
  for (int i = 0; i < S; ++i) src.p[i] = sources[i];
  depth_attend_kernel<<<T, 128, 0, stream>>>(src, S, w, out, d, eps);
  check_launch("depth_attend");
}

void ir_linear_attention_cuda(const float* q, const float* k, const float* v, float* num, float* den, float* out,
                              int T, int width, float den_floor, cudaStream_t stream) {
  linear_attention_kernel<<<(width + 127) / 128, 128, 0, stream>>>(q, k, v, num, den, out, T, width, den_floor);
  check_launch("linear_attention");
}

void ir_ssm_scan_cuda(const float* u, const float* A, const float* B, const float* C, const float* D, float* s,
                      float* y, int T, int n, cudaStream_t stream) {
  if (n <= 0 || n > kIrSsmMax)
    throw std::runtime_error("IR CUDA ssm_scan: n " + std::to_string(n) + " (at most " + std::to_string(kIrSsmMax) +
                             ")");
  const int threads = n < kThreads ? (n + 31) / 32 * 32 : kThreads;
  ssm_scan_kernel<<<1, threads, (size_t)2 * n * sizeof(float), stream>>>(u, A, B, C, D, s, y, T, n);
  check_launch("ssm_scan");
}

void ir_embedding_cuda(const float* table, const int* tokens, float* out, int T, int d, cudaStream_t stream) {
  embedding_kernel<<<T, 128, 0, stream>>>(table, tokens, out, d);
  check_launch("embedding");
}

void ir_pos_embedding_cuda(const float* x, const float* P, float* out, int T, int d, int start_pos,
                           cudaStream_t stream, const int* dpos) {
  pos_embedding_kernel<<<T, 128, 0, stream>>>(x, P, out, d, start_pos, dpos);
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
  check_launch(op == 0 ? "add" : op == 1 ? "mul" : "sub");
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
                  cudaStream_t stream, const int* dpos) {
  rope_kernel<<<T, 128, 0, stream>>>(q, k, hq, hk, hd, start_pos, freqs, dpos);
  check_launch("rope");
}

void ir_rope_rows_cuda(float* x, int T, int heads, int hd, int n_rot, bool neox, int start_pos, const float* freqs,
                       cudaStream_t stream, const int* dpos) {
  if (n_rot <= 0 || n_rot % 2 != 0 || n_rot > hd)
    throw std::runtime_error("IR CUDA rope: n_rot " + std::to_string(n_rot) + " / head_dim " + std::to_string(hd));
  rope_rows_kernel<<<T, 128, 0, stream>>>(x, heads, hd, n_rot, neox ? 1 : 0, start_pos, freqs, dpos);
  check_launch("rope (rows)");
}

void ir_conv1d_causal_cuda(const float* x, const float* w, const float* bias, float* state, float* out, int T, int C,
                           int K, cudaStream_t stream) {
  if (K < 1 || C <= 0) throw std::runtime_error("IR CUDA conv1d_causal: kernel " + std::to_string(K));
  conv1d_causal_kernel<<<(C + 127) / 128, 128, 0, stream>>>(x, w, bias, state, out, T, C, K);
  check_launch("conv1d_causal");
}

void ir_selective_scan_cuda(const float* x, const float* dt, const float* B, const float* C, const float* A,
                            const float* d, float* s, float* y, int T, int D, int N, bool softplus,
                            cudaStream_t stream) {
  selective_scan_kernel<<<(D + 127) / 128, 128, 0, stream>>>(x, dt, B, C, A, d, s, y, T, D, N, softplus ? 1 : 0);
  check_launch("selective_scan");
}

size_t ir_expr_scratch_floats(const std::vector<expr::Stage>& stages, int T, int w) {
  if (stages.size() <= 1) return 0;
  return (size_t)T * w + (stages.size() - 1) * (size_t)T;
}

void ir_expr_cuda(const std::vector<expr::Stage>& stages, const float* const* ins, int n_ins, float* scratch,
                  float* out, int T, int w, cudaStream_t stream) {
  if (n_ins <= 0 || n_ins > (int)expr::Expr::kMaxInputs || stages.empty() || stages.size() - 1 > expr::kMaxColumns)
    throw std::runtime_error("IR CUDA expr: " + std::to_string(n_ins) + " inputs, " + std::to_string(stages.size()) +
                             " stages");
  ExprPtrs p{};
  for (int k = 0; k < n_ins; ++k) p.in[k] = ins[k];
  float* full = scratch;  // a reduction's argument, [T, w], reused stage by stage
  for (size_t k = 0; k + 1 < stages.size(); ++k) p.col[k] = scratch + (size_t)T * w + k * (size_t)T;
  const size_t n = (size_t)T * w;
  for (size_t k = 0; k < stages.size(); ++k) {
    const expr::Stage& s = stages[k];
    if (s.code.empty() || s.code.size() > expr::kMaxCode) throw std::runtime_error("IR CUDA expr: a stage's code");
    ExprCode code{};
    code.n = (int)s.code.size();
    for (size_t i = 0; i < s.code.size(); ++i) code.code[i] = s.code[i];
    const bool last = k + 1 == stages.size();
    expr_kernel<<<blocks_for(n), kThreads, 0, stream>>>(code, p, last ? out : full, n, w);
    check_launch("expr");
    if (last) break;
    row_reduce_kernel<<<(T + 63) / 64, 64, 0, stream>>>(full, const_cast<float*>(p.col[k]), T, w, s.reduce == 2);
    check_launch("expr row reduction");
  }
}

namespace {

__global__ void mla_kv_write_kernel(float* __restrict__ KV, const float* __restrict__ c, const float* __restrict__ kr,
                                    int T, int r, int dr, int start_pos, const int* __restrict__ dpos) {
  if (dpos) start_pos = *dpos;
  const int w = r + dr;
  const size_t n = (size_t)T * w;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const int t = (int)(i / w), e = (int)(i % w);
    KV[((size_t)start_pos + t) * w + e] = e < r ? c[(size_t)t * r + e] : kr[(size_t)t * dr + (e - r)];
  }
}

// One block per (row t, head h). Shared: qa [r], qr_h [dr], u [r], the chunk's scores [kMlaChunk].
__global__ void mla_attention_kernel(const float* __restrict__ qn, const float* __restrict__ qr,
                                     const float* __restrict__ KV, const float* __restrict__ Kb,
                                     const float* __restrict__ Vb, float* __restrict__ out, int H, int r, int dn, int dr,
                                     int dv, float scale, int window, int sinks, int start_pos,
                                     const int* __restrict__ dpos) {
  extern __shared__ float sm[];
  float* qa = sm;
  float* q2 = qa + r;
  float* u = q2 + dr;
  float* sc = u + r;
  __shared__ float m_s, z_s, alpha_s;
  const int t = blockIdx.x / H, h = blockIdx.x % H;
  if (dpos) start_pos = *dpos;
  const int p = start_pos + t;
  const int w = r + dr;
  const float* K = Kb + (size_t)h * r * dn;
  const float* V = Vb + (size_t)h * dv * r;
  const float* qnh = qn + ((size_t)t * H + h) * dn;
  const float* qrh = qr + ((size_t)t * H + h) * dr;
  for (int i = threadIdx.x; i < r; i += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < dn; ++k) acc = __fadd_rn(acc, __fmul_rn(K[(size_t)i * dn + k], qnh[k]));
    qa[i] = acc;
    u[i] = 0.0f;
  }
  for (int e = threadIdx.x; e < dr; e += blockDim.x) q2[e] = qrh[e];
  // section 5.15: [0, sink_hi) then [win_lo, p] (cpu_attention.cpp swa_visible_ranges)
  int win_lo = 0, sink_hi = 0;
  if (window > 0) {
    win_lo = p + 1 > window ? p + 1 - window : 0;
    sink_hi = min(min(sinks, p + 1), win_lo);
  }
  const int nvis = sink_hi + (p + 1 - win_lo);
  const bool one_chunk = nvis <= kMlaChunk;
  if (threadIdx.x == 0) {
    m_s = -INFINITY;
    z_s = 0.0f;
  }
  __syncthreads();
  for (int c0 = 0; c0 < nvis; c0 += kMlaChunk) {
    const int cn = min(kMlaChunk, nvis - c0);
    for (int j = threadIdx.x; j < cn; j += blockDim.x) {
      const int v = c0 + j;
      const int key = v < sink_hi ? v : win_lo + (v - sink_hi);
      const float* row = KV + (size_t)key * w;
      float a = 0.0f, b = 0.0f;
      for (int i = 0; i < r; ++i) a = __fadd_rn(a, __fmul_rn(qa[i], row[i]));
      for (int e = 0; e < dr; ++e) b = __fadd_rn(b, __fmul_rn(q2[e], row[r + e]));
      sc[j] = __fmul_rn(__fadd_rn(a, b), scale);
    }
    __syncthreads();
    if (threadIdx.x == 0) {  // max, then e_j and z in ascending j (the CPU's order within a chunk)
      float cm = m_s;
      for (int j = 0; j < cn; ++j) cm = fmaxf(cm, sc[j]);
      const float alpha = m_s == -INFINITY ? 0.0f : expf(__fsub_rn(m_s, cm));
      float zc = 0.0f;
      for (int j = 0; j < cn; ++j) {
        sc[j] = expf(__fsub_rn(sc[j], cm));
        zc = __fadd_rn(zc, sc[j]);
      }
      z_s = __fadd_rn(__fmul_rn(z_s, alpha), zc);
      m_s = cm;
      alpha_s = alpha;
    }
    __syncthreads();
    const float alpha = alpha_s, z = z_s;
    for (int i = threadIdx.x; i < r; i += blockDim.x) {
      float acc = one_chunk ? 0.0f : __fmul_rn(u[i], alpha);
      for (int j = 0; j < cn; ++j) {
        const int v = c0 + j;
        const int key = v < sink_hi ? v : win_lo + (v - sink_hi);
        const float a = one_chunk ? __fdiv_rn(sc[j], z) : sc[j];
        acc = __fadd_rn(acc, __fmul_rn(a, KV[(size_t)key * w + i]));
      }
      u[i] = acc;
    }
    __syncthreads();
  }
  if (!one_chunk) {
    for (int i = threadIdx.x; i < r; i += blockDim.x) u[i] = __fdiv_rn(u[i], z_s);
    __syncthreads();
  }
  float* o = out + ((size_t)t * H + h) * dv;
  for (int k = threadIdx.x; k < dv; k += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < r; ++i) acc = __fadd_rn(acc, __fmul_rn(V[(size_t)k * r + i], u[i]));
    o[k] = acc;
  }
}

}  // namespace

void ir_mla_kv_write_cuda(float* KV, const float* c, const float* kr, int T, int r, int dr, int start_pos,
                          cudaStream_t stream, const int* dpos) {
  const size_t n = (size_t)T * (r + dr);
  mla_kv_write_kernel<<<blocks_for(n), kThreads, 0, stream>>>(KV, c, kr, T, r, dr, start_pos, dpos);
  check_launch("mla kv write");
}

void ir_mla_attention_cuda(const float* qn, const float* qr, const float* KV, const float* Kb, const float* Vb,
                           float* out, int T, int start_pos, int H, int r, int dn, int dr, int dv, float scale,
                           int window, int sinks, cudaStream_t stream, const int* dpos) {
  const size_t smem = (static_cast<size_t>(r) * 2 + dr + kMlaChunk) * sizeof(float);
  if (smem > 48 * 1024)
    throw std::runtime_error("IR mla_attention on CUDA: kv_rank " + std::to_string(r) + " needs " + std::to_string(smem) +
                             " bytes of shared memory (at most 48 KiB)");
  mla_attention_kernel<<<static_cast<unsigned>(T) * H, 128, smem, stream>>>(qn, qr, KV, Kb, Vb, out, H, r, dn, dr, dv,
                                                                           scale, window, sinks, start_pos, dpos);
  check_launch("mla attention");
}

}  // namespace ember_ir
