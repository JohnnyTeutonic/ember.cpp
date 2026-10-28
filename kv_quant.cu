// Quantised KV cache kernels (kv_quant.h; EMBER_KV=f16|q8|q4). The default f32 cache never reaches
// this file: graph_ir.cpp calls it only when EMBER_KV names another format.
//
// The decode kernel is decode_attn_tiled.cuh's tiled split-K kernel with its loads replaced: the same
// chunking (32-key chunks below 2,048 keys, at most 64 chunks per head), lane layout and fixed-order
// reductions, and its merge (tiled_merge_block) unchanged, so a format's result is the same run to run
// and depends only on L. Each float4 slice a lane loads is dequantised from the codes as it is loaded.
#include "kv_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "decode_attn_tiled.cuh"

namespace ember_kvq {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("kv_quant ") + what + ": " + cudaGetErrorString(e));
}

// ---------------------------------------------------------------------------
// Writes. q8/q4: one warp per block of 32 elements (the warp's lanes its elements); f16: one thread per element.

template <int F>
__global__ void write_block_kernel(uint8_t* __restrict__ Kd, uint8_t* __restrict__ Ks, uint8_t* __restrict__ Vd,
                                   uint8_t* __restrict__ Vs, const float* __restrict__ k, const float* __restrict__ v,
                                   int T, int row, int start_pos, const int* __restrict__ dpos) {
  const int nb = row / kBlock;
  const long long per = static_cast<long long>(T) * nb;
  const long long gw = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (gw >= 2 * per) return;  // the whole warp: gw is the warp's
  const bool isv = gw >= per;
  const long long w = isv ? gw - per : gw;
  const int t = static_cast<int>(w / nb), b = static_cast<int>(w - static_cast<long long>(t) * nb);
  const size_t pos = static_cast<size_t>((dpos ? *dpos : start_pos) + t);
  const float x = (isv ? v : k)[static_cast<size_t>(t) * row + static_cast<size_t>(b) * kBlock + lane];
  uint8_t* D = isv ? Vd : Kd;
  uint8_t* S = isv ? Vs : Ks;
  if (F == kQ8) {
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const __half dh = __float2half_rn(__fdiv_rn(a, 127.0f));
    const float d = __half2float(dh);
    float q = d != 0.0f ? rintf(__fdiv_rn(x, d)) : 0.0f;
    q = fminf(127.0f, fmaxf(-127.0f, q));
    reinterpret_cast<int8_t*>(D)[pos * row + static_cast<size_t>(b) * kBlock + lane] = static_cast<int8_t>(q);
    if (lane == 0) reinterpret_cast<__half*>(S)[pos * nb + b] = dh;
  } else {  // kQ4
    float mn = x, mx = x;
    for (int o = 16; o > 0; o >>= 1) {
      mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    const __half dh = __float2half_rn(__fdiv_rn(__fsub_rn(mx, mn), 15.0f));
    const __half mh = __float2half_rn(mn);
    const float d = __half2float(dh), m = __half2float(mh);
    float q = d != 0.0f ? rintf(__fdiv_rn(__fsub_rn(x, m), d)) : 0.0f;
    q = fminf(15.0f, fmaxf(0.0f, q));
    const int qi = static_cast<int>(q);
    const int hi = __shfl_down_sync(0xffffffffu, qi, 1);
    if ((lane & 1) == 0)
      D[(pos * row + static_cast<size_t>(b) * kBlock + lane) / 2] = static_cast<uint8_t>(qi | (hi << 4));
    if (lane == 0) reinterpret_cast<__half2*>(S)[pos * nb + b] = __halves2half2(dh, mh);
  }
}

__global__ void write_f16_kernel(__half* __restrict__ Kd, __half* __restrict__ Vd, const float* __restrict__ k,
                                 const float* __restrict__ v, int T, int row, int start_pos,
                                 const int* __restrict__ dpos) {
  const size_t n = static_cast<size_t>(T) * row;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= 2 * n) return;
  const bool isv = i >= n;
  const size_t e = isv ? i - n : i;
  const size_t at = static_cast<size_t>(dpos ? *dpos : start_pos) * row + e;
  (isv ? Vd : Kd)[at] = __float2half_rn((isv ? v : k)[e]);
}

// ---------------------------------------------------------------------------
// Loads: elements e .. e + 3 of cache row pos (e % 4 == 0, inside one head), dequantised.

template <int F>
__device__ __forceinline__ float4 load4(const uint8_t* __restrict__ D, const uint8_t* __restrict__ S, size_t pos,
                                        int row, int e) {
  const size_t at = pos * row + e;
  if (F == kF16) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(D + at * 2));
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
  } else if (F == kQ8) {
    const char4 c = __ldg(reinterpret_cast<const char4*>(D + at));
    const float d = __half2float(
        __ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(S) + pos * (row / kBlock) + e / kBlock)));
    return make_float4(__fmul_rn(static_cast<float>(c.x), d), __fmul_rn(static_cast<float>(c.y), d),
                       __fmul_rn(static_cast<float>(c.z), d), __fmul_rn(static_cast<float>(c.w), d));
  } else {  // kQ4
    const unsigned u = __ldg(reinterpret_cast<const unsigned short*>(D + at / 2));
    const unsigned dm = __ldg(reinterpret_cast<const unsigned*>(S) + pos * (row / kBlock) + e / kBlock);
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(dm & 0xffffu)));
    const float m = __half2float(__ushort_as_half(static_cast<unsigned short>(dm >> 16)));
    return make_float4(__fmaf_rn(static_cast<float>(u & 15u), d, m), __fmaf_rn(static_cast<float>((u >> 4) & 15u), d, m),
                       __fmaf_rn(static_cast<float>((u >> 8) & 15u), d, m),
                       __fmaf_rn(static_cast<float>((u >> 12) & 15u), d, m));
  }
}

// ---------------------------------------------------------------------------
// Decode: ember_tiled::tiled_split_block with load4 in place of the float4 cache loads.

constexpr int kMin = ember_tiled::kTiledMinChunk, kMax = ember_tiled::kTiledMaxChunks;
constexpr int kThreads = ember_tiled::kTiledThreads;

template <int F>
__global__ void decode_split_kernel(const float* __restrict__ Q, const uint8_t* __restrict__ Kd,
                                    const uint8_t* __restrict__ Ks, const uint8_t* __restrict__ Vd,
                                    const uint8_t* __restrict__ Vs, int seq_len, int head_dim, float scale,
                                    int num_kv_heads, int num_heads, const int* __restrict__ dpos,
                                    float* __restrict__ partial) {
  const int L = dpos ? *dpos + 1 : seq_len;
  const int h = blockIdx.x, c = blockIdx.y, nc = gridDim.y;
  const int kvh = h / (num_heads / num_kv_heads);
  const int row = num_kv_heads * head_dim, hoff = kvh * head_dim;
  float* __restrict__ part = partial + (static_cast<size_t>(h) * nc + c) * (head_dim + 2);
  const int chunk = ember_tiled::tiled_chunk(L, kMin, kMax);
  const int k0 = c * chunk;
  if (k0 >= L) return;
  const int k1 = min(L, k0 + chunk);
  const int n = k1 - k0;
  const int d4n = head_dim >> 2;
  extern __shared__ float4 kvq_sm4[];
  float4* q4 = kvq_sm4;
  float* s = reinterpret_cast<float*>(q4 + d4n);
  __shared__ float red[kThreads / 32];
  __shared__ float4 pacc4[kThreads];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float4* Q4 = reinterpret_cast<const float4*>(Q + static_cast<size_t>(h) * head_dim);
  for (int j = tid; j < d4n; j += kThreads) q4[j] = Q4[j];
  __syncthreads();
  const int quad = tid >> 2, ql = tid & 3;
  for (int kb = k0; kb < k1; kb += kThreads / 4) {
    const int k = kb + quad;
    float dot = 0.0f;
    if (k < k1) {
#pragma unroll 4
      for (int j = ql; j < d4n; j += 4) {
        const float4 a = q4[j];
        const float4 b = load4<F>(Kd, Ks, static_cast<size_t>(k), row, hoff + 4 * j);
        dot = fmaf(a.x, b.x, dot);
        dot = fmaf(a.y, b.y, dot);
        dot = fmaf(a.z, b.z, dot);
        dot = fmaf(a.w, b.w, dot);
      }
    }
    dot += __shfl_xor_sync(0xffffffff, dot, 1);
    dot += __shfl_xor_sync(0xffffffff, dot, 2);
    if (k < k1 && ql == 0) s[k - k0] = dot * scale;
  }
  __syncthreads();
  float mx = -INFINITY;
  for (int i = tid; i < n; i += kThreads) mx = fmaxf(mx, s[i]);
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  mx = red[0];
  for (int w = 1; w < kThreads / 32; ++w) mx = fmaxf(mx, red[w]);
  __syncthreads();
  float sum = 0.0f;
  for (int i = tid; i < n; i += kThreads) {
    const float e = expf(s[i] - mx);
    s[i] = e;
    sum += e;
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
  if (lane == 0) red[warp] = sum;
  __syncthreads();
  if (tid == 0) {
    float l = 0.0f;
    for (int w = 0; w < kThreads / 32; ++w) l += red[w];
    part[0] = mx;
    part[1] = l;
  }
  const int G = kThreads / d4n;
  const int g = tid / d4n, j = tid - g * d4n;
  if (g < G) {
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll 4
    for (int i = g; i < n; i += G) {
      const float w = s[i];
      const float4 v = load4<F>(Vd, Vs, static_cast<size_t>(k0 + i), row, hoff + 4 * j);
      acc.x = fmaf(w, v.x, acc.x);
      acc.y = fmaf(w, v.y, acc.y);
      acc.z = fmaf(w, v.z, acc.z);
      acc.w = fmaf(w, v.w, acc.w);
    }
    pacc4[g * d4n + j] = acc;
  }
  __syncthreads();
  const float* pacc = reinterpret_cast<const float*>(pacc4);
  for (int d = tid; d < head_dim; d += kThreads) {
    float a = 0.0f;
    for (int gg = 0; gg < G; ++gg) a += pacc[gg * head_dim + d];
    part[2 + d] = a;
  }
}

__global__ void decode_merge_kernel(const float* __restrict__ partial, float* __restrict__ out, int seq_len,
                                    int head_dim, int nc, const int* __restrict__ dpos) {
  const int L = dpos ? *dpos + 1 : seq_len;
  const int h = blockIdx.x;
  ember_tiled::tiled_merge_block(partial + static_cast<size_t>(h) * nc * (head_dim + 2),
                                 out + static_cast<size_t>(h) * head_dim, L, head_dim, kMin, kMax);
}

template <int F>
__global__ void dequant_kernel(const uint8_t* __restrict__ Kd, const uint8_t* __restrict__ Ks,
                               const uint8_t* __restrict__ Vd, const uint8_t* __restrict__ Vs, float* __restrict__ K,
                               float* __restrict__ V, int n, int row) {
  const int r4 = row / 4;
  const size_t per = static_cast<size_t>(n) * r4;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= 2 * per) return;
  const bool isv = i >= per;
  const size_t w = isv ? i - per : i;
  const size_t pos = w / r4;
  const int e = static_cast<int>(w - pos * r4) * 4;
  reinterpret_cast<float4*>(isv ? V : K)[w] = isv ? load4<F>(Vd, Vs, pos, row, e) : load4<F>(Kd, Ks, pos, row, e);
}

template <int F>
void decode_launch(const float* q, const void* Kd, const void* Ks, const void* Vd, const void* Vs, float* out, int hq,
                   int hkv, int hd, float scale, int max_len, int seq_len, float* part, cudaStream_t st,
                   const int* dpos) {
  const int nc = ember_tiled::tiled_grid_chunks(max_len, kMin, kMax);
  const size_t shm = ember_tiled::tiled_shm_bytes(hd, max_len, kMin, kMax);
  decode_split_kernel<F><<<dim3(hq, nc), kThreads, shm, st>>>(
      q, static_cast<const uint8_t*>(Kd), static_cast<const uint8_t*>(Ks), static_cast<const uint8_t*>(Vd),
      static_cast<const uint8_t*>(Vs), seq_len, hd, scale, hkv, hq, dpos, part);
  decode_merge_kernel<<<hq, kThreads, 0, st>>>(part, out, seq_len, hd, nc, dpos);
  check(cudaGetLastError(), "decode");
}

template <int F>
void dequant_launch(const void* Kd, const void* Ks, const void* Vd, const void* Vs, float* K, float* V, int n, int row,
                    cudaStream_t st) {
  const size_t total = 2 * static_cast<size_t>(n) * (row / 4);
  dequant_kernel<F><<<static_cast<unsigned>((total + 255) / 256), 256, 0, st>>>(
      static_cast<const uint8_t*>(Kd), static_cast<const uint8_t*>(Ks), static_cast<const uint8_t*>(Vd),
      static_cast<const uint8_t*>(Vs), K, V, n, row);
  check(cudaGetLastError(), "dequant");
}

}  // namespace

void write_cuda(int f, void* Kd, void* Ks, void* Vd, void* Vs, const float* k, const float* v, int T, int row,
                int start_pos, cudaStream_t st, const int* dpos) {
  if (T <= 0) return;
  if (f == kF16) {
    const size_t total = 2 * static_cast<size_t>(T) * row;
    write_f16_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, st>>>(
        static_cast<__half*>(Kd), static_cast<__half*>(Vd), k, v, T, row, start_pos, dpos);
  } else {
    const size_t threads = 2 * static_cast<size_t>(T) * (row / kBlock) * 32;
    const unsigned grid = static_cast<unsigned>((threads + 127) / 128);
    auto* kd = static_cast<uint8_t*>(Kd);
    auto* ks = static_cast<uint8_t*>(Ks);
    auto* vd = static_cast<uint8_t*>(Vd);
    auto* vs = static_cast<uint8_t*>(Vs);
    if (f == kQ8)
      write_block_kernel<kQ8><<<grid, 128, 0, st>>>(kd, ks, vd, vs, k, v, T, row, start_pos, dpos);
    else if (f == kQ4)
      write_block_kernel<kQ4><<<grid, 128, 0, st>>>(kd, ks, vd, vs, k, v, T, row, start_pos, dpos);
    else
      throw std::runtime_error("kv_quant write: format " + std::to_string(f));
  }
  check(cudaGetLastError(), "write");
}

size_t partial_floats(int hq, int hd, int max_len) {
  return static_cast<size_t>(hq) * ember_tiled::tiled_grid_chunks(max_len, kMin, kMax) * (hd + 2);
}

void decode_cuda(int f, const float* q, const void* Kd, const void* Ks, const void* Vd, const void* Vs, float* out,
                 int hq, int hkv, int hd, float scale, int max_len, int seq_len, float* part, cudaStream_t st,
                 const int* dpos) {
  if (!ember_tiled::tiled_shape_ok(hq, hd, max_len, 1 << 20, kMin, kMax) || hkv <= 0 || hq % hkv != 0)
    throw std::runtime_error("kv_quant decode: shape not supported (heads " + std::to_string(hq) + "/" +
                             std::to_string(hkv) + ", head_dim " + std::to_string(hd) + ", max_len " +
                             std::to_string(max_len) + ")");
  switch (f) {
    case kF16: decode_launch<kF16>(q, Kd, Ks, Vd, Vs, out, hq, hkv, hd, scale, max_len, seq_len, part, st, dpos); break;
    case kQ8: decode_launch<kQ8>(q, Kd, Ks, Vd, Vs, out, hq, hkv, hd, scale, max_len, seq_len, part, st, dpos); break;
    case kQ4: decode_launch<kQ4>(q, Kd, Ks, Vd, Vs, out, hq, hkv, hd, scale, max_len, seq_len, part, st, dpos); break;
    default: throw std::runtime_error("kv_quant decode: format " + std::to_string(f));
  }
}

void dequant_rows_cuda(int f, const void* Kd, const void* Ks, const void* Vd, const void* Vs, float* K, float* V, int n,
                       int row, cudaStream_t st) {
  if (n <= 0) return;
  switch (f) {
    case kF16: dequant_launch<kF16>(Kd, Ks, Vd, Vs, K, V, n, row, st); break;
    case kQ8: dequant_launch<kQ8>(Kd, Ks, Vd, Vs, K, V, n, row, st); break;
    case kQ4: dequant_launch<kQ4>(Kd, Ks, Vd, Vs, K, V, n, row, st); break;
    default: throw std::runtime_error("kv_quant dequant: format " + std::to_string(f));
  }
}

}  // namespace ember_kvq
