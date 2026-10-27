// Tiled split-K decode attention: the block bodies shared by the decode step (cuda_kernels.cu) and the speculative
// verification pass (verify_kernels.cu), so a verification row is the decode step's arithmetic by construction.
//
// One new query row attends to L keys of one head. The keys are cut into nchunks(L) chunks of chunk(L) consecutive
// keys; one block of kTiledThreads threads takes one (head, chunk):
//   scores   a quad of 4 lanes per key, each lane a float4 slice of head_dim (j, j + 4, ...), the 4 lane sums added
//            by a 2-step xor butterfly (the same value in all 4 lanes); 32 keys per pass of the block;
//   softmax  the chunk's max and sum of e^{s - max} by warp shuffles and per-warp slots added in warp order;
//   values   G = 128 / (head_dim / 4) groups of head_dim / 4 lanes; group g adds the keys g, g + G, ... in order
//            into a float4 accumulator; the G partial sums are then added in group order;
// and writes (m, l, unnormalised sum_k p_k v_k) to its slot of the scratch. The merge (one block per head) takes
// the chunks' maximum M, the weights w_c = e^{m_c - M}, L = sum_c w_c l_c by a fixed shuffle tree, and
// out = (sum_c w_c acc_c) / L, adding the chunks in index order. Every reduction has a fixed order, so the result
// is the same run to run, and it depends only on L (not on the cache's capacity or the grid): the graph decode,
// the per-launch decode and the verification rows agree bit for bit at equal L.
//
// The chunking (tiled_chunk): chunk = max(kTiledMinChunk, ceil(L / kTiledMaxChunks)), nchunks = ceil(L / chunk).
// A grid for a cache of max_len positions has tiled_grid_chunks(max_len) chunk blocks per head, an upper bound for
// every L <= max_len; blocks past nchunks(L) return at once. The old kernel (256-key chunks for the cache's
// capacity, experiments/decode_occupancy_2026-10-27) ran 10 blocks for pick_e1_L16 at any length.
#ifndef EMBER_DECODE_ATTN_TILED_CUH
#define EMBER_DECODE_ATTN_TILED_CUH

#include <cuda_runtime.h>
#include <math.h>

namespace ember_tiled {

constexpr int kTiledThreads = 128;
constexpr int kTiledMinChunk = 32;     // keys per chunk below kTiledMinChunk * kTiledMaxChunks keys
constexpr int kTiledMaxChunks = 64;    // chunks per head at most (the merge's weights; the scratch's chunk slots)
constexpr int kTiledChunkMax = 2048;   // keys per chunk at most (the scores in shared memory)
constexpr int kTiledHeadDimMax = 256;  // head_dim % 4 == 0 and <= this

__host__ __device__ inline int tiled_chunk(int L, int min_chunk, int max_chunks) {
  const int c = (L + max_chunks - 1) / max_chunks;
  return c < min_chunk ? min_chunk : c;
}
__host__ __device__ inline int tiled_nchunks(int L, int chunk) { return (L + chunk - 1) / chunk; }
// Chunk blocks per head for a cache of max_len positions (>= tiled_nchunks for every L <= max_len).
__host__ __device__ inline int tiled_grid_chunks(int max_len, int min_chunk, int max_chunks) {
  const int n = (max_len + min_chunk - 1) / min_chunk;
  return n < max_chunks ? n : max_chunks;
}
// True when the tiled kernel takes this shape for a cache of max_len positions.
__host__ inline bool tiled_shape_ok(int heads, int head_dim, int max_len, int heads_max, int min_chunk,
                                    int max_chunks) {
  return heads > 0 && heads <= heads_max && head_dim > 0 && head_dim % 4 == 0 && head_dim <= kTiledHeadDimMax &&
         max_len > 0 && max_chunks <= kTiledMaxChunks && tiled_chunk(max_len, min_chunk, max_chunks) <= kTiledChunkMax;
}
// Dynamic shared memory of the split block: q[head_dim] and the scores s[chunk] for the cache's largest chunk.
__host__ inline size_t tiled_shm_bytes(int head_dim, int max_len, int min_chunk, int max_chunks) {
  return (size_t)(head_dim + tiled_chunk(max_len, min_chunk, max_chunks)) * sizeof(float);
}

// One (head h, chunk c) block. Qh: the head's query row [head_dim]; K, V: the layer's cache [pos][kv_head][head_dim];
// part: this block's slot [m, l, acc[head_dim]]. Every thread of the block calls it.
__device__ __forceinline__ void tiled_split_block(const float* __restrict__ Qh, const float* __restrict__ K,
                                                  const float* __restrict__ V, int L, int c, int kvh,
                                                  int num_kv_heads, int head_dim, float scale, int min_chunk,
                                                  int max_chunks, float* __restrict__ part) {
  const int chunk = tiled_chunk(L, min_chunk, max_chunks);
  const int k0 = c * chunk;
  if (k0 >= L) return;  // past nchunks(L): the merge does not read this slot
  const int k1 = min(L, k0 + chunk);
  const int n = k1 - k0;
  const int d4n = head_dim >> 2;
  extern __shared__ float4 tiled_sm4[];
  float4* q4 = tiled_sm4;                       // [head_dim / 4]
  float* s = reinterpret_cast<float*>(q4 + d4n);  // [chunk]
  __shared__ float red[kTiledThreads / 32];
  __shared__ float4 pacc4[kTiledThreads];       // [G][head_dim / 4]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float4* Q4 = reinterpret_cast<const float4*>(Qh);
  for (int j = tid; j < d4n; j += kTiledThreads) q4[j] = Q4[j];
  __syncthreads();
  const size_t row4 = (size_t)num_kv_heads * d4n;  // float4s per cache position
  const float4* K4 = reinterpret_cast<const float4*>(K) + (size_t)kvh * d4n;
  const float4* V4 = reinterpret_cast<const float4*>(V) + (size_t)kvh * d4n;
  // Scores: quad (tid >> 2) takes key kb + quad; lane (tid & 3) the float4 slices ql, ql + 4, ...
  const int quad = tid >> 2, ql = tid & 3;
  for (int kb = k0; kb < k1; kb += kTiledThreads / 4) {
    const int k = kb + quad;
    float dot = 0.0f;
    if (k < k1) {
      const float4* kr = K4 + (size_t)k * row4;
#pragma unroll 4
      for (int j = ql; j < d4n; j += 4) {
        const float4 a = q4[j];
        const float4 b = __ldg(kr + j);
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
  for (int i = tid; i < n; i += kTiledThreads) mx = fmaxf(mx, s[i]);
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  mx = red[0];
  for (int w = 1; w < kTiledThreads / 32; ++w) mx = fmaxf(mx, red[w]);
  __syncthreads();
  float sum = 0.0f;
  for (int i = tid; i < n; i += kTiledThreads) {
    const float e = expf(s[i] - mx);
    s[i] = e;
    sum += e;
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
  if (lane == 0) red[warp] = sum;
  __syncthreads();  // also publishes s[] = e^{s - max} to the value pass
  if (tid == 0) {
    float l = 0.0f;
    for (int w = 0; w < kTiledThreads / 32; ++w) l += red[w];
    part[0] = mx;
    part[1] = l;
  }
  // Values: group g = tid / d4n adds keys g, g + G, ... (G = 128 / d4n groups; threads past G * d4n idle).
  const int G = kTiledThreads / d4n;
  const int g = tid / d4n, j = tid - g * d4n;
  if (g < G) {
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4* vc = V4 + (size_t)k0 * row4 + j;
#pragma unroll 4
    for (int i = g; i < n; i += G) {
      const float w = s[i];
      const float4 v = __ldg(vc + (size_t)i * row4);
      acc.x = fmaf(w, v.x, acc.x);
      acc.y = fmaf(w, v.y, acc.y);
      acc.z = fmaf(w, v.z, acc.z);
      acc.w = fmaf(w, v.w, acc.w);
    }
    pacc4[g * d4n + j] = acc;
  }
  __syncthreads();
  const float* pacc = reinterpret_cast<const float*>(pacc4);
  for (int d = tid; d < head_dim; d += kTiledThreads) {
    float a = 0.0f;
    for (int gg = 0; gg < G; ++gg) a += pacc[gg * head_dim + d];
    part[2 + d] = a;
  }
}

// The merge of one head: base = the head's chunk slots (stride head_dim + 2), out_h = the head's output row.
__device__ __forceinline__ void tiled_merge_block(const float* __restrict__ base, float* __restrict__ out_h, int L,
                                                  int head_dim, int min_chunk, int max_chunks) {
  const int nc = tiled_nchunks(L, tiled_chunk(L, min_chunk, max_chunks));  // <= max_chunks <= 64
  const int stride = head_dim + 2;
  __shared__ float w[kTiledMaxChunks];
  __shared__ float inv_s;
  const int tid = threadIdx.x;
  if (tid < 32) {
    const int c0 = tid, c1 = tid + 32;
    const float m0 = c0 < nc ? base[(size_t)c0 * stride] : -INFINITY;
    const float m1 = c1 < nc ? base[(size_t)c1 * stride] : -INFINITY;
    float M = fmaxf(m0, m1);
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, o));
    const float w0 = c0 < nc ? expf(m0 - M) : 0.0f;
    const float w1 = c1 < nc ? expf(m1 - M) : 0.0f;
    w[c0] = w0;
    w[c1] = w1;
    // Explicit roundings: no contraction into an fma, so every kernel that inlines this adds the same values.
    const float l0 = c0 < nc ? __fmul_rn(w0, base[(size_t)c0 * stride + 1]) : 0.0f;
    const float l1 = c1 < nc ? __fmul_rn(w1, base[(size_t)c1 * stride + 1]) : 0.0f;
    float l = __fadd_rn(l0, l1);
    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffff, l, o);
    if (tid == 0) inv_s = l > 0.0f ? 1.0f / l : 0.0f;
  }
  __syncthreads();
  const float inv = inv_s;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    const float* p = base + 2 + d;
#pragma unroll 4
    for (int c = 0; c < nc; ++c) acc = fmaf(w[c], p[(size_t)c * stride], acc);
    out_h[d] = acc * inv;
  }
}

}  // namespace ember_tiled

#endif  // EMBER_DECODE_ATTN_TILED_CUH
