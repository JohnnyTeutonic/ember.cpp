// occ_bench: decode attention for one query row per head, the split-K kernel ember ran before this work against
// the tiled kernel (decode_attn_tiled.cuh), on synthetic caches (ember.cpp, experiments/decode_occupancy_2026-10-27).
//
//   occ_bench OUT.jsonl [quick]
//
// Shapes are the picks' (head_dim 64): 5 heads (pick_e1_L16), 6 (pick_e1_60m_L32), 12 (pk_base_v1, pk_sft_v1);
// lengths L 128 / 384 in a 512-position cache (the picks' context), 128 / 1024 / 4096 in a 4096-position one.
// Variants:
//   split_prod   cuda_kernels.cu's decode_attn_split_kernel + merge as ember launches them: 256-key chunks for the
//                cache's capacity (a copy of the two kernels, the scratch passed in);
//   split_cC     the same kernels with chunks of C keys for the capacity (occupancy alone, C = 32, 64, 128);
//   tiled_m_x    decode_attn_tiled.cuh with min_chunk m, max_chunks x (the chunking follows L).
// Per (shape, L, capacity, variant): max |err| against a float64 CPU reference, two runs compared bit for bit,
// the grid (blocks launched and blocks with keys), and the time per layer as a decode graph sees it (a CUDA graph of
// 64 split + merge pairs, replayed 30 times, CUDA events). NVTX ranges "<variant>|H|L|cap|hot" (100 direct launches,
// the cache in L2) and "...|cold" (50 launches, each after a 128 MB write that evicts L2) give nsys the kernels'
// own times. The DRAM floor: bytes of K and V read (2 L kv_heads head_dim 4) over the measured copy bandwidth and
// the L4's 300 GB/s.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "decode_attn_tiled.cuh"

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(1);                                                                             \
    }                                                                                           \
  } while (0)

namespace {

// ---- cuda_kernels.cu's split kernel and merge at 7526469, the scratch a parameter -------------------------------
constexpr int kDecThreads = 128;

__global__ void old_split_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                 const float* __restrict__ V, int seq_len, int head_dim, float scale,
                                 int num_kv_heads, int num_heads, int chunk, const int* step_dev,
                                 float* g_dec_partial) {
  if (step_dev) seq_len = step_dev[1] + 1;
  const int h = blockIdx.x, c = blockIdx.y, nc = gridDim.y;
  float* part = g_dec_partial + ((size_t)h * nc + c) * (head_dim + 2);  // [m, l, acc[head_dim]]
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

__global__ void old_merge_kernel(float* __restrict__ out, int head_dim, int nc, const float* g_dec_partial) {
  const int h = blockIdx.x;
  const float* base = g_dec_partial + (size_t)h * nc * (head_dim + 2);
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

// ---- the tiled kernels as cuda_kernels.cu launches them, the scratch a parameter --------------------------------
__global__ void tiled_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
                             int head_dim, float scale, int num_kv_heads, int num_heads, int mn, int mx,
                             const int* step_dev, float* partial) {
  const int L = step_dev[1] + 1;
  const int h = blockIdx.x, c = blockIdx.y, nc = gridDim.y;
  ember_tiled::tiled_split_block(Q + (size_t)h * head_dim, K, V, L, c, h / (num_heads / num_kv_heads), num_kv_heads,
                                 head_dim, scale, mn, mx, partial + ((size_t)h * nc + c) * (head_dim + 2));
}

__global__ void tiled_merge_kernel(float* __restrict__ out, int head_dim, int nc, int mn, int mx,
                                   const int* step_dev, const float* partial) {
  const int L = step_dev[1] + 1;
  const int h = blockIdx.x;
  ember_tiled::tiled_merge_block(partial + (size_t)h * nc * (head_dim + 2), out + (size_t)h * head_dim, L, head_dim,
                                 mn, mx);
}

__global__ void flush_kernel(float4* p, size_t n, float v) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = make_float4(v, v, v, v);
}

__global__ void copy_kernel(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    b[i] = a[i];
}

struct Variant {
  std::string name;
  int kind;  // 0 split with chunk for the capacity (chunk 0: production), 1 tiled
  int a, b;  // split: chunk (0 = production); tiled: min_chunk, max_chunks
};

struct Bufs {
  float *K, *V, *Q, *O, *part;
  int* step;
  float4* flush;
  size_t flush_n;
};

struct Launch {
  int grid_chunks = 0, active_chunks = 0, chunk = 0;
  size_t shm = 0;
};

Launch plan(const Variant& v, int L, int cap) {
  Launch p;
  if (v.kind == 0) {
    int nc, chunk;
    if (v.a == 0) {  // decode_attn_split for max_len = cap
      nc = (cap + 255) / 256;
      if (nc > 64) nc = 64;
      chunk = (cap + nc - 1) / nc;
    } else {
      chunk = v.a;
      nc = (cap + chunk - 1) / chunk;
    }
    p.grid_chunks = nc;
    p.chunk = chunk;
    p.active_chunks = (L + chunk - 1) / chunk;
    p.shm = (size_t)(64 + chunk) * sizeof(float);
  } else {
    p.grid_chunks = ember_tiled::tiled_grid_chunks(cap, v.a, v.b);
    p.chunk = ember_tiled::tiled_chunk(L, v.a, v.b);
    p.active_chunks = ember_tiled::tiled_nchunks(L, p.chunk);
    p.shm = ember_tiled::tiled_shm_bytes(64, cap, v.a, v.b);
  }
  return p;
}

void launch(const Variant& v, const Launch& p, const Bufs& b, int H, int KV, int D, cudaStream_t st) {
  const float scale = 1.0f / std::sqrt((float)D);
  if (v.kind == 0) {
    old_split_kernel<<<dim3(H, p.grid_chunks), kDecThreads, (size_t)(D + p.chunk) * sizeof(float), st>>>(
        b.Q, b.K, b.V, 0, D, scale, KV, H, p.chunk, b.step, b.part);
    old_merge_kernel<<<H, kDecThreads, 0, st>>>(b.O, D, p.grid_chunks, b.part);
  } else {
    tiled_kernel<<<dim3(H, p.grid_chunks), ember_tiled::kTiledThreads, p.shm, st>>>(b.Q, b.K, b.V, D, scale, KV, H,
                                                                                    v.a, v.b, b.step, b.part);
    tiled_merge_kernel<<<H, ember_tiled::kTiledThreads, 0, st>>>(b.O, D, p.grid_chunks, v.a, v.b, b.step, b.part);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: occ_bench OUT.jsonl [quick]\n");
    return 2;
  }
  const bool quick = argc > 2 && std::string(argv[2]) == "quick";
  FILE* out = std::fopen(argv[1], "w");
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));

  // Bandwidth: a float4 copy of 256 MB (read + write) and the L2 size.
  const size_t bw_n = (256u << 20) / 16;
  float4 *ba, *bb;
  CK(cudaMalloc(&ba, bw_n * 16));
  CK(cudaMalloc(&bb, bw_n * 16));
  CK(cudaMemset(ba, 0, bw_n * 16));
  for (int i = 0; i < 3; ++i) copy_kernel<<<prop.multiProcessorCount * 8, 256, 0, st>>>(ba, bb, bw_n);
  CK(cudaEventRecord(e0, st));
  for (int i = 0; i < 20; ++i) copy_kernel<<<prop.multiProcessorCount * 8, 256, 0, st>>>(ba, bb, bw_n);
  CK(cudaEventRecord(e1, st));
  CK(cudaEventSynchronize(e1));
  float bw_ms = 0;
  CK(cudaEventElapsedTime(&bw_ms, e0, e1));
  const double copy_gbs = 20.0 * 2.0 * bw_n * 16 / (bw_ms * 1e-3) / 1e9;
  CK(cudaFree(bb));
  std::fprintf(out,
               "{\"gpu\": \"%s\", \"sm\": %d, \"sms\": %d, \"l2_bytes\": %d, \"copy_gbs\": %.1f, \"spec_gbs\": 300}\n",
               prop.name, prop.major * 10 + prop.minor, prop.multiProcessorCount, prop.l2CacheSize, copy_gbs);
  std::printf("%s: %d SMs, L2 %d MB, copy %.1f GB/s\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20,
              copy_gbs);

  std::vector<Variant> vars = {{"split_prod", 0, 0, 0}, {"split_c32", 0, 32, 0}, {"split_c64", 0, 64, 0},
                               {"split_c128", 0, 128, 0}};
  for (int mn : {8, 16, 32, 64, 128})
    for (int mx : {16, 32, 64}) vars.push_back({"tiled_" + std::to_string(mn) + "_" + std::to_string(mx), 1, mn, mx});
  if (quick) vars = {{"split_prod", 0, 0, 0}, {"tiled_32_64", 1, 32, 64}};

  struct Shape {
    const char* name;
    int H, KV, D;
  };
  const std::vector<Shape> shapes = {{"pick_e1_L16", 5, 5, 64}, {"pick_e1_60m_L32", 6, 6, 64}, {"pk_base_v1", 12, 12, 64}};
  const std::vector<std::pair<int, int>> cases = {{128, 512}, {384, 512}, {128, 4096}, {1024, 4096}, {4096, 4096}};
  const int MAXCAP = 4096, R = 64, REPS = 30;

  Bufs b{};
  b.flush_n = (128u << 20) / 16;
  b.flush = ba;  // the bandwidth buffer, 256 MB: the first 128 MB
  int fails = 0;
  for (const Shape& sh : shapes) {
    const int H = sh.H, KV = sh.KV, D = sh.D;
    std::mt19937 rng(7 + H);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> K((size_t)MAXCAP * KV * D), V(K.size()), Q((size_t)H * D);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);
    for (auto& x : Q) x = nd(rng) * 2.0f;
    CK(cudaMalloc(&b.K, K.size() * 4));
    CK(cudaMalloc(&b.V, V.size() * 4));
    CK(cudaMalloc(&b.Q, Q.size() * 4));
    CK(cudaMalloc(&b.O, (size_t)H * D * 4));
    CK(cudaMalloc(&b.part, (size_t)H * 256 * (D + 2) * 4));
    CK(cudaMalloc(&b.step, 8));
    CK(cudaMemcpy(b.K, K.data(), K.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.V, V.data(), V.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.Q, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice));
    const float scale = 1.0f / std::sqrt((float)D);
    for (auto [L, cap] : cases) {
      std::vector<double> ref((size_t)H * D);
      for (int h = 0; h < H; ++h) {
        const int kvh = h / (H / KV);
        std::vector<double> s(L);
        double m = -1e300;
        for (int k = 0; k < L; ++k) {
          double dot = 0;
          for (int d = 0; d < D; ++d) dot += (double)Q[h * D + d] * K[((size_t)k * KV + kvh) * D + d];
          s[k] = dot * scale;
          m = std::max(m, s[k]);
        }
        double z = 0;
        for (auto& x : s) z += (x = std::exp(x - m));
        for (int d = 0; d < D; ++d) {
          double a = 0;
          for (int k = 0; k < L; ++k) a += s[k] * V[((size_t)k * KV + kvh) * D + d];
          ref[(size_t)h * D + d] = a / z;
        }
      }
      int stv[2] = {L - 1, L - 1};
      CK(cudaMemcpy(b.step, stv, 8, cudaMemcpyHostToDevice));
      const double bytes = 2.0 * L * KV * D * 4;
      for (const Variant& v : vars) {
        const Launch p = plan(v, L, cap);
        if (p.grid_chunks > 256 || p.chunk > 2048) continue;
        // correctness and run to run
        std::vector<float> o1((size_t)H * D), o2((size_t)H * D);
        launch(v, p, b, H, KV, D, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(o1.data(), b.O, o1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemset(b.O, 0, o1.size() * 4));
        launch(v, p, b, H, KV, D, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(o2.data(), b.O, o2.size() * 4, cudaMemcpyDeviceToHost));
        double err = 0;
        for (size_t i = 0; i < o1.size(); ++i) err = std::max(err, std::fabs((double)o1[i] - ref[i]));
        const bool same = std::memcmp(o1.data(), o2.data(), o1.size() * 4) == 0;
        if (err > 1e-4 || !same) ++fails;
        // a decode graph's view: R pairs in one graph
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int r = 0; r < R; ++r) launch(v, p, b, H, KV, D, st);
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
        CK(cudaEventRecord(e0, st));
        for (int i = 0; i < REPS; ++i) CK(cudaGraphLaunch(ge, st));
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        const double us_graph = ms * 1e3 / (REPS * R);
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
        // nsys ranges
        const std::string tag = v.name + "|" + sh.name + "|L" + std::to_string(L) + "|cap" + std::to_string(cap);
        nvtxRangePushA((tag + "|hot").c_str());
        for (int i = 0; i < 100; ++i) launch(v, p, b, H, KV, D, st);
        CK(cudaStreamSynchronize(st));
        nvtxRangePop();
        nvtxRangePushA((tag + "|cold").c_str());
        for (int i = 0; i < 50; ++i) {
          flush_kernel<<<prop.multiProcessorCount * 4, 256, 0, st>>>(b.flush, b.flush_n, (float)i);
          launch(v, p, b, H, KV, D, st);
        }
        CK(cudaStreamSynchronize(st));
        nvtxRangePop();
        CK(cudaGetLastError());
        std::fprintf(out,
                     "{\"variant\": \"%s\", \"shape\": \"%s\", \"H\": %d, \"KV\": %d, \"D\": %d, \"L\": %d, "
                     "\"cap\": %d, \"chunk\": %d, \"grid_blocks\": %d, \"active_blocks\": %d, \"max_err\": %.3e, "
                     "\"run_to_run_bit_identical\": %s, \"us_per_layer_graph\": %.3f, \"kv_bytes\": %.0f, "
                     "\"floor_us_spec\": %.3f, \"floor_us_copy\": %.3f}\n",
                     v.name.c_str(), sh.name, H, KV, D, L, cap, p.chunk, H * p.grid_chunks, H * p.active_chunks, err,
                     same ? "true" : "false", us_graph, bytes, bytes / 300e9 * 1e6, bytes / (copy_gbs * 1e9) * 1e6);
        std::fflush(out);
        std::printf("%-12s %-16s L %4d cap %4d chunk %4d blocks %4d/%4d err %.1e %s  %.2f us/layer\n",
                    v.name.c_str(), sh.name, L, cap, p.chunk, H * p.active_chunks, H * p.grid_chunks, err,
                    same ? "bits" : "DIFF", us_graph);
      }
    }
    cudaFree(b.K);
    cudaFree(b.V);
    cudaFree(b.Q);
    cudaFree(b.O);
    cudaFree(b.part);
    cudaFree(b.step);
  }
  cudaFree(ba);
  std::fclose(out);
  std::printf(fails ? "FAILED %d\n" : "all ok\n", fails);
  return fails != 0;
}
