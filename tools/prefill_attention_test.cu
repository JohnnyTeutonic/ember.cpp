// The baseline prefill attention kernel (attention_batch_prefill_cuda_kernel: EMBER_PREFILL_ATTN=baseline,
// and the fallback whenever flash attention is off or does not take the shape, e.g. head_dim 80 or above
// 128) vs a double-precision CPU reference of causal grouped-query attention over the KV cache, for
// head dims that are and are not powers of two: a block of head_dim threads rounded up to a power of
// two (unrounded, the softmax's halving reductions drop partials at a head_dim such as 80 or 96, as
// rmsnorm_batch_cuda did at d 192 before ember 69d7ca2; tools/rmsnorm_batch_test.cu). Head dims
// 16 .. 256 with 4 query heads over 2 kv heads (and 3 over 1), batches of 1, 7 and 40 tokens at cache
// offsets 0 and 13. Pass: max |out - ref| / max(1, max |ref|) < 1e-5 everywhere; exit 1 otherwise.
// An optional argument names a file that receives every output as raw float32 (the inputs are seeded,
// so two builds' files compare bit for bit).
//   nvcc -O2 -std=c++17 -DHAS_CUDA -I. tools/prefill_attention_test.cu cuda_kernels.cu logger.cpp -lcublas -o prefill_attention_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_kernels.h"

int main(int argc, char** argv) {
  setenv("EMBER_PREFILL_ATTN", "baseline", 1);  // read once, at the first launch
  FILE* dump = argc > 1 ? std::fopen(argv[1], "wb") : nullptr;
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  int fails = 0;
  double worst_all = 0.0;
  struct Heads {
    int hq, hkv;
  };
  for (int hd : {16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 200, 256}) {
    double worst_hd = 0.0;
    for (Heads h : {Heads{4, 2}, Heads{3, 1}}) {
      for (int start : {0, 13}) {
        for (int T : {1, 7, 40}) {
          const int S = start + T + 3;  // cache capacity (rows past start + T are never read)
          const size_t qn = (size_t)T * h.hq * hd, kn = (size_t)S * h.hkv * hd;
          std::vector<float> q(qn), K(kn), V(kn), out(qn);
          for (auto& v : q) v = nd(rng);
          for (auto& v : K) v = nd(rng);
          for (auto& v : V) v = nd(rng);
          const float scale = 1.0f / std::sqrt((float)hd);
          float *dq, *dK, *dV, *dout;
          cudaMalloc(&dq, qn * 4);
          cudaMalloc(&dK, kn * 4);
          cudaMalloc(&dV, kn * 4);
          cudaMalloc(&dout, qn * 4);
          cudaMemcpy(dq, q.data(), qn * 4, cudaMemcpyHostToDevice);
          cudaMemcpy(dK, K.data(), kn * 4, cudaMemcpyHostToDevice);
          cudaMemcpy(dV, V.data(), kn * 4, cudaMemcpyHostToDevice);
          attention_batch_prefill_cuda(dq, nullptr, nullptr, dK, dV, dout, T, start, S, h.hq, h.hkv, hd, scale, 0, nullptr);
          cudaMemcpy(out.data(), dout, qn * 4, cudaMemcpyDeviceToHost);
          cudaFree(dq);
          cudaFree(dK);
          cudaFree(dV);
          cudaFree(dout);
          if (dump) std::fwrite(out.data(), 4, out.size(), dump);
          double err = 0.0, mref = 0.0;
          const int group = h.hq / h.hkv;
          for (int t = 0; t < T; ++t)
            for (int qh = 0; qh < h.hq; ++qh) {
              const float* qr = q.data() + ((size_t)t * h.hq + qh) * hd;
              const int kvh = qh / group, n = start + t + 1;
              std::vector<double> s(n);
              double mx = -INFINITY;
              for (int j = 0; j < n; ++j) {
                const float* kr = K.data() + ((size_t)j * h.hkv + kvh) * hd;
                double dot = 0.0;
                for (int d = 0; d < hd; ++d) dot += (double)qr[d] * kr[d];
                s[j] = dot * scale;
                mx = std::max(mx, s[j]);
              }
              double sum = 0.0;
              for (double& v : s) sum += (v = std::exp(v - mx));
              for (int d = 0; d < hd; ++d) {
                double ref = 0.0;
                for (int j = 0; j < n; ++j) ref += s[j] / sum * V[((size_t)j * h.hkv + kvh) * hd + d];
                mref = std::max(mref, std::fabs(ref));
                err = std::max(err, std::fabs((double)out[((size_t)t * h.hq + qh) * hd + d] - ref));
              }
            }
          worst_hd = std::max(worst_hd, err / std::max(1.0, mref));
        }
      }
    }
    const bool ok = worst_hd < 1e-5;
    fails += !ok;
    worst_all = std::max(worst_all, worst_hd);
    std::printf("head_dim %4d  worst rel %.3e  %s\n", hd, worst_hd, ok ? "ok" : "FAIL");
  }
  if (dump) std::fclose(dump);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 1;
  }
  std::printf("prefill_attention_test: %s (%d head dims failed; worst %.3e)\n", fails ? "FAIL" : "PASS", fails,
              worst_all);
  return fails ? 1 : 0;
}
