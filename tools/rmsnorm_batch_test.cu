// rmsnorm_batch_cuda (the prefill / verify rmsnorm of the llama device path, and the IR's group
// rmsnorm) vs a double-precision CPU reference, for row widths that are and are not powers of two:
// 32 .. 256 (a block of hidden_size threads rounded up to a power of two; hidden_size threads
// unrounded drop part of the sum of squares at a width such as 192), 320 .. 1024 (256 threads),
// with 1, 7 and 40 rows. The widths include every
// tiny_chat shape's d (192, 320, 384, 448, 512, 640, 704, 896) and the bug's neighbours (96, 160,
// 200, 255). Pass: max |out - ref| / max(1, max |ref|) < 1e-5 everywhere; exit 1 otherwise.
// An optional argument names a file that receives every output row as raw float32 (the inputs are
// seeded, so two builds' files compare bit for bit).
//   nvcc -O2 -std=c++17 -DHAS_CUDA -I. tools/rmsnorm_batch_test.cu cuda_kernels.cu logger.cpp -lcublas -o rmsnorm_batch_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_kernels.h"

int main(int argc, char** argv) {
  FILE* dump = argc > 1 ? std::fopen(argv[1], "wb") : nullptr;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  const float eps = 1e-5f;
  int fails = 0;
  double worst_all = 0.0;
  for (int d : {32, 64, 96, 128, 160, 192, 200, 255, 256, 320, 384, 448, 512, 640, 704, 832, 896, 1024}) {
    double worst_d = 0.0;
    for (int T : {1, 7, 40}) {
      std::vector<float> x((size_t)T * d), w(d), out((size_t)T * d);
      for (auto& v : x) v = nd(rng) * 2.0f;
      for (auto& v : w) v = 1.0f + 0.1f * nd(rng);
      float *dx, *dw, *dout;
      cudaMalloc(&dx, x.size() * 4);
      cudaMalloc(&dw, w.size() * 4);
      cudaMalloc(&dout, out.size() * 4);
      cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
      rmsnorm_batch_cuda(dout, dx, dw, T, d, eps, 0);
      cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
      cudaFree(dx);
      cudaFree(dw);
      cudaFree(dout);
      if (dump) std::fwrite(out.data(), 4, out.size(), dump);
      double err = 0.0, mref = 0.0;
      for (int t = 0; t < T; ++t) {
        const float* xt = x.data() + (size_t)t * d;
        double ss = 0.0;
        for (int i = 0; i < d; ++i) ss += (double)xt[i] * xt[i];
        const double inv = 1.0 / std::sqrt(ss / d + eps);
        for (int i = 0; i < d; ++i) {
          const double ref = (double)xt[i] * inv * w[i];
          mref = std::max(mref, std::fabs(ref));
          err = std::max(err, std::fabs((double)out[(size_t)t * d + i] - ref));
        }
      }
      worst_d = std::max(worst_d, err / std::max(1.0, mref));
    }
    const bool ok = worst_d < 1e-5;
    fails += !ok;
    worst_all = std::max(worst_all, worst_d);
    std::printf("d %4d  worst rel %.3e  %s\n", d, worst_d, ok ? "ok" : "FAIL");
  }
  if (dump) std::fclose(dump);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 1;
  }
  std::printf("rmsnorm_batch_test: %s (%d widths failed; worst %.3e)\n", fails ? "FAIL" : "PASS", fails, worst_all);
  return fails ? 1 : 0;
}
