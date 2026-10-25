// Deterministic CUDA mode (EMBER_DETERMINISTIC=1, off by default; docs/USAGE.md).
//
// With it, CUDA prefill and decode logits are bit-identical run to run on the same GPU and build. The default
// decode path is not: rmsnorm_optimized_kernel adds its warp sums into shared memory with atomicAdd, in whatever
// order the warps arrive (experiments/determinism_2026-10-26). Under the mode:
//   - rmsnorm_vector_cuda_optimized (the decode rmsnorm, per-launch and CUDA-graph) adds the warp sums in warp
//     order (rmsnorm_optimized_det_kernel);
//   - the one-block-per-head decode attention (EMBER_DECODE_ATTN=legacy, or a shape past the split kernel's
//     scratch) reduces its max and softmax sum in warp order (attention_optimized_det_kernel);
//   - every cuBLAS handle gets a fixed workspace (CUBLAS_WORKSPACE_CONFIG=:4096:8 unless the variable is already
//     set) and CUBLAS_ATOMICS_NOT_ALLOWED.
// Unset (or "0"), every path is exactly what it was: the same kernels, launches and cuBLAS setup.
#ifndef EMBER_DETERMINISM_H
#define EMBER_DETERMINISM_H

#include <cstdlib>
#include <cstring>

namespace ember {

// True when EMBER_DETERMINISTIC is set to anything but "" or "0". Read once.
inline bool deterministic() {
  static const bool on = [] {
    const char* v = std::getenv("EMBER_DETERMINISTIC");
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
  }();
  return on;
}

// Called before each cublasCreate: under the mode, pins cuBLAS's workspace configuration (the size can steer its
// kernel choice) unless the caller already set one. A no-op when the mode is off.
inline void deterministic_cublas_env() {
  if (!deterministic() || std::getenv("CUBLAS_WORKSPACE_CONFIG") != nullptr) return;
#ifdef _WIN32
  _putenv_s("CUBLAS_WORKSPACE_CONFIG", ":4096:8");
#else
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 0);
#endif
}

}  // namespace ember

#endif  // EMBER_DETERMINISM_H
