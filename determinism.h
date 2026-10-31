// Deterministic CUDA mode (EMBER_DETERMINISTIC=1, off by default; docs/USAGE.md).
//
// Ordered warp-slot reductions are now the default for the tuned llama decode
// RMSNorm and softmax paths. No unmeasured atomic production opt-in is retained.
// This explicit mode additionally fixes cuBLAS workspace/atomics policy; its
// default is still off, and it does not silently enable itself for the new route.
// Historical pre-change default nondeterminism receipts remain unchanged.
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
