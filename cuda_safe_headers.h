#ifndef CUDA_SAFE_HEADERS_H
#define CUDA_SAFE_HEADERS_H

// Use NVIDIA's actual storage types and conversion rules on every platform.
// Missing toolkit headers are installation errors, never a reason to substitute
// approximate half types, conversions, or vendor include guards.
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

#endif // CUDA_SAFE_HEADERS_H
