/**
 * @file cuda_kernels.cu
 * @brief CUDA kernel implementations for TinyLlama GPU operations
 *
 * This file contains the actual CUDA kernel implementations for the GPU-accelerated
 * operations declared in cuda_kernels.h. The implementations are optimized for
 * NVIDIA GPUs and use CUDA-specific features for maximum performance.
 *
 * Key implementation details:
 * - Uses shared memory for efficient reduction operations
 * - Implements parallel reduction patterns for normalization
 * - Leverages cuBLAS for matrix operations
 * - Supports both FP32 and BF16 formats
 * - Includes error checking and memory management
 */

#include "cuda_kernels.h"
#include "logger.h"
#include "model_macros.h"

#ifdef HAS_CUDA

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Define CUDA_CHECK macro if not already defined
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) gpuErrchk(call)
#endif

// Host conversion functions for BF16
inline uint16_t float32_to_bf16_host(float f) {
    union { float f32; uint32_t u32; } u = {f};
    uint32_t rounding_bias = 0x00007FFF + ((u.u32 >> 16) & 1);
    return (u.u32 + rounding_bias) >> 16;
}

inline float bf16_to_float32_host(uint16_t bf16) {
    union { float f32; uint32_t u32; } u;
    u.u32 = ((uint32_t)bf16) << 16;
    return u.f32;
}

#endif

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cfloat>
#include <cstdlib>
#include <string>

// Place these at the top of the file before any kernel uses them
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ float atomicAddFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val + __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}

/**
 * @brief Converts BF16 to FP32 on the device
 * 
 * This device function converts Brain Floating Point (BF16) values to FP32.
 * It uses CUDA's native BF16 support on architectures >= 800, and falls back
 * to a manual conversion on older architectures.
 * 
 * @param bf16_raw Raw BF16 value as uint16_t
 * @return Converted FP32 value
 */
__device__ inline float bf16_to_float32_device(uint16_t bf16_raw) {
#if __CUDA_ARCH__ >= 800
  __nv_bfloat16 bf16_val;

  memcpy(&bf16_val, &bf16_raw, sizeof(uint16_t));
  return __bfloat162float(bf16_val);
#else

  unsigned int bits = ((unsigned int)bf16_raw) << 16;
  float result;
  memcpy(&result, &bits, sizeof(float));
  return result;
#endif
}

// Add FP32 to BF16 conversion device function
__device__ inline uint16_t float32_to_bf16_device(float fp32_val) {
#if __CUDA_ARCH__ >= 800
    __nv_bfloat16 bf16_val = __float2bfloat16(fp32_val);
    uint16_t bf16_raw;
    memcpy(&bf16_raw, &bf16_val, sizeof(uint16_t));
    return bf16_raw;
#else
    // Manual conversion for older architectures
    unsigned int bits;
    memcpy(&bits, &fp32_val, sizeof(float));
    // Extract upper 16 bits for BF16
    return (uint16_t)(bits >> 16);
#endif
}

/**
 * @brief Kernel for computing sum of squares in RMS normalization
 * 
 * This kernel computes the sum of squares of input elements using parallel
 * reduction with shared memory. It's the first step in RMS normalization.
 * 
 * @param x Input tensor
 * @param partial_sums Output array for partial sums
 * @param n Size of input tensor
 */
__global__ void rmsnorm_sum_squares_kernel(const float* x, float* partial_sums,
                                           int n) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? x[i] * x[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = sdata[0];
  }
}

/**
 * @brief Kernel for applying RMS normalization
 * 
 * This kernel applies the normalization weights and scaling factor to the
 * input tensor. It's the second step in RMS normalization.
 * 
 * @param x Input tensor
 * @param weight Normalization weights
 * @param out Output tensor
 * @param n Size of tensors
 * @param inv_norm_factor Inverse of the normalization factor
 */
__global__ void rmsnorm_apply_kernel(const float* x, const float* weight,
                                     float* out, int n, float inv_norm_factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] * inv_norm_factor * weight[i];
  }
}

__global__ void reduce_partial_sums_kernel(const float* partial_sums, float* total_sum_sq_out, int num_partial_sums) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    float my_sum = 0.0f;
    for (unsigned int i = tid; i < num_partial_sums; i += blockDim.x) {
        my_sum += partial_sums[i];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    // Standard parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        total_sum_sq_out[0] = sdata[0];
    }
}

/**
 * @brief Implementation of RMS normalization with device pointers
 * 
 * This function implements the complete RMS normalization process:
 * 1. Computes sum of squares using parallel reduction
 * 2. Calculates normalization factor
 * 3. Applies normalization with weights
 * 
 * @param x_dev Input tensor (device pointer)
 * @param weight_dev Normalization weights (device pointer)
 * @param out_dev Output tensor (device pointer)
 * @param n Size of tensors
 * @param eps Epsilon for numerical stability
 * @param stream CUDA stream for asynchronous execution
 */
void rmsnorm_vector_cuda(const float* x_dev, const float* weight_dev,
                         float* out_dev, int n, float eps,
                         cudaStream_t stream) {
  const int threads_per_block = 256; // General purpose threads per block
  int num_blocks_reduce_pass1 = (n + threads_per_block - 1) / threads_per_block;
  size_t shared_mem_size_pass1 = threads_per_block * sizeof(float);

  float* partial_sums_dev = nullptr;
  gpuErrchk(cudaMalloc(&partial_sums_dev, num_blocks_reduce_pass1 * sizeof(float)));

  rmsnorm_sum_squares_kernel<<<num_blocks_reduce_pass1, threads_per_block,
                               shared_mem_size_pass1, stream>>>(x_dev,
                                                          partial_sums_dev, n);
  gpuErrchk(cudaGetLastError());

  float* total_sum_sq_dev = nullptr;
  gpuErrchk(cudaMalloc(&total_sum_sq_dev, sizeof(float)));

  const int threads_for_final_reduction = (num_blocks_reduce_pass1 <= 256) ? 256 :
                                          (num_blocks_reduce_pass1 <= 512) ? 512 : 1024; 
                                          // Cap at 1024, ensure kernel handles num_partial_sums correctly if it's less than blockDim.x
  size_t shared_mem_final_reduction = threads_for_final_reduction * sizeof(float);
  
  // Launch a single block for the final reduction of partial_sums_dev
  reduce_partial_sums_kernel<<<1, threads_for_final_reduction, 
                               shared_mem_final_reduction, stream>>>(
                                   partial_sums_dev, total_sum_sq_dev, num_blocks_reduce_pass1);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaFree(partial_sums_dev)); // Free intermediate partial sums

  float h_total_ssq = 0.0f;
  gpuErrchk(cudaMemcpyAsync(&h_total_ssq, total_sum_sq_dev, sizeof(float),
                       cudaMemcpyDeviceToHost, stream));
  gpuErrchk(cudaStreamSynchronize(stream)); // Ensure h_total_ssq is ready
  gpuErrchk(cudaFree(total_sum_sq_dev));

  double total_ssq_double = static_cast<double>(h_total_ssq);
  total_ssq_double /= n;
  float inv_norm_factor = 1.0f / SAFE_SQRT(static_cast<float>(total_ssq_double) + eps);

  int num_blocks_apply = (n + threads_per_block - 1) / threads_per_block;
  rmsnorm_apply_kernel<<<num_blocks_apply, threads_per_block, 0, stream>>>(
      x_dev, weight_dev, out_dev, n, inv_norm_factor);
  gpuErrchk(cudaGetLastError());
}

void rmsnorm_vector_cuda(const std::vector<float>& x_in_host,
                         const std::vector<float>& weight_host,
                         std::vector<float>& out_host, int n, float eps) {
  out_host.resize(n);
  float *x_dev = nullptr, *weight_dev = nullptr, *out_dev = nullptr;
  gpuErrchk(cudaMalloc(&x_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&weight_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));
  gpuErrchk(cudaMemcpy(x_dev, x_in_host.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(weight_dev, weight_host.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice));
  rmsnorm_vector_cuda(x_dev, weight_dev, out_dev, n, eps, 0);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(x_dev));
  gpuErrchk(cudaFree(weight_dev));
  gpuErrchk(cudaFree(out_dev));
}

/**
 * @brief Implementation of matrix-vector multiplication with FP32
 * 
 * This function performs matrix-vector multiplication using cuBLAS,
 * optimized for FP32 precision. It handles the cuBLAS setup and
 * error checking.
 * 
 * @param handle cuBLAS handle
 * @param mat_f32_dev Matrix in row-major format (device pointer)
 * @param vec_f32_dev Vector to multiply (device pointer)
 * @param out_f32_dev Output vector (device pointer)
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 * @param stream CUDA stream for asynchronous execution
 */
void matvec_f32_f32_cuda(cublasHandle_t handle, const float* mat_f32_dev,
                         const float* vec_f32_dev, float* out_f32_dev,
                         int rows, int cols, cudaStream_t stream) {
  // Existing implementation of matvec_f32_f32_cuda
  // C = A^T * B where A is mat_f32_dev (passed as KxM) and B is vec_f32_dev (KxN)
  // M_blas = rows, N_blas = 1, K_blas = cols
  // A is mat_f32_dev, LDA = cols (K_blas)
  // B is vec_f32_dev, LDB = cols (K_blas)
  // C is out_f32_dev, LDC = rows (M_blas)
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                                     rows, 1, cols, 
                                     &alpha, 
                                     mat_f32_dev, cols, 
                                     vec_f32_dev, cols, 
                                     &beta, 
                                     out_f32_dev, rows);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS Sgemm failed in matvec_f32_f32_cuda: " +
                             std::to_string(status));
  }
  gpuErrchk(cudaGetLastError());
}

/**
 * @brief Matrix-vector multiplication with BF16 matrix and FP32 vector/output on GPU
 * 
 * This function performs matrix-vector multiplication (mat * vec = out) where
 * the matrix is in BF16 format, and the vector and output are in FP32 format.
 * It can use BF16 Tensor Cores via cublasGemmEx if supported and enabled.
 * Otherwise, it falls back to converting the BF16 matrix to FP32 and then
 * using cublasSgemm.
 * 
 * Matrix 'mat_bf16_dev' is assumed to be in row-major format with 'rows' rows and 'cols' columns.
 * Vector 'vec_f32_dev' is assumed to have 'cols' elements.
 * Output 'out_f32_dev' will have 'rows' elements.
 * 
 * @param handle cuBLAS handle
 * @param mat_bf16_dev BF16 matrix (device pointer, row-major, rows x cols)
 * @param vec_f32_dev FP32 vector (device pointer, cols elements)
 * @param out_f32_dev FP32 output vector (device pointer, rows elements)
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 * @param use_tensor_cores Whether to attempt using BF16 Tensor Cores with cublasGemmEx
 * @param stream CUDA stream (optional)
 */
void matvec_bf16_f32_cuda(cublasHandle_t handle,
                          const uint16_t* mat_bf16_dev,
                          const float* vec_f32_dev, float* out_f32_dev,
                          int rows, int cols,
                          bool use_tensor_cores, // New flag
                          cudaStream_t stream) {
  cublasStatus_t status;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // For C = A * B (row-major view) where A is mat_bf16_dev (rows x cols) and B is vec_f32_dev (cols x 1)
  // cuBLAS expects column-major. We use A^T for op(A) to simulate row-major A.
  // M_blas = rows (rows of C, rows of op(A))
  // N_blas = 1   (cols of C, cols of op(B))
  // K_blas = cols (cols of op(A), rows of op(B))
  //
  // If op(A) is CUBLAS_OP_T, then A is passed as K_blas x M_blas (cols x rows) column-major. LDA = K_blas = cols.
  // If op(B) is CUBLAS_OP_N, then B is passed as K_blas x N_blas (cols x 1) column-major. LDB = K_blas = cols.
  // C is M_blas x N_blas (rows x 1) column-major. LDC = M_blas = rows.

  // Disable Tensor Cores for matrix-vector operations (N=1) as they are not supported
  // by cublasGemmEx with mixed precision BF16->FP32 (confirmed by standalone test)
  if (use_tensor_cores) {
    Logger::info("[CUDA_KERNEL] matvec_bf16_f32_cuda: Tensor Cores requested, but disabling for matrix-vector (N=1) operation. Mixed precision BF16->FP32 with N=1 returns CUBLAS_STATUS_NOT_SUPPORTED in cublasGemmEx.");
    use_tensor_cores = false;
  }

  if (use_tensor_cores) {
    // Set the cuBLAS stream explicitly
    cublasStatus_t stream_status = cublasSetStream(handle, stream);
    if (stream_status != CUBLAS_STATUS_SUCCESS) {
      Logger::warning("[CUDA_KERNEL] cublasSetStream failed with status: " + std::to_string(stream_status) + ". Falling back to FP32 Sgemm.");
      use_tensor_cores = false;
    } else {
      // Logger::debug("[CUDA_KERNEL] matvec_bf16_f32_cuda attempting cublasGemmEx with Tensor Cores.");
      // Log all parameters for debugging
      Logger::info("[CUDA_KERNEL] cublasGemmEx parameters: M=" + std::to_string(rows) + 
                   ", N=1, K=" + std::to_string(cols) + 
                   ", LDA=" + std::to_string(cols) + 
                   ", LDB=" + std::to_string(cols) + 
                   ", LDC=" + std::to_string(rows) +
                   ", mat_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(mat_bf16_dev)) +
                   ", vec_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(vec_f32_dev)) +
                   ", out_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(out_f32_dev)));
      
      // A is mat_bf16_dev (CUDA_R_16BF)
      // B is vec_f32_dev (CUDA_R_32F)
      // C is out_f32_dev (CUDA_R_32F)
      // Computation type CUDA_R_32F
      status = cublasGemmEx(handle,
                            CUBLAS_OP_T,        // Transpose A (mat_bf16_dev)
                            CUBLAS_OP_N,        // No transpose B (vec_f32_dev)
                            rows,               // M
                            1,                  // N
                            cols,               // K
                            &alpha,             // Alpha
                            mat_bf16_dev,       // A
                            CUDA_R_16BF,        // Atype
                            cols,               // LDA (K_blas for A^T)
                            vec_f32_dev,        // B
                            CUDA_R_32F,         // Btype
                            cols,               // LDB (K_blas for B_NoTrans)
                            &beta,              // Beta
                            out_f32_dev,        // C
                            CUDA_R_32F,         // Ctype
                            rows,               // LDC (M_blas)
                            CUBLAS_COMPUTE_32F, // Compute type (use FP32 for accumulation for stability with BF16 inputs)
                                                // For pure BF16 compute (if B was also BF16), could use CUBLAS_COMPUTE_32F_FAST_16BF
                                                // Or CUBLAS_GEMM_DEFAULT_TENSOR_OP for algo to let cuBLAS pick
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP); // Algorithm
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
      Logger::warning("[CUDA_KERNEL] cublasGemmEx with Tensor Cores failed for matvec_bf16_f32_cuda: " + std::to_string(status) + ". Falling back to FP32 Sgemm.");
      // Fallback to FP32 path
      use_tensor_cores = false; // Ensure fallback path is taken
    } else {
      // Logger::debug("[CUDA_KERNEL] cublasGemmEx with Tensor Cores SUCCEEDED for matvec_bf16_f32_cuda.");
      gpuErrchk(cudaGetLastError());
      return; // Success with Tensor Cores
    }
  }

  // Fallback path: Convert BF16 matrix to FP32 and use cublasSgemm
  if (!use_tensor_cores) {
    // Logger::debug("[CUDA_KERNEL] matvec_bf16_f32_cuda using fallback: BF16->FP32 conversion then Sgemm.");
    // Set the cuBLAS stream for fallback path too
    cublasStatus_t stream_status = cublasSetStream(handle, stream);
    if (stream_status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublasSetStream failed in matvec_bf16_f32_cuda fallback: " + std::to_string(stream_status));
    }
    
    float* mat_fp32_dev = nullptr;
    size_t mat_size_elements = static_cast<size_t>(rows) * cols;
    gpuErrchk(cudaMalloc(&mat_fp32_dev, mat_size_elements * sizeof(float)));

    const int threads = 256;
    const int blocks = (mat_size_elements + threads - 1) / threads;
    convert_bf16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(mat_bf16_dev, mat_fp32_dev, mat_size_elements);
    gpuErrchk(cudaGetLastError());
    // No explicit sync needed here if stream is managed correctly by caller or if Sgemm uses the same stream.

    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           rows, 1, cols,
                           &alpha,
                           mat_fp32_dev, cols,   // A (mat_fp32_dev), LDA = K_blas = cols
                           vec_f32_dev, cols,    // B (vec_f32_dev), LDB = K_blas = cols
                           &beta,
                           out_f32_dev, rows);   // C (out_f32_dev), LDC = M_blas = rows
    
    gpuErrchk(cudaFree(mat_fp32_dev));
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS Sgemm failed in matvec_bf16_f32_cuda fallback: " + std::to_string(status));
    }
    gpuErrchk(cudaGetLastError());
  }
}

void matvec_f32_f32_cuda(cublasHandle_t handle,
                         const std::vector<float>& mat_f32_host,
                         const std::vector<float>& vec_f32_host,
                         std::vector<float>& out_f32_host, int rows, int cols) {
  if (mat_f32_host.size() != (size_t)rows * cols) {
    throw std::runtime_error(
        "matvec_f32_f32_cuda (host/host/host): mat size mismatch.");
  }
  if (vec_f32_host.size() != (size_t)cols) {
    throw std::runtime_error(
        "matvec_f32_f32_cuda (host/host/host): vec size mismatch.");
  }
  out_f32_host.resize(rows);

  float* mat_f32_dev = nullptr;
  float* vec_f32_dev = nullptr;
  float* out_f32_dev = nullptr;

  gpuErrchk(cudaMalloc(&mat_f32_dev, mat_f32_host.size() * sizeof(float)));
  gpuErrchk(cudaMalloc(&vec_f32_dev, vec_f32_host.size() * sizeof(float)));
  gpuErrchk(cudaMalloc(&out_f32_dev, out_f32_host.size() * sizeof(float)));

  gpuErrchk(cudaMemcpy(mat_f32_dev, mat_f32_host.data(),
                       mat_f32_host.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vec_f32_dev, vec_f32_host.data(),
                       vec_f32_host.size() * sizeof(float),
                       cudaMemcpyHostToDevice));

  matvec_f32_f32_cuda(handle, mat_f32_dev, vec_f32_dev, out_f32_dev, rows, cols,
                      0);

  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(out_f32_host.data(), out_f32_dev,
                       out_f32_host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(mat_f32_dev));
  gpuErrchk(cudaFree(vec_f32_dev));
  gpuErrchk(cudaFree(out_f32_dev));
}

__global__ void silu_kernel(const float* x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x_val = x[i];
    out[i] = x_val / (1.0f + expf(-x_val));
  }
}

void silu_cuda(const std::vector<float>& x_host, std::vector<float>& out_host,
               int n) {
  if (x_host.size() != n) {
    throw std::runtime_error("SiLU CUDA: Input vector size mismatch.");
  }
  out_host.resize(n);

  float* x_dev = nullptr;
  float* out_dev = nullptr;
  gpuErrchk(cudaMalloc(&x_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));

  gpuErrchk(cudaMemcpy(x_dev, x_host.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice));

  const int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  silu_kernel<<<num_blocks, threads_per_block>>>(x_dev, out_dev, n);
  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaFree(x_dev));
  gpuErrchk(cudaFree(out_dev));
}

__global__ void softmax_find_max_kernel(const float* x, float* partial_max,
                                        int n) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? x[i] : -INFINITY;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_max[blockIdx.x] = sdata[0];
  }
}

__global__ void softmax_exp_sum_kernel(const float* x, float* partial_sums,
                                       int n, float max_val) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? expf(x[i] - max_val) : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = sdata[0];
  }
}

__global__ void softmax_normalize_kernel(const float* x, float* out, int n,
                                         float max_val, float inv_sum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = expf(x[i] - max_val) * inv_sum;
  }
}

void softmax_vector_cuda(const std::vector<float>& x_host,
                         std::vector<float>& out_host, int n) {
  if (x_host.size() != n) {
    throw std::runtime_error("Softmax CUDA: Input vector size mismatch.");
  }
  if (n == 0) {
    out_host.clear();
    return;
  }
  out_host.resize(n);

  float* x_dev = nullptr;
  float* out_dev = nullptr;
  float* partial_max_dev = nullptr;
  float* partial_sum_dev = nullptr;

  const int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  size_t shared_mem_size = threads_per_block * sizeof(float);

  float* partial_max_host = new float[num_blocks];
  float* partial_sum_host = new float[num_blocks];

  gpuErrchk(cudaMalloc(&x_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&out_dev, n * sizeof(float)));
  gpuErrchk(cudaMalloc(&partial_max_dev, num_blocks * sizeof(float)));
  gpuErrchk(cudaMalloc(&partial_sum_dev, num_blocks * sizeof(float)));

  gpuErrchk(cudaMemcpy(x_dev, x_host.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice));

  softmax_find_max_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      x_dev, partial_max_dev, n);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaMemcpy(partial_max_host, partial_max_dev,
                       num_blocks * sizeof(float), cudaMemcpyDeviceToHost));

  float max_val = -INFINITY;
  for (int i = 0; i < num_blocks; ++i) {
    if (partial_max_host[i] > max_val) {
      max_val = partial_max_host[i];
    }
  }

  softmax_exp_sum_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
      x_dev, partial_sum_dev, n, max_val);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaMemcpy(partial_sum_host, partial_sum_dev,
                       num_blocks * sizeof(float), cudaMemcpyDeviceToHost));

  double exp_sum = 0.0;
  for (int i = 0; i < num_blocks; ++i) {
    exp_sum += partial_sum_host[i];
  }
  float inv_sum = 1.0f / static_cast<float>(exp_sum);

  softmax_normalize_kernel<<<num_blocks, threads_per_block>>>(x_dev, out_dev, n,
                                                              max_val, inv_sum);
  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaMemcpy(out_host.data(), out_dev, n * sizeof(float),
                       cudaMemcpyDeviceToHost));

  delete[] partial_max_host;
  delete[] partial_sum_host;
  gpuErrchk(cudaFree(x_dev));
  gpuErrchk(cudaFree(out_dev));
  gpuErrchk(cudaFree(partial_max_dev));
  gpuErrchk(cudaFree(partial_sum_dev));
}

__global__ void swiglu_kernel(const float* gate, const float* up, float* out,
                              int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float gate_val = gate[i];
    float silu_gate = gate_val / (1.0f + expf(-gate_val));
    out[i] = silu_gate * up[i];
  }
}

void swiglu_cuda(const float* gate_dev, const float* up_dev, float* out_dev,
                 int n, cudaStream_t stream) {
  const int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  swiglu_kernel<<<num_blocks, threads_per_block, 0, stream>>>(gate_dev, up_dev,
                                                              out_dev, n);
  gpuErrchk(cudaGetLastError());
}

// New Batched SwiGLU Kernel
__global__ void swiglu_batch_cuda_kernel(
    float* d_out_batch,              // Output: [num_tokens, intermediate_size]
    const float* d_gate_act_batch,   // Input: Gate activations [num_tokens, intermediate_size]
    const float* d_up_act_batch,     // Input: Up activations [num_tokens, intermediate_size]
    int num_tokens,
    int intermediate_size) {

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) {
        return;
    }

    // Base pointers for the current token
    const float* current_token_gate_ptr = d_gate_act_batch + (size_t)token_idx * intermediate_size;
    const float* current_token_up_ptr = d_up_act_batch + (size_t)token_idx * intermediate_size;
    float* current_token_out_ptr = d_out_batch + (size_t)token_idx * intermediate_size;

    // Threads in the block parallelize over intermediate_size
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        float gate_val = current_token_gate_ptr[i];
        float silu_gate = gate_val / (1.0f + expf(-gate_val)); // SiLU(x) = x * sigmoid(x)
        float up_val = current_token_up_ptr[i];
        current_token_out_ptr[i] = silu_gate * up_val;
    }
}

// Host wrapper for Batched SwiGLU
void swiglu_batch_cuda(
    float* d_out_batch,              // Output: [num_tokens, intermediate_size]
    const float* d_gate_act_batch,   // Input: Gate activations [num_tokens, intermediate_size]
    const float* d_up_act_batch,     // Input: Up activations [num_tokens, intermediate_size]
    int num_tokens,
    int intermediate_size,           // This is typically config_.ffn_hidden_size or similar
    cudaStream_t stream) {


    if (num_tokens == 0 || intermediate_size == 0) {
        Logger::info("[SWIGLU_BATCH_CUDA_SKIP] num_tokens or intermediate_size is 0. Nothing to do.");
        Logger::info("[SWIGLU_BATCH_CUDA_EXIT] Skipped operation.");
        return; // Nothing to do
    }

    // Log initial input values (first token, first few elements)
    int log_count_elements = std::min(3, intermediate_size);
    if (num_tokens > 0 && intermediate_size > 0) {
        if (d_gate_act_batch) {
            std::vector<float> h_gate_sample(log_count_elements);
            // Gate is [num_tokens, intermediate_size]. Offset for T0 is 0.
            gpuErrchk(cudaMemcpyAsync(h_gate_sample.data(), d_gate_act_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        if (d_up_act_batch) {
            std::vector<float> h_up_sample(log_count_elements);
            // Up is [num_tokens, intermediate_size]. Offset for T0 is 0.
            gpuErrchk(cudaMemcpyAsync(h_up_sample.data(), d_up_act_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }

    const int threads_per_block = 256;
    dim3 grid_dim(num_tokens);
    dim3 block_dim(threads_per_block);

    swiglu_batch_cuda_kernel<<<grid_dim, block_dim, 0, stream>>>(
        d_out_batch,
        d_gate_act_batch,
        d_up_act_batch,
        num_tokens,
        intermediate_size
    );
    gpuErrchk(cudaGetLastError());

    // Log output values
    if (num_tokens > 0 && intermediate_size > 0 && d_out_batch) {
        std::vector<float> h_out_sample(log_count_elements);
        // Out is [num_tokens, intermediate_size]. Offset for T0 is 0.
        gpuErrchk(cudaMemcpyAsync(h_out_sample.data(), d_out_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
}
__global__ void rope_kernel(float* x, int num_heads, int head_dim,
                            const float* all_freqs_cis_base, int pos, bool use_adjacent_pairing) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index for pairs
  int total_pairs = num_heads * (head_dim / 2);   // Total (dim_i, dim_{i+1}) or (dim_i, dim_{i+D/2}) pairs across all heads
  if (idx >= total_pairs) return;

  int head_idx = idx / (head_dim / 2);        // The specific head this thread works on
  int dim_pair_idx = idx % (head_dim / 2);  // The index of the pair within this head (0 to head_dim/2 - 1)

  size_t freq_cos_sin_pair_idx = (size_t)pos * (head_dim / 2) + dim_pair_idx;
  size_t freq_base_offset = freq_cos_sin_pair_idx * 2; 

  int base_x0, base_x1;
  if (use_adjacent_pairing) {
    // Adjacent pairing: Rotate x[h*HD + 2*j] with x[h*HD + 2*j + 1]
    base_x0 = head_idx * head_dim + (dim_pair_idx * 2);
    base_x1 = head_idx * head_dim + (dim_pair_idx * 2 + 1);
  } else {
    // Split-half pairing: Rotate x[h*HD + j] with x[h*HD + j + head_dim/2]
    base_x0 = head_idx * head_dim + dim_pair_idx;
    base_x1 = head_idx * head_dim + dim_pair_idx + (head_dim / 2);
  }

  float x0 = x[base_x0];
  float x1 = x[base_x1];
  
  float cos_val = all_freqs_cis_base[freq_base_offset];
  float sin_val = all_freqs_cis_base[freq_base_offset + 1];

  x[base_x0] = x0 * cos_val - x1 * sin_val;
  x[base_x1] = x0 * sin_val + x1 * cos_val;
}

void rope_cuda(float* x_dev, int num_heads, int head_dim,
               const float* all_freqs_cis_dev_base, int pos, bool use_adjacent_pairing, cudaStream_t stream) {
  int total_pairs = num_heads * (head_dim / 2);
  int threads_per_block = 256;
  int num_blocks = (total_pairs + threads_per_block - 1) / threads_per_block;

  rope_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      x_dev, num_heads, head_dim, all_freqs_cis_dev_base, pos, use_adjacent_pairing);
  gpuErrchk(cudaGetLastError());
}

// New Batched RoPE Kernel
__global__ void rope_batch_kernel_cuda(
    float* x_batch_tensor,          // Q_batch or K_batch tensor
    int num_tokens,                 // Number of tokens in this batch
    int num_heads_in_tensor,        // num_q_heads or num_kv_heads for this tensor
    int head_dim,                   // Dimension of each head
    const float* d_all_freqs_cis_base, // Precomputed cos/sin values table
    int start_pos_offset,           // Starting sequence position for the tokens in this batch
    bool use_adjacent_pairing) {

    // Grid: dim3 grid_dim(num_tokens, num_heads_in_tensor);
    // Block: dim3 block_dim(head_dim / 2);
    int pair_within_head_idx = threadIdx.x; // Thread index within the block (0 to head_dim/2 - 1)
    int head_idx = blockIdx.y;              // Block index in Y dimension maps to head index
    int token_idx = blockIdx.x;             // Block index in X dimension maps to token index within the batch

    // Boundary checks
    if (pair_within_head_idx >= (head_dim / 2) || 
        head_idx >= num_heads_in_tensor ||
        token_idx >= num_tokens) {
        return;
    }

    // Determine the actual sequence position for RoPE for the current token
    int current_token_seq_pos = start_pos_offset + token_idx;

    // Fetch cos and sin values from the precomputed table
    // d_all_freqs_cis_base is effectively [max_seq_len][head_dim/2][2]
    // Access: (pos * (head_dim/2) + pair_idx_in_head) * 2 for cos, and +1 for sin
    size_t freq_pair_offset_in_table = ((size_t)current_token_seq_pos * (head_dim / 2) + pair_within_head_idx) * 2;
    float cos_val = d_all_freqs_cis_base[freq_pair_offset_in_table];
    float sin_val = d_all_freqs_cis_base[freq_pair_offset_in_table + 1];

    // Calculate base pointer to the current token's data in x_batch_tensor
    // x_batch_tensor is [num_tokens][num_heads_in_tensor][head_dim]
    float* token_data_start = x_batch_tensor + (size_t)token_idx * num_heads_in_tensor * head_dim;
    // Pointer to the start of the current head's data for this token
    float* head_data_start = token_data_start + (size_t)head_idx * head_dim;

    int x0_idx_in_head, x1_idx_in_head;
    if (use_adjacent_pairing) {
        // Adjacent pairing: (x_0, x_1), (x_2, x_3), ... for a head
        // pair_within_head_idx = 0 maps to (x_0, x_1), pair_within_head_idx = 1 maps to (x_2, x_3)
        x0_idx_in_head = pair_within_head_idx * 2;
        x1_idx_in_head = pair_within_head_idx * 2 + 1;
    } else {
        // Split-half pairing: (x_0, x_{D/2}), (x_1, x_{D/2+1}), ... for a head
        // pair_within_head_idx = 0 maps to (x_0, x_{D/2}), pair_within_head_idx = 1 maps to (x_1, x_{D/2+1})
        x0_idx_in_head = pair_within_head_idx;
        x1_idx_in_head = pair_within_head_idx + (head_dim / 2);
    }

    float x0_val = head_data_start[x0_idx_in_head];
    float x1_val = head_data_start[x1_idx_in_head];


    head_data_start[x0_idx_in_head] = x0_val * cos_val - x1_val * sin_val;
    head_data_start[x1_idx_in_head] = x0_val * sin_val + x1_val * cos_val;
}

// Host wrapper for batched RoPE application
void rope_batch_cuda(float* d_q_batch, float* d_k_batch,
                     const float* d_all_freqs_cis_base, // Moved to 3rd position
                     int num_tokens, int num_q_heads, int num_kv_heads, int head_dim,
                     int start_pos_in_kv_cache, 
                     bool use_adjacent_pairing,
                     cudaStream_t stream) {


    if (head_dim == 0) { 
        Logger::warning("[ROPE_BATCH_CUDA_SKIP] head_dim is 0. Skipping RoPE application.");
        return;
    }
    if (head_dim % 2 != 0) {
        Logger::error("rope_batch_cuda: head_dim must be even. Got " + std::to_string(head_dim));
        Logger::info("[ROPE_BATCH_CUDA_EXIT] Error: head_dim not even.");
        return; 
    }
    if (num_tokens == 0) {
        Logger::info("[ROPE_BATCH_CUDA_EXIT] num_tokens is 0. Nothing to process.");
        return; // Nothing to process
    }

    int log_count_elements = std::min(3, head_dim);

    // Log initial Q values
    if (d_q_batch && num_q_heads > 0 && head_dim > 0) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens, 2); ++token_to_log_idx) {
            std::vector<float> h_q_sample_before(log_count_elements);
            size_t q_offset_elements = (size_t)token_to_log_idx * num_q_heads * head_dim;
            
            gpuErrchk(cudaMemcpyAsync(h_q_sample_before.data(), 
                                      d_q_batch + q_offset_elements, 
                                      log_count_elements * sizeof(float), 
                                      cudaMemcpyDeviceToHost, stream));
        }
    }

    // Log initial K values
    if (d_k_batch && num_kv_heads > 0 && head_dim > 0) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens, 2); ++token_to_log_idx) {
            std::vector<float> h_k_sample_before(log_count_elements);
            size_t k_offset_elements = (size_t)token_to_log_idx * num_kv_heads * head_dim;

            gpuErrchk(cudaMemcpyAsync(h_k_sample_before.data(), 
                                      d_k_batch + k_offset_elements, 
                                      log_count_elements * sizeof(float), 
                                      cudaMemcpyDeviceToHost, stream));
        }
    }

    dim3 block_dim(head_dim / 2);

    // Apply RoPE to Q batch
    if (d_q_batch && num_q_heads > 0) {
        dim3 grid_dim_q(num_tokens, num_q_heads);
        rope_batch_kernel_cuda<<<grid_dim_q, block_dim, 0, stream>>>(
            d_q_batch,
            num_tokens,
            num_q_heads,
            head_dim,
            d_all_freqs_cis_base,
            start_pos_in_kv_cache,
            use_adjacent_pairing
        );
        gpuErrchk(cudaGetLastError()); 
    }

    // Apply RoPE to K batch
    if (d_k_batch && num_kv_heads > 0) {
        dim3 grid_dim_k(num_tokens, num_kv_heads); 
        rope_batch_kernel_cuda<<<grid_dim_k, block_dim, 0, stream>>>(
            d_k_batch,
            num_tokens,
            num_kv_heads, 
            head_dim,
            d_all_freqs_cis_base,
            start_pos_in_kv_cache,
            use_adjacent_pairing
        );
        gpuErrchk(cudaGetLastError()); 
    }

    // Log Q values after RoPE
    if (d_q_batch && num_q_heads > 0 && head_dim > 0) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens, 2); ++token_to_log_idx) {
            std::vector<float> h_q_sample_after(log_count_elements);
            size_t q_offset_elements = (size_t)token_to_log_idx * num_q_heads * head_dim;
            
            gpuErrchk(cudaMemcpyAsync(h_q_sample_after.data(), 
                                      d_q_batch + q_offset_elements, 
                                      log_count_elements * sizeof(float), 
                                      cudaMemcpyDeviceToHost, stream));
        }
    }

    // Log K values after RoPE
    if (d_k_batch && num_kv_heads > 0 && head_dim > 0) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens, 2); ++token_to_log_idx) {
            std::vector<float> h_k_sample_after(log_count_elements);
            size_t k_offset_elements = (size_t)token_to_log_idx * num_kv_heads * head_dim;

            gpuErrchk(cudaMemcpyAsync(h_k_sample_after.data(), 
                                      d_k_batch + k_offset_elements, 
                                      log_count_elements * sizeof(float), 
                                      cudaMemcpyDeviceToHost, stream));
        }
    }
}

__global__ void attention_kernel(const float* Q_current,
                                 const float* K_layer_cache_base,
                                 const float* V_layer_cache_base, float* out,
                                 int current_seq_len, int head_dim, float scale,
                                 int cache_num_kv_heads, int num_q_heads) {


  int q_head_idx = blockIdx.x;
  int d_idx = threadIdx.x; // dimension index for this thread

  if (q_head_idx >= num_q_heads) return; // Should not happen with correct launch

  // Determine the corresponding KV head for this Q head (for GQA/MQA)
  int heads_per_kv = num_q_heads / cache_num_kv_heads;
  int kv_head_idx = q_head_idx / heads_per_kv;

  // Shared memory for scores (one score per k_pos) and for dot product reduction
  extern __shared__ float shared_data[];
  float* scores = shared_data; // First part of shared_data: size current_seq_len
  float* dot_product_terms = &shared_data[current_seq_len]; // Second part: size head_dim (blockDim.x)

  // Pointer to the current query head's data
  const float* q_head_ptr = Q_current + q_head_idx * head_dim;

  for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) {
    // Pointer to the k_pos-th key vector for the relevant kv_head
    size_t k_vec_offset = (size_t)k_pos * cache_num_kv_heads * head_dim +
                          (size_t)kv_head_idx * head_dim;
    const float* k_vec_ptr = K_layer_cache_base + k_vec_offset;

    dot_product_terms[d_idx] = q_head_ptr[d_idx] * k_vec_ptr[d_idx];
    __syncthreads(); // Ensure all terms are written before reduction

    // Reduction in shared memory (e.g., by thread 0 or tree-based)
    if (d_idx == 0) {
      float current_dot = 0.0f;
      for (int r = 0; r < head_dim; ++r) {
        current_dot += dot_product_terms[r];
      }
      scores[k_pos] = current_dot * scale;
    }
    __syncthreads(); 
  }

  float thread_max_score = -INFINITY;
  for (int i = d_idx; i < current_seq_len; i += blockDim.x) { // blockDim.x is head_dim
    if (scores[i] > thread_max_score) {
      thread_max_score = scores[i];
    }
  }
  dot_product_terms[d_idx] = thread_max_score; // Repurpose dot_product_terms shared memory
  __syncthreads();

  float block_max_score = -INFINITY;
  if (d_idx == 0) {
    for (int r = 0; r < head_dim; ++r) { // head_dim might be > current_seq_len for first few tokens
      if (dot_product_terms[r] > block_max_score) { // Check if dot_product_terms[r] was validly written
        block_max_score = dot_product_terms[r];
      }
    }
  }
  __syncthreads();
  if(d_idx == 0) dot_product_terms[0] = block_max_score; // Store it.
  __syncthreads();
  block_max_score = dot_product_terms[0]; // All threads read it.


  // Calculate exp scores and sum
  float thread_exp_sum = 0.0f;
  for (int i = d_idx; i < current_seq_len; i += blockDim.x) {
    float prob = expf(scores[i] - block_max_score);
    scores[i] = prob; // Update scores in-place with exp(val - max)
    thread_exp_sum += prob;
  }
  // Reduce thread_exp_sum across the block
  dot_product_terms[d_idx] = thread_exp_sum; // Repurpose again
  __syncthreads();

  float block_exp_sum = 0.0f;
  if (d_idx == 0) {
    for (int r = 0; r < head_dim; ++r) { // Similar concern as max_score reduction range
      block_exp_sum += dot_product_terms[r];
    }
  }
  __syncthreads();
  // Store and reload block_exp_sum for all threads
  if(d_idx == 0) dot_product_terms[0] = block_exp_sum;
  __syncthreads();
  block_exp_sum = dot_product_terms[0];


  float inv_sum = 1.0f / (block_exp_sum + 1e-9f); // Add epsilon for stability

  // Normalize scores in shared memory
  for (int i = d_idx; i < current_seq_len; i += blockDim.x) {
    scores[i] *= inv_sum;
  }
  __syncthreads(); // Ensure all scores are normalized before weighted sum

  double weighted_val_d = 0.0; // Use double for accumulation
  for (int k_pos = 0; k_pos < current_seq_len; ++k_pos) {
    size_t v_vec_offset = (size_t)k_pos * cache_num_kv_heads * head_dim +
                          (size_t)kv_head_idx * head_dim;
    const float* v_vec_ptr = V_layer_cache_base + v_vec_offset;
    
    weighted_val_d += static_cast<double>(scores[k_pos]) * static_cast<double>(v_vec_ptr[d_idx]);
  }
  
  // Write the final result for this thread's dimension d_idx
  out[q_head_idx * head_dim + d_idx] = static_cast<float>(weighted_val_d);
}

void attention_cuda(const float* Q_current_dev, const float* K_layer_cache_base,
                    const float* V_layer_cache_base, float* out_dev,
                    int num_heads, int current_seq_len, int head_dim,
                    float scale, int cache_max_seq_len, int cache_num_kv_heads,
                    cudaStream_t stream) {
  // Grid: num_heads blocks, each block handles one attention head
  // Block: head_dim threads, each thread handles one output dimension
  dim3 grid_dim(num_heads);
  dim3 block_dim(head_dim);
  // scores[current_seq_len] + dot_product_terms[head_dim] (the kernel's extern shared layout)
  const size_t shm = (size_t)(current_seq_len + head_dim) * sizeof(float);

  attention_kernel<<<grid_dim, block_dim, shm, stream>>>(
      Q_current_dev, K_layer_cache_base, V_layer_cache_base, out_dev,
      current_seq_len, head_dim, scale, cache_num_kv_heads, num_heads);
  gpuErrchk(cudaGetLastError());
}

// Selective dequantization attention implementation
__global__ void attention_selective_dequant_kernel(
    const float* __restrict__ Q_current_dev,
    const int8_t* __restrict__ K_quantized_cache_base,
    const int8_t* __restrict__ V_quantized_cache_base,
    const float* __restrict__ K_scales_cache_base,
    const float* __restrict__ V_scales_cache_base,
    float* __restrict__ selective_k_dequant_buffer,
    float* __restrict__ selective_v_dequant_buffer,
    float* __restrict__ out_dev,
    int num_heads, int current_seq_len, int head_dim,
    float scale, int cache_max_seq_len, int cache_num_kv_heads) {
    
    int head_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (head_idx >= num_heads || dim_idx >= head_dim) return;
    
    // Map query head to corresponding KV head (for GQA/MQA)
    int kv_heads_per_q_group = num_heads / cache_num_kv_heads;
    int kv_head_idx = head_idx / kv_heads_per_q_group;
    
    // Get current query vector for this head and dimension
    float q_val = Q_current_dev[head_idx * head_dim + dim_idx];
    
    float attn_out = 0.0f;
    
    // Selective dequantization and attention computation
    for (int t = 0; t < current_seq_len; ++t) {
        // Calculate offset for this token and KV head in the quantized cache
        size_t cache_offset_base = (size_t)t * cache_num_kv_heads * head_dim + (size_t)kv_head_idx * head_dim;
        size_t scale_offset = (size_t)t * cache_num_kv_heads + kv_head_idx;
        
        // Dequantize only the specific elements we need for this head/dimension
        if (dim_idx == 0) {
            // Only thread 0 in each block dequantizes the entire head for this token
            float k_scale = K_scales_cache_base[scale_offset];
            float v_scale = V_scales_cache_base[scale_offset];
            
            for (int d = 0; d < head_dim; ++d) {
                // Dequantize K values for this token/head
                int8_t k_quant = K_quantized_cache_base[cache_offset_base + d];
                selective_k_dequant_buffer[t * head_dim + d] = k_scale * (float)k_quant;
                
                // Dequantize V values for this token/head  
                int8_t v_quant = V_quantized_cache_base[cache_offset_base + d];
                selective_v_dequant_buffer[t * head_dim + d] = v_scale * (float)v_quant;
            }
        }
        
        __syncthreads(); // Ensure dequantization is complete before using values
        
        // Compute attention score for this token
        float k_val = selective_k_dequant_buffer[t * head_dim + dim_idx];
        float score = q_val * k_val * scale;
        
        // Apply attention score to value
        float v_val = selective_v_dequant_buffer[t * head_dim + dim_idx];
        attn_out += score * v_val;
        
        __syncthreads(); // Ensure all threads are done before next iteration
    }
    
    // Write output
    out_dev[head_idx * head_dim + dim_idx] = attn_out;
}

void attention_cuda_selective_dequant(const float* Q_current_dev, 
                                     const int8_t* K_quantized_cache_base,
                                     const int8_t* V_quantized_cache_base,
                                     const float* K_scales_cache_base,
                                     const float* V_scales_cache_base,
                                     float* selective_k_dequant_buffer,
                                     float* selective_v_dequant_buffer,
                                     float* out_dev,
                                     int num_heads, int current_seq_len, int head_dim,
                                     float scale, int cache_max_seq_len, int cache_num_kv_heads,
                                     cudaStream_t stream) {
    
    if (num_heads == 0 || current_seq_len == 0 || head_dim == 0) {
        Logger::warning("[ATTENTION_SELECTIVE_DEQUANT] Invalid parameters: num_heads=" + 
                       std::to_string(num_heads) + ", current_seq_len=" + std::to_string(current_seq_len) + 
                       ", head_dim=" + std::to_string(head_dim));
        return;
    }
    
    // Grid: num_heads blocks, each block handles one attention head
    // Block: head_dim threads, each thread handles one output dimension  
    dim3 grid_dim(num_heads);
    dim3 block_dim(head_dim);
    
    attention_selective_dequant_kernel<<<grid_dim, block_dim, 0, stream>>>(
        Q_current_dev, K_quantized_cache_base, V_quantized_cache_base,
        K_scales_cache_base, V_scales_cache_base,
        selective_k_dequant_buffer, selective_v_dequant_buffer,
        out_dev, num_heads, current_seq_len, head_dim, scale,
        cache_max_seq_len, cache_num_kv_heads);
    
    gpuErrchk(cudaGetLastError());
}

__global__ void add_vectors_kernel(const float* a, const float* b,
                                   float* result, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    result[i] = a[i] + b[i];
  }
}

void add_vectors_cuda(const float* a_dev, const float* b_dev, float* result_dev,
                      int n, cudaStream_t stream) {
  const int threads_per_block = 256;
  const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

  add_vectors_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      a_dev, b_dev, result_dev, n);
  gpuErrchk(cudaGetLastError());
}

__global__ void add_residual_kernel(const float* matvec_out,
                                    const float* residual, float* result,
                                    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    result[i] = matvec_out[i] + residual[i];
  }
}

void add_residual_cuda(const float* matvec_out_dev, const float* residual_dev,
                       float* result_dev, int n, cudaStream_t stream) {
  const int threads_per_block = 256;
  const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

  add_residual_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      matvec_out_dev, residual_dev, result_dev, n);
  gpuErrchk(cudaGetLastError());
}

// New Batched Add Residual Kernel
__global__ void add_residual_batch_cuda_kernel(
    float* d_output_batch,          // Output: [num_tokens, hidden_size]
    const float* d_input_a_batch,   // Input A: [num_tokens, hidden_size] (e.g., sub-layer output)
    const float* d_input_b_batch,   // Input B: [num_tokens, hidden_size] (e.g., original input to sub-layer)
    int num_tokens,
    int hidden_size) {

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) {
        return;
    }

    // Base pointers for the current token
    const float* current_token_input_a_ptr = d_input_a_batch + (size_t)token_idx * hidden_size;
    const float* current_token_input_b_ptr = d_input_b_batch + (size_t)token_idx * hidden_size;
    float* current_token_output_ptr = d_output_batch + (size_t)token_idx * hidden_size;

    // Threads in the block parallelize over hidden_size
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        current_token_output_ptr[i] = current_token_input_a_ptr[i] + current_token_input_b_ptr[i];
    }
}

// Host wrapper for Batched Add Residual
void add_residual_batch_cuda(
    float* d_output_batch,          // Output: [num_tokens, hidden_size]
    const float* d_input_a_batch,   // Input A: [num_tokens, hidden_size]
    const float* d_input_b_batch,   // Input B: [num_tokens, hidden_size]
    int num_tokens,
    int hidden_size,
    cudaStream_t stream) {


    if (num_tokens == 0 || hidden_size == 0) {
        Logger::info("[ADD_RESIDUAL_BATCH_CUDA_SKIP] num_tokens or hidden_size is 0. Nothing to do.");
        Logger::info("[ADD_RESIDUAL_BATCH_CUDA_EXIT] Skipped operation.");
        return; // Nothing to do
    }

    // Log initial input values (first token, first few elements)
    int log_count_elements = std::min(3, hidden_size);
    if (num_tokens > 0 && hidden_size > 0) {
        if (d_input_a_batch) {
            std::vector<float> h_input_a_sample(log_count_elements);
            // Input A is [num_tokens, hidden_size]. Offset for T0 is 0.
            gpuErrchk(cudaMemcpyAsync(h_input_a_sample.data(), d_input_a_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        if (d_input_b_batch) {
            std::vector<float> h_input_b_sample(log_count_elements);
            // Input B is [num_tokens, hidden_size]. Offset for T0 is 0.
            gpuErrchk(cudaMemcpyAsync(h_input_b_sample.data(), d_input_b_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }

    const int threads_per_block = 256; 
    dim3 grid_dim(num_tokens);
    dim3 block_dim(threads_per_block);

    add_residual_batch_cuda_kernel<<<grid_dim, block_dim, 0, stream>>>(
        d_output_batch,
        d_input_a_batch,
        d_input_b_batch,
        num_tokens,
        hidden_size
    );
    gpuErrchk(cudaGetLastError());

    // Log output values
    if (num_tokens > 0 && hidden_size > 0 && d_output_batch) {
        std::vector<float> h_output_sample(log_count_elements);
        // Output is [num_tokens, hidden_size]. Offset for T0 is 0.
        gpuErrchk(cudaMemcpyAsync(h_output_sample.data(), d_output_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
}
__global__ void update_kv_cache_kernel(float* cache_base_ptr,
                                       const float* current_kv_vector, int pos,
                                       int kv_head_idx, int max_seq_len,
                                       int num_kv_heads, int head_dim) {
  int d = threadIdx.x;
  if (d >= head_dim) return;

  size_t cache_offset = (size_t)pos * num_kv_heads * head_dim +
                        (size_t)kv_head_idx * head_dim + d;

  size_t source_offset = d;

  size_t total_cache_size = (size_t)max_seq_len * num_kv_heads * head_dim;
  if (cache_offset >= total_cache_size) {
    return;
  }

  cache_base_ptr[cache_offset] = current_kv_vector[source_offset];
}

void update_kv_cache_cuda(float* cache_base_ptr,
                          const float* current_kv_head_vector,

                          int pos, int kv_head_idx, int max_seq_len,
                          int num_kv_heads, int head_dim, cudaStream_t stream) {
  dim3 blockDim(head_dim);
  dim3 gridDim(1);

  if (pos < 0 || pos >= max_seq_len) {
    Logger::error("update_kv_cache_cuda: pos out of bounds (" +
                  std::to_string(pos) + " >= " + std::to_string(max_seq_len) +
                  ")");
    return;
  }

  if (kv_head_idx < 0 || kv_head_idx >= num_kv_heads) {
    Logger::error("update_kv_cache_cuda: kv_head_idx out of bounds (" +
                  std::to_string(kv_head_idx) +
                  " >= " + std::to_string(num_kv_heads) + ")");
    return;
  }
  if (!current_kv_head_vector) {
    Logger::error("update_kv_cache_cuda: Input K/V vector pointer is null.");
    return;
  }

  update_kv_cache_kernel<<<gridDim, blockDim, 0, stream>>>(
      cache_base_ptr, current_kv_head_vector, pos, kv_head_idx, max_seq_len,
      num_kv_heads, head_dim);
  gpuErrchk(cudaGetLastError());
}

__global__ void rope_and_update_kv_cache_kernel(
    float* cache_base_ptr, const float* kv_vector_head,

    const float* all_freqs_cis_base, int pos, int kv_head_idx, int max_seq_len,
    int num_kv_heads, int head_dim) {
  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int dim_half = head_dim / 2;

  if (pair_idx >= dim_half) return;

  int idx0 = pair_idx;
  int idx1 = pair_idx + dim_half;
  float kv0 = kv_vector_head[idx0];
  float kv1 = kv_vector_head[idx1];

  size_t freq_base_offset = (size_t)pos * head_dim + (size_t)pair_idx * 2;
  float cos_val = all_freqs_cis_base[freq_base_offset];
  float sin_val = all_freqs_cis_base[freq_base_offset + 1];

  float kv0_rotated = kv0 * cos_val - kv1 * sin_val;
  float kv1_rotated = kv0 * sin_val + kv1 * cos_val;

  size_t cache_offset_0 = (size_t)pos * num_kv_heads * head_dim +
                          (size_t)kv_head_idx * head_dim + idx0;
  size_t cache_offset_1 = cache_offset_0 + dim_half;

  size_t total_cache_size = (size_t)max_seq_len * num_kv_heads * head_dim;
  if (cache_offset_0 >= total_cache_size ||
      cache_offset_1 >= total_cache_size) {
    return;
  }

  cache_base_ptr[cache_offset_0] = kv0_rotated;
  cache_base_ptr[cache_offset_1] = kv1_rotated;
}

void rope_and_update_kv_cache_cuda(float* cache_base_ptr,
                                   const float* kv_vector_head,
                                   const float* all_freqs_cis_base, int pos,
                                   int kv_head_idx, int max_seq_len,
                                   int num_kv_heads, int head_dim,
                                   cudaStream_t stream) {
  if (head_dim % 2 != 0) {
    Logger::error("rope_and_update_kv_cache_cuda: head_dim must be even.");
    return;
  }
  if (pos < 0 || pos >= max_seq_len) {
    Logger::error("rope_and_update_kv_cache_cuda: pos out of bounds.");
    return;
  }
  if (kv_head_idx < 0 || kv_head_idx >= num_kv_heads) {
    Logger::error("rope_and_update_kv_cache_cuda: kv_head_idx out of bounds.");
    return;
  }
  if (!kv_vector_head || !all_freqs_cis_base || !cache_base_ptr) {
    Logger::error(
        "rope_and_update_kv_cache_cuda: Received null device pointer(s).");
    return;
  }

  int threads_per_block = 128;
  int num_blocks = (head_dim / 2 + threads_per_block - 1) / threads_per_block;

  rope_and_update_kv_cache_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      cache_base_ptr, kv_vector_head, all_freqs_cis_base, pos, kv_head_idx,
      max_seq_len, num_kv_heads, head_dim);
  gpuErrchk(cudaGetLastError());
}

__global__ void lookup_embedding_kernel(const void* __restrict__ table_dev,
                                        float* __restrict__ output_dev,
                                        int token_id, int hidden_size,
                                        int vocab_size, bool is_bf16) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= hidden_size) {
    return;
  }

  if (token_id < 0 || token_id >= vocab_size) {
    output_dev[idx] = 0.0f;
    return;
  }
  size_t offset = (size_t)token_id * hidden_size + idx;

  if (is_bf16) {
    const uint16_t* table_bf16 = static_cast<const uint16_t*>(table_dev);

    uint16_t val_bf16 = table_bf16[offset];

    output_dev[idx] = bf16_to_float32_device(val_bf16);
  } else {
    const float* table_f32 = static_cast<const float*>(table_dev);

    output_dev[idx] = table_f32[offset];
  }
}

void lookup_embedding_cuda(const void* table_dev, float* output_dev,
                           int token_id, int hidden_size, int vocab_size,
                           bool is_bf16, cudaStream_t stream) {
  if (!table_dev || !output_dev) {
    Logger::error("lookup_embedding_cuda: Received null device pointer(s).");

    return;
  }
  if (hidden_size <= 0 || vocab_size <= 0) {
    Logger::error("lookup_embedding_cuda: Invalid hidden_size or vocab_size.");
    return;
  }

  int threads_per_block = 256;

  int blocks_per_grid =
      (hidden_size + threads_per_block - 1) / threads_per_block;

  lookup_embedding_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
      table_dev, output_dev, token_id, hidden_size, vocab_size, is_bf16);

  gpuErrchk(cudaGetLastError());
}

__global__ void convert_bf16_to_fp32_kernel(const uint16_t* __restrict__ bf16_in,
                                            float* __restrict__ fp32_out,
                                            size_t n_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_elements) {
    fp32_out[idx] = bf16_to_float32_device(bf16_in[idx]);
  }
}

// KVCache Quantization Kernels (FP32 <-> INT8)
#ifdef HAS_CUDA

// Kernel to find the maximum absolute value in a float array (for per-tensor scaling)
__global__ void find_max_abs_kernel(const float* __restrict__ data, int n_elements, float* __restrict__ max_abs_val) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_max = 0.0f;
    if (i < n_elements) {
        local_max = fabsf(data[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float old_val = *max_abs_val;
        float new_val = sdata[0];
        while (new_val > old_val) {
            float assumed_old = old_val;
            old_val = atomicCAS((unsigned int*)max_abs_val, 
                                __float_as_int(assumed_old), 
                                __float_as_int(new_val));
            // If atomicCAS did not return what we thought was the old value,
            // it means another thread updated it. We re-read and retry if our new_val is still greater.
            if (__int_as_float(old_val) != assumed_old) {
                 // Re-check condition with the actual current max_abs_val (now in old_val after CAS)
                if (new_val <= old_val) break; // Another thread set a higher or equal max, or our new_val is no longer the max
            } else {
                // Our CAS was successful, or new_val was not greater than the initial read.
                break; 
            }
        }
    }
}

// Kernel to quantize FP32 to INT8 using a symmetric per-tensor scale
__global__ void quantize_fp32_to_int8_symmetric_kernel(
    const float* __restrict__ fp32_in,
    int8_t* __restrict__ int8_out,
    float scale,
    int n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        float val_fp32 = fp32_in[i];
        // Symmetric quantization: scaled_value = round(value / scale)
        // Clamp to [-127, 127] for INT8 (as -128 might not be used or handled differently by some dequant schemes)
        float scaled_val = roundf(val_fp32 / scale);
        int8_out[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, scaled_val));
    }
}

// Kernel to dequantize INT8 to FP32 using a symmetric per-tensor scale
__global__ void dequantize_int8_to_fp32_symmetric_kernel(
    const int8_t* __restrict__ int8_in,
    float scale,
    float* __restrict__ fp32_out,
    int n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        fp32_out[i] = (float)int8_in[i] * scale;
    }
}

void quantize_fp32_to_int8_symmetric_per_tensor_cuda(
    const float* fp32_in_dev, 
    int8_t* int8_out_dev, 
    float* scale_out_dev, // Device pointer to store the single scale value
    int num_elements, 
    cudaStream_t stream
) {
    if (num_elements == 0) return;

    const int threads_per_block = 256;
    
    // --- Step 1: Find the maximum absolute value for scaling ---
    float* d_max_abs_val_accumulator = nullptr; // Accumulator on device for max_abs from blocks
    gpuErrchk(cudaMalloc(&d_max_abs_val_accumulator, sizeof(float)));
    gpuErrchk(cudaMemsetAsync(d_max_abs_val_accumulator, 0, sizeof(float), stream)); // Initialize to 0

    // Max reduction kernel requires shared memory proportional to block size
    int num_blocks_reduce = (num_elements + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    find_max_abs_kernel<<<num_blocks_reduce, threads_per_block, shared_mem_size, stream>>>(
        fp32_in_dev, num_elements, d_max_abs_val_accumulator);
    gpuErrchk(cudaGetLastError());

    // Copy the final max absolute value back to `scale_out_dev` which is the designated output for the scale.
    // The actual scale factor will be calculated from this max_abs_val.
    float h_max_abs_val = 0.0f;
    gpuErrchk(cudaMemcpyAsync(&h_max_abs_val, d_max_abs_val_accumulator, sizeof(float), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream)); // Ensure h_max_abs_val is ready

    gpuErrchk(cudaFree(d_max_abs_val_accumulator));

    // Calculate scale: scale = max_abs / 127.0 (so that 127 maps to max_abs)
    // Add a small epsilon to prevent division by zero if all values are zero.
    float scale = (h_max_abs_val > 1e-8) ? (h_max_abs_val / 127.0f) : 1.0f; 
    gpuErrchk(cudaMemcpyAsync(scale_out_dev, &scale, sizeof(float), cudaMemcpyHostToDevice, stream));

    // --- Step 2: Quantize using the calculated scale ---
    int num_blocks_quantize = (num_elements + threads_per_block - 1) / threads_per_block;
    quantize_fp32_to_int8_symmetric_kernel<<<num_blocks_quantize, threads_per_block, 0, stream>>>(
        fp32_in_dev, int8_out_dev, scale, num_elements);
    gpuErrchk(cudaGetLastError());
}

void dequantize_int8_to_fp32_symmetric_per_tensor_cuda(
    const int8_t* int8_in_dev, 
    const float* scale_in_dev, // Device pointer to the single scale value
    float* fp32_out_dev,
    int num_elements, 
    cudaStream_t stream
) {
    if (num_elements == 0) return;

    // Copy scale from device to host to pass as kernel argument
    // (Kernels typically take scalar arguments by value)
    float h_scale;
    gpuErrchk(cudaMemcpyAsync(&h_scale, scale_in_dev, sizeof(float), cudaMemcpyDeviceToHost, stream));
    // We need to synchronize here to ensure h_scale is available before launching the kernel
    // A more optimized approach might involve a device-side broadcast of the scale if it were part of a larger struct.
    gpuErrchk(cudaStreamSynchronize(stream)); 

    const int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    dequantize_int8_to_fp32_symmetric_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        int8_in_dev, h_scale, fp32_out_dev, num_elements);
    gpuErrchk(cudaGetLastError());
}

// New Generic GEMM for FP32
void gemm_f32_f32_cuda(cublasHandle_t handle, 
                       bool transa_user, bool transb_user, 
                       int m_user, int n_user, int k_user, 
                       const float* alpha_user, 
                       const float* A_user, int lda_user, 
                       const float* B_user, int ldb_user, 
                       const float* beta_user, 
                       float* C_user, int ldc_user, 
                       cudaStream_t stream) {
    
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSetStream failed in gemm_f32_f32_cuda with error: " + std::to_string(status) + " (" + cublasGetStatusString(status) + ")");
        throw std::runtime_error("cublasSetStream failed");
    }

    cublasOperation_t opA_cublas, opB_cublas;
    int M_cublas, N_cublas, K_cublas;
    const float *A_cublas_ptr, *B_cublas_ptr;
    int LDA_cublas, LDB_cublas;


    M_cublas = n_user;
    N_cublas = m_user;
    K_cublas = k_user;

    A_cublas_ptr = B_user;    // B_user data is the first matrix for cuBLAS call
    LDA_cublas = (transb_user) ? k_user : ldb_user;

    B_cublas_ptr = A_user;    // A_user data is the second matrix for cuBLAS call
    LDB_cublas = (transa_user) ? m_user : lda_user;

    // Determine cuBLAS operations for A_cublas_ptr (B_user) and B_cublas_ptr (A_user)
    // opA_cublas applies to B_user, opB_cublas applies to A_user.
    if (!transa_user && !transb_user) {
        // User wants C = A * B
        opA_cublas = CUBLAS_OP_N; // for B_user
        opB_cublas = CUBLAS_OP_N; // for A_user
    } else if (transa_user && !transb_user) {
        // User wants C = A^T * B
        opA_cublas = CUBLAS_OP_N; // for B_user
        opB_cublas = CUBLAS_OP_T; // for A_user (A^T)
    } else if (!transa_user && transb_user) {
        // User wants C = A * B^T
        opA_cublas = CUBLAS_OP_T; // for B_user (B^T)
        opB_cublas = CUBLAS_OP_N; // for A_user
    } else { // transa_user && transb_user
        // User wants C = A^T * B^T
        opA_cublas = CUBLAS_OP_T; // for B_user (B^T)
        opB_cublas = CUBLAS_OP_T; // for A_user (A^T)
    }
    

    // For swapped computation: C^T result is (n_user x m_user) in column-major
    int LDC_cublas = ldc_user; // Leading dimension of C (which is n_user for row-major C(m,n))
    
    status = cublasSgemm(handle, opA_cublas, opB_cublas, 
                           M_cublas, N_cublas, K_cublas, 
                           alpha_user, 
                           A_cublas_ptr, LDA_cublas, 
                           B_cublas_ptr, LDB_cublas, 
                           beta_user, 
                           C_user, LDC_cublas); // FIXED: Use LDC_cublas instead of ldc_user

    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSgemm failed in gemm_f32_f32_cuda with error: " + std::to_string(status) + " (" + cublasGetStatusString(status) + ")");
        Logger::error("GEMM params (original user view): transa_user=" + std::to_string(transa_user) + " transb_user=" + std::to_string(transb_user) + 
                       " m=" + std::to_string(m_user) + " n=" + std::to_string(n_user) + " k=" + std::to_string(k_user) + 
                       " lda=" + std::to_string(lda_user) + " ldb=" + std::to_string(ldb_user) + " ldc=" + std::to_string(ldc_user));
        Logger::error("GEMM params (internal cublas view at failure): transA=" + std::to_string(opA_cublas==CUBLAS_OP_T) + " transB=" + std::to_string(opB_cublas==CUBLAS_OP_T) + 
                       " M_c=" + std::to_string(M_cublas) + " N_c=" + std::to_string(N_cublas) + " K_c=" + std::to_string(K_cublas) + 
                       " LDA_c=" + std::to_string(LDA_cublas) + " LDB_c=" + std::to_string(LDB_cublas) + " LDC_c=" + std::to_string(LDC_cublas));
        throw std::runtime_error("cublasSgemm failed");
    }
}


__global__ void attention_batch_prefill_cuda_kernel(
    float* d_output_batch_strided,      // Output: [B, H_q, D_h] effectively, though might be [B, H_q*D_h] and reshaped later
    const float* d_q_batch_strided,     // Q input: [B, H_q, D_h]
    const float* d_k_batch_strided,     // K input for current batch: [B, H_kv, D_h] (may be nullptr)
    const float* d_v_batch_strided,     // V input for current batch: [B, H_kv, D_h] (may be nullptr)
    float* d_kv_cache_k_base,           // K Cache base: [S_max, H_kv, D_h]
    float* d_kv_cache_v_base,           // V Cache base: [S_max, H_kv, D_h]
    int num_tokens_in_batch,            // B
    int num_q_heads,                    // H_q
    int num_kv_heads,                   // H_kv
    int head_dim,                       // D_h
    int start_pos_in_kv_cache,          // S_offset
    float scale,
    int cache_max_seq_len,
    const int* attention_mask_cu) {

    // Grid: (num_tokens_in_batch, num_q_heads)
    // Block: (head_dim)
    int token_idx_in_batch = blockIdx.x; // Current token in the batch we are computing attention FOR
    int q_head_idx = blockIdx.y;         // Current Q head we are computing attention FOR
    int dim_thread_idx = threadIdx.x;    // Thread index within the head dimension

    if (token_idx_in_batch >= num_tokens_in_batch || q_head_idx >= num_q_heads || dim_thread_idx >= head_dim) {
        return;
    }

    // Determine the GQA/MQA mapping: which KV head corresponds to the current Q head
    int kv_heads_per_q_group = num_q_heads / num_kv_heads;
    int kv_head_idx_for_q = q_head_idx / kv_heads_per_q_group;

    // Pointer to the current Q vector for this token and Q head
    const float* q_current_head_ptr = d_q_batch_strided +
                                      (size_t)token_idx_in_batch * num_q_heads * head_dim +
                                      (size_t)q_head_idx * head_dim;

    // Output pointer for the current token and Q head
    float* output_current_head_ptr = d_output_batch_strided +
                                     (size_t)token_idx_in_batch * num_q_heads * head_dim +
                                     (size_t)q_head_idx * head_dim;

    // CORRECTED: KV cache already contains current batch, so total context is available in cache
    int total_context_len = start_pos_in_kv_cache + num_tokens_in_batch;
    int max_attend_pos_for_this_token = start_pos_in_kv_cache + token_idx_in_batch;

    if (total_context_len > cache_max_seq_len) {
        total_context_len = cache_max_seq_len;
    }

    extern __shared__ float sdata[];
    float* s_scores = sdata; // Size: total_context_len
    float* s_dot_product_scratch = &sdata[total_context_len]; // Size: head_dim

    // Calculate QK^T for each K vector in the context with causal masking
    for (int context_pos = 0; context_pos < total_context_len; ++context_pos) {
        // CORRECTED: All K/V comes from cache since cache already contains current batch
        const float* k_vec_ptr = d_kv_cache_k_base +
                                 (size_t)context_pos * num_kv_heads * head_dim +
                                 (size_t)kv_head_idx_for_q * head_dim;
                    

        // Initialize score for this context position 
        if (dim_thread_idx == 0) {
            // Apply causal masking: token can only attend to positions <= its own position
            if (context_pos > max_attend_pos_for_this_token) {
                s_scores[context_pos] = -FLT_MAX; // Mask out future positions
            } else {
                s_scores[context_pos] = 0.0f;
            }
        }
        __syncthreads(); // Ensure s_scores[context_pos] is initialized before use

        // Skip computation if this position is masked
        if (context_pos <= max_attend_pos_for_this_token) {
            // Parallel dot product: each thread computes one element of Q_d * K_d
            // Store into shared memory scratchpad
            float q_val = q_current_head_ptr[dim_thread_idx];
            float k_val = k_vec_ptr[dim_thread_idx];
            s_dot_product_scratch[dim_thread_idx] = q_val * k_val;
            __syncthreads(); // Ensure all partial products are in s_dot_product_scratch

            // Thread 0 sums up the partial products from scratchpad
            if (dim_thread_idx == 0) {
                double dot_prod = 0.0; // USE DOUBLE FOR ACCUMULATION
                for (int d = 0; d < head_dim; ++d) {
                    dot_prod += (double)s_dot_product_scratch[d];
                }
                s_scores[context_pos] = (float)(dot_prod * scale); // Scale and cast back to float
            }
            __syncthreads(); // Ensure s_scores[context_pos] is written by thread 0 before next iteration
        }
    }

    // --- Softmax ---
    // Parallel softmax over s_scores[0...total_context_len-1]
    // 1. Find max score in s_scores (reduction)
    float max_score = -FLT_MAX; 

    float thread_local_max_for_scores = -FLT_MAX;
    for (int i = dim_thread_idx; i < total_context_len; i += blockDim.x) {
         thread_local_max_for_scores = fmaxf(thread_local_max_for_scores, s_scores[i]);
    }
    s_dot_product_scratch[dim_thread_idx] = thread_local_max_for_scores; 
    __syncthreads(); 

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (dim_thread_idx < s) { 
            s_dot_product_scratch[dim_thread_idx] = fmaxf(s_dot_product_scratch[dim_thread_idx], s_dot_product_scratch[dim_thread_idx + s]);
        }
        __syncthreads();
    }
    if (dim_thread_idx == 0) {
        max_score = s_dot_product_scratch[0];
    }
    __syncthreads(); 
    max_score = s_dot_product_scratch[0]; 

    float thread_local_sum_exp = 0.0f;
     for (int i = dim_thread_idx; i < total_context_len; i += blockDim.x) {
         float val = expf(s_scores[i] - max_score);
         s_scores[i] = val; 
         thread_local_sum_exp += val;
    }
    s_dot_product_scratch[dim_thread_idx] = thread_local_sum_exp; 
    __syncthreads(); 
    
    float sum_exp_scores = 0.0f; 
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (dim_thread_idx < s) {
            s_dot_product_scratch[dim_thread_idx] += s_dot_product_scratch[dim_thread_idx + s];
        }
        __syncthreads();
    }
    if (dim_thread_idx == 0) {
        sum_exp_scores = s_dot_product_scratch[0];
    }
    __syncthreads(); 
    sum_exp_scores = s_dot_product_scratch[0]; 

    float inv_sum_exp_scores = 1.0f / (sum_exp_scores + 1e-9f); 
    for (int i = dim_thread_idx; i < total_context_len; i += blockDim.x) {
        s_scores[i] *= inv_sum_exp_scores; 
    }
    __syncthreads(); 

    // --- Weighted Sum of V ---
    double weighted_sum_v_d = 0.0; // USE DOUBLE FOR ACCUMULATION
    for (int context_pos = 0; context_pos < total_context_len; ++context_pos) {
        // CORRECTED: All V comes from cache since cache already contains current batch
        const float* v_vec_ptr = d_kv_cache_v_base +
                                 (size_t)context_pos * num_kv_heads * head_dim +
                                 (size_t)kv_head_idx_for_q * head_dim;
        
        weighted_sum_v_d += (double)s_scores[context_pos] * (double)v_vec_ptr[dim_thread_idx];
    }
    
    output_current_head_ptr[dim_thread_idx] = (float)weighted_sum_v_d;

}

__global__ void rmsnorm_batch_kernel(float* d_out, const float* d_in, const float* d_weight, 
                                   int num_tokens, int hidden_size, float eps) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) {
        return;
    }

    const float* current_token_input = d_in + token_idx * hidden_size;
    float* current_token_output = d_out + token_idx * hidden_size;

    // Calculate sum of squares for the current token's hidden state
    float ssq = 0.0f;
    // A single thread per token for sum of squares calculation is not efficient.
    // This should be a parallel reduction per token.
    // For simplicity in this step, doing it serially per token via one thread block.
    // A more optimized version would use a grid-stride loop or a 2D grid.
    // Or, launch one block per token, and do a parallel reduction within the block.
    if (threadIdx.x == 0) { // Let one thread in the block handle this token
        for (int i = 0; i < hidden_size; ++i) {
            ssq += current_token_input[i] * current_token_input[i];
        }
        ssq /= hidden_size;
        float inv_norm_factor = rsqrtf(ssq + eps);

        for (int i = 0; i < hidden_size; ++i) {
            current_token_output[i] = current_token_input[i] * inv_norm_factor * d_weight[i];
        }
    }
}

// Optimized Batched RMSNorm Kernel - one block per token, parallel reduction within block
__global__ void rmsnorm_batch_kernel_optimized(float* d_out, const float* d_in, const float* d_weight, 
                                             int num_tokens, int hidden_size, float eps) {
    extern __shared__ float sdata[]; // For reduction
    int token_idx = blockIdx.x;

    if (token_idx >= num_tokens) {
        return;
    }

    const float* current_token_input = d_in + token_idx * hidden_size;
    float* current_token_output = d_out + token_idx * hidden_size;

    unsigned int tid = threadIdx.x;
    unsigned int N_per_token = hidden_size; // elements in one token's hidden state

    // Load input into shared memory and calculate sum of squares (partial)
    float local_ssq_sum = 0.0f;
    for (unsigned int i = tid; i < N_per_token; i += blockDim.x) {
        float val = current_token_input[i];
        local_ssq_sum += val * val;
    }
    sdata[tid] = local_ssq_sum;
    __syncthreads();

    // Parallel reduction in shared memory for ssq
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 gets final sum of squares for this token, calculates norm, and applies
    if (tid == 0) {
        float total_ssq = sdata[0];
        total_ssq /= hidden_size;
        float inv_norm_factor = rsqrtf(total_ssq + eps);
        // This final application loop should also be parallelized if hidden_size is large.
        // For now, thread 0 does it.
        for (unsigned int i = 0; i < N_per_token; ++i) {
             current_token_output[i] = current_token_input[i] * inv_norm_factor * d_weight[i];
        }
    }    
}

// Final Optimized Batched RMSNorm Kernel - 2D grid, one warp per row (token), parallel reduction within warp
__global__ void rmsnorm_batch_kernel_final_optimized(float* __restrict__ d_out, 
                                             const float* __restrict__ d_in, 
                                             const float* __restrict__ d_weight, 
                                             int num_tokens, int hidden_size, float eps) {
    int token_idx = blockIdx.x; // Each block processes one token

    if (token_idx >= num_tokens) {
        return;
    }

    const float* current_token_input = d_in + token_idx * hidden_size;
    float* current_token_output = d_out + token_idx * hidden_size;
    
    // MODIFIED: Use double for ssq accumulation
    double ssq_double = 0.0;
    // Parallel reduction for ssq across threads in the block for the current token
    // Each thread handles a subset of the hidden_size elements for this token.
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = current_token_input[i];
        // MODIFIED: Accumulate into double
        ssq_double += static_cast<double>(val) * static_cast<double>(val);
    }

    // Reduce ssq_double across threads in the block
    extern __shared__ double sdata_double[]; // MODIFIED: Shared memory of type double
    sdata_double[threadIdx.x] = ssq_double;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_double[threadIdx.x] += sdata_double[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        // MODIFIED: ssq_double is now fully reduced in sdata_double[0]
        ssq_double = sdata_double[0]; 
    }
    __syncthreads(); 

    if (threadIdx.x == 0) {
        // MODIFIED: Store mean square (as double) in shared memory
        sdata_double[0] = sdata_double[0] / hidden_size; 
    }
    __syncthreads();
    
    // MODIFIED: All threads read the mean_square (as double)
    double mean_square_double = sdata_double[0]; 
    // MODIFIED: Cast to float before rsqrtf
    float mean_square_float = static_cast<float>(mean_square_double);
    float inv_norm_factor = rsqrtf(mean_square_float + eps);

    // Apply normalization in parallel by all threads in the block for this token
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        current_token_output[i] = current_token_input[i] * inv_norm_factor * d_weight[i];
    }
}


void rmsnorm_batch_cuda(float* d_out, float* d_in, const float* d_weight, 
                        int num_tokens, int hidden_size, float eps, 
                        cudaStream_t stream) {

    if (num_tokens == 0 || hidden_size == 0) {
        Logger::info("[RMSNORM_BATCH_CUDA_EXIT] num_tokens or hidden_size is 0. Nothing to do.");
        return; // Nothing to do
    }

    // Log initial input values (first token, first few elements)
    if (num_tokens > 0 && hidden_size > 0) {
        int log_count = std::min(3, hidden_size);
        std::vector<float> h_in_sample(log_count);
        gpuErrchk(cudaMemcpyAsync(h_in_sample.data(), d_in, log_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        std::vector<float> h_weight_sample(log_count);
        gpuErrchk(cudaMemcpyAsync(h_weight_sample.data(), d_weight, log_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }


    // Choose a reasonable block size. For RMSNorm, threads operate along hidden_size.
    // Max threads per block is typically 1024.
    // If hidden_size is small (e.g., < 256), could use hidden_size as blockDim.x
    // If hidden_size is large, use a fixed size like 256 or 512.
    int threads_per_block_x = (hidden_size <= 256) ? hidden_size : 256;
    if (hidden_size == 0) threads_per_block_x = 1; // Avoid 0 blockDim
    
    // We launch one block per token.
    dim3 block_dim(threads_per_block_x);
    dim3 grid_dim(num_tokens);
    
    // Shared memory for reduction within each block (for one token)
    // MODIFIED: Change shared memory size to use sizeof(double)
    size_t shared_mem_size = threads_per_block_x * sizeof(double);
    if (threads_per_block_x == 0) shared_mem_size = sizeof(double); // Min for sdata_double[0]

    rmsnorm_batch_kernel_final_optimized<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        d_out, d_in, d_weight, num_tokens, hidden_size, eps
    );
    gpuErrchk(cudaGetLastError()); // Check for errors immediately after kernel launch

    // Log output values (first token, first few elements)
    if (num_tokens > 0 && hidden_size > 0) {
        int log_count = std::min(3, hidden_size);
        std::vector<float> h_out_sample(log_count);
        gpuErrchk(cudaMemcpyAsync(h_out_sample.data(), d_out, log_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
}

// Host wrapper for Batched Attention Prefill
// ---------------------------------------------------------------------------
// Prefill attention, tiled with online softmax (kernel_variant "flash_v1";
// docs/TTFT.md). Selected by EMBER_PREFILL_ATTN=flash; the default stays the
// baseline kernel above until this variant passes parity and its TTFT has
// been measured under the same protocol.
//
// What it computes: exactly the baseline's causal attention. Query token t
// (absolute position start + t) attends to cache positions 0..start + t, with
// K/V read from the cache the batch was just written into.
//
// Assumptions: head_dim is a multiple of 32 and at most 128 (each lane of a
// warp owns head_dim / 32 output dimensions); num_q_heads % num_kv_heads == 0.
// The wrapper falls back to the baseline kernel otherwise.
//
// Shape: one block per (tile of FA_BM = 32 queries, query head), 4 warps,
// each warp owning FA_BM / 4 = 8 query rows. K/V move through shared memory
// in tiles of FA_BN = 32 positions. K rows are padded by one float, so lane j
// reading K[j][d] hits a different bank per lane. Per row and tile, lane j
// scores key j; a warp max and sum update the row's running max m and
// normaliser l (online softmax), the accumulator is rescaled by
// exp(m_old - m_new), and P V is added with p_j broadcast by shuffle. Tiles
// that lie entirely beyond a block's last query are never loaded (causal
// skip), which is half the work of the baseline's full scan.
//
// Trade-offs: FP32 throughout (no tensor cores), so it is limited by FP32
// FMA throughput rather than memory once tiles are resident. Q is read from
// global memory through the read-only cache (each lane reads the same q
// element: a broadcast) rather than staged in shared memory, which keeps the
// shared footprint under the 48 KB static limit at head_dim 128. Query heads
// sharing a KV head re-load the same K/V tiles (GQA handled by indexing, as in
// the baseline); sharing tiles across a GQA group is the next variant. Tile
// sizes are fixed here; tuning them per GPU is its own experiment.
// Accumulation is FP32 with expf; the baseline accumulates its dot products
// in FP64, so agreement is to FP32 rounding, not bit-exact.
// ---------------------------------------------------------------------------
constexpr int FA_BM = 32;      // queries per block
constexpr int FA_BN = 32;      // keys per K/V tile (= warp size: lane j scores key j)
constexpr int FA_WARPS = 4;
constexpr int FA_ROWS = FA_BM / FA_WARPS;

template <int DPL>  // output dims per lane = head_dim / 32
__global__ void attention_prefill_flash_kernel(
    float* __restrict__ out,            // [B, H_q, D]
    const float* __restrict__ q,        // [B, H_q, D]
    const float* __restrict__ k_cache,  // [S_max, H_kv, D]
    const float* __restrict__ v_cache,  // [S_max, H_kv, D]
    int num_tokens, int num_q_heads, int num_kv_heads, int start_pos, float scale) {
    constexpr int D = DPL * 32;
    __shared__ float sK[FA_BN][D + 1];
    __shared__ float sV[FA_BN][D];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int q_head = blockIdx.y;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int tile_first = blockIdx.x * FA_BM;
    const int tile_last = min(tile_first + FA_BM, num_tokens) - 1;  // last query in this block
    const int kv_end = start_pos + tile_last + 1;                    // keys this block can need
    const size_t kv_stride = (size_t)num_kv_heads * D;

    float m[FA_ROWS], l[FA_ROWS], acc[FA_ROWS][DPL];
#pragma unroll
    for (int i = 0; i < FA_ROWS; ++i) {
        m[i] = -FLT_MAX;
        l[i] = 0.0f;
#pragma unroll
        for (int k = 0; k < DPL; ++k) acc[i][k] = 0.0f;
    }

    for (int kv0 = 0; kv0 < kv_end; kv0 += FA_BN) {
        // Cooperative tile load; positions past kv_end are zero-filled and
        // masked below.
        for (int idx = threadIdx.x; idx < FA_BN * D; idx += blockDim.x) {
            const int j = idx / D, d = idx % D;
            const int pos = kv0 + j;
            const bool in = pos < kv_end;
            const size_t off = (size_t)pos * kv_stride + (size_t)kv_head * D + d;
            sK[j][d] = in ? k_cache[off] : 0.0f;
            sV[j][d] = in ? v_cache[off] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < FA_ROWS; ++i) {
            const int t = tile_first + warp * FA_ROWS + i;
            if (t >= num_tokens) continue;               // warp-uniform
            const int q_pos = start_pos + t;
            if (kv0 > q_pos) continue;                   // whole tile is in this row's future
            const int key = kv0 + lane;
            const float* q_row = q + ((size_t)t * num_q_heads + q_head) * D;
            float s = -FLT_MAX;
            if (key <= q_pos) {
                float dot = 0.0f;
#pragma unroll 8
                for (int d = 0; d < D; ++d) dot += __ldg(q_row + d) * sK[lane][d];
                s = dot * scale;
            }
            float tile_max = s;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, o));
            const float m_new = fmaxf(m[i], tile_max);
            const float p = (key <= q_pos) ? expf(s - m_new) : 0.0f;
            const float corr = expf(m[i] - m_new);       // 0 on the first visited tile
            float p_sum = p;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) p_sum += __shfl_xor_sync(0xffffffff, p_sum, o);
            l[i] = l[i] * corr + p_sum;
#pragma unroll
            for (int k = 0; k < DPL; ++k) acc[i][k] *= corr;
#pragma unroll 4
            for (int jj = 0; jj < FA_BN; ++jj) {
                const float pj = __shfl_sync(0xffffffff, p, jj);
#pragma unroll
                for (int k = 0; k < DPL; ++k) acc[i][k] += pj * sV[jj][lane + 32 * k];
            }
            m[i] = m_new;
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < FA_ROWS; ++i) {
        const int t = tile_first + warp * FA_ROWS + i;
        if (t >= num_tokens) continue;
        float* o_row = out + ((size_t)t * num_q_heads + q_head) * D;
        const float inv_l = 1.0f / l[i];                 // l >= 1: key 0 is always visible
#pragma unroll
        for (int k = 0; k < DPL; ++k) o_row[lane + 32 * k] = acc[i][k] * inv_l;
    }
}

// ---------------------------------------------------------------------------
// Prefill attention v2 (kernel_variant "flash_v2"; EMBER_PREFILL_ATTN=flash2).
//
// Same computation as flash_v1 and the baseline: causal attention of the
// batch's queries over cache positions 0..start + t. The op map
// (docs/TTFT.md) showed v1 spending ~2 shared/L1 loads per FMA and loading
// every K/V tile once per query head, though the 8 query heads of a GQA
// group read the same K/V. v2 changes the work decomposition, not the
// arithmetic:
//
//  * GQA-shared tiles. One block per (KV head, tile of FA2_TQ query
//    tokens) owns all G = H_q / H_kv query heads of that KV head: 64 rows
//    (FA2_ROWS = FA2_TQ * G), row r = (token r / G, head r % G). Each K/V
//    tile is loaded once for all of them.
//  * Register micro-tiles. Scores S = Q K^T for a 64 x 32 tile are computed
//    4 rows x 4 keys per thread, so each pair of shared loads feeds 16 FMAs
//    instead of 1. P V: each thread owns 2 rows x D/4 dims (dims interleaved
//    by 4 so shared reads are conflict-free), with 2 P values reused across
//    D/4 dims.
//  * Online softmax per row, as in v1: running max m, normaliser l, and the
//    accumulator rescaled by exp(m_old - m_new). Two threads per row reduce
//    the tile max and sum with a shuffle. A row entirely masked within a
//    tile leaves m, l and the accumulator unchanged (guarded, not NaN).
//
// Assumptions: head_dim is 32 or 64, and G divides 64. Static shared
// memory stays under 48 KB at D = 64: Q 64 x 65, K 32 x 65, V 32 x 64,
// S 64 x 33 floats. Otherwise the dispatcher uses v1. Still FP32 on CUDA
// cores (no tensor cores), so parity with v1 and the baseline is to FP32
// rounding. Tile sizes are fixed; tuning them per GPU is its own
// experiment.
// ---------------------------------------------------------------------------
constexpr int FA2_ROWS = 64;     // query rows (token x head) per block
constexpr int FA2_BN = 32;       // keys per K/V tile
constexpr int FA2_THREADS = 128;

template <int D>
__global__ void attention_prefill_flash2_kernel(
    float* __restrict__ out, const float* __restrict__ q, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, int num_tokens, int num_q_heads, int num_kv_heads,
    int start_pos, float scale) {
    constexpr int DPT = D / 4;                 // P.V dims per thread
    __shared__ float sQ[FA2_ROWS][D + 1];
    __shared__ float sK[FA2_BN][D + 1];
    __shared__ float sV[FA2_BN][D];
    __shared__ float sS[FA2_ROWS][FA2_BN + 1];
    __shared__ float sM[FA2_ROWS], sL[FA2_ROWS], sCorr[FA2_ROWS];

    const int tid = threadIdx.x;
    const int group = num_q_heads / num_kv_heads;  // G
    const int tq = FA2_ROWS / group;                // query tokens per block
    const int kv_head = blockIdx.y;
    const int tok0 = blockIdx.x * tq;
    const int tok_last = min(tok0 + tq, num_tokens) - 1;
    const int kv_end = start_pos + tok_last + 1;
    const size_t kv_stride = (size_t)num_kv_heads * D;

    // Row r -> (token, query head). Out-of-range tokens load zeros and are
    // never written.
    for (int idx = tid; idx < FA2_ROWS * D; idx += FA2_THREADS) {
        const int r = idx / D, d = idx % D;
        const int t = tok0 + r / group;
        const int h = kv_head * group + r % group;
        sQ[r][d] = t < num_tokens ? q[((size_t)t * num_q_heads + h) * D + d] : 0.0f;
    }
    if (tid < FA2_ROWS) {
        sM[tid] = -FLT_MAX;
        sL[tid] = 0.0f;
    }

    // P.V ownership: rows pr0, pr0 + 1; dims pc + 4k.
    const int pr0 = (tid / 4) * 2;
    const int pc = tid % 4;
    float acc[2][DPT];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int k = 0; k < DPT; ++k) acc[i][k] = 0.0f;

    // Score micro-tile ownership: rows sr0..sr0+3, keys sc0..sc0+3.
    const int sr0 = (tid / 8) * 4;
    const int sc0 = (tid % 8) * 4;

    for (int kv0 = 0; kv0 < kv_end; kv0 += FA2_BN) {
        __syncthreads();  // previous tile's P.V reads of sS/sV are done
        for (int idx = tid; idx < FA2_BN * D; idx += FA2_THREADS) {
            const int j = idx / D, d = idx % D;
            const int pos = kv0 + j;
            const bool in = pos < kv_end;
            const size_t off = (size_t)pos * kv_stride + (size_t)kv_head * D + d;
            sK[j][d] = in ? k_cache[off] : 0.0f;
            sV[j][d] = in ? v_cache[off] : 0.0f;
        }
        __syncthreads();

        // S = Q K^T * scale, causal mask.
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) s[i][j] = 0.0f;
#pragma unroll 8
        for (int d = 0; d < D; ++d) {
            float qv[4], kv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) qv[i] = sQ[sr0 + i][d];
#pragma unroll
            for (int j = 0; j < 4; ++j) kv[j] = sK[sc0 + j][d];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) s[i][j] += qv[i] * kv[j];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = sr0 + i;
            const int q_pos = start_pos + tok0 + r / group;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = kv0 + sc0 + j;
                sS[r][sc0 + j] = (key <= q_pos && key < kv_end) ? s[i][j] * scale : -FLT_MAX;
            }
        }
        __syncthreads();

        // Online softmax: two threads per row, 16 keys each.
        {
            const int r = tid / 2, half = tid % 2;
            float tmax = -FLT_MAX;
#pragma unroll
            for (int j = 0; j < FA2_BN / 2; ++j) tmax = fmaxf(tmax, sS[r][half * 16 + j]);
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, 1));
            // Branch-free: lanes of different rows share a warp, so every
            // lane runs the same shuffles. A row with no visible key in this
            // tile (any == false) gets p = 0 and corr = 1: m, l and its
            // accumulator are left as they were.
            const bool any = tmax > -FLT_MAX;
            const float m_old = sM[r];
            const float m_new = any ? fmaxf(m_old, tmax) : m_old;
            const float corr = any ? expf(m_old - m_new) : 1.0f;  // 0 when m_old is -FLT_MAX
            float p_sum = 0.0f;
#pragma unroll
            for (int j = 0; j < FA2_BN / 2; ++j) {
                const float sv = sS[r][half * 16 + j];
                const float p = (any && sv > -FLT_MAX) ? expf(sv - m_new) : 0.0f;
                sS[r][half * 16 + j] = p;
                p_sum += p;
            }
            p_sum += __shfl_xor_sync(0xffffffff, p_sum, 1);
            if (half == 0) {
                sM[r] = m_new;
                sL[r] = sL[r] * corr + p_sum;
                sCorr[r] = corr;
            }
        }
        __syncthreads();

        // acc = acc * corr + P V.
        const float c0 = sCorr[pr0], c1 = sCorr[pr0 + 1];
#pragma unroll
        for (int k = 0; k < DPT; ++k) {
            acc[0][k] *= c0;
            acc[1][k] *= c1;
        }
#pragma unroll 4
        for (int j = 0; j < FA2_BN; ++j) {
            const float p0 = sS[pr0][j], p1 = sS[pr0 + 1][j];
#pragma unroll
            for (int k = 0; k < DPT; ++k) {
                const float v = sV[j][pc + 4 * k];
                acc[0][k] += p0 * v;
                acc[1][k] += p1 * v;
            }
        }
    }

    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = pr0 + i;
        const int t = tok0 + r / group;
        if (t >= num_tokens) continue;
        const int h = kv_head * group + r % group;
        const float inv_l = 1.0f / sL[r];   // l > 0: key 0 is visible to every row
        float* o = out + ((size_t)t * num_q_heads + h) * D;
#pragma unroll
        for (int k = 0; k < DPT; ++k) o[pc + 4 * k] = acc[i][k] * inv_l;
    }
}

static bool launch_flash2_prefill_attention(float* out, const float* q, const float* k_cache,
                                            const float* v_cache, int num_tokens, int num_q_heads,
                                            int num_kv_heads, int head_dim, int start_pos,
                                            float scale, cudaStream_t stream) {
    if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) return false;
    const int group = num_q_heads / num_kv_heads;
    if (FA2_ROWS % group != 0) return false;
    const int tq = FA2_ROWS / group;
    dim3 grid((num_tokens + tq - 1) / tq, num_kv_heads);
    dim3 block(FA2_THREADS);
    switch (head_dim) {
        case 32: attention_prefill_flash2_kernel<32><<<grid, block, 0, stream>>>(out, q, k_cache, v_cache, num_tokens, num_q_heads, num_kv_heads, start_pos, scale); break;
        case 64: attention_prefill_flash2_kernel<64><<<grid, block, 0, stream>>>(out, q, k_cache, v_cache, num_tokens, num_q_heads, num_kv_heads, start_pos, scale); break;
        default: return false;
    }
    return true;
}

// Read once: which prefill attention kernel to launch (docs/TTFT.md).
// 0 = baseline, 1 = flash (v1), 2 = flash2 (v2, falls back to v1, then to
// the baseline kernel when an assumption does not hold).
// Default (unset) is flash2. It matched the baseline kernel to <= 1e-5
// relative on the first-token logits at every length up to 4096, and was
// 2.8% to 39.8% faster than v1 there, 105x the original TTFT at 4096
// (docs/TTFT.md). EMBER_PREFILL_ATTN=baseline restores the original kernel.
static int prefill_attention_mode() {
    static const int mode = [] {
        const char* v = std::getenv("EMBER_PREFILL_ATTN");
        if (v == nullptr) return 2;
        const std::string s(v);
        if (s == "baseline") return 0;
        return s == "flash" ? 1 : 2;
    }();
    return mode;
}
static bool use_flash_prefill_attention() { return prefill_attention_mode() >= 1; }

// Launch the tiled kernel if its assumptions hold; false means "use the
// baseline kernel instead".
static bool launch_flash_prefill_attention(float* out, const float* q, const float* k_cache,
                                           const float* v_cache, int num_tokens, int num_q_heads,
                                           int num_kv_heads, int head_dim, int start_pos,
                                           float scale, cudaStream_t stream) {
    if (head_dim % 32 != 0 || head_dim > 128 || num_q_heads % num_kv_heads != 0) return false;
    dim3 grid((num_tokens + FA_BM - 1) / FA_BM, num_q_heads);
    dim3 block(FA_WARPS * 32);
    switch (head_dim / 32) {
        case 1: attention_prefill_flash_kernel<1><<<grid, block, 0, stream>>>(out, q, k_cache, v_cache, num_tokens, num_q_heads, num_kv_heads, start_pos, scale); break;
        case 2: attention_prefill_flash_kernel<2><<<grid, block, 0, stream>>>(out, q, k_cache, v_cache, num_tokens, num_q_heads, num_kv_heads, start_pos, scale); break;
        case 3: attention_prefill_flash_kernel<3><<<grid, block, 0, stream>>>(out, q, k_cache, v_cache, num_tokens, num_q_heads, num_kv_heads, start_pos, scale); break;
        case 4: attention_prefill_flash_kernel<4><<<grid, block, 0, stream>>>(out, q, k_cache, v_cache, num_tokens, num_q_heads, num_kv_heads, start_pos, scale); break;
        default: return false;
    }
    return true;
}

void attention_batch_prefill_cuda(
    const float* d_q_batch_strided,   // Input Q: [B, H_q, D_h]
    const float* d_k_batch_strided,   // Input K for current batch (NEW)
    const float* d_v_batch_strided,   // Input V for current batch (NEW)
    float* d_kv_cache_k_base,         // K Cache: [S_max, H_kv, D_h] (changed to non-const)
    float* d_kv_cache_v_base,         // V Cache: [S_max, H_kv, D_h] (changed to non-const)
    float* d_output_batch_strided,    // Output: [B, H_q, D_h]
    int num_tokens_in_batch,          // B
    int start_pos_in_kv_cache,        // Start position for this batch in KV cache (NEW)
    int cache_max_seq_len,            // Max capacity of KV cache (NEW)
    int num_q_heads,                  // H_q
    int num_kv_heads,                 // H_kv
    int head_dim,                     // D_h
    float scale,
    cudaStream_t stream,
    const int* attention_mask_cu      // Optional attention mask
    ) {


    if (num_tokens_in_batch == 0 || head_dim == 0) {
        Logger::info("[ATTN_BATCH_PREFILL_CUDA_SKIP] num_tokens_in_batch or head_dim is 0. Nothing to do.");
        Logger::info("[ATTN_BATCH_PREFILL_CUDA_EXIT] Skipped operation.");
        return; // Nothing to do
    }
    if (num_kv_heads == 0 || num_q_heads % num_kv_heads != 0) {
        Logger::error("[ATTN_BATCH_PREFILL_CUDA_ERROR] Invalid num_kv_heads (" + std::to_string(num_kv_heads) +
                      ") or num_q_heads (" + std::to_string(num_q_heads) + ") for GQA.");
        Logger::info("[ATTN_BATCH_PREFILL_CUDA_EXIT] Error in head configuration.");
        return;
    }

    // Log initial input values (first token, first head, first few elements)
    int log_count_elements = std::min(3, head_dim);
    // Debug samples and the device sync below: EMBER_PREFILL_DEBUG=1 only
    // (same-stream ordering already orders the next op after this kernel).
    static const bool debug = std::getenv("EMBER_PREFILL_DEBUG") != nullptr;
    if (debug && num_tokens_in_batch > 0 && head_dim > 0) {
        // Log Q for Token 0 and Token 1
        if (d_q_batch_strided && num_q_heads > 0) {
            for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens_in_batch, 2); ++token_to_log_idx) {
                std::vector<float> h_q_sample(log_count_elements);
                // Q is [B, H_q, D_h]. 
                // Offset for Token `token_to_log_idx`, Head 0 is `token_to_log_idx * num_q_heads * head_dim`.
                // We are logging the first `log_count_elements` of the first head (H0) for that token.
                const float* q_token_ptr = d_q_batch_strided + (size_t)token_to_log_idx * num_q_heads * head_dim;
                gpuErrchk(cudaMemcpyAsync(h_q_sample.data(), q_token_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
        }

        // Log K_batch for Token 0 and Token 1
        if (d_k_batch_strided && num_kv_heads > 0) {
            for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens_in_batch, 2); ++token_to_log_idx) {
                std::vector<float> h_k_sample(log_count_elements);
                // K_batch is [B, H_kv, D_h].
                const float* k_token_ptr = d_k_batch_strided + (size_t)token_to_log_idx * num_kv_heads * head_dim;
                gpuErrchk(cudaMemcpyAsync(h_k_sample.data(), k_token_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
        }

        // Log V_batch for Token 0 and Token 1
        if (d_v_batch_strided && num_kv_heads > 0) {
            for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens_in_batch, 2); ++token_to_log_idx) {
                std::vector<float> h_v_sample(log_count_elements);
                // V_batch is [B, H_kv, D_h].
                const float* v_token_ptr = d_v_batch_strided + (size_t)token_to_log_idx * num_kv_heads * head_dim;
                gpuErrchk(cudaMemcpyAsync(h_v_sample.data(), v_token_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
            }
        }
        
        // Note: K/V cache sampling might be less informative here if start_pos_in_kv_cache is large,
        // as it shows past context rather than current batch contributions to cache.
        // However, logging it for completeness if cache pointers are valid.
        if (d_kv_cache_k_base && start_pos_in_kv_cache < cache_max_seq_len && num_kv_heads > 0) {
            std::vector<float> h_kc_sample(log_count_elements);
            // K Cache: [S_max, H_kv, D_h]. Pointer to S_offset, H0.
            float* cache_k_ptr = d_kv_cache_k_base + (size_t)start_pos_in_kv_cache * num_kv_heads * head_dim;
            gpuErrchk(cudaMemcpyAsync(h_kc_sample.data(), cache_k_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        if (d_kv_cache_v_base && start_pos_in_kv_cache < cache_max_seq_len && num_kv_heads > 0) {
            std::vector<float> h_vc_sample(log_count_elements);
            // V Cache: [S_max, H_kv, D_h]. Pointer to S_offset, H0.
            float* cache_v_ptr = d_kv_cache_v_base + (size_t)start_pos_in_kv_cache * num_kv_heads * head_dim;
            gpuErrchk(cudaMemcpyAsync(h_vc_sample.data(), cache_v_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }

    dim3 grid_dim(num_tokens_in_batch, num_q_heads); // One block per (token_in_batch, q_head)
    dim3 block_dim(head_dim); // Threads in block work on one head's computation, primarily along head_dim

    int total_context_len = start_pos_in_kv_cache + num_tokens_in_batch;
    if (total_context_len > cache_max_seq_len) total_context_len = cache_max_seq_len;
    size_t shared_mem_bytes = (total_context_len + head_dim) * sizeof(float);

    if (shared_mem_bytes > 48 * 1024) { 
        Logger::warning("[ATTN_BATCH_PREFILL_CUDA_WARN] Requested shared memory (" + std::to_string(shared_mem_bytes) + " bytes) might be large.");
    }

    // Only the launch differs between kernel variants: the debug copies and
    // the device sync around it stay identical, so a TTFT change is the
    // kernel's alone.
    if (prefill_attention_mode() == 2 &&
        launch_flash2_prefill_attention(d_output_batch_strided, d_q_batch_strided,
                                        d_kv_cache_k_base, d_kv_cache_v_base, num_tokens_in_batch,
                                        num_q_heads, num_kv_heads, head_dim, start_pos_in_kv_cache,
                                        scale, stream)) {
        // launched v2
    } else if (use_flash_prefill_attention() &&
        launch_flash_prefill_attention(d_output_batch_strided, d_q_batch_strided,
                                       d_kv_cache_k_base, d_kv_cache_v_base, num_tokens_in_batch,
                                       num_q_heads, num_kv_heads, head_dim, start_pos_in_kv_cache,
                                       scale, stream)) {
        // launched
    } else
    attention_batch_prefill_cuda_kernel<<<grid_dim, block_dim, shared_mem_bytes, stream>>>(
        d_output_batch_strided,
        d_q_batch_strided,
        d_k_batch_strided,
        d_v_batch_strided,
        d_kv_cache_k_base,
        d_kv_cache_v_base,
        num_tokens_in_batch,
        num_q_heads,
        num_kv_heads,
        head_dim,
        start_pos_in_kv_cache,
        scale, 
        cache_max_seq_len,
        attention_mask_cu
    );
    gpuErrchk(cudaGetLastError());

    // Log output values
    if (debug && num_tokens_in_batch > 0 && num_q_heads > 0 && head_dim > 0 && d_output_batch_strided) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens_in_batch, 2); ++token_to_log_idx) {
            std::vector<float> h_out_sample(log_count_elements);
            // Output is [B, H_q, D_h]. 
            const float* out_token_ptr = d_output_batch_strided + (size_t)token_to_log_idx * num_q_heads * head_dim;
            gpuErrchk(cudaMemcpyAsync(h_out_sample.data(), out_token_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }
    if (debug) cudaDeviceSynchronize();

    // Log output values
    if (debug && num_tokens_in_batch > 0 && num_q_heads > 0 && head_dim > 0 && d_output_batch_strided) {
        for (int token_to_log_idx = 0; token_to_log_idx < std::min(num_tokens_in_batch, 2); ++token_to_log_idx) {
            std::vector<float> h_out_sample(log_count_elements);
            // Output is [B, H_q, D_h]. 
            const float* out_token_ptr = d_output_batch_strided + (size_t)token_to_log_idx * num_q_heads * head_dim;
            gpuErrchk(cudaMemcpyAsync(h_out_sample.data(), out_token_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }
}

__global__ void update_kv_cache_batch_kernel(
    float* d_kv_cache_layer_base,         // K or V cache for the layer: [max_seq_len, num_kv_heads, head_dim]
    const float* d_keys_or_values_batch,  // Batch of K or V vectors from current forward pass: [num_tokens_in_batch, num_kv_heads, head_dim]
    int start_pos_in_kv_cache,            // Starting sequence position in cache for this batch
    int num_tokens_in_batch,
    int num_kv_heads,
    int head_dim,
    int cache_max_seq_len) {

    int token_idx_in_batch = blockIdx.x;  // Identifies which token from the batch (0 to num_tokens_in_batch-1)
    int kv_head_idx = blockIdx.y;       // Identifies which KV head (0 to num_kv_heads-1)
    int dim_idx = threadIdx.x;          // Identifies which dimension within the head_dim (0 to head_dim-1)

    // Boundary checks
    if (token_idx_in_batch >= num_tokens_in_batch || 
        kv_head_idx >= num_kv_heads || 
        dim_idx >= head_dim) {
    return;
  }

    // Calculate the global sequence position in the cache where this token's KV vector will be written
    int global_seq_pos = start_pos_in_kv_cache + token_idx_in_batch;

    // Boundary check for cache capacity
    if (global_seq_pos >= cache_max_seq_len) {
        // This should ideally be prevented by checks in the host code before launching,
        // or by model logic that handles sequence length limits.
        // For a kernel, simply returning is a common way to handle out-of-bounds access.
    return;
  }

    // Calculate the offset in the source batch tensor (d_keys_or_values_batch)
    // It's laid out as [token_idx_in_batch][kv_head_idx][dim_idx]
    size_t source_offset = (size_t)token_idx_in_batch * num_kv_heads * head_dim + 
                           (size_t)kv_head_idx * head_dim + 
                           dim_idx;

    // Calculate the offset in the destination KV cache tensor (d_kv_cache_layer_base)
    // It's laid out as [global_seq_pos][kv_head_idx][dim_idx]
    size_t cache_offset = (size_t)global_seq_pos * num_kv_heads * head_dim + 
                          (size_t)kv_head_idx * head_dim + 
                          dim_idx;

    // Perform the copy
    d_kv_cache_layer_base[cache_offset] = d_keys_or_values_batch[source_offset];
}

void update_kv_cache_batch_cuda(
    float* d_kv_cache_layer_base,        // Device pointer to the K or V cache for the current layer
    const float* d_keys_or_values_batch, // Device pointer to the batch of K or V vectors to be written
    int start_pos_in_kv_cache,           // The sequence position in the cache where writing for this batch should begin
    int num_tokens_in_batch,             // Number of tokens in the d_keys_or_values_batch
    int num_kv_heads,                    // Number of K/V heads
    int head_dim,                        // Dimension of each K/V head
    int cache_max_seq_len,               // Maximum sequence length capacity of the cache
                                   cudaStream_t stream) {

    if (num_tokens_in_batch == 0 || head_dim == 0 || num_kv_heads == 0) {
        Logger::debug("[UPDATE_KV_CACHE_BATCH_CUDA_SKIP] Nothing to update (num_tokens_in_batch, head_dim, or num_kv_heads is 0).");
        Logger::info("[UPDATE_KV_CACHE_BATCH_CUDA_EXIT] Skipped update.");
    return;
  }

    if (start_pos_in_kv_cache + num_tokens_in_batch > cache_max_seq_len) {
        Logger::error("[UPDATE_KV_CACHE_BATCH_CUDA_WARN] Batch (start: " + std::to_string(start_pos_in_kv_cache) + ", count: " + std::to_string(num_tokens_in_batch) +
                      ") might exceed KV cache capacity (" + std::to_string(cache_max_seq_len) + "). Kernel will clip if not careful, or this is an error.");
    }

    // Log input K/V values (first token, first head, first few elements)
    int log_count_elements = std::min(3, head_dim);
    if (num_tokens_in_batch > 0 && num_kv_heads > 0 && head_dim > 0) {
        std::vector<float> h_kv_sample_in(log_count_elements);
        // d_keys_or_values_batch is [num_tokens_in_batch][num_kv_heads][head_dim]
        // Offset for first token, first head = 0 (assuming data is contiguous for the first head of the first token)
        gpuErrchk(cudaMemcpyAsync(h_kv_sample_in.data(), d_keys_or_values_batch, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    // Grid: One block per (token_in_batch, kv_head)
    dim3 grid_dim(num_tokens_in_batch, num_kv_heads);
    // Block: One thread per dimension in the head vector
    dim3 block_dim(head_dim);

    update_kv_cache_batch_kernel<<<grid_dim, block_dim, 0, stream>>>(
        d_kv_cache_layer_base,
        d_keys_or_values_batch,
        start_pos_in_kv_cache,
        num_tokens_in_batch,
        num_kv_heads,
        head_dim,
        cache_max_seq_len
    );
  gpuErrchk(cudaGetLastError());

    // Log values written to cache (first token written, first head, first few elements)
    if (num_tokens_in_batch > 0 && num_kv_heads > 0 && head_dim > 0 && start_pos_in_kv_cache < cache_max_seq_len) {
        std::vector<float> h_kv_sample_out(log_count_elements);
        // Cache is [cache_max_seq_len][num_kv_heads][head_dim]
        // Offset for first token written in this batch, first head:
        // (start_pos_in_kv_cache * num_kv_heads * head_dim) + (0 * head_dim)
        // Pointer to start_pos_in_kv_cache, for the first head (head 0)
        float* first_token_first_head_in_cache_ptr = d_kv_cache_layer_base + 
            (size_t)start_pos_in_kv_cache * num_kv_heads * head_dim;
            
        gpuErrchk(cudaMemcpyAsync(h_kv_sample_out.data(), first_token_first_head_in_cache_ptr, log_count_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
}

#endif // HAS_CUDA

/**
 * @brief Optimized RMS normalization using simple warp primitives and shared memory
 *
 * This kernel uses warp-level reductions and a shared memory variable for block-wide sum.
 */
__global__ void rmsnorm_optimized_kernel(const float* __restrict__ x,
                                       const float* __restrict__ weight,
                                       float* __restrict__ out,
                                       int n, float eps) {
    const int tid_global = threadIdx.x + blockIdx.x * blockDim.x;
    const int tid_block = threadIdx.x;

    __shared__ float s_block_sum_sq;

    if (tid_block == 0) {
        s_block_sum_sq = 0.0f;
    }
    __syncthreads();

    float thread_partial_sum_sq = 0.0f;
    for (int i = tid_global; i < n; i += gridDim.x * blockDim.x) {
        float val = x[i];
        thread_partial_sum_sq += val * val;
    }

    // Reduce within warp
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        thread_partial_sum_sq += __shfl_down_sync(0xFFFFFFFF, thread_partial_sum_sq, offset);
    }

    // Warp leaders add to shared memory sum
    if ((tid_block % warpSize) == 0) {
        atomicAdd(&s_block_sum_sq, thread_partial_sum_sq);
    }
    __syncthreads(); // Ensure all warps have added their sums

    float total_sum_sq = s_block_sum_sq;
    float rms_scale = rsqrtf(total_sum_sq / n + eps);

    for (int i = tid_global; i < n; i += gridDim.x * blockDim.x) {
        out[i] = x[i] * weight[i] * rms_scale;
    }
}

/**
 * @brief Optimized softmax using shared memory and warp primitives
 *
 * Correctly finds max and sum of exponentials at block level for accurate softmax.
 */
__global__ void softmax_optimized_kernel(const float* __restrict__ x,
                                        float* __restrict__ out,
                                        int n) {
    const int tid_global = threadIdx.x + blockIdx.x * blockDim.x;
    const int tid_block = threadIdx.x;

    __shared__ float s_block_max_val;
    __shared__ float s_block_sum_exp;

    // Initialize shared variables by the first thread of the block
    if (tid_block == 0) {
        s_block_max_val = -INFINITY;
        s_block_sum_exp = 0.0f;
    }
    __syncthreads();

    // Phase 1: Find max
    float thread_max_val = -INFINITY;
    for (int i = tid_global; i < n; i += gridDim.x * blockDim.x) {
        thread_max_val = fmaxf(thread_max_val, x[i]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        thread_max_val = fmaxf(thread_max_val, __shfl_down_sync(0xFFFFFFFF, thread_max_val, offset));
    }
    if ((tid_block % warpSize) == 0) { // Warp leaders
        atomicMaxFloat(&s_block_max_val, thread_max_val);
    }
    __syncthreads();
    float global_max = s_block_max_val;

    // Phase 2: Compute sum of exp
    float thread_sum_exp = 0.0f;
    for (int i = tid_global; i < n; i += gridDim.x * blockDim.x) {
        thread_sum_exp += expf(x[i] - global_max);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        thread_sum_exp += __shfl_down_sync(0xFFFFFFFF, thread_sum_exp, offset);
    }
    if ((tid_block % warpSize) == 0) { // Warp leaders
        atomicAddFloat(&s_block_sum_exp, thread_sum_exp);
    }
    __syncthreads();
    float global_sum_exp = s_block_sum_exp;

    // Phase 3: Apply softmax
    float inv_sum = (global_sum_exp == 0.0f) ? 0.0f : 1.0f / global_sum_exp;
    for (int i = tid_global; i < n; i += gridDim.x * blockDim.x) {
        out[i] = expf(x[i] - global_max) * inv_sum;
    }
}

/**
 * @brief Corrected and optimized attention kernel
 *
 * Uses shared memory for scores and performs a full softmax operation correctly.
 * Each block computes attention for one head.
 */
__global__ void attention_optimized_kernel(const float* __restrict__ Q_current,      // Base Q for all heads: [num_q_heads, head_dim] (actually, [num_tokens_batch * num_q_heads * head_dim] from model.cpp)
                                          const float* __restrict__ K_layer_cache_base, // Full K cache: [max_seq_len, num_kv_heads, head_dim]
                                          const float* __restrict__ V_layer_cache_base, // Full V cache: [max_seq_len, num_kv_heads, head_dim]
                                          float* __restrict__ out,                    // Output for all heads: [num_q_heads, head_dim]
                                          int current_seq_len, int head_dim, float scale,
                                          int cache_num_kv_heads, int num_q_heads,
                                          const int* step_dev) {
    // Graph decode: the length comes from device memory (step_dev[1] = pos).
    if (step_dev) current_seq_len = step_dev[1] + 1;

    const int head_idx = blockIdx.x; // Current Q head index this block is working on
    const int tid_in_block = threadIdx.x;
    const int block_dim_x = blockDim.x; // Threads per block (e.g., 256)

    if (head_idx >= num_q_heads) return;

    // Map current Q head to its corresponding KV cache head index (for GQA/MQA)
    const int heads_per_kv = num_q_heads / cache_num_kv_heads;
    const int kv_head_idx = head_idx / heads_per_kv;

    // Dynamic shared memory. Layout:
    // s_scores[current_seq_len]: to store Q.K^T results
    // s_reduction_val[1]: for block-wide max and sum_exp
    extern __shared__ float s_data[];
    float* s_scores = s_data;
    float* s_reduction_val = &s_data[current_seq_len]; // Single float for block-wide reduction

    // Pointer to the current Q vector for this specific head
    // Assuming Q_current is already offset for the specific token if in batch generation.
    // Here, Q_current is expected to be just [head_idx * head_dim] for the current token's Q vector.
    const float* q_vec = Q_current + head_idx * head_dim;

    // --- 1. Compute Q.K^T scores and store in s_scores ---
    for (int k_idx = tid_in_block; k_idx < current_seq_len; k_idx += block_dim_x) {
        // Pointer to the k_idx-th K vector in the cache for the relevant kv_head
        const float* k_vec = K_layer_cache_base + ((size_t)k_idx * cache_num_kv_heads + kv_head_idx) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k_vec[d];
        }
        s_scores[k_idx] = score * scale;
    }
    __syncthreads(); // Ensure all scores are computed and stored in s_scores

    // --- 2. Softmax calculation on s_scores ---
    // a. Find max score in s_scores (block-wide reduction)
    if (tid_in_block == 0) *s_reduction_val = -INFINITY;
    __syncthreads();

    float thread_max_score = -INFINITY;
    for (int k_idx = tid_in_block; k_idx < current_seq_len; k_idx += block_dim_x) {
        thread_max_score = fmaxf(thread_max_score, s_scores[k_idx]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) thread_max_score = fmaxf(thread_max_score, __shfl_down_sync(0xFFFFFFFF, thread_max_score, offset));
    if ((tid_in_block % warpSize) == 0) atomicMaxFloat(s_reduction_val, thread_max_score);
    __syncthreads();
    float max_score = *s_reduction_val;
    __syncthreads();  // every thread has read the max before thread 0 reuses the slot

    // b. Subtract max, compute exp, and sum exp (block-wide reduction for sum)
    if (tid_in_block == 0) *s_reduction_val = 0.0f;
    __syncthreads();

    float thread_sum_exp = 0.0f;
    for (int k_idx = tid_in_block; k_idx < current_seq_len; k_idx += block_dim_x) {
        float val = expf(s_scores[k_idx] - max_score);
        s_scores[k_idx] = val; // Store exp(score - max)
        thread_sum_exp += val;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) thread_sum_exp += __shfl_down_sync(0xFFFFFFFF, thread_sum_exp, offset);
    if ((tid_in_block % warpSize) == 0) atomicAddFloat(s_reduction_val, thread_sum_exp);
    __syncthreads();
    float sum_exp = *s_reduction_val;
    float inv_sum_exp = (sum_exp == 0.0f || sum_exp < 1e-9f) ? 0.0f : 1.0f / sum_exp; // add small epsilon for stability

    // c. Normalize s_scores (they now contain softmax probabilities)
    for (int k_idx = tid_in_block; k_idx < current_seq_len; k_idx += block_dim_x) {
        s_scores[k_idx] *= inv_sum_exp;
    }
    __syncthreads();

    // --- 3. Multiply by V and accumulate to output ---
    // Output for this head: out_for_this_head[d_out]
    float* out_for_this_head = out + head_idx * head_dim;

    for (int d_out = tid_in_block; d_out < head_dim; d_out += block_dim_x) {
        float weighted_val = 0.0f;
        for (int k_idx = 0; k_idx < current_seq_len; ++k_idx) { // All threads iterate full k_idx to sum contributions
            const float* v_vec = V_layer_cache_base + ((size_t)k_idx * cache_num_kv_heads + kv_head_idx) * head_dim;
            weighted_val += s_scores[k_idx] * v_vec[d_out];
        }
        out_for_this_head[d_out] = weighted_val; // Each thread writes to its assigned output dimension part
    }
}

/**
 * @brief Optimized wrappers using the new high-performance kernels
 */

void rmsnorm_vector_cuda_optimized(const float* x_dev, const float* weight_dev,
                                 float* out_dev, int n, float eps,
                                 cudaStream_t stream) {
    if (n <= 0) return;
    const int block_size = 256;
    // Ensure grid_size is 1 as the kernel is designed for block-wide reduction.
    const int grid_size = 1;
    rmsnorm_optimized_kernel<<<grid_size, block_size, sizeof(float), stream>>>(
        x_dev, weight_dev, out_dev, n, eps);
    gpuErrchk(cudaGetLastError());
}

void softmax_vector_cuda_optimized(const float* x_dev, float* out_dev, int n,
                                 cudaStream_t stream) {
    if (n <= 0) return;
    const int block_size = 256;
    // Ensure grid_size is 1 as the kernel is designed for block-wide reduction.
    const int grid_size = 1;
    // Shared memory: 2 floats (max_val, sum_exp)
    softmax_optimized_kernel<<<grid_size, block_size, 2 * sizeof(float), stream>>>(
        x_dev, out_dev, n);
    gpuErrchk(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Split-K decode attention ("flash decoding") for one new token.
// Phase 1: grid (heads, chunks); each block takes CHUNK consecutive keys of
// one head: warps compute q.k with lanes striding head_dim (coalesced K rows)
// and a warp reduction, then a chunk-local softmax, and write a partial
// (max m, sum l, unnormalised sum_k p_k v_k) to a static scratch array.
// Phase 2: one block per head merges the chunks:
//   out = sum_c e^{m_c - M} acc_c / sum_c e^{m_c - M} l_c.
// Chunks past the current length write l = 0 and are skipped, so a CUDA
// graph can capture the grid for the cache's full length while the length
// itself comes from device memory (step_dev[1] = pos). The scratch is a
// static __device__ array (nothing allocated, graph-safe; one decode stream
// at a time). EMBER_DECODE_ATTN=legacy selects the one-block-per-head kernel.
namespace {
constexpr int kDecChunkMax = 2048;   // keys per chunk at most (shared memory)
constexpr int kDecChunksMax = 64;    // chunks per head at most
constexpr int kDecHeadsMax = 128;
constexpr int kDecHeadDimMax = 256;
constexpr int kDecThreads = 128;
__device__ float g_dec_partial[(size_t)kDecHeadsMax * kDecChunksMax * (kDecHeadDimMax + 2)];

__global__ void decode_attn_split_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                         const float* __restrict__ V, int seq_len, int head_dim, float scale,
                                         int num_kv_heads, int num_heads, int chunk, const int* step_dev) {
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

__global__ void decode_attn_merge_kernel(float* __restrict__ out, int head_dim, int nc) {
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

bool decode_attn_legacy() {
    static const bool legacy = [] {
        const char* e = std::getenv("EMBER_DECODE_ATTN");
        return e && std::string(e) == "legacy";
    }();
    return legacy;
}

// Launches the split kernel for a cache of max_len positions; false if the
// shape does not fit the scratch (the caller then uses the legacy kernel).
bool decode_attn_split(const float* Q, const float* K, const float* V, float* out, int num_heads, int seq_len,
                       int head_dim, float scale, int max_len, int num_kv_heads, const int* step_dev,
                       cudaStream_t stream) {
    if (decode_attn_legacy() || num_heads > kDecHeadsMax || head_dim > kDecHeadDimMax || max_len <= 0) return false;
    int nc = (max_len + 255) / 256;
    if (nc > kDecChunksMax) nc = kDecChunksMax;
    const int chunk = (max_len + nc - 1) / nc;
    if (chunk > kDecChunkMax) return false;
    const size_t shm = (size_t)(head_dim + chunk) * sizeof(float);
    decode_attn_split_kernel<<<dim3(num_heads, nc), kDecThreads, shm, stream>>>(
        Q, K, V, seq_len, head_dim, scale, num_kv_heads, num_heads, chunk, step_dev);
    decode_attn_merge_kernel<<<num_heads, kDecThreads, 0, stream>>>(out, head_dim, nc);
    gpuErrchk(cudaGetLastError());
    return true;
}
}  // namespace

void attention_cuda_optimized(const float* Q_current_dev, const float* K_layer_cache_base,
                             const float* V_layer_cache_base, float* out_dev,
                             int num_heads, int current_seq_len, int head_dim,
                             float scale, int cache_max_seq_len, int cache_num_kv_heads,
                             cudaStream_t stream) {
    if (current_seq_len == 0) return;
    if (decode_attn_split(Q_current_dev, K_layer_cache_base, V_layer_cache_base, out_dev, num_heads,
                          current_seq_len, head_dim, scale, current_seq_len, cache_num_kv_heads, nullptr, stream))
        return;
    const int threads_per_block = 256;
    size_t shared_mem_size = (cache_max_seq_len + 1) * sizeof(float);

    attention_optimized_kernel<<<num_heads, threads_per_block, shared_mem_size, stream>>>(
        Q_current_dev, K_layer_cache_base, V_layer_cache_base, out_dev,
        current_seq_len, head_dim, scale, cache_num_kv_heads, num_heads, nullptr);
    gpuErrchk(cudaGetLastError());
}

// New BF16 Tensor Core Matrix-Matrix Operations

// Kernel for FP32 to BF16 conversion
__global__ void convert_fp32_to_bf16_kernel(const float* __restrict__ fp32_in,
                                            uint16_t* __restrict__ bf16_out,
                                            size_t n_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        bf16_out[idx] = float32_to_bf16_device(fp32_in[idx]);
    }
}

// Host wrapper for FP32 to BF16 conversion
void convert_fp32_to_bf16_cuda(const float* fp32_in_dev, uint16_t* bf16_out_dev, 
                               size_t n_elements, cudaStream_t stream) {
    if (n_elements == 0) return;
    
    const int threads = 256;
    const int blocks = (n_elements + threads - 1) / threads;
    
    convert_fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(fp32_in_dev, bf16_out_dev, n_elements);
    CUDA_CHECK(cudaGetLastError());
}

// BF16 x BF16 = BF16 Matrix-Matrix multiplication using Tensor Cores
void gemm_bf16_bf16_cuda(cublasHandle_t handle, 
                         bool transa_user, bool transb_user, 
                         int m_user, int n_user, int k_user, 
                         const float* alpha_user, 
                         const uint16_t* A_bf16_user, int lda_user, 
                         const uint16_t* B_bf16_user, int ldb_user, 
                         const float* beta_user, 
                         uint16_t* C_bf16_user, int ldc_user, 
                         cudaStream_t stream) {
    
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSetStream failed in gemm_bf16_bf16_cuda with error: " + std::to_string(status));
        throw std::runtime_error("cublasSetStream failed");
    }

    // Log parameters for debugging
    Logger::info("[GEMM_BF16_BF16] M=" + std::to_string(m_user) + ", N=" + std::to_string(n_user) + 
                 ", K=" + std::to_string(k_user) + ", LDA=" + std::to_string(lda_user) + 
                 ", LDB=" + std::to_string(ldb_user) + ", LDC=" + std::to_string(ldc_user));

    cublasOperation_t opA_cublas, opB_cublas;
    int M_cublas, N_cublas, K_cublas;
    const uint16_t *A_cublas_ptr, *B_cublas_ptr;
    int LDA_cublas, LDB_cublas;

    // Apply the same column-major transformation as gemm_f32_f32_cuda
    M_cublas = n_user;
    N_cublas = m_user;
    K_cublas = k_user;

    A_cublas_ptr = B_bf16_user;    // B_user data is the first matrix for cuBLAS call
    LDA_cublas = (transb_user) ? k_user : ldb_user;

    B_cublas_ptr = A_bf16_user;    // A_user data is the second matrix for cuBLAS call
    LDB_cublas = (transa_user) ? m_user : lda_user;

    // Determine cuBLAS operations
    if (!transa_user && !transb_user) {
        opA_cublas = CUBLAS_OP_N; // for B_user
        opB_cublas = CUBLAS_OP_N; // for A_user
    } else if (transa_user && !transb_user) {
        opA_cublas = CUBLAS_OP_N; // for B_user
        opB_cublas = CUBLAS_OP_T; // for A_user (A^T)
    } else if (!transa_user && transb_user) {
        opA_cublas = CUBLAS_OP_T; // for B_user (B^T)
        opB_cublas = CUBLAS_OP_N; // for A_user
    } else { // transa_user && transb_user
        opA_cublas = CUBLAS_OP_T; // for B_user (B^T)
        opB_cublas = CUBLAS_OP_T; // for A_user (A^T)
    }
    
    int LDC_cublas = ldc_user;

    // Use cublasGemmEx with BF16 Tensor Cores
    status = cublasGemmEx(handle, 
                          opA_cublas, opB_cublas, 
                          M_cublas, N_cublas, K_cublas, 
                          alpha_user, 
                          A_cublas_ptr, CUDA_R_16BF, LDA_cublas, 
                          B_cublas_ptr, CUDA_R_16BF, LDB_cublas, 
                          beta_user, 
                          C_bf16_user, CUDA_R_16BF, LDC_cublas, 
                          CUBLAS_COMPUTE_32F, // Use FP32 for accumulation for stability
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP); // Enable Tensor Cores

    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasGemmEx BF16 failed in gemm_bf16_bf16_cuda with error: " + std::to_string(status));
        Logger::error("GEMM params: transa=" + std::to_string(transa_user) + " transb=" + std::to_string(transb_user) + 
                     " m=" + std::to_string(m_user) + " n=" + std::to_string(n_user) + " k=" + std::to_string(k_user));
        throw std::runtime_error("cublasGemmEx BF16 failed");
    }
    
    Logger::info("[GEMM_BF16_BF16] SUCCESS with Tensor Cores");
}

// FP32 x BF16 = FP32 Matrix-Matrix multiplication (for inputs that need to stay in FP32)
void gemm_f32_to_bf16_f32_cuda(cublasHandle_t handle, 
                               bool transa_user, bool transb_user, 
                               int m_user, int n_user, int k_user, 
                               const float* alpha_user, 
                               const float* A_f32_user, int lda_user, 
                               const uint16_t* B_bf16_user, int ldb_user, 
                               const float* beta_user, 
                               float* C_f32_user, int ldc_user, 
                               cudaStream_t stream) {
    
    cublasStatus_t stream_status = cublasSetStream(handle, stream);
    if (stream_status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasSetStream failed in gemm_f32_to_bf16_f32_cuda with error: " + std::to_string(stream_status));
        throw std::runtime_error("cublasSetStream failed");
    }
    
    // Convert FP32 input A to BF16
    size_t A_size = (size_t)m_user * k_user;
    uint16_t* A_bf16_temp;
    CUDA_CHECK(cudaMalloc(&A_bf16_temp, A_size * sizeof(uint16_t)));
    convert_fp32_to_bf16_cuda(A_f32_user, A_bf16_temp, A_size, stream);
    
    // Allocate BF16 output buffer
    size_t C_size = (size_t)m_user * n_user;
    uint16_t* C_bf16_temp;
    CUDA_CHECK(cudaMalloc(&C_bf16_temp, C_size * sizeof(uint16_t)));
    
    // Convert alpha and beta to BF16 for host-side pointers
    uint16_t alpha_bf16_host = float32_to_bf16_host(*alpha_user);
    uint16_t beta_bf16_host = float32_to_bf16_host(*beta_user);
    
    // DIRECT parameter mapping for cublasGemmEx
    cublasOperation_t opA = transa_user ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transb_user ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    // A is A_bf16_temp (converted from A_f32_user)
    // B is B_bf16_user (already BF16)
    // C is C_bf16_temp (will be converted back to C_f32_user)
    cublasStatus_t status = cublasGemmEx(handle,
        opA, opB,                
        m_user, n_user, k_user, 
        &alpha_bf16_host, // Pointer to BF16 host alpha
        A_bf16_temp, CUDA_R_16BF, lda_user,
        B_bf16_user, CUDA_R_16BF, ldb_user,
        &beta_bf16_host,  // Pointer to BF16 host beta
        C_bf16_temp, CUDA_R_16BF, ldc_user,
        CUBLAS_COMPUTE_16F, // Compute in BF16 (A_16*B_16 -> C_16)
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        Logger::error("cublasGemmEx BF16 (A_bf16*B_bf16->C_bf16) failed in gemm_f32_to_bf16_f32_cuda with error: " + std::to_string(status));
        Logger::error("GEMM params: transA=" + std::to_string(opA) + " transB=" + std::to_string(opB) + 
                     " m=" + std::to_string(m_user) + " n=" + std::to_string(n_user) + " k=" + std::to_string(k_user) + 
                     " lda=" + std::to_string(lda_user) + " ldb=" + std::to_string(ldb_user) + " ldc=" + std::to_string(ldc_user));
        CUDA_CHECK(cudaFree(A_bf16_temp));
        CUDA_CHECK(cudaFree(C_bf16_temp));
        throw std::runtime_error("BF16 Tensor Core GEMM failed");
    }
    
    // Convert BF16 result back to FP32
    convert_bf16_to_fp32_cuda(C_bf16_temp, C_f32_user, C_size, stream);
    
    // Cleanup temporary buffers
    CUDA_CHECK(cudaFree(A_bf16_temp));
    CUDA_CHECK(cudaFree(C_bf16_temp));
}

// Host wrapper for BF16 to FP32 conversion
void convert_bf16_to_fp32_cuda(const uint16_t* bf16_in_dev, float* fp32_out_dev, 
                               size_t n_elements, cudaStream_t stream) {
    if (n_elements == 0) return;
    
    const int threads = 256;
    const int blocks = (n_elements + threads - 1) / threads;
    
    convert_bf16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(bf16_in_dev, fp32_out_dev, n_elements);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Graph decode (19 Oct 2026). One decode step is captured once as a CUDA
// graph and replayed per token, so every per-token value lives in device
// memory: step_dev[0] = input token id, step_dev[1] = position. These are
// the device-position forms of the decode kernels; the arithmetic is the
// same as the per-launch path's.
// ---------------------------------------------------------------------------
__global__ void decode_embed_kernel(const float* __restrict__ table, float* __restrict__ out,
                                    const int* __restrict__ step_dev, int hidden_size, int vocab_size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden_size) return;
  const int tok = step_dev[0];
  out[i] = (tok >= 0 && tok < vocab_size) ? table[(size_t)tok * hidden_size + i] : 0.0f;
}

void decode_embed_cuda(const float* table_dev, float* out_dev, const int* step_dev, int hidden_size,
                       int vocab_size, cudaStream_t stream) {
  decode_embed_kernel<<<(hidden_size + 255) / 256, 256, 0, stream>>>(table_dev, out_dev, step_dev, hidden_size,
                                                                     vocab_size);
  gpuErrchk(cudaGetLastError());
}

__global__ void rope_devpos_kernel(float* x, int num_heads, int head_dim, const float* all_freqs_cis_base,
                                   const int* step_dev, bool use_adjacent_pairing) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_pairs = num_heads * (head_dim / 2);
  if (idx >= total_pairs) return;
  const int pos = step_dev[1];
  const int head_idx = idx / (head_dim / 2);
  const int dim_pair_idx = idx % (head_dim / 2);
  const size_t freq_base_offset = ((size_t)pos * (head_dim / 2) + dim_pair_idx) * 2;
  int base_x0, base_x1;
  if (use_adjacent_pairing) {
    base_x0 = head_idx * head_dim + (dim_pair_idx * 2);
    base_x1 = head_idx * head_dim + (dim_pair_idx * 2 + 1);
  } else {
    base_x0 = head_idx * head_dim + dim_pair_idx;
    base_x1 = head_idx * head_dim + dim_pair_idx + (head_dim / 2);
  }
  const float x0 = x[base_x0];
  const float x1 = x[base_x1];
  const float cos_val = all_freqs_cis_base[freq_base_offset];
  const float sin_val = all_freqs_cis_base[freq_base_offset + 1];
  x[base_x0] = x0 * cos_val - x1 * sin_val;
  x[base_x1] = x0 * sin_val + x1 * cos_val;
}

void rope_devpos_cuda(float* x_dev, int num_heads, int head_dim, const float* all_freqs_cis_dev_base,
                      const int* step_dev, bool use_adjacent_pairing, cudaStream_t stream) {
  const int total_pairs = num_heads * (head_dim / 2);
  rope_devpos_kernel<<<(total_pairs + 255) / 256, 256, 0, stream>>>(x_dev, num_heads, head_dim,
                                                                   all_freqs_cis_dev_base, step_dev,
                                                                   use_adjacent_pairing);
  gpuErrchk(cudaGetLastError());
}

// The KV cache is [pos][kv_head][dim]: position pos's row is `row` floats,
// laid out like k/v.
__global__ void kv_write_devpos_kernel(float* k_cache, float* v_cache, const float* __restrict__ k,
                                       const float* __restrict__ v, const int* step_dev, int row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= row) return;
  const size_t off = (size_t)step_dev[1] * row + i;
  k_cache[off] = k[i];
  v_cache[off] = v[i];
}

void kv_write_devpos_cuda(float* k_cache_dev, float* v_cache_dev, const float* k_dev, const float* v_dev,
                          const int* step_dev, int row, cudaStream_t stream) {
  kv_write_devpos_kernel<<<(row + 255) / 256, 256, 0, stream>>>(k_cache_dev, v_cache_dev, k_dev, v_dev, step_dev,
                                                               row);
  gpuErrchk(cudaGetLastError());
}

void attention_devpos_cuda(const float* Q_current_dev, const float* K_layer_cache_base,
                           const float* V_layer_cache_base, float* out_dev, int num_heads, int head_dim,
                           float scale, int cache_max_seq_len, int cache_num_kv_heads, const int* step_dev,
                           cudaStream_t stream) {
  if (decode_attn_split(Q_current_dev, K_layer_cache_base, V_layer_cache_base, out_dev, num_heads,
                        /*seq_len=*/0, head_dim, scale, cache_max_seq_len, cache_num_kv_heads, step_dev, stream))
    return;
  const size_t shared_mem_size = (cache_max_seq_len + 1) * sizeof(float);
  attention_optimized_kernel<<<num_heads, 256, shared_mem_size, stream>>>(
      Q_current_dev, K_layer_cache_base, V_layer_cache_base, out_dev, /*current_seq_len=*/0, head_dim, scale,
      cache_num_kv_heads, num_heads, step_dev);
  gpuErrchk(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// Mixture-of-experts FFN (see cuda_kernels.h).
namespace {
// Row-major Y[T][O] = X[T][I] W[O][I]^T with column-major cuBLAS: Y^T = W X^T.
void moe_gemm_rm(cublasHandle_t h, const float* X, const float* W, float* Y, int T, int O, int I) {
  const float one = 1.0f, zero = 0.0f;
  cublasStatus_t s = cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, O, T, I, &one, W, I, X, I, &zero, Y, O);
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("moe_ffn_cuda: cublasSgemm failed");
}

// One token per block: top-k of the E router logits (ties to the lower
// index), then the gates exactly as moe_ffn_f32_cpu computes them.
__global__ void moe_route_kernel(const float* __restrict__ logits, float* __restrict__ gates, float* __restrict__ sel,
                                 int E, int k, int renorm) {
  if (threadIdx.x != 0) return;
  const int t = blockIdx.x;
  const float* r = logits + (size_t)t * E;
  float* g = gates + (size_t)t * E;
  float* s = sel + (size_t)t * E;
  for (int e = 0; e < E; ++e) { g[e] = 0.0f; s[e] = 0.0f; }
  for (int j = 0; j < k && j < E; ++j) {
    int best = -1;
    for (int e = 0; e < E; ++e)
      if (s[e] == 0.0f && (best < 0 || r[e] > r[best])) best = e;
    s[best] = 1.0f;
  }
  float mx = -INFINITY;
  for (int e = 0; e < E; ++e)
    if (!renorm || s[e] != 0.0f) mx = fmaxf(mx, r[e]);
  double z = 0.0;
  for (int e = 0; e < E; ++e)
    if (!renorm || s[e] != 0.0f) z += exp((double)(r[e] - mx));
  for (int e = 0; e < E; ++e)
    if (s[e] != 0.0f) g[e] = (float)(exp((double)(r[e] - mx)) / z);
}

__global__ void moe_swiglu_kernel(float* __restrict__ g, const float* __restrict__ u, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float gv = g[i];
    g[i] = gv * (1.0f / (1.0f + expf(-gv))) * u[i];
  }
}

// out[t][i] = sum over the selected experts, ascending, of gate * y[e][t][i].
__global__ void moe_combine_kernel(const float* __restrict__ y, const float* __restrict__ gates,
                                   const float* __restrict__ sel, float* __restrict__ out, int T, int d, int E) {
  const int t = blockIdx.x;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float acc = 0.0f;
    for (int e = 0; e < E; ++e)
      if (sel[(size_t)t * E + e] != 0.0f) acc += gates[(size_t)t * E + e] * y[((size_t)e * T + t) * d + i];
    out[(size_t)t * d + i] = acc;
  }
}
}  // namespace

namespace {
bool moe_use_routed(int T, int k) {
  static const bool dense = [] {
    const char* e = std::getenv("EMBER_MOE_DENSE");
    return e && std::string(e) == "1";
  }();
  return !dense && T * k <= 32;
}
size_t moe_routed_floats(int T, int d, int d_ff, int E, int k) {
  const size_t tk = (size_t)T * k;
  // pointer arrays first (9 x tk pointers, 2 floats each), then logits [T][E],
  // gates [tk], g/u [tk][d_ff], y [tk][d]
  return 18 * tk + (size_t)T * E + tk + 2 * tk * d_ff + tk * d;
}

// Top-k per token (ties to the lower index), the selected indices in
// ascending order with their gates (as moe_ffn_f32_cpu), and the pointer
// arrays for the batched expert GEMMs.
__global__ void moe_route_sel_kernel(const float* __restrict__ logits, int E, int k, int renorm, const float* x,
                                     int d, int d_ff, const float* gate_exps, const float* up_exps,
                                     const float* down_exps, float* gsel, float* g, float* u, float* y,
                                     const float** Ag, const float** Au, const float** Ad, const float** Bx,
                                     const float** Bg, float** Cg, float** Cu, float** Cy) {
  if (threadIdx.x != 0) return;
  const int t = blockIdx.x;
  const float* r = logits + (size_t)t * E;
  int idx[64];
  bool taken[256];
  for (int e = 0; e < E; ++e) taken[e] = false;
  for (int j = 0; j < k; ++j) {
    int best = -1;
    for (int e = 0; e < E; ++e)
      if (!taken[e] && (best < 0 || r[e] > r[best])) best = e;
    taken[best] = true;
    idx[j] = best;
  }
  for (int a = 1; a < k; ++a)  // ascending expert index
    for (int b = a; b > 0 && idx[b - 1] > idx[b]; --b) { const int tmp = idx[b]; idx[b] = idx[b - 1]; idx[b - 1] = tmp; }
  float mx = -INFINITY;
  for (int e = 0; e < E; ++e)
    if (!renorm || taken[e]) mx = fmaxf(mx, r[e]);
  double z = 0.0;
  for (int e = 0; e < E; ++e)
    if (!renorm || taken[e]) z += exp((double)(r[e] - mx));
  for (int j = 0; j < k; ++j) {
    const int e = idx[j];
    const size_t s = (size_t)t * k + j;
    gsel[s] = (float)(exp((double)(r[e] - mx)) / z);
    Ag[s] = gate_exps + (size_t)e * d_ff * d;
    Au[s] = up_exps + (size_t)e * d_ff * d;
    Ad[s] = down_exps + (size_t)e * d * d_ff;
    Bx[s] = x + (size_t)t * d;
    Bg[s] = g + s * d_ff;
    Cg[s] = g + s * d_ff;
    Cu[s] = u + s * d_ff;
    Cy[s] = y + s * d;
  }
}

__global__ void moe_combine_sel_kernel(const float* __restrict__ y, const float* __restrict__ gsel,
                                       float* __restrict__ out, int d, int k) {
  const int t = blockIdx.x;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) acc += gsel[(size_t)t * k + j] * y[((size_t)t * k + j) * d + i];
    out[(size_t)t * d + i] = acc;
  }
}
}  // namespace

size_t moe_ffn_scratch_floats(int T, int d, int d_ff, int E, int k) {
  const size_t dense = (size_t)T * E * 3 + (size_t)T * d_ff * 2 + (size_t)E * T * d;
  return std::max(dense, moe_routed_floats(T, d, d_ff, E, k));
}

void moe_ffn_cuda(cublasHandle_t handle, const float* x, int T, int d, int d_ff, int E, int k, bool renorm,
                  const float* router, const float* gate_exps, const float* up_exps, const float* down_exps,
                  float* scratch, float* out, cudaStream_t stream) {
  if (moe_use_routed(T, k) && E <= 256 && k <= 64) {
    const int tk = T * k;
    const float** Ag = reinterpret_cast<const float**>(scratch);
    const float** Au = Ag + tk;
    const float** Ad = Au + tk;
    const float** Bx = Ad + tk;
    const float** Bg = Bx + tk;
    float** Cg = reinterpret_cast<float**>(scratch) + 5 * (size_t)tk;
    float** Cu = Cg + tk;
    float** Cy = Cu + tk;
    float* logits = reinterpret_cast<float*>(Cy + tk);
    float* gsel = logits + (size_t)T * E;
    float* g = gsel + tk;
    float* u = g + (size_t)tk * d_ff;
    float* y = u + (size_t)tk * d_ff;
    cublasSetStream(handle, stream);
    moe_gemm_rm(handle, x, router, logits, T, E, d);
    moe_route_sel_kernel<<<T, 32, 0, stream>>>(logits, E, k, renorm ? 1 : 0, x, d, d_ff, gate_exps, up_exps,
                                               down_exps, gsel, g, u, y, Ag, Au, Ad, Bx, Bg, Cg, Cu, Cy);
    const float one = 1.0f, zero = 0.0f;
    // Each item: row-major [rows][cols] weight times a column vector (see moe_gemm_rm).
    if (cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, d_ff, 1, d, &one, Ag, d, Bx, d, &zero, Cg, d_ff, tk) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, d_ff, 1, d, &one, Au, d, Bx, d, &zero, Cu, d_ff, tk) !=
            CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("moe_ffn_cuda: cublasSgemmBatched failed");
    const size_t n = (size_t)tk * d_ff;
    moe_swiglu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(g, u, n);
    if (cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, d, 1, d_ff, &one, Ad, d_ff, Bg, d_ff, &zero, Cy, d, tk) !=
        CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("moe_ffn_cuda: cublasSgemmBatched failed");
    moe_combine_sel_kernel<<<T, 128, 0, stream>>>(y, gsel, out, d, k);
    gpuErrchk(cudaGetLastError());
    return;
  }
  float* logits = scratch;
  float* gates = logits + (size_t)T * E;
  float* sel = gates + (size_t)T * E;
  float* g = sel + (size_t)T * E;
  float* u = g + (size_t)T * d_ff;
  float* y = u + (size_t)T * d_ff;
  cublasSetStream(handle, stream);
  moe_gemm_rm(handle, x, router, logits, T, E, d);
  moe_route_kernel<<<T, 32, 0, stream>>>(logits, gates, sel, E, k, renorm ? 1 : 0);
  const size_t n = (size_t)T * d_ff;
  for (int e = 0; e < E; ++e) {
    moe_gemm_rm(handle, x, gate_exps + (size_t)e * d_ff * d, g, T, d_ff, d);
    moe_gemm_rm(handle, x, up_exps + (size_t)e * d_ff * d, u, T, d_ff, d);
    moe_swiglu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(g, u, n);
    moe_gemm_rm(handle, g, down_exps + (size_t)e * d * d_ff, y + (size_t)e * T * d, T, d, d_ff);
  }
  moe_combine_kernel<<<T, 128, 0, stream>>>(y, gates, sel, out, T, d, E);
  gpuErrchk(cudaGetLastError());
}
