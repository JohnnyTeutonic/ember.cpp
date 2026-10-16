#include "model.h"
#include "logger.h"

#include <algorithm>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "cuda_kernels.h"

void TinyLlamaModel::allocate_persistent_batch_buffers() {
    if (d_persistent_batch_input_ != nullptr) {
        Logger::info("[GPU_OPT] Persistent batch buffers already allocated");
        return;
    }
    
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    int n_kv_heads = config_.num_key_value_heads;
    int head_dim = hs / config_.num_attention_heads;
    if (persistent_batch_capacity_ < MAX_BATCH_TOKENS) persistent_batch_capacity_ = MAX_BATCH_TOKENS;
    const size_t cap = static_cast<size_t>(persistent_batch_capacity_);
    
    size_t input_size = cap * hs * sizeof(float);
    size_t norm_size = cap * hs * sizeof(float);
    size_t residual_size = cap * hs * 2 * sizeof(float);
    size_t q_size = cap * hs * sizeof(float);
    size_t k_size = cap * n_kv_heads * head_dim * sizeof(float);
    size_t v_size = cap * n_kv_heads * head_dim * sizeof(float);
    size_t attn_out_size = cap * hs * sizeof(float);
    size_t attn_proj_size = cap * hs * sizeof(float);
    size_t gate_size = cap * is * sizeof(float);
    size_t up_size = cap * is * sizeof(float);
    size_t swiglu_size = cap * is * sizeof(float);
    size_t mlp_down_size = cap * hs * sizeof(float);
    
    size_t total_size = input_size + norm_size + residual_size + q_size + k_size + v_size +
                       attn_out_size + attn_proj_size + gate_size + up_size + swiglu_size + mlp_down_size;
    
    Logger::info("[GPU_OPT] Allocating persistent batch buffers: " + 
                 std::to_string(total_size / (1024*1024)) + " MB total");
    
    gpuErrchk(cudaMalloc(&d_persistent_batch_input_, input_size));
    gpuErrchk(cudaMalloc(&d_persistent_batch_norm_out_, norm_size));
    gpuErrchk(cudaMalloc(&d_persistent_batch_residual_, residual_size));
    gpuErrchk(cudaMalloc(&d_persistent_q_batch_, q_size));
    gpuErrchk(cudaMalloc(&d_persistent_k_batch_, k_size));
    gpuErrchk(cudaMalloc(&d_persistent_v_batch_, v_size));
    gpuErrchk(cudaMalloc(&d_persistent_attn_output_, attn_out_size));
    gpuErrchk(cudaMalloc(&d_persistent_attn_proj_out_, attn_proj_size));
    gpuErrchk(cudaMalloc(&d_persistent_gate_proj_out_, gate_size));
    gpuErrchk(cudaMalloc(&d_persistent_up_proj_out_, up_size));
    gpuErrchk(cudaMalloc(&d_persistent_swiglu_out_, swiglu_size));
    gpuErrchk(cudaMalloc(&d_persistent_mlp_down_out_, mlp_down_size));
    
    Logger::info("[GPU_OPT] Successfully allocated persistent batch buffers");
}

void TinyLlamaModel::free_persistent_batch_buffers() {
    if (d_persistent_batch_input_) {
        Logger::info("[GPU_OPT] Freeing persistent batch buffers");
        gpuErrchk(cudaFree(d_persistent_batch_input_));
        gpuErrchk(cudaFree(d_persistent_batch_norm_out_));
        gpuErrchk(cudaFree(d_persistent_batch_residual_));
        gpuErrchk(cudaFree(d_persistent_q_batch_));
        gpuErrchk(cudaFree(d_persistent_k_batch_));
        gpuErrchk(cudaFree(d_persistent_v_batch_));
        gpuErrchk(cudaFree(d_persistent_attn_output_));
        gpuErrchk(cudaFree(d_persistent_attn_proj_out_));
        gpuErrchk(cudaFree(d_persistent_gate_proj_out_));
        gpuErrchk(cudaFree(d_persistent_up_proj_out_));
        gpuErrchk(cudaFree(d_persistent_swiglu_out_));
        gpuErrchk(cudaFree(d_persistent_mlp_down_out_));
        
        d_persistent_batch_input_ = nullptr;
        d_persistent_batch_norm_out_ = nullptr;
        d_persistent_batch_residual_ = nullptr;
        d_persistent_q_batch_ = nullptr;
        d_persistent_k_batch_ = nullptr;
        d_persistent_v_batch_ = nullptr;
        d_persistent_attn_output_ = nullptr;
        d_persistent_attn_proj_out_ = nullptr;
        d_persistent_gate_proj_out_ = nullptr;
        d_persistent_up_proj_out_ = nullptr;
        d_persistent_swiglu_out_ = nullptr;
        d_persistent_mlp_down_out_ = nullptr;
    }
}

void TinyLlamaModel::resize_persistent_batch_buffers_if_needed(int required_batch_size) {
    if (d_persistent_batch_input_ != nullptr && required_batch_size <= persistent_batch_capacity_) {
        return;
    }
    // Grow to fit. This used to only warn above MAX_BATCH_TOKENS and keep
    // the 2048-token buffers, so a longer prefill wrote past them (a CUDA
    // invalid-argument error at 4096 tokens; docs/TTFT.md). One
    // reallocation on the first long prompt, none after.
    if (d_persistent_batch_input_ != nullptr) {
        Logger::info("[GPU_OPT] Growing persistent batch buffers from " +
                     std::to_string(persistent_batch_capacity_) + " to " +
                     std::to_string(required_batch_size) + " tokens");
        free_persistent_batch_buffers();
    }
    persistent_batch_capacity_ = std::max(MAX_BATCH_TOKENS, required_batch_size);
    allocate_persistent_batch_buffers();
}

#endif // HAS_CUDA 