#include "weight_management.h"

#include <algorithm>
#include <mutex>
#include "ggml_types.h"
#include "logger.h"
#include "quantization.h"
#include "utils.h"
#ifdef HAS_CUDA
#include "cuda_kernels.h"
#endif

void TinyLlamaModel::ensure_embed_tokens_dequantized() {
    if (!this->embed_tokens_f32.empty()) return;
    
    size_t total_elements_embed = static_cast<size_t>(config_.vocab_size) * config_.hidden_size;
    if (!this->embed_tokens_q6k.empty()) {
        dequantize_vector_q6k_to_f32(this->embed_tokens_q6k, this->embed_tokens_f32, total_elements_embed, 1);
    } else if (!this->embed_tokens_q4k.empty()) {
        dequantize_vector_q4k_to_f32(this->embed_tokens_q4k, this->embed_tokens_f32, total_elements_embed, 1);
    } else if (!this->embed_tokens_q5k.empty()) {
        dequantize_vector_q5k_to_f32(this->embed_tokens_q5k, this->embed_tokens_f32, total_elements_embed);
    } else if (!this->embed_tokens_q8k.empty()) {
        dequantize_q8_k(this->embed_tokens_q8k, this->embed_tokens_f32, total_elements_embed, true);
    } else if (!this->embed_tokens_q8_0.empty()) {
        dequantize_vector_q8_0_to_f32(this->embed_tokens_q8_0, this->embed_tokens_f32, total_elements_embed, 1);
    } else if (!this->embed_tokens.empty()) { 
        this->embed_tokens_f32 = bf16vec_to_float_vec(this->embed_tokens);
    }
}

bool TinyLlamaModel::dequantize_all_if_small(size_t max_params) {
    const size_t hs = config_.hidden_size, is = config_.intermediate_size, vs = config_.vocab_size;
    const size_t kv = hs / std::max(config_.num_attention_heads, 1) * config_.num_key_value_heads;
    const size_t per_layer = hs * hs * 2 + hs * kv * 2 + hs * is * 3;
    const size_t params = vs * hs * 2 + per_layer * config_.num_hidden_layers;
    if (params > max_params) return false;
    ensure_embed_tokens_dequantized();
    ensure_lm_head_dequantized();
    for (auto& lw : layers) {  // MoE router (Q8_0 in paperkiln's Q8_0 exports)
        if (lw.moe_router_f32.empty() && !lw.moe_router_q8_0.empty()) {
            const size_t n = lw.moe_router_q8_0.size() * GGML_QK8_0;
            dequantize_vector_q8_0_to_f32(lw.moe_router_q8_0, lw.moe_router_f32, n, 0);
            lw.moe_router_q8_0.clear();
            lw.moe_router_q8_0.shrink_to_fit();
        }
    }
    for (int l = 0; l < static_cast<int>(layers.size()); ++l) {
        ensure_q_proj_dequantized(l);
        ensure_k_proj_dequantized(l);
        ensure_v_proj_dequantized(l);
        ensure_o_proj_dequantized(l);
        ensure_gate_proj_dequantized(l);
        ensure_up_proj_dequantized(l);
        ensure_down_proj_dequantized(l);
    }
    // Free the quantised copies whose f32 copy now exists: nothing can then
    // take a quantised kernel, and the model is not held twice (WASM memory).
    auto drop = [](const std::vector<float>& f32, auto&... quant) {
        if (f32.empty()) return;
        ((quant.clear(), quant.shrink_to_fit()), ...);
    };
    drop(embed_tokens_f32, embed_tokens_q4k, embed_tokens_q5k, embed_tokens_q6k, embed_tokens_q8_0, embed_tokens_q8k);
    drop(lm_head_f32, lm_head_q4k, lm_head_q5k, lm_head_q6k, lm_head_q8_0, lm_head_q8k);
    for (auto& lw : layers) {
        drop(lw.q_proj_f32, lw.q_proj_q4k, lw.q_proj_q5k, lw.q_proj_q6k, lw.q_proj_q8_0, lw.q_proj_q8k);
        drop(lw.k_proj_f32, lw.k_proj_q4k, lw.k_proj_q5k, lw.k_proj_q6k, lw.k_proj_q8_0, lw.k_proj_q8k);
        drop(lw.v_proj_f32, lw.v_proj_q4k, lw.v_proj_q5k, lw.v_proj_q6k, lw.v_proj_q8_0, lw.v_proj_q8k);
        drop(lw.o_proj_f32, lw.o_proj_q4k, lw.o_proj_q5k, lw.o_proj_q6k, lw.o_proj_q8_0, lw.o_proj_q8k);
        drop(lw.gate_proj_f32, lw.gate_proj_q4k, lw.gate_proj_q5k, lw.gate_proj_q6k, lw.gate_proj_q8_0, lw.gate_proj_q8k);
        drop(lw.up_proj_f32, lw.up_proj_q4k, lw.up_proj_q5k, lw.up_proj_q6k, lw.up_proj_q8_0, lw.up_proj_q8k);
        drop(lw.down_proj_f32, lw.down_proj_q4k, lw.down_proj_q5k, lw.down_proj_q6k, lw.down_proj_q8_0, lw.down_proj_q8k);
    }
    return true;
}

void TinyLlamaModel::ensure_lm_head_dequantized() {
    if (!this->lm_head_f32.empty()) return;
    
    size_t total_elements_lm_head = static_cast<size_t>(config_.vocab_size) * config_.hidden_size;
    if (!this->lm_head_q6k.empty()) {
        dequantize_vector_q6k_to_f32(this->lm_head_q6k, this->lm_head_f32, total_elements_lm_head, 1);
    } else if (!this->lm_head_q4k.empty()) {
        dequantize_vector_q4k_to_f32(this->lm_head_q4k, this->lm_head_f32, total_elements_lm_head, 1);
    } else if (!this->lm_head_q5k.empty()) {
        dequantize_vector_q5k_to_f32(this->lm_head_q5k, this->lm_head_f32, total_elements_lm_head);
    } else if (!this->lm_head_q8k.empty()) {
        dequantize_q8_k(this->lm_head_q8k, this->lm_head_f32, total_elements_lm_head, true);
    } else if (!this->lm_head_q8_0.empty()) {
        dequantize_vector_q8_0_to_f32(this->lm_head_q8_0, this->lm_head_f32, total_elements_lm_head, 1);
    } else if (!this->lm_head.empty()) { 
        this->lm_head_f32 = bf16vec_to_float_vec(this->lm_head);
    }
}

void TinyLlamaModel::ensure_q_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.q_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    size_t q_proj_elements = static_cast<size_t>(hs) * hs;
    Logger::info("[DEQUANT_MEM] Layer " + std::to_string(layer_idx) + ": Q-proj dequantization starting, elements=" + std::to_string(q_proj_elements));
    
    if (!lw.q_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.q_proj_q6k, lw.q_proj_f32, q_proj_elements, 0);
    else if (!lw.q_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.q_proj_q4k, lw.q_proj_f32, q_proj_elements, 0);
    else if (!lw.q_proj_q5k.empty()) dequantize_vector_q5k_to_f32(lw.q_proj_q5k, lw.q_proj_f32, q_proj_elements);
    else if (!lw.q_proj_q8k.empty()) dequantize_q8_k(lw.q_proj_q8k, lw.q_proj_f32, q_proj_elements, true);
    else if (!lw.q_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.q_proj_q8_0, lw.q_proj_f32, q_proj_elements, 0);
    else if (!lw.q_proj.empty()) lw.q_proj_f32 = bf16vec_to_float_vec(lw.q_proj);
    
    Logger::info("[DEQUANT_MEM] Layer " + std::to_string(layer_idx) + ": Q-proj dequantization completed, f32 size=" + std::to_string(lw.q_proj_f32.size()));
}

void TinyLlamaModel::clear_layer_dequantized_weights(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers.size())) {
        Logger::warning("clear_layer_dequantized_weights: Invalid layer index " + std::to_string(layer_idx));
        return;
    }

    // Each f32 copy is cleared only when the weight has a quantised (or BF16) source to re-derive it
    // from: for F32-native weights (transformer_cpp exports) the f32 vector IS the master copy, and
    // clearing it is permanent destruction, not a memory optimisation (first hit 2026-07-20 on the
    // 4-layer mid model: "No Q proj weights" on the second forward pass). Per weight, not per
    // layer, so a layer whose weights mix sources keeps its f32-only ones.
    auto& layer = layers[layer_idx];
    auto clear_if_recoverable = [](std::vector<float>& f32, bool recoverable) {
        if (!recoverable) return 0;
        f32.clear();
        f32.shrink_to_fit();
        return 1;
    };
#define EMBER_RECOVERABLE(w)                                                                              \
    (!layer.w##_q4k.empty() || !layer.w##_q5k.empty() || !layer.w##_q6k.empty() || !layer.w##_q8k.empty() || \
     !layer.w##_q8_0.empty() || !layer.w.empty())
    int cleared = 0;
    cleared += clear_if_recoverable(layer.q_proj_f32, EMBER_RECOVERABLE(q_proj));
    cleared += clear_if_recoverable(layer.k_proj_f32, EMBER_RECOVERABLE(k_proj));
    cleared += clear_if_recoverable(layer.v_proj_f32, EMBER_RECOVERABLE(v_proj));
    cleared += clear_if_recoverable(layer.o_proj_f32, EMBER_RECOVERABLE(o_proj));
    cleared += clear_if_recoverable(layer.gate_proj_f32, EMBER_RECOVERABLE(gate_proj));
    cleared += clear_if_recoverable(layer.up_proj_f32, EMBER_RECOVERABLE(up_proj));
    cleared += clear_if_recoverable(layer.down_proj_f32, EMBER_RECOVERABLE(down_proj));
#undef EMBER_RECOVERABLE
    if (cleared)
        Logger::info("Cleared " + std::to_string(cleared) + " dequantized weights of layer " + std::to_string(layer_idx) +
                     " to save memory.");
}

void TinyLlamaModel::ensure_k_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.k_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    size_t k_proj_elements = static_cast<size_t>(config_.num_key_value_heads * (hs / config_.num_attention_heads)) * hs;
    
    if (!lw.k_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.k_proj_q6k, lw.k_proj_f32, k_proj_elements, 0);
    else if (!lw.k_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.k_proj_q4k, lw.k_proj_f32, k_proj_elements, 0);
    else if (!lw.k_proj_q5k.empty()) dequantize_vector_q5k_to_f32(lw.k_proj_q5k, lw.k_proj_f32, k_proj_elements);
    else if (!lw.k_proj_q8k.empty()) dequantize_q8_k(lw.k_proj_q8k, lw.k_proj_f32, k_proj_elements, false);
    else if (!lw.k_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.k_proj_q8_0, lw.k_proj_f32, k_proj_elements, 0);
    else if (!lw.k_proj.empty()) lw.k_proj_f32 = bf16vec_to_float_vec(lw.k_proj);
}

void TinyLlamaModel::ensure_v_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.v_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    size_t v_proj_elements = static_cast<size_t>(config_.num_key_value_heads * (hs / config_.num_attention_heads)) * hs;
    
    if (!lw.v_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.v_proj_q6k, lw.v_proj_f32, v_proj_elements, 0);
    else if (!lw.v_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.v_proj_q4k, lw.v_proj_f32, v_proj_elements, 0);
    else if (!lw.v_proj_q5k.empty()) dequantize_vector_q5k_to_f32(lw.v_proj_q5k, lw.v_proj_f32, v_proj_elements);
    else if (!lw.v_proj_q8k.empty()) dequantize_q8_k(lw.v_proj_q8k, lw.v_proj_f32, v_proj_elements, false);
    else if (!lw.v_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.v_proj_q8_0, lw.v_proj_f32, v_proj_elements, 0);
    else if (!lw.v_proj.empty()) lw.v_proj_f32 = bf16vec_to_float_vec(lw.v_proj);
}

void TinyLlamaModel::ensure_o_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.o_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    size_t o_proj_elements = static_cast<size_t>(hs) * hs;
    
    if (!lw.o_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.o_proj_q6k, lw.o_proj_f32, o_proj_elements, 0);
    else if (!lw.o_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.o_proj_q4k, lw.o_proj_f32, o_proj_elements, 0);
    else if (!lw.o_proj_q5k.empty()) dequantize_vector_q5k_to_f32(lw.o_proj_q5k, lw.o_proj_f32, o_proj_elements);
    else if (!lw.o_proj_q8k.empty()) dequantize_q8_k(lw.o_proj_q8k, lw.o_proj_f32, o_proj_elements, false);
    else if (!lw.o_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.o_proj_q8_0, lw.o_proj_f32, o_proj_elements, 0);
    else if (!lw.o_proj.empty()) lw.o_proj_f32 = bf16vec_to_float_vec(lw.o_proj);
}

void TinyLlamaModel::ensure_gate_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.gate_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    size_t gate_proj_elements = static_cast<size_t>(is) * hs;
    
    if (!lw.gate_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.gate_proj_q6k, lw.gate_proj_f32, gate_proj_elements, 0);
    else if (!lw.gate_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.gate_proj_q4k, lw.gate_proj_f32, gate_proj_elements, 0);
    else if (!lw.gate_proj_q5k.empty()) dequantize_vector_q5k_to_f32(lw.gate_proj_q5k, lw.gate_proj_f32, gate_proj_elements);
    else if (!lw.gate_proj_q8k.empty()) dequantize_q8_k(lw.gate_proj_q8k, lw.gate_proj_f32, gate_proj_elements, false);
    else if (!lw.gate_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.gate_proj_q8_0, lw.gate_proj_f32, gate_proj_elements, 0);
    else if (!lw.gate_proj.empty()) lw.gate_proj_f32 = bf16vec_to_float_vec(lw.gate_proj);
}

void TinyLlamaModel::ensure_up_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.up_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    size_t up_proj_elements = static_cast<size_t>(is) * hs;
    
    if (!lw.up_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.up_proj_q6k, lw.up_proj_f32, up_proj_elements, 0);
    else if (!lw.up_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.up_proj_q4k, lw.up_proj_f32, up_proj_elements, 0);
    else if (!lw.up_proj_q5k.empty()) dequantize_vector_q5k_to_f32(lw.up_proj_q5k, lw.up_proj_f32, up_proj_elements);
    else if (!lw.up_proj_q8k.empty()) dequantize_q8_k(lw.up_proj_q8k, lw.up_proj_f32, up_proj_elements, false);
    else if (!lw.up_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.up_proj_q8_0, lw.up_proj_f32, up_proj_elements, 0);
    else if (!lw.up_proj.empty()) lw.up_proj_f32 = bf16vec_to_float_vec(lw.up_proj);
}

void TinyLlamaModel::ensure_down_proj_dequantized(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    auto& lw = layers[layer_idx];
    if (!lw.down_proj_f32.empty()) return;
    
    int hs = config_.hidden_size;
    int is = config_.intermediate_size;
    size_t down_proj_elements = static_cast<size_t>(hs) * is;
    
    if (!lw.down_proj_q6k.empty()) dequantize_vector_q6k_to_f32(lw.down_proj_q6k, lw.down_proj_f32, down_proj_elements, 0);
    else if (!lw.down_proj_q4k.empty()) dequantize_vector_q4k_to_f32(lw.down_proj_q4k, lw.down_proj_f32, down_proj_elements, 0);
    else if (!lw.down_proj_q5k.empty()) dequantize_vector_q5k_to_f32(lw.down_proj_q5k, lw.down_proj_f32, down_proj_elements);
    else if (!lw.down_proj_q8k.empty()) dequantize_q8_k(lw.down_proj_q8k, lw.down_proj_f32, down_proj_elements, false);
    else if (!lw.down_proj_q8_0.empty()) dequantize_vector_q8_0_to_f32(lw.down_proj_q8_0, lw.down_proj_f32, down_proj_elements, 0);
    else if (!lw.down_proj.empty()) lw.down_proj_f32 = bf16vec_to_float_vec(lw.down_proj);
}

#ifdef HAS_CUDA
void TinyLlamaModel::ensure_f32_concatenated_weights_loaded() {
  // OPTIMIZED: Use concatenated weights for maximum GPU performance.
  // Called once per layer on every decode step: the early return comes
  // before any logging (a log line per call was ~0.5 ms per token).
  int active_num_gpu_layers = config_.num_hidden_layers - config_.num_cpu_offload_layers;
  if (f32_concatenated_weights_loaded_ || active_num_gpu_layers == 0) {
    return;
  }
  Logger::info("Loading concatenated F32 weights for optimal GPU performance");

  Logger::info("Loading F32 concatenated weights on-demand for GPU inference");

  // Concatenated FP32 / Dequantized Layer Weights for GPU layers (w_..._f32_dev_ pointers)
  int hs = config_.hidden_size;
  int is = config_.intermediate_size;
  int kv_dim = (hs / config_.num_attention_heads) * config_.num_key_value_heads;
  
  size_t layer_q_size_f32 = (size_t)hs*hs, layer_k_size_f32 = (size_t)kv_dim*hs, layer_v_size_f32 = (size_t)kv_dim*hs;
  size_t layer_o_size_f32 = (size_t)hs*hs, layer_gate_size_f32 = (size_t)is*hs, layer_up_size_f32 = (size_t)is*hs;
  size_t layer_down_size_f32 = (size_t)hs*is;

  auto process_gpu_layer_weights_to_f32_concat = [&] (
      const std::function<const std::vector<float>&(const LayerWeights&)>& f32_accessor,
      const std::function<const std::vector<uint16_t>&(const LayerWeights&)>& bf16_accessor,
      const std::function<const std::vector<block_q8_0>&(const LayerWeights&)>& q8_accessor,
      const std::function<const std::vector<block_q4_K>&(const LayerWeights&)>& q4k_accessor,
      const std::function<const std::vector<block_q6_K>&(const LayerWeights&)>& q6k_accessor,
      float*& dev_ptr, size_t single_layer_elem_size, const std::string& weight_name) {

    std::vector<float> concatenated_f32;
    concatenated_f32.reserve(active_num_gpu_layers * single_layer_elem_size);

    for (int l_gpu_idx = 0; l_gpu_idx < active_num_gpu_layers; ++l_gpu_idx) {
      int l_model_idx = config_.num_cpu_offload_layers + l_gpu_idx;
      // Individual weight dequantization will be handled by the lambda accessors below
      const LayerWeights& lw = layers[l_model_idx];

      const auto& f32_vec = f32_accessor(lw);
      const auto& bf16_vec = bf16_accessor(lw);
      const auto& q8_vec = q8_accessor(lw);
      const auto& q4k_vec = q4k_accessor(lw);
      const auto& q6k_vec = q6k_accessor(lw);

      if (!f32_vec.empty()) {
        concatenated_f32.insert(concatenated_f32.end(), f32_vec.begin(), f32_vec.end());
      } else if (!bf16_vec.empty()) {
        for (uint16_t bf16_val : bf16_vec) {
          concatenated_f32.push_back(bfloat16_to_float32(bf16_val));
        }
      } else if (!q8_vec.empty()) {
        std::vector<float> temp_f32(q8_vec.size() * GGML_QK8_0);
        for (size_t i = 0; i < q8_vec.size(); ++i) {
          dequantize_q8_0_block(&q8_vec[i], &temp_f32[i * GGML_QK8_0]);
        }
        concatenated_f32.insert(concatenated_f32.end(), temp_f32.begin(), temp_f32.end());
      } else if (!q4k_vec.empty()) {
        std::vector<float> temp_f32(q4k_vec.size() * GGML_QK_K);
        for (size_t i = 0; i < q4k_vec.size(); ++i) {
          dequantize_q4_k_m(&q4k_vec[i], &temp_f32[i * GGML_QK_K], GGML_QK_K);
        }
        concatenated_f32.insert(concatenated_f32.end(), temp_f32.begin(), temp_f32.end());
      } else if (!q6k_vec.empty()) {
        std::vector<float> temp_f32(q6k_vec.size() * GGML_QK_K);
        for (size_t i = 0; i < q6k_vec.size(); ++i) {
          dequantize_q6_k(&q6k_vec[i], &temp_f32[i * GGML_QK_K], GGML_QK_K);
        }
        concatenated_f32.insert(concatenated_f32.end(), temp_f32.begin(), temp_f32.end());
      } else {
        throw std::runtime_error("Layer " + std::to_string(l_model_idx) + ": No " + weight_name + " weights found for GPU processing");
      }
    }

    if (!concatenated_f32.empty()) {
      if (dev_ptr) { cudaFree(dev_ptr); dev_ptr = nullptr; }
      gpuErrchk(cudaMalloc(&dev_ptr, concatenated_f32.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(dev_ptr, concatenated_f32.data(), concatenated_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
      Logger::info("Loaded concatenated " + weight_name + " (F32) to GPU for " + std::to_string(active_num_gpu_layers) + " layers");
    }
  };

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_q_proj_dequantized(layer_idx); 
      return lw.q_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.q_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.q_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.q_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.q_proj_q6k; },
    w_q_f32_dev_, layer_q_size_f32, "Q Proj");
  
  // CRITICAL: Immediately clear Q-proj CPU memory after GPU upload to prevent OOM
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].q_proj_f32.clear();
      layers[layer_idx].q_proj_f32.shrink_to_fit();
    }
    Logger::info("Cleared Q-proj CPU memory immediately after GPU upload");
  }

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_k_proj_dequantized(layer_idx); 
      return lw.k_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.k_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.k_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.k_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.k_proj_q6k; },
    w_k_f32_dev_, layer_k_size_f32, "K Proj");
  
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].k_proj_f32.clear();
      layers[layer_idx].k_proj_f32.shrink_to_fit();
    }
    Logger::info("Cleared K-proj CPU memory immediately after GPU upload");
  }

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_v_proj_dequantized(layer_idx); 
      return lw.v_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.v_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.v_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.v_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.v_proj_q6k; },
    w_v_f32_dev_, layer_v_size_f32, "V Proj");
  
  // Clear V-proj CPU memory immediately
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].v_proj_f32.clear();
      layers[layer_idx].v_proj_f32.shrink_to_fit();
    }
  }

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_o_proj_dequantized(layer_idx); 
      return lw.o_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.o_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.o_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.o_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.o_proj_q6k; },
    w_o_f32_dev_, layer_o_size_f32, "O Proj");
  
  // Clear O-proj CPU memory immediately  
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].o_proj_f32.clear();
      layers[layer_idx].o_proj_f32.shrink_to_fit();
    }
  }

  if (config_.num_experts == 0) {
  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_gate_proj_dequantized(layer_idx); 
      return lw.gate_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.gate_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.gate_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.gate_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.gate_proj_q6k; },
    w_gate_f32_dev_, layer_gate_size_f32, "Gate Proj");
  
  // Clear Gate-proj CPU memory immediately
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].gate_proj_f32.clear();
      layers[layer_idx].gate_proj_f32.shrink_to_fit();
    }
  }

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_up_proj_dequantized(layer_idx); 
      return lw.up_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.up_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.up_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.up_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.up_proj_q6k; },
    w_up_f32_dev_, layer_up_size_f32, "Up Proj");
  
  // Clear Up-proj CPU memory immediately
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].up_proj_f32.clear();
      layers[layer_idx].up_proj_f32.shrink_to_fit();
    }
  }

  process_gpu_layer_weights_to_f32_concat(
    [this](const LayerWeights& lw) -> const std::vector<float>& { 
      int layer_idx = &lw - &layers[0]; 
      const_cast<TinyLlamaModel*>(this)->ensure_down_proj_dequantized(layer_idx); 
      return lw.down_proj_f32; 
    },
    [](const LayerWeights& lw) -> const std::vector<uint16_t>& { return lw.down_proj; },
    [](const LayerWeights& lw) -> const std::vector<block_q8_0>& { return lw.down_proj_q8_0; },
    [](const LayerWeights& lw) -> const std::vector<block_q4_K>& { return lw.down_proj_q4k; },
    [](const LayerWeights& lw) -> const std::vector<block_q6_K>& { return lw.down_proj_q6k; },
    w_down_f32_dev_, layer_down_size_f32, "Down Proj");
  
  // Clear Down-proj CPU memory immediately
  if (config_.enable_memory_efficient_layers) {
    int first_gpu_layer = config_.num_cpu_offload_layers;
    int last_gpu_layer = config_.num_hidden_layers - 1;
    for (int layer_idx = first_gpu_layer; layer_idx <= last_gpu_layer; ++layer_idx) {
      layers[layer_idx].down_proj_f32.clear();
      layers[layer_idx].down_proj_f32.shrink_to_fit();
    }
  }

  } else {
    // Mixture of experts: router and expert stacks per GPU layer, f32.
    auto upload = [&](const std::function<const std::vector<float>&(LayerWeights&)>& get, float*& dev,
                      const char* what) {
      std::vector<float> cat;
      for (int g = 0; g < active_num_gpu_layers; ++g) {
        LayerWeights& lw = layers[config_.num_cpu_offload_layers + g];
        if (lw.moe_router_f32.empty() && !lw.moe_router_q8_0.empty())
          dequantize_vector_q8_0_to_f32(lw.moe_router_q8_0, lw.moe_router_f32, lw.moe_router_q8_0.size() * GGML_QK8_0, 0);
        const std::vector<float>& v = get(lw);
        if (v.empty())
          throw std::runtime_error("MoE layer " + std::to_string(config_.num_cpu_offload_layers + g) + ": no " + what +
                                   " weights for the GPU");
        cat.insert(cat.end(), v.begin(), v.end());
      }
      if (dev) { cudaFree(dev); dev = nullptr; }
      gpuErrchk(cudaMalloc(&dev, cat.size() * sizeof(float)));
      gpuErrchk(cudaMemcpy(dev, cat.data(), cat.size() * sizeof(float), cudaMemcpyHostToDevice));
    };
    upload([](LayerWeights& lw) -> const std::vector<float>& { return lw.moe_router_f32; }, w_moe_router_dev_, "router");
    upload([](LayerWeights& lw) -> const std::vector<float>& { return lw.moe_gate_exps_f32; }, w_moe_gate_dev_, "expert gate");
    upload([](LayerWeights& lw) -> const std::vector<float>& { return lw.moe_up_exps_f32; }, w_moe_up_dev_, "expert up");
    upload([](LayerWeights& lw) -> const std::vector<float>& { return lw.moe_down_exps_f32; }, w_moe_down_dev_, "expert down");
    Logger::info("Loaded MoE router and expert stacks (F32) to GPU for " + std::to_string(active_num_gpu_layers) +
                 " layers (" + std::to_string(config_.num_experts) + " experts)");
  }

  f32_concatenated_weights_loaded_ = true;
  Logger::info("Successfully loaded all concatenated F32 weights for GPU layers");
}

// BF16 Tensor Core weight management functions
void TinyLlamaModel::ensure_bf16_concatenated_weights_loaded() {
  Logger::info("Loading concatenated BF16 weights for Tensor Core acceleration");
  
  int active_num_gpu_layers = config_.num_hidden_layers - config_.num_cpu_offload_layers;
  if (bf16_concatenated_weights_loaded_ || active_num_gpu_layers == 0) {
    return;
  }

  Logger::info("Converting F32 weights to BF16 for Tensor Core acceleration");

  // Ensure F32 weights are loaded first
  ensure_f32_concatenated_weights_loaded();

  int hs = config_.hidden_size;
  int is = config_.intermediate_size;
  int kv_dim = (hs / config_.num_attention_heads) * config_.num_key_value_heads;
  
  size_t layer_q_size = (size_t)hs*hs;
  size_t layer_k_size = (size_t)kv_dim*hs; 
  size_t layer_v_size = (size_t)kv_dim*hs;
  size_t layer_o_size = (size_t)hs*hs;
  size_t layer_gate_size = (size_t)is*hs;
  size_t layer_up_size = (size_t)is*hs;
  size_t layer_down_size = (size_t)hs*is;

  // Convert F32 weights to BF16 on GPU
  auto convert_f32_to_bf16 = [&](float* f32_dev_ptr, uint16_t*& bf16_dev_ptr, size_t total_elements, const std::string& weight_name) {
    if (f32_dev_ptr && total_elements > 0) {
      if (bf16_dev_ptr) { cudaFree(bf16_dev_ptr); bf16_dev_ptr = nullptr; }
      gpuErrchk(cudaMalloc(&bf16_dev_ptr, total_elements * sizeof(uint16_t)));
      convert_fp32_to_bf16_cuda(f32_dev_ptr, bf16_dev_ptr, total_elements, 0);
      gpuErrchk(cudaDeviceSynchronize());
      Logger::info("Converted " + weight_name + " from F32 to BF16 on GPU for Tensor Cores");
    }
  };

  size_t total_q_elements = active_num_gpu_layers * layer_q_size;
  size_t total_k_elements = active_num_gpu_layers * layer_k_size;
  size_t total_v_elements = active_num_gpu_layers * layer_v_size;
  size_t total_o_elements = active_num_gpu_layers * layer_o_size;
  size_t total_gate_elements = active_num_gpu_layers * layer_gate_size;
  size_t total_up_elements = active_num_gpu_layers * layer_up_size;
  size_t total_down_elements = active_num_gpu_layers * layer_down_size;

  convert_f32_to_bf16(w_q_f32_dev_, w_q_bf16_dev_, total_q_elements, "Q Proj");
  convert_f32_to_bf16(w_k_f32_dev_, w_k_bf16_dev_, total_k_elements, "K Proj");
  convert_f32_to_bf16(w_v_f32_dev_, w_v_bf16_dev_, total_v_elements, "V Proj");
  convert_f32_to_bf16(w_o_f32_dev_, w_o_bf16_dev_, total_o_elements, "O Proj");
  convert_f32_to_bf16(w_gate_f32_dev_, w_gate_bf16_dev_, total_gate_elements, "Gate Proj");
  convert_f32_to_bf16(w_up_f32_dev_, w_up_bf16_dev_, total_up_elements, "Up Proj");
  convert_f32_to_bf16(w_down_f32_dev_, w_down_bf16_dev_, total_down_elements, "Down Proj");

  bf16_concatenated_weights_loaded_ = true;
  Logger::info("Successfully loaded all concatenated BF16 weights for Tensor Core acceleration");
}

void TinyLlamaModel::free_bf16_concatenated_weights() {
  if (!bf16_concatenated_weights_loaded_) return;
  
  Logger::info("Freeing BF16 concatenated weights");
  if (w_q_bf16_dev_) { cudaFree(w_q_bf16_dev_); w_q_bf16_dev_ = nullptr; }
  if (w_k_bf16_dev_) { cudaFree(w_k_bf16_dev_); w_k_bf16_dev_ = nullptr; }
  if (w_v_bf16_dev_) { cudaFree(w_v_bf16_dev_); w_v_bf16_dev_ = nullptr; }
  if (w_o_bf16_dev_) { cudaFree(w_o_bf16_dev_); w_o_bf16_dev_ = nullptr; }
  if (w_gate_bf16_dev_) { cudaFree(w_gate_bf16_dev_); w_gate_bf16_dev_ = nullptr; }
  if (w_up_bf16_dev_) { cudaFree(w_up_bf16_dev_); w_up_bf16_dev_ = nullptr; }
  if (w_down_bf16_dev_) { cudaFree(w_down_bf16_dev_); w_down_bf16_dev_ = nullptr; }
  
  bf16_concatenated_weights_loaded_ = false;
  Logger::info("Successfully freed all BF16 concatenated weights");
}

void TinyLlamaModel::ensure_layer_weights_on_gpu(int layer_idx) {
#ifdef HAS_CUDA
  if (layer_idx < 0 || layer_idx >= layers.size()) return;
  
  // Check if this layer is supposed to be on GPU
  int first_gpu_layer = config_.num_cpu_offload_layers;
  int last_gpu_layer = config_.num_hidden_layers - 1;
  if (layer_idx < first_gpu_layer || layer_idx > last_gpu_layer) return;
  
  LayerWeights& lw = layers[layer_idx];
  
  // Check if weights are already loaded on GPU for this layer
  if (lw.q_proj_f32_dev && lw.k_proj_f32_dev && lw.v_proj_f32_dev && 
      lw.o_proj_f32_dev && lw.gate_proj_f32_dev && lw.up_proj_f32_dev && lw.down_proj_f32_dev) {
    return; // Already loaded
  }
  
  // AGGRESSIVE MEMORY MANAGEMENT: Free previous layer weights to make room
  // Keep only the current layer and maybe the next one
  if (layer_idx > first_gpu_layer) {
    int prev_layer = layer_idx - 1;
    if (prev_layer >= first_gpu_layer && prev_layer < layers.size()) {
      free_layer_gpu_weights(prev_layer);
    }
  }
  
  // If still hitting memory limits, free ALL other GPU layers except current
  if (layer_idx > first_gpu_layer + 1) {
    for (int i = first_gpu_layer; i < layer_idx - 1; ++i) {
      free_layer_gpu_weights(i);
    }
  }
  
  Logger::info("JIT loading layer " + std::to_string(layer_idx) + " weights to GPU (with aggressive eviction)");
  
  // Dequantize and load each weight matrix individually
  auto load_single_weight = [&](
    const std::function<void()>& ensure_dequantized,
    const std::function<const std::vector<float>&()>& get_f32_weights,
    float*& dev_ptr,
    const std::string& weight_name
  ) {
    ensure_dequantized();
    const auto& f32_weights = get_f32_weights();
    if (!f32_weights.empty()) {
      if (dev_ptr) { cudaFree(dev_ptr); dev_ptr = nullptr; }
      
      // Try allocation with error handling
      cudaError_t malloc_result = cudaMalloc(&dev_ptr, f32_weights.size() * sizeof(float));
      if (malloc_result != cudaSuccess) {
        Logger::warning("GPU memory allocation failed for " + weight_name + " in layer " + std::to_string(layer_idx) + 
                       ". Attempting emergency cleanup...");
        
        // Emergency cleanup: free ALL other layer weights
        for (int emergency_idx = first_gpu_layer; emergency_idx <= last_gpu_layer; ++emergency_idx) {
          if (emergency_idx != layer_idx) {
            free_layer_gpu_weights(emergency_idx);
          }
        }
        
        // Try allocation again after cleanup
        malloc_result = cudaMalloc(&dev_ptr, f32_weights.size() * sizeof(float));
        if (malloc_result != cudaSuccess) {
          throw std::runtime_error("GPU OOM: Cannot allocate " + std::to_string(f32_weights.size() * sizeof(float)) + 
                                   " bytes for " + weight_name + " in layer " + std::to_string(layer_idx) + 
                                   " even after emergency cleanup. Try reducing --n-gpu-layers.");
        }
        Logger::info("Emergency cleanup successful, allocated " + weight_name);
      }
      
      gpuErrchk(cudaMemcpy(dev_ptr, f32_weights.data(), f32_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
      
      // Immediately clear CPU memory to save RAM
      if (config_.enable_memory_efficient_layers) {
        const_cast<std::vector<float>&>(f32_weights).clear();
        const_cast<std::vector<float>&>(f32_weights).shrink_to_fit();
      }
    }
  };
  
  // Load Q projection
  load_single_weight(
    [this, layer_idx]() { ensure_q_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.q_proj_f32; },
    lw.q_proj_f32_dev,
    "Q Proj"
  );
  
  // Load K projection  
  load_single_weight(
    [this, layer_idx]() { ensure_k_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.k_proj_f32; },
    lw.k_proj_f32_dev,
    "K Proj"
  );
  
  // Load V projection
  load_single_weight(
    [this, layer_idx]() { ensure_v_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.v_proj_f32; },
    lw.v_proj_f32_dev, 
    "V Proj"
  );
  
  // Load O projection
  load_single_weight(
    [this, layer_idx]() { ensure_o_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.o_proj_f32; },
    lw.o_proj_f32_dev,
    "O Proj"
  );
  
  // Load Gate projection
  load_single_weight(
    [this, layer_idx]() { ensure_gate_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.gate_proj_f32; },
    lw.gate_proj_f32_dev,
    "Gate Proj"
  );
  
  // Load Up projection
  load_single_weight(
    [this, layer_idx]() { ensure_up_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.up_proj_f32; },
    lw.up_proj_f32_dev,
    "Up Proj"
  );
  
  // Load Down projection
  load_single_weight(
    [this, layer_idx]() { ensure_down_proj_dequantized(layer_idx); },
    [&lw]() -> const std::vector<float>& { return lw.down_proj_f32; },
    lw.down_proj_f32_dev,
    "Down Proj"
  );
#else
  // CPU-only build - this function is a no-op
  Logger::info("CPU-only build: ensure_layer_weights_on_gpu is a no-op for layer " + std::to_string(layer_idx));
#endif
}

void TinyLlamaModel::free_layer_gpu_weights(int layer_idx) {
#ifdef HAS_CUDA
  if (layer_idx < 0 || layer_idx >= layers.size()) return;
  
  LayerWeights& lw = layers[layer_idx];
  
  if (lw.q_proj_f32_dev) { cudaFree(lw.q_proj_f32_dev); lw.q_proj_f32_dev = nullptr; }
  if (lw.k_proj_f32_dev) { cudaFree(lw.k_proj_f32_dev); lw.k_proj_f32_dev = nullptr; }
  if (lw.v_proj_f32_dev) { cudaFree(lw.v_proj_f32_dev); lw.v_proj_f32_dev = nullptr; }
  if (lw.o_proj_f32_dev) { cudaFree(lw.o_proj_f32_dev); lw.o_proj_f32_dev = nullptr; }
  if (lw.gate_proj_f32_dev) { cudaFree(lw.gate_proj_f32_dev); lw.gate_proj_f32_dev = nullptr; }
  if (lw.up_proj_f32_dev) { cudaFree(lw.up_proj_f32_dev); lw.up_proj_f32_dev = nullptr; }
  if (lw.down_proj_f32_dev) { cudaFree(lw.down_proj_f32_dev); lw.down_proj_f32_dev = nullptr; }
  
  Logger::info("Freed GPU weights for layer " + std::to_string(layer_idx) + " (~200MB freed)");
#else  
  // CPU-only build - this function is a no-op
  Logger::info("CPU-only build: free_layer_gpu_weights is a no-op for layer " + std::to_string(layer_idx));
#endif
}

#endif // HAS_CUDA (GPU weight-management implementations)

// NOTE: map_gguf_weights is required by BOTH CUDA and CPU builds (it maps
// GGUF tensors into host-side model fields). It was previously trapped inside
// the HAS_CUDA block opened far above, which broke CPU-only linking.
// Load-time dequantisation of the formats ember has no kernels for: ggml's
// legacy 32-wide blocks (Q4_0, Q4_1, Q5_0, Q5_1), F16 and BF16, written straight
// into the f32 vector in the tensor's own element order. Block layouts as in
// ggml-common.h: Q4_0 {d, qs[16]}, Q4_1 {d, m, qs[16]}, Q5_0 {d, qh[4], qs[16]},
// Q5_1 {d, m, qh[4], qs[16]}; element j of a block uses the low nibble of qs[j],
// element j+16 the high nibble, and the Q5 formats take bit j (resp. j+16) of qh
// as the fifth bit. x = (q - 8) d, q d + m, (q - 16) d, q d + m respectively.
// Also the K-quants Q4_K, Q5_K and Q6_K (256-wide super-blocks), for the
// interpreter and any tensor the llama path keeps no blocks for.
// Returns false for any other type.
bool dequant_legacy_to_f32(const GGUFTensorInfo& info, const uint8_t* src, std::vector<float>& out) {
  const GGMLType t = info.type;
  const size_t n = info.num_elements;
  if (t == GGMLType::GGML_TYPE_F16 || t == GGMLType::GGML_TYPE_BF16) {
    out.resize(n);
    dequant_span_to_f32(t, src, 0, n, out.data());
    return true;
  }
  const bool q4_0 = t == GGMLType::GGML_TYPE_Q4_0, q4_1 = t == GGMLType::GGML_TYPE_Q4_1;
  const bool q5_0 = t == GGMLType::GGML_TYPE_Q5_0, q5_1 = t == GGMLType::GGML_TYPE_Q5_1;
  const bool k_quant = t == GGMLType::GGML_TYPE_Q4_K || t == GGMLType::GGML_TYPE_Q5_K || t == GGMLType::GGML_TYPE_Q6_K;
  if (!(q4_0 || q4_1 || q5_0 || q5_1 || k_quant)) return false;
  const size_t width = k_quant ? GGML_QK_K : 32;
  if (n % width != 0)
    throw std::runtime_error(std::string(ggml_type_name(t)) + " tensor with " + std::to_string(n) +
                             " elements (not a multiple of " + std::to_string(width) + ")");
  const size_t bsz = ggml_type_size(t), nb = n / width;
  if (info.size_in_bytes < nb * bsz) throw std::runtime_error("legacy-quantised tensor is shorter than its element count implies");
  out.resize(n);
  dequant_span_to_f32(t, src, 0, n, out.data());
  return true;
}

// Every f16 value as f32, built once from fp16_to_fp32 itself (so a lookup equals the call bit for
// bit): the interpreter dequantises an F16 weight row by row on every call, so a large F16 model
// converts every weight once per decode step.
static const float* fp16_table() {
  static const std::vector<float> table = [] {
    std::vector<float> t(65536);
    for (uint32_t h = 0; h < 65536; ++h) t[h] = fp16_to_fp32(static_cast<uint16_t>(h), false);
    return t;
  }();
  return table.data();
}

bool dequant_span_to_f32(GGMLType t, const uint8_t* src, size_t first, size_t n, float* out) {
  if (t == GGMLType::GGML_TYPE_F16) {
    const float* table = fp16_table();
    const uint8_t* p = src + 2 * first;
    for (size_t i = 0; i < n; ++i) {
      uint16_t h;
      std::memcpy(&h, p + 2 * i, 2);
      out[i] = table[h];
    }
    return true;
  }
  if (t == GGMLType::GGML_TYPE_BF16) {  // the top 16 bits of an f32: exact
    for (size_t i = 0; i < n; ++i) {
      uint16_t h;
      std::memcpy(&h, src + 2 * (first + i), 2);
      const uint32_t bits = static_cast<uint32_t>(h) << 16;
      std::memcpy(out + i, &bits, 4);
    }
    return true;
  }
  if (t == GGMLType::GGML_TYPE_Q8_0) {
    const block_q8_0* blocks = reinterpret_cast<const block_q8_0*>(src) + first / GGML_QK8_0;
    for (size_t b = 0; b < n / GGML_QK8_0; ++b) dequantize_q8_0_block(blocks + b, out + b * GGML_QK8_0);
    return true;
  }
  // K-quants: 256-wide super-blocks (first and n multiples of GGML_QK_K), ember's block dequantisers.
  if (t == GGMLType::GGML_TYPE_Q4_K) {
    const block_q4_K* blocks = reinterpret_cast<const block_q4_K*>(src) + first / GGML_QK_K;
    for (size_t b = 0; b < n / GGML_QK_K; ++b) dequantize_q4_k_m(blocks + b, out + b * GGML_QK_K, GGML_QK_K);
    return true;
  }
  if (t == GGMLType::GGML_TYPE_Q5_K) {
    const block_q5_K* blocks = reinterpret_cast<const block_q5_K*>(src) + first / GGML_QK_K;
    for (size_t b = 0; b < n / GGML_QK_K; ++b) dequantize_q5_k(blocks + b, out + b * GGML_QK_K);
    return true;
  }
  if (t == GGMLType::GGML_TYPE_Q6_K) {
    const block_q6_K* blocks = reinterpret_cast<const block_q6_K*>(src) + first / GGML_QK_K;
    for (size_t b = 0; b < n / GGML_QK_K; ++b) dequantize_q6_k(blocks + b, out + b * GGML_QK_K, GGML_QK_K);
    return true;
  }
  const bool q4_0 = t == GGMLType::GGML_TYPE_Q4_0, q4_1 = t == GGMLType::GGML_TYPE_Q4_1;
  const bool q5_0 = t == GGMLType::GGML_TYPE_Q5_0, q5_1 = t == GGMLType::GGML_TYPE_Q5_1;
  if (!(q4_0 || q4_1 || q5_0 || q5_1)) return false;
  const bool has_min = q4_1 || q5_1, five_bit = q5_0 || q5_1;
  const size_t bsz = ggml_type_size(t), b0 = first / 32, nb = n / 32;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* p = src + (b0 + b) * bsz;
    uint16_t dh, mh = 0;
    std::memcpy(&dh, p, 2);
    p += 2;
    if (has_min) { std::memcpy(&mh, p, 2); p += 2; }
    uint32_t qh = 0;
    if (five_bit) { std::memcpy(&qh, p, 4); p += 4; }
    const float d = fp16_to_fp32(dh, false), m = has_min ? fp16_to_fp32(mh, false) : 0.0f;
    const int offset = has_min ? 0 : (five_bit ? 16 : 8);
    float* y = out + b * 32;
    for (int j = 0; j < 16; ++j) {
      int lo = p[j] & 0x0F, hi = p[j] >> 4;
      if (five_bit) {
        lo |= ((qh >> j) & 1) << 4;
        hi |= ((qh >> (j + 16)) & 1) << 4;
      }
      y[j] = static_cast<float>(lo - offset) * d + m;
      y[j + 16] = static_cast<float>(hi - offset) * d + m;
    }
  }
  return true;
}

void map_gguf_weights(const GGUFData& gguf, TinyLlamaModel& model) {
  Logger::info("Mapping GGUF weights to model fields (ULTRA-OPTIMIZED VERSION)...");

  const uint8_t* actual_data_block_start = nullptr;
  
  // Determine which data source to use
  if (gguf.mapped_tensor_data != nullptr && gguf.mapped_tensor_data_size > 0) {
    const uint8_t* mmap_buffer_start = static_cast<const uint8_t*>(gguf.mapped_tensor_data);
    actual_data_block_start = mmap_buffer_start + gguf.offset_diff_for_mmap;
    Logger::info("map_gguf_weights: Using mmap mode (ZERO-COPY). Size: " +
               std::to_string(gguf.mapped_tensor_data_size) + " bytes.");
  } else if (!gguf.tensor_data.empty()) {
    actual_data_block_start = gguf.tensor_data.data();
    Logger::info("map_gguf_weights: Using non-mmap mode. Size: " +
                 std::to_string(gguf.tensor_data.size()) + " bytes.");
  } else {
    Logger::error("GGUF tensor data is not available. Cannot map weights.");
    return;
  }

  const size_t num_tensors = gguf.tensor_infos_map.size();
  Logger::info("Processing " + std::to_string(num_tensors) + " tensors with ultra-optimized parallel mapping...");

  // Pre-allocate containers to avoid reallocations during parallel processing
  std::vector<std::pair<std::string, GGUFTensorInfo>> tensor_pairs;
  tensor_pairs.reserve(num_tensors);
  for (const auto& pair : gguf.tensor_infos_map) {
    tensor_pairs.emplace_back(pair.first, pair.second);
  }

  // Reserve capacity for major model containers to reduce allocations
  const size_t typical_blocks = 4096;
  if (model.lm_head_q8_0.capacity() == 0) model.lm_head_q8_0.reserve(32768);
  if (model.embed_tokens_q8_0.capacity() == 0) model.embed_tokens_q8_0.reserve(32768);
  
  for (auto& layer : model.layers) {
    if (layer.q_proj_q8_0.capacity() == 0) layer.q_proj_q8_0.reserve(typical_blocks);
    if (layer.k_proj_q8_0.capacity() == 0) layer.k_proj_q8_0.reserve(typical_blocks);
    if (layer.v_proj_q8_0.capacity() == 0) layer.v_proj_q8_0.reserve(typical_blocks);
    if (layer.o_proj_q8_0.capacity() == 0) layer.o_proj_q8_0.reserve(typical_blocks);
    if (layer.gate_proj_q8_0.capacity() == 0) layer.gate_proj_q8_0.reserve(typical_blocks);
    if (layer.up_proj_q8_0.capacity() == 0) layer.up_proj_q8_0.reserve(typical_blocks);
    if (layer.down_proj_q8_0.capacity() == 0) layer.down_proj_q8_0.reserve(typical_blocks);
  }

  // BLAZING FAST: Sort tensors by type and process in bulk
  std::vector<size_t> global_tensor_indices;
  std::vector<std::vector<size_t>> layer_tensor_indices(model.layers.size());
  
  global_tensor_indices.reserve(10); // output.weight, token_embd.weight, output_norm.weight, etc.
  for (auto& layer_indices : layer_tensor_indices) {
    layer_indices.reserve(9); // 7 weights + 2 norms per layer
  }
  
  // ULTRA-FAST categorization without string operations
  for (size_t i = 0; i < tensor_pairs.size(); ++i) {
    const std::string& name = tensor_pairs[i].first;
    if (name[0] == 'o' || name[0] == 't') { // output.weight, token_embd.weight, output_norm.weight
      global_tensor_indices.push_back(i);
    } else if (name.size() > 4 && name[0] == 'b' && name[1] == 'l' && name[2] == 'k' && name[3] == '.') {
      // Extract layer index without substr - MUCH faster
      size_t layer_start = 4;
      size_t layer_end = name.find('.', layer_start);
      if (layer_end != std::string::npos) {
        int layer_idx = 0;
        for (size_t pos = layer_start; pos < layer_end; ++pos) {
          layer_idx = layer_idx * 10 + (name[pos] - '0');
        }
        if (layer_idx >= 0 && static_cast<size_t>(layer_idx) < model.layers.size()) {
          layer_tensor_indices[layer_idx].push_back(i);
        }
      }
    }
  }
  
  std::atomic<int> processed_count{0};
  std::atomic<int> error_count{0};
  std::mutex first_error_mu;
  std::string first_error;  // reported if any tensor fails to map

  // Process global tensors sequentially (small count, avoid overhead)
  for (size_t idx : global_tensor_indices) {
    try {
    const std::string& target_field_key = tensor_pairs[idx].first;
    const GGUFTensorInfo& info = tensor_pairs[idx].second;
    const uint8_t* tensor_data_ptr = actual_data_block_start + info.offset;

      // Global tensors with optimized type dispatch
    if (target_field_key == "output.weight") {
        switch (info.type) {
          case GGMLType::GGML_TYPE_Q6_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q6_K);
        model.lm_head_q6k.resize(num_blocks);
        std::memcpy(model.lm_head_q6k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q4_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q4_K);
        model.lm_head_q4k.resize(num_blocks);
        std::memcpy(model.lm_head_q4k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q5_K: {
        model.lm_head_q5k.resize(info.size_in_bytes / sizeof(block_q5_K));
        std::memcpy(model.lm_head_q5k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q8_0: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q8_0);
        model.lm_head_q8_0.resize(num_blocks);
        std::memcpy(model.lm_head_q8_0.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q8_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q8_K);
        model.lm_head_q8k.resize(num_blocks);
        std::memcpy(model.lm_head_q8k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_F32: {
        size_t num_elements = info.size_in_bytes / sizeof(float);
        model.lm_head_f32.resize(num_elements);
        std::memcpy(model.lm_head_f32.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          default:
            if (!dequant_legacy_to_f32(info, tensor_data_ptr, model.lm_head_f32)) throw std::runtime_error(target_field_key + ": unsupported tensor type " + ggml_type_name(info.type));
            break;
        }
        processed_count++;
      continue;
      }
      
      if (target_field_key == "token_embd.weight") {
        switch (info.type) {
          case GGMLType::GGML_TYPE_Q4_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q4_K);
        model.embed_tokens_q4k.resize(num_blocks);
        std::memcpy(model.embed_tokens_q4k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q5_K: {
        model.embed_tokens_q5k.resize(info.size_in_bytes / sizeof(block_q5_K));
        std::memcpy(model.embed_tokens_q5k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q8_0: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q8_0);
        model.embed_tokens_q8_0.resize(num_blocks);
        std::memcpy(model.embed_tokens_q8_0.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q8_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q8_K);
        model.embed_tokens_q8k.resize(num_blocks);
        std::memcpy(model.embed_tokens_q8k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_Q6_K: {
        size_t num_blocks = info.size_in_bytes / sizeof(block_q6_K);
        model.embed_tokens_q6k.resize(num_blocks);
        std::memcpy(model.embed_tokens_q6k.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          case GGMLType::GGML_TYPE_F32: {
        size_t num_elements = info.size_in_bytes / sizeof(float);
        model.embed_tokens_f32.resize(num_elements);
        std::memcpy(model.embed_tokens_f32.data(), tensor_data_ptr, info.size_in_bytes);
            break;
          }
          default:
            if (!dequant_legacy_to_f32(info, tensor_data_ptr, model.embed_tokens_f32)) throw std::runtime_error(target_field_key + ": unsupported tensor type " + ggml_type_name(info.type));
            break;
        }
        processed_count++;
      continue;
      }
      
      if (target_field_key == "output_norm.weight") {
      if (info.type == GGMLType::GGML_TYPE_F32) {
        size_t num_elements = info.size_in_bytes / sizeof(float);
        model.final_norm_f32.resize(num_elements);
        std::memcpy(model.final_norm_f32.data(), tensor_data_ptr, info.size_in_bytes);
      } else if (!dequant_legacy_to_f32(info, tensor_data_ptr, model.final_norm_f32)) throw std::runtime_error(target_field_key + ": unsupported tensor type " + ggml_type_name(info.type));
        processed_count++;
      continue;
    }

    } catch (const std::exception& e) {
      error_count++;
      std::lock_guard<std::mutex> lk(first_error_mu);
      if (first_error.empty()) first_error = e.what();
    }
  }
  
  // ULTRA-FAST: Process layers in parallel with pre-sorted tensors
  #pragma omp parallel for schedule(static) if(model.layers.size() > 4)
  for (int layer_idx = 0; layer_idx < (int)layer_tensor_indices.size(); ++layer_idx) {
    const auto& layer_indices = layer_tensor_indices[layer_idx];
    if (layer_indices.empty()) continue;
    
    LayerWeights& layer = model.layers[layer_idx];
    
    try {
      for (size_t idx : layer_indices) {
        const std::string& name = tensor_pairs[idx].first;
        const GGUFTensorInfo& info = tensor_pairs[idx].second;
        const uint8_t* tensor_data_ptr = actual_data_block_start + info.offset;
        
        // BLAZING FAST: Direct character matching (faster than hashing)
        const size_t last_dot = name.find_last_of('.');
        if (last_dot == std::string::npos) continue;
        
        const char* field = name.c_str() + name.find('.', 4) + 1;
        
                // BLAZING FAST: Direct character-based dispatch without string operations
        #define LEGACY_OR_THROW(target_f32) \
          default: \
            if (!dequant_legacy_to_f32(info, tensor_data_ptr, target_f32)) \
              throw std::runtime_error(name + ": unsupported tensor type " + ggml_type_name(info.type)); \
            break;
        #define FAST_COPY_WEIGHT(target_vec, block_type) \
          target_vec.resize(info.size_in_bytes / sizeof(block_type)); \
          std::memcpy(target_vec.data(), tensor_data_ptr, info.size_in_bytes);
        
        // IMPROVED: Pattern matching based on tensor name structure
        const char* name_cstr = name.c_str();
        const size_t name_len = name.length();
        
        if (name_len > 10 && name.find("attn_") != std::string::npos) {
          if (name.find("attn_q.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.q_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.q_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q5_K: FAST_COPY_WEIGHT(layer.q_proj_q5k, block_q5_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.q_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.q_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.q_proj, uint16_t); break;
              case GGMLType::GGML_TYPE_F32: FAST_COPY_WEIGHT(layer.q_proj_f32, float);
                Logger::info("[MAP_DEBUG] " + name + " F32 copied, elems=" +
                             std::to_string(layer.q_proj_f32.size()) +
                             " into layer_idx=" + std::to_string(layer_idx));
                break; // transformer_cpp fp32 exports
              LEGACY_OR_THROW(layer.q_proj_f32)
            }
          } else if (name.find("attn_k.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.k_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.k_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q5_K: FAST_COPY_WEIGHT(layer.k_proj_q5k, block_q5_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.k_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.k_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.k_proj, uint16_t); break;
              case GGMLType::GGML_TYPE_F32: FAST_COPY_WEIGHT(layer.k_proj_f32, float); break; // transformer_cpp fp32 exports
              LEGACY_OR_THROW(layer.k_proj_f32)
            }
          } else if (name.find("attn_v.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.v_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.v_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q5_K: FAST_COPY_WEIGHT(layer.v_proj_q5k, block_q5_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.v_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.v_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.v_proj, uint16_t); break;
              case GGMLType::GGML_TYPE_F32: FAST_COPY_WEIGHT(layer.v_proj_f32, float); break; // transformer_cpp fp32 exports
              LEGACY_OR_THROW(layer.v_proj_f32)
            }
          } else if (name.find("attn_output.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.o_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.o_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q5_K: FAST_COPY_WEIGHT(layer.o_proj_q5k, block_q5_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.o_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.o_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.o_proj, uint16_t); break;
              case GGMLType::GGML_TYPE_F32: FAST_COPY_WEIGHT(layer.o_proj_f32, float); break; // transformer_cpp fp32 exports
              LEGACY_OR_THROW(layer.o_proj_f32)
            }
          } else if (name.find("attn_norm.weight") != std::string::npos) {
            if (info.type == GGMLType::GGML_TYPE_F32) { FAST_COPY_WEIGHT(layer.input_layernorm_f32, float); }
            else if (!dequant_legacy_to_f32(info, tensor_data_ptr, layer.input_layernorm_f32))
              throw std::runtime_error(name + ": unsupported tensor type " + ggml_type_name(info.type));
          }
        } else if (name_len > 10 && name.find("ffn_") != std::string::npos) {
          if (name.find("ffn_gate_inp.weight") != std::string::npos) {  // MoE router
            if (info.type == GGMLType::GGML_TYPE_F32) { FAST_COPY_WEIGHT(layer.moe_router_f32, float); }
            else if (info.type == GGMLType::GGML_TYPE_Q8_0) { FAST_COPY_WEIGHT(layer.moe_router_q8_0, block_q8_0); }
            else if (!dequant_legacy_to_f32(info, tensor_data_ptr, layer.moe_router_f32))
              throw std::runtime_error(name + ": unsupported tensor type " + ggml_type_name(info.type));
          } else if (name.find("_exps.weight") != std::string::npos) {  // MoE expert stacks
            std::vector<float>& dst = name.find("ffn_gate_exps") != std::string::npos ? layer.moe_gate_exps_f32
                                    : name.find("ffn_up_exps") != std::string::npos   ? layer.moe_up_exps_f32
                                                                                     : layer.moe_down_exps_f32;
            if (info.type == GGMLType::GGML_TYPE_F32) { FAST_COPY_WEIGHT(dst, float); }
            else if (!dequant_legacy_to_f32(info, tensor_data_ptr, dst))
              throw std::runtime_error(name + ": MoE expert stacks must be F32, F16 or Q4_0/Q4_1/Q5_0/Q5_1 (got " +
                                       std::string(ggml_type_name(info.type)) + ")");
          } else if (name.find("ffn_gate.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.gate_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.gate_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q5_K: FAST_COPY_WEIGHT(layer.gate_proj_q5k, block_q5_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.gate_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.gate_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.gate_proj, uint16_t); break;
              case GGMLType::GGML_TYPE_F32: FAST_COPY_WEIGHT(layer.gate_proj_f32, float); break; // transformer_cpp fp32 exports
              LEGACY_OR_THROW(layer.gate_proj_f32)
            }
          } else if (name.find("ffn_up.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.up_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.up_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q5_K: FAST_COPY_WEIGHT(layer.up_proj_q5k, block_q5_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.up_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.up_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.up_proj, uint16_t); break;
              case GGMLType::GGML_TYPE_F32: FAST_COPY_WEIGHT(layer.up_proj_f32, float); break; // transformer_cpp fp32 exports
              LEGACY_OR_THROW(layer.up_proj_f32)
            }
          } else if (name.find("ffn_down.weight") != std::string::npos) {
            switch (info.type) {
              case GGMLType::GGML_TYPE_Q8_0: FAST_COPY_WEIGHT(layer.down_proj_q8_0, block_q8_0); break;
              case GGMLType::GGML_TYPE_Q4_K: FAST_COPY_WEIGHT(layer.down_proj_q4k, block_q4_K); break;
              case GGMLType::GGML_TYPE_Q5_K: FAST_COPY_WEIGHT(layer.down_proj_q5k, block_q5_K); break;
              case GGMLType::GGML_TYPE_Q6_K: FAST_COPY_WEIGHT(layer.down_proj_q6k, block_q6_K); break;
              case GGMLType::GGML_TYPE_Q8_K: FAST_COPY_WEIGHT(layer.down_proj_q8k, block_q8_K); break;
              case GGMLType::GGML_TYPE_BF16: FAST_COPY_WEIGHT(layer.down_proj, uint16_t); break;
              case GGMLType::GGML_TYPE_F32: FAST_COPY_WEIGHT(layer.down_proj_f32, float); break; // transformer_cpp fp32 exports
              LEGACY_OR_THROW(layer.down_proj_f32)
            }
          } else if (name.find("ffn_norm.weight") != std::string::npos) {
            if (info.type == GGMLType::GGML_TYPE_F32) { FAST_COPY_WEIGHT(layer.post_attention_layernorm_f32, float); }
            else if (!dequant_legacy_to_f32(info, tensor_data_ptr, layer.post_attention_layernorm_f32))
              throw std::runtime_error(name + ": unsupported tensor type " + ggml_type_name(info.type));
          }
        }
         
         #undef FAST_COPY_WEIGHT
         #undef LEGACY_OR_THROW
      }
      processed_count++;
    } catch (const std::exception& e) {
      error_count++;
      std::lock_guard<std::mutex> lk(first_error_mu);
      if (first_error.empty()) first_error = e.what();
    }
  }

  Logger::info("Finished mapping GGUF weights: " + std::to_string(processed_count.load()) + "/" + 
               std::to_string(num_tensors) + " tensors processed successfully (errors: " + 
               std::to_string(error_count.load()) + ") with ultra-optimized parallel mapping");
  // A tensor that failed to map would leave the model silently incomplete.
  if (error_count.load() > 0)
    throw std::runtime_error("GGUF weight mapping failed for " + std::to_string(error_count.load()) +
                             " layer(s); first error: " + first_error);
}

#ifndef HAS_CUDA

void TinyLlamaModel::ensure_f32_concatenated_weights_loaded() {
  Logger::info("CPU-only build: ensure_f32_concatenated_weights_loaded is a no-op");
}

void TinyLlamaModel::ensure_bf16_concatenated_weights_loaded() {
  Logger::info("CPU-only build: ensure_bf16_concatenated_weights_loaded is a no-op");
}

void TinyLlamaModel::free_bf16_concatenated_weights() {
  Logger::info("CPU-only build: free_bf16_concatenated_weights is a no-op");
}

#endif // HAS_CUDA

