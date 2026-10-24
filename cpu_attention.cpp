#include "cpu_attention.h"
#include "parallel_for.h"
#include "logger.h"
#include "utils.h"
#include <algorithm>
#include <limits>

void update_kv_cache_batch_cpu(
    KVCache* kv_cache,
    int layer_idx,
    const std::vector<float>& k_batch_for_layer,
    const std::vector<float>& v_batch_for_layer,
    int num_tokens_in_batch,
    int start_pos_in_sequence,
    int num_kv_heads,
    int head_dim
) {
    if (!kv_cache) {
        Logger::error("update_kv_cache_batch_cpu: KVCache is null.");
        return;
    }
    if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= kv_cache->layers.size()) {
        Logger::error("update_kv_cache_batch_cpu: layer_idx " + std::to_string(layer_idx) + " is out of bounds for KVCache layers (size " + std::to_string(kv_cache->layers.size()) + ").");
        return;
    }
    Logger::info("[CPU_KV_UPDATE] Layer=" + std::to_string(layer_idx) + 
                ", start_pos=" + std::to_string(start_pos_in_sequence) + 
                ", num_tokens=" + std::to_string(num_tokens_in_batch) +
                ", k_batch_first_vals=[" + std::to_string(k_batch_for_layer[0]) + 
                "," + std::to_string(k_batch_for_layer[1]) + "," + std::to_string(k_batch_for_layer[2]) + "]");
    KVCacheLayer& layer_cache = kv_cache->layers[layer_idx];
    int kv_dim = num_kv_heads * head_dim;

    if (k_batch_for_layer.size() != static_cast<size_t>(num_tokens_in_batch * kv_dim)) {
        Logger::error("[KV_BATCH_UPDATE L" + std::to_string(layer_idx) + "] k_batch_for_layer size mismatch. Expected " +
                      std::to_string(num_tokens_in_batch * kv_dim) + ", got " + std::to_string(k_batch_for_layer.size()));
        return;
    }
    
    if (v_batch_for_layer.size() != static_cast<size_t>(num_tokens_in_batch * kv_dim)) {
        Logger::error("[KV_BATCH_UPDATE L" + std::to_string(layer_idx) + "] v_batch_for_layer size mismatch. Expected " +
                      std::to_string(num_tokens_in_batch * kv_dim) + ", got " + std::to_string(v_batch_for_layer.size()));
        return;
    }
    
    size_t expected_total_elements_in_layer_cache = static_cast<size_t>(kv_cache->max_seq_len_config_) * static_cast<size_t>(kv_cache->max_batch_size) * kv_dim;
    if (layer_cache.k.size() != expected_total_elements_in_layer_cache || layer_cache.v.size() != expected_total_elements_in_layer_cache) {
        Logger::error("[KV_BATCH_UPDATE L" + std::to_string(layer_idx) + 
                      "] Precondition failed: Layer cache not sized to max_seq_len_config. K size: " + std::to_string(layer_cache.k.size()) +
                      ", V size: " + std::to_string(layer_cache.v.size()) + 
                      ", Expected size: " + std::to_string(expected_total_elements_in_layer_cache) +
                      ". Check KVCache::initialize.");
        return;
    }
    for (int token_idx_in_batch = 0; token_idx_in_batch < num_tokens_in_batch; ++token_idx_in_batch) {
        size_t current_token_batch_offset = static_cast<size_t>(token_idx_in_batch) * kv_dim;

        int global_seq_pos = start_pos_in_sequence + token_idx_in_batch;

        if (global_seq_pos >= kv_cache->max_seq_len_config_ * kv_cache->max_batch_size) {
            Logger::error("[KV_BATCH_UPDATE L" + std::to_string(layer_idx) + 
                          "] Error: global_seq_pos (" + std::to_string(global_seq_pos) +
                          ") is out of bounds for total cache size. Skipping update for this token.");
            continue; 
        }

        size_t destination_offset_in_layer_cache = static_cast<size_t>(global_seq_pos) * kv_dim;
        size_t k_size_before = layer_cache.k.size();
        std::string k_vals_to_log = " vals to copy: ";
        for(int i = 0; i < std::min(3, kv_dim); ++i) { k_vals_to_log += std::to_string(k_batch_for_layer[current_token_batch_offset + i]) + " "; }
        if (kv_dim > 3) k_vals_to_log += "...";

        
        std::copy(k_batch_for_layer.begin() + current_token_batch_offset,
                  k_batch_for_layer.begin() + current_token_batch_offset + kv_dim,
                  layer_cache.k.begin() + destination_offset_in_layer_cache);
        

        size_t v_size_before = layer_cache.v.size();
        std::string v_vals_to_log = " vals to copy: ";
        for(int i = 0; i < std::min(3, kv_dim); ++i) { v_vals_to_log += std::to_string(v_batch_for_layer[current_token_batch_offset + i]) + " "; }
        if (kv_dim > 3) v_vals_to_log += "...";

        std::copy(v_batch_for_layer.begin() + current_token_batch_offset,
                  v_batch_for_layer.begin() + current_token_batch_offset + kv_dim,
                  layer_cache.v.begin() + destination_offset_in_layer_cache);

    }
    
}

void attention_batch_cpu(
    const std::vector<float>& q_batch_roped,
    KVCacheLayer& current_layer_kv_cache,
    std::vector<float>& batch_attn_output,
    int num_tokens_in_batch,
    int start_pos_in_sequence,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attention_scale
) {
    size_t expected_q_size = (size_t)num_tokens_in_batch * num_q_heads * head_dim;
    if (q_batch_roped.size() != expected_q_size) {
        Logger::error("[ATTN_BATCH_CPU] q_batch_roped size mismatch. Expected: " + std::to_string(expected_q_size) +
                      ", Got: " + std::to_string(q_batch_roped.size()));
        std::fill(batch_attn_output.begin(), batch_attn_output.end(), 0.0f);
        return;
    }
    Logger::info("[ATTENTION_BATCH_CPU_ENTRY] Called with num_tokens=" + std::to_string(num_tokens_in_batch));
    size_t expected_output_size = (size_t)num_tokens_in_batch * num_q_heads * head_dim;
    batch_attn_output.assign(expected_output_size, 0.0f);

    // Tokens are independent (each writes its own output rows; scores are
    // per-iteration), so they run in parallel with unchanged arithmetic.
#if defined(__EMSCRIPTEN_PTHREADS__)
    ember::parallel_for(num_tokens_in_batch, [&](int64_t token_idx) {
#else
#pragma omp parallel for schedule(dynamic)
    for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
#endif
        size_t q_token_offset = (size_t)token_idx * num_q_heads * head_dim;
        size_t attn_out_token_offset = (size_t)token_idx * num_q_heads * head_dim;
        int current_token_absolute_pos = start_pos_in_sequence + token_idx;

        for (int h_q = 0; h_q < num_q_heads; ++h_q) {
            const float* q_head_for_token_ptr = q_batch_roped.data() + q_token_offset + (h_q * head_dim);
            int kv_group_head_idx = h_q / (num_q_heads / num_kv_heads); 
            
            bool log_details_for_this_head = (token_idx == 0 && h_q == 0);


            int history_len = current_token_absolute_pos + 1;
            if (history_len <= 0) {
                 Logger::warning("[ATTN_BATCH_CPU] Token_idx " + std::to_string(token_idx) + ", Q_Head " + std::to_string(h_q) +
                                 ": history_len is " + std::to_string(history_len) + ". Skipping score calculation for this head.");
                continue;
            }
            std::vector<float> scores(history_len);

            for (int t_hist = 0; t_hist < history_len; ++t_hist) { 
                size_t k_cache_offset = ((size_t)t_hist * num_kv_heads + kv_group_head_idx) * head_dim;
                                if (token_idx == 0 && h_q == 0 && t_hist < 3) {
                  Logger::info("[CPU_ATTN_MEM] T" + std::to_string(token_idx) + "_H" + std::to_string(h_q) + 
                              " accessing K_cache[pos=" + std::to_string(t_hist) + ",kv_head=" + std::to_string(kv_group_head_idx) + 
                              "]: offset=" + std::to_string(k_cache_offset) + 
                              ", k_vals=[" + std::to_string(current_layer_kv_cache.k[k_cache_offset]) + 
                              "," + std::to_string(current_layer_kv_cache.k[k_cache_offset + 1]) + 
                              "," + std::to_string(current_layer_kv_cache.k[k_cache_offset + 2]) + "]");
              }
                if (k_cache_offset + head_dim > current_layer_kv_cache.k.size()) {
                     Logger::error("[ATTN_BATCH_CPU] K cache out of bounds. Token_idx " + std::to_string(token_idx) +
                                   " (abs_pos " + std::to_string(current_token_absolute_pos) + "), Q_Head " + std::to_string(h_q) +
                                   ", history_pos " + std::to_string(t_hist) +
                                   ". Required k_cache_offset " + std::to_string(k_cache_offset + head_dim) +
                                   " > cache_k_size " + std::to_string(current_layer_kv_cache.k.size()));
                    scores[t_hist] = -std::numeric_limits<float>::infinity(); 
                    continue;
                }

                float current_dot_product = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    current_dot_product += q_head_for_token_ptr[d] * current_layer_kv_cache.k[k_cache_offset + d];
                }
                                if (token_idx == 0 && h_q == 0 && t_hist < 2) {
                Logger::info("[CPU_ATTN_SCORE] T0_H0 pos=" + std::to_string(t_hist) + 
                            ", q_vals=[" + std::to_string(q_head_for_token_ptr[0]) + 
                            "," + std::to_string(q_head_for_token_ptr[1]) + "] " +
                            ", k_vals=[" + std::to_string(current_layer_kv_cache.k[k_cache_offset]) + 
                            "," + std::to_string(current_layer_kv_cache.k[k_cache_offset + 1]) + "]" +
                            ", dot=" + std::to_string(current_dot_product) + ", scale=" + std::to_string(attention_scale));
            }
                scores[t_hist] = current_dot_product * attention_scale;

            }

            softmax_vector_cpu(scores, scores); 
            if (token_idx == 0 && h_q == 0) {
                std::string scores_str = "";
                for (int i = 0; i < std::min(3, (int)scores.size()); i++) {
                    scores_str += std::to_string(scores[i]) + " ";
                }
                Logger::info("[CPU_SOFTMAX] T0_H0 first_3_probs=[" + scores_str + "]");
            }
            float* current_attn_out_head_ptr = batch_attn_output.data() + attn_out_token_offset + (h_q * head_dim);

            for (int t_hist = 0; t_hist < history_len; ++t_hist) {
                if (scores[t_hist] == -std::numeric_limits<float>::infinity() || scores[t_hist] == 0.0f) continue;

                size_t v_cache_offset = ((size_t)t_hist * num_kv_heads + kv_group_head_idx) * head_dim;
                if (v_cache_offset + head_dim > current_layer_kv_cache.v.size()) {
                     Logger::error("[ATTN_BATCH_CPU] V cache out of bounds. Token_idx " + std::to_string(token_idx) +
                                   " (abs_pos " + std::to_string(current_token_absolute_pos) + "), Q_Head " + std::to_string(h_q) +
                                   ", history_pos " + std::to_string(t_hist) +
                                   ". Required v_cache_offset " + std::to_string(v_cache_offset + head_dim) +
                                   " > cache_v_size " + std::to_string(current_layer_kv_cache.v.size()));
                    continue; 
                }

                for (int d = 0; d < head_dim; ++d) {
                    float val_before = (log_details_for_this_head && t_hist < 2 && d < 2) ? current_attn_out_head_ptr[d] : 0.0f;
                    current_attn_out_head_ptr[d] += scores[t_hist] * current_layer_kv_cache.v[v_cache_offset + d];
                }
            }
        } 
    }
#if defined(__EMSCRIPTEN_PTHREADS__)
    );
#endif 
}

void swa_visible_ranges(int pos, int window, int sinks, int& sink_hi, int& win_lo) {
    // paperkiln ops.cpp swa_attention, ranges(): [0, sink_hi) then [win_lo, pos].
    win_lo = pos + 1 > window ? pos + 1 - window : 0;
    sink_hi = std::min(std::min(sinks, pos + 1), win_lo);
}

void attention_window_batch_cpu(
    const std::vector<float>& q_batch_roped,
    KVCacheLayer& current_layer_kv_cache,
    std::vector<float>& batch_attn_output,
    int num_tokens_in_batch,
    int start_pos_in_sequence,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attention_scale,
    int window,
    int sinks
) {
    // window 0 is unbounded attention: today's kernel, untouched.
    if (window <= 0) {
        attention_batch_cpu(q_batch_roped, current_layer_kv_cache, batch_attn_output, num_tokens_in_batch,
                            start_pos_in_sequence, num_q_heads, num_kv_heads, head_dim, attention_scale);
        return;
    }
    const size_t expected_size = (size_t)num_tokens_in_batch * num_q_heads * head_dim;
    batch_attn_output.assign(expected_size, 0.0f);
    if (q_batch_roped.size() != expected_size) {
        Logger::error("[ATTN_WINDOW_CPU] q_batch_roped size mismatch. Expected: " + std::to_string(expected_size) +
                      ", Got: " + std::to_string(q_batch_roped.size()));
        return;
    }
    const size_t needed = (size_t)(start_pos_in_sequence + num_tokens_in_batch) * num_kv_heads * head_dim;
    if (needed > current_layer_kv_cache.k.size() || needed > current_layer_kv_cache.v.size()) {
        Logger::error("[ATTN_WINDOW_CPU] KV cache holds fewer than " +
                      std::to_string(start_pos_in_sequence + num_tokens_in_batch) + " positions");
        return;
    }

    // The visible keys in ascending position (sinks, then the window), each
    // scored, softmaxed and accumulated exactly as attention_batch_cpu does
    // over its full history: when the window covers [0, pos] the two kernels
    // perform the same operations in the same order.
#if defined(__EMSCRIPTEN_PTHREADS__)
    ember::parallel_for(num_tokens_in_batch, [&](int64_t token_idx) {
#else
#pragma omp parallel for schedule(dynamic)
    for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
#endif
        const size_t token_offset = (size_t)token_idx * num_q_heads * head_dim;
        const int pos = start_pos_in_sequence + (int)token_idx;
        int sink_hi = 0, win_lo = 0;
        swa_visible_ranges(pos, window, sinks, sink_hi, win_lo);
        std::vector<int> visible;
        visible.reserve((size_t)sink_hi + (pos + 1 - win_lo));
        for (int t = 0; t < sink_hi; ++t) visible.push_back(t);
        for (int t = win_lo; t <= pos; ++t) visible.push_back(t);
        std::vector<float> scores(visible.size());

        for (int h_q = 0; h_q < num_q_heads; ++h_q) {
            const float* q_head = q_batch_roped.data() + token_offset + (h_q * head_dim);
            const int kv_head = h_q / (num_q_heads / num_kv_heads);
            for (size_t i = 0; i < visible.size(); ++i) {
                const size_t k_off = ((size_t)visible[i] * num_kv_heads + kv_head) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += q_head[d] * current_layer_kv_cache.k[k_off + d];
                scores[i] = dot * attention_scale;
            }
            softmax_vector_cpu(scores, scores);
            float* out_head = batch_attn_output.data() + token_offset + (h_q * head_dim);
            for (size_t i = 0; i < visible.size(); ++i) {
                if (scores[i] == -std::numeric_limits<float>::infinity() || scores[i] == 0.0f) continue;
                const size_t v_off = ((size_t)visible[i] * num_kv_heads + kv_head) * head_dim;
                for (int d = 0; d < head_dim; ++d) out_head[d] += scores[i] * current_layer_kv_cache.v[v_off + d];
            }
        }
    }
#if defined(__EMSCRIPTEN_PTHREADS__)
    );
#endif
}

void attention_rows_batch_cpu(
    const std::vector<float>& q_batch_roped,
    const KVCacheLayer& current_layer_kv_cache,
    std::vector<float>& batch_attn_output,
    int num_tokens_in_batch,
    const std::vector<std::vector<int>>& rows,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attention_scale
) {
    const size_t expected_size = (size_t)num_tokens_in_batch * num_q_heads * head_dim;
    batch_attn_output.assign(expected_size, 0.0f);
    if (q_batch_roped.size() != expected_size || rows.size() != (size_t)num_tokens_in_batch) {
        Logger::error("[ATTN_ROWS_CPU] q or row lists do not match the batch");
        return;
    }
    const size_t row_floats = (size_t)num_kv_heads * head_dim;
    // The body of attention_window_batch_cpu with the visible list given.
#if defined(__EMSCRIPTEN_PTHREADS__)
    ember::parallel_for(num_tokens_in_batch, [&](int64_t token_idx) {
#else
#pragma omp parallel for schedule(dynamic)
    for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
#endif
        const size_t token_offset = (size_t)token_idx * num_q_heads * head_dim;
        const std::vector<int>& visible = rows[token_idx];
        std::vector<float> scores(visible.size());
        bool in_range = !visible.empty();
        for (int r : visible)
            if (r < 0 || ((size_t)r + 1) * row_floats > current_layer_kv_cache.k.size() ||
                ((size_t)r + 1) * row_floats > current_layer_kv_cache.v.size())
                in_range = false;
        if (in_range) {
            for (int h_q = 0; h_q < num_q_heads; ++h_q) {
                const float* q_head = q_batch_roped.data() + token_offset + (h_q * head_dim);
                const int kv_head = h_q / (num_q_heads / num_kv_heads);
                for (size_t i = 0; i < visible.size(); ++i) {
                    const size_t k_off = ((size_t)visible[i] * num_kv_heads + kv_head) * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) dot += q_head[d] * current_layer_kv_cache.k[k_off + d];
                    scores[i] = dot * attention_scale;
                }
                softmax_vector_cpu(scores, scores);
                float* out_head = batch_attn_output.data() + token_offset + (h_q * head_dim);
                for (size_t i = 0; i < visible.size(); ++i) {
                    if (scores[i] == -std::numeric_limits<float>::infinity() || scores[i] == 0.0f) continue;
                    const size_t v_off = ((size_t)visible[i] * num_kv_heads + kv_head) * head_dim;
                    for (int d = 0; d < head_dim; ++d) out_head[d] += scores[i] * current_layer_kv_cache.v[v_off + d];
                }
            }
        }
    }
#if defined(__EMSCRIPTEN_PTHREADS__)
    );
#endif
}

void update_kv_cache_batch_cpu_sequence_aware(
    KVCache* kv_cache,
    int layer_idx,
    const std::vector<float>& k_batch_for_layer,
    const std::vector<float>& v_batch_for_layer,
    int num_tokens_in_batch,
    const std::vector<int>& sequence_indices,
    const std::vector<int>& position_in_sequence,
    int num_kv_heads,
    int head_dim
) {
    if (!kv_cache) {
        Logger::error("update_kv_cache_batch_cpu_sequence_aware: KVCache is null.");
        return;
    }
    if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= kv_cache->layers.size()) {
        Logger::error("update_kv_cache_batch_cpu_sequence_aware: layer_idx " + std::to_string(layer_idx) + 
                      " is out of bounds for KVCache layers (size " + std::to_string(kv_cache->layers.size()) + ").");
        return;
    }
    
    KVCacheLayer& layer_cache = kv_cache->layers[layer_idx];
    int kv_dim = num_kv_heads * head_dim;

    for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
        size_t current_token_batch_offset = static_cast<size_t>(token_idx) * kv_dim;
        
        int seq_idx = sequence_indices[token_idx];
        int pos_in_seq = position_in_sequence[token_idx];
        
        int sequence_base_offset = seq_idx * kv_cache->max_seq_len_config_;
        int actual_cache_position = sequence_base_offset + pos_in_seq;
        if (actual_cache_position >= kv_cache->max_seq_len_config_ * kv_cache->max_batch_size) {
            Logger::error("[KV_BATCH_UPDATE_SEQ_AWARE L" + std::to_string(layer_idx) + 
                          "] Error: actual_cache_position (" + std::to_string(actual_cache_position) +
                          ") is out of bounds for total cache size. Skipping update for this token.");
            continue;
        }
        
        size_t destination_offset_in_layer_cache = static_cast<size_t>(actual_cache_position) * kv_dim;
        
        std::copy(k_batch_for_layer.begin() + current_token_batch_offset,
                  k_batch_for_layer.begin() + current_token_batch_offset + kv_dim,
                  layer_cache.k.begin() + destination_offset_in_layer_cache);
                  
        std::copy(v_batch_for_layer.begin() + current_token_batch_offset,
                  v_batch_for_layer.begin() + current_token_batch_offset + kv_dim,
                  layer_cache.v.begin() + destination_offset_in_layer_cache);
    }
}

void attention_batch_cpu_sequence_aware(
    const std::vector<float>& q_batch_roped,
    KVCacheLayer& current_layer_kv_cache,
    std::vector<float>& batch_attn_output,
    int num_tokens_in_batch,
    const std::vector<int>& sequence_indices,
    const std::vector<int>& position_in_sequence,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attention_scale,
    int max_seq_len_per_sequence
) {
    size_t expected_q_size = (size_t)num_tokens_in_batch * num_q_heads * head_dim;
    if (q_batch_roped.size() != expected_q_size) {
        Logger::error("[ATTN_BATCH_CPU_SEQ_AWARE] q_batch_roped size mismatch. Expected: " + std::to_string(expected_q_size) +
                      ", Got: " + std::to_string(q_batch_roped.size()));
        std::fill(batch_attn_output.begin(), batch_attn_output.end(), 0.0f);
        return;
    }
    
    batch_attn_output.assign((size_t)num_tokens_in_batch * num_q_heads * head_dim, 0.0f);

    for (int token_idx = 0; token_idx < num_tokens_in_batch; ++token_idx) {
        size_t q_token_offset = (size_t)token_idx * num_q_heads * head_dim;
        size_t attn_out_token_offset = (size_t)token_idx * num_q_heads * head_dim;
        
        int seq_idx = sequence_indices[token_idx];
        int pos_in_seq = position_in_sequence[token_idx];
        int sequence_base_offset = seq_idx * max_seq_len_per_sequence;

        for (int h_q = 0; h_q < num_q_heads; ++h_q) {
            const float* q_head_for_token_ptr = q_batch_roped.data() + q_token_offset + (h_q * head_dim);
            int kv_group_head_idx = h_q / (num_q_heads / num_kv_heads);
            
            int history_len = pos_in_seq + 1;
            std::vector<float> scores(history_len);

            for (int t_hist = 0; t_hist < history_len; ++t_hist) {
                size_t k_cache_offset = ((size_t)(sequence_base_offset + t_hist) * num_kv_heads + kv_group_head_idx) * head_dim;
                
                if (k_cache_offset + head_dim > current_layer_kv_cache.k.size()) {
                    scores[t_hist] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float current_dot_product = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    current_dot_product += q_head_for_token_ptr[d] * current_layer_kv_cache.k[k_cache_offset + d];
                }
                scores[t_hist] = current_dot_product * attention_scale;
            }

            softmax_vector_cpu(scores, scores);
            
            float* current_attn_out_head_ptr = batch_attn_output.data() + attn_out_token_offset + (h_q * head_dim);

            for (int t_hist = 0; t_hist < history_len; ++t_hist) {
                if (scores[t_hist] == -std::numeric_limits<float>::infinity() || scores[t_hist] == 0.0f) continue;

                size_t v_cache_offset = ((size_t)(sequence_base_offset + t_hist) * num_kv_heads + kv_group_head_idx) * head_dim;
                if (v_cache_offset + head_dim > current_layer_kv_cache.v.size()) {
                    continue;
                }

                for (int d = 0; d < head_dim; ++d) {
                    current_attn_out_head_ptr[d] += scores[t_hist] * current_layer_kv_cache.v[v_cache_offset + d];
                }
            }
        }
    }
} 