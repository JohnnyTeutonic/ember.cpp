#pragma once

#include <vector>
#include "kv_cache.h"

void update_kv_cache_batch_cpu(
    KVCache* kv_cache,
    int layer_idx,
    const std::vector<float>& k_batch_for_layer,
    const std::vector<float>& v_batch_for_layer,
    int num_tokens_in_batch,
    int start_pos_in_sequence,
    int num_kv_heads,
    int head_dim
);

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
);

// Sliding-window attention with attention sinks, paperkiln's swa_attention
// (microtorch/src/ops.cpp): query at absolute position p sees keys
// [0, sink_hi) and [win_lo, p], with win_lo = max(0, p + 1 - window) and
// sink_hi = min(sinks, p + 1, win_lo). The window is the last `window`
// positions including p; sinks are the first `sinks` positions, not counted
// in the window, and a sink inside the window is seen once. window 0 means
// unbounded (sinks then have no effect): it calls attention_batch_cpu.
void swa_visible_ranges(int pos, int window, int sinks, int& sink_hi, int& win_lo);

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
);

// Attention over an explicit list of visible kv rows per query: token t of
// the batch sees rows[t] (ascending), each scored, softmaxed (maximum and sum
// over the listed rows only, in list order) and accumulated exactly as
// attention_window_batch_cpu does, so a list holding rows 0 .. pos performs
// attention_batch_cpu's operations in its order. Used by the IR's
// turn_state mask (graph_ir.cpp), whose rows may be a compact cache.
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
);

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
);

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
); 