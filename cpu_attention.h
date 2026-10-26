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

// The visible kv rows of each query of a batch at positions start_pos .. start_pos + T - 1 over a cache whose row r
// holds position r: [0, sink_hi) then [win_lo, p] (swa_visible_ranges), or 0 .. p when window is 0.
std::vector<std::vector<int>> attention_visible_rows(int num_tokens_in_batch, int start_pos_in_sequence, int window,
                                                     int sinks);
// attention_rows_batch_cpu with T5's relative position bias (paperkiln IR_SPEC 5.15 position_bias, revision 1.3):
// each score is (q . k) * scale, rounded, then + bias[b(p - j) * num_q_heads + h], rounded, with p the query's
// position (query_pos[t]), j the key's (key_pos[row], or the row itself when key_pos is null) and
// b(n) = buckets[min(n, buckets.size() - 1)]; bias is the table [B][num_q_heads] row-major. Under grouped-query
// attention the query head's column. Softmax and accumulation as attention_rows_batch_cpu.
void attention_bias_batch_cpu(
    const std::vector<float>& q_batch_roped,
    const KVCacheLayer& current_layer_kv_cache,
    std::vector<float>& batch_attn_output,
    int num_tokens_in_batch,
    const std::vector<std::vector<int>>& rows,
    const std::vector<int>* key_pos,
    const std::vector<int>& query_pos,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attention_scale,
    const float* bias,
    const std::vector<int>& buckets
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