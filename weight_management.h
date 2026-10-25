#pragma once

#include "gguf_structs.h"
#include "model.h"

// Dequantises a GGUF tensor of type F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K or Q6_K into
// f32 (element order unchanged); false for any other type.
bool dequant_legacy_to_f32(const GGUFTensorInfo& info, const uint8_t* src, std::vector<float>& out);

// Dequantises elements [first, first + n) of a tensor of type F16, BF16, Q8_0,
// Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K or Q6_K stored at src (for the block types,
// first and n are multiples of the block width: 32, or 256 for the K-quants).
// Each value equals the one the whole-tensor dequantisers (dequant_legacy_to_f32,
// dequantize_vector_q8_0_to_f32, dequantize_vector_q4k/q5k/q6k_to_f32) give it:
// they share these per-block routines. False for any other type.
bool dequant_span_to_f32(GGMLType t, const uint8_t* src, size_t first, size_t n, float* out);
