#pragma once

#include "gguf_structs.h"
#include "model.h"

// Dequantises a GGUF tensor of type F16, Q4_0, Q4_1, Q5_0 or Q5_1 into f32
// (element order unchanged); false for any other type.
bool dequant_legacy_to_f32(const GGUFTensorInfo& info, const uint8_t* src, std::vector<float>& out);
