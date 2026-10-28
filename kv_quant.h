#pragma once
// Quantised KV cache for the IR interpreter (EMBER_KV=f32|f16|q8|q4; research item R2,
// experiments/kv_quant_r2_2026-10-28). f32 is the default and leaves every existing path as it was.
//
// A cache row is one position's [kv_heads][head_dim] floats. The formats, per row:
//   f16  IEEE half per element, round to nearest even (2 bytes per element).
//   q8   blocks of 32 consecutive elements (whole blocks inside a head: head_dim % 32 == 0), llama.cpp's
//        q8_0 rule: d = half(max |x| / 127), q = clamp(rint(x / d), -127, 127), x' = q * d
//        (1 byte per element + a half scale per block: 1.0625 bytes per element).
//   q4   blocks of 32, asymmetric (llama.cpp's q4_1 form): d = half((max - min) / 15), m = half(min),
//        q = clamp(rint((x - m) / d), 0, 15), x' = fma(q, d, m); element 2i in the low nibble of byte i,
//        2i + 1 in the high one; (d, m) a half pair per block (0.625 bytes per element).
// Every rounding is explicit (division, product and fma rounded once each, rint half to even), so the
// CPU form below and the CUDA kernels (kv_quant.cu) quantise a given float to the same code.
//
// CPU: the interpreter keeps its f32 cache and replaces each written row by its dequantised value
// (roundtrip_rows): the numerics of a quantised cache, not its memory. CUDA stores the codes and
// dequantises in the attention kernels' load path: the tiled split-K decode kernel's form
// (decode_attn_tiled.cuh) reading codes, and prefill through a dequantised f32 copy of the rows
// [0, start_pos + T) of the layer (one copy shared by the layers) read by the existing prefill kernel.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace ember_kvq {

enum Format : int { kF32 = 0, kF16 = 1, kQ8 = 2, kQ4 = 3 };
constexpr int kBlock = 32;

// EMBER_KV's value: "" or "f32" -> kF32, "f16", "q8", "q4"; anything else throws.
inline Format parse_format(const char* s) {
  const std::string v = s ? s : "";
  if (v.empty() || v == "f32") return kF32;
  if (v == "f16") return kF16;
  if (v == "q8") return kQ8;
  if (v == "q4") return kQ4;
  throw std::runtime_error("EMBER_KV=" + v + ": expected f32, f16, q8 or q4");
}
inline const char* format_name(int f) { return f == kF16 ? "f16" : f == kQ8 ? "q8" : f == kQ4 ? "q4" : "f32"; }

// Bytes of one row's codes and of its scales (row = kv_heads * head_dim elements).
inline size_t data_bytes(int f, size_t row) {
  return f == kF16 ? row * 2 : f == kQ8 ? row : f == kQ4 ? row / 2 : row * 4;
}
inline size_t scale_bytes(int f, size_t row) {
  return f == kQ8 ? row / kBlock * 2 : f == kQ4 ? row / kBlock * 4 : 0;
}
// Why a shape cannot take format f ("" when it can).
inline std::string shape_error(int f, int head_dim) {
  if (f == kF32) return "";
  if (head_dim % 4 != 0) return "head_dim % 4 != 0";
  if ((f == kQ8 || f == kQ4) && head_dim % kBlock != 0) return "head_dim % 32 != 0 (q8 and q4 blocks sit inside a head)";
  return "";
}

// IEEE binary16 conversions, round to nearest even (CUDA's __float2half_rn), subnormals and
// infinities included; NaN stays NaN.
inline uint16_t f32_to_f16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const uint32_t ax = u & 0x7fffffffu;
  if (ax >= 0x7f800000u) return static_cast<uint16_t>(sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u));
  if (ax >= 0x477ff000u) return static_cast<uint16_t>(sign | 0x7c00u);  // rounds past 65504: infinity
  if (ax < 0x38800000u) {  // below 2^-14: a half subnormal (or zero)
    if (ax < 0x33000000u) return static_cast<uint16_t>(sign);  // below half the smallest subnormal: 0
    const uint32_t e = ax >> 23;
    const uint32_t m = (ax & 0x7fffffu) | 0x800000u;
    const uint32_t shift = 126 - e;  // r = m * 2^(e - 126) units of 2^-24, m with its implicit bit
    const uint32_t base = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u);
    const uint32_t half_ulp = 1u << (shift - 1);
    uint32_t r = base;
    if (rem > half_ulp || (rem == half_ulp && (base & 1u))) ++r;
    return static_cast<uint16_t>(sign | r);
  }
  uint32_t r = ((ax >> 13) - (112u << 10));  // rebias the exponent, keep 10 mantissa bits
  const uint32_t rem = ax & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;  // a carry into the exponent is correct
  return static_cast<uint16_t>(sign | r);
}
inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  uint32_t u;
  if (e == 0) {
    if (m == 0) {
      u = sign;
    } else {  // subnormal: m * 2^-24, exact in float
      const float v = static_cast<float>(m) * 5.9604644775390625e-8f;
      std::memcpy(&u, &v, 4);
      u |= sign;
    }
  } else if (e == 31) {
    u = sign | 0x7f800000u | (m << 13);
  } else {
    u = sign | ((e + 112u) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// One block of 32 floats quantised and dequantised in place (q8, q4) or each element through half (f16).
inline void roundtrip_block(int f, float* x) {
  if (f == kQ8) {
    float amax = 0.0f;
    for (int i = 0; i < kBlock; ++i) amax = std::fmax(amax, std::fabs(x[i]));
    const float d = f16_to_f32(f32_to_f16(amax / 127.0f));
    for (int i = 0; i < kBlock; ++i) {
      float q = d != 0.0f ? std::nearbyint(x[i] / d) : 0.0f;
      q = std::fmin(127.0f, std::fmax(-127.0f, q));
      x[i] = q * d;
    }
  } else if (f == kQ4) {
    float mn = x[0], mx = x[0];
    for (int i = 1; i < kBlock; ++i) {
      mn = std::fmin(mn, x[i]);
      mx = std::fmax(mx, x[i]);
    }
    const float d = f16_to_f32(f32_to_f16((mx - mn) / 15.0f));
    const float m = f16_to_f32(f32_to_f16(mn));
    for (int i = 0; i < kBlock; ++i) {
      float q = d != 0.0f ? std::nearbyint((x[i] - m) / d) : 0.0f;
      q = std::fmin(15.0f, std::fmax(0.0f, q));
      x[i] = std::fma(q, d, m);
    }
  }
}

// rows rows of `row` floats each, in place: what the cache gives back after storing them in format f.
inline void roundtrip_rows(int f, float* x, size_t rows, size_t row) {
  if (f == kF32) return;
  const size_t n = rows * row;
  if (f == kF16) {
    for (size_t i = 0; i < n; ++i) x[i] = f16_to_f32(f32_to_f16(x[i]));
    return;
  }
  for (size_t b = 0; b + kBlock <= n; b += kBlock) roundtrip_block(f, x + b);
}

#ifdef HAS_CUDA
// The rows of k, v [T][row] stored in format f at positions start_pos + t (or *dpos + t when dpos is
// given: the captured decode step). Kd/Vd: codes [max_len][data_bytes(row)]; Ks/Vs: scales
// [max_len][scale_bytes(row)] (null for f16).
void write_cuda(int f, void* Kd, void* Ks, void* Vd, void* Vs, const float* k, const float* v, int T, int row,
                int start_pos, cudaStream_t stream, const int* dpos = nullptr);
// One query row per head (q [hq][hd]) over the L = seq_len (or *dpos + 1) cached positions: the tiled
// split-K decode attention (decode_attn_tiled.cuh's chunking, lane layout and merge), codes dequantised as
// they are loaded. part: partial_floats(hq, hd, max_len) floats of scratch.
size_t partial_floats(int hq, int hd, int max_len);
void decode_cuda(int f, const float* q, const void* Kd, const void* Ks, const void* Vd, const void* Vs, float* out,
                 int hq, int hkv, int hd, float scale, int max_len, int seq_len, float* part, cudaStream_t stream,
                 const int* dpos = nullptr);
// Rows [0, n) dequantised to f32 [n][row] (for the prefill kernel).
void dequant_rows_cuda(int f, const void* Kd, const void* Ks, const void* Vd, const void* Vs, float* K, float* V, int n,
                       int row, cudaStream_t stream);
#endif

}  // namespace ember_kvq
