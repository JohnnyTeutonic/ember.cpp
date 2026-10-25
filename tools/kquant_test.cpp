// kquant_test: ember's K-quant dequantisers (Q4_K, Q5_K, Q6_K) against gguf-py's reference, bit for bit.
//
//   kquant_test blocks FILE          random blocks and gguf-py's values (tools/kquant_blocks.py writes FILE)
//   kquant_test gguf Q.gguf REF.gguf every quantised tensor of Q.gguf against REF.gguf's F32 copy
//                                    (REF from tools/quantize_gguf.py Q.gguf REF.gguf --dequantize)
//
// FILE holds records: int32 ggml type, int32 block count n, n raw blocks, then n * 256 float32 values.
// Each block goes through all three of ember's routes: the block routine the llama path's
// ensure_*_dequantized uses (dequantize_vector_q4k/q5k/q6k_to_f32), dequant_span_to_f32 (the
// interpreter's row-wise route) and dequant_legacy_to_f32 (whole tensors). Any differing bit fails.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gguf_parser.h"
#include "quantization.h"
#include "weight_management.h"

namespace {

struct Tally {
  size_t values = 0, bit_diffs = 0;
  double max_abs = 0.0;
};

void compare(const float* got, const float* ref, size_t n, Tally& t) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t a, b;
    std::memcpy(&a, got + i, 4);
    std::memcpy(&b, ref + i, 4);
    t.values++;
    if (a != b) {
      t.bit_diffs++;
      const double d = std::fabs(double(got[i]) - double(ref[i]));
      if (!(d <= t.max_abs)) t.max_abs = d;  // NaN-safe: a NaN difference is recorded too
    }
  }
}

std::vector<float> by_blocks(GGMLType t, const uint8_t* raw, size_t nb) {
  std::vector<float> out;
  const size_t n = nb * GGML_QK_K;
  if (t == GGMLType::GGML_TYPE_Q4_K) {
    std::vector<block_q4_K> b(nb);
    std::memcpy(b.data(), raw, nb * sizeof(block_q4_K));
    dequantize_vector_q4k_to_f32(b, out, n);
  } else if (t == GGMLType::GGML_TYPE_Q5_K) {
    std::vector<block_q5_K> b(nb);
    std::memcpy(b.data(), raw, nb * sizeof(block_q5_K));
    dequantize_vector_q5k_to_f32(b, out, n);
  } else {
    std::vector<block_q6_K> b(nb);
    std::memcpy(b.data(), raw, nb * sizeof(block_q6_K));
    dequantize_vector_q6k_to_f32(b, out, n);
  }
  return out;
}

bool k_type(GGMLType t) {
  return t == GGMLType::GGML_TYPE_Q4_K || t == GGMLType::GGML_TYPE_Q5_K || t == GGMLType::GGML_TYPE_Q6_K;
}

int run_blocks(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "cannot open " << path << "\n";
    return 2;
  }
  bool ok = true;
  for (;;) {
    int32_t type = 0, nb = 0;
    if (!f.read(reinterpret_cast<char*>(&type), 4)) break;
    f.read(reinterpret_cast<char*>(&nb), 4);
    const GGMLType t = static_cast<GGMLType>(type);
    if (!k_type(t) || nb <= 0) {
      std::cerr << "bad record: type " << type << ", " << nb << " blocks\n";
      return 2;
    }
    const size_t bsz = ggml_type_size(t), n = static_cast<size_t>(nb) * GGML_QK_K;
    std::vector<uint8_t> raw(static_cast<size_t>(nb) * bsz);
    std::vector<float> ref(n);
    f.read(reinterpret_cast<char*>(raw.data()), raw.size());
    f.read(reinterpret_cast<char*>(ref.data()), n * 4);
    if (!f) {
      std::cerr << "truncated record for " << ggml_type_name(t) << "\n";
      return 2;
    }
    Tally tb, ts, tl;
    compare(by_blocks(t, raw.data(), nb).data(), ref.data(), n, tb);
    std::vector<float> span(n);
    if (!dequant_span_to_f32(t, raw.data(), 0, n, span.data())) {
      std::cerr << "dequant_span_to_f32 refuses " << ggml_type_name(t) << "\n";
      return 1;
    }
    compare(span.data(), ref.data(), n, ts);
    GGUFTensorInfo info;
    info.type = t;
    info.num_elements = n;
    info.size_in_bytes = raw.size();
    std::vector<float> whole;
    if (!dequant_legacy_to_f32(info, raw.data(), whole)) {
      std::cerr << "dequant_legacy_to_f32 refuses " << ggml_type_name(t) << "\n";
      return 1;
    }
    compare(whole.data(), ref.data(), n, tl);
    for (const auto& [route, tally] : {std::pair<const char*, Tally&>{"blocks", tb}, {"span", ts}, {"tensor", tl}}) {
      std::printf("%-4s %-6s %6d blocks, %8zu values: %zu differing bits (max |d| %.3g)\n", ggml_type_name(t), route,
                  nb, tally.values, tally.bit_diffs, tally.max_abs);
      ok = ok && tally.bit_diffs == 0;
    }
  }
  std::printf("%s\n", ok ? "PASS: every value bit-identical to gguf-py" : "FAIL");
  return ok ? 0 : 1;
}

const uint8_t* tensor_base(const GGUFData& g) {
  return g.mapped_tensor_data ? static_cast<const uint8_t*>(g.mapped_tensor_data) + g.offset_diff_for_mmap
                              : g.tensor_data.data();
}

int run_gguf(const char* qpath, const char* refpath) {
  const GGUFData q = load_gguf_meta(qpath, true);
  const GGUFData r = load_gguf_meta(refpath, true);
  bool ok = true;
  size_t n_tensors = 0;
  Tally all;
  for (const auto& [name, info] : q.tensor_infos_map) {
    if (info.type == GGMLType::GGML_TYPE_F32) continue;
    const auto it = r.tensor_infos_map.find(name);
    if (it == r.tensor_infos_map.end() || it->second.type != GGMLType::GGML_TYPE_F32 ||
        it->second.num_elements != info.num_elements) {
      std::printf("%s: no F32 tensor of %zu elements in the reference\n", name.c_str(), info.num_elements);
      ok = false;
      continue;
    }
    std::vector<float> got;
    if (!dequant_legacy_to_f32(info, tensor_base(q) + info.offset, got)) {
      std::printf("%s: type %s not dequantised by ember\n", name.c_str(), ggml_type_name(info.type));
      ok = false;
      continue;
    }
    const float* ref = reinterpret_cast<const float*>(tensor_base(r) + it->second.offset);
    Tally t;
    compare(got.data(), ref, info.num_elements, t);
    if (k_type(info.type)) {  // the llama path's block routine as well
      compare(by_blocks(info.type, tensor_base(q) + info.offset, info.num_elements / GGML_QK_K).data(), ref,
              info.num_elements, t);
    }
    if (t.bit_diffs) {
      std::printf("%s (%s): %zu differing bits, max |d| %.3g\n", name.c_str(), ggml_type_name(info.type), t.bit_diffs,
                  t.max_abs);
      ok = false;
    }
    all.values += t.values;
    all.bit_diffs += t.bit_diffs;
    n_tensors++;
  }
  std::printf("%zu quantised tensors, %zu values compared: %zu differing bits\n%s\n", n_tensors, all.values,
              all.bit_diffs, ok ? "PASS: every tensor bit-identical to gguf-py's dequantised F32" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "blocks") return run_blocks(argv[2]);
  if (argc == 4 && std::string(argv[1]) == "gguf") return run_gguf(argv[2], argv[3]);
  std::cerr << "usage: kquant_test blocks FILE | kquant_test gguf Q.gguf REF.gguf\n";
  return 2;
}
