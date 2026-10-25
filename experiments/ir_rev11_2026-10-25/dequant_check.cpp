// dequant_check: ember's dequantisers (the interpreter's route dequant_span_to_f32 and the whole-tensor route
// dequant_legacy_to_f32 / dequantize_vector_q8_0_to_f32 the interpreter's binding uses) on paperkiln's
// spec/conformance/quant/blocks.json (IR_SPEC 4.7), bit for bit, for one build: native, or WebAssembly in node
// (NODERAWFS). Prints one JSON object: per case the type and whether every value's bits equal the fixture's.
//   dequant_check BLOCKS.json
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_structs.h"
#include "quantization.h"
#include "weight_management.h"

using nlohmann::json;

static std::vector<uint8_t> unhex(const std::string& h) {
  std::vector<uint8_t> out(h.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<uint8_t>(std::stoul(h.substr(2 * i, 2), nullptr, 16));
  return out;
}

static std::vector<uint32_t> bits(const std::string& hex) {
  std::vector<uint32_t> out(hex.size() / 8);
  for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<uint32_t>(std::stoul(hex.substr(8 * i, 8), nullptr, 16));
  return out;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: dequant_check BLOCKS.json\n";
    return 2;
  }
  std::ifstream f(argv[1]);
  std::stringstream ss;
  ss << f.rdbuf();
  const json doc = json::parse(ss.str());
  json out = json::object();
  bool all = true;
  for (const auto& c : doc.at("cases")) {
    const GGMLType t = static_cast<GGMLType>(c.at("ggml_type").get<int>());
    const size_t n = c.at("rows").get<size_t>() * c.at("n").get<size_t>();
    const std::vector<uint8_t> raw = unhex(c.at("bytes").get<std::string>());
    const std::vector<uint32_t> want = bits(c.at("expect").at("f32").get<std::string>());
    std::vector<float> span(n);
    bool span_ok;
    if (t == GGMLType::GGML_TYPE_F32) {
      std::memcpy(span.data(), raw.data(), n * 4);
      span_ok = true;
    } else {
      span_ok = dequant_span_to_f32(t, raw.data(), 0, n, span.data());
    }
    std::vector<float> whole;
    bool whole_ok = true;
    if (t == GGMLType::GGML_TYPE_Q8_0) {
      std::vector<block_q8_0> blocks(raw.size() / sizeof(block_q8_0));
      std::memcpy(blocks.data(), raw.data(), blocks.size() * sizeof(block_q8_0));
      dequantize_vector_q8_0_to_f32(blocks, whole, n);
    } else if (t == GGMLType::GGML_TYPE_F32) {
      whole = span;
    } else {
      GGUFTensorInfo info;
      info.type = t;
      info.num_elements = n;
      info.size_in_bytes = raw.size();
      whole_ok = dequant_legacy_to_f32(info, raw.data(), whole);
    }
    size_t bad_span = 0, bad_whole = 0;
    for (size_t i = 0; i < n; ++i) {
      uint32_t a, b;
      std::memcpy(&a, &span[i], 4);
      if (i < whole.size()) std::memcpy(&b, &whole[i], 4); else b = ~want[i];
      const bool nan_w = (want[i] & 0x7f800000u) == 0x7f800000u && (want[i] & 0x7fffffu);
      const bool nan_a = (a & 0x7f800000u) == 0x7f800000u && (a & 0x7fffffu);
      const bool nan_b = (b & 0x7f800000u) == 0x7f800000u && (b & 0x7fffffu);
      bad_span += !(a == want[i] || (nan_a && nan_w));
      bad_whole += !(b == want[i] || (nan_b && nan_w));
    }
    const bool ok = span_ok && whole_ok && bad_span == 0 && bad_whole == 0;
    all = all && ok;
    out[c.at("name").get<std::string>()] = {{"type", c.at("type")}, {"n", n}, {"span_bad", bad_span},
                                            {"whole_bad", bad_whole}, {"pass", ok}};
  }
  out["pass"] = all;
  std::cout << out.dump() << "\n";
  return all ? 0 : 1;
}
