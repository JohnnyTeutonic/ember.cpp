// ir_device_dump: the device half of tools/ir_device_gate.cpp, written to a file, so two ember builds can be
// compared bit for bit on one GPU (a change that should move no device result: diff the two files). The
// prompts are ir_device_gate's (the same seeded random ids, or --ids FILE); for each one the file gets the
// device's prefill logits (every row; --ids: the last row), its token-by-token decode logits (the captured
// CUDA graph after the first step; --ids: the last row) and gen greedy ids, raw float32 / int32 in that
// order. Only the public interpreter API is used, so the file compiles against older trees.
//
//   ir_device_dump MODEL.gguf OUT.bin [--n 40] [--len 24] [--gen 32] [--seed 0] [--ids FILE]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_parser.h"
#include "graph_ir.h"

using json = nlohmann::json;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: ir_device_dump MODEL.gguf OUT.bin [--n 40] [--len 24] [--gen 32] [--seed 0] [--ids FILE]\n";
    return 2;
  }
  int n = 40, len = 24, gen = 32;
  unsigned seed = 0;
  std::string ids_file;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--n") n = std::atoi(argv[i + 1]);
    else if (k == "--len") len = std::atoi(argv[i + 1]);
    else if (k == "--gen") gen = std::atoi(argv[i + 1]);
    else if (k == "--seed") seed = static_cast<unsigned>(std::atoi(argv[i + 1]));
    else if (k == "--ids") ids_file = argv[i + 1];
    else {
      std::cerr << "unknown option " << k << "\n";
      return 2;
    }
  }
  const GGUFData gguf = load_gguf_meta(argv[1], true);
  const std::string ir = ember_ir::find_ir(gguf);
  if (ir.empty() || !ember_ir::device_supported(ir)) {
    std::cerr << "no IR, or the device does not run it\n";
    return 1;
  }
  ember_ir::Interpreter dev(ir, gguf, true);
  if (!dev.on_gpu()) {
    std::cerr << "not on the GPU\n";
    return 1;
  }
  const int V = dev.vocab();
  std::vector<std::vector<int>> prompts;
  if (!ids_file.empty()) {
    std::ifstream f(ids_file);
    for (const auto& s : json::parse(f)) prompts.push_back(s.get<std::vector<int>>());
  } else {  // ir_device_gate's prompts: the same generator, range and order
    const int hi = std::min<int>(V, static_cast<int>(gguf.tokenizer_tokens.empty() ? V : gguf.tokenizer_tokens.size()));
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, hi - 1);
    for (int s = 0; s < n; ++s) {
      std::vector<int> ids(len);
      for (int& t : ids) t = pick(rng);
      prompts.push_back(ids);
    }
  }
  FILE* out = std::fopen(argv[2], "wb");
  if (!out) return 1;
  const bool last = !ids_file.empty();
  for (const auto& ids : prompts) {
    const std::vector<float> pre = dev.forward(ids, 0, !last);
    std::fwrite(pre.data(), 4, pre.size(), out);
    std::vector<float> d;
    for (size_t t = 0; t < ids.size(); ++t) {
      d = dev.forward({ids[t]}, static_cast<int>(t), false);
      if (!last) std::fwrite(d.data(), 4, d.size(), out);
    }
    if (last) std::fwrite(d.data(), 4, d.size(), out);
    std::vector<float> l = d;
    int pos = static_cast<int>(ids.size());
    std::vector<int> g;
    for (int k = 0; k < gen; ++k) {
      const int tok = static_cast<int>(std::max_element(l.end() - V, l.end()) - (l.end() - V));
      g.push_back(tok);
      if (k + 1 < gen) l = dev.forward({tok}, pos++, false);
    }
    std::fwrite(g.data(), 4, g.size(), out);
  }
  std::fclose(out);
  std::cout << json{{"gguf", argv[1]}, {"prompts", prompts.size()}, {"vocab", V}}.dump() << "\n";
  return 0;
}
