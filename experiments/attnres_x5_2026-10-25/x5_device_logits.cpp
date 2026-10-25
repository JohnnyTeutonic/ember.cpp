// x5_device_logits: tools/ir_logits on the CUDA device (one interpreter, as a server holds it), for the X5 gate's
// log-probs against mtstudio score. Only the public interpreter API, so it builds against older trees.
//
//   x5_device_logits MODEL.gguf < {"seqs": [[id, ...], ...], "prompt_len": [P, ...]} > out.json
//
// out: [{"prefill": [[logit, ...] per position], "decode": [[...] per position], "free": [ids]}, ...]
// "prefill" is one forward of the sequence from position 0 (all rows); "decode" runs it one token at a time
// (the captured CUDA graph after the first step); "free" is free-running greedy: the prompt's prefill, then
// len - P argmax tokens decoded.
#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_parser.h"
#include "graph_ir.h"

using json = nlohmann::json;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: x5_device_logits MODEL.gguf < input.json\n";
    return 2;
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
  const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  const json inp = json::parse(in);
  json out = json::array();
  for (size_t s = 0; s < inp["seqs"].size(); ++s) {
    const std::vector<int> ids = inp["seqs"][s].get<std::vector<int>>();
    const int P = inp["prompt_len"][s].get<int>();
    json rec;
    const std::vector<float> all = dev.forward(ids, 0, true);
    json rows = json::array();
    for (size_t t = 0; t < ids.size(); ++t) rows.push_back(std::vector<float>(all.begin() + t * V, all.begin() + (t + 1) * V));
    rec["prefill"] = std::move(rows);
    json drows = json::array();
    for (size_t t = 0; t < ids.size(); ++t) drows.push_back(dev.forward({ids[t]}, static_cast<int>(t), false));
    rec["decode"] = std::move(drows);
    std::vector<int> prompt(ids.begin(), ids.begin() + P);
    std::vector<float> l = dev.forward(prompt, 0, false);
    std::vector<int> g;
    int pos = P;
    for (size_t k = 0; k + P < ids.size(); ++k) {
      const int tok = static_cast<int>(std::max_element(l.end() - V, l.end()) - (l.end() - V));
      g.push_back(tok);
      if (k + 1 + P < ids.size()) l = dev.forward({tok}, pos++, false);
    }
    rec["free"] = g;
    out.push_back(std::move(rec));
  }
  std::cout << out.dump() << "\n";
  return 0;
}
