// ir_logits: ember's IR interpreter on raw token ids, for checking a graph against a reference
// (paperkiln's tools/ir_numpy.py, HF transformers) without a tokenizer or the model loader.
//
//   EMBER_IR=graph.json ir_logits MODEL.gguf < seqs.json > logits.json
//
//   stdin:  [[id, id, ...], ...]               token sequences
//   stdout: [{"prefill": [[logit, ...] per position], "decode": [[...] per position]}, ...]
//
// "prefill" runs each sequence as one forward from position 0 (all rows); "decode" runs it one
// token at a time through the KV / recurrent state, as generation does. Both start fresh. The IR
// comes from EMBER_IR, else the GGUF's paperkiln.ir. CPU only.
#include <cstdlib>
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
    std::cerr << "usage: EMBER_IR=graph.json ir_logits MODEL.gguf < seqs.json\n";
    return 2;
  }
  const GGUFData gguf = load_gguf_meta(argv[1], true);
  const std::string ir = ember_ir::find_ir(gguf);
  if (ir.empty()) {
    std::cerr << "no IR: set EMBER_IR or use a GGUF with paperkiln.ir\n";
    return 1;
  }
  const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  const json seqs = json::parse(in);
  const int V = ember_ir::Interpreter(ir, gguf, false).vocab();
  json out = json::array();
  for (const json& s : seqs) {
    const std::vector<int> ids = s.get<std::vector<int>>();
    json rec;
    {
      ember_ir::Interpreter m(ir, gguf, false);
      const std::vector<float> all = m.forward(ids, 0, true);
      json rows = json::array();
      for (size_t t = 0; t < ids.size(); ++t)
        rows.push_back(std::vector<float>(all.begin() + t * V, all.begin() + (t + 1) * V));
      rec["prefill"] = std::move(rows);
    }
    {
      ember_ir::Interpreter m(ir, gguf, false);
      json rows = json::array();
      for (size_t t = 0; t < ids.size(); ++t) rows.push_back(m.forward({ids[t]}, static_cast<int>(t), false));
      rec["decode"] = std::move(rows);
    }
    out.push_back(std::move(rec));
  }
  std::cout << out.dump() << "\n";
  return 0;
}
