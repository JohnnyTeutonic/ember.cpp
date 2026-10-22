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
// comes from ember_ir::find_ir: EMBER_IR, else the GGUF's paperkiln.ir, else the IR built from its
// metadata. EMBER_IR_PRINT=path writes that IR to path. EMBER_IR_LAST=1 writes only each sequence's
// last row of each (long prompts: a full [T, vocab] dump would be gigabytes). CPU only.
#include <cstdlib>
#include <fstream>
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
    std::cerr << "no IR: set EMBER_IR, use a GGUF with paperkiln.ir, or EMBER_IR_BUILD=1 for llama/mistral\n";
    return 1;
  }
  if (const char* dump = std::getenv("EMBER_IR_PRINT")) std::ofstream(dump) << ir << "\n";
  const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  const json seqs = json::parse(in);
  const int V = ember_ir::Interpreter(ir, gguf, false).vocab();
  const char* last_env = std::getenv("EMBER_IR_LAST");
  const bool last = last_env && std::string(last_env) == "1";
  json out = json::array();
  for (const json& s : seqs) {
    const std::vector<int> ids = s.get<std::vector<int>>();
    json rec;
    {
      ember_ir::Interpreter m(ir, gguf, false);
      const std::vector<float> all = m.forward(ids, 0, !last);
      json rows = json::array();
      if (last)
        rows.push_back(all);
      else
        for (size_t t = 0; t < ids.size(); ++t)
          rows.push_back(std::vector<float>(all.begin() + t * V, all.begin() + (t + 1) * V));
      rec["prefill"] = std::move(rows);
    }
    {
      ember_ir::Interpreter m(ir, gguf, false);
      json rows = json::array();
      std::vector<float> d;
      for (size_t t = 0; t < ids.size(); ++t) {
        d = m.forward({ids[t]}, static_cast<int>(t), false);
        if (!last) rows.push_back(d);
      }
      if (last) rows.push_back(d);
      rec["decode"] = std::move(rows);
    }
    out.push_back(std::move(rec));
  }
  std::cout << out.dump() << "\n";
  return 0;
}
