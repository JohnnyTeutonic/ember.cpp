// e4b scratch tool: a GGUF on the session's default path (for a llama-labelled Mixtral file: the tuned llama path,
// not the IR interpreter), greedy over raw ids by fresh prefills (TinyLlamaSession::score_rows, the last row's
// logits), as HF's greedy reference is the argmax of its own forward.
//   e4b_llama_path MODEL.gguf CASES.json > rows.jsonl   (one row per case: first-step logits, greedy ids)
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include "api.h"

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  nlohmann::json cases;
  std::ifstream(argv[2]) >> cases;
  tinyllama::TinyLlamaSession s(argv[1], argv[1], 1, 0, true);
  const int V = s.score_vocab();
  const int max_new = cases.at("max_new").get<int>();
  for (const auto& c : cases.at("cases")) {
    std::vector<int> ids = c.at("ids").get<std::vector<int>>();
    std::vector<float> first;
    std::vector<int> out;
    for (int k = 0; k < max_new; ++k) {
      const std::vector<float> lg = s.score_rows(ids, static_cast<int>(ids.size()) - 1, false);
      if (k == 0) first = lg;
      int best = 0;
      for (int v = 1; v < V; ++v)
        if (lg[v] > lg[best]) best = v;
      out.push_back(best);
      ids.push_back(best);
    }
    std::cout << nlohmann::json{{"first", first}, {"greedy", out}}.dump() << "\n";
  }
  return 0;
}
