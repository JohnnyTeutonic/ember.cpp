// Raw-token, fixed-work CPU forward benchmark. The controller freezes the
// GGUF, prompts and continuations and checks every output row before timing.
// Load is excluded. JSON, argmax and optional row writes are outside forward
// timers; no tokenizer, sampling, HTTP or speculative decoding is measured.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "gguf_parser.h"
#include "graph_ir.h"

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: ir_hf_cpu_bench MODEL.gguf");
    const GGUFData gguf = load_gguf_meta(argv[1], true);
    const std::string doc = ember_ir::find_ir(gguf);
    if (doc.empty()) throw std::runtime_error("no supported IR for model");
    ember_ir::Interpreter model(doc, gguf, false);
    if (model.on_gpu()) throw std::runtime_error("CPU benchmark must not use a GPU");
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    if (threads != 1) throw std::runtime_error("CPU benchmark requires exactly one OpenMP thread");
    std::cout << json({{"ready", true}, {"backend", "cpu"}, {"vocab", model.vocab()},
                       {"max_len", model.max_len()}, {"threads", threads}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
      const json request = json::parse(line);
      const auto prompt = request.at("prompt_ids").get<std::vector<int>>();
      const auto continuation = request.at("continuation_ids").get<std::vector<int>>();
      if (prompt.empty() || continuation.empty() || prompt.size() + continuation.size() >
          static_cast<size_t>(model.max_len())) throw std::runtime_error("invalid context length");
      for (const auto* ids : {&prompt, &continuation})
        for (int id : *ids)
          if (id < 0 || id >= model.vocab()) throw std::runtime_error("out-of-vocabulary token");
      std::ofstream dump;
      if (request.contains("rows")) {
        dump.open(request.at("rows").get<std::string>(), std::ios::binary | std::ios::trunc);
        if (!dump) throw std::runtime_error("cannot open row dump");
      }
      std::vector<int> observed;
      std::vector<double> times;
      for (size_t step = 0; step < continuation.size(); ++step) {
        const std::vector<int> ids = step ? std::vector<int>{continuation[step-1]} : prompt;
        const int position = step ? static_cast<int>(prompt.size() + step - 1) : 0;
        const auto begin = Clock::now();
        const auto row = model.forward(ids, position, false);
        const auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end-begin).count());
        if (row.size() != static_cast<size_t>(model.vocab()) ||
            !std::all_of(row.begin(), row.end(), [](float x) { return std::isfinite(x); }))
          throw std::runtime_error("invalid output logits");
        observed.push_back(static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin()));
        if (dump.is_open()) {
          dump.write(reinterpret_cast<const char*>(row.data()), row.size()*sizeof(float));
          if (!dump) throw std::runtime_error("row dump failed");
        }
      }
      std::cout << json({{"key", request.at("key")}, {"backend", "cpu"},
                         {"argmax", observed}, {"forward_ms", times}}).dump() << std::endl;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 2;
  }
}
