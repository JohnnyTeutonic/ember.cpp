// ir_path_check: which path ember serves a GGUF on (graph_ir.h tuned_llama_match; IR_SPEC 9.1): the tuned llama
// path for the graph it implements, the IR interpreter for any other document, the llama path for a file without
// one.
//
//   ir_path_check MODEL.gguf ...      (EMBER_IR=doc replaces each file's document, as in the server)
//
// One JSON line per file: {"gguf", "family_hint", "path": "tuned" | "interpreter" | "llama (no IR)", "why"}.
#include <iostream>

#include <nlohmann/json.hpp>

#include "gguf_parser.h"
#include "graph_ir.h"
#include "logger.h"

int main(int argc, char** argv) {
  Logger::set_level(Logger::Level::OFF);
  int rc = 0;
  for (int i = 1; i < argc; ++i) {
    nlohmann::json out{{"gguf", argv[i]}};
    try {
      const GGUFData g = load_gguf_meta(argv[i], true);
      const std::string ir = ember_ir::find_ir(g);
      if (ir.empty()) {
        out["path"] = "llama (no IR)";
      } else {
        std::string why;
        out["family_hint"] = nlohmann::json::parse(ir).value("family_hint", nlohmann::json());
        out["path"] = ember_ir::tuned_llama_match(ir, g, &why) ? "tuned" : "interpreter";
        out["why"] = why;
      }
    } catch (const std::exception& e) {
      out["error"] = e.what();
      rc = 1;
    }
    std::cout << out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
  }
  return rc;
}
