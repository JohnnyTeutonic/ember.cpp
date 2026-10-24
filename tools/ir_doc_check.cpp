// ir_doc_check: ember's document validation (graph_ir_validate.h; paperkiln spec/IR_SPEC.md section 10.1, the
// document rules) on IR documents without a GGUF, for comparing ember with paperkiln's tools/ir_validate.py.
//
//   ir_doc_check DOC.ir.json ...
//
// One JSON line per document: {"path", "accepted", "errors": [...]}. Container rules (24-28) and tensor
// binding need the GGUF: ir_adapter's check runs the whole load. Exit status 1 when any is refused.
#include <fstream>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "graph_ir_validate.h"

int main(int argc, char** argv) {
  int bad = 0;
  for (int i = 1; i < argc; ++i) {
    nlohmann::json out{{"path", argv[i]}};
    std::vector<std::string> errs;
    try {
      std::ifstream f(argv[i], std::ios::binary);
      if (!f) throw std::runtime_error(std::string("cannot read ") + argv[i]);
      std::stringstream ss;
      ss << f.rdbuf();
      errs = ember_ir::ir_document_errors(nlohmann::json::parse(ss.str()));
    } catch (const std::exception& e) {
      errs = {e.what()};
    }
    out["accepted"] = errs.empty();
    out["errors"] = errs;
    bad += !errs.empty();
    std::cout << out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
  }
  return bad ? 1 : 0;
}
