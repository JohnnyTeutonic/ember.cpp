// ir_expr_eval: ember's `expr` op (graph_ir_expr.h) with paperkiln's expr_eval contract, so
// paperkiln's tests/test_ir_expr.py gates ember against the numpy reference (tools/ir_expr.py):
//
//   EXPR_EVAL_BIN=/path/to/ir_expr_eval python -m pytest tests/test_ir_expr.py   (in paperkiln)
//
//   stdin:  [{"expr": "x*sigmoid(1.702*x)", "inputs": [[[row], ...], ...]},
//            {"expr": "max(a,", "n_inputs": 2}, ...]
//   stdout: [{"out": [[row], ...]}, {"error": "expr: ... at column 7 ..."}, ...]
//
// A case without "inputs" is only parsed (with "n_inputs" inputs, default 1). Values travel as
// JSON doubles, exact for floats both ways; a non-finite output is "nan", "inf" or "-inf".
//   g++ -O2 -std=c++17 -ffp-contract=off -I. -I<nlohmann/include> tools/ir_expr_eval.cpp graph_ir_expr.cpp -o ir_expr_eval
#include <cmath>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph_ir_expr.h"

using json = nlohmann::json;

namespace {

json number(float v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  return static_cast<double>(v);
}

json run(const json& c) {
  const std::string text = c.at("expr").get<std::string>();
  if (!c.contains("inputs")) {
    ember_ir::expr::Expr::parse(text, c.value("n_inputs", size_t(1)));
    return {{"ok", true}};
  }
  std::vector<std::vector<float>> bufs;
  int T = -1, w = -1;
  for (const json& m : c.at("inputs")) {
    const int R = static_cast<int>(m.size()), C = R ? static_cast<int>(m[0].size()) : 0;
    if (T >= 0 && (R != T || C != w)) throw std::invalid_argument("expr: every input must have one shape [T, w]");
    T = R, w = C;
    std::vector<float> x(static_cast<size_t>(R) * C);
    for (int i = 0; i < R; ++i)
      for (int j = 0; j < C; ++j) x[static_cast<size_t>(i) * C + j] = static_cast<float>(m[i][j].get<double>());
    bufs.push_back(std::move(x));
  }
  std::vector<const float*> ins;
  for (const auto& b : bufs) ins.push_back(b.data());
  const ember_ir::expr::Expr e = ember_ir::expr::Expr::parse(text, ins.size());
  std::vector<float> y(static_cast<size_t>(T) * w);
  ember_ir::expr::eval(e, ins, T, w, y.data());
  json rows = json::array();
  for (int i = 0; i < T; ++i) {
    json row = json::array();
    for (int j = 0; j < w; ++j) row.push_back(number(y[static_cast<size_t>(i) * w + j]));
    rows.push_back(std::move(row));
  }
  return {{"out", std::move(rows)}};
}

}  // namespace

int main() {
  const std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  const json cases = json::parse(in);
  json out = json::array();
  for (const json& c : cases) {
    try {
      out.push_back(run(c));
    } catch (const std::exception& e) {
      out.push_back({{"error", e.what()}});
    }
  }
  std::cout << out.dump() << "\n";
  return 0;
}
