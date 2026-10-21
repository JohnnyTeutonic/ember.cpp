#pragma once
// The IR's `expr` op on the CPU: a runtime-defined elementwise operation (a
// paper's new activation or gate) written in paperkiln's closed grammar
// (paperkiln include/microtorch/ir.hpp, op `expr`; microtorch/expr.hpp).
//
// The parser is vendored from paperkiln src/expr.cpp (1159d63e) with its
// error messages unchanged, so both engines accept the same strings and reject
// the rest at the same column. The evaluator is ember's own forward over row-
// major [T, w] buffers with paperkiln's semantics: every primitive in f32 and
// rounded after it (this file's .cpp is compiled with -ffp-contract=off), a
// constant subexpression folded with the same operations, row reductions
// [T, 1] in ascending column order, broadcast back where they meet a full
// value. tools/ir_expr_eval.cpp exposes it with paperkiln's expr_eval JSON
// contract, so paperkiln's tests/test_ir_expr.py gates it (EXPR_EVAL_BIN).
#include <cstddef>
#include <string>
#include <vector>

namespace ember_ir {
namespace expr {

enum class Kind {
  NUMBER, INPUT, NEG, ADD, SUB, MUL, DIV, POW,
  EXP, LOG, SQRT, TANH, SIGMOID, ERF, ABS, RELU,  // one argument
  MAX, MIN,                                       // two
  WHERE,                                          // three
  ROWSUM, ROWMEAN, ROWMAX,                        // one, over the row
};

struct Node {
  Kind kind = Kind::NUMBER;
  float value = 0.0f;      // NUMBER
  size_t input = 0;        // INPUT: 0 for x, 1 for a, 2 for b, ...
  std::vector<Node> args;  // operands, in order
  size_t column = 0;       // where the node starts in the text (1-based)
};

class Expr {
 public:
  static constexpr size_t kMaxInputs = 23;  // x, then a .. v
  static constexpr size_t kMaxDepth = 200;  // nesting, so a hostile string cannot exhaust the stack

  // Throws std::invalid_argument naming the problem and its column, e.g.
  //   expr: expected an operand at column 5 in "x + * 2".
  static Expr parse(const std::string& text, size_t n_inputs = 1);

  const std::string& text() const { return text_; }
  size_t n_inputs() const { return n_inputs_; }
  const Node& root() const { return root_; }

 private:
  std::string text_;
  size_t n_inputs_ = 1;
  Node root_;
};

// out [T, w] = the expression over ins (n_inputs() buffers, each [T, w] row-major).
void eval(const Expr& e, const std::vector<const float*>& ins, int T, int w, float* out);

// The parse of (text, n_inputs), cached: an IR's expressions are parsed once at load.
const Expr& cached(const std::string& text, size_t n_inputs);

}  // namespace expr
}  // namespace ember_ir
