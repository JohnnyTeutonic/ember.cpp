#include "graph_ir_expr.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace ember_ir {
namespace expr {

namespace {

// ---- the parser: vendored from paperkiln src/expr.cpp (1159d63e), messages unchanged ----

struct Fn {
  const char* name;
  Kind kind;
  size_t arity;
};
constexpr Fn kFns[] = {
    {"exp", Kind::EXP, 1},         {"log", Kind::LOG, 1},         {"sqrt", Kind::SQRT, 1},
    {"tanh", Kind::TANH, 1},       {"sigmoid", Kind::SIGMOID, 1}, {"erf", Kind::ERF, 1},
    {"abs", Kind::ABS, 1},         {"relu", Kind::RELU, 1},       {"max", Kind::MAX, 2},
    {"min", Kind::MIN, 2},         {"where", Kind::WHERE, 3},     {"rowsum", Kind::ROWSUM, 1},
    {"rowmean", Kind::ROWMEAN, 1}, {"rowmax", Kind::ROWMAX, 1},
};
constexpr float kPi = 3.14159265358979323846f;  // the nearest float to pi

enum class Tok { NUM, NAME, OP, LPAREN, RPAREN, COMMA, END };

struct Token {
  Tok tok;
  std::string text;
  float value = 0.0f;
  size_t column = 0;  // 1-based
};

class Parser {
 public:
  Parser(const std::string& text, size_t n_inputs) : s_(text), n_(n_inputs) { lex(); }

  Node parse() {
    if (toks_.front().tok == Tok::END) fail("empty expression", 1);
    Node root = expr(0);
    if (peek().tok != Tok::END) fail("expected an operator or the end, got '" + peek().text + "'", peek().column);
    return root;
  }

 private:
  [[noreturn]] void fail(const std::string& what, size_t column) const {
    throw std::invalid_argument("expr: " + what + " at column " + std::to_string(column) + " in \"" + s_ + "\"");
  }

  void lex() {
    size_t i = 0;
    while (i < s_.size()) {
      const char ch = s_[i];
      const size_t col = i + 1;
      if (ch == ' ' || ch == '\t') {
        ++i;
      } else if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '.') {
        // digits [. digits] | . digits, then an optional exponent.
        size_t j = i, mant = 0;
        while (j < s_.size() && std::isdigit(static_cast<unsigned char>(s_[j]))) ++j, ++mant;
        if (j < s_.size() && s_[j] == '.') {
          ++j;
          while (j < s_.size() && std::isdigit(static_cast<unsigned char>(s_[j]))) ++j, ++mant;
        }
        if (mant == 0) fail("malformed number", col);
        if (j < s_.size() && (s_[j] == 'e' || s_[j] == 'E')) {
          size_t k = j + 1;
          if (k < s_.size() && (s_[k] == '+' || s_[k] == '-')) ++k;
          const size_t d0 = k;
          while (k < s_.size() && std::isdigit(static_cast<unsigned char>(s_[k]))) ++k;
          if (k == d0) fail("malformed number (exponent without digits)", col);
          j = k;
        }
        const std::string lit = s_.substr(i, j - i);
        const float v = std::strtof(lit.c_str(), nullptr);
        if (!std::isfinite(v)) fail("number " + lit + " is out of float range", col);
        toks_.push_back({Tok::NUM, lit, v, col});
        i = j;
      } else if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_') {
        size_t j = i;
        while (j < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[j])) || s_[j] == '_')) ++j;
        toks_.push_back({Tok::NAME, s_.substr(i, j - i), 0.0f, col});
        i = j;
      } else if (ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^') {
        toks_.push_back({Tok::OP, std::string(1, ch), 0.0f, col});
        ++i;
      } else if (ch == '(') {
        toks_.push_back({Tok::LPAREN, "(", 0.0f, col});
        ++i;
      } else if (ch == ')') {
        toks_.push_back({Tok::RPAREN, ")", 0.0f, col});
        ++i;
      } else if (ch == ',') {
        toks_.push_back({Tok::COMMA, ",", 0.0f, col});
        ++i;
      } else {
        fail(std::string("unexpected character '") + ch + "'", col);
      }
    }
    toks_.push_back({Tok::END, "end", 0.0f, s_.size() + 1});
  }

  const Token& peek() const { return toks_[pos_]; }
  const Token& next() { return toks_[pos_++]; }
  bool at_op(char op) const { return peek().tok == Tok::OP && peek().text[0] == op; }

  void enter(size_t depth) const {
    if (depth > Expr::kMaxDepth) fail("nesting deeper than " + std::to_string(Expr::kMaxDepth), peek().column);
  }

  static Node binary(Kind k, Node a, Node b, size_t column) {
    Node n;
    n.kind = k;
    n.column = column;
    n.args.push_back(std::move(a));
    n.args.push_back(std::move(b));
    return n;
  }

  // expr := term (('+' | '-') term)*
  Node expr(size_t depth) {
    enter(depth);
    Node left = term(depth + 1);
    while (at_op('+') || at_op('-')) {
      const Token op = next();
      Node right = term(depth + 1);
      left = binary(op.text[0] == '+' ? Kind::ADD : Kind::SUB, std::move(left), std::move(right), op.column);
    }
    return left;
  }

  // term := unary (('*' | '/') unary)*
  Node term(size_t depth) {
    enter(depth);
    Node left = unary(depth + 1);
    while (at_op('*') || at_op('/')) {
      const Token op = next();
      Node right = unary(depth + 1);
      left = binary(op.text[0] == '*' ? Kind::MUL : Kind::DIV, std::move(left), std::move(right), op.column);
    }
    return left;
  }

  // unary := '-' unary | power
  Node unary(size_t depth) {
    enter(depth);
    if (at_op('-')) {
      const Token op = next();
      Node n;
      n.kind = Kind::NEG;
      n.column = op.column;
      n.args.push_back(unary(depth + 1));
      return n;
    }
    return power(depth + 1);
  }

  // power := primary ('^' unary)?   (right-associative through unary)
  Node power(size_t depth) {
    enter(depth);
    Node base = primary(depth + 1);
    if (at_op('^')) {
      const Token op = next();
      Node exponent = unary(depth + 1);
      return binary(Kind::POW, std::move(base), std::move(exponent), op.column);
    }
    return base;
  }

  Node primary(size_t depth) {
    enter(depth);
    const Token t = next();
    Node n;
    n.column = t.column;
    if (t.tok == Tok::NUM) {
      n.kind = Kind::NUMBER;
      n.value = t.value;
      return n;
    }
    if (t.tok == Tok::LPAREN) {
      Node inner = expr(depth + 1);
      if (peek().tok != Tok::RPAREN) fail("expected ')', got '" + peek().text + "'", peek().column);
      next();
      return inner;
    }
    if (t.tok != Tok::NAME) {
      fail(t.tok == Tok::END ? "expected an operand, got the end" : "expected an operand, got '" + t.text + "'",
           t.column);
    }
    for (const Fn& f : kFns) {
      if (t.text != f.name) continue;
      if (peek().tok != Tok::LPAREN) fail("'" + t.text + "' is a function: expected '('", peek().column);
      next();
      n.kind = f.kind;
      for (;;) {
        n.args.push_back(expr(depth + 1));
        if (peek().tok == Tok::COMMA) {
          next();
          continue;
        }
        if (peek().tok != Tok::RPAREN) fail("expected ',' or ')', got '" + peek().text + "'", peek().column);
        next();
        break;
      }
      if (n.args.size() != f.arity)
        fail("'" + t.text + "' takes " + std::to_string(f.arity) + " argument" + (f.arity == 1 ? "" : "s") +
                 ", got " + std::to_string(n.args.size()),
             t.column);
      return n;
    }
    if (peek().tok == Tok::LPAREN) fail("unknown function '" + t.text + "'", t.column);
    if (t.text == "pi") {
      n.kind = Kind::NUMBER;
      n.value = kPi;
      return n;
    }
    // x is the first input, a the second, b the third, ...
    size_t idx = n_;
    if (t.text == "x") {
      idx = 0;
    } else if (t.text.size() == 1 && t.text[0] >= 'a' && t.text[0] < static_cast<char>('a' + n_ - 1)) {
      idx = static_cast<size_t>(t.text[0] - 'a') + 1;
    }
    if (idx >= n_) {
      std::string names = "x";
      for (size_t k = 1; k < n_; ++k) names += std::string(", ") + static_cast<char>('a' + k - 1);
      fail("unknown name '" + t.text + "' (inputs: " + names + "; constant: pi)", t.column);
    }
    n.kind = Kind::INPUT;
    n.input = idx;
    return n;
  }

  const std::string& s_;
  size_t n_;
  std::vector<Token> toks_;
  size_t pos_ = 0;
};

// ---- the forward: ember's own, paperkiln's Builder step for step on plain buffers ----

// A subexpression's value: a constant (folded in f32), a column [T, 1] (a
// row reduction, broadcast only where it meets a full value) or a full
// [T, w] buffer. The shapes are an implementation detail: the semantics are
// those of [T, w] throughout.
struct Value {
  enum Shape { SCALAR = 0, COL = 1, FULL = 2 } shape = SCALAR;
  float c = 0.0f;
  std::vector<float> v;  // T (COL) or T * w (FULL) entries
};

// The primitives, each rounded to f32 (paperkiln's host formulas: ops::sigmoid
// is 1 / (1 + exp(-a)), ops::relu std::max(0, a), maximum a >= b ? a : b).
float unary_op(Kind k, float a) {
  switch (k) {
    case Kind::NEG: return -a;
    case Kind::EXP: return std::exp(a);
    case Kind::LOG: return std::log(a);
    case Kind::SQRT: return std::sqrt(a);
    case Kind::TANH: return std::tanh(a);
    case Kind::SIGMOID: return 1.0f / (1.0f + std::exp(-a));
    case Kind::ERF: return std::erf(a);
    case Kind::ABS: return std::fabs(a);
    case Kind::RELU: return std::max(0.0f, a);
    default: throw std::logic_error("expr: unary");
  }
}

float binary_op(Kind k, float a, float b) {
  switch (k) {
    case Kind::ADD: return a + b;
    case Kind::SUB: return a - b;
    case Kind::MUL: return a * b;
    case Kind::DIV: return a / b;
    case Kind::POW: return std::pow(a, b);
    case Kind::MAX: return a >= b ? a : b;
    case Kind::MIN: return a <= b ? a : b;
    default: throw std::logic_error("expr: binary");
  }
}

class Evaluator {
 public:
  Evaluator(const std::vector<const float*>& ins, int T, int w) : ins_(ins), T_(T), w_(w) {}

  void result(const Node& root, float* out) {
    Value r = to(eval(root), Value::FULL);
    std::copy(r.v.begin(), r.v.end(), out);
  }

 private:
  size_t size(Value::Shape s) const {
    return s == Value::SCALAR ? 1 : static_cast<size_t>(T_) * (s == Value::COL ? 1 : w_);
  }

  // r brought to `shape` (never narrower than its own).
  Value to(Value r, Value::Shape shape) const {
    if (r.shape == shape) return r;
    Value out;
    out.shape = shape;
    out.v.resize(size(shape));
    if (r.shape == Value::SCALAR) {
      std::fill(out.v.begin(), out.v.end(), r.c);
    } else {  // COL -> FULL
      for (int t = 0; t < T_; ++t) std::fill_n(out.v.begin() + static_cast<size_t>(t) * w_, w_, r.v[t]);
    }
    return out;
  }

  static Value constant(float c) {
    Value out;
    out.c = c;
    return out;
  }

  Value eval(const Node& n) {
    switch (n.kind) {
      case Kind::NUMBER: return constant(n.value);
      case Kind::INPUT: {
        Value out;
        out.shape = Value::FULL;
        out.v.assign(ins_[n.input], ins_[n.input] + size(Value::FULL));
        return out;
      }
      case Kind::NEG: case Kind::EXP: case Kind::LOG: case Kind::SQRT: case Kind::TANH: case Kind::SIGMOID:
      case Kind::ERF: case Kind::ABS: case Kind::RELU: {
        Value a = eval(n.args[0]);
        if (a.shape == Value::SCALAR) return constant(unary_op(n.kind, a.c));
        for (float& v : a.v) v = unary_op(n.kind, v);
        return a;
      }
      case Kind::ADD: case Kind::SUB: case Kind::MUL: case Kind::DIV: case Kind::POW: case Kind::MAX:
      case Kind::MIN:
        return binary(n.kind, eval(n.args[0]), eval(n.args[1]));
      case Kind::WHERE: {
        Value c = eval(n.args[0]), a = eval(n.args[1]), b = eval(n.args[2]);
        if (c.shape == Value::SCALAR && a.shape == Value::SCALAR && b.shape == Value::SCALAR)
          return constant(c.c > 0.0f ? a.c : b.c);
        const auto s = std::max({c.shape, a.shape, b.shape});
        c = to(std::move(c), s), a = to(std::move(a), s), b = to(std::move(b), s);
        for (size_t i = 0; i < a.v.size(); ++i) a.v[i] = c.v[i] > 0.0f ? a.v[i] : b.v[i];
        return a;
      }
      case Kind::ROWSUM: case Kind::ROWMEAN: case Kind::ROWMAX: {
        const Value full = to(eval(n.args[0]), Value::FULL);
        Value col;
        col.shape = Value::COL;
        col.v.resize(static_cast<size_t>(T_));
        for (int t = 0; t < T_; ++t) {
          const float* row = full.v.data() + static_cast<size_t>(t) * w_;
          float acc;
          if (n.kind == Kind::ROWMAX) {  // the first largest entry, scanning ascending columns
            acc = row[0];
            for (int j = 1; j < w_; ++j) acc = row[j] > acc ? row[j] : acc;
          } else {  // 0 + row[0] + row[1] + ... in ascending columns, as the reference
            acc = 0.0f;
            for (int j = 0; j < w_; ++j) acc = acc + row[j];
          }
          col.v[t] = acc;
        }
        if (n.kind == Kind::ROWMEAN) return binary(Kind::DIV, std::move(col), constant(static_cast<float>(w_)));
        return col;
      }
    }
    throw std::logic_error("expr: unhandled node");
  }

  Value binary(Kind k, Value a, Value b) {
    if (a.shape == Value::SCALAR && b.shape == Value::SCALAR) return constant(binary_op(k, a.c, b.c));
    const auto s = std::max(a.shape, b.shape);
    // paperkiln forms x - c as x + (-c): the same IEEE result per element.
    if (k == Kind::SUB && b.shape == Value::SCALAR) {
      k = Kind::ADD;
      b.c = -b.c;
    }
    if (a.shape == Value::SCALAR || b.shape == Value::SCALAR) {
      const bool left = a.shape == Value::SCALAR;
      Value& t = left ? b : a;
      const float c = left ? a.c : b.c;
      for (float& v : t.v) v = left ? binary_op(k, c, v) : binary_op(k, v, c);
      return std::move(t);
    }
    a = to(std::move(a), s), b = to(std::move(b), s);
    for (size_t i = 0; i < a.v.size(); ++i) a.v[i] = binary_op(k, a.v[i], b.v[i]);
    return a;
  }

  const std::vector<const float*>& ins_;
  int T_, w_;
};

// ---- the device form (lower) ----

Op op_of(Kind k) {
  switch (k) {
    case Kind::NEG: return Op::NEG;
    case Kind::EXP: return Op::EXP;
    case Kind::LOG: return Op::LOG;
    case Kind::SQRT: return Op::SQRT;
    case Kind::TANH: return Op::TANH;
    case Kind::SIGMOID: return Op::SIGMOID;
    case Kind::ERF: return Op::ERF;
    case Kind::ABS: return Op::ABS;
    case Kind::RELU: return Op::RELU;
    case Kind::ADD: return Op::ADD;
    case Kind::SUB: return Op::SUB;
    case Kind::MUL: return Op::MUL;
    case Kind::DIV: return Op::DIV;
    case Kind::POW: return Op::POW;
    case Kind::MAX: return Op::MAX;
    case Kind::MIN: return Op::MIN;
    case Kind::WHERE: return Op::WHERE;
    default: throw std::logic_error("expr: no instruction");
  }
}

class Lowering {
 public:
  explicit Lowering(const Expr& e) : e_(e) {}

  std::vector<Stage> run() {
    Stage last;
    emit_stage(e_.root(), last);
    stages_.push_back(std::move(last));
    return std::move(stages_);
  }

 private:
  // n's value when it is a constant in eval (every leaf a number, no row
  // reduction), folded with eval's operations in eval's order.
  static bool constant(const Node& n, float& v) {
    switch (n.kind) {
      case Kind::NUMBER: v = n.value; return true;
      case Kind::INPUT: case Kind::ROWSUM: case Kind::ROWMEAN: case Kind::ROWMAX: return false;
      case Kind::WHERE: {
        float c, a, b;
        if (!constant(n.args[0], c) || !constant(n.args[1], a) || !constant(n.args[2], b)) return false;
        v = c > 0.0f ? a : b;
        return true;
      }
      default: break;
    }
    float a, b;
    if (!constant(n.args[0], a)) return false;
    if (n.args.size() == 1) {
      v = unary_op(n.kind, a);
      return true;
    }
    if (!constant(n.args[1], b)) return false;
    v = binary_op(n.kind, a, b);
    return true;
  }

  void emit_stage(const Node& n, Stage& s) {
    size_t depth = 0, peak = 0;
    emit(n, s.code, depth, peak);
    if (s.code.size() > kMaxCode || peak > kMaxStack)
      throw std::invalid_argument("expr: \"" + e_.text() + "\" needs " + std::to_string(s.code.size()) +
                                  " instructions and a stack of " + std::to_string(peak) + " in one stage (the CUDA form "
                                  "takes " + std::to_string(kMaxCode) + " and " + std::to_string(kMaxStack) + ")");
  }

  void emit(const Node& n, std::vector<Instr>& code, size_t& depth, size_t& peak) {
    auto push = [&](Op op, int idx = 0, float c = 0.0f) {
      code.push_back({op, idx, c});
      peak = std::max(peak, ++depth);
    };
    float v;
    if (constant(n, v)) return push(Op::CONST, 0, v);
    switch (n.kind) {
      case Kind::INPUT: return push(Op::INPUT, static_cast<int>(n.input));
      case Kind::ROWSUM: case Kind::ROWMEAN: case Kind::ROWMAX: {
        Stage s;
        s.reduce = n.kind == Kind::ROWMAX ? 2 : 1;
        emit_stage(n.args[0], s);  // its own reductions become earlier stages
        stages_.push_back(std::move(s));
        if (stages_.size() > kMaxColumns)
          throw std::invalid_argument("expr: \"" + e_.text() + "\" has more than " + std::to_string(kMaxColumns) +
                                      " row reductions (the CUDA form's limit)");
        push(Op::COL, static_cast<int>(stages_.size() - 1));
        if (n.kind == Kind::ROWMEAN) {  // eval: the column divided by (float) w
          push(Op::WIDTH);
          code.push_back({Op::DIV, 0, 0.0f});
          --depth;
        }
        return;
      }
      default: break;
    }
    for (const Node& a : n.args) emit(a, code, depth, peak);
    code.push_back({op_of(n.kind), 0, 0.0f});
    depth -= n.args.size() - 1;
  }

  const Expr& e_;
  std::vector<Stage> stages_;
};

}  // namespace

std::vector<Stage> lower(const Expr& e) { return Lowering(e).run(); }

Expr Expr::parse(const std::string& text, size_t n_inputs) {
  if (n_inputs == 0 || n_inputs > kMaxInputs)
    throw std::invalid_argument("expr: " + std::to_string(n_inputs) + " inputs (1 to " + std::to_string(kMaxInputs) +
                                ")");
  Expr e;
  e.text_ = text;
  e.n_inputs_ = n_inputs;
  e.root_ = Parser(e.text_, n_inputs).parse();
  return e;
}

void eval(const Expr& e, const std::vector<const float*>& ins, int T, int w, float* out) {
  if (ins.size() != e.n_inputs())
    throw std::invalid_argument("expr: " + std::to_string(ins.size()) + " inputs given, \"" + e.text() +
                                "\" was parsed for " + std::to_string(e.n_inputs()));
  if (T <= 0 || w <= 0) throw std::invalid_argument("expr: inputs must be [T, w] with T, w > 0");
  Evaluator(ins, T, w).result(e.root(), out);
}

const Expr& cached(const std::string& text, size_t n_inputs) {
  static std::mutex mu;
  static std::map<std::pair<std::string, size_t>, Expr> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto key = std::make_pair(text, n_inputs);
  auto it = cache.find(key);
  if (it == cache.end()) it = cache.emplace(key, Expr::parse(text, n_inputs)).first;
  return it->second;
}

}  // namespace expr
}  // namespace ember_ir
