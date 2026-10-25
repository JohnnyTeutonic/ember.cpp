#include "graph_ir_recurrence.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>

using nlohmann::json;

namespace ember_ir {
namespace rec {
namespace {

[[noreturn]] void fail(int rule, const std::string& msg) {
  throw std::runtime_error("IR_SPEC 6.10 (rule " + std::to_string(rule) + "): " + msg);
}

const std::set<std::string> kFunctions = {"exp", "log", "sqrt", "tanh", "erf", "abs", "sigmoid", "relu",
                                          "max", "min", "where"};
const std::set<std::string> kRowReductions = {"rowsum", "rowmean", "rowmax"};

bool is_identifier(const std::string& s) {
  if (s.empty() || !(std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_')) return false;
  for (char c : s)
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
  return true;
}

bool is_reserved(const std::string& s) {
  return kFunctions.count(s) || kRowReductions.count(s) || s == "pi" || s == "sum" || s == "per_layer";
}

// ---------------------------------------------------------------- tokens

enum class Tok { Ident, Number, LParen, RParen, LBracket, RBracket, Comma, Plus, Minus, Star, Slash, Caret, Eq, End };

struct Token {
  Tok kind = Tok::End;
  std::string text;
  size_t column = 0;  // 1-based
  bool integer = false;  // a Number of digits only
};

std::vector<Token> lex(const std::string& s, const std::string& where) {
  std::vector<Token> out;
  size_t i = 0;
  while (i < s.size()) {
    const char c = s[i];
    if (c == ' ' || c == '\t') {
      ++i;
      continue;
    }
    Token t;
    t.column = i + 1;
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t j = i;
      while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
      t.kind = Tok::Ident;
      t.text = s.substr(i, j - i);
      i = j;
    } else if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
      size_t j = i, mant = 0;
      while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j, ++mant;
      bool integer = true;
      if (j < s.size() && s[j] == '.') {
        integer = false;
        ++j;
        while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j, ++mant;
      }
      if (mant == 0) fail(31, where + " does not parse: a malformed number at column " + std::to_string(t.column));
      if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
        integer = false;
        size_t k = j + 1;
        if (k < s.size() && (s[k] == '+' || s[k] == '-')) ++k;
        const size_t d0 = k;
        while (k < s.size() && std::isdigit(static_cast<unsigned char>(s[k]))) ++k;
        if (k == d0)
          fail(31, where + " does not parse: a malformed number (an exponent without digits) at column " +
                       std::to_string(t.column));
        j = k;
      }
      t.kind = Tok::Number;
      t.text = s.substr(i, j - i);
      t.integer = integer;
      i = j;
    } else {
      static const std::map<char, Tok> kPunct = {{'(', Tok::LParen}, {')', Tok::RParen}, {'[', Tok::LBracket},
                                                 {']', Tok::RBracket}, {',', Tok::Comma}, {'+', Tok::Plus},
                                                 {'-', Tok::Minus}, {'*', Tok::Star}, {'/', Tok::Slash},
                                                 {'^', Tok::Caret}, {'=', Tok::Eq}};
      const auto it = kPunct.find(c);
      if (it == kPunct.end())
        fail(31, where + " does not parse: unexpected character '" + std::string(1, c) + "' at column " +
                     std::to_string(t.column));
      t.kind = it->second;
      t.text = std::string(1, c);
      ++i;
    }
    out.push_back(std::move(t));
  }
  Token end;
  end.kind = Tok::End;
  end.column = s.size() + 1;
  out.push_back(end);
  return out;
}

// ---------------------------------------------------------------- the compiled form

enum class K {
  Num, Idx, Read, Neg, Add, Sub, Mul, Div, Pow,
  Exp, Log, Sqrt, Tanh, Erf, Abs, Sigmoid, Relu, Max, Min, Where, Sum
};

struct Ix {
  int index = -1;  // -1: the constant off
  long off = 0;
};

struct E {
  K k = K::Num;
  float v = 0.0f;
  int id = -1;  // Idx, Sum: the index; Read: the tensor
  std::vector<Ix> ix;
  std::vector<E> a;
};

enum class TKind { Input, Weight, State, Let, Out };

struct Tensor {
  std::string name;
  TKind kind = TKind::Input;
  std::vector<int> sizes;
  std::vector<size_t> strides;
  size_t numel = 1;
  int pos = 0;  // within its kind's list
};

struct Assign {
  int tensor = -1;
  std::vector<int> lhs;  // index ids
  E body;
  std::string where;
  size_t body_tok = 0;  // the first token of the body (compile time)
  std::vector<Token> toks;
};

}  // namespace

struct Program {
  std::vector<std::string> index_names;
  std::vector<int> index_sizes;
  std::map<std::string, int> index_of;
  std::vector<Tensor> tensors;
  std::map<std::string, int> tensor_of;
  std::vector<int> inputs, weights, states, lets;
  int out = -1;
  std::vector<Assign> init, let, update;
  Assign readout;
  std::vector<int> init_of_state;  // per state: index into init, or -1
  std::vector<Recurrence::StateTensor> state_info;
  std::vector<Recurrence::WeightDecl> weight_info;
  std::vector<size_t> in_widths;
  size_t out_width = 0;
  bool exact = true;
  // evaluation scratch
  mutable std::vector<int> iv;
  mutable std::vector<const float*> data;
  mutable std::vector<std::vector<float>> let_buf, new_state;

  float eval(const E& e) const;
  void assign(const Assign& a, float* out) const;
};

namespace {

// Which tensors a body may read (section 6.10.2), by section.
struct ReadRules {
  bool inputs = false, weights = false, state = false;
  int lets = 0;  // lets 0 .. lets - 1
};

class BodyParser {
 public:
  BodyParser(Program& p, const std::vector<Token>& toks, size_t pos, const std::string& where,
             const ReadRules& rules, const std::vector<int>& scope)
      : p_(p), t_(toks), i_(pos), where_(where), rules_(rules), in_scope_(p.index_names.size(), false) {
    for (int s : scope) in_scope_[s] = true;
  }

  E parse() {
    E e = expr(0);
    if (peek().kind != Tok::End) syntax("unexpected '" + peek().text + "'");
    return e;
  }

 private:
  const Token& peek() const { return t_[i_]; }
  const Token& next() { return t_[i_++]; }
  [[noreturn]] void syntax(const std::string& what) const {
    fail(31, where_ + " does not parse: " + what + " at column " + std::to_string(peek().column));
  }
  void expect(Tok k, const char* what) {
    if (peek().kind != k) syntax(std::string("expected ") + what);
    ++i_;
  }
  void enter(int depth) const {
    if (depth > 256) syntax("nesting deeper than the limit");
  }

  E expr(int depth) {
    enter(depth);
    E left = term(depth + 1);
    while (peek().kind == Tok::Plus || peek().kind == Tok::Minus) {
      const K k = next().kind == Tok::Plus ? K::Add : K::Sub;
      E right = term(depth + 1);
      E n;
      n.k = k;
      n.a.push_back(std::move(left));
      n.a.push_back(std::move(right));
      left = std::move(n);
    }
    return left;
  }
  E term(int depth) {
    enter(depth);
    E left = unary(depth + 1);
    while (peek().kind == Tok::Star || peek().kind == Tok::Slash) {
      const K k = next().kind == Tok::Star ? K::Mul : K::Div;
      E right = unary(depth + 1);
      E n;
      n.k = k;
      n.a.push_back(std::move(left));
      n.a.push_back(std::move(right));
      left = std::move(n);
    }
    return left;
  }
  E unary(int depth) {
    enter(depth);
    if (peek().kind == Tok::Minus) {
      ++i_;
      E n;
      n.k = K::Neg;
      n.a.push_back(unary(depth + 1));
      return n;
    }
    return power(depth + 1);
  }
  E power(int depth) {
    enter(depth);
    E base = primary(depth + 1);
    if (peek().kind == Tok::Caret) {
      ++i_;
      p_.exact = false;
      E n;
      n.k = K::Pow;
      n.a.push_back(std::move(base));
      n.a.push_back(unary(depth + 1));
      return n;
    }
    return base;
  }

  int declared_index(const Token& t) const {
    const auto it = p_.index_of.find(t.text);
    if (it == p_.index_of.end())
      fail(32, where_ + ": index '" + t.text + "' at column " + std::to_string(t.column) + " is not declared");
    return it->second;
  }

  Ix ix() {
    Ix r;
    const Token& t = peek();
    if (t.kind == Tok::Number) {
      if (!t.integer) syntax("an index offset or position must be an integer");
      ++i_;
      r.off = std::strtol(t.text.c_str(), nullptr, 10);
      return r;
    }
    if (t.kind != Tok::Ident) syntax("expected an index");
    ++i_;
    r.index = declared_index(t);
    if (!in_scope_[r.index])
      fail(32, where_ + ": index '" + t.text + "' at column " + std::to_string(t.column) + " is not in scope");
    if (peek().kind == Tok::Plus || peek().kind == Tok::Minus) {
      const bool minus = next().kind == Tok::Minus;
      const Token& n = peek();
      if (n.kind != Tok::Number || !n.integer) syntax("expected an integer offset");
      ++i_;
      const long v = std::strtol(n.text.c_str(), nullptr, 10);
      r.off = minus ? -v : v;
    }
    return r;
  }

  E primary(int depth) {
    enter(depth);
    const Token& t = peek();
    if (t.kind == Tok::Number) {
      ++i_;
      E n;
      n.k = K::Num;
      n.v = std::strtof(t.text.c_str(), nullptr);
      if (!std::isfinite(n.v)) syntax("the number " + t.text + " is not finite in f32");
      return n;
    }
    if (t.kind == Tok::LParen) {
      ++i_;
      E e = expr(depth + 1);
      expect(Tok::RParen, "')'");
      return e;
    }
    if (t.kind != Tok::Ident) syntax("expected an operand");
    const std::string name = t.text;
    const size_t col = t.column;
    ++i_;
    if (name == "pi") {
      E n;
      n.k = K::Num;
      n.v = static_cast<float>(3.14159265358979323846);
      return n;
    }
    if (kRowReductions.count(name))
      fail(31, where_ + " uses " + name + " at column " + std::to_string(col) +
                   " (the row reductions are not part of a recurrence body; sum replaces them)");
    if (name == "sum") {
      expect(Tok::LParen, "'(' after sum");
      const Token& it = peek();
      if (it.kind != Tok::Ident) syntax("expected sum's index");
      ++i_;
      const int id = declared_index(it);
      if (in_scope_[id])
        fail(32, where_ + ": sum over index '" + it.text + "' at column " + std::to_string(it.column) +
                     ", which is already in scope");
      expect(Tok::Comma, "',' after sum's index");
      in_scope_[id] = true;
      E arg = expr(depth + 1);
      in_scope_[id] = false;
      expect(Tok::RParen, "')'");
      E n;
      n.k = K::Sum;
      n.id = id;
      n.a.push_back(std::move(arg));
      return n;
    }
    if (peek().kind == Tok::LParen) {
      static const std::map<std::string, std::pair<K, size_t>> kFn = {
          {"exp", {K::Exp, 1}},   {"log", {K::Log, 1}},         {"sqrt", {K::Sqrt, 1}}, {"tanh", {K::Tanh, 1}},
          {"erf", {K::Erf, 1}},   {"abs", {K::Abs, 1}},         {"sigmoid", {K::Sigmoid, 1}},
          {"relu", {K::Relu, 1}}, {"max", {K::Max, 2}},         {"min", {K::Min, 2}},   {"where", {K::Where, 3}}};
      const auto f = kFn.find(name);
      if (f == kFn.end()) fail(31, where_ + ": unknown function '" + name + "' at column " + std::to_string(col));
      ++i_;
      E n;
      n.k = f->second.first;
      if (n.k == K::Exp || n.k == K::Log || n.k == K::Tanh || n.k == K::Erf || n.k == K::Sigmoid) p_.exact = false;
      n.a.push_back(expr(depth + 1));
      while (peek().kind == Tok::Comma) {
        ++i_;
        n.a.push_back(expr(depth + 1));
      }
      expect(Tok::RParen, "')'");
      if (n.a.size() != f->second.second)
        fail(31, where_ + ": " + name + " at column " + std::to_string(col) + " takes " +
                     std::to_string(f->second.second) + " argument(s), not " + std::to_string(n.a.size()));
      return n;
    }
    if (kFunctions.count(name)) syntax("expected '(' after " + name);
    if (peek().kind == Tok::LBracket) {
      ++i_;
      const auto it = p_.tensor_of.find(name);
      if (it == p_.tensor_of.end())
        fail(32, where_ + " reads '" + name + "' at column " + std::to_string(col) + ", which is not declared");
      const Tensor& tn = p_.tensors[it->second];
      const bool ok = (tn.kind == TKind::Input && rules_.inputs) || (tn.kind == TKind::Weight && rules_.weights) ||
                      (tn.kind == TKind::State && rules_.state) || (tn.kind == TKind::Let && tn.pos < rules_.lets);
      if (!ok)
        fail(32, where_ + " reads '" + name + "' at column " + std::to_string(col) + ", which its section cannot read");
      E n;
      n.k = K::Read;
      n.id = it->second;
      n.ix.push_back(ix());
      while (peek().kind == Tok::Comma) {
        ++i_;
        n.ix.push_back(ix());
      }
      expect(Tok::RBracket, "']'");
      if (n.ix.size() != tn.sizes.size())
        fail(32, where_ + " reads '" + name + "' at column " + std::to_string(col) + " with " +
                     std::to_string(n.ix.size()) + " index(es); it has " + std::to_string(tn.sizes.size()) +
                     " dimension(s)");
      return n;
    }
    // A bare name: an index's value.
    const auto ii = p_.index_of.find(name);
    if (ii != p_.index_of.end()) {
      if (!in_scope_[ii->second])
        fail(32, where_ + ": index '" + name + "' at column " + std::to_string(col) + " is not in scope");
      E n;
      n.k = K::Idx;
      n.id = ii->second;
      return n;
    }
    if (p_.tensor_of.count(name))
      fail(32, where_ + " reads '" + name + "' at column " + std::to_string(col) +
                   " without indices; it has " + std::to_string(p_.tensors[p_.tensor_of.at(name)].sizes.size()) +
                   " dimension(s)");
    fail(32, where_ + ": '" + name + "' at column " + std::to_string(col) + " is not declared");
  }

  Program& p_;
  const std::vector<Token>& t_;
  size_t i_;
  std::string where_;
  ReadRules rules_;
  std::vector<bool> in_scope_;
};

// name '[' index (',' index)* ']' at toks[pos]; returns the name and the index ids; pos moves past it.
std::pair<std::string, std::vector<int>> head(const Program& p, const std::vector<Token>& toks, size_t& pos,
                                              const std::string& where, bool distinct) {
  auto syntax = [&](const std::string& what) {
    fail(31, where + " does not parse: " + what + " at column " + std::to_string(toks[pos].column));
  };
  if (toks[pos].kind != Tok::Ident) syntax("expected a name");
  const std::string name = toks[pos++].text;
  if (toks[pos].kind != Tok::LBracket) syntax("expected '[' after " + name);
  ++pos;
  std::vector<int> ids;
  for (;;) {
    if (toks[pos].kind != Tok::Ident) syntax("expected an index name");
    const Token& t = toks[pos++];
    const auto it = p.index_of.find(t.text);
    if (it == p.index_of.end())
      fail(32, where + ": index '" + t.text + "' at column " + std::to_string(t.column) + " is not declared");
    if (distinct && std::find(ids.begin(), ids.end(), it->second) != ids.end())
      fail(32, where + ": the assignment repeats index '" + t.text + "' at column " + std::to_string(t.column));
    ids.push_back(it->second);
    if (toks[pos].kind == Tok::Comma) {
      ++pos;
      continue;
    }
    if (toks[pos].kind != Tok::RBracket) syntax("expected ',' or ']'");
    ++pos;
    break;
  }
  return {name, ids};
}

std::vector<std::string> strings(const json& node, const char* key, bool required, bool nonempty) {
  std::vector<std::string> out;
  if (!node.contains(key) || node.at(key).is_null()) {
    if (required) fail(31, std::string("'") + key + "' is missing");
    return out;
  }
  const json& v = node.at(key);
  if (!v.is_array()) fail(31, std::string("'") + key + "' is not a list of strings");
  for (const auto& x : v) {
    if (!x.is_string()) fail(31, std::string("'") + key + "' holds a value that is not a string");
    out.push_back(x.get<std::string>());
  }
  if (nonempty && out.empty()) fail(31, std::string("'") + key + "' is empty");
  return out;
}

}  // namespace

float Program::eval(const E& e) const {
  switch (e.k) {
    case K::Num: return e.v;
    case K::Idx: return static_cast<float>(iv[e.id]);
    case K::Read: {
      const Tensor& t = tensors[e.id];
      size_t flat = 0;
      for (size_t d = 0; d < e.ix.size(); ++d) {
        const long c = (e.ix[d].index >= 0 ? iv[e.ix[d].index] : 0) + e.ix[d].off;
        if (c < 0 || c >= t.sizes[d]) return 0.0f;  // outside the shape: +0
        flat += static_cast<size_t>(c) * t.strides[d];
      }
      return data[e.id][flat];
    }
    case K::Neg: return -eval(e.a[0]);
    case K::Add: { const float x = eval(e.a[0]); const float y = eval(e.a[1]); return x + y; }
    case K::Sub: { const float x = eval(e.a[0]); const float y = eval(e.a[1]); return x - y; }
    case K::Mul: { const float x = eval(e.a[0]); const float y = eval(e.a[1]); return x * y; }
    case K::Div: { const float x = eval(e.a[0]); const float y = eval(e.a[1]); return x / y; }
    case K::Pow: { const float x = eval(e.a[0]); const float y = eval(e.a[1]); return std::pow(x, y); }
    case K::Exp: return std::exp(eval(e.a[0]));
    case K::Log: return std::log(eval(e.a[0]));
    case K::Sqrt: return std::sqrt(eval(e.a[0]));
    case K::Tanh: return std::tanh(eval(e.a[0]));
    case K::Erf: return std::erf(eval(e.a[0]));
    case K::Abs: return std::fabs(eval(e.a[0]));
    case K::Sigmoid: {
      const float x = eval(e.a[0]);
      const float ex = std::exp(-x);
      const float den = 1.0f + ex;
      return 1.0f / den;
    }
    case K::Relu: { const float x = eval(e.a[0]); return x > 0.0f ? x : 0.0f; }
    case K::Max: { const float x = eval(e.a[0]); const float y = eval(e.a[1]); return x >= y ? x : y; }
    case K::Min: { const float x = eval(e.a[0]); const float y = eval(e.a[1]); return x <= y ? x : y; }
    case K::Where: return eval(e.a[0]) > 0.0f ? eval(e.a[1]) : eval(e.a[2]);
    case K::Sum: {
      float acc = 0.0f;  // +0
      const int n = index_sizes[e.id];
      for (int j = 0; j < n; ++j) {
        iv[e.id] = j;
        const float x = eval(e.a[0]);
        acc = acc + x;
      }
      return acc;
    }
  }
  return 0.0f;
}

// The body at every point of the assignment's index space, row-major (the last index fastest).
void Program::assign(const Assign& a, float* out) const {
  const size_t k = a.lhs.size();
  for (size_t d = 0; d < k; ++d) iv[a.lhs[d]] = 0;
  const size_t n = tensors[a.tensor].numel;
  for (size_t flat = 0; flat < n; ++flat) {
    out[flat] = eval(a.body);
    for (size_t d = k; d-- > 0;) {
      if (++iv[a.lhs[d]] < index_sizes[a.lhs[d]]) break;
      iv[a.lhs[d]] = 0;
    }
  }
}

Recurrence::Recurrence() = default;
Recurrence::~Recurrence() = default;
Recurrence::Recurrence(Recurrence&&) noexcept = default;
Recurrence& Recurrence::operator=(Recurrence&&) noexcept = default;

const std::vector<Recurrence::StateTensor>& Recurrence::state() const { return p_->state_info; }
const std::vector<Recurrence::WeightDecl>& Recurrence::weights() const { return p_->weight_info; }
const std::vector<size_t>& Recurrence::input_widths() const { return p_->in_widths; }
size_t Recurrence::out_width() const { return p_->out_width; }
bool Recurrence::exact() const { return p_->exact; }

Recurrence Recurrence::compile(const json& node) {
  auto P = std::make_unique<Program>();
  Program& p = *P;
  std::set<std::string> names;  // every declared name, distinct across all kinds
  auto declare = [&](const std::string& name, const std::string& what) {
    if (!is_identifier(name)) fail(31, what + " name '" + name + "' is not an identifier");
    if (is_reserved(name)) fail(31, what + " name '" + name + "' is reserved");
    if (!names.insert(name).second) fail(31, "the name '" + name + "' is declared twice");
  };
  // index
  if (!node.contains("index") || !node.at("index").is_object() || node.at("index").empty())
    fail(31, "'index' is not a map of names to positive integers");
  for (const auto& kv : node.at("index").items()) {
    if (!kv.value().is_number_integer() || kv.value().get<int64_t>() <= 0)
      fail(31, "index '" + kv.key() + "' has size " + kv.value().dump() + ", not a positive integer");
    if (kv.value().get<int64_t>() > (1 << 24))  // a capability limit (section 10.3), not a validation rule
      throw std::runtime_error("IR recurrence: index '" + kv.key() + "' of size " + kv.value().dump() +
                               " is larger than ember's interpreter takes (2^24)");
    declare(kv.key(), "index");
    p.index_of[kv.key()] = static_cast<int>(p.index_names.size());
    p.index_names.push_back(kv.key());
    p.index_sizes.push_back(static_cast<int>(kv.value().get<int64_t>()));
  }
  p.iv.assign(p.index_names.size(), 0);
  auto add_tensor = [&](const std::string& name, TKind kind, const std::vector<int>& ids, std::vector<int>& list) {
    Tensor t;
    t.name = name;
    t.kind = kind;
    for (int id : ids) t.sizes.push_back(p.index_sizes[id]);
    t.strides.assign(t.sizes.size(), 1);
    for (size_t d = t.sizes.size(); d-- > 1;) t.strides[d - 1] = t.strides[d] * static_cast<size_t>(t.sizes[d]);
    for (int s : t.sizes) {
      t.numel *= static_cast<size_t>(s);
      if (t.numel > (size_t(1) << 28))  // a capability limit (section 10.3)
        throw std::runtime_error("IR recurrence: tensor '" + name + "' holds more than 2^28 elements, more than ember's interpreter takes");
    }
    t.pos = static_cast<int>(list.size());
    const int id = static_cast<int>(p.tensors.size());
    p.tensor_of[name] = id;
    p.tensors.push_back(std::move(t));
    list.push_back(id);
    return id;
  };
  // inputs: one declaration per value of in
  size_t n_in = 0;
  if (node.contains("in")) n_in = node.at("in").is_array() ? node.at("in").size() : 1;
  const std::vector<std::string> inputs = strings(node, "inputs", true, true);
  if (inputs.size() != n_in)
    fail(31, "'inputs' declares " + std::to_string(inputs.size()) + " value(s) for the " + std::to_string(n_in) +
                 " of 'in'");
  for (size_t v = 0; v < inputs.size(); ++v) {
    const std::string where = "inputs[" + std::to_string(v) + "]";
    const auto toks = lex(inputs[v], where);
    size_t pos = 0;
    const auto [name, ids] = head(p, toks, pos, where, false);
    if (toks[pos].kind != Tok::End)
      fail(31, where + " does not parse: unexpected '" + toks[pos].text + "' at column " + std::to_string(toks[pos].column));
    declare(name, "input");
    const int id = add_tensor(name, TKind::Input, ids, p.inputs);
    p.in_widths.push_back(p.tensors[id].numel);
  }
  // weights: declarations to tensor names
  if (node.contains("weights") && !node.at("weights").is_null()) {
    if (!node.at("weights").is_object()) fail(31, "'weights' is not a map of declarations to tensor names");
    for (const auto& kv : node.at("weights").items()) {
      if (!kv.value().is_string() || kv.value().get<std::string>().empty())
        fail(31, "weight '" + kv.key() + "' does not name a tensor");
      const std::string where = "weights '" + kv.key() + "'";
      const auto toks = lex(kv.key(), where);
      size_t pos = 0;
      const auto [name, ids] = head(p, toks, pos, where, false);
      if (toks[pos].kind != Tok::End)
        fail(31, where + " does not parse: unexpected '" + toks[pos].text + "' at column " + std::to_string(toks[pos].column));
      declare(name, "weight");
      const int id = add_tensor(name, TKind::Weight, ids, p.weights);
      p.weight_info.push_back({name, kv.value().get<std::string>(), p.tensors[id].numel});
    }
  }
  // The assignments' heads first (the names every body may read), then the bodies.
  auto parse_heads = [&](const std::vector<std::string>& src, const char* section, std::vector<Assign>& dst,
                         std::vector<std::string>& names_out) {
    for (size_t k = 0; k < src.size(); ++k) {
      Assign a;
      a.where = std::string(section) + "[" + std::to_string(k) + "]";
      a.toks = lex(src[k], a.where);
      size_t pos = 0;
      auto h = head(p, a.toks, pos, a.where, true);
      if (a.toks[pos].kind != Tok::Eq)
        fail(31, a.where + " does not parse: expected '=' at column " + std::to_string(a.toks[pos].column));
      a.body_tok = pos + 1;
      a.lhs = h.second;
      names_out.push_back(h.first);
      dst.push_back(std::move(a));
    }
  };
  std::vector<std::string> init_names, let_names, update_names, readout_names;
  const std::vector<std::string> init_src = strings(node, "init", false, false);
  const std::vector<std::string> let_src = strings(node, "let", false, false);
  const std::vector<std::string> update_src = strings(node, "update", true, true);
  if (!node.contains("readout") || !node.at("readout").is_string()) fail(31, "'readout' is not one assignment");
  parse_heads(update_src, "update", p.update, update_names);
  parse_heads(let_src, "let", p.let, let_names);
  parse_heads(init_src, "init", p.init, init_names);
  {
    std::vector<Assign> ro;
    parse_heads({node.at("readout").get<std::string>()}, "readout", ro, readout_names);
    ro[0].where = "readout";
    p.readout = std::move(ro[0]);
  }
  for (size_t k = 0; k < p.update.size(); ++k) {
    declare(update_names[k], "state");
    p.update[k].tensor = add_tensor(update_names[k], TKind::State, p.update[k].lhs, p.states);
  }
  for (size_t k = 0; k < p.let.size(); ++k) {
    declare(let_names[k], "let");
    p.let[k].tensor = add_tensor(let_names[k], TKind::Let, p.let[k].lhs, p.lets);
  }
  {
    declare(readout_names[0], "readout");
    std::vector<int> outs;
    p.readout.tensor = add_tensor(readout_names[0], TKind::Out, p.readout.lhs, outs);
    p.out = p.readout.tensor;
    p.out_width = p.tensors[p.out].numel;
  }
  p.init_of_state.assign(p.states.size(), -1);
  for (size_t k = 0; k < p.init.size(); ++k) {
    const auto it = p.tensor_of.find(init_names[k]);
    if (it == p.tensor_of.end() || p.tensors[it->second].kind != TKind::State)
      fail(33, p.init[k].where + " assigns '" + init_names[k] + "', which is not a state tensor (an update's)");
    const Tensor& st = p.tensors[it->second];
    if (p.init_of_state[st.pos] >= 0) fail(33, p.init[k].where + " assigns '" + init_names[k] + "' a second time");
    std::vector<int> sizes;
    for (int id : p.init[k].lhs) sizes.push_back(p.index_sizes[id]);
    if (sizes != st.sizes) fail(33, p.init[k].where + " assigns '" + init_names[k] + "' with other sizes than its update");
    p.init_of_state[st.pos] = static_cast<int>(k);
    p.init[k].tensor = it->second;
  }
  // Bodies, with what each section reads (section 6.10.2).
  auto body = [&](Assign& a, const ReadRules& r) { a.body = BodyParser(p, a.toks, a.body_tok, a.where, r, a.lhs).parse(); };
  for (auto& a : p.init) body(a, ReadRules{false, true, false, 0});
  for (size_t k = 0; k < p.let.size(); ++k) body(p.let[k], ReadRules{true, true, true, static_cast<int>(k)});
  for (auto& a : p.update) body(a, ReadRules{true, true, true, static_cast<int>(p.let.size())});
  body(p.readout, ReadRules{true, true, true, static_cast<int>(p.let.size())});
  for (auto* v : {&p.init, &p.let, &p.update})
    for (auto& a : *v) std::vector<Token>().swap(a.toks);
  std::vector<Token>().swap(p.readout.toks);
  // The public description and the scratch.
  for (size_t s = 0; s < p.states.size(); ++s) {
    const Tensor& t = p.tensors[p.states[s]];
    StateTensor st;
    st.name = t.name;
    for (int x : t.sizes) st.shape.push_back(x);
    st.numel = t.numel;
    st.init_body = p.init_of_state[s] >= 0;
    p.state_info.push_back(std::move(st));
    p.new_state.emplace_back(t.numel);
  }
  for (int id : p.lets) p.let_buf.emplace_back(p.tensors[id].numel);
  p.data.assign(p.tensors.size(), nullptr);
  Recurrence r;
  r.p_ = std::move(P);
  return r;
}

std::string Recurrence::slot_errors(const json& slot) const {
  const std::string name = slot.contains("name") ? slot.at("name").dump() : "?";
  if (slot.value("op", json()) != "recurrence") return "the slot " + name + " of a recurrence is not of op recurrence";
  if (!slot.contains("tensors") || !slot.at("tensors").is_array()) return "the slot " + name + " declares no tensors";
  std::map<std::string, const json*> declared;
  for (const auto& t : slot.at("tensors")) {
    if (!t.is_object() || !t.contains("name") || !t.at("name").is_string())
      return "the slot " + name + " holds a tensor without a name";
    if (!declared.emplace(t.at("name").get<std::string>(), &t).second)
      return "the slot " + name + " declares tensor '" + t.at("name").get<std::string>() + "' twice";
  }
  for (const auto& st : p_->state_info) {
    const auto it = declared.find(st.name);
    if (it == declared.end()) return "the slot " + name + " has no tensor '" + st.name + "' (the update assigns it)";
    const json& t = *it->second;
    const json want_shape = st.shape;
    if (t.value("shape", json()) != want_shape)
      return "the slot " + name + "'s tensor '" + st.name + "' has shape " + t.value("shape", json()).dump() +
             ", the update's is " + want_shape.dump();
    const json init = t.value("init", json());
    if (st.init_body ? init != "body" : init != json(0))
      return "the slot " + name + "'s tensor '" + st.name + "' has init " + init.dump() + ", not " +
             (st.init_body ? "\"body\" (the node's init assigns it)" : "0 (the node's init does not assign it)");
  }
  if (declared.size() != p_->state_info.size()) {
    for (const auto& kv : declared) {
      bool found = false;
      for (const auto& st : p_->state_info) found = found || st.name == kv.first;
      if (!found) return "the slot " + name + "'s tensor '" + kv.first + "' is not one the update assigns";
    }
  }
  return {};
}

void Recurrence::init(const std::vector<const float*>& weights, const std::vector<float*>& state) const {
  const Program& p = *p_;
  for (size_t w = 0; w < p.weights.size(); ++w) p.data[p.weights[w]] = weights[w];
  for (size_t s = 0; s < p.states.size(); ++s) {
    const int k = p.init_of_state[s];
    if (k < 0)
      std::fill(state[s], state[s] + p.state_info[s].numel, 0.0f);
    else
      p.assign(p.init[k], state[s]);
  }
}

void Recurrence::step(const std::vector<const float*>& in, const std::vector<const float*>& weights,
                      const std::vector<float*>& state, float* out) const {
  const Program& p = *p_;
  for (size_t v = 0; v < p.inputs.size(); ++v) p.data[p.inputs[v]] = in[v];
  for (size_t w = 0; w < p.weights.size(); ++w) p.data[p.weights[w]] = weights[w];
  for (size_t s = 0; s < p.states.size(); ++s) p.data[p.states[s]] = state[s];
  for (size_t l = 0; l < p.lets.size(); ++l) {
    p.assign(p.let[l], p.let_buf[l].data());
    p.data[p.lets[l]] = p.let_buf[l].data();
  }
  // The whole new state from the state before the token, then it replaces the old.
  for (size_t s = 0; s < p.states.size(); ++s) p.assign(p.update[s], p.new_state[s].data());
  for (size_t s = 0; s < p.states.size(); ++s) std::copy(p.new_state[s].begin(), p.new_state[s].end(), state[s]);
  p.assign(p.readout, out);
}

}  // namespace rec
}  // namespace ember_ir
