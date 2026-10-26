// IR_SPEC section 10.1 at load (graph_ir_validate.h): a port of paperkiln's tools/ir_validate.py.
#include "graph_ir_validate.h"

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include "gguf_structs.h"
#include "graph_ir.h"
#include "graph_ir_expr.h"
#include "graph_ir_recurrence.h"
#include "graph_ir_schema.h"
#include "ir_digest.h"

using nlohmann::json;

namespace ember_ir {
namespace {

std::string E(const std::string& section, const std::string& msg) { return "IR_SPEC " + section + ": " + msg; }

std::string short_dump(const json& v) {
  std::string s = v.dump(-1, ' ', false, json::error_handler_t::replace);
  return s.size() > 60 ? s.substr(0, 60) : s;
}

// ---------------------------------------------------------------- the schema (rule 1)

const json& schema() {
  static const json s = [] {
    std::string text;
    for (const char* p : kIrSchemaPieces) text += p;
    return json::parse(text);
  }();
  return s;
}

// JSON equality where a boolean equals only a boolean (the validator's _json_eq).
bool json_eq(const json& a, const json& b) {
  if (a.is_boolean() || b.is_boolean()) return a.is_boolean() && b.is_boolean() && a == b;
  return a == b;
}

bool type_ok(const json& v, const json& t) {
  if (t.is_array()) {
    for (const auto& x : t)
      if (type_ok(v, x)) return true;
    return false;
  }
  const std::string n = t.get<std::string>();
  if (n == "object") return v.is_object();
  if (n == "array") return v.is_array();
  if (n == "string") return v.is_string();
  if (n == "boolean") return v.is_boolean();
  if (n == "null") return v.is_null();
  if (n == "integer") return v.is_number_integer();
  if (n == "number") return v.is_number();
  throw std::logic_error("IR schema: unknown type " + n);
}

bool pattern_match(const std::string& pattern, const std::string& key) {
  static std::map<std::string, std::regex> cache;
  auto it = cache.find(pattern);
  if (it == cache.end()) it = cache.emplace(pattern, std::regex(pattern, std::regex::ECMAScript)).first;
  return std::regex_search(key, it->second);
}

using Failures = std::vector<std::pair<std::string, std::string>>;

void eval(const json& s, const json& v, const std::string& path, Failures& out);

Failures collect(const json& s, const json& v, const std::string& path) {
  Failures out;
  eval(s, v, path, out);
  return out;
}

// The keywords spec/ir.schema.json uses, as tools/ir_validate.py evaluates them.
void eval(const json& s, const json& v, const std::string& path, Failures& out) {
  if (s.is_boolean()) {
    if (!s.get<bool>()) out.emplace_back(path, "is not allowed");
    return;
  }
  if (s.empty()) return;
  if (s.contains("$ref")) {
    const std::string ref = s.at("$ref").get<std::string>();
    const std::string pre = "#/$defs/";
    if (ref.compare(0, pre.size(), pre) != 0) throw std::logic_error("IR schema: $ref " + ref);
    eval(schema().at("$defs").at(ref.substr(pre.size())), v, path, out);
  }
  if (s.contains("type") && !type_ok(v, s.at("type"))) {
    out.emplace_back(path, "must be of type " + s.at("type").dump() + ", not " + short_dump(v));
    return;
  }
  if (s.contains("enum")) {
    bool any = false;
    for (const auto& e : s.at("enum")) any = any || json_eq(v, e);
    if (!any) out.emplace_back(path, short_dump(v) + " is not one of " + s.at("enum").dump());
  }
  if (s.contains("const") && !json_eq(v, s.at("const"))) out.emplace_back(path, "must be " + s.at("const").dump());
  if (v.is_number() && s.contains("minimum") && v.get<double>() < s.at("minimum").get<double>())
    out.emplace_back(path, v.dump() + " is below the minimum " + s.at("minimum").dump());
  if (v.is_string() && s.contains("minLength") && v.get<std::string>().size() < s.at("minLength").get<size_t>())
    out.emplace_back(path, "must not be empty");
  if (v.is_array()) {
    if (s.contains("minItems") && v.size() < s.at("minItems").get<size_t>())
      out.emplace_back(path, "needs at least " + s.at("minItems").dump() + " item(s)");
    if (s.contains("items"))
      for (size_t i = 0; i < v.size(); ++i) eval(s.at("items"), v[i], path + "[" + std::to_string(i) + "]", out);
  }
  if (v.is_object()) {
    if (s.contains("required"))
      for (const auto& k : s.at("required"))
        if (!v.contains(k.get<std::string>())) out.emplace_back(path, "lacks the required key '" + k.get<std::string>() + "'");
    static const json kEmpty = json::object();
    const json& props = s.contains("properties") ? s.at("properties") : kEmpty;
    const json& pats = s.contains("patternProperties") ? s.at("patternProperties") : kEmpty;
    for (const auto& kv : v.items()) {
      bool matched = false;
      if (props.contains(kv.key())) {
        matched = true;
        eval(props.at(kv.key()), kv.value(), path + "." + kv.key(), out);
      }
      for (const auto& p : pats.items())
        if (pattern_match(p.key(), kv.key())) {
          matched = true;
          eval(p.value(), kv.value(), path + "." + kv.key(), out);
        }
      if (!matched && s.contains("additionalProperties")) {
        const json& ap = s.at("additionalProperties");
        if (ap.is_boolean() && !ap.get<bool>()) {
          // A node names its op, so the message says which op does not define the key (rule 4).
          const std::string op = v.contains("op") && v.at("op").is_string() ? " (" + v.at("op").get<std::string>() + ")" : "";
          out.emplace_back(path + op, "has the key '" + kv.key() + "', which its specification does not define");
        } else {
          eval(ap, kv.value(), path + "." + kv.key(), out);
        }
      }
    }
  }
  if (s.contains("allOf"))
    for (const auto& sub : s.at("allOf")) eval(sub, v, path, out);
  if (s.contains("anyOf")) {
    bool any = false;
    for (const auto& sub : s.at("anyOf")) any = any || collect(sub, v, path).empty();
    if (!any) {
      // The closest alternative's failures (the fewest), as the validator reports them.
      Failures best;
      bool first = true;
      for (const auto& sub : s.at("anyOf")) {
        Failures f = collect(sub, v, path);
        if (first || f.size() < best.size()) best = std::move(f);
        first = false;
      }
      if (best.empty()) best.emplace_back(path, "matches no alternative");
      out.insert(out.end(), best.begin(), best.end());
    }
  }
  if (s.contains("if")) {
    if (collect(s.at("if"), v, path).empty()) {
      if (s.contains("then")) eval(s.at("then"), v, path, out);
    } else if (s.contains("else")) {
      eval(s.at("else"), v, path, out);
    }
  }
}

void add_unique(std::vector<std::string>& out, std::set<std::string>& seen, const std::string& m) {
  if (seen.insert(m).second) out.push_back(m);
}

// ---------------------------------------------------------------- the document rules (3, 5-23)

using Errs = std::vector<std::string>;

const json& kNull() {
  static const json n;
  return n;
}
const json& kObject() {
  static const json o = json::object();
  return o;
}
const json& get(const json& o, const std::string& k) { return o.is_object() && o.contains(k) ? o.at(k) : kNull(); }
bool is_int(const json& v) { return v.is_number_integer(); }
bool is_pl(const json& v) { return v.is_object() && v.contains("per_layer"); }
// Python's truthiness, for the rules the validator states with any().
bool truthy(const json& v) {
  if (v.is_null()) return false;
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_number()) return v.get<double>() != 0.0;
  if (v.is_string()) return !v.get<std::string>().empty();
  return !v.empty();
}
// A value as a message shows it: a string bare, anything else as JSON (null as None, as the validator prints).
std::string show(const json& v) {
  if (v.is_string()) return v.get<std::string>();
  if (v.is_null()) return "None";
  return v.dump();
}
std::string q(const std::string& s) { return "'" + s + "'"; }

// The value per application: the per_layer list (nullopt when its length is not L) or the scalar L times.
std::optional<std::vector<json>> layers(const json& v, int L) {
  if (is_pl(v)) {
    const json& pl = v.at("per_layer");
    if (!pl.is_array() || static_cast<int>(pl.size()) != L) return std::nullopt;
    return std::vector<json>(pl.begin(), pl.end());
  }
  return std::vector<json>(static_cast<size_t>(std::max(L, 0)), v);
}

struct Sect {
  std::string sect;
  size_t j;
  const json* n;
};

std::vector<Sect> sections(const json& doc) {
  std::vector<Sect> out;
  for (const char* sect : {"embed", "block", "head"}) {
    const json& nodes = get(get(doc, sect), "nodes");
    if (!nodes.is_array()) continue;
    for (size_t j = 0; j < nodes.size(); ++j)
      if (nodes[j].is_object()) out.push_back({sect, j, &nodes[j]});
  }
  return out;
}

const std::set<std::string> kNotPerLayer = {"op", "in", "out", "state", "tensor_index", "sizes", "expr", "weight", "bias",
                                            "position_bias",  // attention's (IR_SPEC 5.15, revision 1.3)
                                            "router", "experts", "state_weight", "tied_to", "a", "b", "c", "d",
                                            // recurrence (IR_SPEC 6.10.1: every key structural)
                                            "index", "inputs", "weights", "init", "let", "update", "readout"};
const std::set<std::string> kSlotNotPerLayer = {"name", "kind", "op", "dtype", "tensors", "evict", "kv_rows"};
const std::map<std::string, size_t> kArity = {
    {"embedding", 1}, {"pos_embedding", 2}, {"linear", 1}, {"scale", 1}, {"split", 1}, {"rmsnorm", 1},
    {"layernorm", 1}, {"gelu", 1}, {"silu", 1}, {"sigmoid", 1}, {"relu", 1}, {"add", 2}, {"mul", 2}, {"sub", 2},
    {"affine", 1}, {"mul_col", 2}, {"rms_row", 1}, {"rope", 3}, {"attention", 3}, {"moe_ffn", 1}, {"append", 2},
    {"linear_attention", 3}, {"ssm_scan", 1}, {"conv1d_causal", 1}, {"selective_scan", 4}, {"depth_route", 1}};
const std::set<std::string> kAtLeastOne = {"list", "depth_attend", "expr", "recurrence"};
const std::map<std::string, std::string> kStateful = {{"attention", "kv"},
                                                      {"linear_attention", "recurrent"},
                                                      {"ssm_scan", "recurrent"},
                                                      {"conv1d_causal", "recurrent"},
                                                      {"selective_scan", "recurrent"},
                                                      {"recurrence", "recurrent"}};
const std::set<std::string> kRecurrent = {"linear_attention", "ssm_scan", "conv1d_causal", "selective_scan",
                                          "recurrence"};

std::string op_of(const json& n) { return get(n, "op").is_string() ? get(n, "op").get<std::string>() : ""; }

// (key, name) for every tensor a node names.
std::vector<std::pair<std::string, std::string>> tensor_names(const json& n) {
  std::vector<std::pair<std::string, std::string>> out;
  for (const char* k : {"weight", "bias", "router", "state_weight", "tied_to"})
    if (get(n, k).is_string()) out.emplace_back(k, get(n, k).get<std::string>());
  const std::string op = op_of(n);
  if (op == "ssm_scan" || op == "selective_scan")
    for (const char* k : {"a", "b", "c", "d"})
      if (get(n, k).is_string()) out.emplace_back(k, get(n, k).get<std::string>());
  if (get(n, "experts").is_object())
    for (const auto& kv : n.at("experts").items())
      if (kv.value().is_string()) out.emplace_back("experts." + kv.key(), kv.value().get<std::string>());
  if (op == "recurrence" && get(n, "weights").is_object())
    for (const auto& kv : n.at("weights").items())
      if (kv.value().is_string()) out.emplace_back("weights." + kv.key(), kv.value().get<std::string>());
  return out;
}

size_t count_i(const std::string& s) {
  size_t c = 0;
  for (size_t p = s.find("{i}"); p != std::string::npos; p = s.find("{i}", p + 3)) ++c;
  return c;
}

// Rules 4-10 and 14 over every node, and the names and graph inputs (3.5, 3.7, 7.2).
void node_errors(const json& doc, Errs& errs) {
  const json& blk = get(doc, "block").is_object() ? doc.at("block") : kObject();
  const int L = is_int(get(blk, "repeat")) ? blk.at("repeat").get<int>() : 0;
  for (const auto& [sect, j, np] : sections(doc)) {
    const json& n = *np;
    const std::string op = op_of(n);
    const std::string where = sect + "[" + std::to_string(j) + "] " + show(get(n, "op"));
    std::vector<json> ins;
    if (get(n, "in").is_array()) ins.assign(n.at("in").begin(), n.at("in").end());
    const json& outs = get(n, "out");
    // Rule 6: per-layer values.
    for (const auto& kv : n.items()) {
      if (!is_pl(kv.value())) continue;
      const json& pl = kv.value().at("per_layer");
      if (sect != "block")
        errs.push_back(E("3.8", where + ": per-layer " + q(kv.key()) + " outside the block"));
      else if (kNotPerLayer.count(kv.key()))
        errs.push_back(E("3.8", where + ": " + q(kv.key()) + " is structural and cannot be per-layer"));
      else if (!pl.is_array() || static_cast<int>(pl.size()) != L)
        errs.push_back(E("3.8", where + ": per-layer " + q(kv.key()) + " has " +
                                    (pl.is_array() ? std::to_string(pl.size()) : std::string("no list")) +
                                    " entries for block.repeat " + std::to_string(L)));
    }
    // Rule 7: arity.
    if (auto it = kArity.find(op); it != kArity.end() && ins.size() != it->second)
      errs.push_back(E("10.1 rule 7", where + " reads " + std::to_string(ins.size()) + " value(s), not " +
                                          std::to_string(it->second)));
    if (kAtLeastOne.count(op) && ins.empty()) errs.push_back(E("10.1 rule 7", where + " reads no value"));
    if (op == "expr" && ins.size() > 23)
      errs.push_back(E("5.19", where + " reads " + std::to_string(ins.size()) + " values (at most 23)"));
    if (op == "split" || op == "rope") {
      if (!outs.is_array())
        errs.push_back(E("10.1 rule 7", where + " writes a list of names"));
      else if (op == "rope" && outs.size() != 2)
        errs.push_back(E("5.14", where + " writes " + std::to_string(outs.size()) + " values, not [q, k]"));
    } else if (!get(n, "op").is_null() && !outs.is_string()) {
      errs.push_back(E("10.1 rule 7", where + " writes one value"));
    }
    // Rule 9: expr parses.
    if (op == "expr" && get(n, "expr").is_string()) {
      try {
        expr::Expr::parse(n.at("expr").get<std::string>(), std::max<size_t>(1, ins.size()));
      } catch (const std::exception& e) {
        errs.push_back(E("5.19", where + ": " + e.what()));
      }
    }
    // Rules 10 and 14: values.
    if (op == "split" && get(n, "sizes").is_array() && outs.is_array() && outs.size() != n.at("sizes").size())
      errs.push_back(E("5.5", where + ": " + std::to_string(n.at("sizes").size()) + " sizes for " +
                                  std::to_string(outs.size()) + " outputs"));
    if (op == "rope" && L) {
      const auto hd = layers(get(n, "head_dim"), L), nr = layers(get(n, "n_rot"), L), fs = layers(get(n, "freq_scale"), L);
      for (int i = 0; i < L; ++i) {
        if (nr && hd && !(*nr)[i].is_null() && is_int((*hd)[i]) && is_int((*nr)[i])) {
          const int64_t a = (*nr)[i].get<int64_t>(), h = (*hd)[i].get<int64_t>();
          if (!(0 < a && a <= h && a % 2 == 0)) {
            errs.push_back(E("5.14", where + ": n_rot " + std::to_string(a) + " is not even and in 1 .. head_dim " +
                                         std::to_string(h)));
            break;
          }
        }
        if (fs && !(*fs)[i].is_null() && (*fs)[i].is_number() && !((*fs)[i].get<double>() > 0)) {
          errs.push_back(E("5.14", where + ": freq_scale " + show((*fs)[i]) + " is not positive"));
          break;
        }
      }
    }
    if (op == "attention" && L) {
      const auto h = layers(get(n, "heads"), L), kvh = layers(get(n, "kv_heads"), L), w = layers(get(n, "window"), L),
                 s = layers(get(n, "sinks"), L), c = layers(get(n, "causal"), L);
      for (int i = 0; h && kvh && i < L; ++i)
        if (is_int((*h)[i]) && is_int((*kvh)[i]) && (*kvh)[i].get<int64_t>() > 0 &&
            (*h)[i].get<int64_t>() % (*kvh)[i].get<int64_t>()) {
          errs.push_back(E("5.15", where + ": heads " + show((*h)[i]) + " is not a multiple of kv_heads " + show((*kvh)[i])));
          break;
        }
      for (int i = 0; i < L; ++i)
        if ((w && is_int((*w)[i]) && (*w)[i].get<int64_t>() < 0) || (s && is_int((*s)[i]) && (*s)[i].get<int64_t>() < 0)) {
          errs.push_back(E("5.15", where + ": window and sinks must be >= 0"));
          break;
        }
      bool not_true = false;
      for (int i = 0; c && i < L; ++i) not_true = not_true || !((*c)[i].is_boolean() && (*c)[i].get<bool>());
      if (not_true) errs.push_back(E("5.15", where + ": causal must be true (Q6)"));
    }
    if (op == "linear_attention" && !(get(n, "causal").is_boolean() && n.at("causal").get<bool>()))
      errs.push_back(E("6.4", where + ": causal must be true"));
    if (op == "moe_ffn" && is_int(get(n, "top_k")) && is_int(get(n, "n_experts"))) {
      const int64_t k = n.at("top_k").get<int64_t>(), e = n.at("n_experts").get<int64_t>();
      if (!(1 <= k && k <= e))
        errs.push_back(E("5.16", where + ": top_k " + std::to_string(k) + " is not in 1 .. n_experts " + std::to_string(e)));
    }
    if (op == "conv1d_causal" && ((is_int(get(n, "channels")) && n.at("channels").get<int64_t>() < 1) ||
                                  (is_int(get(n, "kernel")) && n.at("kernel").get<int64_t>() < 2)))
      errs.push_back(E("6.6", where + ": channels must be >= 1 and kernel >= 2"));
    // Names: {i} outside the block or twice (3.7).
    for (const auto& [key, name] : tensor_names(n)) {
      if (name.find("{i}") != std::string::npos && sect != "block")
        errs.push_back(E("3.7", where + ": " + key + " " + q(name) + " uses {i} outside the block"));
      if (count_i(name) > 1) errs.push_back(E("3.7", where + ": " + key + " " + q(name) + " holds {i} more than once"));
    }
    // The graph inputs (3.5): tokens for embeddings, pos as rope's third and pos_embedding's second input.
    for (size_t k = 0; k < ins.size(); ++k) {
      if (!ins[k].is_string()) continue;
      const std::string nm = ins[k].get<std::string>();
      const bool ok = nm == "tokens" ? (op == "embedding" && k == 0)
                    : nm == "pos"    ? ((op == "rope" && k == 2) || (op == "pos_embedding" && k == 1))
                                     : true;
      if (!ok) errs.push_back(E("3.5", where + " reads the graph input " + q(nm) + " as its input " + std::to_string(k)));
    }
    if ((op == "rope" && ins.size() == 3 && ins[2] != "pos") || (op == "pos_embedding" && ins.size() == 2 && ins[1] != "pos"))
      errs.push_back(E("3.5", where + ": the position input must be the graph input 'pos'"));
    if (op == "embedding" && !ins.empty() && ins[0] != "tokens")
      errs.push_back(E("3.5", where + ": an embedding reads the graph input 'tokens'"));
    const std::vector<json> out_list = outs.is_array() ? std::vector<json>(outs.begin(), outs.end()) : std::vector<json>{outs};
    for (const auto& o : out_list)
      if (o == "tokens" || o == "pos") errs.push_back(E("3.5", where + " writes the graph input " + q(o.get<std::string>())));
    if (n.contains("tensor_index") && (sect != "block" || !blk.contains("weight_index")))
      errs.push_back(E("7.2", where + ": tensor_index needs a block with weight_index"));
  }
}

// Rule 12 (attention against its kv slot) and rule 13 (a recurrent slot's tensors fit its op).
void slot_fit(const json& node, const json& slot, int L, Errs& errs) {
  const std::string op = op_of(node);
  if (op == "attention") {
    for (const char* key : {"window", "sinks", "kv_heads", "head_dim"}) {
      const auto a = layers(get(node, key), L), b = layers(get(slot, key), L);
      if (!a || !b) continue;
      for (int i = 0; i < L; ++i)
        if ((*a)[i] != (*b)[i]) {
          const bool q7 = std::string(key) == "kv_heads" || std::string(key) == "head_dim";
          errs.push_back(E("5.15", "attention " + std::string(key) + " " + show((*a)[i]) + " (application " +
                                       std::to_string(i) + ") differs from its kv state " + show(get(slot, "name")) +
                                       "'s " + show((*b)[i]) + (q7 ? " (Q7)" : "")));
          break;
        }
    }
    return;
  }
  if (op == "recurrence") {  // rules 31-33 (graph_ir_recurrence.h)
    // A slot whose tensors rule 13 refuses (no name, shape or init 0 / "body") is not compared with the node: the
    // fault is rule 13's alone.
    if (get(slot, "tensors").is_array())
      for (const auto& t : slot.at("tensors"))
        if (!t.is_object() || !(get(t, "init") == json(0) || get(t, "init") == "body") || !get(t, "shape").is_array())
          return;
    try {
      const std::string why = rec::Recurrence::compile(node).slot_errors(slot);
      if (!why.empty()) errs.push_back(E("6.10", "(rule 33) " + why));
    } catch (const std::exception& e) {
      std::string m = e.what();
      const std::string tag = "IR_SPEC 6.10 ";
      errs.push_back(E("6.10", m.rfind(tag, 0) == 0 ? m.substr(tag.size()) : m));
    }
    return;
  }
  json shapes = json::object();
  if (get(slot, "tensors").is_array())
    for (const auto& t : slot.at("tensors"))
      if (t.is_object()) shapes[show(get(t, "name"))] = get(t, "shape");
  json want;
  if (op == "linear_attention") {
    want = json::object();
    want["num"] = json::array({get(node, "heads"), get(node, "head_dim")});
    want["den"] = want["num"];
  } else if (op == "ssm_scan") {
    want = json::object();
    want["s"] = json::array({get(node, "n")});
  } else if (op == "conv1d_causal") {
    const json& k = get(node, "kernel");
    want = json::object();
    want["x"] = json::array({is_int(k) ? json(k.get<int64_t>() - 1) : k, get(node, "channels")});
  } else {
    want = json::object();
    want["s"] = json::array({get(node, "channels"), get(node, "d_state")});
  }
  if (shapes != want)
    errs.push_back(E("6.3", "recurrent state " + show(get(slot, "name")) + ": tensors " + shapes.dump() + " do not fit " +
                                op + " (want " + want.dump() + ")"));
}

// Rules 3, 11-13 and 18: the model fields, the slots, the stateful nodes, weight_index.
void state_errors(const json& doc, Errs& errs) {
  const json& model = get(doc, "model").is_object() ? doc.at("model") : kObject();
  const json& blk = get(doc, "block").is_object() ? doc.at("block") : kObject();
  for (const char* k : {"d", "vocab", "ctx", "n_layers"})
    if (!is_int(get(model, k)) || model.at(k).get<int64_t>() <= 0)
      errs.push_back(E("3.3", std::string("model.") + k + " missing or not a positive integer"));
  if (get(blk, "repeat") != get(model, "n_layers"))
    errs.push_back(E("3.3", "block.repeat " + show(get(blk, "repeat")) + " != model.n_layers " + show(get(model, "n_layers"))));
  const int L = is_int(get(blk, "repeat")) ? blk.at("repeat").get<int>() : 0;
  if (blk.contains("weight_index")) {
    const json& m = get(blk.at("weight_index"), "per_layer");
    bool ok = m.is_array() && static_cast<int>(m.size()) == L;
    for (size_t i = 0; ok && i < m.size(); ++i) ok = is_int(m[i]) && m[i].get<int64_t>() >= 0;
    if (!ok) {
      errs.push_back(E("7.1", "block.weight_index must be {per_layer: [block.repeat integers >= 0]}"));
    } else if (!m.empty()) {
      std::set<int64_t> used;
      for (const auto& x : m) used.insert(x.get<int64_t>());
      const int64_t mx = *used.rbegin();
      if (static_cast<int64_t>(used.size()) != mx + 1) {
        std::string s;
        for (int64_t x : used) s += (s.empty() ? "" : ", ") + std::to_string(x);
        errs.push_back(E("7.1", "block.weight_index uses stored layers [" + s + "], not every one of 0.." + std::to_string(mx)));
      }
    }
  }
  std::vector<std::string> order;
  std::map<std::string, const json*> slots;
  if (get(blk, "state").is_array())
    for (const auto& s : blk.at("state")) {
      if (!s.is_object()) continue;
      const std::string name = show(get(s, "name"));
      if (slots.count(name))
        errs.push_back(E("6.1", "state slot " + name + " declared twice"));
      else
        order.push_back(name);
      slots[name] = &s;
      for (const auto& kv : s.items()) {
        if (!is_pl(kv.value())) continue;
        const json& pl = kv.value().at("per_layer");
        if (kSlotNotPerLayer.count(kv.key()))
          errs.push_back(E("3.8", "state slot " + name + ": " + q(kv.key()) + " is structural and cannot be per-layer"));
        else if (!pl.is_array() || static_cast<int>(pl.size()) != L)
          errs.push_back(E("3.8", "state slot " + name + ": per-layer " + q(kv.key()) + " needs block.repeat " +
                                      std::to_string(L) + " entries"));
      }
      const json& kind = get(s, "kind");
      if (kind == "kv") {
        for (const char* k : {"heads", "kv_heads", "head_dim", "max_len", "window", "sinks"})
          if (!(is_int(get(s, k)) || is_pl(get(s, k)))) errs.push_back(E("6.2", "kv state " + name + ": " + k + " missing"));
        if (s.contains("kv_from")) errs.push_back(E("7.6", "kv state " + name + ": kv_from is reserved and not implemented"));
      } else if (kind == "recurrent") {
        const json& ts = get(s, "tensors");
        if (!get(s, "op").is_string() || !kRecurrent.count(s.at("op").get<std::string>()) || get(s, "dtype") != "f32" ||
            !truthy(ts))
          errs.push_back(E("6.3", "recurrent state " + name + ": op (a recurrent op), dtype f32 and tensors required"));
        if (ts.is_array())
          for (const auto& t : ts) {
            bool ok = t.is_object() && truthy(get(t, "name")) && truthy(get(t, "shape")) && get(t, "shape").is_array() &&
                      (get(t, "init") == json(0) || (get(t, "init") == "body" && get(s, "op") == "recurrence"));
            if (ok)
              for (const auto& x : t.at("shape")) ok = ok && is_int(x) && x.get<int64_t>() > 0;
            if (!ok)
              errs.push_back(E("6.3", "recurrent state " + name + ": tensor " + t.dump() + " needs name, shape and init 0"));
          }
      } else {
        errs.push_back(E("6.1", "state slot " + name + ": unknown kind " + show(kind)));
      }
    }
  std::set<std::string> used;
  std::vector<std::string> reader_order;
  std::map<std::string, int> readers;
  for (const auto& [sect, j, np] : sections(doc)) {
    const json& node = *np;
    const std::string op = op_of(node);
    const auto want = kStateful.find(op);
    if (want == kStateful.end()) {
      if (node.contains("state"))
        errs.push_back(E("6.1", sect + "[" + std::to_string(j) + "] " + show(get(node, "op")) + " names state " +
                                    q(show(node.at("state"))) + " but has no state"));
      continue;
    }
    if (sect != "block") {
      errs.push_back(E("6.1", sect + ": " + op + " outside the block has no state"));
      continue;
    }
    const std::string st = show(get(node, "state"));
    const auto it = get(node, "state").is_string() ? slots.find(st) : slots.end();
    if (it == slots.end() || get(*it->second, "kind") != want->second ||
        (want->second == "recurrent" && get(*it->second, "op") != op)) {
      errs.push_back(E("6.1", sect + ": " + op + " reads state " + q(st) + ", not a declared " + want->second + " slot" +
                                  (want->second == "recurrent" ? " of op " + op : std::string())));
      continue;
    }
    used.insert(st);
    if (op == "attention" || op == "recurrence") {
      if (!readers.count(st)) reader_order.push_back(st);
      ++readers[st];
    }
    slot_fit(node, *it->second, L, errs);
  }
  for (const auto& name : reader_order)
    if (readers[name] > 1) {
      if (get(*slots.at(name), "kind") == "kv")
        errs.push_back(E("6.1", "kv state " + name + " is read by " + std::to_string(readers[name]) +
                                    " attention nodes (exactly one is defined)"));
      else
        errs.push_back(E("6.10", "(rule 33) the slot " + name + " of a recurrence is read by " +
                                     std::to_string(readers[name]) + " nodes (the node alone reads it)"));
    }
  for (const auto& name : order)
    if (!used.count(name)) errs.push_back(E("6.1", "state slot " + name + " is read by no node"));
}

// Rules 20, 21 and 23: depth_route first and once, its steps, the router name, kv_rows active.
void depth_route_errors(const json& doc, Errs& errs) {
  const json& blk = get(doc, "block").is_object() ? doc.at("block") : kObject();
  const json& nodes = get(blk, "nodes").is_array() ? blk.at("nodes") : json::array();
  for (const auto& [sect, j, np] : sections(doc))
    if (op_of(*np) == "depth_route" && sect != "block") errs.push_back(E("7.4", "depth_route outside the block (" + sect + ")"));
  std::vector<size_t> at;
  for (size_t j = 0; j < nodes.size(); ++j)
    if (nodes[j].is_object() && op_of(nodes[j]) == "depth_route") at.push_back(j);
  std::vector<const json*> kvs;
  if (get(blk, "state").is_array())
    for (const auto& s : blk.at("state"))
      if (s.is_object() && get(s, "kind") == "kv") kvs.push_back(&s);
  for (const json* s : kvs)
    if (s->contains("kv_rows") && s->at("kv_rows") != "active")
      errs.push_back(E("7.5", "kv state " + show(get(*s, "name")) + ": kv_rows " + q(show(s->at("kv_rows"))) +
                                  " (only \"active\" is defined)"));
  if (at.empty()) return;
  if (at != std::vector<size_t>{0}) {
    errs.push_back(E("7.4", "depth_route must be the block's first node, once"));
    return;
  }
  const json& r = nodes[0];
  const json& Rj = get(r, "n_rec");
  if (!is_int(Rj) || Rj.get<int64_t>() < 1) {
    errs.push_back(E("7.4", "depth_route: n_rec must be an integer >= 1"));
    return;
  }
  const int64_t R = Rj.get<int64_t>();
  if (get(r, "route") != "token_choice")
    errs.push_back(E("7.7", "depth_route: route " + q(show(get(r, "route"))) + " (only token_choice, causal, is defined)"));
  if (get(r, "gate") != "score" && get(r, "gate") != "none")
    errs.push_back(E("7.4", "depth_route: gate " + q(show(get(r, "gate"))) + " is not score or none"));
  const json& w = get(r, "weight");
  if (!w.is_string() || w.get<std::string>().find('{') != std::string::npos)
    errs.push_back(E("7.7", "depth_route: weight must be the router's full tensor name (constraint 4: no substitution)"));
  if (!get(r, "out").is_string()) errs.push_back(E("7.4", "depth_route: out must name the depth trace"));
  const json& st = get(get(r, "step"), "per_layer");
  const json& rep = get(blk, "repeat");
  bool ints = st.is_array() && is_int(rep) && static_cast<int64_t>(st.size()) == rep.get<int64_t>();
  for (size_t i = 0; ints && i < st.size(); ++i) ints = is_int(st[i]);
  if (!ints) {
    errs.push_back(E("7.4", "depth_route: step must be {per_layer: [block.repeat integers]}"));
    return;
  }
  std::vector<int64_t> s;
  for (const auto& x : st) s.push_back(x.get<int64_t>());
  std::vector<std::pair<int64_t, int64_t>> runs;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == -1) continue;
    if (i == 0 || s[i - 1] != s[i]) runs.emplace_back(s[i], 0);
    ++runs.back().second;
  }
  std::vector<size_t> inside;
  for (size_t i = 0; i < s.size(); ++i)
    if (s[i] != -1) inside.push_back(i);
  const bool contiguous = !inside.empty() && inside.back() - inside.front() + 1 == inside.size();
  bool range_bad = false, keys_bad = static_cast<int64_t>(runs.size()) != R;
  for (int64_t x : s) range_bad = range_bad || x < -1 || x >= R;
  for (size_t k = 0; !keys_bad && k < runs.size(); ++k) keys_bad = runs[k].first != static_cast<int64_t>(k);
  if (range_bad || keys_bad || !contiguous) {
    errs.push_back(E("7.4", "depth_route: steps " + st.dump() + " are not -1 or the consecutive steps 0 .. " +
                                std::to_string(R - 1) + " in order"));
  } else {
    std::set<int64_t> lens;
    for (const auto& run : runs) lens.insert(run.second);
    if (lens.size() != 1) errs.push_back(E("7.4", "depth_route: steps " + st.dump() + " differ in length"));
  }
  bool all_active = !kvs.empty();
  for (const json* k : kvs) all_active = all_active && get(*k, "kv_rows") == "active";
  if (!all_active) errs.push_back(E("7.5", "depth_route needs kv_rows \"active\" on every kv slot"));
  for (const auto& n : nodes)
    if (n.is_object() && op_of(n) == "attention" && get(n, "mask") == "turn_state") {
      errs.push_back(E("9.2", "depth_route with mask turn_state is not defined (llama_state)"));
      break;
    }
  for (const auto& [sect, j, np] : sections(doc)) {
    const std::string op = op_of(*np);
    if (op == "list" || op == "append" || op == "depth_attend") {
      errs.push_back(E("7.4", "depth_route with list values (list, append, depth_attend) is not defined"));
      break;
    }
  }
  const json& out = get(r, "out");
  if (out.is_string())
    for (const auto& [sect, j, np] : sections(doc)) {
      const json& ins = get(*np, "in");
      if (ins.is_array() && std::find(ins.begin(), ins.end(), out) != ins.end())
        errs.push_back(E("7.4", sect + "[" + std::to_string(j) + "] " + show(get(*np, "op")) + " reads depth_route's out " +
                                    q(out.get<std::string>()) + " (the depth report)"));
    }
}

// Rules 15-17: the turn_state mask, kv evict and the state_weight's placement.
void state_token_doc_errors(const json& doc, Errs& errs) {
  const json& blk = get(doc, "block").is_object() ? doc.at("block") : kObject();
  const int L = is_int(get(blk, "repeat")) ? blk.at("repeat").get<int>() : 0;
  size_t sws = 0;
  for (const auto& [sect, j, np] : sections(doc)) {
    const json& n = *np;
    if (op_of(n) != "embedding" || !n.contains("state_weight")) continue;
    ++sws;
    if (sect != "embed")
      errs.push_back(E("6.8.1", sect + "[" + std::to_string(j) + "]: an embedding state_weight outside the embed section (Q8)"));
    if (n.at("state_weight").is_string() && n.at("state_weight").get<std::string>().find('{') != std::string::npos)
      errs.push_back(E("6.8.1", "state_weight names its tensor in full (Q8)"));
  }
  if (sws > 1) errs.push_back(E("6.8.1", "more than one embedding with a state_weight (Q8)"));
  std::map<std::string, const json*> slots;
  if (get(blk, "state").is_array())
    for (const auto& s : blk.at("state"))
      if (s.is_object()) slots[show(get(s, "name"))] = &s;
  std::map<std::string, std::set<std::string>> masks;
  auto any_of = [](const std::optional<std::vector<json>>& v) {
    if (!v) return false;
    for (const auto& x : *v)
      if (truthy(x)) return true;
    return false;
  };
  if (get(blk, "nodes").is_array())
    for (const auto& n : blk.at("nodes")) {
      if (!n.is_object() || op_of(n) != "attention") continue;
      const json m = n.contains("mask") ? n.at("mask") : json("causal");
      if (m != "causal" && m != "turn_state")
        errs.push_back(E("6.8.2", "attention mask " + q(show(m)) + " (causal or turn_state)"));
      if (m == "turn_state") {
        const json& slot = slots.count(show(get(n, "state"))) ? *slots.at(show(get(n, "state"))) : kObject();
        auto field = [&](const json& o, const char* k) { return layers(o.contains(k) ? o.at(k) : json(0), L); };
        if (any_of(field(n, "window")) || any_of(field(n, "sinks")) || any_of(field(slot, "window")) ||
            any_of(field(slot, "sinks")) || sws == 0)
          errs.push_back(E("6.8.2", "attention mask turn_state needs window 0, sinks 0 (every application, and its kv slot) "
                                    "and an embedding state_weight"));
      }
      masks[show(get(n, "state"))].insert(show(m));
    }
  if (get(blk, "state").is_array())
    for (const auto& st : blk.at("state")) {
      if (!st.is_object() || !st.contains("evict")) continue;
      const std::string name = show(get(st, "name"));
      if (st.at("evict") != "turn_state")
        errs.push_back(E("6.8.3", "kv state " + name + ": evict " + q(show(st.at("evict"))) + " (only turn_state is defined)"));
      else if (masks[name] != std::set<std::string>{"turn_state"})
        errs.push_back(E("6.8.3", "kv state " + name + ": evict turn_state needs every attention node reading it to mask "
                                  "turn_state"));
    }
}

// Rule 36 (revision 1.2): every value a node reads is written before it, in the order of section 3.5 (embed, the
// block as application 0, head; tokens and pos excepted); the embed section writes x and the head logits; under
// depth_route the block's nodes after it read only x, pos and values written before them in the same
// application, and the head only x and the embed section's values (section 7.4(b)).
void read_errors(const json& doc, Errs& errs) {
  auto names = [](const json& v) {
    std::vector<std::string> out;
    if (v.is_string()) out.push_back(v.get<std::string>());
    if (v.is_array())
      for (const auto& x : v)
        if (x.is_string()) out.push_back(x.get<std::string>());
    return out;
  };
  auto nodes_of = [](const json& doc, const char* sect) -> const json& {
    const json& n = get(get(doc, sect), "nodes");
    return n.is_array() ? n : kNull();
  };
  std::set<std::string> written, embed_written;
  auto check = [&](const std::string& sect, size_t j, const json& n, const std::set<std::string>& scope,
                   bool inputs_free, const char* section, const std::string& why) {
    for (const auto& v : names(get(n, "in")))
      if (!scope.count(v) && !(inputs_free && (v == "tokens" || v == "pos")))
        errs.push_back(E(std::string(section) + " (rule 36)", sect + "[" + std::to_string(j) + "] " +
                                                                    show(get(n, "op")) + " reads " + q(v) + ", " + why));
  };
  for (const char* sect : {"embed", "block", "head"})
    if (!get(get(doc, sect), "nodes").is_array()) return;  // rule 1 names the section
  const json& embed = nodes_of(doc, "embed");
  for (size_t j = 0; embed.is_array() && j < embed.size(); ++j) {
    check("embed", j, embed[j], written, true, "3.5", "which no node before it writes");
    for (const auto& o : names(get(embed[j], "out"))) written.insert(o);
  }
  if (!written.count("x")) errs.push_back(E("3.5 (rule 36)", "the embed section writes no 'x'"));
  embed_written = written;
  const json& block = nodes_of(doc, "block");
  size_t route = block.is_array() ? block.size() : 0;
  if (block.is_array() && !block.empty() && op_of(block[0]) == "depth_route") route = 0;  // else rule 20
  std::set<std::string> step = {"x", "pos"};  // what a recursion step sees (7.4(b))
  for (size_t j = 0; block.is_array() && j < block.size(); ++j) {
    if (j <= route) {
      check("block", j, block[j], written, true, "3.5", "which no node before it writes (application 0)");
      if (j == route)  // depth_route's out is written before the step's nodes (reading it is rule 17a's)
        for (const auto& o : names(get(block[j], "out"))) step.insert(o);
    } else {
      check("block", j, block[j], step, false, "7.4",
            "outside what a depth_route step sees (x, pos and the values written before it in the application)");
      for (const auto& o : names(get(block[j], "out"))) step.insert(o);
    }
    for (const auto& o : names(get(block[j], "out"))) written.insert(o);
  }
  const bool mor = block.is_array() && route < block.size();
  std::set<std::string> head_scope = mor ? embed_written : written;
  if (mor) head_scope.insert("x");
  const json& head = nodes_of(doc, "head");
  bool logits = false;
  for (size_t j = 0; head.is_array() && j < head.size(); ++j) {
    check("head", j, head[j], head_scope, true, mor ? "7.4" : "3.5",
          mor ? "a block value other than x (under depth_route the head reads x and the embed section's values)"
              : "which no node before it writes");
    for (const auto& o : names(get(head[j], "out"))) {
      head_scope.insert(o);
      logits = logits || o == "logits";
    }
  }
  if (!logits) errs.push_back(E("3.5 (rule 36)", "the head writes no 'logits'"));
}

// ---------------------------------------------------------------- the container rules (24, 26-28)

std::optional<double> meta_num(const GGUFData& g, const std::string& key) {
  auto it = g.metadata.find(key);
  if (it == g.metadata.end()) return std::nullopt;
  return std::visit(
      [](const auto& v) -> std::optional<double> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
          return static_cast<double>(v);
        else
          return std::nullopt;
      },
      it->second);
}

std::optional<std::string> meta_str(const GGUFData& g, const std::string& key) {
  auto it = g.metadata.find(key);
  if (it == g.metadata.end()) return std::nullopt;
  if (const auto* s = std::get_if<std::string>(&it->second)) return *s;
  return std::nullopt;
}

std::string num(double v) {
  if (v == static_cast<double>(static_cast<int64_t>(v))) return std::to_string(static_cast<int64_t>(v));
  return json(v).dump();
}

std::string ne_str(const std::vector<uint64_t>& s) {
  std::string out;
  for (uint64_t x : s) out += (out.empty() ? "" : ", ") + std::to_string(x);
  return "[" + out + "]";
}

void container_errors(const json& doc, const GGUFData& g, Errs& errs) {
  // Rule 24: the file's paperkiln.ir_version, a uint32 equal to its embedded document's ir_version.
  if (const auto emb = meta_str(g, "paperkiln.ir")) {
    json e;
    // Rule 35 governs the document read (the Interpreter's parse_ir_document): a file whose own document holds a
    // duplicate name loads with another document in its place, as the reference validator accepts it.
    try {
      e = json::parse(*emb);
    } catch (const std::exception&) {
      errs.push_back(E("3.1", "paperkiln.ir is not a JSON document"));
    }
    const auto it = g.metadata.find("paperkiln.ir_version");
    const json& v = get(e, "ir_version");
    if (it == g.metadata.end() || !std::holds_alternative<uint32_t>(it->second) || !is_int(v) ||
        static_cast<int64_t>(std::get<uint32_t>(it->second)) != v.get<int64_t>())
      errs.push_back(E("3.1", "paperkiln.ir_version " +
                                  (it == g.metadata.end() ? std::string("None")
                                   : std::holds_alternative<uint32_t>(it->second)
                                       ? std::to_string(std::get<uint32_t>(it->second))
                                       : std::string("(not a uint32)")) +
                                  " does not match the document's ir_version " + show(v)));
  }
  // Rule 30: paperkiln.ir_sha256, when present, is a string equal to the digest of the document embedded in
  // the file (section 3.10), whatever document is run in its place (EMBER_IR, the adapter's "ir").
  if (const auto key = g.metadata.find("paperkiln.ir_sha256"); key != g.metadata.end()) {
    const auto* want = std::get_if<std::string>(&key->second);
    const auto emb = meta_str(g, "paperkiln.ir");
    if (!want) {
      errs.push_back(E("3.10", "paperkiln.ir_sha256 is not a string"));
    } else if (!emb) {
      errs.push_back(E("3.10", "paperkiln.ir_sha256 is present but the file embeds no paperkiln.ir"));
    } else {
      const DigestResult d = ir_digest(*emb);
      if (!d.ok)
        errs.push_back(E("3.10", "paperkiln.ir_sha256 is present but the embedded document has no digest (" + d.refused + ")"));
      else if (d.sha256 != *want)
        errs.push_back(E("3.10", "paperkiln.ir_sha256 " + *want + " is not the embedded document's digest " + d.sha256));
    }
  }
  auto ne =[&](const std::string& name) -> const std::vector<uint64_t>* {
    auto it = g.tensor_infos_map.find(name);
    return it == g.tensor_infos_map.end() ? nullptr : &it->second.shape;
  };
  const std::string arch = meta_str(g, "general.architecture").value_or("");
  const json& model = get(doc, "model");
  const json& blk = get(doc, "block");
  // Rule 26: the depth_route router's ne is {d, n_rec}.
  const json& nodes = get(blk, "nodes");
  if (nodes.is_array() && !nodes.empty() && op_of(nodes[0]) == "depth_route" && get(nodes[0], "weight").is_string()) {
    const json& r = nodes[0];
    if (const auto* got = ne(r.at("weight").get<std::string>())) {
      const bool ok = got->size() == 2 && is_int(get(model, "d")) && is_int(get(r, "n_rec")) &&
                      static_cast<int64_t>((*got)[0]) == model.at("d").get<int64_t>() &&
                      static_cast<int64_t>((*got)[1]) == r.at("n_rec").get<int64_t>();
      if (!ok)
        errs.push_back(E("7.4", "depth_route: " + r.at("weight").get<std::string>() + " has ne " + ne_str(*got) +
                                    ", not [d, n_rec] = [" + show(get(model, "d")) + ", " + show(get(r, "n_rec")) + "]"));
    }
  }
  // Rule 27: under weight_index, architecture paperkiln and <arch>.block_count = the stored layers.
  const json& wmap = get(get(blk, "weight_index"), "per_layer");
  bool wints = wmap.is_array() && !wmap.empty();
  for (size_t i = 0; wints && i < wmap.size(); ++i) wints = is_int(wmap[i]);
  if (wints) {
    int64_t mx = 0;
    for (const auto& x : wmap) mx = std::max(mx, x.get<int64_t>());
    if (arch != "paperkiln") errs.push_back(E("7.1", "block.weight_index needs architecture paperkiln, not " + (arch.empty() ? std::string("None") : arch)));
    const auto bc = meta_num(g, arch + ".block_count");
    if (!bc || *bc != static_cast<double>(mx + 1))
      errs.push_back(E("7.1", arch + ".block_count " + (bc ? num(*bc) : std::string("None")) + " != the " +
                                  std::to_string(mx + 1) + " stored layers weight_index uses"));
  }
  // Rule 28: state tokens.
  const auto k = meta_num(g, "paperkiln.state_tokens.k");
  std::vector<std::string> sws;
  const json& enodes = get(get(doc, "embed"), "nodes");
  if (enodes.is_array())
    for (const auto& n : enodes)
      if (op_of(n) == "embedding" && n.contains("state_weight")) sws.push_back(show(n.at("state_weight")));
  if ((k && *k > 0) != !sws.empty())
    errs.push_back(E("6.8.1", "paperkiln.state_tokens.k " + (k ? num(*k) : std::string("None")) +
                                  " without an embedding state_weight, or the reverse"));
  for (const auto& sw : sws) {
    const auto* got = ne(sw);
    if (!got) {
      errs.push_back(E("6.8.1", "embedding state_weight " + sw + " is not a tensor of the file"));
    } else if (!(got->size() == 2 && is_int(get(model, "d")) && k &&
                 static_cast<int64_t>((*got)[0]) == model.at("d").get<int64_t>() &&
                 static_cast<double>((*got)[1]) == *k)) {
      errs.push_back(E("6.8.1", "embedding state_weight " + sw + " has ne " + ne_str(*got) +
                                    ", not [d, paperkiln.state_tokens.k] = [" + show(get(model, "d")) + ", " +
                                    (k ? num(*k) : std::string("None")) + "]"));
    }
  }
  if (!sws.empty()) {
    if (arch != "paperkiln") errs.push_back(E("8.2", "state tokens need architecture paperkiln, not " + (arch.empty() ? std::string("None") : arch)));
    static const std::map<std::string, std::string> kMarkers = {
        {"eot", "<|endoftext|>"}, {"user", "<|user|>"}, {"assistant", "<|assistant|>"}};  // paperkiln/1
    const json& sp = get(get(doc, "tokenizer"), "specials");
    for (const char* role : {"eot", "user", "assistant"}) {
      const json& id = get(sp, role);
      if (!is_int(id)) {
        errs.push_back(E("8.2", std::string("state tokens: the tokenizer names no ") + role + " id (the template needs it)"));
        continue;
      }
      const int64_t i = id.get<int64_t>();
      const auto& toks = g.tokenizer_tokens;
      if (!toks.empty() && !(0 <= i && i < static_cast<int64_t>(toks.size()) && toks[i] == kMarkers.at(role)))
        errs.push_back(E("8.2", std::string("state tokens: specials.") + role + " = " + std::to_string(i) + " is the token " +
                                    (0 <= i && i < static_cast<int64_t>(toks.size()) ? q(toks[i]) : std::string("None")) +
                                    ", not " + q(kMarkers.at(role))));
    }
    if (const auto tmpl = meta_str(g, "paperkiln.chat_template")) {
      std::string name = "None";
      try {
        const json rec = json::parse(*tmpl);
        const std::string nm = show(get(rec, "name"));
        name = nm.find('/') == std::string::npos ? nm + "/" + show(get(rec, "version")) : nm;
      } catch (const std::exception&) {
      }
      if (name != "paperkiln/1") errs.push_back(E("8.2", "state tokens are defined over paperkiln/1 only, not " + name));
    }
  }
  // Rule 39 (container): each application's position_bias table is ne {heads, B}, B greater than every bucket.
  const int L = is_int(get(blk, "repeat")) ? blk.at("repeat").get<int>() : 0;
  const json& wmi = get(get(blk, "weight_index"), "per_layer");
  for (const auto& [sect, j, np] : sections(doc)) {
    const json& n = *np;
    const json& pb = get(n, "position_bias");
    if (op_of(n) != "attention" || !pb.is_object() || !get(pb, "weight").is_string() || !get(pb, "buckets").is_array())
      continue;
    int64_t top = -1;
    for (const auto& b : pb.at("buckets"))
      if (is_int(b)) top = std::max(top, b.get<int64_t>());
    const auto heads = layers(get(n, "heads"), std::max(L, 1));
    for (int i = 0; i < (sect == "block" ? L : 1); ++i) {
      std::string w = pb.at("weight").get<std::string>();
      const size_t at = w.find("{i}");
      if (at != std::string::npos) {
        const bool mapped = wmi.is_array() && i < static_cast<int>(wmi.size()) && is_int(wmi[i]) && !n.contains("tensor_index");
        w.replace(at, 3, std::to_string(mapped ? wmi[i].get<int64_t>() : i));
      }
      const auto* got = ne(w);
      if (!got || !heads || !is_int((*heads)[i])) continue;  // rule 25 names a missing tensor
      const int64_t H = (*heads)[i].get<int64_t>();
      if (!(got->size() == 2 && static_cast<int64_t>((*got)[0]) == H && static_cast<int64_t>((*got)[1]) > top)) {
        errs.push_back(E("5.15 (rule 39)", sect + "[" + std::to_string(j) + "] attention position_bias " + q(w) +
                                               (sect == "block" ? " (application " + std::to_string(i) + ")" : "") +
                                               " has ne " + ne_str(*got) + ", not {heads, B} = {" + std::to_string(H) +
                                               ", B} with B greater than every bucket (" + std::to_string(top) + ")"));
        break;
      }
    }
  }
}

// ---------------------------------------------------------------- rule 38 (revision 1.3): the architecture

// A document as rule 38 compares it (IR_SPEC 3.1): "_" members gone; the informative keys and attributes gone;
// optional attributes at their section 3.5 defaults gone; values renamed in the order they are written (tokens and
// pos kept) and slots in the order nodes name them, with what each name the block reads before writing it holds
// after the block; add's and mul's operands in one order; every number the f32 it rounds to; a linear's weight
// the tensor it reads (its tied_to resolved against the file's tensors).
json rule38_form(const json& doc, const GGUFData& g) {
  std::function<json(const json&)> strip = [&](const json& v) -> json {
    if (v.is_object()) {
      json o = json::object();
      for (const auto& kv : v.items())
        if (kv.key().empty() || kv.key()[0] != '_') o[kv.key()] = strip(kv.value());
      return o;
    }
    if (v.is_array()) {
      json a = json::array();
      for (const auto& x : v) a.push_back(strip(x));
      return a;
    }
    if (v.is_number()) return json(static_cast<double>(static_cast<float>(v.get<double>())));
    return v;
  };
  const json d = strip(doc);
  std::map<std::string, std::string> cur, slot;
  int next = 0;
  auto read = [&](const std::string& nm) {
    if (nm == "tokens" || nm == "pos") return nm;
    auto it = cur.find(nm);
    return it == cur.end() ? "?" + nm : it->second;
  };
  auto has_tensor = [&](std::string nm) {
    const size_t at = nm.find("{i}");
    if (at != std::string::npos) nm.replace(at, 3, "0");
    return g.tensor_infos_map.count(nm) > 0;
  };
  auto num_is = [](const json& v, double x) { return v.is_number() && v.get<double>() == static_cast<double>(static_cast<float>(x)); };
  auto node_form = [&](json n, bool in_block, std::vector<std::string>* carried, std::set<std::string>* written) {
    const std::string op = n.value("op", "");
    std::vector<std::string> ins;
    if (n.contains("in") && n.at("in").is_array())
      for (const auto& x : n.at("in"))
        if (x.is_string()) {
          const std::string nm = x.get<std::string>();
          if (in_block && carried && written && !written->count(nm) && nm != "tokens" && nm != "pos" &&
              std::find(carried->begin(), carried->end(), nm) == carried->end())
            carried->push_back(nm);
          ins.push_back(read(nm));
        }
    if ((op == "add" || op == "mul") && ins.size() == 2 && ins[1] < ins[0]) std::swap(ins[0], ins[1]);
    n["in"] = ins;
    if (n.contains("state") && n.at("state").is_string()) {
      const std::string st = n.at("state").get<std::string>();
      if (!slot.count(st)) slot[st] = "s" + std::to_string(slot.size());
      n["state"] = slot[st];
    }
    // informative attributes and optional ones at their defaults (IR_SPEC 3.5)
    if (op == "attention") {
      n.erase("softmax");
      if (n.value("mask", json("causal")) == "causal") n.erase("mask");
    }
    if (op == "moe_ffn") n.erase("routing");
    if ((op == "linear" || op == "scale") && n.contains("bias") && n.at("bias").is_null()) n.erase("bias");
    if (op == "linear") {
      if (n.contains("tied_to") && n.at("tied_to").is_string() &&
          !(n.contains("weight") && n.at("weight").is_string() && has_tensor(n.at("weight").get<std::string>())))
        n["weight"] = n.at("tied_to");
      n.erase("tied_to");
    }
    if (op == "rmsnorm" && n.contains("weight_offset") && num_is(n.at("weight_offset"), 0.0)) n.erase("weight_offset");
    if (op == "rope") {
      if (n.contains("kv_heads") && n.at("kv_heads") == get(n, "heads")) n.erase("kv_heads");
      if (n.contains("n_rot") && n.at("n_rot") == get(n, "head_dim")) n.erase("n_rot");
      if (n.contains("freq_scale") && num_is(n.at("freq_scale"), 1.0)) n.erase("freq_scale");
    }
    if (op == "depth_attend" && n.contains("eps") && num_is(n.at("eps"), 1e-5)) n.erase("eps");
    if (op == "embedding" && n.contains("state_weight") && n.at("state_weight").is_null()) n.erase("state_weight");
    if (op == "conv1d_causal" && n.contains("bias") && n.at("bias").is_null()) n.erase("bias");
    if (op == "selective_scan") {
      if (n.contains("d") && n.at("d").is_null()) n.erase("d");
      if (n.value("dt_softplus", json(true)) == json(true)) n.erase("dt_softplus");
    }
    json outs = json::array();
    const json& o = get(n, "out");
    std::vector<std::string> on;
    if (o.is_string()) on.push_back(o.get<std::string>());
    if (o.is_array())
      for (const auto& x : o)
        if (x.is_string()) on.push_back(x.get<std::string>());
    for (const auto& nm : on) {
      const std::string v = "v" + std::to_string(next++);
      cur[nm] = v;
      if (written) written->insert(nm);
      outs.push_back(v);
    }
    n["out"] = o.is_string() && outs.size() == 1 ? outs[0] : outs;
    return n;
  };
  json out = json::object();
  json model = get(d, "model");
  if (model.is_object() && model.value("dtype", json("f32")) == "f32") model.erase("dtype");
  out["model"] = model;
  json embed = json::array();
  for (const auto& n : get(get(d, "embed"), "nodes").is_array() ? d.at("embed").at("nodes") : json::array())
    embed.push_back(node_form(n, false, nullptr, nullptr));
  out["embed"] = embed;
  const std::map<std::string, std::string> before = cur;
  std::vector<std::string> carried;
  std::set<std::string> written;
  json block = json::array();
  const json& bl = get(d, "block");
  for (const auto& n : get(bl, "nodes").is_array() ? bl.at("nodes") : json::array())
    block.push_back(node_form(n, true, &carried, &written));
  json carry = json::array();
  for (const auto& nm : carried) {
    const auto b = before.find(nm);
    carry.push_back(json::array({b == before.end() ? "?" + nm : b->second, read(nm)}));
  }
  std::sort(carry.begin(), carry.end());
  json slots = json::array();
  if (get(bl, "state").is_array())
    for (json s : bl.at("state")) {
      const std::string nm = show(get(s, "name"));
      s["name"] = slot.count(nm) ? slot[nm] : "?" + nm;
      if (get(s, "kind") == "kv" && s.value("dtype", json("f32")) == "f32") s.erase("dtype");
      slots.push_back(s);
    }
  std::sort(slots.begin(), slots.end(), [](const json& a, const json& b) { return a.at("name") < b.at("name"); });
  out["block"] = {{"repeat", get(bl, "repeat")}, {"nodes", block}, {"state", slots}, {"carried", carry}};
  if (bl.is_object() && bl.contains("weight_index")) out["block"]["weight_index"] = bl.at("weight_index");
  json head = json::array();
  for (const auto& n : get(get(d, "head"), "nodes").is_array() ? d.at("head").at("nodes") : json::array())
    head.push_back(node_form(n, false, nullptr, nullptr));
  out["head"] = head;
  return out;
}

// The first place two rule-38 forms differ ("" when equal), as path: a against b.
std::string first_difference(const json& a, const json& b, const std::string& path) {
  if (a.type() != b.type() && !(a.is_number() && b.is_number()))
    return path + ": " + short_dump(a) + " against " + short_dump(b);
  if (a.is_object()) {
    std::set<std::string> keys;
    for (const auto& kv : a.items()) keys.insert(kv.key());
    for (const auto& kv : b.items()) keys.insert(kv.key());
    for (const auto& k : keys) {
      if (!a.contains(k)) return path + "." + k + ": absent against " + short_dump(b.at(k));
      if (!b.contains(k)) return path + "." + k + ": " + short_dump(a.at(k)) + " against absent";
      const std::string d = first_difference(a.at(k), b.at(k), path + "." + k);
      if (!d.empty()) return d;
    }
    return "";
  }
  if (a.is_array()) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
      const std::string d = first_difference(a[i], b[i], path + "[" + std::to_string(i) + "]");
      if (!d.empty()) return d;
    }
    if (a.size() != b.size())
      return path + ": " + std::to_string(a.size()) + " entries against " + std::to_string(b.size());
    return "";
  }
  if (a.is_number() && b.is_number()) return a.get<double>() == b.get<double>() ? "" : path + ": " + a.dump() + " against " + b.dump();
  return a == b ? "" : path + ": " + short_dump(a) + " against " + short_dump(b);
}

void architecture_errors(const GGUFData& g, Errs& errs) {
  const auto emb = meta_str(g, "paperkiln.ir");
  if (!emb) return;  // a third-party file: no document to compare
  const std::string arch = meta_str(g, "general.architecture").value_or("");
  if (arch == "paperkiln") return;
  const std::string tail = "; a file whose model is not that architecture's graph has general.architecture "
                           "\"paperkiln\", which no llama.cpp release builds a graph for";
  if (!has_ir_builder(arch)) {
    errs.push_back(E("3.1 (rule 38)", "general.architecture " + q(arch) +
                                          " is neither \"paperkiln\" nor an architecture of section 9.3" + tail));
    return;
  }
  json doc;
  try {
    doc = json::parse(*emb);
  } catch (const std::exception&) {
    return;  // rule 1 / 3.1 name it
  }
  json built;
  try {
    built = json::parse(build_ir_from_gguf(g, false));
  } catch (const std::exception& e) {
    errs.push_back(E("3.1 (rule 38)", "general.architecture " + q(arch) + ": its metadata builds no graph (" + e.what() +
                                          ")" + tail));
    return;
  }
  const std::string d = first_difference(rule38_form(doc, g), rule38_form(built, g), "document");
  if (!d.empty())
    errs.push_back(E("3.1 (rule 38)", "general.architecture " + q(arch) + ": the embedded document is not the " + arch +
                                          " graph its metadata describes (section 9.3; first difference, document "
                                          "against graph: " + d + ")" + tail));
}

}  // namespace

json parse_ir_document(const std::string& text) {
  // nlohmann's parse keeps the last of two members with one name; its callback sees every key as it is read,
  // so each open object keeps the names it has seen.
  std::vector<std::set<std::string>> open;
  std::string dup;
  int dup_depth = -1;
  const json::parser_callback_t cb = [&](int depth, json::parse_event_t ev, json& parsed) {
    if (ev == json::parse_event_t::object_start) {
      open.emplace_back();
    } else if (ev == json::parse_event_t::object_end) {
      if (!open.empty()) open.pop_back();
    } else if (ev == json::parse_event_t::key && dup_depth < 0 && !open.empty() && parsed.is_string() &&
               !open.back().insert(parsed.get<std::string>()).second) {
      dup = parsed.get<std::string>();
      dup_depth = depth;
    }
    return true;
  };
  json j = json::parse(text, cb);
  if (dup_depth >= 0)
    throw std::runtime_error(E("3.2 (rule 35)", "an object holds the member name " +
                                                    json(dup).dump(-1, ' ', false, json::error_handler_t::replace) +
                                                    " twice (nesting depth " + std::to_string(dup_depth) +
                                                    "): a document names each member of an object once"));
  return j;
}

namespace {

// The section and rule a schema failure breaks (IR_SPEC 10.1, revision 1.2): rule 1 only for what no other rule
// names (the types of values, the closed key sets of the document, its sections and its slots, the required
// sections); the version is rule 2, the model's fields rule 3, an op or an attribute its op does not define rule 4,
// an attribute value outside its closed set rule 5, a required attribute rule 8, causal not true rule 14, a slot's
// kind rule 11 and its fields rules 12 and 13, kv_from rule 22, weight_index rule 18 and a recurrence's
// declarations rule 31.
std::string schema_rule(const json& doc, const std::string& path, const std::string& msg) {
  static const std::regex kNode(R"(^doc\.(embed|block|head)\.nodes\[(\d+)\](?: \([A-Za-z0-9_]+\))?(.*)$)");
  static const std::regex kSlot(R"(^doc\.block\.state\[(\d+)\](.*)$)");
  static const std::set<std::string> kClosed = {"gelu.approx", "rope.pairing", "rope.rope_heads", "attention.mask",
                                                "moe_ffn.gate_norm", "moe_ffn.activation", "linear_attention.feature_map",
                                                "depth_route.route", "depth_route.gate"};
  static const std::set<std::string> kRecDecl = {"index", "inputs", "weights", "init", "let", "update", "readout"};
  const bool unknown_key = msg.rfind("has the key '", 0) == 0;
  const bool required = msg.rfind("lacks the required key '", 0) == 0;
  auto key_of = [&](const std::string& rest) {  // ".mask" or ".mask.per_layer[0]" -> "mask"
    if (rest.size() < 2 || rest[0] != '.') return std::string();
    const size_t e = rest.find_first_of(".[", 1);
    return rest.substr(1, e == std::string::npos ? std::string::npos : e - 1);
  };
  std::smatch m;
  if (std::regex_match(path, m, kNode)) {
    const json& nodes = get(get(doc, m[1].str()), "nodes");
    const size_t j = std::stoul(m[2].str());
    const std::string op = nodes.is_array() && j < nodes.size() ? op_of(nodes[j]) : "";
    const std::string rest = m[3].str(), key = key_of(rest);
    if (key == "position_bias")  // attention's (5.15, revision 1.3); a per-layer one breaks rule 6
      return nodes.is_array() && j < nodes.size() && is_pl(get(nodes[j], "position_bias")) ? "3.8 (rule 6)"
                                                                                           : "5.15 (rule 39)";
    const bool io = key == "in" || key == "out" || (required && (msg.find("'in'") != std::string::npos ||
                                                                 msg.find("'out'") != std::string::npos));
    if (io) return "3.5 (rule 7)";  // the arity: what a node reads and writes
    if (unknown_key || key == "op") return "3.5 (rule 4)";
    if (required && rest.empty()) return "3.5 (rule 8)";
    if (key == "tensor_index" || kClosed.count(op + "." + key)) return "3.5 (rule 5)";
    if (key == "causal" && (op == "attention" || op == "linear_attention"))
      return op == "attention" ? "5.15 (rule 14)" : "6.4 (rule 14)";
    if (op == "recurrence" && kRecDecl.count(key)) return "6.10 (rule 31)";
    return "3.5 (rule 1)";
  }
  if (std::regex_match(path, m, kSlot)) {
    const json& slots = get(get(doc, "block"), "state");
    const size_t j = std::stoul(m[1].str());
    const json& slot = slots.is_array() && j < slots.size() ? slots[j] : kNull();
    const std::string rest = m[2].str(), key = key_of(rest);
    if (unknown_key && get(slot, "kind") == "recurrent") return "6.3 (rule 13)";
    if (unknown_key) return msg.find("'kv_from'") != std::string::npos ? "7.6 (rule 22)" : "6.1 (rule 1)";
    if (key == "kind" || (required && msg.find("'kind'") != std::string::npos)) return "6.1 (rule 11)";
    if (get(slot, "kind") == "kv") {
      if (key == "dtype" || key == "evict" || key == "kv_rows") return "6.2 (rule 5)";
      return "6.2 (rule 12)";
    }
    if (get(slot, "kind") == "recurrent") return "6.3 (rule 13)";
    return "6.1 (rule 1)";
  }
  if (path.rfind("doc.ir_version", 0) == 0) return "2.3 (rule 2)";
  if (path.rfind("doc.model", 0) == 0) {  // the model's fields d, vocab, ctx and n_layers are rule 3
    bool field = false;
    for (const char* k : {"vocab", "d", "n_layers", "ctx"})
      field = field || key_of(path.substr(9)) == k || (required && msg.find(std::string("'") + k + "'") != std::string::npos);
    return field ? "3.3 (rule 3)" : "3.3 (rule 1)";
  }
  if (path.rfind("doc.block.weight_index", 0) == 0) return "7.1 (rule 18)";
  if (path.rfind("doc.block.repeat", 0) == 0) return "3.3 (rule 3)";
  return "3.2 (rule 1)";
}

}  // namespace

std::vector<std::string> ir_schema_errors(const json& doc) {
  Failures f;
  eval(schema(), doc, "doc", f);
  std::vector<std::string> out;
  std::set<std::string> seen;
  for (const auto& [p, m] : f) add_unique(out, seen, E(schema_rule(doc, p, m), p + " " + m));
  return out;
}

std::vector<std::string> ir_document_errors(const json& doc) {
  if (!doc.is_object()) return {E("10.1 rule 1", "the document is not a JSON object")};
  Errs errs = ir_schema_errors(doc);
  const json& v = get(doc, "ir_version");
  if (is_int(v) && v.get<int64_t>() != 0 && v.get<int64_t>() != 1)
    errs.push_back(E("2.3", "ir_version " + v.dump() + " is not supported (ember reads 0, 1)"));
  node_errors(doc, errs);
  state_errors(doc, errs);
  depth_route_errors(doc, errs);
  state_token_doc_errors(doc, errs);
  read_errors(doc, errs);
  std::vector<std::string> out;
  std::set<std::string> seen;
  for (const auto& m : errs) add_unique(out, seen, m);
  return out;
}

std::vector<std::string> ir_container_errors(const json& doc, const GGUFData& gguf) {
  Errs errs;
  container_errors(doc, gguf, errs);
  architecture_errors(gguf, errs);
  std::vector<std::string> out;
  std::set<std::string> seen;
  for (const auto& m : errs) add_unique(out, seen, m);
  return out;
}

void validate_ir(const json& doc, const GGUFData& gguf) {
  std::vector<std::string> errs = ir_document_errors(doc);
  if (doc.is_object())
    for (const auto& m : ir_container_errors(doc, gguf)) errs.push_back(m);
  if (errs.empty()) return;
  std::string msg;
  for (const auto& e : errs) msg += (msg.empty() ? "" : "; ") + e;
  throw std::runtime_error(msg);
}

}  // namespace ember_ir
