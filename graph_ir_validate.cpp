// IR_SPEC section 10.1 at load (graph_ir_validate.h): a port of paperkiln's tools/ir_validate.py.
#include "graph_ir_validate.h"

#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <utility>

#include "graph_ir_schema.h"
#include "gguf_structs.h"

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

}  // namespace

std::vector<std::string> ir_schema_errors(const json& doc) {
  Failures f;
  eval(schema(), doc, "doc", f);
  std::vector<std::string> out;
  std::set<std::string> seen;
  for (const auto& [p, m] : f) add_unique(out, seen, E("10.1 rule 1", p + " " + m));
  return out;
}

std::vector<std::string> ir_document_errors(const json& doc) {
  if (!doc.is_object()) return {E("10.1 rule 1", "the document is not a JSON object")};
  std::vector<std::string> out;
  std::set<std::string> seen;
  for (const auto& m : ir_schema_errors(doc)) add_unique(out, seen, m);
  return out;
}

void validate_ir(const json& doc, const GGUFData& gguf) {
  (void)gguf;
  const std::vector<std::string> errs = ir_document_errors(doc);
  if (errs.empty()) return;
  std::string msg;
  for (const auto& e : errs) msg += (msg.empty() ? "" : "; ") + e;
  throw std::runtime_error(msg);
}

}  // namespace ember_ir
