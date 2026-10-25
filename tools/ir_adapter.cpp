// ir_adapter: ember's IR interpreter behind paperkiln's conformance protocol (spec/IR_SPEC.md
// section 11.3), for tools/ir_conformance.py --adapter cmd:"ir_adapter".
//
//   ir_adapter [--conformance] [--log FILE]
//
// JSON Lines: one request per stdin line ({"id", "method", "params"}), one reply per stdout line in
// request order ({"id", "result"}, {"id", "error"} or {"id", "skip"}). Nothing else reaches stdout:
// the process's stdout is moved to stderr at start and replies go to the original descriptor.
// --log FILE writes ember's log there (off by default). CPU interpreter only ("backend": "cpu",
// "exact": true).
//
// Methods: hello, check, open, forward, truncate, snapshot, restore, close, digest, dequantize, and run_node
// for recurrence nodes (other ops and run_node_grad reply skip). A session owns one GGUF, one interpreter and
// the ids it has seen.
//   check / open: "ir" names a document that replaces the file's (ember's EMBER_IR); null reads the
//     GGUF's paperkiln.ir (or the IR ember builds from third-party metadata). EMBER_IR from the
//     environment is ignored. open's "evict": null as declared, false keeps every row
//     (EMBER_EVICT=0), true requires the document's "evict": "turn_state".
//   forward: ids at positions len .. len + n - 1, one call of the interpreter (all rows).
//   truncate: section 6.9. The interpreter drops the state past resume_point(n) and re-prefills
//     resume_point(n) .. n - 1 from the ids held (an evicted turn's start under eviction; the
//     nearest recurrent snapshot below n).
//   snapshot / restore: a copy of the whole sequence state (kv rows, state-token bookkeeping, MoR
//     row sets, recurrent tensors and snapshots) and the ids, by tag.
//   digest: ember's digest of the text (ir_digest.h; IR_SPEC 3.10), with its canonical form, or "refused".
//   dequantize: the f32 values of n elements stored as the bytes, by the interpreter's route
//     (dequant_span_to_f32, IR_SPEC 4.7): F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K.
//   run_node: a recurrence node (graph_ir_recurrence.h) as application 0 ({i} is 0), token by token over the
//     inputs' rows from the slot's initial state (or "state"'s tensors); outputs {out: [T, w]} and the slot's
//     tensors after the last row.
// Reports: logits and depth (forward only), kv_rows, kv_positions, recurrent. ember keeps one kv
// slot per block; kv_rows and kv_positions give, per application, the rows its cache holds after
// the call. ember evicts lazily: a finished segment's text rows go when the first position of a
// later segment is processed (the start of the next call into it), not at the end of the call
// that completes the state block (IR_SPEC.md 6.8.3 allows either).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gguf_parser.h"
#include "graph_ir.h"
#include "graph_ir_recurrence.h"
#include "ir_digest.h"
#include "logger.h"
#include "quantization.h"
#include "weight_management.h"

#ifdef _WIN32
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define fdopen _fdopen
#define fileno _fileno
#else
#include <unistd.h>
#endif

using json = nlohmann::json;

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value ? value : "");
#else
  if (value)
    setenv(name, value, 1);
  else
    unsetenv(name);
#endif
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

// A tensor as section 11.2 writes it: 8 hex digits per value, the binary32 bit pattern.
json tensor(const std::vector<int64_t>& shape, const float* v, size_t n) {
  static const char* hex = "0123456789abcdef";
  std::string s(n * 8, '0');
  for (size_t i = 0; i < n; ++i) {
    uint32_t b;
    std::memcpy(&b, v + i, 4);
    for (int d = 7; d >= 0; --d, b >>= 4) s[i * 8 + d] = hex[b & 15];
  }
  return json{{"shape", shape}, {"f32", s}};
}

// A tensor of section 11.2 as floats (and its shape).
std::vector<float> floats(const json& t, std::vector<int64_t>* shape = nullptr) {
  const std::string hex = t.at("f32").get<std::string>();
  if (hex.size() % 8) throw std::runtime_error("a tensor's f32 hex is not 8 digits per value");
  std::vector<float> v(hex.size() / 8);
  for (size_t i = 0; i < v.size(); ++i) {
    const uint32_t b = static_cast<uint32_t>(std::stoul(hex.substr(i * 8, 8), nullptr, 16));
    std::memcpy(&v[i], &b, 4);
  }
  if (shape) *shape = t.at("shape").get<std::vector<int64_t>>();
  return v;
}

std::vector<uint8_t> unhex(const std::string& h) {
  if (h.size() % 2) throw std::runtime_error("bytes: an odd number of hex digits");
  std::vector<uint8_t> out(h.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<uint8_t>(std::stoul(h.substr(2 * i, 2), nullptr, 16));
  return out;
}

// IR_SPEC 4.7's table: elements and bytes per block.
struct StoredType {
  GGMLType type;
  size_t elems, bytes;
};
const std::map<std::string, StoredType>& stored_types() {
  static const std::map<std::string, StoredType> t = {
      {"F32", {GGMLType::GGML_TYPE_F32, 1, 4}},       {"F16", {GGMLType::GGML_TYPE_F16, 1, 2}},
      {"BF16", {GGMLType::GGML_TYPE_BF16, 1, 2}},     {"Q8_0", {GGMLType::GGML_TYPE_Q8_0, 32, 34}},
      {"Q4_0", {GGMLType::GGML_TYPE_Q4_0, 32, 18}},   {"Q4_1", {GGMLType::GGML_TYPE_Q4_1, 32, 20}},
      {"Q5_0", {GGMLType::GGML_TYPE_Q5_0, 32, 22}},   {"Q5_1", {GGMLType::GGML_TYPE_Q5_1, 32, 24}},
      {"Q4_K", {GGMLType::GGML_TYPE_Q4_K, 256, 144}}, {"Q5_K", {GGMLType::GGML_TYPE_Q5_K, 256, 176}},
      {"Q6_K", {GGMLType::GGML_TYPE_Q6_K, 256, 210}}};
  return t;
}

struct Loaded {
  std::unique_ptr<GGUFData> gguf;  // outlives ip: weights kept quantised point into its map
  std::unique_ptr<ember_ir::Interpreter> ip;
};

// evict: null (as declared), false (keep every row) or true (the declaration required).
Loaded load(const json& p, const json& evict) {
  const std::string path = p.at("gguf").get<std::string>();
  Loaded L;
  L.gguf = std::make_unique<GGUFData>(load_gguf_meta(path, true));
  std::string ir;
  if (p.contains("ir") && !p.at("ir").is_null())
    ir = read_file(p.at("ir").get<std::string>());
  else
    ir = ember_ir::find_ir(*L.gguf);
  if (ir.empty()) throw std::runtime_error("no IR: the GGUF has no paperkiln.ir and ember runs it on the llama path");
  set_env("EMBER_EVICT", evict.is_boolean() && !evict.get<bool>() ? "0" : nullptr);
  try {
    L.ip = std::make_unique<ember_ir::Interpreter>(ir, *L.gguf, false);
  } catch (...) {
    set_env("EMBER_EVICT", nullptr);
    throw;
  }
  set_env("EMBER_EVICT", nullptr);
  if (evict.is_boolean() && evict.get<bool>() && !L.ip->evict_declared())
    throw std::runtime_error("evict true: the document's kv slot declares no \"evict\": \"turn_state\"");
  return L;
}

struct Snap {
  std::vector<int> ids;
  ember_ir::Interpreter::State state;
};

struct Session {
  Loaded m;
  std::vector<int> ids;
  std::map<std::string, Snap> snaps;
};

struct Skip {
  std::string why;
};

class Adapter {
 public:
  json handle(const std::string& method, const json& p) {
    if (method == "hello") return hello();
    if (method == "check") {
      try {
        load(p, nullptr);
      } catch (const std::exception& e) {
        return {{"accepted", false}, {"message", e.what()}};
      }
      return {{"accepted", true}};
    }
    if (method == "open") {
      Session s;
      s.m = load(p, p.value("evict", json(nullptr)));
      const std::string id = "s" + std::to_string(++next_);
      json r{{"session", id}, {"vocab", s.m.ip->vocab()}, {"applications", s.m.ip->n_layers()},
             {"state_k", s.m.ip->state_k()}};
      sessions_.emplace(id, std::move(s));
      return r;
    }
    if (method == "close") {
      if (!sessions_.erase(sid(p))) throw std::runtime_error("no session " + sid(p));
      return json::object();
    }
    if (method == "forward") return forward(session(p), p);
    if (method == "truncate") return truncate(session(p), p);
    if (method == "snapshot") {
      Session& s = session(p);
      s.snaps[p.at("tag").get<std::string>()] = Snap{s.ids, s.m.ip->save_state()};
      return {{"len", s.ids.size()}};
    }
    if (method == "restore") {
      Session& s = session(p);
      const auto it = s.snaps.find(p.at("tag").get<std::string>());
      if (it == s.snaps.end()) throw std::runtime_error("no snapshot " + p.at("tag").dump());
      check_reports(*s.m.ip, p, false);
      s.m.ip->load_state(it->second.state);
      s.ids = it->second.ids;
      json r{{"len", s.ids.size()}};
      reports(s, p, r, false, {});
      return r;
    }
    if (method == "run_node") return run_node(p);
    if (method == "run_node_grad") throw Skip{"ember is an inference engine: no backward (IR_SPEC 6.10.5 is for training engines)"};
    if (method == "digest") {
      const ember_ir::DigestResult d = ember_ir::ir_digest(p.at("text").get<std::string>(), p.value("prose", false));
      if (!d.ok) return {{"refused", d.refused}};
      return {{"sha256", d.sha256}, {"canonical", d.canonical}};
    }
    if (method == "dequantize") return dequantize(p);
    throw std::runtime_error("unknown method '" + method + "'");
  }

 private:
  static json hello() {
    return {{"engine", "ember.cpp"},
            {"version", "ir_adapter 2 (IR versions 0 and 1; spec revision 1.1)"},
            {"backend", "cpu"},
            {"exact", true},
            {"profiles", {"core", "state", "recursion", "thirdparty", "recurrence"}},
            {"methods", {"hello", "check", "open", "forward", "truncate", "snapshot", "restore", "close", "digest",
                         "dequantize", "run_node"}}};
  }

  static json dequantize(const json& p) {
    const std::string name = p.at("type").get<std::string>();
    const auto it = stored_types().find(name);
    if (it == stored_types().end()) throw std::runtime_error("dequantize: type " + name + " is not one ember reads");
    const StoredType& st = it->second;
    const size_t n = p.at("n").get<size_t>();
    if (n % st.elems)
      throw std::runtime_error("dequantize: " + std::to_string(n) + " elements of " + name + " is not whole blocks of " +
                               std::to_string(st.elems));
    const std::vector<uint8_t> raw = unhex(p.at("bytes").get<std::string>());
    if (raw.size() != n / st.elems * st.bytes)
      throw std::runtime_error("dequantize: " + std::to_string(raw.size()) + " bytes for " + std::to_string(n) +
                               " elements of " + name);
    std::vector<float> v(n);
    if (st.type == GGMLType::GGML_TYPE_F32)
      std::memcpy(v.data(), raw.data(), raw.size());
    else if (!dequant_span_to_f32(st.type, raw.data(), 0, n, v.data()))
      throw std::runtime_error("dequantize: no route for " + name);
    return {{"values", tensor({static_cast<int64_t>(n)}, v.data(), v.size())}};
  }

  // A recurrence node alone, as application 0 (IR_SPEC 11.3 run_node; 6.10.3).
  static json run_node(const json& p) {
    const json& node = p.at("node");
    if (node.value("op", "") != "recurrence")
      throw Skip{"run_node: ember's adapter runs recurrence nodes only (whole-model methods for the other ops)"};
    const ember_ir::rec::Recurrence prog = ember_ir::rec::Recurrence::compile(node);
    std::vector<std::string> ins;
    if (node.at("in").is_string())
      ins.push_back(node.at("in").get<std::string>());
    else
      ins = node.at("in").get<std::vector<std::string>>();
    std::vector<std::vector<float>> in_vals;
    int64_t T = -1;
    for (size_t k = 0; k < ins.size(); ++k) {
      std::vector<int64_t> shape;
      in_vals.push_back(floats(p.at("inputs").at(ins[k]), &shape));
      const int64_t rows = shape.empty() ? 0 : shape[0];
      if (T >= 0 && rows != T) throw std::runtime_error("run_node: the inputs have different row counts");
      T = rows;
      if (rows > 0 && in_vals.back().size() != static_cast<size_t>(rows) * prog.input_widths()[k])
        throw std::runtime_error("IR_SPEC 6.10 (rule 34): recurrence input '" + ins[k] + "' is " +
                                 std::to_string(in_vals.back().size() / rows) + " wide; its declaration is " +
                                 std::to_string(prog.input_widths()[k]));
    }
    std::vector<std::vector<float>> w_vals;
    std::vector<const float*> w;
    for (const auto& wd : prog.weights()) {
      std::string name = wd.tensor;
      const auto at = name.find("{i}");
      if (at != std::string::npos) name.replace(at, 3, "0");
      if (!p.at("weights").contains(name)) throw std::runtime_error("run_node: no weight '" + name + "'");
      w_vals.push_back(floats(p.at("weights").at(name)));
      if (w_vals.back().size() != wd.numel)
        throw std::runtime_error("IR_SPEC 6.10 (rule 34): recurrence weight " + wd.decl + " reads '" + name +
                                 "', which has " + std::to_string(w_vals.back().size()) +
                                 " elements; its declaration has " + std::to_string(wd.numel));
    }
    for (const auto& v : w_vals) w.push_back(v.data());
    const std::string slot = node.at("state").get<std::string>();
    std::vector<std::vector<float>> st_vals;
    std::vector<float*> st;
    for (const auto& s : prog.state()) st_vals.emplace_back(s.numel);
    for (auto& v : st_vals) st.push_back(v.data());
    prog.init(w, st);
    const json given = p.contains("state") ? p.at("state") : json();
    if (given.is_object() && given.contains(slot)) {  // the slot's declaration and its tensors before the call
      const json& g = given.at(slot);
      if (g.contains("decl")) {
        const std::string why = prog.slot_errors(g.at("decl"));
        if (!why.empty()) throw std::runtime_error("IR_SPEC 6.10 (rule 33): " + why);
      }
      for (size_t s = 0; s < prog.state().size(); ++s)
        if (g.contains("tensors") && g.at("tensors").contains(prog.state()[s].name)) {
          const std::vector<float> v = floats(g.at("tensors").at(prog.state()[s].name));
          if (v.size() != st_vals[s].size())
            throw std::runtime_error("run_node: state tensor " + prog.state()[s].name + " has the wrong size");
          std::copy(v.begin(), v.end(), st_vals[s].begin());
        }
    }
    const size_t ow = prog.out_width();
    const int64_t rows = T < 0 ? 0 : T;
    std::vector<float> out(static_cast<size_t>(rows) * ow);
    std::vector<const float*> row(ins.size());
    for (int64_t t = 0; t < rows; ++t) {
      for (size_t k = 0; k < ins.size(); ++k) row[k] = in_vals[k].data() + static_cast<size_t>(t) * prog.input_widths()[k];
      prog.step(row, w, st, out.data() + static_cast<size_t>(t) * ow);
    }
    json state = json::object();
    for (size_t s = 0; s < prog.state().size(); ++s)
      state[prog.state()[s].name] = tensor(prog.state()[s].shape, st_vals[s].data(), st_vals[s].size());
    json outputs = json::object();
    outputs[node.at("out").get<std::string>()] = tensor({rows, static_cast<int64_t>(ow)}, out.data(), out.size());
    json r;
    r["outputs"] = outputs;
    r["state"] = json::object();
    r["state"][slot] = state;
    return r;
  }
  static std::string sid(const json& p) { return p.at("session").get<std::string>(); }
  Session& session(const json& p) {
    auto it = sessions_.find(sid(p));
    if (it == sessions_.end()) throw std::runtime_error("no session " + sid(p));
    return it->second;
  }

  // Checked before a call changes the session: an unknown report is an error, a report that is
  // not this method's is an error, one the model cannot give skips the call.
  static void check_reports(const ember_ir::Interpreter& ip, const json& p, bool fwd) {
    if (!p.contains("report") || p.at("report").is_null()) return;
    for (const auto& want : p.at("report")) {
      const std::string k = want.get<std::string>();
      if (k == "logits" || k == "depth") {
        if (!fwd) throw std::runtime_error("'" + k + "' is a forward report");
        if (k == "depth" && !ip.has_depth_route()) throw Skip{"depth: the document has no depth_route"};
      } else if (k != "kv_rows" && k != "kv_positions" && k != "recurrent") {
        throw std::runtime_error("unknown report '" + k + "'");
      }
    }
  }

  json forward(Session& s, const json& p) {
    check_reports(*s.m.ip, p, true);
    const std::vector<int> ids = p.at("ids").get<std::vector<int>>();
    const int len = static_cast<int>(s.ids.size());
    if (s.m.ip->seq_len() != len)
      throw std::runtime_error("session state covers " + std::to_string(s.m.ip->seq_len()) + " positions, the ids " +
                               std::to_string(len));
    const std::vector<float> logits = s.m.ip->forward(ids, len, true);
    s.ids.insert(s.ids.end(), ids.begin(), ids.end());
    json r{{"len", s.ids.size()}};
    reports(s, p, r, true, logits, ids.size());
    return r;
  }

  json truncate(Session& s, const json& p) {
    check_reports(*s.m.ip, p, false);
    const int n = p.at("n").get<int>();
    const int len = static_cast<int>(s.ids.size());
    if (n < 0 || n > len) throw std::runtime_error("truncate to " + std::to_string(n) + " of " + std::to_string(len));
    ember_ir::Interpreter& ip = *s.m.ip;
    const int r = ip.resume_point(n);
    ip.truncate(r);
    if (r < n) ip.forward(std::vector<int>(s.ids.begin() + r, s.ids.begin() + n), r, true);
    s.ids.resize(n);
    json out{{"len", n}};
    reports(s, p, out, false, {});
    return out;
  }

  // The reports a call asks for; a report the engine cannot give skips the call.
  void reports(Session& s, const json& p, json& r, bool fwd, const std::vector<float>& logits, size_t n = 0) {
    if (!p.contains("report") || p.at("report").is_null()) return;
    ember_ir::Interpreter& ip = *s.m.ip;
    for (const auto& want : p.at("report")) {
      const std::string k = want.get<std::string>();
      if (k == "logits" || k == "depth") {
        if (!fwd) throw std::runtime_error("'" + k + "' is a forward report");
        if (k == "logits") {
          const int64_t V = ip.vocab();
          if (logits.size() != n * static_cast<size_t>(V))
            throw std::runtime_error("logits: " + std::to_string(logits.size()) + " values for " + std::to_string(n) + " rows");
          r["logits"] = tensor({static_cast<int64_t>(n), V}, logits.data(), logits.size());
        } else {
          if (ip.last_depths().empty() && n > 0) throw Skip{"depth: the document has no depth_route"};
          r["depth"] = n > 0 ? ip.last_depths() : std::vector<int>{};
        }
      } else if (k == "kv_rows" || k == "kv_positions") {
        json slots = json::object();
        if (!ip.kv_slot().empty()) {
          json per = json::array();
          for (int a = 0; a < ip.n_layers(); ++a) {
            const std::vector<int> pos = ip.kv_positions(a);
            if (k == "kv_rows")
              per.push_back(pos.size());
            else
              per.push_back(pos);
          }
          slots[ip.kv_slot()] = per;
        }
        r[k] = slots;
      } else if (k == "recurrent") {
        json slots = json::object();
        for (const auto& t : ip.recurrent_tensors()) {
          if (!slots.contains(t.slot)) slots[t.slot] = json::array();
          json& per = slots[t.slot];
          for (int a = 0; a < ip.n_layers(); ++a) {
            if (static_cast<int>(per.size()) <= a) per.push_back(json::object());
            const std::vector<float> v = ip.recurrent_values(t.slot, t.tensor, a);
            per[a][t.tensor] = tensor(t.shape, v.data(), v.size());
          }
        }
        r["recurrent"] = slots;
      } else {
        throw std::runtime_error("unknown report '" + k + "'");
      }
    }
  }

  std::map<std::string, Session> sessions_;
  int next_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--conformance") continue;
    if (a == "--log" && i + 1 < argc) {
      Logger::set_logfile(argv[++i]);
      continue;
    }
    std::cerr << "usage: ir_adapter [--conformance] [--log FILE]  (JSON Lines on stdin/stdout, IR_SPEC.md 11.3)\n";
    return 2;
  }
  bool logging = false;
  for (int i = 1; i < argc; ++i) logging = logging || std::string(argv[i]) == "--log";
  if (!logging) Logger::set_level(Logger::Level::OFF);
  // Replies only on stdout: anything else the libraries print goes to stderr.
  std::fflush(stdout);
  const int out_fd = dup(fileno(stdout));
  FILE* out = out_fd >= 0 ? fdopen(out_fd, "w") : nullptr;
  if (!out || dup2(fileno(stderr), fileno(stdout)) < 0) {
    std::cerr << "ir_adapter: cannot separate stdout\n";
    return 1;
  }
  set_env("EMBER_IR", nullptr);  // "ir" in check / open is the only override
  Adapter ad;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    json reply;
    json req;
    try {
      req = json::parse(line);
    } catch (const std::exception& e) {
      reply = {{"id", nullptr}, {"error", std::string("bad request: ") + e.what()}};
    }
    if (reply.is_null()) {
      reply["id"] = req.value("id", json(nullptr));
      try {
        reply["result"] = ad.handle(req.at("method").get<std::string>(), req.value("params", json::object()));
      } catch (const Skip& s) {
        reply.erase("result");
        reply["skip"] = s.why;
      } catch (const std::exception& e) {
        reply.erase("result");
        reply["error"] = e.what();
      }
    }
    const std::string s = reply.dump(-1, ' ', false, json::error_handler_t::replace) + "\n";
    std::fwrite(s.data(), 1, s.size(), out);
    std::fflush(out);
  }
  return 0;
}
