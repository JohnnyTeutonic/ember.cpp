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
// Methods: hello, check, open, forward, truncate, snapshot, restore, close, digest, dequantize, render_chat,
// parse_reply, and run_node for recurrence, expr, attention, mla_attention, split and rope nodes (other ops and
// run_node_grad reply skip). A session owns one GGUF, one
// interpreter and the ids it has seen.
//   check / open: "ir" names a document that replaces the file's (ember's EMBER_IR); null reads the
//     GGUF's paperkiln.ir (or the IR ember builds from third-party metadata). EMBER_IR from the
//     environment is ignored. check with "gguf" null checks the document "ir" alone: the document rules
//     (graph_ir_validate.h ir_document_errors, with rule 35's parse), no container rule and no binding.
//     open's "evict": null as declared, false keeps every row (EMBER_EVICT=0), true requires the
//     document's "evict": "turn_state".
//   forward: ids at positions len .. len + n - 1, one call of the interpreter (all rows); empty ids are an error.
//   truncate: section 6.9, 0 <= n <= len (n = len changes nothing; n < 0 or n > len is an error). The
//     interpreter drops the state past resume_point(n) and re-prefills resume_point(n) .. n - 1 from the ids
//     held (an evicted turn's start under eviction; the nearest recurrent snapshot below n, or the start).
//   snapshot / restore: a copy of the whole sequence state (kv rows, state-token bookkeeping, MoR
//     row sets, recurrent tensors and snapshots) and the ids, by tag; a tag used again replaces its snapshot,
//     and restoring a tag never snapshotted is an error.
//   An error reply (or a skip) leaves the session as it was before the call (IR_SPEC 11.3): forward, truncate
//   and restore save the sequence state first and put it back when the call fails.
//   digest: ember's digest of the text (ir_digest.h; IR_SPEC 3.10), with its canonical form, or "refused".
//   dequantize: the f32 values of n elements stored as the bytes, by the interpreter's route
//     (dequant_span_to_f32, IR_SPEC 4.7): F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K.
//   render_chat: the text a conversation renders to under a template record (IR_SPEC 8.1.3, 8.1.6), through the
//     vendored chat_template.h ember's chat paths render with, or "refused" with its reason; "tools" (sent only
//     when a case has them) are the conversation's tool definitions, read with the request's member order.
//   parse_reply: a reply's ids split at the call id (IR_SPEC 8.1.6, 11.3), the split chat_reply.h's reply_of
//     makes for the server's /chat and /v1/chat/completions (chat::parse_reply), each part decoded by joining
//     its ids' texts from "tokens"; the calls in OpenAI's form (chat_reply.h tool_calls_json). The reported
//     content is the split's part 0 as decoded: reply_of then strips its ends and makes it valid UTF-8.
//   run_node: a recurrence node (graph_ir_recurrence.h) as application 0 ({i} is 0), token by token over the
//     inputs' rows from the slot's initial state (or "state"'s tensors); outputs {out: [T, w]} and the slot's
//     tensors after the last row. An expr node (graph_ir_expr.h): the interpreter's expression over its inputs
//     [T, w], their shapes checked as the interpreter checks them (rule 37); state {}. An attention node with section
//     11.2's defaults (its own slot, rows at positions 0 .. T-1, mask causal) through the interpreter's kernels,
//     position_bias and relative_kv included; state {} (a kv slot is not reported).
// Reports: logits and depth (forward only), kv_rows, kv_positions, recurrent; depth is [] for a document
// without depth_route, kv_rows / kv_positions {} without a kv slot, recurrent {} without one. ember keeps one kv
// slot per block; kv_rows and kv_positions give, per application, the rows its cache holds after
// the call. ember evicts lazily: a finished segment's text rows go when the first position of a
// later segment is processed (the start of the next call into it), not at the end of the call
// that completes the state block (IR_SPEC.md 6.8.3 allows either).
#include <algorithm>
#include <cmath>
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

#include "chat_reply.h"
#include "chat_template.h"
#include "cpu_attention.h"
#include "gguf_parser.h"
#include "graph_ir.h"
#include "graph_ir_expr.h"
#include "graph_ir_ops.h"
#include "graph_ir_recurrence.h"
#include "graph_ir_validate.h"
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
  // raw: the request line, for the members whose order matters (render_chat's tool definitions).
  json handle(const std::string& method, const json& p, const std::string& raw = std::string()) {
    if (method == "hello") return hello();
    if (method == "check") {
      try {
        if (p.contains("gguf") && !p.at("gguf").is_null()) {
          load(p, nullptr);
        } else {  // the document alone: the document rules (IR_SPEC 11.3)
          if (!p.contains("ir") || p.at("ir").is_null()) throw std::runtime_error("check: neither gguf nor ir");
          const std::vector<std::string> errs =
              ember_ir::ir_document_errors(ember_ir::parse_ir_document(read_file(p.at("ir").get<std::string>())));
          std::string msg;
          for (const auto& e : errs) msg += (msg.empty() ? "" : "; ") + e;
          if (!errs.empty()) return {{"accepted", false}, {"message", msg}};
        }
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
    if (method == "forward") return guarded(session(p), [&](Session& s) { return forward(s, p); });
    if (method == "truncate") return guarded(session(p), [&](Session& s) { return truncate(s, p); });
    if (method == "snapshot") {
      Session& s = session(p);
      s.snaps[p.at("tag").get<std::string>()] = Snap{s.ids, s.m.ip->save_state()};  // a tag used again: replaced
      return {{"len", s.ids.size()}};
    }
    if (method == "restore") {
      Session& s = session(p);
      const auto it = s.snaps.find(p.at("tag").get<std::string>());
      if (it == s.snaps.end()) throw std::runtime_error("no snapshot " + p.at("tag").dump());
      check_reports(*s.m.ip, p, false);
      return guarded(s, [&](Session& ss) {
        ss.m.ip->load_state(it->second.state);
        ss.ids = it->second.ids;
        json r{{"len", ss.ids.size()}};
        reports(ss, p, r, false, {});
        return r;
      });
    }
    if (method == "render_chat") return render_chat(p, raw);
    if (method == "parse_reply") return parse_reply(p);
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
            {"version", "ir_adapter 5 (IR versions 0 and 1; spec revision 1.4.2)"},
            {"backend", "cpu"},
            {"exact", true},
            {"profiles", {"core", "state", "recursion", "thirdparty", "recurrence", "chat"}},
            {"methods", {"hello", "check", "open", "forward", "truncate", "snapshot", "restore", "close", "digest",
                         "dequantize", "run_node", "render_chat", "parse_reply"}}};
  }

  // The call f on s; when it fails (an error or a skip), the session's ids and sequence state as they were
  // before it (IR_SPEC 11.3: an error leaves the session unchanged).
  template <class F>
  static json guarded(Session& s, F f) {
    const std::vector<int> ids = s.ids;
    const ember_ir::Interpreter::State state = s.m.ip->save_state();
    try {
      return f(s);
    } catch (...) {
      s.m.ip->load_state(state);
      s.ids = ids;
      throw;
    }
  }

  // IR_SPEC 8.1.3 and 8.1.6 through chat_template.h: the record's fields and values, the conversation's structure,
  // its reserved texts and the tool definitions are checked there; any refusal is a result. The params are read
  // from the request line as ordered JSON (a definition renders with its members in the order received).
  static json render_chat(const json& p, const std::string& raw) {
    namespace chat = microtorch::chat;
    try {
      chat::json req = chat::json::parse(raw, nullptr, false);
      const chat::json q = !req.is_discarded() && req.is_object() && req.contains("params") && req["params"].is_object()
                               ? req["params"]
                               : chat::json::parse(p.dump());
      const chat::Template t = chat::Template::from_json(q.at("record"));
      const std::vector<chat::Message> m = chat::messages_from_json(q.at("messages"));
      const chat::json th = q.contains("thinking") ? q.at("thinking") : chat::json();
      const int thinking = th.is_boolean() ? (th.get<bool>() ? 1 : 0) : -1;
      std::vector<chat::json> tools;
      if (q.contains("tools")) {  // a list; tool_json refuses a definition that is not an object
        if (!q.at("tools").is_array()) throw chat::Error("tools is a list of tool definitions");
        for (const auto& d : q.at("tools")) tools.push_back(d);
      }
      return {{"text", chat::render(m, t, q.value("generation_prompt", false), thinking, tools)}};
    } catch (const chat::Error& e) {
      return {{"refused", e.what()}};
    } catch (const nlohmann::json::exception& e) {  // a field of the wrong type
      return {{"refused", std::string("chat template: ") + e.what()}};
    }
  }

  // IR_SPEC 8.1.6 and 11.3: {"tokens": {id: text}, "ids", "call_id"} -> {"parsed": true, "content", "tool_calls"}
  // or {"parsed": false}, by chat::parse_reply, the split chat_reply.h's reply_of serves replies with.
  static json parse_reply(const json& p) {
    namespace chat = microtorch::chat;
    std::map<int, std::string> text;
    for (auto it = p.at("tokens").begin(); it != p.at("tokens").end(); ++it) {
      size_t used = 0;
      const int id = std::stoi(it.key(), &used);
      if (used != it.key().size()) throw std::runtime_error("parse_reply: token key '" + it.key() + "' is not an id");
      text[id] = it.value().get<std::string>();
    }
    const auto decode = [&](const std::vector<int>& part) {
      std::string s;
      for (int id : part) {
        const auto f = text.find(id);
        if (f == text.end()) throw std::runtime_error("parse_reply: no text for id " + std::to_string(id));
        s += f->second;
      }
      return s;
    };
    const chat::ParsedReply r =
        chat::parse_reply(p.at("ids").get<std::vector<int>>(), p.at("call_id").get<int>(), decode);
    if (!r.parsed) return {{"parsed", false}};
    return {{"parsed", true}, {"content", r.content}, {"tool_calls", ember::tool_calls_json<json>(r.tool_calls)}};
  }

  // An expr node alone (IR_SPEC 5.19): the interpreter's evaluation (graph_ir_expr.h) and its shape checks.
  // An attention node alone (IR_SPEC 5.15; section 11.2's defaults): a kv slot of its own heads, kv_heads, head_dim,
  // window and sinks, empty at the start, its rows at positions 0 .. T-1, mask causal; the interpreter's kernels
  // (cpu_attention.h), position_bias through attention_bias_batch_cpu, relative_kv through attention_relkv_batch_cpu.
  // The kv slot is not reported (state {}).
  static json run_attention(const json& node, const json& inputs, const json& weights) {
    auto attr = [&](const char* k, const json& dflt) {  // application 0's value
      const json& v = node.contains(k) ? node.at(k) : dflt;
      return v.is_object() && v.contains("per_layer") ? v.at("per_layer").at(0) : v;
    };
    if (attr("mask", json("causal")) != "causal") throw Skip{"run_node: attention mask turn_state needs token ids"};
    const int H = attr("heads", nullptr).get<int>(), hd = attr("head_dim", nullptr).get<int>();
    const int KV = attr("kv_heads", json(H)).get<int>();
    const int window = attr("window", json(0)).get<int>(), sinks = attr("sinks", json(0)).get<int>();
    const float scale = attr("scale", nullptr).get<float>();
    const float softcap = node.contains("softcap") ? attr("softcap", nullptr).get<float>() : 0.0f;  // revision 1.5
    if (node.contains("softcap") && !(softcap > 0.0f))
      throw std::runtime_error("IR_SPEC 5.15 (rule 10): softcap is not positive");
    const std::vector<std::string> ins = node.at("in").get<std::vector<std::string>>();
    if (ins.size() != 3) throw std::runtime_error("IR_SPEC 10.1 rule 7: attention reads [q, k, v]");
    std::vector<int64_t> sq, sk, sv;
    std::vector<float> q = floats(inputs.at(ins[0]), &sq), k = floats(inputs.at(ins[1]), &sk),
                       v = floats(inputs.at(ins[2]), &sv);
    if (sq.size() != 2 || sk != sv || sk.size() != 2 || sq[0] != sk[0] || sq[1] != H * hd || sk[1] != KV * hd)
      throw std::runtime_error("run_node: attention q must be [T, heads x head_dim] and k, v [T, kv_heads x head_dim]");
    const int T = static_cast<int>(sq[0]);
    KVCacheLayer cache;
    cache.k = std::move(k);
    cache.v = std::move(v);
    std::vector<float> out;
    if (node.contains("position_bias")) {
      const json& pb = node.at("position_bias");
      std::string wn = pb.at("weight").get<std::string>();
      const auto at = wn.find("{i}");
      if (at != std::string::npos) wn.replace(at, 3, "0");
      if (!weights.contains(wn)) throw std::runtime_error("run_node: no weight '" + wn + "'");
      std::vector<int64_t> sw;
      const std::vector<float> W = floats(weights.at(wn), &sw);
      const std::vector<int> buckets = pb.at("buckets").get<std::vector<int>>();
      int top = -1;
      for (int b : buckets) top = std::max(top, b);
      if (buckets.empty() || *std::min_element(buckets.begin(), buckets.end()) < 0 || sw.size() != 2 || sw[1] != H ||
          sw[0] <= top)
        throw std::runtime_error("IR_SPEC 5.15 (rule 39): position_bias table must be [B, heads] with B above every bucket");
      std::vector<int> qpos(static_cast<size_t>(T));
      for (int t = 0; t < T; ++t) qpos[t] = t;
      attention_bias_batch_cpu(q, cache, out, T, attention_visible_rows(T, 0, window, sinks), nullptr, qpos, H, KV, hd,
                               scale, W.data(), buckets, softcap);
    } else if (node.contains("relative_kv")) {  // revision 1.4
      const json& rk = node.at("relative_kv");
      if (node.contains("position_bias"))
        throw std::runtime_error("IR_SPEC 5.15 (rule 40): relative_kv and position_bias on one node");
      if (!rk.is_object() || !rk.contains("key") || !rk.at("key").is_string() || !rk.contains("max_distance") ||
          !rk.at("max_distance").is_number_integer() || rk.at("max_distance").get<int64_t>() < 0)
        throw std::runtime_error("IR_SPEC 5.15 (rule 40): relative_kv is {key, max_distance} with an optional value");
      const int K = rk.at("max_distance").get<int>();
      std::vector<std::vector<float>> tables(2);
      int widths[2] = {0, 0};
      const char* members[2] = {"key", "value"};
      for (int m = 0; m < 2; ++m) {
        if (!rk.contains(members[m]) || rk.at(members[m]).is_null()) continue;
        if (!rk.at(members[m]).is_string())
          throw std::runtime_error("IR_SPEC 5.15 (rule 40): relative_kv value is a name or null");
        std::string wn = rk.at(members[m]).get<std::string>();
        const auto at = wn.find("{i}");
        if (at != std::string::npos) wn.replace(at, 3, "0");
        if (!weights.contains(wn)) throw std::runtime_error("run_node: no weight '" + wn + "'");
        std::vector<int64_t> sw;
        tables[m] = floats(weights.at(wn), &sw);
        if (sw.size() != 2 || sw[0] != K + 1 || (sw[1] != hd && sw[1] != static_cast<int64_t>(H) * hd))
          throw std::runtime_error(std::string("IR_SPEC 5.15 (rule 40): relative_kv ") + members[m] +
                                   " must be [max_distance + 1, head_dim or heads x head_dim]");
        widths[m] = static_cast<int>(sw[1]);
      }
      std::vector<int> qpos(static_cast<size_t>(T));
      for (int t = 0; t < T; ++t) qpos[t] = t;
      attention_relkv_batch_cpu(q, cache, out, T, attention_visible_rows(T, 0, window, sinks), nullptr, qpos, H, KV, hd,
                                scale, tables[0].data(), widths[0], tables[1].empty() ? nullptr : tables[1].data(),
                                widths[1], K, softcap);
    } else {
      attention_window_batch_cpu(q, cache, out, T, 0, H, KV, hd, scale, window, sinks, softcap);
    }
    json r;
    r["outputs"] = json::object();
    r["outputs"][node.at("out").get<std::string>()] = tensor({T, static_cast<int64_t>(H) * hd}, out.data(), out.size());
    r["state"] = json::object();
    return r;
  }

  // An mla_attention node alone (IR_SPEC 5.20; section 11.2's defaults): its own kv slot (kv_heads 1, head_dim
  // kv_rank + qk_rope, the node's window and sinks), empty at the start, rows [c | kr] at positions 0 .. T-1,
  // through the interpreter's kernel (graph_ir_ops.h). The kv slot is not reported (state {}).
  static json run_mla(const json& node, const json& inputs, const json& weights) {
    auto attr = [&](const char* k, const json& dflt) {  // application 0's value
      const json& v = node.contains(k) ? node.at(k) : dflt;
      return v.is_object() && v.contains("per_layer") ? v.at("per_layer").at(0) : v;
    };
    for (const char* k : {"heads", "kv_rank", "qk_nope", "qk_rope", "v_head_dim", "k_b", "v_b", "scale", "causal"})
      if (!node.contains(k)) throw std::runtime_error(std::string("IR_SPEC 5.20 (rule 8): mla_attention needs ") + k);
    if (attr("causal", nullptr) != json(true)) throw std::runtime_error("IR_SPEC 5.20 (rule 14): mla_attention causal must be true");
    const int H = attr("heads", nullptr).get<int>(), r = attr("kv_rank", nullptr).get<int>();
    const int dn = attr("qk_nope", nullptr).get<int>(), dr = attr("qk_rope", nullptr).get<int>();
    const int dv = attr("v_head_dim", nullptr).get<int>();
    const int window = attr("window", json(0)).get<int>(), sinks = attr("sinks", json(0)).get<int>();
    const std::vector<std::string> ins = node.at("in").get<std::vector<std::string>>();
    if (ins.size() != 4) throw std::runtime_error("IR_SPEC 10.1 rule 7: mla_attention reads [qn, qr, c, kr]");
    std::vector<int64_t> s[4];
    std::vector<float> v[4];
    for (int k = 0; k < 4; ++k) {
      if (!inputs.contains(ins[k])) throw std::runtime_error("run_node: no input '" + ins[k] + "'");
      v[k] = floats(inputs.at(ins[k]), &s[k]);
      if (s[k].size() != 2 || s[k][0] != s[0][0]) throw std::runtime_error("run_node: mla_attention inputs are [T, w]");
    }
    const int64_t want[4] = {static_cast<int64_t>(H) * dn, static_cast<int64_t>(H) * dr, r, dr};
    for (int k = 0; k < 4; ++k)
      if (s[k][1] != want[k])
        throw std::runtime_error("IR_SPEC 5.20 (rule 37): mla_attention input '" + ins[k] + "' is " + std::to_string(s[k][1]) +
                                 " wide, not " + std::to_string(want[k]));
    auto tensor_of = [&](const char* key, const std::vector<int64_t>& shape) {
      std::string wn = node.at(key).get<std::string>();
      const auto at = wn.find("{i}");
      if (at != std::string::npos) wn.replace(at, 3, "0");
      if (!weights.contains(wn)) throw std::runtime_error("run_node: no weight '" + wn + "'");
      std::vector<int64_t> sw;
      std::vector<float> w = floats(weights.at(wn), &sw);
      if (sw != shape) throw std::runtime_error(std::string("IR_SPEC 5.20 (rule 42): mla_attention ") + key + " has the wrong shape");
      return w;
    };
    const std::vector<float> Kb = tensor_of("k_b", {H, r, dn}), Vb = tensor_of("v_b", {H, dv, r});
    const int T = static_cast<int>(s[0][0]);
    std::vector<float> kv(static_cast<size_t>(T) * (r + dr));
    for (int t = 0; t < T; ++t) {
      std::memcpy(kv.data() + static_cast<size_t>(t) * (r + dr), v[2].data() + static_cast<size_t>(t) * r, r * sizeof(float));
      std::memcpy(kv.data() + static_cast<size_t>(t) * (r + dr) + r, v[3].data() + static_cast<size_t>(t) * dr,
                  dr * sizeof(float));
    }
    std::vector<float> out(static_cast<size_t>(T) * H * dv);
    ember_ir::cpu::mla_attention(v[0].data(), v[1].data(), kv.data(), out.data(), T,
                                 attention_visible_rows(T, 0, window, sinks), H, r, dn, dr, dv, Kb.data(), Vb.data(),
                                 attr("scale", nullptr).get<float>());
    json res;
    res["outputs"] = json::object();
    res["outputs"][node.at("out").get<std::string>()] = tensor({T, static_cast<int64_t>(H) * dv}, out.data(), out.size());
    res["state"] = json::object();
    return res;
  }

  // A split node alone (IR_SPEC 5.5, groups of revision 1.5): the interpreter's copies (graph_ir_ops.h).
  static json run_split(const json& node, const json& inputs) {
    const std::vector<std::string> ins = node.at("in").get<std::vector<std::string>>();
    if (ins.size() != 1) throw std::runtime_error("IR_SPEC 10.1 rule 7: split reads one value");
    const std::vector<int> sizes = node.at("sizes").get<std::vector<int>>();
    const std::vector<std::string> outs = node.at("out").get<std::vector<std::string>>();
    if (sizes.size() != outs.size()) throw std::runtime_error("IR_SPEC 5.5 (rule 10): sizes and outputs differ in count");
    const json& gj = node.contains("groups") ? node.at("groups") : json(1);
    if (!gj.is_number_integer() || gj.get<int64_t>() < 1)
      throw std::runtime_error("IR_SPEC 5.5 (rule 10): split groups is not a positive integer");
    const int g = gj.get<int>();
    if (!inputs.contains(ins[0])) throw std::runtime_error("run_node: no input '" + ins[0] + "'");
    std::vector<int64_t> sx;
    const std::vector<float> x = floats(inputs.at(ins[0]), &sx);
    if (sx.size() != 2) throw std::runtime_error("run_node: split input is not [T, w]");
    int64_t total = 0;
    for (int s : sizes) total += s;
    if (total * g != sx[1])
      throw std::runtime_error("IR_SPEC 5.5 (rule 37): split sizes times groups do not sum to the input width");
    const int T = static_cast<int>(sx[0]);
    std::vector<std::vector<float>> parts;
    std::vector<float*> ptrs;
    for (int s : sizes) parts.emplace_back(static_cast<size_t>(T) * g * s);
    for (auto& p : parts) ptrs.push_back(p.data());
    ember_ir::cpu::split_groups(x.data(), T, sizes, g, ptrs);
    json res;
    res["outputs"] = json::object();
    for (size_t k = 0; k < outs.size(); ++k)
      res["outputs"][outs[k]] = tensor({T, static_cast<int64_t>(g) * sizes[k]}, parts[k].data(), parts[k].size());
    res["state"] = json::object();
    return res;
  }

  // A rope node alone (IR_SPEC 5.14): q [T, heads x head_dim], k [T, kv_heads x head_dim], each row at its position
  // pos[t]; the angle table as the interpreter builds it (graph_ir.cpp build_rope_tables): freq_j = 1 / powf(theta,
  // f32(2j) / f32(n_rot)), angle (f32(pos) * freq_scale) * freq_j, divided by f_j under freq_factors, cosf and sinf
  // times attn_factor; with two factor sets the run's length max(pos) + 1 selects the long one past original_ctx
  // (section 5.14's single-node rule). Each rotated element x0 * cos - x1 * sin or x0 * sin + x1 * cos, no
  // fused multiply-add; pairing interleaved or neox; the first n_rot dims of each head rotate.
  static json run_rope(const json& node, const json& inputs, const json& weights) {
    auto attr = [&](const char* k, const json& dflt) {  // application 0's value
      const json& v = node.contains(k) ? node.at(k) : dflt;
      return v.is_object() && v.contains("per_layer") ? v.at("per_layer").at(0) : v;
    };
    if (attr("rope_heads", json("all")) != "all")
      throw Skip{"run_node: rope_heads \"first\" is not implemented in ember (refused at load by name)"};
    const int H = attr("heads", nullptr).get<int>(), hd = attr("head_dim", nullptr).get<int>();
    const int KV = attr("kv_heads", json(H)).get<int>(), n_rot = attr("n_rot", json(hd)).get<int>();
    const float theta = attr("theta", nullptr).get<float>(), fs = attr("freq_scale", json(1.0)).get<float>();
    const float m = attr("attn_factor", json(1.0)).get<float>();
    const std::string pairing = attr("pairing", nullptr).get<std::string>();
    if (pairing != "interleaved" && pairing != "neox") throw std::runtime_error("IR_SPEC 5.14 (rule 5): pairing " + pairing);
    if (n_rot <= 0 || n_rot % 2 || n_rot > hd) throw std::runtime_error("IR_SPEC 5.14 (rule 10): n_rot is not even in 1 .. head_dim");
    if (!(fs > 0.0f)) throw std::runtime_error("IR_SPEC 5.14 (rule 10): freq_scale is not positive");
    if (!(m > 0.0f)) throw std::runtime_error("IR_SPEC 5.14 (rule 10): attn_factor is not positive");
    const std::vector<std::string> ins = node.at("in").get<std::vector<std::string>>();
    const std::vector<std::string> outs = node.at("out").get<std::vector<std::string>>();
    if (ins.size() != 3 || outs.size() != 2) throw std::runtime_error("IR_SPEC 10.1 rule 7: rope reads [q, k, pos] and writes two");
    std::vector<int64_t> sq, sk;
    std::vector<float> q = floats(inputs.at(ins[0]), &sq), k = floats(inputs.at(ins[1]), &sk);
    const json& pj = inputs.at(ins[2]);
    const std::vector<int> pos = (pj.is_object() && pj.contains("ids") ? pj.at("ids") : pj).get<std::vector<int>>();
    if (sq.size() != 2 || sk.size() != 2 || sq[0] != sk[0] || sq[1] != H * hd || sk[1] != KV * hd ||
        static_cast<int64_t>(pos.size()) != sq[0])
      throw std::runtime_error("IR_SPEC 5.14 (rule 37): rope q must be [T, heads x head_dim], k [T, kv_heads x head_dim], "
                               "pos T positions");
    const int T = static_cast<int>(sq[0]), half = n_rot / 2;
    int n = 0;
    for (int p : pos) {
      if (p < 0) throw std::runtime_error("run_node: rope position " + std::to_string(p) + " is negative");
      n = std::max(n, p + 1);
    }
    // The frequency factors in force: a name, or the set {short, long, original_ctx} the length selects.
    std::vector<float> f;
    if (node.contains("freq_factors")) {
      const json& ff = node.at("freq_factors");
      std::string name;
      if (ff.is_string()) {
        name = ff.get<std::string>();
      } else if (ff.is_object() && ff.contains("short") && ff.contains("long") && ff.contains("original_ctx") &&
                 ff.at("short").is_string() && ff.at("long").is_string() && ff.at("original_ctx").is_number_integer() &&
                 ff.at("original_ctx").get<int64_t>() >= 1) {
        name = (n > ff.at("original_ctx").get<int>() ? ff.at("long") : ff.at("short")).get<std::string>();
      } else {
        throw std::runtime_error("IR_SPEC 5.14 (rule 41): freq_factors is a name or {short, long, original_ctx}");
      }
      const auto at = name.find("{i}");
      if (at != std::string::npos) name.replace(at, 3, "0");
      if (!weights.contains(name)) throw std::runtime_error("run_node: no weight '" + name + "'");
      std::vector<int64_t> sw;
      f = floats(weights.at(name), &sw);
      if (sw.size() != 1 || sw[0] != half)
        throw std::runtime_error("IR_SPEC 5.14 (rule 41): freq_factors '" + name + "' is not of ne {n_rot / 2}");
    }
    const bool neox = pairing == "neox";
    auto rotate = [&](std::vector<float>& x, int heads) {
      for (int t = 0; t < T; ++t)
        for (int j = 0; j < half; ++j) {
          const float freq = 1.0f / std::pow(theta, float(2 * j) / n_rot);
          float val = (static_cast<float>(pos[t]) * fs) * freq;
          if (!f.empty()) val = val / f[j];
          const float c = std::cos(val) * m, s = std::sin(val) * m;
          for (int h = 0; h < heads; ++h) {
            float* r = x.data() + (static_cast<size_t>(t) * heads + h) * hd;
            const size_t i0 = neox ? j : 2 * j, i1 = neox ? j + half : 2 * j + 1;
            const float v0 = r[i0], v1 = r[i1];
            const float a = v0 * c, b = v1 * s, e = v0 * s, g = v1 * c;
            r[i0] = a - b;
            r[i1] = e + g;
          }
        }
    };
    rotate(q, H);
    rotate(k, KV);
    json r;
    r["outputs"] = json::object();
    r["outputs"][outs[0]] = tensor(sq, q.data(), q.size());
    r["outputs"][outs[1]] = tensor(sk, k.data(), k.size());
    r["state"] = json::object();
    return r;
  }

  static json run_expr(const json& node, const json& inputs) {
    if (!node.contains("expr") || !node.at("expr").is_string()) throw std::runtime_error("IR_SPEC 5.19: expr needs its expr");
    const std::vector<std::string> ins = node.at("in").get<std::vector<std::string>>();
    if (ins.empty()) throw std::runtime_error("IR_SPEC 10.1 rule 7: expr reads at least one value");
    std::vector<std::vector<float>> vals;
    std::vector<int64_t> shape0;
    for (size_t k = 0; k < ins.size(); ++k) {
      if (!inputs.contains(ins[k])) throw std::runtime_error("run_node: no input '" + ins[k] + "'");
      std::vector<int64_t> shape;
      vals.push_back(floats(inputs.at(ins[k]), &shape));
      if (shape.size() != 2) throw std::runtime_error("run_node: expr input '" + ins[k] + "' is not [T, w]");
      if (k == 0) shape0 = shape;
      if (shape != shape0)
        throw std::runtime_error("IR_SPEC 5.19 (rule 37): expr inputs must share one shape [T, w]");
    }
    const auto& e = ember_ir::expr::cached(node.at("expr").get<std::string>(), ins.size());
    std::vector<const float*> ptrs;
    for (const auto& v : vals) ptrs.push_back(v.data());
    std::vector<float> out(vals[0].size());
    ember_ir::expr::eval(e, ptrs, static_cast<int>(shape0[0]), static_cast<int>(shape0[1]), out.data());
    json r;
    r["outputs"] = json::object();
    r["outputs"][node.at("out").get<std::string>()] = tensor(shape0, out.data(), out.size());
    r["state"] = json::object();
    return r;
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
    if (node.value("op", "") == "expr") return run_expr(node, p.at("inputs"));
    if (node.value("op", "") == "attention") return run_attention(node, p.at("inputs"), p.value("weights", json::object()));
    if (node.value("op", "") == "mla_attention") return run_mla(node, p.at("inputs"), p.value("weights", json::object()));
    if (node.value("op", "") == "split") return run_split(node, p.at("inputs"));
    if (node.value("op", "") == "rope") return run_rope(node, p.at("inputs"), p.value("weights", json::object()));
    if (node.value("op", "") != "recurrence")
      throw Skip{"run_node: ember's adapter runs recurrence, expr, attention, mla_attention, split and rope nodes only "
                 "(whole-model methods for the other ops)"};
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
      } else if (k != "kv_rows" && k != "kv_positions" && k != "recurrent") {
        throw std::runtime_error("unknown report '" + k + "'");
      }
    }
  }

  json forward(Session& s, const json& p) {
    check_reports(*s.m.ip, p, true);
    const std::vector<int> ids = p.at("ids").get<std::vector<int>>();
    if (ids.empty()) throw std::runtime_error("forward: ids is empty (IR_SPEC 11.3: at least one id)");
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
    if (n < len) {  // n = len changes nothing (IR_SPEC 11.3)
      ember_ir::Interpreter& ip = *s.m.ip;
      const int r = ip.resume_point(n);
      ip.truncate(r);
      if (r < n) ip.forward(std::vector<int>(s.ids.begin() + r, s.ids.begin() + n), r, true);
      s.ids.resize(n);
    }
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
        } else {  // [] for a document without depth_route (IR_SPEC 11.3)
          r["depth"] = n > 0 && ip.has_depth_route() ? ip.last_depths() : std::vector<int>{};
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
        reply["result"] = ad.handle(req.at("method").get<std::string>(), req.value("params", json::object()), line);
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
