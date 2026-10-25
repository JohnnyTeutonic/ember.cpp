// POST /score (server.cpp): the log-probability of given continuations under the model, with the
// semantics of paperkiln's `mtstudio score` (tools/mtstudio.cpp score_cmd, the forced-choice scorer of
// its benchmarks; LOCAL_LLM_STACK.md section 3) on token ids.
//
//   request:  {"requests": [{"key": any, "prompt_ids": [int], "continuation_ids": [int]}, ...]}
//   response: {"results": [{"i", "key", "prompt_tokens", "continuation_ids", "over_ctx",
//                           "logprobs", "argmax", "sum"}, ...], "stats": {...}}
//
// Each request is one fresh forward over prompt_ids + continuation_ids[:-1] from position 0 (no state
// is carried between requests; TinyLlamaSession::score_rows). For every continuation token j, at
// position P + j (P = the prompt's length), from the logits row before it: its log-probability under
// the softmax at temperature 1, in double from the float logits (max, then the sum of exp(x - max),
// then x_c - max - log(sum), in vocabulary order), and the argmax with the specials mtstudio bans
// from generation (<unk>, <s>, </s>, <pad>, by name in the vocabulary) left out, the first maximum
// winning; "sum" adds the log-probabilities in order. A request with P + C > ctx + 1 ids (ctx =
// score_context(), mtstudio's T) is reported "over_ctx": true and not scored. Records carry
// mtstudio's fields: "i" (the request's index), "prompt_tokens", "continuation_ids", "key" when the
// request has one.
//
// Refusals are mtstudio's, whose score fails the whole run on the first bad line: here the whole
// request is refused (HTTP 400) before anything runs, naming the first bad request. A request must
// be an object; prompt_ids and continuation_ids are read as mtstudio reads them (nlohmann's
// get<std::vector<int>>); an empty prompt or continuation, a prompt id outside [0, vocab + k) (k
// the model's state tokens, 0 without), a continuation id outside [0, vocab), or (state-token
// models) a prompt whose state ids are not complete blocks after an <|endoftext|> that ends a reply
// (nn::state_ids_error, ported below) is refused. mtstudio's text fields ("prompt",
// "continuation", "messages") are not served: ember scores ids only.
#ifndef EMBER_SCORE_H
#define EMBER_SCORE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ember {

// What validation needs to know about the model.
struct ScoreModel {
  size_t vocab = 0;    // the logits' width: continuation ids lie in [0, vocab)
  size_t state_k = 0;  // per-turn state tokens: prompt ids lie in [0, vocab + state_k)
  int eot = -1, user = -1, assistant = -1;  // the state-token template's markers (state_k > 0)
  size_t ctx = 0;      // positions one forward holds
};

// One parsed request: mtstudio's record so far, and the ids.
struct ScoreItem {
  nlohmann::json rec;
  std::vector<int> prompt, cont;
  bool over_ctx = false;
};

// paperkiln's nn::state_ids_error (src/llama.cpp): "" when every id >= vocab sits in a complete,
// ordered block vocab .. vocab + k - 1 directly after an <|endoftext|> that closes an exchange.
inline std::string score_state_ids_error(const std::vector<int>& ids, const ScoreModel& m) {
  enum { NONE, USER, REPLY } phase = NONE;
  bool closed = false;
  for (size_t p = 0; p < ids.size(); ++p) {
    const int id = ids[p];
    if (id < 0 || static_cast<size_t>(id) >= m.vocab + m.state_k)
      return "id " + std::to_string(id) + " at position " + std::to_string(p) + " is out of range";
    if (static_cast<size_t>(id) >= m.vocab) {
      if (!closed || static_cast<size_t>(id) != m.vocab)
        return "state id " + std::to_string(id) + " at position " + std::to_string(p) +
               " does not start a block after an <|endoftext|> that ends a reply";
      for (size_t j = 0; j < m.state_k; ++j)
        if (p + j >= ids.size() || static_cast<size_t>(ids[p + j]) != m.vocab + j)
          return "the state block at position " + std::to_string(p) + " is not the ids " + std::to_string(m.vocab) +
                 " .. " + std::to_string(m.vocab + m.state_k - 1) + " in order";
      p += m.state_k - 1;
      closed = false;
      continue;
    }
    closed = false;
    if (id == m.user) {
      phase = USER;
    } else if (id == m.assistant) {
      if (phase == USER) phase = REPLY;
    } else if (id == m.eot) {
      closed = phase == REPLY;
      phase = NONE;
    }
  }
  return "";
}

// Parses and checks a /score body in mtstudio's order (per request: the ids, empty, prompt range,
// continuation range, state blocks; then over_ctx). Returns "" or the refusal (HTTP 400).
inline std::string parse_score_body(const nlohmann::json& body, const ScoreModel& m, std::vector<ScoreItem>& out) {
  if (!body.is_object() || !body.contains("requests") || !body["requests"].is_array())
    return "body must be {\"requests\": [{\"key\", \"prompt_ids\", \"continuation_ids\"}, ...]}";
  const nlohmann::json& reqs = body["requests"];
  out.clear();
  out.reserve(reqs.size());
  for (size_t n = 0; n < reqs.size(); ++n) {
    const nlohmann::json& r = reqs[n];
    const std::string at = "request " + std::to_string(n);
    if (!r.is_object()) return at + " is not an object";
    if (!r.contains("prompt_ids") || !r.contains("continuation_ids"))
      return at + ": prompt_ids and continuation_ids are required (ember scores token ids; mtstudio's text fields "
                  "are not served)";
    ScoreItem it;
    try {
      it.prompt = r["prompt_ids"].get<std::vector<int>>();
      it.cont = r["continuation_ids"].get<std::vector<int>>();
    } catch (const nlohmann::json::exception& e) {
      return at + ": prompt_ids and continuation_ids must be lists of integers (" + e.what() + ")";
    }
    if (it.prompt.empty() || it.cont.empty()) return at + " has an empty prompt or continuation";
    for (int t : it.prompt)
      if (t < 0 || static_cast<size_t>(t) >= m.vocab + m.state_k) return at + ": prompt id out of range";
    for (int t : it.cont)
      if (t < 0 || static_cast<size_t>(t) >= m.vocab) return at + ": continuation id out of range";
    if (m.state_k > 0) {
      std::vector<int> ids = it.prompt;
      ids.insert(ids.end(), it.cont.begin(), it.cont.end());
      const std::string err = score_state_ids_error(ids, m);
      if (!err.empty()) return at + ": " + err;
    }
    it.rec = {{"i", n}, {"prompt_tokens", it.prompt.size()}, {"continuation_ids", it.cont}};
    if (r.contains("key")) it.rec["key"] = r["key"];
    it.over_ctx = it.prompt.size() + it.cont.size() > m.ctx + 1;
    out.push_back(std::move(it));
  }
  return "";
}

// banned[k]: vocabulary id k is one of the specials mtstudio leaves out of the argmax, found by name
// among ids 0 .. vocab - 1 (the first id holding the name, as mtstudio's vocabulary map keeps).
inline std::vector<char> score_banned(const std::vector<std::string>& tokens, size_t vocab) {
  std::vector<char> banned(vocab, 0);
  const size_t n = tokens.size() < vocab ? tokens.size() : vocab;
  for (const char* sp : {"<unk>", "<s>", "</s>", "<pad>"})
    for (size_t k = 0; k < n; ++k)
      if (tokens[k] == sp) {
        banned[k] = 1;
        break;
      }
  return banned;
}

// mtstudio's arithmetic on the continuation's rows (rows[j * V ..] precedes continuation token j):
// sets rec's over_ctx (false), logprobs, argmax and sum.
inline void score_fill(nlohmann::json& rec, const std::vector<int>& c, const float* rows, size_t V,
                       const std::vector<char>& banned) {
  std::vector<double> lp;
  std::vector<int> am;
  lp.reserve(c.size());
  am.reserve(c.size());
  double sum = 0.0;
  for (size_t j = 0; j < c.size(); ++j) {
    const float* row = rows + j * V;
    double mx = -1e300;
    int best = -1;
    double best_v = -1e300;
    for (size_t k = 0; k < V; ++k) {
      const double x = row[k];
      mx = std::max(mx, x);
      if (!banned[k] && (best < 0 || x > best_v)) {
        best = static_cast<int>(k);
        best_v = x;
      }
    }
    double z = 0.0;
    for (size_t k = 0; k < V; ++k) z += std::exp(static_cast<double>(row[k]) - mx);
    const double l = static_cast<double>(row[static_cast<size_t>(c[j])]) - mx - std::log(z);
    lp.push_back(l);
    am.push_back(best);
    sum += l;
  }
  rec["over_ctx"] = false;
  rec["logprobs"] = lp;
  rec["argmax"] = am;
  rec["sum"] = sum;
}

}  // namespace ember

#endif  // EMBER_SCORE_H
