// ember chat in the browser: ember.cpp's own TinyLlamaSession compiled to
// WebAssembly (deploy/wasm/build.sh), so a free static page can serve the
// paperkiln BPE chat model with no server. Same contract as the server's
// /chat for BPE models (server.cpp): the prompt is exactly the SFT template,
// <|user|>U<|assistant|>A<|endoftext|> per past turn then <|user|>U<|assistant|>,
// with the oldest turns dropped until prompt + max_new_tokens fits the
// context; a request's seed reseeds the sampler before generating.
//
//   ember_load(path)        load a GGUF from the in-memory filesystem; 1 if BPE
//   ember_chat(json)        {"user_input", "history": [{"user","assistant"}],
//                            "temperature", "top_k", "top_p", "seed",
//                            "max_new_tokens", "repeat_penalty",
//                            "repeat_last_n"} -> {"reply", "stop_reason",
//                            "tokens", "prompt_tokens", "history_turns_used"}
//                            or {"error"}
//
// Step by step, for a page that computes the logits itself (the WebGPU
// forward pass in web/webgpu/): the same request rules, sampler and stops,
// with the forward pass outside.
//   ember_prepare(json)     the ember_chat request -> {"prompt_ids", "max_new_tokens",
//                            "ctx", "prompt_tokens", "history_turns_used"} or {"error"};
//                            starts a reply (reseeds, sets the penalty)
//   ember_sample(ptr, n)    next token from n float logits at ptr (penalty over prompt +
//                            reply so far, then the session's sampler); a stop id (1-3)
//                            ends the reply and is not part of it
//   ember_finish(stop_id)   -> {"reply", "stop_reason", "tokens", "prompt_tokens",
//                            "history_turns_used"}; stop_id -1 means the length limit
#include <emscripten/emscripten.h>

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "api.h"

using json = nlohmann::json;

namespace {
std::unique_ptr<tinyllama::TinyLlamaSession> g_session;
std::string g_out;  // returned to JS; valid until the next call

struct Request {
    std::string prompt;
    int max_new = 64, top_k = 40, prompt_tokens = 0, turns_used = 0, ctx = 0, repeat_last_n = 64;
    float temperature = 0.7f, top_p = 0.9f, repeat_penalty = 1.0f;
    bool has_seed = false;
    unsigned seed = 0;
};

// A /chat request under the public page's caps (as the server's public mode) and the context
// contract (LOCAL_LLM_STACK.md section 3): oldest turns dropped until prompt + max_new fits, a
// prompt that alone fills the context rejected.
Request parse_request(const json& r) {
    Request q;
    const std::string user = r.value("user_input", "");
    if (user.empty() || user.size() > 600) throw std::runtime_error("message must be 1-600 characters");
    q.max_new = std::max(1, std::min(r.value("max_new_tokens", 64), 64));
    q.temperature = std::max(0.0f, std::min(r.value("temperature", 0.7f), 1.5f));
    q.top_k = std::max(0, std::min(r.value("top_k", 40), 200));
    q.top_p = std::max(0.0f, std::min(r.value("top_p", 0.9f), 1.0f));
    std::vector<std::pair<std::string, std::string>> turns;
    if (r.contains("history") && r["history"].is_array())
        for (const auto& h : r["history"]) turns.emplace_back(h.value("user", ""), h.value("assistant", ""));
    // Public-page cap: the NEWEST 16 turns (the contract drops the oldest).
    if (turns.size() > 16) turns.erase(turns.begin(), turns.end() - 16);
    const std::string tail = "<|user|>" + user + "<|assistant|>";
    auto build = [&](size_t from) {
        std::string p;
        for (size_t i = from; i < turns.size(); ++i)
            p += "<|user|>" + turns[i].first + "<|assistant|>" + turns[i].second + "<|endoftext|>";
        return p + tail;
    };
    q.ctx = g_session->get_config().max_position_embeddings;
    size_t from = 0;
    q.prompt = build(from);
    while (from < turns.size() && g_session->bpe_token_count(q.prompt) + q.max_new > q.ctx) q.prompt = build(++from);
    q.prompt_tokens = g_session->bpe_token_count(q.prompt);
    if (q.prompt_tokens >= q.ctx)
        throw std::runtime_error("message is " + std::to_string(q.prompt_tokens) + " tokens; the model's context is " +
                                 std::to_string(q.ctx));
    q.turns_used = static_cast<int>(turns.size() - from);
    q.has_seed = r.contains("seed");
    if (q.has_seed) q.seed = r["seed"].get<unsigned>();
    // The contract's repetition penalty (point 4), clamped as a public page: p in [1, 2], n in [0, ctx].
    q.repeat_penalty = std::max(1.0f, std::min(r.value("repeat_penalty", 1.0f), 2.0f));
    q.repeat_last_n = std::max(0, std::min(r.value("repeat_last_n", 64), q.ctx));
    return q;
}

// The step-by-step reply in progress (ember_prepare .. ember_finish).
Request g_req;
std::vector<int> g_context;  // prompt ids, then the reply's ids so far
size_t g_prompt_len = 0;

std::string strip(const std::string& s) {
    const char* ws = " \t\n\v\f\r";
    const auto a = s.find_first_not_of(ws), b = s.find_last_not_of(ws);
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}
}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int ember_load(const char* path) {
    try {
        g_session = std::make_unique<tinyllama::TinyLlamaSession>(path, "", 1, 0, /*mmap=*/false);
        return g_session->is_bpe() ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ember_load: %s\n", e.what());
        g_session.reset();
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE const char* ember_chat(const char* request) {
    json out;
    try {
        if (!g_session) throw std::runtime_error("no model loaded");
        const Request q = parse_request(json::parse(request));
        if (q.has_seed) g_session->set_seed(q.seed);
        g_session->set_repeat_penalty(q.repeat_penalty, q.repeat_last_n);
        out["reply"] = g_session->generate(q.prompt, q.max_new, q.temperature, q.top_k, q.top_p, "", false);
        g_session->set_repeat_penalty(1.0f, 64);
        out["stop_reason"] = g_session->last_stop_reason();
        out["tokens"] = g_session->last_generated_tokens();
        out["prompt_tokens"] = q.prompt_tokens;
        out["history_turns_used"] = q.turns_used;
    } catch (const std::exception& e) {
        out = json{{"error", e.what()}};
    }
    g_out = out.dump();
    return g_out.c_str();
}

EMSCRIPTEN_KEEPALIVE const char* ember_prepare(const char* request) {
    json out;
    try {
        if (!g_session || !g_session->is_bpe()) throw std::runtime_error("no chat model loaded");
        g_req = parse_request(json::parse(request));
        if (g_req.has_seed) g_session->set_seed(g_req.seed);
        g_session->set_repeat_penalty(g_req.repeat_penalty, g_req.repeat_last_n);
        g_context = g_session->bpe_encode(g_req.prompt);
        g_prompt_len = g_context.size();
        out["prompt_ids"] = g_context;
        out["max_new_tokens"] = g_req.max_new;
        out["ctx"] = g_req.ctx;
        out["prompt_tokens"] = g_req.prompt_tokens;
        out["history_turns_used"] = g_req.turns_used;
    } catch (const std::exception& e) {
        out = json{{"error", e.what()}};
    }
    g_out = out.dump();
    return g_out.c_str();
}

EMSCRIPTEN_KEEPALIVE int ember_sample(const float* logits, int n) {
    if (!g_session || n <= 0) return -1;
    std::vector<float> lg(logits, logits + n);
    const int t = g_session->sample_step(lg, g_context, g_req.temperature, g_req.top_k, g_req.top_p);
    if (t != 1 && t != 2 && t != 3) g_context.push_back(t);
    return t;
}

EMSCRIPTEN_KEEPALIVE const char* ember_finish(int stop_id) {
    json out;
    try {
        if (!g_session) throw std::runtime_error("no model loaded");
        const std::vector<int> reply(g_context.begin() + static_cast<long>(g_prompt_len), g_context.end());
        out["reply"] = strip(g_session->bpe_decode(reply));
        out["stop_reason"] = stop_id == 1 ? "eos" : (stop_id == 2 || stop_id == 3) ? "turn" : "length";
        out["tokens"] = static_cast<int>(reply.size());
        out["prompt_tokens"] = g_req.prompt_tokens;
        out["history_turns_used"] = g_req.turns_used;
        g_session->set_repeat_penalty(1.0f, 64);
    } catch (const std::exception& e) {
        out = json{{"error", e.what()}};
    }
    g_out = out.dump();
    return g_out.c_str();
}

}  // extern "C"
