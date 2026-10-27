// ember chat in the browser: ember.cpp's own TinyLlamaSession compiled to
// WebAssembly (deploy/wasm/build.sh), so a free static page can serve a
// chat model with no server. Same contract as the server's /chat
// (server.cpp; LOCAL_LLM_STACK.md section 3, points 3-5): the request is a
// prompt conversation (chat_request.h) rendered by the model's chat template
// (the record resolved at load, chat_tokenizer.h; paperkiln/1 writes
// <|user|>U<|assistant|>A<|endoftext|> per past turn then <|user|>U<|assistant|>),
// with the oldest whole exchanges dropped until prompt + max_new_tokens fits
// the context (the capped context: ember_set_ctx) and the system message
// pinned; a request's seed reseeds the sampler before generating.
//
//   ember_set_ctx(n)        the context cap for the next ember_load: the KV state holds at most n
//                            positions, as EMBER_IR_CTX does for the native server (graph_ir.h), so
//                            a third-party file that declares its whole trained context (Qwen3-0.6B:
//                            40,960 positions, 9.4 GB of f32 KV) fits the heap; a file with a smaller
//                            context keeps its own. n <= 0: no cap. Default EMBER_WASM_CTX (2048,
//                            set by build.sh). Returns the cap now set.
//   ember_load(path)        load a GGUF from the in-memory filesystem; 1 in chat mode
//   ember_ctx()             the loaded model's context (after the cap); 0 with no model
//   ember_ir_doc()          the IR document the interpreter runs ("" on the tuned llama path)
//   ember_chat(json)        {"user_input", "system", "history": [{"user","assistant"}]
//                            or [{"role","content"}] (a leading {"role": "system"} entry
//                            instead of "system"), "thinking", "temperature", "top_k",
//                            "top_p", "seed", "max_new_tokens", "repeat_penalty",
//                            "repeat_last_n"} -> {"reply", "stop_reason",
//                            "tokens", "prompt_tokens", "history_turns_used"}
//                            or {"error"}
//
// Step by step, for a page that computes the logits itself (the WebGPU
// forward pass in web/webgpu/): the same request rules, sampler and stops,
// with the forward pass outside.
//   ember_prepare(json)     the ember_chat request -> {"prompt_ids", "stop_ids", "max_new_tokens",
//                            "ctx", "prompt_tokens", "history_turns_used"} or {"error"};
//                            starts a reply (reseeds, sets the penalty)
//   ember_sample(ptr, n)    next token from n float logits at ptr (penalty over prompt +
//                            reply so far, then the session's sampler); an id in stop_ids
//                            (the template's stop set) ends the reply and is not part of it
//   ember_finish(stop_id)   -> {"reply", "stop_reason", "tokens", "prompt_tokens",
//                            "history_turns_used"}; stop_id -1 means the length limit
#include <emscripten/emscripten.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "api.h"
#include "chat_request.h"
#include "utf8_text.h"

using json = nlohmann::json;

#ifndef EMBER_WASM_CTX
#define EMBER_WASM_CTX 2048
#endif

namespace {
std::unique_ptr<tinyllama::TinyLlamaSession> g_session;
int g_ctx_cap = EMBER_WASM_CTX;  // ember_set_ctx; <= 0: the file's own context
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
    if (!g_session->chat_format()) throw std::runtime_error("the model has no chat template (raw completion only)");
    ember::Conversation c = ember::chat_conversation(r);
    if (c.user.empty() || c.user.size() > 600) throw std::runtime_error("message must be 1-600 characters");
    if (c.system.size() > 600) throw std::runtime_error("system message must be at most 600 characters");
    q.max_new = std::max(1, std::min(r.value("max_new_tokens", 64), 64));
    q.temperature = std::max(0.0f, std::min(r.value("temperature", 0.7f), 1.5f));
    q.top_k = std::max(0, std::min(r.value("top_k", 40), 200));
    q.top_p = std::max(0.0f, std::min(r.value("top_p", 0.9f), 1.0f));
    // Public-page cap: the NEWEST 16 turns (the contract drops the oldest).
    if (c.turns.size() > 16) c.turns.erase(c.turns.begin(), c.turns.end() - 16);
    q.ctx = g_session->get_config().max_position_embeddings;
    const ember::ChatPrompt p = ember::build_chat_prompt(*g_session->chat_format(), c, q.max_new, q.ctx,
                                                         [](const std::string& t) { return g_session->bpe_token_count(t); });
    if (!p.error.empty()) throw std::runtime_error(p.error);
    q.prompt = p.prompt;
    q.prompt_tokens = p.prompt_tokens;
    q.turns_used = p.turns_used;
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

EMSCRIPTEN_KEEPALIVE int ember_set_ctx(int n) {
    g_ctx_cap = n > 0 ? n : 0;
    return g_ctx_cap;
}

// The IR document the session's interpreter runs, or "" when the tuned llama path serves the file
// (the page's WebGPU path then runs llama_webgpu.js; otherwise ir_webgpu.js runs this document).
EMSCRIPTEN_KEEPALIVE const char* ember_ir_doc() {
    g_out = g_session ? g_session->ir_document() : std::string();
    return g_out.c_str();
}

EMSCRIPTEN_KEEPALIVE int ember_ctx() { return g_session ? g_session->get_config().max_position_embeddings : 0; }

EMSCRIPTEN_KEEPALIVE int ember_load(const char* path) {
    try {
        // The cap reaches the loader as the native server's EMBER_IR_CTX does: it bounds the
        // interpreter's KV state and the context the request rules use (model.cpp, graph_ir.cpp).
        if (g_ctx_cap > 0)
            setenv("EMBER_IR_CTX", std::to_string(g_ctx_cap).c_str(), 1);
        else
            unsetenv("EMBER_IR_CTX");
        g_session.reset();  // free the previous model before the next is read
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
        out["reply"] = ember::utf8_valid(g_session->generate(q.prompt, q.max_new, q.temperature, q.top_k, q.top_p, "", false));
        g_session->set_repeat_penalty(1.0f, 64);
        out["stop_reason"] = g_session->last_stop_reason();
        out["tokens"] = g_session->last_generated_tokens();
        out["prompt_tokens"] = q.prompt_tokens;
        out["history_turns_used"] = q.turns_used;
    } catch (const std::exception& e) {
        out = json{{"error", e.what()}};
    }
    g_out = ember::dump_valid(out);
    return g_out.c_str();
}

EMSCRIPTEN_KEEPALIVE const char* ember_prepare(const char* request) {
    json out;
    try {
        if (!g_session || !g_session->is_bpe()) throw std::runtime_error("no chat model loaded");
        g_req = parse_request(json::parse(request));
        if (g_req.has_seed) g_session->set_seed(g_req.seed);
        g_session->set_repeat_penalty(g_req.repeat_penalty, g_req.repeat_last_n);
        g_context = g_session->bpe_prompt_ids(g_req.prompt);  // the ids generate() runs (state blocks included)
        g_prompt_len = g_context.size();
        out["prompt_ids"] = g_context;
        out["stop_ids"] = g_session->chat_format()->stop_ids();
        out["max_new_tokens"] = g_req.max_new;
        out["ctx"] = g_req.ctx;
        out["prompt_tokens"] = g_req.prompt_tokens;
        out["history_turns_used"] = g_req.turns_used;
    } catch (const std::exception& e) {
        out = json{{"error", e.what()}};
    }
    g_out = ember::dump_valid(out);
    return g_out.c_str();
}

EMSCRIPTEN_KEEPALIVE int ember_sample(const float* logits, int n) {
    if (!g_session || n <= 0) return -1;
    std::vector<float> lg(logits, logits + n);
    const int t = g_session->sample_step(lg, g_context, g_req.temperature, g_req.top_k, g_req.top_p);
    if (!g_session->is_chat_stop(t)) g_context.push_back(t);
    return t;
}

EMSCRIPTEN_KEEPALIVE const char* ember_finish(int stop_id) {
    json out;
    try {
        if (!g_session) throw std::runtime_error("no model loaded");
        const std::vector<int> reply(g_context.begin() + static_cast<long>(g_prompt_len), g_context.end());
        out["reply"] = ember::utf8_valid(strip(g_session->bpe_decode(reply)));
        out["stop_reason"] = g_session->is_chat_eos(stop_id)    ? "eos"
                             : g_session->is_chat_stop(stop_id) ? "turn"
                                                                : "length";
        out["tokens"] = static_cast<int>(reply.size());
        out["prompt_tokens"] = g_req.prompt_tokens;
        out["history_turns_used"] = g_req.turns_used;
        g_session->set_repeat_penalty(1.0f, 64);
    } catch (const std::exception& e) {
        out = json{{"error", e.what()}};
    }
    g_out = ember::dump_valid(out);
    return g_out.c_str();
}

}  // extern "C"
