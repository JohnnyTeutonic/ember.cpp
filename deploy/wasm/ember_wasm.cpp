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
//                            "max_new_tokens"} -> {"reply", "stop_reason",
//                            "tokens", "prompt_tokens", "history_turns_used"}
//                            or {"error"}
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
        const json r = json::parse(request);
        const std::string user = r.value("user_input", "");
        // Public-page caps, as the server's public mode.
        if (user.empty() || user.size() > 600) throw std::runtime_error("message must be 1-600 characters");
        const int max_new = std::max(1, std::min(r.value("max_new_tokens", 64), 64));
        const float temperature = std::max(0.0f, std::min(r.value("temperature", 0.7f), 1.5f));
        const int top_k = std::max(0, std::min(r.value("top_k", 40), 200));
        const float top_p = std::max(0.0f, std::min(r.value("top_p", 0.9f), 1.0f));
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
        const int ctx = g_session->get_config().max_position_embeddings;
        size_t from = 0;
        std::string prompt = build(from);
        while (from < turns.size() && g_session->bpe_token_count(prompt) + max_new > ctx) prompt = build(++from);
        const int prompt_tokens = g_session->bpe_token_count(prompt);
        // The context contract (LOCAL_LLM_STACK.md section 3): reject only a
        // prompt that fills the context; generation stops at the limit.
        if (prompt_tokens >= ctx)
            throw std::runtime_error("message is " + std::to_string(prompt_tokens) + " tokens; the model's context is " +
                                     std::to_string(ctx));
        if (r.contains("seed")) g_session->set_seed(r["seed"].get<unsigned>());
        // The contract's repetition penalty (point 4), clamped as a public page:
        // p in [1, 2], n in [0, ctx].
        const float repeat_penalty = std::max(1.0f, std::min(r.value("repeat_penalty", 1.0f), 2.0f));
        const int repeat_last_n = std::max(0, std::min(r.value("repeat_last_n", 64), ctx));
        g_session->set_repeat_penalty(repeat_penalty, repeat_last_n);
        out["reply"] = g_session->generate(prompt, max_new, temperature, top_k, top_p, "", false);
        g_session->set_repeat_penalty(1.0f, 64);
        out["stop_reason"] = g_session->last_stop_reason();
        out["tokens"] = g_session->last_generated_tokens();
        out["prompt_tokens"] = prompt_tokens;
        out["history_turns_used"] = static_cast<int>(turns.size() - from);
    } catch (const std::exception& e) {
        out = json{{"error", e.what()}};
    }
    g_out = out.dump();
    return g_out.c_str();
}

}  // extern "C"
