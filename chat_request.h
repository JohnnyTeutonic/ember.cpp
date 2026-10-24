#pragma once
// A chat request as a prompt conversation (LOCAL_LLM_STACK.md section 3, points 3-5), shared by
// the server's /chat and /v1/chat/completions (server.cpp), the WebAssembly ember_chat /
// ember_prepare (deploy/wasm/ember_wasm.cpp) and tools/chat_template_check.cpp, so every path
// renders through the same code:
//   request JSON -> Conversation (system, (user, assistant) turns, the new user message, thinking)
//   -> the context rule (build_chat_prompt) -> ChatFormat::prompt -> chat::render of the record.
// Request errors throw RequestError, the template's refusals chat::Error; both are HTTP 400.
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat_tokenizer.h"

namespace ember {

struct RequestError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Conversation {
    std::string system;  // used when has_system
    bool has_system = false;
    std::vector<std::pair<std::string, std::string>> turns;  // (user, assistant), oldest first
    std::string user;
    int thinking = -1;  // -1 the template's default, 0 off, 1 on
};

namespace detail {
inline int thinking_field(const nlohmann::json& v, const char* name) {
    if (!v.is_boolean()) throw RequestError(std::string(name) + " must be true or false");
    return v.get<bool>() ? 1 : 0;
}
}  // namespace detail

// POST /chat (point 4): user_input; history as [{"role", "content"}] or [{"user", "assistant"}]. A
// role-form history pairs strictly, as mtstudio's history_turns (paperkiln 7a6c7659; section 3 point
// 3a): after an optional leading system entry, user and assistant alternate, starting with user and
// ending with assistant; anything else is a 400 naming the rule, in mtstudio's words. Pair-form
// entries are whole turns. The system message as "system" or
// a leading {"role": "system"} history entry, not both and nowhere else (empty means none); and
// "thinking" (a template's no-think generation prompt when false).
inline Conversation chat_conversation(const nlohmann::json& r) {
    Conversation c;
    if (!r.contains("user_input") || !r["user_input"].is_string())
        throw RequestError("Missing 'user_input' field in request JSON");
    c.user = r["user_input"].get<std::string>();
    if (r.contains("system")) {
        if (!r["system"].is_string()) throw RequestError("system must be a string");
        c.system = r["system"].get<std::string>();
        c.has_system = true;
    }
    if (r.contains("history") && r["history"].is_array()) {
        const auto& hist = r["history"];
        std::string pending_user;
        bool have_user = false;
        for (size_t i = 0; i < hist.size(); ++i) {
            const auto& h = hist[i];
            if (!h.is_object()) continue;
            if (h.contains("role")) {
                const std::string role = h.value("role", "");
                const std::string content = h.value("content", "");
                if (role == "system") {
                    if (i != 0) throw RequestError("a system message is allowed only as the first history entry");
                    if (c.has_system)
                        throw RequestError("give the system message once: the 'system' field or a leading history entry");
                    c.system = content;
                    c.has_system = true;
                } else if (role == "user") {
                    if (have_user)
                        throw RequestError("history has two user messages in a row: user and assistant messages "
                                           "must alternate (section 3 point 3a)");
                    pending_user = content;
                    have_user = true;
                } else if (role == "assistant") {
                    if (!have_user)
                        throw RequestError("history has an assistant message without a user message before it: user "
                                           "and assistant messages must alternate, starting with user (section 3 "
                                           "point 3a)");
                    c.turns.emplace_back(pending_user, content);
                    have_user = false;
                } else {
                    throw RequestError("history role '" + role +
                                       "' is not allowed here (user, assistant; system only first)");
                }
            } else {
                if (have_user)
                    throw RequestError("history has a user message without its assistant reply before a {user, "
                                       "assistant} pair");
                c.turns.emplace_back(h.value("user", ""), h.value("assistant", ""));
            }
        }
        if (have_user)
            throw RequestError("history ends with a user message: it must end with an assistant reply (the new "
                               "message goes in user_input)");
    }
    c.has_system = c.has_system && !c.system.empty();
    if (r.contains("thinking")) c.thinking = detail::thinking_field(r["thinking"], "thinking");
    return c;
}

// The text of an OpenAI message's content: a string, or text parts joined; anything else refused.
inline std::string openai_content(const nlohmann::json& m, size_t i) {
    const std::string at = "messages[" + std::to_string(i) + "]";
    if (!m.contains("content") || m["content"].is_null()) throw RequestError(at + " has no content");
    const auto& c = m["content"];
    if (c.is_string()) return c.get<std::string>();
    if (!c.is_array()) throw RequestError(at + ".content must be a string or a list of text parts");
    std::string s;
    for (const auto& part : c) {
        if (!part.is_object() || part.value("type", "") != "text" || !part.contains("text") || !part["text"].is_string())
            throw RequestError(at + ".content: only text parts are supported (this model takes text only)");
        s += part["text"].get<std::string>();
    }
    return s;
}

// POST /v1/chat/completions (point 4): messages map one to one. System and developer messages
// before the first user message join, in order, with "\n\n" into the system message (an empty
// one is dropped); a later one, or a tool message, is refused; user and assistant messages must
// alternate from the user, and the last message must be from the user. Thinking: "thinking" or
// chat_template_kwargs.enable_thinking (llama.cpp's and vLLM's field), both booleans.
inline Conversation openai_conversation(const nlohmann::json& body) {
    if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
        throw RequestError("body must be JSON with a non-empty 'messages' array");
    const auto& msgs = body["messages"];
    Conversation c;
    std::vector<chat::Message> m;
    for (size_t i = 0; i < msgs.size(); ++i) {
        const auto& x = msgs[i];
        if (!x.is_object() || !x.contains("role") || !x["role"].is_string())
            throw RequestError("messages[" + std::to_string(i) + "] needs a string role");
        const std::string role = x["role"].get<std::string>();
        if (role == "tool")
            throw RequestError("tool messages are not supported: this model's chat template has no tool role");
        if (role == "system" || role == "developer") {
            if (!m.empty())
                throw RequestError("a " + role +
                                   " message is allowed only before the first user message (this model's chat "
                                   "template has one system message, first)");
            const std::string s = openai_content(x, i);
            if (s.empty()) continue;
            c.system += (c.has_system ? "\n\n" : "") + s;
            c.has_system = true;
            continue;
        }
        if (role != "user" && role != "assistant")
            throw RequestError("messages[" + std::to_string(i) + "] has unknown role '" + role + "'");
        m.push_back({role, openai_content(x, i)});
    }
    if (m.empty() || m.back().role != "user") throw RequestError("the last message must be from the user");
    for (size_t i = 0; i < m.size(); ++i)
        if (m[i].role != (i % 2 == 0 ? "user" : "assistant"))
            throw RequestError("messages must alternate user and assistant, starting from the user (message " +
                               std::to_string(i) + " is " + m[i].role + ")");
    for (size_t i = 0; i + 1 < m.size(); i += 2) c.turns.emplace_back(m[i].content, m[i + 1].content);
    c.user = m.back().content;
    if (body.contains("thinking")) c.thinking = detail::thinking_field(body["thinking"], "thinking");
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object() &&
        body["chat_template_kwargs"].contains("enable_thinking"))
        c.thinking = detail::thinking_field(body["chat_template_kwargs"]["enable_thinking"],
                                            "chat_template_kwargs.enable_thinking");
    return c;
}

struct ChatPrompt {
    std::string prompt;
    int prompt_tokens = 0;
    int turns_used = 0;
    std::string error;  // non-empty: the newest message (with the system message) alone does not fit
};

// The context rule (point 5): the prompt from the past turns oldest first plus the new message;
// while turns remain and prompt tokens + max_new > ctx, the OLDEST whole exchange is dropped. The
// system message is pinned (fold mode: it folds into the first exchange kept) and never cut.
// count(text) is the token count generate() will run (state ids included).
template <class Count>
ChatPrompt build_chat_prompt(const ChatFormat& f, const Conversation& c, int max_new, int ctx, Count count) {
    const std::string* sys = c.has_system ? &c.system : nullptr;
    auto build = [&](size_t from) { return f.prompt(c.turns, from, c.user, sys, c.thinking); };
    size_t from = 0;
    ChatPrompt out;
    out.prompt = build(from);
    while (from < c.turns.size() && count(out.prompt) + max_new > ctx) out.prompt = build(++from);
    out.prompt_tokens = count(out.prompt);
    out.turns_used = static_cast<int>(c.turns.size() - from);
    if (out.prompt_tokens >= ctx)
        out.error = "message is " + std::to_string(out.prompt_tokens) + " tokens; the model's context is " +
                    std::to_string(ctx);
    return out;
}

}  // namespace ember
