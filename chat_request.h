#pragma once
// A chat request as a prompt conversation (LOCAL_LLM_STACK.md section 3, points 3-5), shared by
// the server's /chat and /v1/chat/completions (server.cpp), the WebAssembly ember_chat /
// ember_prepare (deploy/wasm/ember_wasm.cpp) and tools/chat_template_check.cpp, so every path
// renders through the same code:
//   request JSON -> Conversation (system, (user, assistant) turns, the new user message, thinking)
//   -> the context rule (build_chat_prompt) -> ChatFormat::prompt -> chat::render of the record.
// Request errors throw RequestError, the template's refusals chat::Error; both are HTTP 400.
// Messages as keramos IR_SPEC 8.1.1 reads them (v1.4.2): every history entry is an object with a string
// content (an assistant entry with tool_calls may leave it null: ""), or a pair of strings; tool_calls (even
// null or []) and tool_call_id are refused under a record without tool turns, and under one with them on any
// message but an assistant's and a tool's respectively; a null tool_call_id answers a call without an id.
//
// Records with tool turns (paperkiln/2, chat::Template::has_tool; schema v2): the request becomes the
// whole message list (Conversation::general), tool calls and results included, plus the request's
// tool definitions, and the context rule drops the oldest whole exchange (chat::exchanges). Behaviour
// follows the resolved record's fields: under every other record (paperkiln/1 and the rest) the v1
// rules below are unchanged, and tool definitions, tool messages and tool_calls are HTTP 400.
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
    // A record with tool turns: `messages` is the whole prompt conversation (a non-empty system
    // message first, then user, assistant and tool messages; the v1 fields above unused but
    // thinking), `tools` the tool definitions (member order kept: chat::json is ordered_json).
    bool general = false;
    std::vector<chat::Message> messages;
    std::vector<chat::json> tools;
};

namespace detail {
inline int thinking_field(const nlohmann::json& v, const char* name) {
    if (!v.is_boolean()) throw RequestError(std::string(name) + " must be true or false");
    return v.get<bool>() ? 1 : 0;
}

inline bool tool_turns(const chat::Template* t) { return t && t->has_tool; }

// A request's "tools" under a record without tool turns: absent, null or [] (no definitions);
// anything else is refused.
inline void refuse_tools_v1(const nlohmann::json& r) {
    if (!r.is_object() || !r.contains("tools") || r["tools"].is_null()) return;
    if (r["tools"].is_array() && r["tools"].empty()) return;
    throw RequestError("tools are not supported: this model's chat template has no tool turns");
}

// The tool definitions: `ordered` (the request's "tools" as chat::json, member order kept; null when
// the caller has none) or, failing that, r["tools"] (member order then alphabetical). A list of
// objects; absent, null or [] is none.
inline std::vector<chat::json> tool_definitions(const nlohmann::json& r, const chat::json& ordered) {
    const chat::json src = !ordered.is_null()                     ? ordered
                           : r.is_object() && r.contains("tools") ? chat::json::parse(r["tools"].dump())
                                                                  : chat::json();
    std::vector<chat::json> out;
    if (src.is_null()) return out;
    if (!src.is_array()) throw RequestError("tools must be a list of tool definitions (JSON objects)");
    for (const auto& d : src) {
        if (!d.is_object()) throw RequestError("tools must be a list of tool definitions (JSON objects)");
        out.push_back(d);
    }
    return out;
}

// A history entry's content (IR_SPEC 8.1.1: every message has a string content). Missing or null is
// refused, except on an assistant entry with tool_calls (OpenAI's null content beside calls), where it is "".
inline std::string history_content(const nlohmann::json& h, size_t i, bool calls_may_omit = false) {
    const std::string at = "history[" + std::to_string(i) + "]";
    const bool calls = h.contains("tool_calls") && !h["tool_calls"].is_null();
    if (!h.contains("content") || h["content"].is_null()) {
        if (calls_may_omit && calls && h.value("role", "") == "assistant") return std::string();
        throw RequestError(at + " needs a string content (an assistant entry with tool_calls may leave it null)");
    }
    if (!h["content"].is_string()) throw RequestError(at + ".content must be a string");
    return h["content"].get<std::string>();
}

// A message's tool_calls or tool_call_id under a record without tool turns (IR_SPEC 8.1.1): any such member,
// null or an empty list included, is refused.
inline void refuse_tool_members_v1(const nlohmann::json& m, const std::string& at) {
    if (m.contains("tool_calls"))
        throw RequestError(at + ": tool_calls are not supported: this model's chat template has no tool turns");
    if (m.contains("tool_call_id"))
        throw RequestError(at + ": tool_call_id is not supported: this model's chat template has no tool messages");
}

// A {user, assistant} pair entry: both strings.
inline std::pair<std::string, std::string> history_pair(const nlohmann::json& h, size_t i) {
    const std::string at = "history[" + std::to_string(i) + "]";
    if (!h.contains("user") || !h["user"].is_string() || !h.contains("assistant") || !h["assistant"].is_string())
        throw RequestError(at + " needs a string role and content, or string user and assistant");
    return {h["user"].get<std::string>(), h["assistant"].get<std::string>()};
}

// One message of a tool-turn conversation, through the header's parser (its tool_calls and
// tool_call_id rules): role, content, tool_calls and tool_call_id as given; render's structure rules
// decide whether they belong where they are.
inline chat::Message tool_turn_message(const std::string& role, const std::string& content, const nlohmann::json& m) {
    chat::json x = chat::json::object();
    x["role"] = role;
    x["content"] = content;
    if (m.contains("tool_calls")) x["tool_calls"] = chat::json::parse(m["tool_calls"].dump());
    if (m.contains("tool_call_id")) x["tool_call_id"] = chat::json::parse(m["tool_call_id"].dump());
    try {
        return chat::messages_from_json(chat::json::array({x}))[0];
    } catch (const chat::Error& e) {
        throw RequestError(e.what());
    }
}

// The leading system message of a tool-turn conversation, when it is non-empty (an empty one is none).
inline void push_system(std::vector<chat::Message>& m, const std::string& s) {
    if (!s.empty()) m.insert(m.begin(), chat::Message("system", s));
}
}  // namespace detail

// The request's "tools" with the member order the client wrote: the body parsed as chat::json
// (ordered_json). Null when the body has none or is not JSON.
inline chat::json ordered_tools(const std::string& body) {
    chat::json o = chat::json::parse(body, nullptr, false);
    if (o.is_discarded() || !o.is_object() || !o.contains("tools")) return chat::json();
    return o["tools"];
}

// POST /chat under a record with tool turns: "system" or a leading {"role": "system"} history entry
// (not both, nowhere else; empty means none); history entries {"role", "content", "tool_calls"?,
// "tool_call_id"?} (roles user, assistant, tool) or {"user", "assistant"} pairs; user_input, which may
// be omitted when the history ends on a tool message; "tools"; "thinking". The order of user,
// assistant and tool messages is the record's structure rule (chat::check_structure, at render).
inline Conversation chat_conversation_tools(const nlohmann::json& r, const chat::json& tools) {
    Conversation c;
    c.general = true;
    std::string system;
    bool has_system = false;
    if (r.contains("system")) {
        if (!r["system"].is_string()) throw RequestError("system must be a string");
        system = r["system"].get<std::string>();
        has_system = true;
    }
    if (r.contains("history") && r["history"].is_array()) {
        const auto& hist = r["history"];
        for (size_t i = 0; i < hist.size(); ++i) {
            const auto& h = hist[i];
            if (!h.is_object()) throw RequestError("history[" + std::to_string(i) + "] must be an object");
            if (h.contains("role")) {
                if (!h["role"].is_string()) throw RequestError("history[" + std::to_string(i) + "].role must be a string");
                const std::string role = h["role"].get<std::string>();
                const std::string content = detail::history_content(h, i, /*calls_may_omit=*/true);
                if (role == "system") {
                    if (h.contains("tool_calls") || h.contains("tool_call_id"))
                        throw RequestError("history[" + std::to_string(i) +
                                           "]: tool_calls belong on assistant messages, tool_call_id on tool messages");
                    if (i != 0) throw RequestError("a system message is allowed only as the first history entry");
                    if (has_system)
                        throw RequestError("give the system message once: the 'system' field or a leading history entry");
                    system = content;
                    has_system = true;
                } else if (role == "user" || role == "assistant" || role == "tool") {
                    c.messages.push_back(detail::tool_turn_message(role, content, h));
                } else {
                    throw RequestError("history role '" + role +
                                       "' is not allowed here (user, assistant, tool; system only first)");
                }
            } else {
                const auto pr = detail::history_pair(h, i);
                c.messages.push_back({"user", pr.first});
                c.messages.push_back({"assistant", pr.second});
            }
        }
    }
    if (r.contains("user_input")) {
        if (!r["user_input"].is_string()) throw RequestError("user_input must be a string");
        c.messages.push_back({"user", r["user_input"].get<std::string>()});
    } else if (c.messages.empty() || c.messages.back().role != "tool") {
        throw RequestError("Missing 'user_input' field in request JSON (it may be left out only when the history "
                           "ends on a tool message)");
    }
    detail::push_system(c.messages, system);
    c.tools = detail::tool_definitions(r, tools);
    if (r.contains("thinking")) c.thinking = detail::thinking_field(r["thinking"], "thinking");
    return c;
}

// POST /chat (point 4): user_input; history as [{"role", "content"}] or [{"user", "assistant"}]. A
// role-form history pairs strictly, as mtstudio's history_turns (paperkiln 7a6c7659; section 3 point
// 3a): after an optional leading system entry, user and assistant alternate, starting with user and
// ending with assistant; anything else is a 400 naming the rule, in mtstudio's words. Pair-form
// entries are whole turns. The system message as "system" or
// a leading {"role": "system"} history entry, not both and nowhere else (empty means none); and
// "thinking" (a template's no-think generation prompt when false).
// t: the model's record; one with tool turns takes chat_conversation_tools (tools: the request's
// "tools" from ordered_tools(body), or null). Under any other record a non-empty "tools" or a
// history entry's non-empty "tool_calls" is refused, as is the tool role.
inline Conversation chat_conversation(const nlohmann::json& r, const chat::Template* t = nullptr,
                                      const chat::json& tools = chat::json()) {
    if (detail::tool_turns(t)) return chat_conversation_tools(r, tools);
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
            if (!h.is_object()) throw RequestError("history[" + std::to_string(i) + "] must be an object");
            if (h.contains("role")) {
                if (!h["role"].is_string()) throw RequestError("history[" + std::to_string(i) + "].role must be a string");
                const std::string role = h["role"].get<std::string>();
                detail::refuse_tool_members_v1(h, "history[" + std::to_string(i) + "]");
                const std::string content = detail::history_content(h, i);
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
                c.turns.push_back(detail::history_pair(h, i));
            }
        }
        if (have_user)
            throw RequestError("history ends with a user message: it must end with an assistant reply (the new "
                               "message goes in user_input)");
    }
    c.has_system = c.has_system && !c.system.empty();
    if (r.contains("thinking")) c.thinking = detail::thinking_field(r["thinking"], "thinking");
    detail::refuse_tools_v1(r);
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

namespace detail {
inline void openai_thinking(const nlohmann::json& body, Conversation& c) {
    if (body.contains("thinking")) c.thinking = thinking_field(body["thinking"], "thinking");
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object() &&
        body["chat_template_kwargs"].contains("enable_thinking"))
        c.thinking = thinking_field(body["chat_template_kwargs"]["enable_thinking"],
                                    "chat_template_kwargs.enable_thinking");
}
}  // namespace detail

// POST /v1/chat/completions under a record with tool turns: system and developer messages before the
// first other message join as under v1 (in order, "\n\n", an empty one dropped; a later one is
// refused); user, assistant and tool messages map one to one, an assistant message's tool_calls (its
// content may then be null) and a tool message's tool_call_id included; "tools" passes through; the
// last message is the user's or a tool's. tool_choice: absent, null or "auto" (the model decides).
inline Conversation openai_conversation_tools(const nlohmann::json& body, const chat::json& tools) {
    const auto& msgs = body["messages"];
    Conversation c;
    c.general = true;
    std::string system;
    for (size_t i = 0; i < msgs.size(); ++i) {
        const auto& x = msgs[i];
        if (!x.is_object() || !x.contains("role") || !x["role"].is_string())
            throw RequestError("messages[" + std::to_string(i) + "] needs a string role");
        const std::string role = x["role"].get<std::string>();
        if (role == "system" || role == "developer") {
            if (!c.messages.empty())
                throw RequestError("a " + role +
                                   " message is allowed only before the first user message (this model's chat "
                                   "template has one system message, first)");
            if (x.contains("tool_calls") || x.contains("tool_call_id"))
                throw RequestError("messages[" + std::to_string(i) +
                                   "]: tool_calls belong on assistant messages, tool_call_id on tool messages");
            const std::string s = openai_content(x, i);
            if (s.empty()) continue;
            system += (system.empty() ? "" : "\n\n") + s;
            continue;
        }
        if (role != "user" && role != "assistant" && role != "tool")
            throw RequestError("messages[" + std::to_string(i) + "] has unknown role '" + role + "'");
        const bool calls = x.contains("tool_calls") && !x["tool_calls"].is_null();
        const bool no_content = !x.contains("content") || x["content"].is_null();
        const std::string content = role == "assistant" && calls && no_content ? std::string() : openai_content(x, i);
        c.messages.push_back(detail::tool_turn_message(role, content, x));
    }
    if (c.messages.empty() || (c.messages.back().role != "user" && c.messages.back().role != "tool"))
        throw RequestError("the last message must be from the user or a tool");
    detail::push_system(c.messages, system);
    c.tools = detail::tool_definitions(body, tools);
    if (body.contains("tool_choice") && !body["tool_choice"].is_null() &&
        !(body["tool_choice"].is_string() && body["tool_choice"].get<std::string>() == "auto"))
        throw RequestError("tool_choice is not supported except \"auto\" (the model decides whether to call a tool)");
    detail::openai_thinking(body, c);
    return c;
}

// POST /v1/chat/completions (point 4): messages map one to one. System and developer messages
// before the first user message join, in order, with "\n\n" into the system message (an empty
// one is dropped); a later one, or a tool message, is refused; user and assistant messages must
// alternate from the user, and the last message must be from the user. Thinking: "thinking" or
// chat_template_kwargs.enable_thinking (llama.cpp's and vLLM's field), both booleans.
// t and tools as chat_conversation's: a record with tool turns takes openai_conversation_tools;
// under any other a non-empty "tools" or an assistant message's non-empty tool_calls is refused.
inline Conversation openai_conversation(const nlohmann::json& body, const chat::Template* t = nullptr,
                                        const chat::json& tools = chat::json()) {
    if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
        throw RequestError("body must be JSON with a non-empty 'messages' array");
    if (detail::tool_turns(t)) return openai_conversation_tools(body, tools);
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
        detail::refuse_tool_members_v1(x, "messages[" + std::to_string(i) + "]");
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
    detail::openai_thinking(body, c);
    detail::refuse_tools_v1(body);
    return c;
}

struct ChatPrompt {
    std::string prompt;
    int prompt_tokens = 0;
    int turns_used = 0;
    std::string error;  // non-empty: the newest message (with the system message) alone does not fit
};

// The context rule for a tool-turn conversation (Conversation::general): an exchange is a user
// message and everything before the next user message (replies, tool calls, tool results;
// chat::exchanges). While more than the newest exchange remains and prompt tokens + max_new > ctx,
// the OLDEST whole exchange is dropped; the system turn, with its tool definitions, is pinned. When
// the system turn and the newest exchange alone are >= ctx tokens, the request is refused naming the
// counts; an exchange is never cut. history_turns_used counts the past exchanges kept (as v1's past
// turns: for a user/assistant-only conversation the same number).
template <class Count>
ChatPrompt build_general_prompt(const ChatFormat& f, const Conversation& c, int max_new, int ctx, Count count) {
    const std::vector<chat::Message>& m = c.messages;
    const size_t sys = !m.empty() && m[0].role == "system" ? 1 : 0;
    const auto ex = chat::exchanges(m);
    auto kept = [&](size_t k) {
        if (k == 0) return m;  // the whole conversation: anything before the first user message is refused by render
        std::vector<chat::Message> out(m.begin(), m.begin() + static_cast<std::ptrdiff_t>(sys));
        if (k < ex.size()) out.insert(out.end(), m.begin() + static_cast<std::ptrdiff_t>(ex[k].first), m.end());
        return out;
    };
    auto build = [&](size_t k) { return chat::render(kept(k), f.tmpl, /*generation_prompt=*/true, c.thinking, c.tools); };
    size_t k = 0;
    ChatPrompt out;
    out.prompt = build(k);  // the whole conversation first: the record's refusals come from here
    while (k + 1 < ex.size() && count(out.prompt) + max_new > ctx) out.prompt = build(++k);
    out.prompt_tokens = count(out.prompt);
    out.turns_used = ex.empty() ? 0 : static_cast<int>(ex.size() - 1 - k);
    if (out.prompt_tokens >= ctx) {
        std::string head;  // the pinned system turn (with the tool definitions), as rendered
        for (const auto& p : chat::render_pieces(kept(k), f.tmpl, true, c.thinking, c.tools))
            if (p.role == "system" || p.kind == "bos") head += p.text;
        const int pinned = head.empty() ? 0 : count(head);
        out.error = pinned > 0 ? "the system turn and the newest exchange are " + std::to_string(out.prompt_tokens) +
                                     " tokens (the system turn " + std::to_string(pinned) +
                                     "); the model's context is " + std::to_string(ctx) +
                                     " (an exchange is never cut)"
                               : "the newest exchange is " + std::to_string(out.prompt_tokens) +
                                     " tokens; the model's context is " + std::to_string(ctx) +
                                     " (an exchange is never cut)";
    }
    return out;
}

// The context rule (point 5): the prompt from the past turns oldest first plus the new message;
// while turns remain and prompt tokens + max_new > ctx, the OLDEST whole exchange is dropped. The
// system message is pinned (fold mode: it folds into the first exchange kept) and never cut.
// count(text) is the token count generate() will run (state ids included).
template <class Count>
ChatPrompt build_chat_prompt(const ChatFormat& f, const Conversation& c, int max_new, int ctx, Count count) {
    if (c.general) return build_general_prompt(f, c, max_new, ctx, count);
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
