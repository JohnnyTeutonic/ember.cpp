// Vendored verbatim from paperkiln (microtorch) include/microtorch/chat_template.hpp
// at paperkiln 22f8fe6d, whose tests/chat_template_cases.json ember's
// tools/chat_template_check.cpp runs. ember.cpp renders every chat prompt through it
// (chat_tokenizer.h). Do not edit here: change it in paperkiln (its tests check C++/Python
// parity) and re-vendor. One local change, keramos IR_SPEC 8.1.6 at v1.4.2: a tool_call_id of
// null is accepted on a tool message that answers a call without an id (and refused against a
// call with one); 22f8fe6d refuses every null tool_call_id.
#pragma once
// Conversations and chat templates (LOCAL_LLM_STACK.md section 3, points 3-5): the C++ twin of
// tools/chat_template.py, header-only, for mtstudio and for ember.cpp to vendor (as it vendors
// bpe_tokenizer.hpp). Depends on the standard library and nlohmann/json only (as ir.hpp does;
// ember vendors nlohmann/json too). tests/test_chat_template.cpp runs the cases file
// tests/chat_template_cases.json, which tests/test_chat_template.py writes from the Python twin.
//
// A conversation is a list of messages {role: system | user | assistant, content, loss?}: at most
// one system message, first; then user and assistant strictly alternating from the user. A prompt
// conversation (a reply is wanted) ends on a user message, a training one on an assistant
// message. The tool role is reserved and refused.
//
// A template is a named, versioned record (GGUF key paperkiln.chat_template, the record as JSON):
// bos, then each message as open + content + close, the system message by its mode (fold: its
// text + join opens the first user message's content; turn: open + text + close before the first
// exchange; none: refused), then generation_prompt when a reply is wanted. Content holding one of
// the record's `reserved` strings is refused (the Python twin and its parse refuse the same).
// content "trim" strips from each message's own content exactly what Python's str.strip() strips
// (HF's Jinja trim): the 29 code points str.isspace() accepts, UTF-8 encoded. past_assistant "strip_think" renders an
// assistant message before the last user message as its content after the last "</think>", with
// leading newlines stripped, when it contains "</think>"; prompt conversations always end on a
// user message, so HF's re-wrapping of replies after the last user message never arises and is
// not implemented (render refuses a training conversation under strip_think). A close's trailing
// text after its stop token (ChatML's "<|im_end|>\n") is rendered in the next prompt and never
// generated: the stop sets hold the stop tokens only, never "\n", and engines re-render the
// history every turn.
//
// Resolution (resolve(), the same in every engine): (1) a declared paperkiln.chat_template;
// (2) ids 0-3 are "<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>" (both paperkiln BPE
// vocabularies, 8k "bpe" and 16k "llama3_bpe"): paperkiln/1; (3) tokenizer.ggml.model "word":
// paperkiln/legacy-word; (4) a third-party record whose `match` holds the sha256 of the file's
// exact tokenizer.chat_template, else whose `markers` are each a single token and whose
// `architectures` hold general.architecture (llama.cpp b6000's Gemma 3 GGUF has no
// tokenizer.chat_template); (5) no chat mode, with the reason.
//
// v1's defaults, each behind one constant shared with tools/chat_template.py: a system message
// folds into the first user message (kSystemModeV1, SYSTEM_MODE_V1 there), and the SFT loss
// covers every assistant turn (kDefaultSftTurns, DEFAULT_SFT_TURNS there).
//
// Schema v2 (paperkiln/2, LOCAL_LLM_STACK.md section 3, point 3g; never inferred, always declared):
// the system message is its own turn behind <|system|> (no fold); an assistant message may carry
// tool_calls [{id?, type?: "function", function: {name, arguments}}], each rendered after its
// content as <|tool_call|> + name + "\n" + arguments (the arguments string verbatim); exactly as
// many tool messages follow, in call order (tool_call_id, when given, equals the call's id; ids are
// never rendered), each <|tool|> + content, then an assistant message; tool definitions (JSON
// objects) render inside the system turn as Python's json.dumps(d, ensure_ascii=False) text
// (detail::dump_py; HF's tojson), joined by "\n", after the system text and "\n\n". A prompt
// conversation ends on a user message or on the last tool message of a call. parse_reply splits a
// generated reply on the <|tool_call|> id, never on text. A user/assistant-only conversation renders
// exactly as under paperkiln/1. The eight control tokens (kV2Control) are appended after the
// vocabulary's last id with GGUF token type 3; check_control_tokens refuses a file without them.
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace microtorch {
namespace chat {

using json = nlohmann::ordered_json;

inline constexpr const char* kSystemModeV1 = "fold";
inline constexpr const char* kDefaultSftTurns = "all";
inline constexpr const char* kGGUFKey = "paperkiln.chat_template";
inline constexpr const char* kV1 = "paperkiln/1";
inline constexpr const char* kLegacyWord = "paperkiln/legacy-word";
inline constexpr const char* kV2 = "paperkiln/2";
inline const std::array<const char*, 4> kV1Specials = {"<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>"};
// paperkiln/2's control tokens, appended in this order after a vocabulary's last id (the 16k
// Llama 3-scheme vocabulary: ids 16384-16391), GGUF token type 3.
inline const std::array<const char*, 8> kV2Control = {"<|system|>",  "<|tool_call|>", "<|tool|>",    "<|spare_3|>",
                                                      "<|spare_4|>", "<|spare_5|>",   "<|spare_6|>", "<|spare_7|>"};

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Template {
    std::string name;
    json version;  // an integer (1) or a string ("legacy-word")
    std::string bos, content = "raw";
    std::string system_mode = "none", system_join, system_open, system_close;
    std::string user_open, user_close, assistant_open, assistant_close;
    std::string generation_prompt;
    bool has_nothink = false;
    std::string generation_prompt_nothink;
    std::vector<std::string> stop_eos, stop_turn;
    std::string past_assistant = "keep";
    bool has_default_system = false;
    std::string default_system;
    std::vector<std::string> reserved, markers, match, architectures;
    // v2: tool turns (`tool` {open, close} and `tool_call` {open, join} come together) and tool
    // definitions (`tools` {join, sep}).
    bool has_tool = false, has_tools = false;
    std::string tool_open, tool_close, tool_call_open, tool_call_join, tools_join, tools_sep;

    std::string id() const { return name + "/" + (version.is_string() ? version.get<std::string>() : version.dump()); }

    // The record, in tools/chat_template.py's field order.
    json to_json() const {
        json r = {{"name", name}, {"version", version}, {"bos", bos}, {"content", content}};
        json s = {{"mode", system_mode}};
        if (system_mode == "fold") s["join"] = system_join;
        if (system_mode == "turn") {
            s["open"] = system_open;
            s["close"] = system_close;
        }
        r["system"] = s;
        r["user"] = {{"open", user_open}, {"close", user_close}};
        r["assistant"] = {{"open", assistant_open}, {"close", assistant_close}};
        if (has_tool) {
            r["tool"] = {{"open", tool_open}, {"close", tool_close}};
            r["tool_call"] = {{"open", tool_call_open}, {"join", tool_call_join}};
        }
        if (has_tools) r["tools"] = {{"join", tools_join}, {"sep", tools_sep}};
        r["generation_prompt"] = generation_prompt;
        r["stop"] = {{"eos", stop_eos}, {"turn", stop_turn}};
        r["past_assistant"] = past_assistant;
        r["reserved"] = reserved;
        if (!markers.empty()) r["markers"] = markers;
        if (has_default_system) r["default_system"] = default_system;
        if (has_nothink) r["generation_prompt_nothink"] = generation_prompt_nothink;
        if (!match.empty()) r["match"] = match;
        if (!architectures.empty()) r["architectures"] = architectures;
        return r;
    }

    // A record from JSON; an unknown field or value is refused (never guessed).
    template <class J>
    static Template from_json(const J& r) {
        static const char* known[] = {"name", "version", "bos", "content", "system", "user", "assistant",
                                      "generation_prompt", "generation_prompt_nothink", "stop", "past_assistant",
                                      "default_system", "reserved", "markers", "match", "architectures",
                                      "tool", "tool_call", "tools"};
        if (!r.is_object()) throw Error("chat template: the record is not a JSON object");
        for (auto it = r.begin(); it != r.end(); ++it) {
            const std::string k = it.key();
            bool ok = !k.empty() && k[0] == '_';
            for (const char* q : known) ok = ok || k == q;
            if (!ok) throw Error("chat template field '" + k + "' is not implemented");
        }
        for (const char* k : {"name", "version", "user", "assistant", "generation_prompt", "stop"})
            if (!r.contains(k)) throw Error(std::string("chat template has no '") + k + "'");
        Template t;
        t.name = r["name"].template get<std::string>();
        if (!r["version"].is_string() && !r["version"].is_number_integer())
            throw Error("chat template version must be an integer or a string");
        t.version = json::parse(r["version"].dump());
        t.bos = r.value("bos", std::string());
        t.content = r.value("content", std::string("raw"));
        if (t.content != "raw" && t.content != "trim")
            throw Error("chat template content '" + t.content + "' is not implemented (raw, trim)");
        if (r.contains("system")) {
            const auto& s = r["system"];
            t.system_mode = s.value("mode", std::string("none"));
            if (t.system_mode != "fold" && t.system_mode != "turn" && t.system_mode != "none")
                throw Error("chat template system mode '" + t.system_mode + "' is not implemented (fold, turn, none)");
            t.system_join = s.value("join", std::string());
            t.system_open = s.value("open", std::string());
            t.system_close = s.value("close", std::string());
        }
        auto side = [&](const char* k, std::string& open, std::string& close) {
            const auto& s = r[k];
            for (auto it = s.begin(); it != s.end(); ++it)
                if (it.key() != "open" && it.key() != "close")
                    throw Error(std::string("chat template ") + k + " takes open and close only");
            open = s.value("open", std::string());
            close = s.value("close", std::string());
        };
        side("user", t.user_open, t.user_close);
        side("assistant", t.assistant_open, t.assistant_close);
        auto pair = [&](const char* k, const char* a, const char* b, std::string& x, std::string& y) {
            const auto& s = r[k];
            if (!s.is_object()) throw Error(std::string("chat template ") + k + " is an object");
            for (auto it = s.begin(); it != s.end(); ++it)
                if (it.key() != a && it.key() != b)
                    throw Error(std::string("chat template ") + k + " takes " + a + " and " + b + " only");
            x = s.value(a, std::string());
            y = s.value(b, std::string());
        };
        if (r.contains("tool") != r.contains("tool_call")) throw Error("chat template tool and tool_call come together");
        if (r.contains("tools") && !r.contains("tool")) throw Error("chat template tools needs tool and tool_call");
        if (r.contains("tool")) {
            if (t.system_mode != "turn") throw Error("chat template tool turns need system mode turn");
            t.has_tool = true;
            pair("tool", "open", "close", t.tool_open, t.tool_close);
            pair("tool_call", "open", "join", t.tool_call_open, t.tool_call_join);
        }
        if (r.contains("tools")) {
            t.has_tools = true;
            pair("tools", "join", "sep", t.tools_join, t.tools_sep);
        }
        t.generation_prompt = r["generation_prompt"].template get<std::string>();
        if (r.contains("generation_prompt_nothink")) {
            t.has_nothink = true;
            t.generation_prompt_nothink = r["generation_prompt_nothink"].template get<std::string>();
        }
        auto strs = [&](const J& a) {
            std::vector<std::string> v;
            for (const auto& x : a) v.push_back(x.template get<std::string>());
            return v;
        };
        t.stop_eos = strs(r["stop"].value("eos", J::array()));
        t.stop_turn = strs(r["stop"].value("turn", J::array()));
        t.past_assistant = r.value("past_assistant", std::string("keep"));
        if (t.past_assistant != "keep" && t.past_assistant != "strip_think")
            throw Error("chat template past_assistant '" + t.past_assistant + "' is not implemented");
        if (r.contains("default_system")) {
            t.has_default_system = true;
            t.default_system = r["default_system"].template get<std::string>();
        }
        if (r.contains("reserved")) t.reserved = strs(r["reserved"]);
        if (r.contains("markers")) t.markers = strs(r["markers"]);
        if (r.contains("match")) t.match = strs(r["match"]);
        if (r.contains("architectures")) t.architectures = strs(r["architectures"]);
        return t;
    }
};

// ---- the registry ---------------------------------------------------------------------------
inline Template v1() {
    Template t;
    t.name = "paperkiln";
    t.version = 1;
    t.system_mode = kSystemModeV1;
    t.system_join = "\n\n";
    t.user_open = "<|user|>";
    t.assistant_open = "<|assistant|>";
    t.assistant_close = "<|endoftext|>";
    t.generation_prompt = "<|assistant|>";
    t.stop_eos = {"<|endoftext|>"};
    t.stop_turn = {"<|user|>", "<|assistant|>"};
    t.reserved = {"<|endoftext|>", "<|user|>", "<|assistant|>"};
    return t;
}

// Schema v2: tools/chat_template.py _v2(), field for field.
inline Template v2() {
    Template t = v1();
    t.version = 2;
    t.system_mode = "turn";
    t.system_join.clear();
    t.system_open = "<|system|>";
    t.has_tool = true;
    t.tool_open = "<|tool|>";
    t.tool_call_open = "<|tool_call|>";
    t.tool_call_join = "\n";
    t.has_tools = true;
    t.tools_join = "\n\n";
    t.tools_sep = "\n";
    t.stop_turn = {"<|user|>", "<|assistant|>", "<|system|>", "<|tool|>"};
    for (const char* c : kV2Control) t.reserved.push_back(c);
    return t;
}

// mtstudio's word-level form: "user: U assistant: A " per past turn, then "user: X assistant:".
// eos is the vocabulary's end-of-text word (GGUF-defined); "user:" and "assistant:" stop as
// multi-token sequences.
inline Template legacy_word() {
    Template t;
    t.name = "paperkiln";
    t.version = "legacy-word";
    t.system_mode = kSystemModeV1;
    t.system_join = "\n\n";
    t.user_open = "user: ";
    t.user_close = " ";
    t.assistant_open = "assistant: ";
    t.assistant_close = " ";
    t.generation_prompt = "assistant:";
    t.stop_turn = {"user:", "assistant:"};
    return t;
}

// Third-party records: ember owns them and gates them against HF apply_chat_template (G5).
inline Template chatml(const std::string& name, bool qwen25) {
    Template t;
    t.name = name;
    t.version = 1;
    t.system_mode = "turn";
    t.system_open = "<|im_start|>system\n";
    t.system_close = "<|im_end|>\n";
    t.user_open = "<|im_start|>user\n";
    t.user_close = "<|im_end|>\n";
    t.assistant_open = "<|im_start|>assistant\n";
    t.assistant_close = "<|im_end|>\n";
    t.generation_prompt = "<|im_start|>assistant\n";
    t.stop_eos = {"<|im_end|>", "<|endoftext|>"};
    t.reserved = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"};
    t.markers = {"<|im_start|>", "<|im_end|>"};
    if (qwen25) {
        t.has_default_system = true;
        t.default_system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
        t.match = {"d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4"};
        t.architectures = {"qwen2"};
    } else {
        t.past_assistant = "strip_think";
        t.has_nothink = true;
        t.generation_prompt_nothink = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        t.match = {"57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361"};
        t.architectures = {"qwen3"};
    }
    return t;
}

inline Template gemma3() {
    Template t;
    t.name = "gemma3";
    t.version = 1;
    t.bos = "<bos>";
    t.content = "trim";
    t.system_mode = "fold";
    t.system_join = "\n\n";
    t.user_open = "<start_of_turn>user\n";
    t.user_close = "<end_of_turn>\n";
    t.assistant_open = "<start_of_turn>model\n";
    t.assistant_close = "<end_of_turn>\n";
    t.generation_prompt = "<start_of_turn>model\n";
    t.stop_eos = {"<end_of_turn>", "<eos>"};
    t.reserved = {"<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>"};
    t.markers = {"<start_of_turn>", "<end_of_turn>"};
    t.match = {"7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"};
    t.architectures = {"gemma3"};
    return t;
}

inline std::vector<Template> registry() {
    return {v1(), v2(), legacy_word(), chatml("chatml-qwen2.5", true), chatml("chatml-qwen3", false), gemma3()};
}

inline Template get(const std::string& id) {
    std::string known;
    for (const auto& t : registry()) {
        if (t.id() == id) return t;
        known += (known.empty() ? "" : ", ") + t.id();
    }
    throw Error("no chat template '" + id + "' (known: " + known + ")");
}

// ---- conversations ---------------------------------------------------------------------------
// A v2 tool call: {id?, type?: "function", function: {name, arguments}}.
struct ToolCall {
    bool has_id = false;
    std::string id, name, arguments;
};

struct Message {
    std::string role, content;
    int loss = -1;  // -1 unset, 0 false, 1 true (assistant messages only)
    bool has_tool_calls = false;  // the message has a tool_calls member (refused outside v2 assistant messages)
    std::vector<ToolCall> tool_calls;
    bool has_tool_call_id = false;  // tool messages: the call's id, checked, never rendered
    std::string tool_call_id;
    bool tool_call_id_null = false;  // tool_call_id null: answers a call without an id (IR_SPEC 8.1.6)

    Message() = default;
    Message(std::string r, std::string c, int l = -1) : role(std::move(r)), content(std::move(c)), loss(l) {}
};

template <class J>
std::vector<Message> messages_from_json(const J& a) {
    if (!a.is_array()) throw Error("a conversation needs a messages list");
    std::vector<Message> out;
    for (const auto& m : a) {
        if (!m.is_object() || !m.contains("content") || !m["content"].is_string())
            throw Error("every message needs a role and a string content");
        Message x;
        x.role = m.value("role", std::string());
        x.content = m["content"].template get<std::string>();
        if (m.contains("loss")) {
            if (!m["loss"].is_boolean()) throw Error("loss is a bool on assistant messages only");
            x.loss = m["loss"].template get<bool>() ? 1 : 0;
        }
        if (m.contains("tool_calls")) {
            x.has_tool_calls = true;
            const auto& tc = m["tool_calls"];
            if (!tc.is_null()) {
                if (!tc.is_array() || tc.empty()) throw Error("tool_calls is a non-empty list");
                for (const auto& c : tc) {
                    if (!c.is_object() || !c.contains("function") || !c["function"].is_object())
                        throw Error("a tool call is {id?, type?: function, function: {name, arguments}}");
                    if (c.contains("type") && !(c["type"].is_string() && c["type"].template get<std::string>() == "function"))
                        throw Error("tool call type is not implemented (function)");
                    const auto& f = c["function"];
                    ToolCall k;
                    if (!f.contains("name") || !f["name"].is_string())
                        throw Error("a tool call's name is a non-empty string without a newline");
                    k.name = f["name"].template get<std::string>();
                    if (k.name.empty() || k.name.find('\n') != std::string::npos)
                        throw Error("a tool call's name is a non-empty string without a newline");
                    if (f.contains("arguments")) {
                        if (!f["arguments"].is_string()) throw Error("a tool call's arguments are a string (rendered verbatim)");
                        k.arguments = f["arguments"].template get<std::string>();
                    }
                    if (c.contains("id")) {
                        if (!c["id"].is_string()) throw Error("a tool call's id is a string");
                        k.has_id = true;
                        k.id = c["id"].template get<std::string>();
                    }
                    x.tool_calls.push_back(std::move(k));
                }
            }
        }
        if (m.contains("tool_call_id")) {
            x.has_tool_call_id = true;
            if (m["tool_call_id"].is_null())
                x.tool_call_id_null = true;
            else if (!m["tool_call_id"].is_string())
                throw Error("tool_call_id is a string or null");
            else
                x.tool_call_id = m["tool_call_id"].template get<std::string>();
        }
        out.push_back(std::move(x));
    }
    return out;
}

// Messages from (user, assistant) pairs, then `user` when has_user.
inline std::vector<Message> from_pairs(const std::vector<std::pair<std::string, std::string>>& turns,
                                       const std::string* user = nullptr, const std::string* system = nullptr) {
    std::vector<Message> m;
    if (system) m.push_back({"system", *system});
    for (const auto& t : turns) {
        m.push_back({"user", t.first});
        m.push_back({"assistant", t.second});
    }
    if (user) m.push_back({"user", *user});
    return m;
}

// The structure rules (tools/chat_template.py _structure). Returns the index of the first
// non-system message. tools_ok: the record defines tool turns (v2).
inline size_t check_structure(const std::vector<Message>& m, bool prompt, bool tools_ok = false) {
    if (m.empty()) throw Error("a conversation needs a non-empty messages list");
    for (const auto& x : m) {
        if (x.role == "tool" && !tools_ok) throw Error("role 'tool' is reserved: no template defines it yet");
        if (x.role != "system" && x.role != "user" && x.role != "assistant" && x.role != "tool")
            throw Error("unknown role '" + x.role + "'");
        if (x.has_tool_calls && (!tools_ok || x.role != "assistant"))
            throw Error("tool_calls belong on assistant messages of a template with tool turns");
        if (x.has_tool_call_id && x.role != "tool") throw Error("tool_call_id belongs on tool messages");
    }
    const size_t first = m[0].role == "system" ? 1 : 0;
    if (first == m.size()) throw Error("a system message needs a user message after it");
    std::string want = "user";
    size_t pending = 0, call = 0;  // results still owed to the last calls; the next call they answer
    const Message* caller = nullptr;
    for (size_t i = first; i < m.size(); ++i) {
        if (m[i].role == "system") throw Error("a system message is allowed only as the first message");
        if (m[i].role != want) {
            if (!tools_ok)
                throw Error("message " + std::to_string(i) + " is " + m[i].role + ", expected " + want +
                            " (user and assistant alternate from the user)");
            throw Error("message " + std::to_string(i) + " is " + m[i].role + ", expected " + want +
                        " (user, then assistant; an assistant message with k tool calls, then k tool messages, then "
                        "assistant)");
        }
        if (m[i].loss >= 0 && m[i].role != "assistant") throw Error("loss is a bool on assistant messages only");
        if (m[i].role == "user") {
            want = "assistant";
        } else if (m[i].role == "assistant") {
            pending = m[i].tool_calls.size();
            call = 0;
            caller = &m[i];
            want = pending ? "tool" : "user";
        } else {
            const ToolCall& c = caller->tool_calls[call++];
            --pending;
            // A call without an id counts as id null (IR_SPEC 8.1.6): a null tool_call_id answers only such a call.
            if (m[i].has_tool_call_id && m[i].tool_call_id_null && c.has_id)
                throw Error("tool_call_id null is not the call's id '" + c.id + "' (tool messages answer the calls in "
                            "order; null answers a call without an id)");
            if (m[i].has_tool_call_id && !m[i].tool_call_id_null && (!c.has_id || m[i].tool_call_id != c.id))
                throw Error("tool_call_id '" + m[i].tool_call_id + "' is not the call's id (tool messages answer the "
                            "calls in order)");
            want = pending ? "tool" : "assistant";
        }
    }
    const std::string& last = m.back().role;
    if (prompt && !(last == "user" || (last == "tool" && pending == 0)))
        throw Error(std::string("a prompt conversation ends on a user message") +
                    (tools_ok ? " or on the last tool message of a call" : ""));
    if (!prompt && last != "assistant") throw Error("a training conversation ends on an assistant message");
    return first;
}

struct Piece {
    std::string text, role, kind;  // kind: bos | open | content | call_open | call | close | generation
    int index;                     // the message's position; -1 for bos, the generation prompt and a default system
};

// Python's str.strip(): drop leading and trailing code points for which str.isspace() is true.
// Invalid UTF-8 at an end is kept (never stripped).
inline bool py_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f ||
           c == 0x3000;
}

inline std::string strip_py_ws(const std::string& s) {
    // Decode the code point starting at i; returns its length (0 when invalid).
    auto decode = [&](size_t i, uint32_t& c) -> size_t {
        const unsigned char b = static_cast<unsigned char>(s[i]);
        size_t n = b < 0x80 ? 1 : (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xe ? 3 : (b >> 3) == 0x1e ? 4 : 0;
        if (n == 0 || i + n > s.size()) return 0;
        c = n == 1 ? b : b & (0xff >> (n + 1));
        for (size_t k = 1; k < n; ++k) {
            const unsigned char x = static_cast<unsigned char>(s[i + k]);
            if ((x >> 6) != 0x2) return 0;
            c = (c << 6) | (x & 0x3f);
        }
        return n;
    };
    size_t b = 0;
    while (b < s.size()) {
        uint32_t c = 0;
        const size_t n = decode(b, c);
        if (n == 0 || !py_space(c)) break;
        b += n;
    }
    size_t e = s.size();
    while (e > b) {
        size_t st = e - 1;  // back up to the lead byte of the last code point
        while (st > b && (static_cast<unsigned char>(s[st]) >> 6) == 0x2 && e - st < 4) --st;
        uint32_t c = 0;
        const size_t n = decode(st, c);
        if (n != e - st || !py_space(c)) break;
        e = st;
    }
    return s.substr(b, e - b);
}

namespace detail {
// Python's json.dumps(x, ensure_ascii=False) with its default separators ", " and ": ", member order
// kept (HF transformers' Jinja tojson). Strings are escaped as nlohmann::json writes them, which is
// Python's escaping without ensure_ascii; non-integer numbers are refused (tool_json in Python).
inline std::string dump_py(const json& x) {
    if (x.is_object()) {
        std::string s = "{";
        bool first = true;
        for (auto it = x.begin(); it != x.end(); ++it) {
            if (!first) s += ", ";
            first = false;
            s += json(it.key()).dump() + ": " + dump_py(it.value());
        }
        return s + "}";
    }
    if (x.is_array()) {
        std::string s = "[";
        for (size_t i = 0; i < x.size(); ++i) s += (i ? ", " : "") + dump_py(x[i]);
        return s + "]";
    }
    if (x.is_number_float())
        throw Error("a tool definition holds a non-integer number (refused: its text is not portable)");
    return x.dump();
}
}  // namespace detail

// A tool definition's text (tools/chat_template.py tool_json).
inline std::string tool_json(const json& d) {
    if (!d.is_object()) throw Error("a tool definition is a JSON object");
    return detail::dump_py(d);
}

// thinking: -1 the template's default (HF's: on), 0 off (generation_prompt_nothink when the record
// has one), 1 on. tools: v2 tool definitions, rendered inside the system turn (parse requests with
// ordered_json so that member order is the request's).
inline std::vector<Piece> render_pieces(const std::vector<Message>& m, const Template& t, bool generation_prompt,
                                        int thinking = -1, const std::vector<json>& tools = {}) {
    if (!tools.empty() && !t.has_tools) throw Error("this model's template has no tool definitions");
    const size_t first = check_structure(m, generation_prompt, t.has_tool);
    const bool trim = t.content == "trim";
    auto check_reserved = [&](const std::string& s) {
        for (const auto& r : t.reserved)
            if (!r.empty() && s.find(r) != std::string::npos)
                throw Error("content holds the template's reserved text '" + r + "'");
        return s;
    };
    auto body = [&](std::string s) {
        if (trim) s = strip_py_ws(s);
        check_reserved(s);
        return s;
    };
    std::vector<Piece> out;
    auto add = [&](const std::string& text, const std::string& role, const char* kind, int idx) {
        if (!text.empty()) out.push_back({text, role, kind, idx});
    };
    add(t.bos, "", "bos", -1);
    bool have_sys = first == 1;
    std::string sys = have_sys ? m[0].content : std::string();
    const int sys_idx = have_sys ? 0 : -1;
    if (!have_sys && t.has_default_system && !t.default_system.empty()) {
        have_sys = true;
        sys = t.default_system;
    }
    std::string defs;
    for (size_t k = 0; k < tools.size(); ++k) defs += (k ? t.tools_sep : "") + check_reserved(tool_json(tools[k]));
    if (!tools.empty()) {
        const std::string head = have_sys ? body(sys) : std::string();
        sys = (head.empty() ? std::string() : head + t.tools_join) + defs;
        have_sys = true;
    }
    bool fold = false;
    std::string fold_text;
    if (have_sys) {
        if (t.system_mode == "none") throw Error("this model's template has no system role");
        if (t.system_mode == "turn") {
            add(t.system_open, "system", "open", sys_idx);
            add(tools.empty() ? body(sys) : sys, "system", "content", sys_idx);
            add(t.system_close, "system", "close", sys_idx);
        } else {
            check_reserved(sys);
            fold = true;
            fold_text = sys + t.system_join;
        }
    }
    size_t last_user = first;
    for (size_t i = first; i < m.size(); ++i)
        if (m[i].role == "user") last_user = i;
    for (size_t i = first; i < m.size(); ++i) {
        const std::string& role = m[i].role;
        const bool user = role == "user", tool = role == "tool";
        std::string text = body(m[i].content);
        if (user && fold) {
            text = fold_text + text;
            fold = false;
        }
        if (role == "assistant" && t.past_assistant == "strip_think" && i < last_user) {
            const size_t p = text.rfind("</think>");
            if (p != std::string::npos) {
                text = text.substr(p + 8);
                text.erase(0, text.find_first_not_of('\n') == std::string::npos ? text.size()
                                                                                    : text.find_first_not_of('\n'));
            }
        }
        const int idx = static_cast<int>(i);
        add(user ? t.user_open : tool ? t.tool_open : t.assistant_open, role, "open", idx);
        add(text, role, "content", idx);
        for (const auto& c : m[i].tool_calls) {
            add(t.tool_call_open, role, "call_open", idx);
            add(check_reserved(c.name) + t.tool_call_join + check_reserved(c.arguments), role, "call", idx);
        }
        add(user ? t.user_close : tool ? t.tool_close : t.assistant_close, role, "close", idx);
    }
    if (generation_prompt)
        add(thinking == 0 && t.has_nothink ? t.generation_prompt_nothink : t.generation_prompt, "", "generation", -1);
    else if (t.past_assistant == "strip_think")
        throw Error("a strip_think template renders prompt conversations only (HF re-wraps the reasoning of "
                    "replies after the last user message; prompts never have one)");
    return out;
}

inline std::string render(const std::vector<Message>& m, const Template& t, bool generation_prompt, int thinking = -1,
                          const std::vector<json>& tools = {}) {
    std::string s;
    for (const auto& p : render_pieces(m, t, generation_prompt, thinking, tools)) s += p.text;
    return s;
}

// The exchanges after the system message, as [start, end) message index ranges: a user message and
// everything before the next user message (replies, tool calls, tool results). The context rule
// (section 3, point 5) drops the oldest whole exchange; the system turn is pinned.
inline std::vector<std::pair<size_t, size_t>> exchanges(const std::vector<Message>& m) {
    std::vector<std::pair<size_t, size_t>> out;
    for (size_t i = 0; i < m.size(); ++i)
        if (m[i].role == "user") {
            if (!out.empty()) out.back().second = i;
            out.push_back({i, m.size()});
        }
    return out;
}

// A generated v2 reply (ids without the stop token), split on ids, never on text: cut at every
// call_id (the <|tool_call|> control token) and each part decoded. Part 0 is the content; every
// later part must be name + "\n" + arguments with a non-empty name; the calls get ids call_0,
// call_1, ... in order. parsed false: a part does not parse (the engine returns the whole decoded
// reply as content). BPE pieces that spell "<|tool_call|>" are content.
struct ParsedReply {
    bool parsed = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
};

template <class Decode>
ParsedReply parse_reply(const std::vector<int>& ids, int call_id, Decode decode) {
    std::vector<std::vector<int>> parts(1);
    for (int x : ids) {
        if (x == call_id) parts.emplace_back();
        else parts.back().push_back(x);
    }
    ParsedReply r;
    r.content = decode(parts[0]);
    for (size_t k = 1; k < parts.size(); ++k) {
        const std::string text = decode(parts[k]);
        const size_t nl = text.find('\n');
        if (nl == std::string::npos || nl == 0) return ParsedReply{false, {}, {}};
        ToolCall c;
        c.has_id = true;
        c.id = "call_" + std::to_string(k - 1);
        c.name = text.substr(0, nl);
        c.arguments = text.substr(nl + 1);
        r.tool_calls.push_back(std::move(c));
    }
    r.parsed = true;
    return r;
}

// A paperkiln/2 file's vocabulary must hold each control token as one token (tokens: the GGUF's
// tokenizer.ggml.tokens; types: tokenizer.ggml.token_type, 3 = control); otherwise it is refused.
inline void check_control_tokens(const Template& t, const std::vector<std::string>& tokens,
                                 const std::vector<int32_t>& types) {
    if (!t.has_tool) return;
    for (const auto& r : t.reserved) {
        bool ok = false;
        for (size_t i = 0; i < tokens.size() && !ok; ++i)
            ok = tokens[i] == r && i < types.size() && types[i] == 3;
        if (!ok) throw Error("chat template " + t.id() + " needs '" + r + "' as a control token of the vocabulary");
    }
}

// ---- Jinja (llama.cpp --jinja, LM Studio) and an Ollama Modelfile, for paperkiln records -----
namespace detail {
inline std::string jstr(const std::string& s) {
    std::string o = "'";
    for (char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == '\'') o += "\\'";
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o + "'";
}
inline std::string go(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s.compare(i, 2, "{{") == 0) {
            o += "{{ \"{{\" }}";
            ++i;
        } else {
            o += s[i];
        }
    }
    return o;
}
inline void paperkiln_only(const Template& t, const char* what) {
    if (t.name != "paperkiln") throw Error(std::string(what) + " writes paperkiln records only");
}
}  // namespace detail

// tokenizer.chat_template: the same text as render() for HF-style messages (tools/chat_template.py
// jinja(); the two are compared in the cases file).
namespace detail {
// tools/chat_template.py _jinja_tools: a record with tool turns (paperkiln/2).
inline std::string jinja_tools(const Template& t) {
    std::string j;
    if (!t.bos.empty()) j += "{{- " + jstr(t.bos) + " -}}";
    j += "{%- if messages and messages[0]['role'] == 'system' -%}"
         "{%- set sys = messages[0]['content'] -%}{%- set msgs = messages[1:] -%}"
         "{%- else -%}{%- set sys = none -%}{%- set msgs = messages -%}{%- endif -%}";
    if (t.has_tools)
        j += "{%- if tools -%}{%- set ns = namespace(d='') -%}{%- for x in tools -%}"
             "{%- if not loop.first -%}{%- set ns.d = ns.d + " + jstr(t.tools_sep) + " -%}{%- endif -%}"
             "{%- set ns.d = ns.d + (x | tojson) -%}{%- endfor -%}"
             "{%- if sys -%}{%- set sys = sys + " + jstr(t.tools_join) + " + ns.d -%}"
             "{%- else -%}{%- set sys = ns.d -%}{%- endif -%}{%- endif -%}";
    j += "{%- if sys is not none -%}{{- " + jstr(t.system_open) + " + sys + " + jstr(t.system_close) +
         " -}}{%- endif -%}";
    j += "{%- for m in msgs -%}"
         "{%- if m['role'] == 'user' -%}{{- " + jstr(t.user_open) + " + m['content'] + " + jstr(t.user_close) + " -}}"
         "{%- elif m['role'] == 'assistant' -%}{{- " + jstr(t.assistant_open) + " + m['content'] -}}"
         "{%- for c in (m['tool_calls'] or []) -%}{{- " + jstr(t.tool_call_open) + " + c['function']['name'] + " +
         jstr(t.tool_call_join) + " -}}"
         "{%- if c['function']['arguments'] is string -%}{{- c['function']['arguments'] -}}"
         "{%- else -%}{{- c['function']['arguments'] | tojson -}}{%- endif -%}{%- endfor -%}"
         "{{- " + jstr(t.assistant_close) + " -}}"
         "{%- elif m['role'] == 'tool' -%}{{- " + jstr(t.tool_open) + " + m['content'] + " + jstr(t.tool_close) + " -}}"
         "{%- else -%}{{ raise_exception('roles: one leading system message, then user, assistant and tool') }}"
         "{%- endif -%}{%- endfor -%}"
         "{%- if add_generation_prompt -%}{{- " + jstr(t.generation_prompt) + " -}}{%- endif -%}";
    return j;
}

// tools/chat_template.py _modelfile_tools.
inline std::string modelfile_tools(const std::string& gguf_name, const Template& t) {
    std::string head = go(t.bos);
    head += "{{ if or .System .Tools }}" + go(t.system_open) + "{{ if .System }}{{ .System }}{{ end }}"
            "{{ if .Tools }}{{ if .System }}" + go(t.tools_join) + "{{ end }}"
            "{{ range $i, $t := .Tools }}{{ if $i }}" + go(t.tools_sep) + "{{ end }}{{ json $t }}{{ end }}"
            "{{ end }}" + go(t.system_close) + "{{ end }}";
    const std::string msgs =
        "{{ range .Messages }}{{ if eq .Role \"user\" }}" + go(t.user_open) + "{{ .Content }}" + go(t.user_close) +
        "{{ else if eq .Role \"assistant\" }}" + go(t.assistant_open) + "{{ .Content }}"
        "{{ range .ToolCalls }}" + go(t.tool_call_open) + "{{ .Function.Name }}" + go(t.tool_call_join) +
        "{{ json .Function.Arguments }}{{ end }}" + go(t.assistant_close) +
        "{{ else if eq .Role \"tool\" }}" + go(t.tool_open) + "{{ .Content }}" + go(t.tool_close) +
        "{{ end }}{{ end }}" + go(t.generation_prompt);
    std::string s = "# Ollama Modelfile for " + gguf_name + ": chat template " + t.id() +
                    " (LOCAL_LLM_STACK.md section 3, point 3).\nFROM ./" + gguf_name + "\nTEMPLATE \"\"\"" + head +
                    msgs + "\"\"\"\n";
    std::vector<std::string> stops = t.stop_eos;
    stops.insert(stops.end(), t.stop_turn.begin(), t.stop_turn.end());
    for (const auto& x : stops) s += "PARAMETER stop " + json(x).dump() + "\n";
    return s;
}
}  // namespace detail

inline std::string jinja(const Template& t) {
    using detail::jstr;
    detail::paperkiln_only(t, "jinja()");
    if (t.content != "raw" || t.past_assistant != "keep" || t.has_default_system)
        throw Error("jinja() covers raw content, kept replies and no default system");
    if (t.has_tool) return detail::jinja_tools(t);
    std::string j;
    if (!t.bos.empty()) j += "{{- " + jstr(t.bos) + " -}}";
    j += "{%- if messages and messages[0]['role'] == 'system' -%}"
         "{%- set sys = messages[0]['content'] -%}{%- set msgs = messages[1:] -%}"
         "{%- else -%}{%- set sys = none -%}{%- set msgs = messages -%}{%- endif -%}";
    if (t.system_mode == "none")
        j += "{%- if sys is not none -%}{{ raise_exception('this model\\'s template has no system role') }}"
             "{%- endif -%}";
    else if (t.system_mode == "turn")
        j += "{%- if sys is not none -%}{{- " + jstr(t.system_open) + " + sys + " + jstr(t.system_close) +
             " -}}{%- endif -%}";
    j += "{%- for m in msgs -%}"
         "{%- if (m['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
         "{{ raise_exception('user and assistant messages alternate from the user') }}{%- endif -%}"
         "{%- if m['role'] == 'user' -%}{{- " + jstr(t.user_open) + " -}}";
    if (t.system_mode == "fold")
        j += "{%- if loop.first and sys is not none -%}{{- sys + " + jstr(t.system_join) + " -}}{%- endif -%}";
    j += "{{- m['content'] + " + jstr(t.user_close) + " -}}"
         "{%- else -%}{{- " + jstr(t.assistant_open) + " + m['content'] + " + jstr(t.assistant_close) + " -}}"
         "{%- endif -%}{%- endfor -%}"
         "{%- if add_generation_prompt -%}{{- " + jstr(t.generation_prompt) + " -}}{%- endif -%}";
    return j;
}

inline std::string modelfile(const std::string& gguf_name, const Template& t) {
    using detail::go;
    detail::paperkiln_only(t, "modelfile()");
    if (t.has_tool) return detail::modelfile_tools(gguf_name, t);
    // No action trims whitespace, so the record's text is emitted exactly; the fold goes to the first user
    // message (Ollama keeps a system message in .Messages as well as in .System), as in tools/chat_template.py.
    std::string msgs = "{{ $first := true }}{{ range .Messages }}{{ if eq .Role \"user\" }}" + go(t.user_open);
    if (t.system_mode == "fold") msgs += "{{ if and $first $.System }}{{ $.System }}" + go(t.system_join) + "{{ end }}";
    msgs += "{{ $first = false }}{{ .Content }}" + go(t.user_close) + "{{ else if eq .Role \"assistant\" }}" +
            go(t.assistant_open) + "{{ .Content }}" + go(t.assistant_close) + "{{ end }}{{ end }}" + go(t.generation_prompt);
    std::string head = go(t.bos);
    if (t.system_mode == "turn")
        head += "{{ if .System }}" + go(t.system_open) + "{{ .System }}" + go(t.system_close) + "{{ end }}";
    std::string s = "# Ollama Modelfile for " + gguf_name + ": chat template " + t.id() +
                    " (LOCAL_LLM_STACK.md section 3, point 3).\nFROM ./" + gguf_name + "\nTEMPLATE \"\"\"" + head +
                    msgs + "\"\"\"\n";
    std::vector<std::string> stops = t.stop_eos;
    stops.insert(stops.end(), t.stop_turn.begin(), t.stop_turn.end());
    for (const auto& x : stops) s += "PARAMETER stop " + json(x).dump() + "\n";
    return s;
}

// ---- resolution ------------------------------------------------------------------------------
namespace detail {
inline std::string sha256_hex(const std::string& msg) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::string m = msg;
    const uint64_t bits = static_cast<uint64_t>(msg.size()) * 8;
    m += static_cast<char>(0x80);
    while (m.size() % 64 != 56) m += '\0';
    for (int i = 7; i >= 0; --i) m += static_cast<char>((bits >> (8 * i)) & 0xff);
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(uint8_t(m[off + 4 * i])) << 24) | (uint32_t(uint8_t(m[off + 4 * i + 1])) << 16) |
                   (uint32_t(uint8_t(m[off + 4 * i + 2])) << 8) | uint32_t(uint8_t(m[off + 4 * i + 3]));
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    char buf[65];
    for (int i = 0; i < 8; ++i) std::snprintf(buf + 8 * i, 9, "%08x", h[i]);
    return std::string(buf, 64);
}
}  // namespace detail

struct Resolution {
    bool chat = false;
    Template tmpl;
    std::string why;
};

// declared: the GGUF's paperkiln.chat_template ("" when absent); tokens: tokenizer.ggml.tokens;
// model: tokenizer.ggml.model; chat_template: tokenizer.chat_template ("" when absent); arch:
// general.architecture.
inline Resolution resolve(const std::string& declared, const std::vector<std::string>& tokens, const std::string& model,
                          const std::string& chat_template, const std::string& arch) {
    Resolution r;
    r.chat = true;
    if (!declared.empty()) {
        r.tmpl = Template::from_json(json::parse(declared));
        r.why = "declared";
        return r;
    }
    bool v1specials = tokens.size() >= 4;
    for (size_t i = 0; v1specials && i < 4; ++i) v1specials = tokens[i] == kV1Specials[i];
    if (v1specials) {
        r.tmpl = v1();
        r.why = "paperkiln specials at ids 0-3";
        return r;
    }
    if (model == "word") {
        r.tmpl = legacy_word();
        r.why = "word-level vocabulary";
        return r;
    }
    const auto reg = registry();
    if (!chat_template.empty()) {
        const std::string h = detail::sha256_hex(chat_template);
        for (const auto& t : reg)
            for (const auto& m : t.match)
                if (m == h) {
                    r.tmpl = t;
                    r.why = "tokenizer.chat_template sha256 " + h.substr(0, 12);
                    return r;
                }
    }
    for (const auto& t : reg) {
        if (t.markers.empty()) continue;
        bool arch_ok = false;
        for (const auto& a : t.architectures) arch_ok = arch_ok || a == arch;
        bool all = arch_ok;
        for (const auto& mk : t.markers) {
            bool found = false;
            for (const auto& tok : tokens) found = found || tok == mk;
            all = all && found;
        }
        if (all) {
            r.tmpl = t;
            r.why = "markers and architecture " + arch;
            return r;
        }
    }
    r.chat = false;
    r.why = "no declared template, no paperkiln specials, and no known tokenizer.chat_template or markers: raw "
            "completion only";
    return r;
}

}  // namespace chat
}  // namespace microtorch
