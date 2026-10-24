// Vendored verbatim from paperkiln (microtorch) include/microtorch/chat_template.hpp
// at paperkiln 7a6c7659, whose tests/chat_template_cases.json ember's
// tools/chat_template_check.cpp runs. ember.cpp renders every chat prompt through it
// (chat_tokenizer.h). Do not edit here: change it in paperkiln (its tests check C++/Python
// parity) and re-vendor.
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
// Jonathan's decisions pending (24 Oct 2026), each behind one constant: kSystemModeV1 (fold or
// none) here and in tools/chat_template.py (SYSTEM_MODE_V1); the default loss policy
// (DEFAULT_SFT_TURNS there, kDefaultSftTurns here).
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
inline const std::array<const char*, 4> kV1Specials = {"<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>"};

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
                                      "default_system", "reserved", "markers", "match", "architectures"};
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
    return {v1(), legacy_word(), chatml("chatml-qwen2.5", true), chatml("chatml-qwen3", false), gemma3()};
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
struct Message {
    std::string role, content;
    int loss = -1;  // -1 unset, 0 false, 1 true (assistant messages only)
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
// non-system message.
inline size_t check_structure(const std::vector<Message>& m, bool prompt) {
    if (m.empty()) throw Error("a conversation needs a non-empty messages list");
    for (const auto& x : m) {
        if (x.role == "tool") throw Error("role 'tool' is reserved: no template defines it yet");
        if (x.role != "system" && x.role != "user" && x.role != "assistant")
            throw Error("unknown role '" + x.role + "'");
    }
    const size_t first = m[0].role == "system" ? 1 : 0;
    if (first == m.size()) throw Error("a system message needs a user message after it");
    for (size_t i = first; i < m.size(); ++i) {
        const char* want = (i - first) % 2 == 0 ? "user" : "assistant";
        if (m[i].role == "system") throw Error("a system message is allowed only as the first message");
        if (m[i].role != want)
            throw Error("message " + std::to_string(i) + " is " + m[i].role + ", expected " + want +
                        " (user and assistant alternate from the user)");
        if (m[i].loss >= 0 && m[i].role != "assistant") throw Error("loss is a bool on assistant messages only");
    }
    const std::string& last = m.back().role;
    if (prompt && last != "user") throw Error("a prompt conversation ends on a user message");
    if (!prompt && last != "assistant") throw Error("a training conversation ends on an assistant message");
    return first;
}

struct Piece {
    std::string text, role, kind;  // kind: bos | open | content | close | generation
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

// thinking: -1 the template's default (HF's: on), 0 off (generation_prompt_nothink when the record
// has one), 1 on.
inline std::vector<Piece> render_pieces(const std::vector<Message>& m, const Template& t, bool generation_prompt,
                                        int thinking = -1) {
    const size_t first = check_structure(m, generation_prompt);
    const bool trim = t.content == "trim";
    auto check_reserved = [&](const std::string& s) {
        for (const auto& r : t.reserved)
            if (!r.empty() && s.find(r) != std::string::npos)
                throw Error("content holds the template's reserved text '" + r + "'");
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
    bool fold = false;
    std::string fold_text;
    if (have_sys) {
        if (t.system_mode == "none") throw Error("this model's template has no system role");
        if (t.system_mode == "turn") {
            add(t.system_open, "system", "open", sys_idx);
            add(body(sys), "system", "content", sys_idx);
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
        const bool user = m[i].role == "user";
        std::string text = body(m[i].content);
        if (user && fold) {
            text = fold_text + text;
            fold = false;
        }
        if (!user && t.past_assistant == "strip_think" && i < last_user) {
            const size_t p = text.rfind("</think>");
            if (p != std::string::npos) {
                text = text.substr(p + 8);
                text.erase(0, text.find_first_not_of('\n') == std::string::npos ? text.size()
                                                                                    : text.find_first_not_of('\n'));
            }
        }
        add(user ? t.user_open : t.assistant_open, m[i].role, "open", static_cast<int>(i));
        add(text, m[i].role, "content", static_cast<int>(i));
        add(user ? t.user_close : t.assistant_close, m[i].role, "close", static_cast<int>(i));
    }
    if (generation_prompt)
        add(thinking == 0 && t.has_nothink ? t.generation_prompt_nothink : t.generation_prompt, "", "generation", -1);
    else if (t.past_assistant == "strip_think")
        throw Error("a strip_think template renders prompt conversations only (HF re-wraps the reasoning of "
                    "replies after the last user message; prompts never have one)");
    return out;
}

inline std::string render(const std::vector<Message>& m, const Template& t, bool generation_prompt, int thinking = -1) {
    std::string s;
    for (const auto& p : render_pieces(m, t, generation_prompt, thinking)) s += p.text;
    return s;
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
inline std::string jinja(const Template& t) {
    using detail::jstr;
    detail::paperkiln_only(t, "jinja()");
    if (t.content != "raw" || t.past_assistant != "keep" || t.has_default_system)
        throw Error("jinja() covers raw content, kept replies and no default system");
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
    std::string msgs = "{{- range $i, $m := .Messages }}{{- if eq $m.Role \"user\" }}" + go(t.user_open);
    if (t.system_mode == "fold") msgs += "{{- if and $.System (eq $i 0) }}{{ $.System }}" + go(t.system_join) + "{{ end }}";
    msgs += "{{ $m.Content }}" + go(t.user_close) + "{{- else if eq $m.Role \"assistant\" }}" + go(t.assistant_open) +
            "{{ $m.Content }}" + go(t.assistant_close) + "{{- end }}{{- end }}" + go(t.generation_prompt);
    std::string head = go(t.bos);
    if (t.system_mode == "turn")
        head += "{{- if .System }}" + go(t.system_open) + "{{ .System }}" + go(t.system_close) + "{{ end }}";
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
