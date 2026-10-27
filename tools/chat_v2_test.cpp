// chat_v2_test: ember's side of conversation schema v2 (paperkiln/2; LOCAL_LLM_STACK.md section 3) on a
// v2 vocabulary, without a model: the setup a session builds at load, the request paths of /chat,
// /v1/chat/completions and the WebAssembly build (chat_request.h), the exchange context rule, and replies
// split on the <|tool_call|> id, plain and streamed (chat_reply.h).
//
//   chat_v2_test TOY_DIR BASE.gguf [fuzz N]
//
// TOY_DIR holds experiments/chat_v2_2026-10-27/make_toy_v2.py's outputs (toy_v2.gguf, toy_v2_nocontrol.gguf,
// toy_v2_badtype.gguf); BASE.gguf is the v1 file they were made from. Prints one JSON report; exit 1 on any
// failed check.
#include <algorithm>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat_gguf.h"
#include "chat_reply.h"
#include "chat_request.h"
#include "gguf_parser.h"

using json = nlohmann::json;
namespace chat = microtorch::chat;

namespace {

json report = json::object();
size_t n_checks = 0, n_fail = 0;

void check(bool ok, const std::string& what) {
    ++n_checks;
    if (!ok) {
        ++n_fail;
        report["failures"].push_back(what);
    }
}

struct Loaded {
    ember::ChatSetup setup;
    std::vector<std::string> tokens;
};

Loaded load(const std::string& path) {
    GGUFData gd = load_gguf_meta(path, true);
    ember::ChatVocab v = ember::chat_vocab_from_gguf(gd, path);
    Loaded l;
    l.tokens = v.g.tokens;
    l.setup = ember::make_chat_setup(std::move(v));
    return l;
}

// GPT-2's bytes_to_unicode: the vocabulary's text for one raw byte.
std::string byte_token(unsigned char b) {
    std::vector<int> bs;
    for (int c = '!'; c <= '~'; ++c) bs.push_back(c);
    for (int c = 0xA1; c <= 0xAC; ++c) bs.push_back(c);
    for (int c = 0xAE; c <= 0xFF; ++c) bs.push_back(c);
    int cp = -1, n = 0;
    for (int x = 0; x < 256; ++x) {
        bool in = false;
        for (int y : bs) in = in || y == x;
        if (x == b) cp = in ? x : 256 + n;
        if (!in) ++n;
    }
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

std::string refused(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

// A plain reply and the same ids streamed: pieces, then the reply finish() gives.
struct Streamed {
    std::vector<std::string> pieces;
    size_t sent_after_call = 0;  // bytes sent by next() at or after the first call id
    std::string joined;
    ember::ChatReply reply;
};

Streamed stream(const std::vector<int>& ids, int call_id, const ember::ChatDecode& decode) {
    Streamed s;
    ember::ReplyBuilder b(call_id, decode);
    bool seen = false;
    for (int id : ids) {
        seen = seen || (call_id >= 0 && id == call_id);
        const std::string p = b.next(id);
        if (seen) s.sent_after_call += p.size();
        s.pieces.push_back(p);
        s.joined += p;
    }
    const std::string end = b.finish(s.reply);
    s.pieces.push_back(end);
    s.joined += end;
    return s;
}

bool same_calls(const std::vector<chat::ToolCall>& a, const std::vector<chat::ToolCall>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].id != b[i].id || a[i].name != b[i].name || a[i].arguments != b[i].arguments) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: chat_v2_test TOY_DIR BASE.gguf [fuzz N]\n");
        return 2;
    }
    const std::string dir = argv[1], base = argv[2];
    const int fuzz_n = argc > 4 && std::string(argv[3]) == "fuzz" ? std::stoi(argv[4]) : 20000;

    // 1. Load: the v2 file's setup, the refusals, and the v1 base unchanged.
    Loaded toy = load(dir + "/toy_v2.gguf");
    check(toy.setup.tok != nullptr && toy.setup.tmpl.id() == "paperkiln/2", "toy_v2 resolves to paperkiln/2 in chat mode");
    if (!toy.setup.tok) {
        report["why"] = toy.setup.why;
        std::cout << report.dump(1) << "\n";
        return 1;
    }
    const ember::ChatTokenizer& tok = *toy.setup.tok;
    const ember::ChatFormat& f = tok.format();
    const ember::ChatDecode decode = [&](const std::vector<int>& v) { return tok.decode(v); };
    auto enc = [&](const std::string& s) { return tok.encode(s); };
    const int call = f.tool_call_id;
    json load_r = {{"template", f.name}, {"why", f.why}, {"kind", tok.kind()}, {"vocab", tok.size()},
                   {"eos_ids", f.eos_ids}, {"turn_ids", f.turn_ids}, {"tool_call_id", call}};
    check(call == 16385, "tool_call_id is 16385");
    check(f.eos_ids == std::vector<int>{1} && f.turn_ids == std::vector<int>({2, 3, 16384, 16386}),
          "stops: eos <|endoftext|>; turn <|user|>, <|assistant|>, <|system|>, <|tool|>");
    check(!f.is_stop(call), "<|tool_call|> is not a stop");
    for (size_t k = 0; k < chat::kV2Control.size(); ++k) {
        const int id = 16384 + static_cast<int>(k);
        check(enc(chat::kV2Control[k]) == std::vector<int>{id}, std::string("encode ") + chat::kV2Control[k]);
        check(tok.decode({id}) == chat::kV2Control[k], std::string("decode ") + chat::kV2Control[k]);
    }
    for (const char* bad : {"toy_v2_nocontrol.gguf", "toy_v2_badtype.gguf"}) {
        const std::string e = refused([&] { load(dir + "/" + bad); });
        check(has(e, "control token") && has(e, "refused"), std::string(bad) + " refused at load");
        load_r["refusals"][bad] = e;
    }
    {
        Loaded v1 = load(base);
        check(v1.setup.tok && v1.setup.tmpl.id() == "paperkiln/1" && v1.setup.tok->format().tool_call_id == -1,
              "the v1 base: paperkiln/1, no tool_call_id");
        load_r["base"] = {{"template", v1.setup.tmpl.id()}, {"why", v1.setup.why}};
    }
    report["load"] = load_r;

    // 2. Requests (chat_request.h) under paperkiln/2 and paperkiln/1.
    const chat::Template v2 = chat::v2(), v1 = chat::v1();
    auto chat_prompt = [&](const std::string& body, const chat::Template& t) {
        const json r = json::parse(body);
        ember::ChatFormat g = f;
        g.tmpl = t;
        const ember::Conversation c = ember::chat_conversation(r, &t, t.has_tool ? ember::ordered_tools(body) : chat::json());
        return ember::build_chat_prompt(g, c, 1, 1 << 30, [&](const std::string& s) { return static_cast<int>(enc(s).size()); }).prompt;
    };
    auto oa_prompt = [&](const std::string& body, const chat::Template& t) {
        const json r = json::parse(body);
        ember::ChatFormat g = f;
        g.tmpl = t;
        const ember::Conversation c = ember::openai_conversation(r, &t, t.has_tool ? ember::ordered_tools(body) : chat::json());
        return ember::build_chat_prompt(g, c, 1, 1 << 30, [&](const std::string& s) { return static_cast<int>(enc(s).size()); }).prompt;
    };
    json req_r = json::array();
    auto expect_text = [&](const std::string& name, const std::function<std::string()>& f2, const std::string& want) {
        std::string got, err;
        try {
            got = f2();
        } catch (const std::exception& e) {
            err = e.what();
        }
        check(err.empty() && got == want, "request: " + name);
        req_r.push_back({{"name", name}, {"ok", err.empty() && got == want}, {"got", err.empty() ? json(got) : json(nullptr)},
                         {"error", err.empty() ? json(nullptr) : json(err)}});
    };
    auto expect_refused = [&](const std::string& name, const std::function<std::string()>& f2, const std::string& part) {
        std::string err;
        try {
            f2();
        } catch (const std::exception& e) {
            err = e.what();
        }
        check(has(err, part), "request refused: " + name);
        req_r.push_back({{"name", name}, {"ok", has(err, part)}, {"error", err}, {"rule", part}});
    };
    const std::string tools_body =
        R"("tools":[{"type":"function","function":{"name":"get_weather","description":"Weather now.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}])";
    const std::string tools_text =
        R"({"type": "function", "function": {"name": "get_weather", "description": "Weather now.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})";
    const std::string call_hist =
        R"([{"role":"user","content":"Weather in Paris?"},{"role":"assistant","content":"","tool_calls":[{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]},{"role":"tool","tool_call_id":"call_0","content":"{\"temp_c\": 18}"}])";
    const std::string after_call = "<|user|>Weather in Paris?<|assistant|><|tool_call|>get_weather\n{\"city\": \"Paris\"}"
                                   "<|endoftext|><|tool|>{\"temp_c\": 18}<|assistant|>";
    expect_text("/chat tools in the request's member order", [&] {
        return chat_prompt(R"({"user_input":"Weather?",)" + tools_body + "}", v2);
    }, "<|system|>" + tools_text + "<|user|>Weather?<|assistant|>");
    expect_text("/chat system and tools", [&] {
        return chat_prompt(R"({"system":"Be brief.","user_input":"Weather?",)" + tools_body + "}", v2);
    }, "<|system|>Be brief.\n\n" + tools_text + "<|user|>Weather?<|assistant|>");
    expect_text("/chat user_input left out after a tool message", [&] {
        return chat_prompt(R"({"history":)" + call_hist + "}", v2);
    }, after_call);
    expect_text("/chat a pair-form entry, then a call", [&] {
        return chat_prompt(R"({"history":[{"user":"hi","assistant":"hello"},)" + call_hist.substr(1) + "}", v2);
    }, "<|user|>hi<|assistant|>hello<|endoftext|>" + after_call);
    expect_text("/chat v2 without system or tools renders as v1", [&] {
        return chat_prompt(R"({"user_input":"b","history":[{"role":"user","content":"a"},{"role":"assistant","content":"x"}]})", v2);
    }, chat_prompt(R"({"user_input":"b","history":[{"role":"user","content":"a"},{"role":"assistant","content":"x"}]})", v1));
    expect_text("/chat empty system and tools: [] under v1 are none", [&] {
        return chat_prompt(R"({"system":"","user_input":"a","tools":[]})", v1);
    }, "<|user|>a<|assistant|>");
    expect_refused("/chat user_input missing, history ends on an assistant", [&] {
        return chat_prompt(R"({"history":[{"role":"user","content":"a"},{"role":"assistant","content":"x"}]})", v2);
    }, "Missing 'user_input'");
    expect_refused("/chat a user message after a tool result (the reply is owed)", [&] {
        return chat_prompt(R"({"user_input":"and?","history":)" + call_hist + "}", v2);
    }, "expected assistant");
    expect_refused("/chat tools not a list", [&] { return chat_prompt(R"({"user_input":"a","tools":{"x":1}})", v2); },
                   "tools must be a list");
    expect_refused("/chat a tool definition with a float", [&] {
        return chat_prompt(R"({"user_input":"a","tools":[{"x":1.5}]})", v2);
    }, "non-integer number");
    expect_refused("/chat a call name with a newline", [&] {
        return chat_prompt(R"({"history":[{"role":"user","content":"a"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"a\nb","arguments":"{}"}}]},{"role":"tool","content":"r"}]})", v2);
    }, "without a newline");
    expect_refused("/chat v1: tools", [&] { return chat_prompt(R"({"user_input":"a","tools":[{"x":1}]})", v1); },
                   "tools are not supported");
    expect_refused("/chat v1: tool_calls in the history", [&] {
        return chat_prompt(R"({"user_input":"a","history":[{"role":"user","content":"q"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":"{}"}}]}]})", v1);
    }, "tool_calls are not supported");
    expect_refused("/chat v1: a tool message", [&] {
        return chat_prompt(R"({"user_input":"a","history":[{"role":"tool","content":"q"}]})", v1);
    }, "history role 'tool' is not allowed");
    const std::string oa_msgs =
        R"([{"role":"system","content":"Be brief."},{"role":"developer","content":"Use tools."},{"role":"user","content":"Weather in Paris?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]},{"role":"tool","tool_call_id":"call_0","content":[{"type":"text","text":"{\"temp_c\": 18}"}]}])";
    expect_text("OpenAI: system + developer joined, null content with tool_calls, a tool message's text parts", [&] {
        return oa_prompt(R"({"messages":)" + oa_msgs + "," + tools_body + "}", v2);
    }, "<|system|>Be brief.\n\nUse tools.\n\n" + tools_text + after_call);
    expect_text("OpenAI: tool_choice auto", [&] {
        return oa_prompt(R"({"tool_choice":"auto","messages":[{"role":"user","content":"a"}]})", v2);
    }, "<|user|>a<|assistant|>");
    expect_refused("OpenAI: the last message an assistant's", [&] {
        return oa_prompt(R"({"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]})", v2);
    }, "the last message must be from the user or a tool");
    expect_refused("OpenAI: tool_choice required", [&] {
        return oa_prompt(R"({"tool_choice":"required","messages":[{"role":"user","content":"a"}]})", v2);
    }, "tool_choice is not supported");
    expect_refused("OpenAI: a system message after the user's", [&] {
        return oa_prompt(R"({"messages":[{"role":"user","content":"a"},{"role":"system","content":"b"}]})", v2);
    }, "allowed only before the first user message");
    expect_refused("OpenAI: a tool result for a call that does not exist", [&] {
        return oa_prompt(R"({"messages":[{"role":"user","content":"a"},{"role":"tool","content":"b"}]})", v2);
    }, "expected assistant");
    expect_refused("OpenAI: a tool message before the first user message (never dropped as an exchange)", [&] {
        return oa_prompt(R"({"messages":[{"role":"tool","content":"r"},{"role":"user","content":"a"}]})", v2);
    }, "message 0 is tool, expected user");
    expect_refused("/chat: a tool message before the first user message", [&] {
        return chat_prompt(R"({"user_input":"a","history":[{"role":"tool","content":"r"}]})", v2);
    }, "message 0 is tool, expected user");
    expect_refused("OpenAI v1: a tool message", [&] {
        return oa_prompt(R"({"messages":[{"role":"user","content":"a"},{"role":"tool","content":"b"}]})", v1);
    }, "tool messages are not supported");
    expect_refused("OpenAI v1: tools", [&] {
        return oa_prompt(R"({"messages":[{"role":"user","content":"a"}],"tools":[{"x":1}]})", v1);
    }, "tools are not supported");
    expect_refused("OpenAI v1: assistant tool_calls", [&] {
        return oa_prompt(R"({"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":"{}"}}]},{"role":"user","content":"b"}]})", v1);
    }, "tool_calls are not supported");
    {  // the control strings become single control ids in the prompt
        const std::vector<int> ids = enc(oa_prompt(R"({"messages":)" + oa_msgs + "," + tools_body + "}", v2));
        auto count_id = [&](int x) { return std::count(ids.begin(), ids.end(), x); };
        check(count_id(16384) == 1 && count_id(16385) == 1 && count_id(16386) == 1 && count_id(2) == 1 && count_id(3) == 2 &&
                  count_id(1) == 1,
              "the prompt's control strings are single ids");
    }
    report["requests"] = req_r;

    // 3. The context rule over exchanges (build_general_prompt).
    {
        const std::string body = R"({"system":"You are terse.",)" + tools_body +
                                 R"(,"history":[{"role":"user","content":"First question, about the weather in Paris today?"},{"role":"assistant","content":"Checking.","tool_calls":[{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]},{"role":"tool","tool_call_id":"call_0","content":"{\"temp_c\": 18, \"sky\": \"clear\"}"},{"role":"assistant","content":"18 C and clear."},{"role":"user","content":"Second question: and in Rome?"},{"role":"assistant","content":"Warmer, I expect."},{"role":"user","content":"Third: check Oslo."},{"role":"assistant","content":"","tool_calls":[{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Oslo\"}"}}]},{"role":"tool","tool_call_id":"call_0","content":"{\"temp_c\": 4}"}]})";
        const json r = json::parse(body);
        const ember::Conversation c = ember::chat_conversation(r, &v2, ember::ordered_tools(body));
        auto count = [&](const std::string& s) { return static_cast<int>(enc(s).size()); };
        const auto ex = chat::exchanges(c.messages);
        check(ex.size() == 3, "three exchanges");
        std::vector<std::string> from;  // the prompt kept from exchange k
        for (size_t k = 0; k < ex.size(); ++k) {
            std::vector<chat::Message> m = {c.messages[0]};
            m.insert(m.end(), c.messages.begin() + static_cast<long>(ex[k].first), c.messages.end());
            from.push_back(chat::render(m, v2, true, -1, c.tools));
        }
        const int max_new = 8;
        json ctx_r = json::array();
        auto run = [&](int ctx, int want_k, bool want_error) {
            const ember::ChatPrompt p = ember::build_chat_prompt(f, c, max_new, ctx, count);
            const bool ok = want_error ? !p.error.empty() && has(p.error, std::to_string(p.prompt_tokens)) &&
                                             has(p.error, "the system turn")
                                       : p.error.empty() && p.prompt == from[want_k] &&
                                             p.turns_used == static_cast<int>(ex.size()) - 1 - want_k;
            check(ok, "context rule at ctx " + std::to_string(ctx));
            ctx_r.push_back({{"ctx", ctx}, {"prompt_tokens", p.prompt_tokens}, {"history_turns_used", p.turns_used},
                             {"error", p.error}, {"ok", ok}});
        };
        const int n0 = count(from[0]), n1 = count(from[1]), n2 = count(from[2]);
        run(n0 + max_new, 0, false);      // everything fits
        run(n0 + max_new - 1, 1, false);  // one token short: the oldest exchange (with its call) goes
        run(n1 + max_new, 1, false);
        run(n1 + max_new - 1, 2, false);  // only the newest exchange (ending on its tool result)
        run(n2 + 1, 2, false);            // the newest alone, even when max_new does not fit
        run(n2, 2, true);                 // the system turn + the newest exchange fill the context: refused
        // A user/assistant-only v2 conversation without a system message keeps v1's turns at every ctx.
        const std::string pairs =
            R"({"user_input":"last","history":[{"user":"one two three","assistant":"four five"},{"user":"six","assistant":"seven eight nine ten"},{"user":"eleven","assistant":"twelve"}]})";
        bool same = true;
        const json pr = json::parse(pairs);
        const ember::Conversation cv2 = ember::chat_conversation(pr, &v2), cv1 = ember::chat_conversation(pr, &v1);
        ember::ChatFormat g1 = f;
        g1.tmpl = v1;
        for (int ctx = 4; ctx < 64; ++ctx) {
            const ember::ChatPrompt a = ember::build_chat_prompt(f, cv2, 2, ctx, count);
            const ember::ChatPrompt b = ember::build_chat_prompt(g1, cv1, 2, ctx, count);
            same = same && a.prompt == b.prompt && a.turns_used == b.turns_used && a.prompt_tokens == b.prompt_tokens &&
                   a.error.empty() == b.error.empty();
        }
        check(same, "user/assistant-only v2 keeps v1's turns and prompts at ctx 4..63");
        report["context"] = {{"counts", {n0, n1, n2}}, {"max_new", max_new}, {"runs", ctx_r}, {"v1_equal_ctx_4_63", same}};
    }

    // 4. Replies: plain (reply_of) and streamed (ReplyBuilder), on ids.
    {
        auto cat = [](std::vector<int> a, const std::vector<int>& b) {
            a.insert(a.end(), b.begin(), b.end());
            return a;
        };
        const int nl = enc("\n").at(0), lead = ember::token_id(toy.tokens, byte_token(0xC3)), cont = ember::token_id(toy.tokens, byte_token(0xA9));
        const int f0 = ember::token_id(toy.tokens, byte_token(0xF0)), f9 = ember::token_id(toy.tokens, byte_token(0x9F)),
                  f8 = ember::token_id(toy.tokens, byte_token(0x98)), f80 = ember::token_id(toy.tokens, byte_token(0x80));
        check(lead >= 0 && cont >= 0 && f0 >= 0 && f9 >= 0 && f8 >= 0 && f80 >= 0, "byte tokens found");
        const std::vector<int> args = enc("{\"city\": \"Paris\"}"), name = enc("get_weather");
        struct Case {
            std::string name;
            std::vector<int> ids;
            std::string content;
            std::vector<std::pair<std::string, std::string>> calls;
        };
        const std::vector<Case> cases = {
            {"no call", enc(" Hello there.\n"), "Hello there.", {}},
            {"content then a call", cat(cat(cat(enc("Let me check.  \n"), {call}), cat(name, {nl})), args),
             "Let me check.", {{"get_weather", "{\"city\": \"Paris\"}"}}},
            {"a call alone", cat(cat({call}, cat(name, {nl})), args), "", {{"get_weather", "{\"city\": \"Paris\"}"}}},
            {"two calls", cat(cat(cat({call}, cat(name, {nl})), args), cat(cat({call}, enc("get_time")), {nl})), "",
             {{"get_weather", "{\"city\": \"Paris\"}"}, {"get_time", ""}}},
            {"arguments keep their whitespace", cat(cat({call}, cat(name, {nl})), enc(" {} \n")), "",
             {{"get_weather", " {} \n"}}},
            {"a call without a newline: the whole reply is content", cat(enc("Sure "), cat({call}, name)),
             "Sure <|tool_call|>get_weather", {}},
            {"an empty name: content", cat(enc("a"), cat({call, nl}, args)), "a<|tool_call|>\n{\"city\": \"Paris\"}", {}},
            {"a spare token in the content", cat(enc("x "), cat({16387}, enc(" y"))), "x <|spare_3|> y", {}},
            {"a spare token in a call's arguments", cat(cat({call}, cat(name, {nl})), {16391}), "",
             {{"get_weather", "<|spare_7|>"}}},
            {"a character cut by the call", cat(enc("Caf"), cat({lead, call}, cat(name, {nl}))), "Caf\xEF\xBF\xBD",
             {{"get_weather", ""}}},
            {"a character split across tokens before the call", cat(enc("Caf"), cat({lead, cont, call}, cat(name, {nl}))),
             "Caf\xC3\xA9", {{"get_weather", ""}}},
            {"a four-byte character over four tokens", {f0, f9, f8, f80}, "\xF0\x9F\x98\x80", {}},
            {"arguments ending inside a character", cat(cat({call}, cat(name, {nl})), {f0, f9}), "",
             {{"get_weather", "\xEF\xBF\xBD"}}},
            {"no ids", {}, "", {}},
        };
        json rep_r = json::array();
        for (const Case& k : cases) {
            const ember::ChatReply p = ember::reply_of(k.ids, call, decode);
            const Streamed s = stream(k.ids, call, decode);
            bool calls_ok = p.tool_calls.size() == k.calls.size();
            for (size_t i = 0; calls_ok && i < k.calls.size(); ++i)
                calls_ok = p.tool_calls[i].id == "call_" + std::to_string(i) && p.tool_calls[i].name == k.calls[i].first &&
                           p.tool_calls[i].arguments == k.calls[i].second;
            bool valid = true;
            for (const auto& piece : s.pieces) valid = valid && ember::utf8_valid(piece) == piece;
            const bool ok = p.content == k.content && calls_ok && s.joined == p.content && s.reply.content == p.content &&
                            same_calls(s.reply.tool_calls, p.tool_calls) && s.sent_after_call == 0 && valid;
            check(ok, "reply: " + k.name);
            json calls = ember::tool_calls_json<json>(p.tool_calls);
            rep_r.push_back({{"name", k.name}, {"ids", k.ids}, {"content", p.content}, {"tool_calls", calls},
                             {"pieces", s.pieces}, {"ok", ok}});
        }
        report["replies"] = rep_r;

        // Fuzz: random replies over text, whitespace, byte, spare and call tokens; streamed equals plain, the
        // pieces are valid UTF-8, nothing goes out after the first call id before the end, and under v1
        // (call_id -1) the reply is v1's (the stripped decoded ids).
        std::vector<int> alphabet = {call, call, call, nl, nl, lead, cont, f0, f9, f8, f80, 16387, 16390};
        for (const char* t : {"get_weather", "{\"a\": 1}", " ", "  ", "\t", "Hello", " world", ".", "x"})
            for (int id : enc(t)) alphabet.push_back(id);
        std::mt19937 rng(20261027);
        size_t eq = 0, with_calls = 0, unparsed_with_call = 0, v1_eq = 0;
        for (int n = 0; n < fuzz_n; ++n) {
            std::vector<int> ids(rng() % 24);
            for (int& x : ids) x = alphabet[rng() % alphabet.size()];
            const ember::ChatReply p = ember::reply_of(ids, call, decode);
            const Streamed s = stream(ids, call, decode);
            bool valid = true;
            for (const auto& piece : s.pieces) valid = valid && ember::utf8_valid(piece) == piece;
            eq += s.joined == p.content && s.reply.content == p.content && same_calls(s.reply.tool_calls, p.tool_calls) &&
                  s.sent_after_call == 0 && valid;
            with_calls += !p.tool_calls.empty();
            unparsed_with_call += p.tool_calls.empty() && std::count(ids.begin(), ids.end(), call) > 0;
            const Streamed s1 = stream(ids, -1, decode);
            v1_eq += s1.joined == ember::utf8_valid(ember::strip_reply_ws(decode(ids))) && s1.reply.tool_calls.empty() &&
                     s1.reply.content == s1.joined;
        }
        check(eq == static_cast<size_t>(fuzz_n) && v1_eq == static_cast<size_t>(fuzz_n), "fuzz: streamed equals plain");
        report["fuzz"] = {{"replies", fuzz_n}, {"stream_equals_plain", eq}, {"with_calls", with_calls},
                          {"call_id_but_not_parsed", unparsed_with_call}, {"v1_stream_equals_v1_reply", v1_eq}};
    }

    report["checks"] = n_checks;
    report["failed"] = n_fail;
    std::cout << report.dump(1) << "\n";
    return n_fail == 0 ? 0 : 1;
}
