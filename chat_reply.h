#pragma once
// A chat reply from its generated ids (the stop id cut), plain and streamed, shared by the server's
// /chat and /v1/chat/completions (server.cpp), the WebAssembly ember_chat / ember_finish
// (deploy/wasm/ember_wasm.cpp) and tools/chat_v2_test.cpp.
//
// A record with tool turns (paperkiln/2; ChatFormat::tool_call_id >= 0) splits the reply on ids, never
// on decoded text (chat::parse_reply): when it parses with at least one call, the reply is the content
// before the first <|tool_call|> plus the calls (ids call_0, call_1, ... per reply; name and arguments
// verbatim); otherwise (no call, or a part that does not parse) it is the whole decoded reply, every
// token as its text (a control or spare token's text included), as under v1. The content is stripped
// of leading and trailing whitespace (the C isspace set) as generate() strips a v1 reply, and is valid
// UTF-8 (utf8_valid); names and arguments are made valid UTF-8 and are otherwise untouched. Under
// every other record the reply is v1's: the stripped decoded ids.
//
// Streamed (ReplyBuilder): content pieces go out as tokens are generated (utf8_text.h's ReplyStream:
// a cut character held, trailing whitespace held) until the first <|tool_call|> id; after it nothing
// is sent until the end, where the rest of the content goes out (the content before the call, or, when
// the reply does not parse, the whole reply), and then the calls, each whole. Joined, the pieces equal
// the plain reply's content byte for byte, and the calls equal its calls.
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "chat_tokenizer.h"
#include "utf8_text.h"

namespace ember {

using ChatDecode = std::function<std::string(const std::vector<int>&)>;

// generate()'s strip (api.cpp): leading and trailing " \t\n\v\f\r".
inline std::string strip_reply_ws(const std::string& s) {
    const char* ws = " \t\n\v\f\r";
    const auto a = s.find_first_not_of(ws), b = s.find_last_not_of(ws);
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

struct ChatReply {
    std::string content;                      // valid UTF-8, stripped
    std::vector<chat::ToolCall> tool_calls;   // empty: no call (always empty under a v1 record)
};

// The plain reply of ids. call_id: ChatFormat::tool_call_id (-1: no tool turns).
inline ChatReply reply_of(const std::vector<int>& ids, int call_id, const ChatDecode& decode) {
    ChatReply r;
    if (call_id >= 0) {
        chat::ParsedReply p = chat::parse_reply(ids, call_id, decode);
        if (p.parsed && !p.tool_calls.empty()) {
            r.content = utf8_valid(strip_reply_ws(p.content));
            for (auto& c : p.tool_calls) {
                c.name = utf8_valid(c.name);
                c.arguments = utf8_valid(c.arguments);
                r.tool_calls.push_back(std::move(c));
            }
            return r;
        }
    }
    r.content = utf8_valid(strip_reply_ws(decode(ids)));
    return r;
}

// tool_calls in OpenAI's form, as requests carry them back:
// [{"id": "call_0", "type": "function", "function": {"name", "arguments"}}, ...].
template <class Json>
Json tool_calls_json(const std::vector<chat::ToolCall>& calls) {
    Json a = Json::array();
    for (const auto& c : calls)
        a.push_back(Json{{"id", c.id}, {"type", "function"}, {"function", Json{{"name", c.name}, {"arguments", c.arguments}}}});
    return a;
}

// A streamed reply: next(id) for each generated id in order (stop ids excluded), returning the newly
// final content text (often empty); finish(out) at the end returns the rest of the content and sets
// out to the reply, equal to reply_of(the ids seen).
class ReplyBuilder {
public:
    ReplyBuilder(int call_id, ChatDecode decode) : call_id_(call_id), decode_(std::move(decode)) {}

    std::string next(int id) {
        ids_.push_back(id);
        if (call_id_ >= 0 && id == call_id_) in_call_ = true;
        if (in_call_) return std::string();
        return stream_.next(decode_(ids_), false);
    }

    std::string finish(ChatReply& out) {
        out = reply_of(ids_, call_id_, decode_);
        if (!in_call_) return stream_.next(decode_(ids_), true);
        if (!out.tool_calls.empty()) {  // parsed: the content before the first call
            std::vector<int> before(ids_.begin(), ids_.begin() + first_call());
            return stream_.next(decode_(before), true);
        }
        return stream_.next(decode_(ids_), true);  // not parsed: the whole reply is content
    }

    const std::vector<int>& ids() const { return ids_; }

private:
    std::ptrdiff_t first_call() const {
        for (size_t i = 0; i < ids_.size(); ++i)
            if (ids_[i] == call_id_) return static_cast<std::ptrdiff_t>(i);
        return static_cast<std::ptrdiff_t>(ids_.size());
    }

    int call_id_;
    ChatDecode decode_;
    std::vector<int> ids_;
    bool in_call_ = false;
    ReplyStream stream_;
};

}  // namespace ember
