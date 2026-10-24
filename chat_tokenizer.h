#pragma once
// The tokenizer and chat template behind ember's chat mode (TinyLlamaSession::is_bpe()).
//
// The template is data (LOCAL_LLM_STACK.md section 3, point 3): a record from paperkiln's
// chat_template.h (vendored), resolved at load by chat::resolve, the same in every engine:
//   (1) the GGUF's paperkiln.chat_template, (2) paperkiln's chat specials at ids 0-3: paperkiln/1,
//   (3) a word-level vocabulary: paperkiln/legacy-word, (4) a third-party record (chatml-qwen2.5,
//   chatml-qwen3, gemma3) by the sha256 of tokenizer.chat_template, else by its markers and
//   general.architecture (llama.cpp b6000's Gemma 3 GGUF has no tokenizer.chat_template),
//   (5) no chat mode (raw completion only).
// Every prompt is chat::render of the conversation under that record; the stop ids are the
// record's stop strings, each of which must be a single token of the vocabulary. ember never
// hard-codes a format.
//
// The vocabulary forms ember tokenises for chat mode, all vendored or ported:
// - tokenizer.ggml.model "bpe": paperkiln's byte-level BPE with id-pair merges (bpe_tokenizer.h).
// - "gpt2" with pre "llama-bpe" (paperkiln's 16k Llama 3-scheme vocabulary) or "qwen2" (Qwen2.5,
//   Qwen3: single digits, added tokens split out by their GGUF token types): paperkiln's exact
//   llama.cpp port (llama3_tokenizer.h).
// - "llama" (SentencePiece; Gemma 3): llama.cpp b6000's SPM session (spm_tokenizer.h).
// A word-level vocabulary ("word") resolves to paperkiln/legacy-word but keeps ember's own
// word-level tokenizer and the server's word-level path (TinyLlamaSession::chat_template()).
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bpe_tokenizer.h"
#include "chat_template.h"
#include "llama3_tokenizer.h"
#include "spm_tokenizer.h"

namespace ember {

namespace chat = microtorch::chat;

// The id of `text` among the tokens (the last one, as llama.cpp's token_to_id), or -1.
inline int token_id(const std::vector<std::string>& tokens, const std::string& text) {
    for (size_t i = tokens.size(); i-- > 0;)
        if (tokens[i] == text) return static_cast<int>(i);
    return -1;
}

// A chat model's template and where its reply ends: generation stops at an id in eos_ids (stop
// reason "eos") or turn_ids ("turn": the model starting the next turn); the stop id is cut from
// the reply. A close's text after its stop token (ChatML's "<|im_end|>\n") is never generated: the
// next prompt renders it from the history.
struct ChatFormat {
    chat::Template tmpl;
    std::string name;  // tmpl.id(): "paperkiln/1", "chatml-qwen3/1", ...
    std::string why;   // the resolution's reason
    std::vector<int> eos_ids, turn_ids;

    // The record with its stop strings mapped to ids; chat::Error when one is not a single token.
    static ChatFormat of(const chat::Template& t, const std::vector<std::string>& tokens, std::string why) {
        ChatFormat f;
        f.tmpl = t;
        f.name = t.id();
        f.why = std::move(why);
        auto ids = [&](const std::vector<std::string>& stops, const char* field) {
            std::vector<int> v;
            for (const auto& s : stops) {
                const int id = token_id(tokens, s);
                if (id < 0)
                    throw chat::Error("chat template " + t.id() + ": stop." + field + " '" + s +
                                      "' is not a single token of this vocabulary (ember stops on single tokens only)");
                v.push_back(id);
            }
            return v;
        };
        f.eos_ids = ids(t.stop_eos, "eos");
        f.turn_ids = ids(t.stop_turn, "turn");
        if (f.eos_ids.empty()) throw chat::Error("chat template " + t.id() + " has no stop.eos token");
        return f;
    }

    bool is_eos(int id) const {
        for (int e : eos_ids)
            if (e == id) return true;
        return false;
    }
    bool is_stop(int id) const {
        if (is_eos(id)) return true;
        for (int t : turn_ids)
            if (t == id) return true;
        return false;
    }
    std::vector<int> stop_ids() const {
        std::vector<int> v = eos_ids;
        v.insert(v.end(), turn_ids.begin(), turn_ids.end());
        return v;
    }
    // The markers paperkiln/1's per-turn state tokens are defined over (state_tokens.h): the record
    // writes <|user|>U<|assistant|>A<|endoftext|> and nothing else around a message.
    bool v1_markers() const {
        const chat::Template v = chat::v1();
        return tmpl.name == v.name && tmpl.version == v.version && tmpl.bos.empty() && tmpl.content == "raw" &&
               tmpl.user_open == v.user_open && tmpl.user_close == v.user_close &&
               tmpl.assistant_open == v.assistant_open && tmpl.assistant_close == v.assistant_close &&
               tmpl.generation_prompt == v.generation_prompt && tmpl.system_mode != "turn" &&
               tmpl.stop_eos == v.stop_eos && tmpl.stop_turn == v.stop_turn;
    }

    // The prompt for `user` after turns[from..] ((user, assistant) pairs, oldest first), with the
    // system message (nullptr: none) and the thinking mode (-1 the template's default, 0 off, 1 on):
    // the record's rendering with a generation prompt. chat::Error when the template refuses it.
    std::string prompt(const std::vector<std::pair<std::string, std::string>>& turns, size_t from,
                       const std::string& user, const std::string* system = nullptr, int thinking = -1) const {
        const std::vector<std::pair<std::string, std::string>> kept(turns.begin() + static_cast<std::ptrdiff_t>(from),
                                                                    turns.end());
        return chat::render(chat::from_pairs(kept, &user, system), tmpl, /*generation_prompt=*/true, thinking);
    }
};

class ChatTokenizer {
public:
    virtual ~ChatTokenizer() = default;
    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& ids) const = 0;
    virtual size_t size() const = 0;
    virtual const char* kind() const = 0;
    const ChatFormat& format() const { return format_; }

protected:
    ChatFormat format_;
};

template <class T>
class ChatTokenizerOf : public ChatTokenizer {
public:
    ChatTokenizerOf(T t, const char* kind, ChatFormat format) : t_(std::move(t)), kind_(kind) {
        format_ = std::move(format);
    }
    std::vector<int> encode(const std::string& text) const override { return t_.encode(text); }
    std::string decode(const std::vector<int>& ids) const override { return t_.decode(ids); }
    size_t size() const override { return t_.size(); }
    const char* kind() const override { return kind_; }

private:
    T t_;
    const char* kind_;
};

// True when a GGUF's tokens start with paperkiln's four chat specials.
inline bool has_chat_specials(const std::vector<std::string>& tokens) {
    if (tokens.size() < chat::kV1Specials.size()) return false;
    for (size_t i = 0; i < chat::kV1Specials.size(); ++i)
        if (tokens[i] != chat::kV1Specials[i]) return false;
    return true;
}

// A GGUF's tokenizer block and the keys template resolution reads.
struct ChatVocab {
    microtorch::bpe::GgufTokenizer g;  // model, pre, tokens, merges, token_type
    std::vector<float> scores;         // tokenizer.ggml.scores (SentencePiece)
    bool add_space_prefix = true;      // tokenizer.ggml.add_space_prefix (SentencePiece; llama.cpp's default)
    std::string arch;                  // general.architecture
    std::string chat_template;         // tokenizer.chat_template ("" when absent)
    std::string declared;              // paperkiln.chat_template ("" when absent)
};

// What load found: the chat tokenizer (nullptr: no chat mode ember serves), the resolved record
// (also for a word-level vocabulary, which has no chat tokenizer here), and why.
struct ChatSetup {
    std::unique_ptr<ChatTokenizer> tok;
    bool resolved = false;
    chat::Template tmpl;
    std::string why;
};

inline ChatSetup make_chat_setup(ChatVocab v) {
    ChatSetup s;
    chat::Resolution r;
    try {
        r = chat::resolve(v.declared, v.g.tokens, v.g.model, v.chat_template, v.arch);
    } catch (const std::exception& e) {
        s.why = std::string("paperkiln.chat_template refused (") + e.what() + "): raw completion only";
        return s;
    }
    s.why = r.why;
    if (!r.chat) return s;
    s.resolved = true;
    s.tmpl = r.tmpl;
    auto& g = v.g;
    const bool bpe = g.model == "bpe";
    const bool l3 = g.model == "gpt2" && (g.pre == "llama-bpe" || g.pre == "qwen2");
    const bool spm = g.model == "llama";
    if (!bpe && !l3 && !spm) {
        if (g.model != "word") s.why += "; ember has no chat tokenizer for tokenizer.ggml.model '" + g.model + "'";
        return s;
    }
    ChatFormat f;
    try {
        f = ChatFormat::of(r.tmpl, g.tokens, r.why);
    } catch (const chat::Error& e) {
        s.resolved = false;
        s.why = r.why + "; " + e.what() + ": raw completion only";
        return s;
    }
    if (bpe) {
        if (g.merges.empty()) throw std::runtime_error("BPE GGUF has no tokenizer.ggml.merges");
        s.tok = std::make_unique<ChatTokenizerOf<microtorch::bpe::Tokenizer>>(
            microtorch::bpe::Tokenizer(std::move(g.tokens), g.merges), "bpe", std::move(f));
    } else if (l3) {
        if (g.merges.empty()) throw std::runtime_error(g.pre + " GGUF has no tokenizer.ggml.merges");
        const bool qwen = g.pre == "qwen2";
        s.tok = std::make_unique<ChatTokenizerOf<microtorch::llama3::Tokenizer>>(
            microtorch::llama3::Tokenizer(std::move(g.tokens), g.merges, g.token_type, qwen ? 1 : 3),
            qwen ? "qwen2_bpe" : "llama3_bpe", std::move(f));
    } else {
        s.tok = std::make_unique<ChatTokenizerOf<SpmTokenizer>>(
            SpmTokenizer(std::move(g.tokens), std::move(v.scores), g.token_type, v.add_space_prefix), "spm",
            std::move(f));
    }
    return s;
}

}  // namespace ember
