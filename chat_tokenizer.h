#pragma once
// The tokenizer and chat format behind ember's chat mode (TinyLlamaSession::is_bpe()): one interface
// over the vocabulary forms it runs, all vendored from paperkiln.
// - tokenizer.ggml.model "bpe": paperkiln's byte-level BPE with id-pair merges (bpe_tokenizer.h).
// - tokenizer.ggml.model "gpt2" with pre "llama-bpe" and ids 0-3 = <unk>, <|endoftext|>, <|user|>,
//   <|assistant|>: the llama.cpp-compatible Llama 3 scheme (llama3_tokenizer.h), encoded exactly as
//   llama.cpp does. A real Llama 3 GGUF has the same model/pre pair but other tokens at ids 0-3,
//   so it keeps ember's Llama 3 path.
//   Both paperkiln forms share the SFT template <|user|>U<|assistant|>A<|endoftext|> and stop at
//   ids 1-3 (ChatFormat::paperkiln).
// - tokenizer.ggml.model "gpt2" with pre "qwen2" (Qwen2.5, Qwen3): the same encoder with the qwen2
//   pre-tokeniser (single digits), Qwen's added tokens (<|im_start|>, <|im_end|>, <|endoftext|>,
//   ...) split out by their GGUF token types, and the ChatML template (ChatFormat::chatml) when the
//   GGUF's tokenizer.chat_template is ChatML-style or the architecture is qwen2/qwen3.
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bpe_tokenizer.h"
#include "llama3_tokenizer.h"

namespace ember {

// How a chat model's conversation is written and where its reply ends. A prompt is
//   prefix + (user_open U assistant_open A turn_end) per past turn + user_open U assistant_open
// and generation stops at an id in eos_ids (stop reason "eos") or turn_ids ("turn": the model
// starting the next turn); the stop id is cut from the reply.
struct ChatFormat {
    std::string name;
    std::string prefix, user_open, assistant_open, turn_end;
    std::vector<int> eos_ids, turn_ids;

    // paperkiln's SFT template: no BOS, no separators; <|endoftext|> (id 1) ends a reply, a
    // generated <|user|> (2) or <|assistant|> (3) starts the next turn.
    static ChatFormat paperkiln() {
        return {"paperkiln", "", "<|user|>", "<|assistant|>", "<|endoftext|>", {1}, {2, 3}};
    }
    // ChatML as HF's apply_chat_template writes it for Qwen2.5 / Qwen3 with one user message (or a
    // history of plain user/assistant turns) and add_generation_prompt: an optional default system
    // turn, then <|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n. Stops at <|im_end|> and
    // <|endoftext|> (HF's generation_config eos ids).
    static ChatFormat chatml(const std::string& default_system, int im_end, int endoftext) {
        ChatFormat f;
        f.name = "chatml";
        if (!default_system.empty()) f.prefix = "<|im_start|>system\n" + default_system + "<|im_end|>\n";
        f.user_open = "<|im_start|>user\n";
        f.assistant_open = "<|im_end|>\n<|im_start|>assistant\n";
        f.turn_end = "<|im_end|>\n";
        f.eos_ids = {im_end};
        if (endoftext >= 0 && endoftext != im_end) f.eos_ids.push_back(endoftext);
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
    // The prompt for `user` after turns[from..] ((user, assistant) pairs, oldest first).
    std::string prompt(const std::vector<std::pair<std::string, std::string>>& turns, size_t from,
                       const std::string& user) const {
        std::string p = prefix;
        for (size_t i = from; i < turns.size(); ++i)
            p += user_open + turns[i].first + assistant_open + turns[i].second + turn_end;
        return p + user_open + user + assistant_open;
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
    ChatFormat format_ = ChatFormat::paperkiln();
};

template <class T>
class ChatTokenizerOf : public ChatTokenizer {
public:
    ChatTokenizerOf(T t, const char* kind, ChatFormat format = ChatFormat::paperkiln()) : t_(std::move(t)), kind_(kind) {
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
    const auto& s = microtorch::bpe::specials();
    if (tokens.size() < s.size()) return false;
    for (size_t i = 0; i < s.size(); ++i)
        if (tokens[i] != s[i]) return false;
    return true;
}

// The id of `text` among the tokens (the last one, as llama.cpp's token_to_id), or -1.
inline int token_id(const std::vector<std::string>& tokens, const std::string& text) {
    for (size_t i = tokens.size(); i-- > 0;)
        if (tokens[i] == text) return static_cast<int>(i);
    return -1;
}

// Qwen2.5's default system message, which its chat template writes when the conversation has none.
inline const char* kQwenDefaultSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

// The chat tokenizer for a GGUF's tokenizer block, or nullptr when ember's own tokenizers handle it.
// arch: general.architecture; chat_template: tokenizer.chat_template ("" when absent).
inline std::unique_ptr<ChatTokenizer> make_chat_tokenizer(microtorch::bpe::GgufTokenizer g, const std::string& arch = "",
                                                          const std::string& chat_template = "") {
    if (g.model == "bpe") {
        if (g.merges.empty()) throw std::runtime_error("BPE GGUF has no tokenizer.ggml.merges");
        return std::make_unique<ChatTokenizerOf<microtorch::bpe::Tokenizer>>(
            microtorch::bpe::Tokenizer(std::move(g.tokens), g.merges), "bpe");
    }
    if (g.model == "gpt2" && g.pre == "llama-bpe" && has_chat_specials(g.tokens)) {
        if (g.merges.empty()) throw std::runtime_error("llama-bpe GGUF has no tokenizer.ggml.merges");
        return std::make_unique<ChatTokenizerOf<microtorch::llama3::Tokenizer>>(
            microtorch::llama3::Tokenizer(std::move(g.tokens), g.merges, g.token_type), "llama3_bpe");
    }
    const bool chatml = chat_template.find("<|im_start|>") != std::string::npos || arch == "qwen2" || arch == "qwen3";
    if (g.model == "gpt2" && g.pre == "qwen2" && chatml) {
        if (g.merges.empty()) throw std::runtime_error("qwen2 GGUF has no tokenizer.ggml.merges");
        const int im_end = token_id(g.tokens, "<|im_end|>"), eot = token_id(g.tokens, "<|endoftext|>");
        if (token_id(g.tokens, "<|im_start|>") < 0 || im_end < 0)
            throw std::runtime_error("ChatML GGUF has no <|im_start|> / <|im_end|> tokens");
        // The default system turn: Qwen2.5's template writes one when there is none; Qwen3's does not.
        const std::string system =
            chat_template.find(kQwenDefaultSystem) != std::string::npos ? kQwenDefaultSystem : "";
        return std::make_unique<ChatTokenizerOf<microtorch::llama3::Tokenizer>>(
            microtorch::llama3::Tokenizer(std::move(g.tokens), g.merges, g.token_type, 1), "qwen2_bpe",
            ChatFormat::chatml(system, im_end, eot));
    }
    return nullptr;
}

}  // namespace ember
