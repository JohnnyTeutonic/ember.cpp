#pragma once
// Per-turn state tokens: the template contract (paperkiln
// experiments/tiny_chat_long_2026-10/IR_SPEC_DRAFT.md sections 1.2 and 1.5;
// LOCAL_LLM_STACK.md section 3 point 3 extended). With k = the GGUF's
// paperkiln.state_tokens.k > 0, every exchange of the served template is
//   <|user|> U <|assistant|> A <|endoftext|> S_0 S_1 .. S_{k-1}
// with S_j = vocab + j. The ids never come from the tokeniser; the engine
// inserts them, as the training serialiser does:
//   - after every <|endoftext|> that closes an exchange (one with an
//     <|assistant|> after the latest <|user|>), the k state ids;
//   - an assistant reply followed by a <|user|> with no <|endoftext|> between
//     them (a reply that stopped early: max_new_tokens, a stop string, an
//     abort) gets <|endoftext|> and then the k state ids before that <|user|>.
// Raw ids >= vocab are accepted only as a complete, ordered state block
// directly after an <|endoftext|> that closes an exchange (kept as given, not
// doubled); anything else throws std::invalid_argument naming the position.
#include <stdexcept>
#include <string>
#include <vector>

namespace ember {

inline std::vector<int> insert_state_blocks(const std::vector<int>& ids, int vocab, int k, int eot, int user,
                                            int assistant) {
  if (k <= 0) {
    for (size_t i = 0; i < ids.size(); ++i)
      if (ids[i] >= vocab)
        throw std::invalid_argument("token id " + std::to_string(ids[i]) + " at position " + std::to_string(i) +
                                    " is past the vocabulary (" + std::to_string(vocab) + ") and the model has no state tokens");
    return ids;
  }
  std::vector<int> out;
  out.reserve(ids.size() + 4 * static_cast<size_t>(k));
  auto block = [&] {
    for (int j = 0; j < k; ++j) out.push_back(vocab + j);
  };
  bool in_reply = false;  // an <|assistant|> after the latest <|user|>, its reply not yet closed
  const size_t n = ids.size();
  for (size_t i = 0; i < n; ++i) {
    const int id = ids[i];
    if (id >= vocab)
      throw std::invalid_argument("state id " + std::to_string(id) + " at position " + std::to_string(i) +
                                  " is not part of a complete state block directly after an <|endoftext|> that ends "
                                  "an assistant reply");
    if (id == user) {
      if (in_reply) {  // the reply stopped early: close it as training does
        out.push_back(eot);
        block();
      }
      in_reply = false;
      out.push_back(id);
    } else if (id == assistant) {
      in_reply = true;
      out.push_back(id);
    } else if (id == eot && in_reply) {
      in_reply = false;
      out.push_back(id);
      if (i + 1 < n && ids[i + 1] >= vocab) {  // a state block given with the ids: it must be complete
        for (int j = 0; j < k; ++j)
          if (i + 1 + j >= n || ids[i + 1 + j] != vocab + j)
            throw std::invalid_argument("the state block after the <|endoftext|> at position " + std::to_string(i) +
                                        " is not the " + std::to_string(k) + " ids " + std::to_string(vocab) + " .. " +
                                        std::to_string(vocab + k - 1) + " in order");
        i += static_cast<size_t>(k);
      }
      block();
    } else {
      out.push_back(id);
    }
  }
  return out;
}

}  // namespace ember
