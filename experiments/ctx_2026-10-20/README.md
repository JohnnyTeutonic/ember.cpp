# Context-limit parity, 20 Oct 2026

The context rule (LOCAL_LLM_STACK.md section 3, contract point 5), shared by ember's server,
the WebAssembly `ember_chat` and mtstudio /chat (paperkiln de33d142): drop the oldest whole
history turns while prompt + max_new_tokens > ctx; reject a prompt that still fills the
context (HTTP 400, "message is N tokens; the model's context is C"); generation never runs
the model over more than ctx positions, so a reply that reaches the limit has
ctx - prompt + 1 tokens and stop_reason "length".

`tools/chat_ctx_parity.py` sends 13 greedy cases to ember and to mtstudio: four messages
that fill 50-95% of the context with max_new_tokens 1024, four short messages with
max_new_tokens 1024, histories of 2, 4, 8 and 12 turns, and one message longer than the
context. It compares reply, stop_reason, tokens and history_turns_used (or status and error
text). `ctx_parity.log`:

| Model (ctx) | CPU build | CUDA build |
|---|---|---|
| gpt2_tiny (128), IR interpreter | 13/13 | 13/13 |
| chat_sftfinal_v3 (256), llama path | 13/13 | 13/13 |

The gpt2 cases reach the limit both ways: prompt 119 + 10 reply tokens and prompt 20 + 109
reply tokens, each 129 = ctx + 1, stop "length"; a 12-turn history keeps 1 turn on gpt2 and
4 on the chat model, as in mtstudio.
