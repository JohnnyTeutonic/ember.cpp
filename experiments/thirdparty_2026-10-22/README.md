# Third-party GGUFs through the IR interpreter: Qwen2.5 and Qwen3 on the CPU, 22 Oct 2026

ember runs a standard llama.cpp GGUF whose `general.architecture` is `qwen2` or `qwen3`. The file
has no `paperkiln.ir`, so `ember_ir::build_ir_from_gguf` (`graph_ir.cpp`) builds the IR from its
metadata and tensor names and the interpreter runs it (LOCAL_LLM_STACK.md section 7, "Third-party
architectures via the IR"). The llama fast path is untouched.

New in the interpreter: RoPE pairing `neox` (the pair x[j], x[j + n_rot/2]) and `n_rot` (partial
rotary), the `rmsnorm` `group` attribute (per-head q/k norm, one weight [group] shared by every
head), grouped-query checks (query head h reads kv head h / (heads / kv_heads), as the CPU
attention kernel already did), and BF16 tensors. Kept-quantised weights (F16, Q8_0, legacy
blocks; models over 64M parameters on the CPU) are read in place from the GGUF's memory map
instead of copied, and F16 rows dequantise through a lookup table built from ember's own
fp16_to_fp32. The table gives bit-identical first-token logits (40/40 prompts against the build
without it). On Qwen2.5 the server's peak RSS falls from 2.9 to 1.1 GB, and a 30-token prefill
takes about 12 s instead of 57 s on one loaded core. The new RoPE and norm attributes have CPU
kernels only: `device_supported` keeps a CUDA build on the CPU interpreter for these files.

Tokeniser and template (`chat_tokenizer.h`): a `gpt2`/`qwen2` vocabulary goes through paperkiln's
exact llama.cpp port (`llama3_tokenizer.h`, vendored at paperkiln a8510d93). It runs with
`max_digits` 1, the qwen2 pre-tokeniser regex in llama.cpp b6000 (`LLAMA_VOCAB_PRE_TYPE_QWEN2`:
the llama3 regex with `\p{N}` for `\p{N}{1,3}`). Added tokens are split out by their GGUF token
types. `/chat`, `/v1/chat/completions` and the WASM build write ChatML:
`<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n` per turn, preceded by Qwen2.5's
default system turn when the GGUF's template contains it (Qwen3's has none). Generation stops at
`<|im_end|>` or `<|endoftext|>`.

## Method

- Reference: HF transformers 4.57 loads the same GGUF (`gguf_file=...`, float32), with HF's own
  tokenizer and `apply_chat_template` (`tools/thirdparty_hf_reference.py`). Greedy decoding is
  plain argmax: `repetition_penalty=1.0`, overriding the checkpoint's generation_config, which
  sets 1.1 for Qwen2.5-Instruct. Up to 32 new tokens, stopping at the generation_config's eos ids.
- ember: the CPU build (`-DHAS_CUDA=OFF`), `tinyllama_server MODEL PORT 127.0.0.1 - 0`, `/chat`
  with `top_k` 1 and `return_ids`. First-token logits come from `tools/ir_first_logits.cpp` on
  HF's prompt ids. `tools/thirdparty_gate.py` compares the two; `EMBER_IR_CTX=1024`.
- The built IR is compared with `python tools/gguf_ir.py MODEL.gguf` from paperkiln (the file at
  sha256 13401df7...), field by field except `_comment`.
- Tokeniser (`tools/thirdparty_tokenizer_check.py`, no model): ember's ids against HF's for 21
  hard cases, all 200 prompts of paperkiln's `corpus/eval/chat_prompts_v1.jsonl` raw, and all 200
  in the chat template. ember's template text is also checked against `apply_chat_template`.
- Prompts: the first 40 of `chat_prompts_v1.jsonl` (everyday greetings).

## Results

| Model (file, sha256) | built IR = gguf_ir.py | prompt ids | greedy | first-token max abs dlogit (of range) | top-1 | tokeniser | template text |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct f16 (`qwen2.5-0.5b-instruct-fp16.gguf`, 8e0ae260...) | yes | 40/40 | 40/40 | 5.6e-5 (1.4e-6) | 40/40 | 421/421 | 200/200 |
| Qwen3-0.6B Q8_0 (`Qwen3-0.6B-Q8_0.gguf`, 9465e63a...) | yes | 40/40 | 40/40 | 4.8e-5 (1.0e-6) | 40/40 | 421/421 | 200/200 |

Qwen2.5's 40 greetings all get the same 10-token reply ("Hello! How can I assist you today?"), so
on that set the logits are the stronger check: median 1.26e-6 of the logit range, worst 1.37e-6.
Qwen3 thinks before it answers. Its 40 replies are 32 tokens each of varied reasoning text
("<think>\nOkay, the user just said ...") and all 40 match HF token for token. The Q8_0 weights
stay in their blocks and are read in place, and HF dequantises the same blocks to f32. Server peak
RSS 1.07 GB.

The template check covers one user message and a two-turn history (`hf_ref.jsonl.template.json`).
Qwen2.5 writes its default system turn and Qwen3 none. Both match HF's text exactly.

Regression on paperkiln's own families (`regress.jsonl`): this CPU build (0f3dc58) against the
338b24a CPU build on 40 prompts, greedy, 60 new tokens. Prompt ids, reply ids and stop ids are
identical on 40/40 for each of chat_sftfinal_v3 (llama path, and the interpreter with
`EMBER_INTERPRETER=1`), moe_tiny_v3, gpt2_tiny, kimi_tiny, mamba_tiny, attnres_tiny,
flex_highway_tiny and swa_tiny. The `reply_same_as_mtstudio` column compares against whatever
reference file sits beside each model, and both builds score the same on it. chat_sftfinal_v3's
file holds another model's replies (1/40), and swa_tiny's gives 11/40 on both builds.

Files: `qwen2.5-0.5b/` and `qwen3-0.6b/` each hold `summary.json` and `gate.jsonl` (per prompt: ids, replies, logit
differences), `hf_ref.jsonl` (HF's templated text, prompt ids and greedy ids),
`hf_ref.jsonl.template.json` (template text checks, one turn and two), `ember_built.ir.json` and
`gguf_ir.json`, and `tokenizer_check.json`.
