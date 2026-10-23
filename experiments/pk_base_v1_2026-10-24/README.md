# The 16k Llama 3-scheme vocabulary: model gate, pk_base_v1 and the arXiv-chain chat models, 24 Oct 2026

paperkiln's next vocabulary is the Llama 3 scheme, 16,384 entries (byte-level BPE, llama3
pre-tokeniser, `tokenizer.ggml.model` "gpt2", pre "llama-bpe", the four chat specials at ids 0-3;
LOCAL_LLM_STACK.md section 10, 21 Oct). Every model here carries it. ember serves it through
`chat_tokenizer.h` (paperkiln's `llama3_tokenizer.hpp`, vendored); no engine change was needed.

Setup, all on the CPU in WSL:
- **ember:** `~/ember_wt/build_cpu/tinyllama_server` (CPU build, `-DHAS_CUDA=OFF`), sources equal to
  ember main cbdcb91 (checked file by file, line endings aside), `tinyllama_server MODEL 8097 127.0.0.1 - 0`.
  Llama-family files run on the tuned llama path; the Cerebras-GPT model (family_hint gpt2) runs on
  the IR interpreter from its embedded `paperkiln.ir`.
- **mtstudio:** CPU build of paperkiln master c8f0b8af (`git archive`, out of tree in
  `~/pk_cpu_1024`, vendored transformer_core), `mtstudio chat SPEC --port 8098`, full recompute (no
  `--kv-cache`). SPEC is the run's own spec with `out_dir` set to the run folder and `data.vocab` set
  to the run's f32 GGUF (`specs/`); mtstudio reads `<name>.safetensors` and `events.jsonl` there.
  Both servers' `/health` were checked for the right model before any request.
- **Harness:** `tools/chat_parity.py` (this commit): greedy (`top_k` 1, temperature 1, `top_p` 1),
  60 new tokens, the first 40 prompts of `tinychat_hf_prompts_v1.jsonl` (sha256 a906b25a...). It
  compares reply text, generated token count and stop reason, and ember's prompt token count against
  paperkiln's tokeniser: `tools/llama3_bpe.py` (HF tokenizers configured as llama.cpp encodes) for a
  "gpt2"/"llama-bpe" GGUF, `tools/bpe.py` for the older "bpe" vocabularies. New: `PARITY_PROMPTS`
  and `PARITY_MAX_NEW` override the prompt file and the reply cap.
- **Driver:** `gate.py` starts both servers, runs the harness, and stops both. Its `ref` job replays a
  recorded parity file through ember alone (used for Q8_0 against the f32 mtstudio replies).

## Models (C:/ml_artifacts/next_base/)

| file | shape | sha256 |
|---|---|---|
| `gate/gate_llama3.gguf` | llama d256 L4 H4, ctx 256, 7.6M params, 400 steps (val 6.19) | f365148a8f9ea22fe67106ad85fc3c94f014a2acf71c699b248acb4b9b6a921a |
| `gate/gate_llama3.safetensors` | (mtstudio's weights) | 60be38090eea55422ffcd1901cb9d3e37e22d8c7c6846c51770dee2170e5219c |
| `base/pk_base_v1.gguf` | llama d768 L12 H12 d_ff 2048, ctx 512, 97.5M params, 120k steps (best val 2.676) | 510d288cf10ead4d3902247e3d898b6fa39e1d31d7465d3a8b6820761d6c9178 |
| `base/pk_base_v1_q8_0_all.gguf` | the same, Q8_0 | 50b0a101f5bb1f971ce28637ae825848aff33afb55cf190ccfd79c3a56028433 |
| `base/pk_base_v1.safetensors` | (mtstudio's weights) | 1251f204342299c54a30d15601976bf5dc4c44611d42bbb988c2e2ed25a686fd |
| `sft_2302_13971/arxiv_2302_13971.gguf` | LLaMA paper: llama d384 L6, ctx 512, 17.8M params, SFT 1,500 steps on lever 1 (val 2.447) | 1b231df76c47d27defafcf763ac51ae48e740c5dc238070e4a82d695bfc89400 |
| `sft_2302_13971/arxiv_2302_13971.safetensors` | (mtstudio's weights) | 2cbc52f7a2184b82f19aaee553f0abaabdbf40fa6f8c5094b342ce79d603bef0 |
| `sft_2304_03208/arxiv_2304_03208.gguf` | Cerebras-GPT paper: flex (layernorm, GELU, learned positions) d384 L6, ctx 512, 23.4M params, SFT 1,500 steps (val 2.618) | 58f4649cef57755427f69d8319b0bccabaa7e8b2c6255082303011beaab3f143 |
| `sft_2304_03208/arxiv_2304_03208.safetensors` | (mtstudio's weights) | 4647a9eff24921a9d7c1d7a99044f659bb0e7ec425f3bde5f5d3d24a87c9f89d |

## Results (`summary.json`)

| model | prompts | reply identical | token count | stop reason | prompt tokens = llama3_bpe.py | distinct replies | wall |
|---|---|---|---|---|---|---|---|
| gate_llama3 (llama3-vocabulary model gate) | 40 frozen | 40/40 | 40/40 | 40/40 | 40/40 | 6 | 192 s |
| pk_base_v1 f32 | 40 frozen | 40/40 | 40/40 | 40/40 | 40/40 | 40 (4 reach 60 tokens) | 2,265 s |
| pk_base_v1 Q8_0 (ember) vs mtstudio f32 | 40 frozen | 36/40 | 36/40 | 39/40 | 40/40 | 40 | 442 s (ember only) |
| arxiv_2302_13971 (LLaMA) | 40 frozen | 40/40 | 40/40 | 40/40 | 40/40 | 39 | 326 s |
| arxiv_2302_13971 | 5 demo, 120 tokens | 5/5 | 5/5 | 5/5 | 5/5 | 5 | 281 s |
| arxiv_2304_03208 (Cerebras-GPT) | 40 frozen | 40/40 | 40/40 | 40/40 | 40/40 | 40 | 260 s |
| arxiv_2304_03208 | 5 demo, 120 tokens | 5/5 | 5/5 | 5/5 | 5/5 | 5 | 239 s |

The gate model is barely trained (val 6.19), so its 40 replies collapse onto 6 strings; the token
counts and the prompt-token check still cover the vocabulary end to end. pk_base_v1 gives 40
different replies, so its 40/40 is the stronger whole-model check.

**Q8_0.** The quantised file's replies equal ember's own f32 replies on the same 36 prompts. The
other four part from f32 mid-reply after a shared prefix (prompt 9 at character 31, 10 at 11, 12 at
139, 34 at 89), each a plausible continuation, as base models with small top-1 margins do (fw_moe:
25/40, `../moe_fw_2026-10-21/`). Reported, not gated: first-token logits were not dumped in this run.

## Demo (`demo_transcript.md`, `arxiv_*_demo.jsonl`)

Five prompts (`demo_prompts.jsonl`), greedy, 120 new tokens; every reply is ember's and mtstudio's
alike. Both models greet fluently ("I'm doing well, thank you for asking. How can I help you today?")
and open the birthday prompt with a sympathetic line ("It's natural to feel upset...", "It's natural
to feel fear...") before repeating themselves (LLaMA) or listing loose steps (Cerebras-GPT); they have
no facts: "What is the
capital of France?" gives "The capital of France is a fundamental concept in finance, particularly
in the United States..." (LLaMA) and "...a fundamental concept in the United States..."
(Cerebras-GPT), looping to the 120-token cap; the autumn-leaves prompt drifts off topic in both.

## Commands

From the workspace root, each under `scripts/cpu_budget.py run --session ember --cores 4`:

```
python3 gate.py gate_llama3 /mnt/c/ml_artifacts/next_base/gate gate_llama3 \
    /mnt/c/ml_artifacts/next_base/gate/gate_llama3.gguf 40,~/e24/out/gate_llama3_parity.jsonl
python3 gate.py pk_base_v1 /mnt/c/ml_artifacts/next_base/base pk_base_v1 \
    /mnt/c/ml_artifacts/next_base/base/pk_base_v1.gguf 40,~/e24/out/pk_base_v1_parity.jsonl
python3 gate.py pk_base_v1_q8 /mnt/c/ml_artifacts/next_base/base pk_base_v1 \
    /mnt/c/ml_artifacts/next_base/base/pk_base_v1_q8_0_all.gguf \
    ref,~/e24/out/pk_base_v1_q8_0_all_parity.jsonl,~/e24/out/pk_base_v1_parity.jsonl
python3 gate.py arxiv_2302_13971 /mnt/c/ml_artifacts/next_base/sft_2302_13971 arxiv_2302_13971 \
    /mnt/c/ml_artifacts/next_base/sft_2302_13971/arxiv_2302_13971.gguf \
    40,~/e24/out/arxiv_2302_13971_parity.jsonl 5,~/e24/out/arxiv_2302_13971_demo.jsonl,demo_prompts.jsonl,120
# and the same for 2304_03208
```

`gate.py` runs `PARITY_EMBER_PORT=8097 PARITY_MTS_PORT=8098 PARITY_GGUF=MODEL
PAPERKILN_TOOLS=~/pk_cpu_1024/src/tools python3 ember.cpp/tools/chat_parity.py N OUT` for each job.
