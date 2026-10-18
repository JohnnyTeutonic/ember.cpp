# Chat parity, 18 Oct 2026

Model: paperkiln `chat_sft_bpe.gguf` (sha256 686e09274662f954..., d384, 6 layers,
byte-level BPE 8192, ctx 256), served by `tinyllama_server` in BPE chat mode.
Reference: mtstudio chat (paperkiln) on the same weights (`chat_sft.safetensors`,
sha256 172cba33dcb4f3a8...) with spec `sft_all.json`, which states
`arch.rope_heads: "all"` as the training run recorded.

Check (`tools/chat_parity.py`): first N prompts of `tinychat_hf_prompts_v1.jsonl`,
greedy (`top_k=1`, max 60 new tokens), both servers; compares reply text, generated
token count, stop reason, and ember's prompt token count against `tools/bpe.py`.

| Build | N | Reply identical | Tokens | Stop reason | Prompt tokens == bpe.py |
|---|---|---|---|---|---|
| CPU | 40 | 40 | 40 | 40 | 40 |
| CUDA (TF32 default) | 40 | 40 | 40 | 40 | 40 |

Sampled replies (temperature > 0) are not compared: the two samplers consume the
RNG differently. Logits are not compared directly here.

## Batch prefill for short prompts

`generate` ran prompts under 32 tokens one token at a time. Chat prompts are
mostly 10-30 tokens. `tools/chat_ttft.py`: first 20 eval prompts, greedy,
TTFT from a 1-token request, then a 60-token reply. Two alternating runs per
setting (`EMBER_PREFILL_MIN`); `ttft_min*.json` are the CUDA runs (second of
each pair), `ttft_cpu_min*.json` the CPU runs.

| Build | Threshold | TTFT p50 (ms), two runs | Reply wall p50 (ms) |
|---|---|---|---|
| CUDA | 32 | 338, 294 | 568, 551 |
| CUDA | 1 | 53, 55 | 295, 290 |
| CPU | 32 | 1251, 983 | 2245, 1803 |
| CPU | 1 | 650, 592 | 1526, 1390 |

Greedy replies identical between thresholds, 20/20 on each build. Default is
now 1 (always batch). 40-prompt greedy parity against mtstudio rerun with the
new default: 40/40 on CPU and CUDA (`*_prefill1.jsonl`). The machine was shared
with a paperkiln eval on CPU during these runs.
