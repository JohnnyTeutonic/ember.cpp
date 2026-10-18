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
