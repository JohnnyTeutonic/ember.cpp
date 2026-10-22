# The IR interpreter on CUDA against the CPU, third-party families (22 Oct 2026)

`tools/ir_device_gate` runs one GGUF on the device and on the CPU interpreter in one process. Short
prompts: 40 seeded random prompts of 24 token ids, the logits at every position, the device's
token-by-token decode (the captured CUDA graph after the first step), and 32 greedy tokens on each.
Long prompts (`--ids`): chat prompts from paperkiln's `corpus/eval/chat_prompts_v1.jsonl`, UltraChat
indices 100-199, kept when longer than 512 tokens (`tools/chat_prompt_ids.py --min-tokens 513`)
through each model's own chat template. For each one: the last prompt row's logits from the device's
prefill and from its decode through the whole prompt, then 32 greedy tokens. "rel" is
max |dlogit| over the CPU's logit range.

The real-model rule (LOCAL_LLM_STACK section 5.3): greedy replies identical and top-1 agreeing at
every position; logit closeness judged against a float64 evaluation of the same IR, not the flat
1e-4 that the receipts' `pass` field applies.

| Run | Where | Greedy | Top-1 | Worst rel |
|---|---|---|---|---|
| synthetic mistral_swa, gptneox, gpt2, mamba, gemma3 (+ per-layer freq_scale, + weight_offset) | local RTX 4060 | 40/40 each | 960/960 each | 1.5e-6 |
| gpt2_tiny with its gelu as an `expr` | local | 40/40 | 960/960 | 3.3e-7 |
| Qwen2.5-0.5B-Instruct f16 | local | 40/40 | 960/960 | 5.5e-5 |
| Gemma-3-270m-it f32 | local | 40/40 | 960/960 | 4.8e-5 |
| Mamba-130m f32 | local | 40/40 | 960/960 | 1.0e-4 |
| Pythia-160m f32 | local | 40/40 | 959/960 | 2.0e-4 |
| Qwen3-0.6B Q8_0 | Colab L4 | 40/40 | 960/960 | 4.3e-6 |
| Gemma-3-270m-it, 5 prompts of 794-1,324 tokens (past the 512 window) | Colab L4 | 5/5 | 10/10 | 2.9e-6 |
| Qwen2.5-0.5B-Instruct, 6 prompts of 558-1,289 tokens | Colab L4 | 6/6 | 12/12 | 1.8e-6 |
| Qwen3-0.6B, 6 prompts of 537-1,268 tokens | Colab L4 | 6/6 | 12/12 | 1.8e-6 |
| Pythia-160m f32 (`colab_l4_pythia/`, session ember-gate-1023p) | Colab L4 | 40/40 | 960/960 | 3.9e-4 |

Also on the Colab VM, on the CPU (`colab_l4/gemma_cpu/`; the Colab run is session ember-gate-1022d): Gemma-3-270m-it on the same five
past-window prompts through `tools/thirdparty_ember_check.py --ids`: ember's built IR equal to
`gguf_ir.build`, top-1 agreeing with numpy on every prompt, ember vs HF 2.1e-6 of the range
(numpy f32 vs HF 6.8e-6).

Notes:
- Local runs were on the laptop's RTX 4060 before GPU work moved to Colab (the WSL service faults of
  22 Oct). The local Qwen3 run was cut by the first fault; the Colab run replaces it.
- Pythia: on the RTX 4060 one position in 960 flipped (the row with top-1 23/24 in
  `local/pythia_160m.json`, recorded before the gate saved flip positions and margins); on the L4,
  the same prompts (same seed) with the file converted byte-identically (sha256 97522db7..., at
  revision 50f5173d with transformers 4.57.0, as paperkiln converted the local copy) give 960/960 and
  no flip, so `pythia_flips_float64.json` has none to settle. The worst difference, 3.9e-4 of the
  range, sits on Pythia's ill-conditioned positions (LOCAL_LLM_STACK section 5.3: float32
  conditioning at depth).
- Checkpoints on the VM: the Qwen GGUFs from Qwen's repositories (Qwen3's sha256 equal to the local
  copy's; Qwen2.5's matching the recorded prefix 8e0ae260);
  Gemma-3-270m-it converted by b6000 from unsloth's ungated copy of the release (a different file
  from the local conversion of Google's repository: sha256 36b7792e... vs 0a174d42...). The
  interpreter comparison is the same graph and weights on both sides of each run either way.
