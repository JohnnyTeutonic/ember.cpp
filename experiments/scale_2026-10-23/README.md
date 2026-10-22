# The `scale` op (23 Oct 2026)

`{"op": "scale", "in": ["x"], "weight": "<[d]>", "bias": "<[d]>" | null, "out": ...}`: y[t][j] =
(x[t][j] * g[j]) + b[j], each product and sum rounded on its own (EXACT), agreed with paperkiln for
a per-feature gain without a [d, d] diagonal linear. ember 67d9c0b (CPU, CUDA, device plan).

Gate, ember CPU (ember 67d9c0b), on paperkiln's Primer-norm gate model re-exported with scale
(paperkiln 3c1912ed; GGUF sha256 148642cd9c83c079..., 325,120 bytes smaller than the diag export,
same weights; per block expr, scale, ..., expr, scale, ..., expr, ...; head expr, scale, linear):
- logits on the scale GGUF equal the diag GGUF's exactly (max |d| 0.0, prefill and decode, three
  sequences up to 58 tokens): `scale_gate.py`;
- ember's server on the scale GGUF: replies identical to mtstudio's recorded references on 40/40
  prompts (text, token counts, stop reasons; the request of tools/chat_parity.py);
- smoke test (`scale_smoke.py`): gpt2_tiny with a scale node per block, ember vs a numpy evaluation
  of the same IR 1.9e-7 of the range, top-1 13/13, decode equal to prefill.
CUDA vs CPU: with the next Colab device-gate run.
