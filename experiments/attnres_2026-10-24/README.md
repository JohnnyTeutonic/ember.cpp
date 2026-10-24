# llama_attnres: the first gate (24 Oct 2026)

The model is paperkiln's ar3 (`/mnt/c/ml_artifacts/attnres_gate/ar3/`, paperkiln c48ca962, family_hint
`llama_attnres`; IR_SPEC_DRAFT.md section 2). It has 3 llama blocks at d 32 and uses the full
(sublayer) form of attention residuals. Each sublayer reads `depth_attend` over the bank list,
followed by a per-layer `affine`: attention mix x (2i + 1), FFN mix x (2i + 2), and in the head
depth_attend then x 7. Every sublayer appends to the bank. ember runs it from the IR embedded in
ar3.gguf on the CPU, and needed no engine change.

`attnres_gate.py` (receipt `attnres_gate_cpu.json`, ember 2cb4d2a build):

| check | result |
|---|---|
| sha256: ar3.gguf 0e2aea08, ar3.ir.json 7428c891, score_in 7c53d154, score_out 81e677bb | all match |
| F32: log-probs of the 40 continuations against `mtstudio score` | worst 2.0e-7 relative (gate 1e-5); paperkiln's numpy IR is 3.04e-7 |
| F32: greedy (ember's argmax at every scored position) | 40/40 |
| F32: decode against prefill | bit-equal |
| ar3_q8_0.gguf (7 pseudo-queries [d] all F32, the matrices Q8_0) | loads; log-probs within 1.7e-4, greedy 40/40, decode equal to prefill (reported, not gated) |
| blk.0.attnres_attn_q is not read (its depth_attend has one source and copies it) | a copy with that tensor set to NaN gives bit-equal logits |

Run from the workspace root in WSL, under the CPU budget:

    python3 scripts/cpu_budget.py run --session ember --job "ember llama_attnres gate ar3 (CPU)" --cores 4 \
      --resume "rerun attnres_gate.py" -- python3 ember.cpp/experiments/attnres_2026-10-24/attnres_gate.py

Not done here:
- **CUDA and WASM legs** (section 5, gate 5). `list`, `append`, `depth_attend` and `affine` all have
  device kernels, so the device gate should need no new code. It waits for a Colab device session,
  because the local GPU is not used.
- **Recursion under attention residuals** (section 3: the node key `"tensor_index": "application"`,
  pseudo-queries `rec_app.{i}.*`) is not implemented. It is outside the sections this work covered,
  and no model uses it yet.
