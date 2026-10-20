# Recurrent state in the IR interpreter (Kimi linear attention, Mamba ssm_scan), 20 Oct 2026

By a sub-agent against paperkiln's op specs (ir.hpp at `families` 176b425) and CPU reference
ops, reviewed, merged onto the CUDA work and re-gated (`merge_gate.log`).

- New state kind `recurrent` ({name, kind, op, dtype f32, tensors [{name, shape, init}]}), any
  number per block next to at most one kv slot (SRD's kv + recurrent fits).
- `linear_attention`: phi(x) = x > 0 ? x+1 : (exp(x)-1)+1; num += phi(k) v and den += phi(k)
  before out = (phi(q) num) / max(den, 1e-8); no scale, softmax or positions.
- `ssm_scan`: s_t[i] = (sum_j A[i][j] s_{t-1}[j], left to right) + b[i] u[t,i]; y = c s_t + d u.
  Weights are node attributes naming GGUF tensors (blk.{i}.ssm_{a,b,c,d}.weight), as paperkiln's
  exporter writes them.
- Prefill runs the decode update row by row, so it equals token-by-token decode bit for bit.
- Runtime state: a call at position 0 resets; prefix reuse restores a snapshot (taken at the
  end of every call, after a multi-row call's second-to-last row, and after every row of a
  speculative call; at most 64 / 256 MiB), api.cpp asking `resume_point()` for the nearest
  restorable position; speculative decoding resumes from the snapshot after the last accepted
  row. Recurrent IRs run on the CPU interpreter (no device kernels yet).

| Check | Result |
|---|---|
| op level vs paperkiln's ops::kimi_attention / ops::ssm_scan (`op_gate.log`): prefill, decode, split prefill + decode, a speculative rewind, a run after another sequence | max abs diff 0 |
| Kimi end to end: gpt2_tiny weights with a Kimi IR (`fixtures/`) vs ParityLM(KIMI) mtstudio | 40/40 (CPU build, CUDA build, EMBER_SPEC=4) |
| Mamba end to end: a GGUF built from paperkiln's MambaModel (random, perturbed weights) vs paperkiln greedy | 40/40 generated ids (CPU and CUDA builds) |
| reset, prefix reuse and speculation (`state_*.log`; Kimi, Mamba, a kv + recurrent hybrid) | identical to from-scratch replies in every case |
| regressions: swa_tiny, gpt2 and llama via the IR on the GPU | 40/40 each |

Trained Kimi, Mamba and SRD models from paperkiln are still to be gated.
