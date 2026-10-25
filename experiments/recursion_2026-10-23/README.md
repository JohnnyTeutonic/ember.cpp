# Tied-block recursion: the block `weight_index` (23 Oct 2026)

`block.weight_index = {"per_layer": [m_0, ..., m_{L-1}]}`, L = `block.repeat` (paperkiln ir.hpp, paperkiln
567a441d): application i of the block reads every tensor name with `{i}` replaced by m_i, so several
applications share one stored layer. Everything else per application stays indexed by i: its own kv
cache and recurrent state, its per-layer attributes, its device-plan entry. Absent: the identity. Load
refuses a map whose length is not `block.repeat`, a negative entry, a stored layer 0 .. max m_i that is
never used, or a stored count other than the file's `<arch>.block_count`. Per-layer `affine` scale and
shift are accepted too (paperkiln's optional embedding re-injection, inject "add"). Files with the map
carry family_hint `llama_recursive`, so they run only through the interpreter (the llama path is
skipped). Weights are bound and uploaded by name, so a shared layer is stored once on either device.

Gate, ember CPU, on paperkiln's recursion gate models (paperkiln b1b620dd, `tests/test_recursion_ir.py`;
`/mnt/c/ml_artifacts/recursion_gate/`, F32, the IR embedded): `recursion_gate.py`, receipt
`recursion_gate_cpu.json`.

| model | applications / stored | weight_index | log-prob worst (rel.) | greedy | decode = prefill |
|---|---|---|---:|---|---|
| p1_b2x3_c1 (1 + [2] x 3 + 1) | 8 / 4 | 0 1 2 1 2 1 2 3 | 2.0e-7 | 40/40 | yes |
| p1_b2x3_c1_inject (the same, inject "add") | 8 / 4 | 0 1 2 1 2 1 2 3 | 1.8e-7 | 40/40 | yes |
| b2x4 ([2] x 4) | 8 / 2 | 0 1 0 1 0 1 0 1 | 2.7e-7 | 40/40 | yes |

Log-probabilities are against `mtstudio score`'s (the gate's tolerance is 1e-5 of max(1, max |ref|));
greedy is ember's argmax at every scored position against the continuation (numpy's greedy, equal
to mtstudio's argmax). The files hold tensors only for their stored layers, so a load that ignored the
map would fail.

Smoke test (`recursion_smoke.py`): gpt2_tiny's 2 stored layers applied 4 times ([0, 1, 0, 1] and
[0, 0, 1, 1]) through an external IR, ember vs paperkiln's ir_numpy of the same IR within 3.0e-7 of
the range, top-1 at every position, decode equal to prefill, logits moved by up to 3.5 against the
plain model (the map is read). The four malformed maps above are each refused at load.

CUDA: the three models on the device against the CPU (`tools/ir_device_gate`, Colab T4), 40/40 greedy and 320/320
top-1 each, worst 4.9e-7, and device outputs bit-identical across ember 51eed18 and 66c8b2c
(`../mor_cuda_2026-10-25`).

## Mixture-of-Recursions: `depth_route` and kv_rows "active" (causal token choice)

paperkiln's spec (ir.hpp at paperkiln c4427679; tiny-chat amendment 4, 63d9ffe9, registered before the
engine code), family_hint `llama_mor`. ember (CPU):
- **Routing.** At the first application of step 0, each token's router logits are x W^T, W =
  `rec.0.router.weight` [n_rec, d], computed as linear computes it. The depth is 1 + argmax, ties to the
  lower index. The gate is the max-shifted softmax's p[depth - 1], with z summed in ascending order
  (`cpu::mor_route`).
- **Running a step.** Step k keeps its input x_in and runs only on the tokens deeper than k: one at a
  time, in position order, at their absolute positions. Each step application keeps a compact kv
  cache of just the rows written there, with their positions, so attention's causal kernel and RoPE
  run unchanged.
- **Gate.** After the step's last application, gate "score" sets x = x_in + g (x - x_in): the
  difference rounded, then the product, then the sum, compiled without contraction (`cpu::mor_gate`).
  Inactive tokens keep x_in. A step with no active token runs nothing.
- **Resuming.** A forward at start_pos (a new sequence, prefix reuse, a rewind) keeps each compact
  cache's rows below start_pos.
- **Load checks.** Load refuses a depth_route that isn't the block's first node, any route except
  token_choice, steps that aren't consecutive, ascending and of equal length, a kv state without
  kv_rows "active", kv_from (reserved), recurrent state or windowed attention alongside it, and a
  router name with {i}.
- **CUDA.** From ember 66c8b2c a CUDA build runs these files on the device, ROUNDING against this CPU path, which stays
  the EXACT reference: on a T4, mor3 and mor2p0 give greedy 40/40, top-1 1299/1299, logits within 2.0e-7 and
  log-probabilities within 2.6e-7 of the CPU, depth traces 40/40 in prefill and decode, rows held equal to the CPU's
  per application, and rewinds 160/160 (`../mor_cuda_2026-10-25`).

Gate (`mor_gate.py`, receipt `mor_gate_cpu.json`) on paperkiln's two MoR models
(`/mnt/c/ml_artifacts/mor_gate/`):

| model | recursion | log-prob worst (rel.) | greedy | depth, prefill | depth, decode | decode vs prefill | smallest router margin |
|---|---|---:|---|---|---|---:|---:|
| mor3 | 1 + [2] x 3 + 1, gate score | 2.3e-7 | 40/40 | 40/40 | 40/40 | 0.0 | 2.9e-5 |
| mor2p0 | 0 + [3] x 2 + 1, gate score | 1.9e-7 | 40/40 | 40/40 | 40/40 | 0.0 | 6.4e-3 |

The depths are mtstudio's per-token trace, token for token. Nine malformed variants of mor3's IR are
each refused at load (`mor_refusals.py`, `mor_refusals.json`). The weight_index gate, the recursion
smoke test and the scale smoke test give the same numbers as before.
