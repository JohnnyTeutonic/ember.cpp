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

CUDA: the interpreter compiles with HAS_CUDA; the device run waits for the next Colab device gate.
Reserved, not built: paperkiln's step 2 (per-token recursion depth, Mixture-of-Recursions).
