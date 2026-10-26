# The IR conformance adapter's self-test

`adapter_selftest.py` drives `tools/ir_adapter.cpp` (IR_SPEC section 11.3) over JSON Lines on paperkiln's gate
models and keramos's suite; its docstring lists the checks. Receipt: `adapter_selftest_cpu.json` (189 checks
since spec revision 1.3, `../ir_rev13_2026-10-27`). Run (WSL, under the CPU budget):
`IR_ADAPTER_BIN=<build>/ir_adapter python3 adapter_selftest.py` (without `IR_ADAPTER_BIN` it tests
`~/ember_wt/build_cpu/ir_adapter` and writes the receipt here).

## Defaults: the relabelled models and the v1.3.0 suite

Since keramos v1.3.0's rule 38 an IR-only file labelled `general.architecture` "gpt2" is refused at load. Two of
the gate models, `kimi_tiny` and `swa_tiny`, are such files, and the older suites' `pk_kimi` and `pk_recurrence`
too, so the self-test run with its old defaults (`copy_models.sh`'s copies, `microtorch/spec/conformance`, the
keramos working tree) stopped at a rule-38 refusal (request 631, `../verify_2026-10-27/receipts/cpu/cpu_regress.txt`). The defaults are now what the revision 1.3 gate ran
with (`../ir_rev13_2026-10-27/gates.sh`):

- gate models: `~/ember_wt/adapter_models_r13`, the ext4 cache of `../ir_rev13_2026-10-27/relabel_models.sh`
  (`copy_models.sh`'s models as symlinks; `kimi_tiny` and `swa_tiny` relabelled by keramos v1.3.0's
  `tools/gguf_relabel.py`: architecture "paperkiln", the `gpt2.*` keys renamed, document, digest and tensors
  unchanged);
- `IR_SUITE`, `IR12_SUITE`, `IR13_SUITE`: `~/keramos_v1.3.0/conformance`, a git archive of the tag (keramos's
  checkout untouched), whose `pk_kimi` and `pk_recurrence` are relabelled at the tag.

The script runs `relabel_models.sh` when either is missing; the variables still override them. The originals in
`/mnt/c/ml_artifacts` are only read (receipts hash their bytes).

Rerun on 27 Oct 2026 through `../cuda_state_2026-10-25/cpu_regress.sh` (no model or suite variable), ember
4ddd218's sources, CPU build: **189/189, 3072 requests**; the report equals the
committed `adapter_selftest_cpu.json` (1ab6a25's) in every check, byte for byte apart from the `bin` path.
