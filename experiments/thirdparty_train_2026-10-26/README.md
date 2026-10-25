# Third-party ops trained by paperkiln: GQA, NEOX and partial RoPE, per-head q/k RMSNorm, MQA, Mamba-1 (26 Oct 2026)

Five small models paperkiln 70638baf trained with third-party ops (keramos pin v1.1.4, no IR change), in
`C:/ml_artifacts/thirdparty_train_gate/<name>/`: vocab 15, d 32, T 16, 2 layers, 40 steps. paperkiln's numpy IR
against `mtstudio score`: worst 2.5e-7 to 4.3e-7, greedy 40/40 on all five.

ember **5ad59b0** (`5ad59b0569eeb6c0faa2d64058ae1fcab04533ab`, origin/main when tested), built from `git archive` on
ext4 (CPU) and on the VM (CUDA), runs all five on the CPU and on CUDA with no engine change. Every sha256 in each
`manifest.json` matches (34/34 files), and each GGUF's embedded `paperkiln.ir` equals its `<name>.ir.json`.

| case | ops | arch, family_hint | path ember serves it on |
|---|---|---|---|
| tp_gqa | 4 q / 2 kv heads, interleaved RoPE | llama, llama | **tuned llama path** (CPU and CUDA) |
| tp_qwen | 4/2 GQA, NEOX, per-head q/k RMSNorm (group 8, before RoPE), theta 1e6 | paperkiln, llama | interpreter |
| tp_mqa_partial | 1 kv head, interleaved, n_rot 4 of head_dim 8 | paperkiln, llama | interpreter |
| tp_neox_partial | MHA, NEOX, n_rot 4 of 8 | paperkiln, llama | interpreter |
| tp_mamba1 | Mamba-1, llama.cpp tensor names | paperkiln, mamba1 | interpreter |

## The path each case takes

Three readings agree for every file, F32 and Q8_0 (`gate_cpu.json`, `path`): `tools/ir_path_check`
(graph_ir.h `tuned_llama_match`), `tinyllama_server`'s own load log at gpu_layers 0, and `tp_session_logits`, which
builds the session exactly as server.cpp does and reports whether it holds an interpreter. On the T4 the same
session at gpu_layers -1 reports tp_gqa on the tuned path on CUDA and the other four on the interpreter on CUDA.

- tp_gqa: the structural match accepts it (4/2 heads from `llama.attention.head_count_kv`); the tuned path serves it.
- Cases 2-4 carry family_hint "llama" and go to the interpreter. In these files the reason the match gives is
  `model.d 32 != the file's embedding_length`: their architecture is "paperkiln", so the llama metadata the match
  reads is absent, and the match stops before it reaches the ops. The server log says so:
  `[IR] family_hint llama, but not the tuned llama path's graph (model.d 32 != the file's embedding_length): the
  interpreter runs it`.
- **Relabelled probe** (`relabel_probe.txt`, `tp_relabel.py`): copies of cases 2-5 rewritten as architecture
  "llama" (each `paperkiln.<key>` hyperparameter written as `llama.<key>`; the IR and every tensor unchanged), so the
  match reads the same dimensions as the IR and has to judge the ops. It still refuses every one, on the op:
  tp_qwen `the block has 17 nodes` (the two q/k norms), tp_mqa_partial and tp_neox_partial `rope is not interleaved
  on every head, n_rot = head_dim, freq_scale 1`, tp_mamba1 `family_hint is not llama`. Served from the copies, the
  logits, decode and greedy ids are bit-equal to the original files'. So none of cases 2-4 can land on the tuned
  path, either as shipped or as a llama-architecture file.

## CPU (`gate_cpu.json`; ext4 Release build, 2 cores)

Log-probs of the 40 continuations against `mtstudio score`, worst |d| / max(1, |ref|) (ROUNDING 1e-5), on the path
the server takes (`tp_session_logits`, gpu_layers 0); `tools/ir_logits` (the interpreter) beside it:

| | tp_gqa (tuned) | tp_qwen | tp_mqa_partial | tp_neox_partial | tp_mamba1 |
|---|---|---|---|---|---|
| log-probs vs mtstudio | 1.6e-7 | 2.2e-7 | 2.2e-7 | 1.7e-7 | 2.2e-7 |
| greedy: argmax at every scored position = continuation (= mtstudio's argmax) / free-running | 40/40 (40/40) / 40/40 | 40/40 (40/40) / 40/40 | 40/40 (40/40) / 40/40 | 40/40 (40/40) / 40/40 | 40/40 (40/40) / 40/40 |
| decode vs prefill | 1.1e-7 of the range | bit-equal | bit-equal | bit-equal | bit-equal |
| served path vs `ir_logits` | prefill 1.1e-7, decode bit-equal | bit-equal | bit-equal | bit-equal | bit-equal |
| Q8_0: loads, greedy (free-running), decode = prefill; log-probs vs the F32 references | yes, 40/40 (40/40), 1.2e-7; 2.1e-4 | yes, 40/40 (40/40), bit-equal; 3.3e-4 | yes, 40/40 (40/40), bit-equal; 1.5e-4 | yes, 40/40 (40/40), bit-equal; 2.2e-4 | (F32 only) |

The tuned path's token-by-token decode and its batched prefill use different CPU kernels, so they agree in
rounding rather than bit for bit; its decode is bit-equal to the interpreter's. The Q8_0 log-probs move by the
quantisation against mtstudio's F32 references (reported, not gated; greedy is gated).

## CUDA (`cuda_gate.json`; Colab T4, sm_75, nvcc 13.0.88, session `ember-tptrain-1026-t4`, stopped)

`tp_session_logits` at gpu_layers -1 (the server's own path and device choice), `EMBER_GEMM_TF32=0`; max |dlogit|
over the CPU row's range against the CPU run on the same path and file; ROUNDING 1e-5:

| | tp_gqa (tuned) | tp_qwen | tp_mqa_partial | tp_neox_partial | tp_mamba1 |
|---|---|---|---|---|---|
| log-probs vs mtstudio | 2.7e-7 | 3.2e-7 | 2.6e-7 | 2.6e-7 | 3.3e-7 |
| greedy teacher-forced (= mtstudio's argmax) / free-running | 40/40 (40/40) / 40/40 | 40/40 (40/40) / 40/40 | 40/40 (40/40) / 40/40 | 40/40 (40/40) / 40/40 | 40/40 (40/40) / 40/40 |
| prefill vs CPU; decode vs CPU | 2.0e-7; 1.8e-7 | 2.0e-7; 2.0e-7 | 1.7e-7; 1.8e-7 | 2.1e-7; 1.7e-7 | 1.8e-7; 1.8e-7 |
| top-1 vs CPU, prefill and decode | 473/473, 473/473 | 473/473, 473/473 | 473/473, 473/473 | 473/473, 473/473 | 473/473, 473/473 |
| decode vs device prefill | 1.8e-7 | 1.9e-7 | 1.5e-7 | 2.0e-7 | 1.8e-7 |
| `state_device_gate` (interpreter, device vs CPU in one process) | pass, 2.5e-7 | pass, 2.0e-7 | pass, 1.8e-7 | pass, 2.1e-7 | pass, 1.8e-7 |
| `ir_device_gate`, 40 random prompts x 8 ids, 8 greedy | 40/40, 320/320, 1.7e-7 | 40/40, 320/320, 2.4e-7 | 40/40, 320/320, 2.2e-7 | 40/40, 320/320, 2.1e-7 | 40/40, 320/320, 1.9e-7 |

The Q8_0 files on the device, against the CPU on the same file: greedy 40/40 both ways, top-1 473/473, prefill,
decode and decode-vs-prefill within 2.4e-7, `state_device_gate` and `ir_device_gate` pass (log-probs vs the F32
references 1.5e-4 to 3.3e-4, as on the CPU). tp_gqa with the interpreter forced on the device (`EMBER_INTERPRETER=1`):
log-probs 2.7e-7, greedy 40/40 both ways, 1.7e-7 against the CPU interpreter, top-1 complete (Q8_0 likewise). The
default-TF32 run of tp_gqa equals the TF32-off run exactly: the T4 has no TF32 path, so this says nothing about an L4.
No L4 was free (paperkiln's three L4 sessions held the account's slots; a fourth returned TooManyAssignmentsError),
so the driver took a T4, as it is written to.

## Files

- `tp_session_logits.cpp`: the session as server.cpp builds it, then api.cpp's prefill, token-by-token decode and
  free-running greedy on raw ids, on whichever path and device the session chose (it reaches the session's private
  members by redefining `private` around api.h; the library is unchanged). Built as an added CMake target.
- CPU: `tp_build_cpu.sh` (ember from the archive on ext4, ir_logits, ir_path_check, tinyllama_server,
  tp_session_logits), `tp_job.sh` (the CPU-budget wrapper, session ember, 2 cores), `tp_gate_cpu_run.sh` /
  `tp_gate_cpu.py` (`gate_cpu.json`), `tp_relabel_probe.sh` with `tp_relabel.py`, `tp_relabel_cmp.py`,
  `tp_inspect_relabel.py` (`relabel_probe.txt`).
- Colab (microtorch/AGENTS.md path C): `tp_stage.sh` (with `tp_inputs.py`), `tp_drive.sh` (an L4 asked for first,
  then a T4; the session stopped on every exit path), `tp_vm.sh`, `tp_analyze.py` (`cuda_gate.json`). Receipts from
  the VM in `receipts/` (`vm.txt`, `hardware.txt`, `bins.sha256`). The logits dumps, builds and staging copies were
  deleted once the comparisons were written.

The archive the scripts read (`tp_ember.tgz`) is made in this folder with Windows git and deleted after the run:

    git -c core.autocrlf=false archive --format=tar.gz --prefix=ember/ -o tp_ember.tgz 5ad59b0569eeb6c0faa2d64058ae1fcab04533ab

Then, in WSL: `bash tp_job.sh build_cpu tp_build_cpu.sh`, `bash tp_job.sh gate_cpu tp_gate_cpu_run.sh`,
`bash tp_relabel_probe.sh`, `bash tp_stage.sh ~/tptrain_stage 5ad59b0569eeb6c0faa2d64058ae1fcab04533ab &&
bash tp_drive.sh ember-tptrain-1026 90 ~/tptrain_stage`, and `python3 tp_analyze.py
/mnt/c/ml_artifacts/ember_tptrain_gate/ember-tptrain-1026-t4/out ~/tptrain/work cuda_gate.json`.
