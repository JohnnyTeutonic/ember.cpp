# `"tensor_index": "application"` on every backend: paperkiln's ar_rec (27 Oct 2026)

The model is paperkiln's ar_rec (`C:/ml_artifacts/attnres_gate_rec/ar_rec/`, paperkiln b9d5ce06, keramos v1.3.5
c4696bf5; files from `tests/test_llama_attnres_ir.py` with `ATTNRES_GATE_DIR`): llama_attnres in the full form under
fixed recursion 1 + [2] x 2 + 1, so 6 applications over 4 stored layers (`weight_index` [0, 1, 2, 1, 2, 3]), d 32,
2 heads, 16 positions, 40 CPU steps on a 15-word vocabulary. Each application's two depth_attend nodes carry
`"tensor_index": "application"` (IR_SPEC 7.2), so application i reads its own pseudo-queries `rec_app.{i}.attnres_{attn,ffn}_q`
while every other `{i}` name in the block binds the stored layer m_i. paperkiln's figures: numpy IR against
`mtstudio score` 3.2e-7, greedy 40/40; the document without its tags (application i reading `rec_app.{m_i}`) 2.1e-2
away (max |logit d| / max(1, max |ref|), first 8 prompts).

ember **ec3201b** (ember main and origin/main when tested) runs it on the CPU interpreter, in WebAssembly, on CUDA
and on WebGPU (SwiftShader) with no engine change: aa50a61's binding (`AppIndex` and `layer_name` in graph_ir.cpp)
resolves a tagged node's names by application in binding, running and compiling the device plan, and every path
here goes through it.

Every sha256 in ar_rec's `manifest.json` matches (7/7: ar_rec.gguf 62c43b66, ar_rec_q8_0.gguf 868bd9b9,
ar_rec.ir.json 22c3e187, v.gguf 879b0dcb, score_in 7c53d154, score_out 8af0e9c6, spec.json f6d095e0). The document
embedded in both GGUFs equals ar_rec.ir.json. Its 13 pseudo-queries are [d] F32 in both files (the Q8_0 file's
matrices are Q8_0).

## Results

The untagged document is ar_rec.ir.json with both `tensor_index` keys removed (sha256 in `tig_gate_cpu.json`),
served through `EMBER_IR` (CPU, WebAssembly in node, CUDA) or `--doc` (WebGPU).

| check | CPU (EXACT reference) | WebAssembly | CUDA, Colab T4 | WebGPU, SwiftShader |
|---|---|---|---|---|
| F32: log-probs of the 40 continuations vs `mtstudio score`, worst \|d\| / max(1, \|ref\|) (gate 1e-5) | 2.2e-7 | 2.2e-7 | 3.4e-7 | |
| F32: logits vs keramos v1.3.5 `ir_numpy` over the same file, max \|d\| / row range (gate 1e-5) | 2.7e-7 | | | |
| F32: against ember's CPU, max \|dlogit\| / row range; top-1 | | 0 (bit-equal); 473/473 | 2.0e-7; 1419/1419 | 1.5e-6; 320/320 |
| F32: greedy (argmax at every scored position) / free-running greedy | 40/40 | 40/40 / 40/40 | 40/40 / 40/40 | 40/40 (8 tokens) |
| F32: decode vs prefill | bit-equal | bit-equal | 2.0e-7 | (one token per step) |
| Q8_0: logits vs `ir_numpy` over the Q8_0 file (gate 1e-5) | 2.4e-7 | | | |
| Q8_0: against ember's CPU; top-1 | | 0 (bit-equal); 473/473 | 2.1e-7; 1419/1419 | 7.4e-7; 320/320 |
| Q8_0: greedy / free-running; decode vs prefill | 40/40; bit-equal | 40/40 / 40/40; bit-equal | 40/40 / 40/40; 2.4e-7 | 40/40 |
| Q8_0: log-probs vs `mtstudio score` (F32 weights; the quantisation's distance, reported) | 1.6e-4 | 1.6e-4 | 1.6e-4 | |
| random ids, 40 x 8 ids + 8 greedy (`ir_wasm_gate`, `ir_device_gate`, `ir_gate.mjs`), F32; Q8_0 | | 40/40, 320/320, bit-equal; the same | 40/40, 320/320, 1.1e-6; 40/40, 320/320, 4.2e-7 | the row above |
| `EMBER_DETERMINISTIC=1` run to run, both files | | | byte-equal | |

**The run uses the application index.** On every backend the tagged and untagged documents give different
functions, each matching its own reference:

| | CPU | WebAssembly | CUDA (T4) | WebGPU |
|---|---|---|---|---|
| untagged document vs its own reference (`ir_numpy` of the untagged document on the CPU; ember's CPU untagged run elsewhere), F32; Q8_0 | 2.0e-7; 2.4e-7 | bit-equal (F32) | 2.1e-7; 2.0e-7 | 9.0e-7; 3.9e-6 |
| tagged vs untagged, paperkiln's measure, first 8 prompts, F32 (paperkiln's numpy: 2.1037e-2) | 2.1037e-2 (1.0e-7 of paperkiln's figure) | the CPU's (bit-equal runs) | 2.1037e-2 | |
| the same, Q8_0; all 40 sequences, F32 | 2.1069e-2; 2.1210e-2 | | 2.1069e-2; 2.1210e-2 | |
| untagged run's log-probs vs `mtstudio score`, F32 (tagged: 2.2e-7) | 2.6e-3 | 2.6e-3 | 2.6e-3 | |
| untagged document against the tagged reference (`ir_gate.mjs`, must fail), F32; Q8_0 | | | | fails: worst 0.71 of the row range (median 1.7e-2), top-1 316/320; the same for Q8_0 (0.71, 316/320) |
| `rec_app.{4,5}.*` (read only through the tags) set to NaN: tagged logits; untagged logits | all NaN; bit-equal to the untagged run | | all NaN; byte-equal to the untagged run (both files) | |

The greedy continuations survive the untagged reading (40/40 on every backend), so greedy alone would not tell
the readings apart; the log-probs and logits do, by four orders of magnitude over the gate. `rec_app.0.attnres_attn_q`
is written as zeros and never read (its depth_attend has one source): a copy with it set to NaN gives bit-equal
tagged logits on the CPU, both files.

- CPU (`tig_gate_cpu.json`): `tools/ir_logits` on a CPU-only Release build of ec3201b from a git archive (sha256
  4402739f).
- WebAssembly (`wasm_ar_rec*.json`, `wasm_random_ar_rec*.json`, `wasm_untagged_probe.json`): `tools/state_wasm_gate`
  and `tools/ir_wasm_gate` built by `../wasm_state_2026-10-25/build.sh` (Emscripten 6.0.6, `-O3 -msimd128`, 1gb heap;
  `receipts/wasm_build.txt`), compared by that folder's `compare.py`: every integer result and greedy id equal to
  native's, decode = prefill and eviction on = off bit for bit inside the WebAssembly engine. node hands `EMBER_IR`
  to the engine, so the untagged probe ran the untagged document in WebAssembly.
- CUDA (`cuda_gate.json`, `receipts/cuda_*`): one Colab T4 session (`ember-tig-1027`, 6 minutes; build 200 s at sm_75,
  nvcc 13.0.88, driver 580.82.07). `../attnres_x5_2026-10-25/x5_device_logits.cpp` (refuses unless the interpreter is
  on the GPU) for the log-probs, decode and free-running greedy; `tools/state_device_gate` (device against the CPU in
  one process, kv rows held 40/40, pass on both files); `tools/ir_device_gate` on random ids. The default mode's two
  runs were also byte-equal at this size.
- WebGPU (`webgpu_gate.json`, `webgpu/<run>/summary.json`, `check.jsonl`): `tools/webgpu_parity/ir_gate.mjs`
  (`deploy/wasm/web/webgpu/ir_webgpu.js`, which resolves `tensor_index` itself) through `swiftshader_run.py`: Dawn's
  node bindings (npm `webgpu` 0.6.2) on SwiftShader only (`google | swiftshader | swiftshader-device-subzero- |
  SwiftShader driver 5.0.0`), the GPU-counter monitor read 9-11 times per run and saw no hardware context in any of
  the six runs; 132 dispatches per token, 1.7-2.3 s per prompt on two cores. References from the native
  `ir_wasm_gate` (`tig_wgref.sh`, `EMBER_IR_CTX` 1024). `tools/webgpu_parity/run.py` (a browser on a hardware GPU)
  was not used: its `EMBER_WEBGPU_REMOTE` guard keeps it to remote VMs.

## Regression

No engine code changed, so no earlier gate moves; on this build (`receipts/cpu_regress.txt`):
`../attnres_2026-10-24/attnres_gate.py` (ar3) writes a receipt equal to the committed one in every field but `bin`,
and `../ir_adapter_2026-10-25/adapter_selftest.py` passes 189/189 checks (3,072 requests, no failure), every record
identical to the committed receipt's. The ir_adapter binary is a73412f6, the same binary as 1ab6a25's and 7526469's.

## Files

- `tig_gate_cpu.py` (CPU gate, both files, the untagged document and the NaN copies), `tig_job.sh` (the CPU-budget
  wrapper, session ember, 2 cores), `tig_regress.sh`.
- `tig_wasm.sh`, `tig_inputs.py`, `tig_logprobs.py`.
- `tig_stage.sh`, `tig_nan.py`, `tig_vm.sh`, `tig_drive.sh` (microtorch/AGENTS.md path C: a T4 asked for first, an L4
  only if none is granted; the session stopped on every exit path, and it was stopped), `tig_analyze_cuda.py`.
- `tig_wgref.sh` (WSL), `tig_webgpu.py` (Windows).

The build tree, the archive of ec3201b, the staging copies and the device logits dumps were deleted once the
receipts were written. To rerun: `git -c core.autocrlf=false archive --format=tar.gz --prefix=ember/ -o ARCHIVE ec3201b`,
then in WSL a CPU build of it at `~/tig/ember` (cmake Release, `HAS_CUDA=OFF`; targets ir_logits, tinyllama_server,
ir_adapter), `bash tig_job.sh gate_cpu python3 tig_gate_cpu.py OUT.json ~/tig/ember/build_cpu`,
`bash tig_job.sh wasm bash tig_wasm.sh OUT`, `bash tig_stage.sh ARCHIVE ec3201b ~/tig/stage && bash tig_drive.sh
ember-tig-1027 60 ~/tig/stage`, `python3 tig_analyze_cuda.py <session>/out OUT.json`, `bash tig_job.sh wgref bash tig_wgref.sh`,
and on Windows `python tig_webgpu.py webgpu_gate.json` with `WEBGPU_NODE` set.
