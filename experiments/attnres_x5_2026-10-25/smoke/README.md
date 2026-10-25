# Arm X5 real-vocabulary smoke model at e115fd3: CPU and WebAssembly (26 Oct 2026)

paperkiln's `tc_smoke_20m_X5` (`C:/ml_artifacts/attnres_gate_x5/smoke_20m_X5/`): 20m_L16 with `llama_attnres`
(full form, scale `"sources"`, 33 depth sources), d 320, 16 layers, 5 heads, d_ff 864, the 16,384-entry Llama 3
vocabulary embedded, 300 steps on an L4, F32, ir_version 1. Prompts: the spot-check's 40 Carry-5 items (0, 26, ...,
1014), the target user turn with its gold history in the served template, 8 greedy tokens each. paperkiln's numpy
IR against `mtstudio score`: worst 1.9e-6, greedy 40/40.

Every sha256 in `manifest.json` matches (5/5: tc_smoke_20m_X5.gguf 3735ee54, .ir.json 67d1d837, score_in 1c901ad9,
score_out 392e8e4f, spec.json 9d1ce971). The GGUF's embedded IR equals `tc_smoke_20m_X5.ir.json`.

ember **e115fd3** (`e115fd3d23ad5cee1ab7fb3bd0c8bb249ef69555`) from a git archive, CPU Release on ext4 (81 s at 2
cores). No CUDA leg here: the X5 post jobs' own 40-item spot-check on the L4 is the L4 gate.

| check | CPU (e115fd3) | WebAssembly (e115fd3) |
|---|---|---|
| log-probs of the 40 continuations (320 positions) vs `mtstudio score`, worst \|d\| / max(1, \|ref\|) (gate 1e-5) | 1.6e-6 | 1.7e-6 |
| greedy: argmax at every scored position = paperkiln's continuation (= mtstudio's argmax) | 40/40 (40/40) | 40/40 |
| free-running greedy = continuation | 40/40 (native state_wasm_gate) | 40/40; ids equal native's |
| decode vs prefill | bit-equal | bit-equal |
| against ember's CPU, max \|dlogit\| over the CPU row's range; top-1 | | 2.0e-7; 3699/3699 rows (prefill, decode) |
| `blk.0.attnres_attn_q` (all zeros) set to NaN | logits bit-equal: never read | |

- CPU (`x5s_gate_cpu.json`): `tools/ir_logits` on the embedded IR, one process per sequence (prompts 53 to 145
  tokens; a single [T, 16,384] JSON dump of all 40 would be gigabytes).
- WebAssembly (`x5s_wasm.json`): `tools/state_wasm_gate`, native (g++ -O3 against this build's
  `libtinyllama_core.a`) and WebAssembly (Emscripten 6.0.6, `-O3 -msimd128`, 1gb heap, node), built by
  `../../wasm_state_2026-10-25/build.sh` in 111 s; compared by that folder's `compare.py` in four chunks of ten
  sequences (each chunk's logits about 190 MB per engine, deleted once compared). Every integer result equal.

## Served: tinyllama_server /chat (CPU)

`x5s_chat_cpu.json`: the same 40 items through e115fd3's `tinyllama_server` (gpu_layers 0, `EMBER_GEMM_TF32=0`),
carry5_eval.py's gold mode and greedy body (temperature 1, top_k 1, top_p 1, seed 7, return_ids),
max_new_tokens 8. The server reports chat template `paperkiln` v1 (declared), context 512, BPE tokenizer, vocab
16,384.

- prompt_ids equal score_in's prompt_ids: **40/40** (the score files' prompts are the served form);
- generated ids (with the stop id) equal paperkiln's continuation up to the stop: **40/40** (eos at id 1 after six
  ids, or the 8-token length);
- every sampled position's logits (`TINYLLAMA_LOGITS_DUMP`, nine significant digits) against ir_logits' decode rows:
  bit-equal, 40/40 requests, 290 positions, max \|d\| 0.

## Files

`x5s_build.sh`, `x5s_cpu_legs.sh` (`x5s_gate_cpu.py`, `x5s_chat.py`), `x5s_wasm.sh` (`x5s_wasm.py`), `x5s_job.sh`
(the CPU-budget wrapper, session ember, 2 cores); logs as `.txt`. The archive was made with
`git -c core.autocrlf=false archive --format=tar.gz --prefix=ember/ e115fd3` and extracted to `~/x5s/src`; then, in
WSL, `bash x5s_job.sh build x5s_build.sh`, `bash x5s_job.sh cpu_legs x5s_cpu_legs.sh OUT`,
`bash x5s_job.sh wasm x5s_wasm.sh OUT`. The build, archive and logits were deleted afterwards.
