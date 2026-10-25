# CUDA: per-turn state tokens on the device, and the fallback prefill attention at any head_dim (25 Oct 2026)

Two CUDA changes, gated on one Colab T4 (sm_75, session `ember-cuda-1025b`, stopped at the end by its driver;
an earlier session, `ember-cuda-1025`, stopped itself when cmake ran before boost was installed):

- **ember 606e1d4**: the fallback prefill attention kernel (`attention_batch_prefill_cuda_kernel`) launched
  head_dim threads, and its softmax reductions halve `blockDim.x`, so any head_dim that is not a power of two
  lost partials, as `rmsnorm_batch_cuda` did at d 192 before 69d7ca2. The block is now head_dim rounded up to a
  power of two; the extra threads read and write nothing and hold the reductions' identities. A power-of-two
  head_dim launches as before. The kernel serves `EMBER_PREFILL_ATTN=baseline` and every shape flash attention
  does not take (head_dim 80, 160, 192, 200, ...). No other block reduction in `cuda_kernels.cu` or
  `graph_ir_cuda.cu` depends on a launch size that can be other than a power of two or whole warps
  (`rmsnorm_batch_kernel_optimized` halves too, but is never launched).
- **ember aadeed2**: the interpreter's CUDA path runs the extended embedding (`state_weight`), attention mask
  `turn_state` and kv `evict turn_state` (IR_SPEC 6.8, 6.9), so `device_supported()` accepts these files. The
  bookkeeping is the CPU path's: lazy eviction on the next turn's first position, state rows held, positions
  absolute, and a rewind into a segment whose rows were dropped re-prefilling from start(m) (`resume_point`).
  The mask kernel reads each kv row's code 2 s + c over the compact cache, an online softmax over chunks of
  1024 rows; eviction compacts each block's device cache in place. Device results are ROUNDING against the CPU,
  which stays the EXACT reference.

Receipt: `cuda_gate.json` (`analyze.py` over the VM's `out/`), the VM's records in `receipts/`, the full output
in `/mnt/c/ml_artifacts/ember_cuda_state/ember-cuda-1025b/`.

## (a) Kernel tests (T4)

| test | aadeed2 | aa50a61 (before) |
|---|---|---|
| `tools/prefill_attention_test.cu`: baseline kernel vs float64, head dims 16-256 | PASS, worst 2.3e-7 | FAIL at 48, 80, 96, 112, 160, 192, 200 (worst 10.5) |
| its outputs, aadeed2 vs aa50a61, bit for bit | equal at 16, 32, 64, 128, 256; differ only at the seven fixed dims | |
| `tools/rmsnorm_batch_test.cu` (18 widths) | PASS, outputs bit-equal to aa50a61 | PASS |
| `tools/turn_state_attention_test.cu`: the mask kernel vs float64 (head dims 32-128, 0/9/1100 held rows, row base from the host and from device memory), compaction and the extended embedding bit for bit | PASS, worst 2.8e-7 | |
| `tools/decode_attention_test.cu` (unchanged kernels) | all ok | |

## (b) State tokens and llama_attnres: CUDA vs CPU (`tools/state_device_gate`)

On paperkiln's state4 (`C:/ml_artifacts/state_gate/state4`, the 40 `score_in` sequences, eviction as the IR
declares it, and off with `EMBER_EVICT=0`), max |dlogit| over the CPU row's range:

| check | state4 | state4 Q8_0 |
|---|---|---|
| greedy: argmax at every scored position equal to the continuation / free-running greedy | 40/40, 40/40 (CPU 40/40, 40/40) | 40/40, 40/40 |
| top-1 vs CPU (prefill on, prefill off, decode; every position) | 9264/9264 | 9264/9264 |
| one-call prefill on / off vs CPU | 6.2e-7 / 6.2e-7 | 5.9e-7 |
| token-by-token decode (the CUDA graph) vs CPU; vs the device's prefill | 4.4e-7; 4.3e-7 | 4.4e-7; 5.2e-7 |
| turn-by-turn calls (eviction between them) vs CPU | 4.1e-7 | 5.9e-7 |
| kv rows held after the context, device on and off = CPU = mtstudio's (manifest) | 40/40; mean 25.35 evicting vs 76.2 | 40/40 |
| rewinds into turns 0 and 1: resume point = CPU's, direct forward refused, re-prefill vs CPU, rows held = CPU's | 80/80, 80/80, 6.5e-7 | 80/80, 80/80, 5.9e-7 |
| session script (`make_inputs.py`: generation, a regenerated reply, an edited earlier message, a rewind into turn 0; 12 calls) | 12/12; vs CPU 1.0e-6, on vs off 5.0e-7, vs a fresh prefill 5.9e-7; 2/2 direct rewinds refused | |

The synthetic k = 4 model of the CPU gate (`chat_state_k4.gguf`, sha256 68b302da...) on that gate's own session
script (`state_tokens_gate.py` (c), 12 calls with rewinds into evicted turns): 12/12, vs CPU 8.5e-7, on vs off
5.6e-7, vs fresh 7.7e-7, 2/2 direct rewinds refused, resume points and rows held equal to the CPU's.

ar3 (llama_attnres, `C:/ml_artifacts/attnres_gate/ar3`): greedy 40/40 both ways, top-1 1419/1419, prefill
2.0e-7, decode 2.0e-7. `ir_device_gate` on random ids: state4 40/40, 960/960, 7.6e-7; ar3 40/40, 320/320, 1.8e-7.

## (c) The device-gate set: nothing moved

`ir_device_gate` (aadeed2), 40 prompts x 24 ids, 32 greedy: synthetic mistral_swa, gptneox, gpt2, mamba,
gemma3 (+ per-layer freq_scale, + weight_offset) and gpt2_tiny with its gelu as an `expr`: 40/40 greedy and
960/960 top-1 each, worst 2.4e-6. `tools/ir_device_dump` from aadeed2 and from aa50a61 on the same T4 (every
prefill row, the graph decode, 32 greedy ids): identical files on all 16 runs: the eight synthetic ones, ar3,
Qwen3-0.6B Q8_0, Pythia-160m, Gemma-3-270m-it, Qwen2.5-0.5B-Instruct f16, and the long chat prompts past 512
tokens for Gemma-3-270m, Qwen2.5 and Qwen3 (the checkpoints fetched on the VM; sha256 in
`receipts/thirdparty.sha256`, Qwen3 and Pythia equal to the local copies).

## (d) The rmsnorm regression (SFT L32, d 192)

`tc_sft_L32_r1.gguf` (sha256 1187e448...) through aadeed2's `tinyllama_server` on CUDA (`EMBER_GEMM_TF32=0`), 40
tinychat prompts, both request variants of `../rmsnorm_2026-10-24`: replies 40/40 equal to mtstudio's CPU
records, and ids 40/40 equal to the T4 records of the 69d7ca2 fix.

## CPU regression

`receipts/cpu_regress.txt`. Byte-identical: recursion, MoR, MoR refusals, attnres, state4, the parity replays
and G5 third party; WASM 40/40 and 60/60 with receipts equal but for the timing; the conformance adapter's
self-test 119/119. The synthetic state-token gate's receipt differs from the committed one only in its numpy
reference fields, and the same run with aa50a61's build gives it byte for byte. regress.py bits: 15 of 16 cases
bit-equal to 69d7ca2; its hand-made recursion IR on gpt2_tiny is refused at load since the IR_SPEC validation
(e358153), by aa50a61 and aadeed2 alike.

## Files

- `make_inputs.py`, `stage.sh`, `drive.sh`, `vm.sh`: the Colab run (path C); `analyze.py` writes `cuda_gate.json`.
- `cpu_regress.sh`, `rungate.py`, `wasm_regress.sh`: the CPU regression against another build without touching
  the committed receipts.
