# CUDA: Mixture-of-Recursions on the device (25 Oct 2026)

ember 66c8b2c runs Mixture-of-Recursions documents (`depth_route`, causal token choice; kv state `kv_rows`
`"active"`; IR_SPEC 7.4, 7.5) on the interpreter's CUDA path, so `device_supported()` accepts them and CUDA builds
stop serving them on the CPU. The CPU path is unchanged and stays the EXACT reference; device results are ROUNDING
against it. Gated on Colab T4s (sm_75). Session `ember-mor-1025` ran every leg but lost its VM (it left the session
list) during the last third-party dumps, before `out/` came home; its log came home (`receipts/vm_session1.txt`), and
the third-party results below are from it. Session `ember-mor-1025b` reran the local legs (`/content/LEGS` "local")
and gives the rest of `receipts/`; its driver stopped it at the end. The VMs built 4b6a0b6 and 258d747 (66c8b2c plus
this folder's scripts) and ember main before it, 51eed18. The legs both sessions ran gave the same numbers.

How the device runs it (graph_ir.h, `mor_step_device`):
- **Routing.** At the first application of step 0 the router logits x W^T come from cuBLAS, as `linear`; the depth
  (1 + argmax, ties to the lower index) and the gate p[depth - 1] are computed on the host by the CPU's own
  `cpu::mor_route`, so a depth is EXACT given its logits. One host sync per forward.
- **A step** runs batched over its active tokens (depth > step), gathered into compact rows in position order. RoPE
  reads each row's absolute position from a device array (`ir_rope_at_cuda`, `ir_rope_rows_at_cuda`: the old
  kernels' per-pair arithmetic). Attention appends the rows to the application's compact kv cache after the rows it
  holds and reads causally there (the existing prefill and decode kernels at that row base), so query t sees exactly
  the rows held at positions <= its own. The block's other inputs are re-gathered at each application and every slot
  it touches is returned after the step, as the CPU builds each token's row from the full values.
- **Gate.** After the step's last application, gate `"score"` blends without contraction (`ir_mor_gate_cuda`:
  `__fsub_rn`, `__fmul_rn`, `__fadd_rn`, bit-equal to `cpu::mor_gate` given its inputs); the rows go back into the
  full x, inactive rows keeping x_in. A step with no active token runs nothing and writes no rows.
- **Bookkeeping** is the CPU's: the rows each application holds (`mor_rows_`), a forward at start_pos keeping the
  rows below it (rewinds, prefix reuse), `last_depths()` for the trace; `held_positions(i)` reads an application's
  rows on either path.
- **Decode** runs per launch: a forward through a depth_route never takes the captured CUDA graph, since the routing
  reads the host. A `pos_embedding` inside a MoR block keeps the document on the CPU.

Receipt: `cuda_gate.json` (`analyze.py` over the VM's `out/`), the VM's records in `receipts/`; the full output in
`/mnt/c/ml_artifacts/ember_mor_cuda/ember-mor-1025/`.

## (a) Kernel test (T4)

`tools/mor_kernels_test.cu`: PASS, 51 of 51 bit comparisons: gather and scatter against host row copies; the gate
against `cpu::mor_gate`; the position-array RoPE (interleaved, NEOX full and partial; head dims 16, 64, 80) against the
old kernels at contiguous positions and, at scattered positions, against one T = 1 launch per row.

## (b) Mixture-of-Recursions: CUDA vs CPU (`tools/mor_device_gate`)

The models of `../recursion_2026-10-23`'s MoR gate (`C:/ml_artifacts/mor_gate`), their 40 `score_in` rows, with
mtstudio score's depth traces and log-probabilities as references; max |dlogit| over the CPU row's range:

| check | mor3 (1 + [2] x 3 + 1) | mor2p0 (0 + [3] x 2 + 1) |
|---|---|---|
| greedy: teacher-forced, device and CPU / free-running, device = CPU | 40/40, 40/40 / 40/40 | 40/40, 40/40 / 40/40 |
| top-1 vs CPU (one-call prefill, token-by-token decode, two-call prefill; every position) | 1299/1299 | 1299/1299 |
| prefill / decode / two-call vs CPU | 1.8e-7 / 1.7e-7 / 1.8e-7 | 2.0e-7 / 2.0e-7 / 2.0e-7 |
| decode vs the device's prefill | 1.9e-7 | 2.3e-7 |
| log-probabilities, device vs CPU / vs mtstudio (CPU vs mtstudio) | 2.6e-7 / 2.8e-7 (2.3e-7) | 2.6e-7 / 3.2e-7 (1.9e-7) |
| depth trace = CPU's: prefill, decode, two-call (= mtstudio's) | 40/40, 40/40, 40/40 (40/40) | 40/40, 40/40, 40/40 (40/40) |
| rows held per application = CPU's after each run (mean rows per application) | 40/40 (8.36) | 40/40 (8.36) |
| rewinds to L/2 in one call and to L-3 token by token, altered then original suffix: vs CPU, vs a fresh CPU prefill, depths and rows held = CPU's | 160/160, 2.0e-7, 1.8e-7 | 160/160, 2.0e-7, 2.0e-7 |

The Q8_0 files (`mor3_q8_0`, `mor2p0_q8_0`) give the same device-against-CPU results: top-1 1299/1299 each, logits
within 2.4e-7, depths, rows held, greedy and rewinds complete. The tool's own `pass` is false for them only because
they were given the F32 models' mtstudio references, which the CPU misses by the same amount (its log-probabilities
move by the quantisation, 1.3e-2 and 3.6e-4; one of mor3_q8_0's 40 depth traces differs from the F32 model's);
`analyze.py` gates them device against CPU.

The tiny-chat MoR3 model (`tc_e1_mor_MoR3_r1.gguf`, d 640, 8 applications over 4 stored layers, vocab 16384, sha256
fa0f8a68...; 16 seeded random prompts of 24 ids, 32-id continuations): top-1 2640/2640, prefill 6.4e-7, decode 3.9e-7,
two-call 4.8e-7, decode vs prefill 5.8e-7, log-probabilities 3.4e-7, depths 16/16 three ways, rows held 16/16 (mean
34.4 of 55 positions per application), free-running greedy device = CPU 16/16, rewinds 64/64. Time on the T4 VM:

| | prefill, 24 tokens | decode, per token |
|---|---:|---:|
| ember CPU (the VM's CPU) | 346 ms | 24.2 ms |
| ember CUDA (T4) | 3.8 ms | 1.41 ms |

`ir_device_gate` on random ids: mor3 and mor2p0 40/40 greedy, 320/320 top-1 (worst 2.1e-7); the tiny-chat MoR3 model
10/10, 240/240 (4.6e-7).

## (c) The device-gate set: nothing moved

`ir_device_gate` (66c8b2c), 40 prompts x 24 ids, 32 greedy (ctx-16 models 8 and 8): synthetic mistral_swa, gptneox,
gpt2, mamba, gemma3 (+ per-layer freq_scale, + weight_offset), gpt2_tiny's expr IR and state4: 40/40 and 960/960
each (worst 2.4e-6); ar3 and the three tied-block recursion models of `../recursion_2026-10-23` (p1_b2x3_c1, its
inject variant, b2x4): 40/40 and 320/320 each (worst 4.9e-7), their first device receipt. Third-party, 10 prompts x 24
ids, 24 greedy, by the real-model rule of `../thirdparty_2026-10-22/device_gate` (greedy and top-1 decide; the flat
1e-4 of the `pass` field is not the criterion): Gemma-3-270m-it 10/10, 240/240 (2.5e-5), Qwen2.5-0.5B-Instruct f16
10/10, 240/240 (5.9e-5), Pythia-160m 10/10, 240/240 (2.7e-4, its known float32 conditioning; 3.9e-4 on the L4 there).
Qwen3-0.6B's CPU half was killed by the VM's memory (rc 137: the IR declares 40960 positions of f32 kv on both
sides); its device dump below is bit-identical to 51eed18's.

`tools/ir_device_dump` from 66c8b2c and from 51eed18 on the same T4 (every prefill row, the token-by-token decode,
32 greedy ids): identical files on all 18 runs: the eight synthetic ones, state4 and its Q8_0, ar3, the three
tied-block recursion models (session 2), and Qwen3-0.6B Q8_0, Pythia-160m, Gemma-3-270m-it and Qwen2.5-0.5B-Instruct
f16 (session 1, the checkpoints fetched on the VM). The long chat prompts past 512 tokens were the session-1 leg the
VM was lost in; they were not rerun (`../cuda_state_2026-10-25` has them identical at aadeed2, and no code they run
changed here).

## CPU regression

`receipts/cpu_regress.txt`. This work's CPU build (ext4, Release) against the receipts as committed at HEAD:
- byte-identical: recursion gate, MoR gate, MoR refusals, attnres gate, state4 gate, the parity replays
  (`regress.py parity`), G5 third party;
- `regress.py bits` (51eed18's script): 16/16 bit-equal to ember 69d7ca2, the receipt equal to the committed one but
  for the build path it names;
- `state_tokens_gate` (51eed18's script): every ember-side field equal to the committed receipt; its nine
  numpy-reference distances (paperkiln's ir_numpy, worst 8.5e-7) differ, as ember 541a190 records for any rerun;
- the conformance adapter's self-test: 156/156.
The CUDA tree compiles (`../ir_rev11_2026-10-25/build_cuda_check.sh`, sm_75) and links the device tools.

## Files

- `make_inputs.py`, `stage.sh`, `drive.sh`, `vm.sh`: the Colab run (microtorch/AGENTS.md path C); `analyze.py` writes `cuda_gate.json`
  from session 2's `out/` and session 1's log.
