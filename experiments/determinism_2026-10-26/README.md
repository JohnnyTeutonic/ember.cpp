# Deterministic CUDA mode: `EMBER_DETERMINISTIC=1` (26 Oct 2026)

ember **fe01b35** (`fe01b35e3e50a049a7f1f5c9a5712d80c3be48b1`) adds an opt-in deterministic mode. With
`EMBER_DETERMINISTIC=1`, CUDA prefill and decode logits are bit-identical run to run on the same GPU and build. Unset,
every kernel, launch, cuBLAS setup and `/health` byte is what it was at d74e052 (ember main before it).

## Where the run-to-run differences came from

ember's CUDA decode was not bit-reproducible (`../attnres_x5_2026-10-25`: the same build twice on pick_e1_L16, 40/658
and 160/3941 decode positions bit-equal, up to 8.6e-6; fresh-prompt first tokens bit-equal). Every CUDA path was
checked for an order-dependent reduction:

| source | path | under the mode |
|---|---|---|
| `rmsnorm_optimized_kernel`: the warp sums go into one shared float by `atomicAdd`, in the order the warps arrive | the tuned llama decode, per-launch and in the captured CUDA graph (every layer's two norms and the final norm) | `rmsnorm_optimized_det_kernel`: the same arithmetic; each warp leader stores its sum in its own slot and thread 0 adds the slots in warp order from 0 (one of the orders the atomic kernel can produce) |
| `attention_optimized_kernel`: softmax sum by `atomicAddFloat` (CAS loop), max by `atomicMaxFloat` | the one-block-per-head decode attention: `EMBER_DECODE_ATTN=legacy`, or a shape past the split kernel's scratch (128 heads, head_dim 256, 131,072 positions); the interpreter's decode falls back to it there too | `attention_optimized_det_kernel`: max and sum in warp order |
| cuBLAS: its kernel choice can follow the workspace configuration, and some routines use atomics when allowed | every GEMM and matvec (the llama path's handle and the interpreter's) | `CUBLAS_WORKSPACE_CONFIG=:4096:8` before `cublasCreate` (unless the caller set it) and `CUBLAS_ATOMICS_NOT_ALLOWED` on each handle. cuBLAS is deterministic for a fixed GPU, toolkit and single active stream; ember's streams never run concurrently |

Already fixed-order, so unchanged: the split-K decode attention (`decode_attn_split_kernel` + merge: shuffle and
slot reductions, chunks merged in order), flash and flash2 prefill attention, the batched rmsnorm (shared tree),
MoE routing and combine (one thread per token, experts in index order), every interpreter kernel
(`graph_ir_cuda.cu` has no atomics), the KV INT8 quantiser's max (`find_max_abs_kernel`: one block per head at
head_dim <= 256, so its CAS loop runs once). `softmax_optimized_kernel` also adds by `atomicAddFloat`, but nothing
calls `softmax_vector_cuda_optimized`.

So the only source on any gated path was the decode rmsnorm. Prefix reuse explains why some Carry-5 first tokens
differed run to run: the reused KV rows were written by the previous turn's decode.

`/health` carries `"deterministic": true` under the mode (the field is absent when off, so the default `/health`
is unchanged) and the server's startup log says so.

## Gates (`cuda_gate.json`, `analyze.py`; `pass: true`)

Colab T4 (sm_75), both trees built as paperkiln's `ensure_ember` builds them (Release; `CMAKE_CUDA_ARCHITECTURES` the
GPU's own, 75, since no L4 could be had: paperkiln's three L4 sessions held the slots and a fourth returned
TooManyAssignmentsError). Session `ember-det2-1026-t4` (`receipts/s2_*`); session `ember-det-1026-t4` ran the same
legs first and gave the same numbers (`receipts/s1_vm.txt`), but its VM was lost before `out/` came home. Both VMs
were lost at the same point, the third-party gemma3_270m dump (as `../mor_cuda_2026-10-25`'s first session was), so
that leg and the four after it did not run. Each 360-request run is its own server process
(`../attnres_x5_2026-10-25/x5_plain_collect.py`: the first 40 tinychat prompts and the 40 Carry-5 items' 320
requests, greedy, `EMBER_GEMM_TF32=0`, gpu_layers -1, every sampled position's full logit vector at nine digits).

**On: run to run.**

| | replies, ids | every sampled position bit-identical | requests bit-identical at every position |
|---|---|---|---|
| pick_e1_L16 (20m_L16), on vs on | 360/360 | 4,599/4,599 (tinychat 658, Carry-5 3,941); worst \|d\| 0 | 360/360 |
| chat_sft_bpe (TinyChat SFT, d384 L6), on vs on | 360/360 | 5,657/5,657 (647; 5,010); worst \|d\| 0 | 360/360 |

For comparison, off vs off (pick, new build twice): 42/658 and 155/3941 positions bit-equal, worst 9.5e-6.

**Off: unchanged.** Off-mode decode is still not reproducible run to run, so off is held to d74e052 by what can be
fixed: the code itself and the outputs that are deterministic.
- SASS (`cuobjdump -sass` of `tinyllama_server`, kernel by kernel, the anonymous-namespace hash normalised): all 78
  of d74e052's kernels are identical in fe01b35, on sm_75 (the VM, `receipts/s2_sass_sm75.json`) and on sm_89 (the
  L4's architecture, built in WSL with `-DCMAKE_CUDA_ARCHITECTURES=89`, `receipts/sass_sm89_wsl.json`); the build
  adds the two `_det` kernels and nothing else. Off, the wrappers launch the same kernels with the same arguments,
  and the cuBLAS setup is untouched.
- Tuned llama path, new off vs d74e052: replies, stop and ids 360/360 on both models, fresh-prompt first-token
  logits bit-equal 40/40 on both; the decode positions differ by the same spread as off vs off (pick: 48/658,
  189/3941 bit-equal, worst 8.6e-6; chat_sft_bpe: 474/647, 4074/5010, worst 6.7e-6).
- Interpreter (`tools/ir_device_dump`: prefill logits, token-by-token decode through the captured graph, greedy ids):
  d74e052 = fe01b35 off = fe01b35 on = fe01b35 on, file for file, on all 21 dumps (`receipts/s2_dumps.txt`): ar_x5
  and its Q8_0 file, state4 and its Q8_0 file, mor3, mor3_q8_0, mor2p0, ar3, the three recursion models, the five
  synthetic families and the two gemma3 IR variants, gpt2_tiny, Qwen3-0.6B Q8_0 and Pythia-160m. The interpreter's
  device path was already deterministic run to run; the mode changes none of its outputs.

**On against the CPU** (this work's CPU build, `cpu_legs.sh`): pick_e1_L16 replies, stop and ids 360/360. chat_sft_bpe
348/360, the same 12 requests that d74e052's and fe01b35-off's CUDA replies miss the CPU on (`receipts/s2_vs_cpu.json`:
5 first divergences at the same position in every CUDA run, the rest Carry-5 turns whose history then differs): an
existing CUDA-vs-CPU difference of that model, not moved by the mode.

**Cost** (T4, `bench.py`; off and on alternating in separate server processes; 120 greedy decodes of up to 64 tokens,
80 prefills of ~85-tokens prompts with prefix reuse off):

| | decode ms/token p50, off / on | prefill wall ms p50, off / on |
|---|---|---|
| pick_e1_L16 (two pairs) | 1.586, 1.582 / 1.588, 1.586 (on/off 1.002) | 3.71, 3.80 / 3.72, 3.73 (0.993) |
| chat_sft_bpe | 0.747 / 0.746 (1.000) | 1.77 / 1.80 (1.016) |

Within the noise: no measurable cost. Session 1 measured the same (pick 1.573/1.573 and 1.562/1.564 ms/token).

**CPU regression** (WSL, under the CPU budget; `receipts/cpu_regress.txt`): recursion, MoR, MoR refusals, attnres,
state4, regress_parity and G5 third party byte-identical to the committed receipts; the adapter self-test 156/156.
`state_tokens_gate_cpu.json` differs from the committed file in its numpy-reference distances only, and fe01b35's
receipt equals d74e052's byte for byte. regress_bits' old build (69d7ca2) is gone; with d74e052's `ir_logits` as the
old side, 16/16 cases bit-equal.

## What stays nondeterministic

- Across GPUs, architectures, CUDA or cuBLAS versions, or builds: the mode promises run-to-run identity on one GPU and
  one build, not across them. No L4 ran here; on an L4 the mode's claim is run to run on that L4.
- Off mode, by design: the decode rmsnorm keeps its atomics.
- Multi-process or MPI serving (`tinyllama_distributed`) was not looked at.

## Files

- `stage.sh`, `drive.sh`, `vm.sh`: the Colab run (microtorch/AGENTS.md path C; L4 asked for first, then a T4; a
  partial pack before the third-party dumps; the driver stops when the session leaves the listing).
- `bench.py` (cost), `sass_compare.py` (SASS), `vs_cpu.py` (replies against the CPU), `analyze.py` (`cuda_gate.json`).
- `cpu_legs.sh`: the CPU references and `../cuda_state_2026-10-25/cpu_regress.sh` with this work's build.
- `receipts/`: the VMs' logs (`s1_vm.txt`, `s2_vm.txt`), session 2's hardware, binaries' and models' sha256, SASS,
  pair comparisons, benches, dumps; the sm_89 SASS check; the CPU regression. The full VM output (records) is in
  `C:/ml_artifacts/ember_determinism/ember-det2-1026-t4/`; logits dumps, builds and staging copies were deleted.

Run (WSL): `git archive` fe01b35 and d74e052 to `ember_new.tgz`, `ember_head.tgz` with `COMMITS`, then
`bash stage.sh <dir> ~/det/stage && bash drive.sh <session> 110 ~/det/stage`,
`bash cpu_legs.sh <CPU build> <work> <REF>`, and `python3 analyze.py <VM out> <work> cuda_gate.json`.
