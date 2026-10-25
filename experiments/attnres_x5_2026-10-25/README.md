# llama_attnres at the tiny-chat 20m_L16 shape: paperkiln's arm X5 model, and the pin for the X5 post jobs (25-26 Oct 2026)

The model is paperkiln's ar_x5 (`C:/ml_artifacts/attnres_gate_x5/ar_x5/`, family_hint `llama_attnres`, ir_version 1):
d 320, 16 layers, 5 heads, d_ff 864, rope on every head, attention residuals in the full form with scale
`"sources"` (33 depth sources; per-layer affines 2i + 1 and 2i + 2, 33 in the head), `blk.0.attnres_attn_q`
written as zeros, 40 steps on a toy vocabulary (15 ids, 16 positions). Its ops are ar3's
(`../attnres_2026-10-24/`): embedding, list, depth_attend, affine, rmsnorm, llama attention and SwiGLU, append.
paperkiln's numpy IR against `mtstudio score`: worst 2.66e-6, greedy 40/40.

ember **e115fd3** (`e115fd3d23ad5cee1ab7fb3bd0c8bb249ef69555`, ember main and origin/main when tested) runs it on the
CPU, in WebAssembly and on CUDA with no engine change. The L16 control's generation ran at **cbdcb915**
(`cbdcb915d26a66a276919328b8d808a450d51307`); e115fd3 serves that control's own SFT GGUF exactly as cbdcb915 does:
bit for bit on the CPU, and on CUDA within the device's own run-to-run spread with identical replies.

Every sha256 in ar_x5's `manifest.json` matches (7/7: ar_x5.gguf 226fe90b, ar_x5.ir.json 7d9e577a, ar_x5_q8_0.gguf
75779888, v.gguf 879b0dcb, score_in b0f08ef5, score_out 4739183c, spec.json 86fe2dd3). The plain-llama model is
`C:/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf`, sha256 96f6afdc..., the file post_e1_L16_r1's summary records
as served at cbdcb915.

## ar_x5

| check | CPU (e115fd3) | WebAssembly (e115fd3) | CUDA, Colab T4 (e115fd3) |
|---|---|---|---|
| log-probs of the 40 continuations vs `mtstudio score`, worst \|d\| / max(1, \|ref\|) (gate 1e-5) | 5.6e-7 | 5.6e-7 | 6.7e-7 |
| greedy: argmax at every scored position = continuation (= mtstudio's argmax) | 40/40 | 40/40 | 40/40 (40/40) |
| free-running greedy = continuation | | 40/40 | 40/40 |
| decode vs prefill | bit-equal | bit-equal | 2.4e-7 of the range (ROUNDING) |
| against ember's CPU, max \|dlogit\| over the CPU row's range; top-1 | | 9.1e-8; 1419/1419 rows | 1.9e-7; 473/473 |
| ar_x5_q8_0.gguf (33 pseudo-queries F32, the matrices Q8_0): loads; log-probs; greedy; decode = prefill | yes; 7.1e-5; 40/40; bit-equal | | yes; 7.2e-5; 40/40 (free 40/40); 1.8e-7 |
| random ids, 40 x 8 ids, 8 greedy (`ir_wasm_gate`, `ir_device_gate`) | | 40/40, 320/320, 8.4e-8 | 40/40, 320/320, 2.2e-7 |

- CPU (`x5_gate_cpu.json`): `blk.0.attnres_attn_q` is all zeros, and a copy with it set to NaN gives bit-equal logits,
  so it is never read (its depth_attend has one source).
- WebAssembly (`wasm_ar_x5.json`, `wasm_ar_x5_logprobs.json`, `wasm_random_ar_x5.json`): `tools/state_wasm_gate` and
  `tools/ir_wasm_gate` built by `../wasm_state_2026-10-25/build.sh` (Emscripten 6.0.6, `-O3 -msimd128`, 1gb heap),
  compared by that folder's `compare.py`. Every integer result and the greedy ids equal native's; decode = prefill
  bit for bit inside the WebAssembly engine.
- CUDA (`cuda_gate.json` section a; `receipts/s1_*`): `x5_device_logits.cpp` (`tools/ir_logits` on the device, one
  interpreter as a server holds it) for the log-probs; `tools/state_device_gate` (device vs CPU, decode, free-running
  greedy; pass on F32 and Q8_0); `tools/ir_device_gate` on random ids.
- cbdcb915 refuses the file at load ("IR version 1 is not supported (ember reads 0)"). cbdcb915 with ce41ffb's diff
  (the one-line version accept) gives ar_x5 logits bit-equal to e115fd3's on the CPU (F32 and Q8_0) and on the T4
  (`x5_device_logits` outputs identical files), with the same `ir_device_gate` numbers.

## The pin: the post jobs' build

paperkiln's `tiny_chat_run.ensure_ember` builds `cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_ARCHITECTURES=89` and `make -C build -j<nproc> tinyllama_server`. No L4 could be had (paperkiln's
three L4 sessions held the account's slots; a fourth returned TooManyAssignmentsError), so the CUDA legs ran on a
T4 (sm_75) and the pin's exact configuration was compiled there: e115fd3 from a git archive, CUDA toolkit and cuBLAS
found, rc 0 in 172 s on 2 vCPUs, sm_89 cubins only in the binary (`receipts/s1_pin_build.json`,
`s1_cmake_pin.txt`, `s1_make_pin.txt`, `s1_pin_server_elf.txt`; nvcc 13.0.88, driver 580.82.07). It compiles; it
was not run on an sm_89 device here.

## Plain llama: e115fd3 against cbdcb915

The L16 control's SFT GGUF through `tinyllama_server`, the post jobs' path: `EMBER_GEMM_TF32=0`, gpu_layers -1 on
CUDA (0 on the CPU). `x5_plain_collect.py` sends carry5_eval.py's greedy body (temperature 1, top_k 1, top_p 1,
seed 7, return_ids) for 360 requests: the first 40 tinychat prompts (first turn, 48 new tokens) and the spot-check's
40 Carry-5 items (0, 26, ..., 1014) as `carry5_eval.generate` runs them (five self turns with the replies so far as
history, then gold, twin and noctx; 24 new tokens). The server dumps every sampled position's full logit vector
(`TINYLLAMA_LOGITS_DUMP`, nine significant digits, so each value reads back as the same float32). cbdcb915 predates
the nine-digit dump (9238a86), so its tree gets that one line (`x5_dump9.py`); nothing else in it changes.

| | requests: replies + stop, ids, prompt ids equal | first-token logits bit-equal | every sampled position bit-equal |
|---|---|---|---|
| CPU, e115fd3 vs cbdcb915 (`plain_cpu_new_vs_old.json`) | 360/360 each | 360/360 | 4,599/4,599 positions (360/360 requests) |
| CUDA (T4), e115fd3 vs cbdcb915, session 1 | 360/360 each | 40/40 fresh prompts; 148/320 Carry-5 | 0/360 requests; worst first-token \|d\| 7.5e-6 |

The first-token logits of every fresh prompt are bit-equal on the T4, and the decode positions differ in rounding.
The second session (`x5_vm2.sh`; `cuda_gate.json` section c, `receipts/s2_ctl_*.json`) ran each build twice,
interleaved, and compared every pair at every sampled position:

| T4 pair | replies, ids | fresh-prompt first token bit-equal | positions bit-equal (tinychat; Carry-5) | worst \|dlogit\| | top-1 at every position |
|---|---|---|---|---|---|
| e115fd3 vs e115fd3 | 360/360 | 40/40 | 40/658; 160/3941 | 8.6e-6 | 4599/4599 |
| cbdcb915 vs cbdcb915 | 360/360 | 40/40 | 43/658; 153/3941 | 1.3e-5 | 4599/4599 |
| e115fd3 vs cbdcb915 (four pairs) | 360/360 | 40/40 | 41-45/658; 151-162/3941 | 6.4e-6 to 1.0e-5 | 4599/4599 |

So neither build reproduces its own CUDA decode logits bit for bit (the decode reductions use atomics,
`../latency_2026-10-19/`), and the two builds are no further apart than either is from itself (1.0e-5 against 1.3e-5).
Where the device is deterministic (a fresh prompt's prefill) they are bit-equal. On the CPU they are bit-equal
everywhere, so 9238a86's change to the CPU prefill LM head moved no output of this F32 model. Each build's CUDA
replies also equal its CPU replies 360/360. d 320 and head_dim 64 are untouched by the rmsnorm (69d7ca2) and prefill
attention (606e1d4) fixes: rmsnorm launches 256 threads at d 320 before and after, and head_dim 64 is a power of two.

## Files

- CPU: `x5_build_cpu.sh` (e115fd3, cbdcb915 and cbdcb915 + ce41ffb from git archives, on ext4), `x5_cpu_legs.sh`
  (`x5_gate_cpu.py`, the plain-llama runs), `x5_job.sh` (the CPU-budget wrapper, session ember, 2 cores).
- WebAssembly: `x5_wasm.sh`, `x5_inputs.py`, `x5_wasm_logprobs.py`.
- Colab: `x5_stage.sh` / `x5_vm.sh` (session 1, `ember-x5-1025-t4`), `x5_stage2.sh` / `x5_vm2.sh` (session 2,
  `ember-x5ctl-1026-t4`), `x5_drive.sh` (microtorch/AGENTS.md path C: an L4 asked for first, then a T4; the session
  stopped on every exit path; both were stopped), `x5_device_logits.cpp`, `x5_analyze.py` (`cuda_gate.json`).
- Plain llama: `x5_dump9.py`, `x5_plain_collect.py`, `x5_plain_compare.py`.
- Receipts: `x5_gate_cpu.json`, `plain_cpu_new_vs_old.json`, `wasm_*.json`, `cuda_gate.json`; the VMs' records in
  `receipts/` (`s1_`, `s2_`; logs as `.txt`). The logits dumps, builds and staging copies were deleted once the
  comparisons were written.

The archives the scripts read (`ember_new.tgz`, `ember_old.tgz`, `ce41ffb.patch`, `COMMITS`) are made in this folder
with Windows git and deleted after the run:

    git -c core.autocrlf=false archive --format=tar.gz --prefix=ember/ -o ember_new.tgz e115fd3
    git -c core.autocrlf=false archive --format=tar.gz --prefix=ember/ -o ember_old.tgz cbdcb915
    git -c core.autocrlf=false diff ce41ffb^ ce41ffb > ce41ffb.patch

Then, in WSL: `bash x5_job.sh build_cpu x5_build_cpu.sh`, `bash x5_job.sh cpu_legs x5_cpu_legs.sh OUT`,
`bash x5_job.sh wasm x5_wasm.sh OUT`, `bash x5_stage.sh ~/x5g/stage && bash x5_drive.sh ember-x5-1025 110 ~/x5g/stage`,
`bash x5_stage2.sh ~/x5g/stage2 && VM=x5_vm2.sh bash x5_drive.sh ember-x5ctl-1026 70 ~/x5g/stage2`, and
`python3 x5_analyze.py <session 1 out> <session 2 out> ~/x5g/work OUT`.
