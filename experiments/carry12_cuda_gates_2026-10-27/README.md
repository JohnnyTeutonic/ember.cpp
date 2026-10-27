# Carry-12's CUDA pin: state tokens and llama_attnres at ember ec3201b (27 Oct 2026)

Carry-12 pins ember **ec3201b** (`ec3201b04e6f2fcb6dffd3496b0c72b3bec0c54a`) with `EMBER_DETERMINISTIC=1` for its S-gist
(state tokens) and X (llama_attnres) cells. ec3201b made the tiled split-K decode attention (`decode_attn_tiled.cuh`,
`../decode_occupancy_2026-10-27/`) the default, and that change was gated on plain llama only (pick_e1_L16,
pick_e1_60m_L32). Here the earlier state-token and llama_attnres CUDA gates are run again at ec3201b, in the default
mode and under the deterministic mode, with the decode attention at its default and at `EMBER_DECODE_ATTN=split` (the
old kernel). **Every gate passes, and no engine code changed.**

Files (every sha256 in each `manifest.json` matches: state4 12/12, ar3 7/7, ar_x5 7/7):

- paperkiln's **state4** (`C:/ml_artifacts/state_gate/state4`: d 64, 2 layers, 2 heads of 32, k = 4 state tokens,
  `mask: turn_state`, evict turn_state, ctx 128) and state4_q8_0;
- paperkiln's **ar3** (`C:/ml_artifacts/attnres_gate/ar3`: llama_attnres, full form, d 32, 3 layers, 2 heads of 16,
  ctx 16) and ar3_q8_0;
- paperkiln's **ar_x5** (`C:/ml_artifacts/attnres_gate_x5/ar_x5`: llama_attnres at the tiny-chat 20m_L16 shape, d 320,
  16 layers, 5 heads of 64, 33 depth sources, ctx 16) and ar_x5_q8_0.

ember main is **39badc6** (= origin/main when tested). ec3201b..39badc6 changes no CUDA source (`*.cu`, `*.cuh`, their
headers): its engine changes are the CPU interpreter's `attention.relative_kv` (refused on the device by
`device_supported`) and the chat schema v2 paths. 39badc6 was built and run beside the pin: its outputs are byte-equal to
the pin's on every file (below). main has since moved to b1128f5, which changes files under `experiments/` only.

## Does the tiled kernel run on these files?

The interpreter's device attention (`graph_ir.cpp` at ec3201b, lines 3193-3208) sends a `mask: turn_state` node to its
own kernel, `ir_turn_state_attention_cuda`, for prefill and decode alike; any other causal node decodes (T = 1) through
`attention_devpos_cuda` (inside the captured graph) or `attention_decode_at_cuda` (the first step), which take the
tiled kernel by default (`decode_attn_device` in `cuda_kernels.cu`). Measured (nsys `--cuda-graph-trace=node`, every
kernel inside the graphs; `receipts/s1/nsys/`, `receipts/s2/nsys/`):

| file, configuration | attention kernels launched (instances) |
|---|---|
| state4: mode tiled, mode split, default tiled | `turn_state_attention_kernel` 6,736, the same in all three; no decode attention kernel |
| ar3: mode tiled, default tiled | `decode_attn_tiled_kernel` 2,259 + `decode_attn_tiled_merge_kernel` 2,259; `attention_batch_prefill_cuda_kernel` 240 |
| ar3: mode split | `decode_attn_split_kernel` 2,259 + `decode_attn_merge_kernel` 2,259; the same prefill kernel |
| ar_x5: mode tiled, default tiled | `decode_attn_tiled_kernel` 12,048 + merge 12,048; `attention_prefill_flash2_kernel` 1,280 |
| ar_x5: mode split | `decode_attn_split_kernel` 12,048 + merge 12,048; the same prefill kernel |
| ar_x5, long leg (600 positions), mode tiled | `decode_attn_tiled_kernel` 115,072 + merge 115,072 |

So **state4 never takes the tiled path** (its outputs with `EMBER_DECODE_ATTN=split` are byte-equal to the default's in
both modes, every logit of every row), and its gate is the earlier receipts' (`../cuda_state_2026-10-25/`), reproduced
here to the digit. **ar3 and ar_x5 do take it**, in both modes, so for them the comparison with the old kernel is the
gate.

## Results (Colab T4, sm_75, nvcc 13.0.88, driver 580.82.07)

Gate bounds as in the earlier gates: log-probs of the 40 continuations against `mtstudio score` (`score_out.jsonl`),
worst |d| / max(1, max |ref|), 1e-5 on F32 (the Q8_0 files' distance is the quantisation's, reported);
device against the CPU interpreter (the EXACT reference, `tools/state_device_gate` in one process, and
`tools/ir_device_gate` on random ids), max |dlogit| over the CPU row's range within 1e-4, every argmax and greedy count
complete. The four configurations gave the same figures to two digits unless a cell says otherwise.

| check | state4 | state4_q8_0 | ar3 | ar3_q8_0 | ar_x5 | ar_x5_q8_0 |
|---|---|---|---|---|---|---|
| log-probs vs mtstudio | 1.9e-6 | 7.6e-4 (reported) | 2.4e-7 | 1.7e-4 (reported) | 6.7e-7 | 7.2e-5 (reported) |
| greedy at every scored position / free-running greedy | 40/40, 40/40 | 40/40, 40/40 | 40/40, 40/40 | 40/40, 40/40 | 40/40, 40/40 | 40/40, 40/40 |
| `state_device_gate`: top-1 vs CPU; prefill vs CPU; decode vs CPU | 9264/9264; 6.2e-7; 4.4e-7 | 9264/9264; 5.9e-7; 4.4e-7 | 1419/1419; 2.0e-7; 1.8e-7 (split 2.0e-7) | 1419/1419; 2.0e-7; 2.0e-7 | 1419/1419; 1.9e-7; 1.8e-7 | 1419/1419; 1.8e-7; 2.1e-7 (split 1.6e-7) |
| `state_device_gate`: free greedy device / CPU; kv rows held = CPU's; rewinds; session script | 40/40 / 40/40; 40/40; 80/80, 80/80 refused; 12/12 (vs CPU 1.0e-6) | 40/40 / 40/40; 40/40; 80/80, 80/80 refused | 40/40 / 40/40; 40/40 | the same | the same | the same |
| decode vs prefill on the device (graph decode) | 4.3e-7 | 5.2e-7 | 1.6e-7 (split 1.8e-7) | 1.9e-7 | 2.4e-7 | 1.8e-7 |
| `ir_device_gate` random ids: greedy, top-1, prefill, decode | 40/40, 960/960, 7.6e-7, 9.7e-7 | 40/40, 960/960, 9.9e-7, 1.4e-6 | 40/40, 320/320, 1.8e-7, 2.1e-7 | 40/40, 320/320, 1.9e-7, 1.9e-7 | 40/40, 320/320, 2.2e-7, 2.2e-7 | 40/40, 320/320, 2.1e-7, 2.2e-7 |
| `EMBER_DETERMINISTIC=1`, two runs byte-equal (tiled; split) | yes; yes | yes; yes | yes; yes | yes; yes | yes; yes | yes; yes |
| default mode, two runs byte-equal | yes | yes | yes | yes | yes | yes |
| tiled vs split (both modes): outputs byte-equal | **yes** | **yes** | no | no | no | no |
| tiled vs split: argmax at every prefill and decode row; free-running ids | 3088/3088 each; 40/40 | the same | 473/473 each; 40/40 | 473/473 each; 40/40 | 473/473 each; 40/40 | 473/473 each; 40/40 |
| tiled vs split: prefill rows bit-equal; decode rows bit-equal; worst decode \|dlogit\| (over the range) | all; all; 0 | all; all; 0 | 473/473; 42/473; 7.2e-7 (1.6e-7) | 473/473; 48/473; 9.5e-7 (1.8e-7) | 473/473; 97/473; 1.9e-6 (1.7e-7) | 473/473; 100/473; 1.9e-6 (1.8e-7) |
| 39badc6 vs ec3201b under the mode: outputs byte-equal; its `state_device_gate` | yes; pass | yes; pass | yes; pass | yes; pass | yes; pass | yes; pass |

The earlier receipts (aadeed2 / e115fd3, T4): state4 9264/9264, 6.2e-7, decode 4.4e-7, session script 12/12 (1.0e-6),
`ir_device_gate` 40/40, 960/960, 7.6e-7; state4_q8_0 5.9e-7; ar3 1419/1419, 2.0e-7, `ir_device_gate` 320/320, 1.8e-7;
ar_x5 1419/1419, 1.9e-7, log-probs 6.7e-7, Q8_0 7.2e-5 (`cuda_gate.json`'s `earlier` blocks carry them). The
tiled kernel moves the attnres files' decode logits by rounding (worst 1.9e-6 absolute, 1.8e-7 of the row's range),
never an argmax, a greedy id or a prefill bit; under the mode it is as reproducible as the old kernel. The EMBER_IR
control (the unchanged document through `EMBER_IR`) is byte-equal to the embedded one on all four attnres files.

**Long leg.** The files' 16 positions give the tiled kernel one 32-key chunk per head, while Carry-12's X cell decodes
at 245-430 cached tokens of a 1,024-position cache. So ar3 and ar_x5 (and their Q8_0 files) also ran their own documents
with `model.ctx` and the kv state's `max_len` raised to 1,024 and nothing else changed (`c12_inputs.py`; through
`EMBER_IR`), on 8 seeded random sequences of 600 ids (300 free-running), up to 19 chunks per head (second session;
the first session's long runs stopped at load, "IR rope: position 599 exceeds the rope table", because only `model.ctx`
had been raised):

| check (long leg) | ar3 | ar3_q8_0 | ar_x5 | ar_x5_q8_0 |
|---|---|---|---|---|
| `ir_device_gate`, 4 x 600 ids + 64 greedy, device vs CPU (mode tiled, default tiled, mode split): greedy, top-1, prefill, decode | 4/4, 2400/2400, 2.6e-7, 2.7e-7 (split 2.2e-7) | 4/4, 2400/2400, 2.4e-7, 2.5e-7 (split 2.1e-7) | 4/4, 2400/2400, 2.6e-7, 2.4e-7 (split 2.7e-7) | 4/4, 2400/2400, 2.8e-7, 2.7e-7 |
| decode vs prefill (tiled; split) | 2.4e-7; 2.2e-7 | 2.4e-7; 2.0e-7 | 3.7e-7; 3.3e-7 | 3.7e-7; 3.7e-7 |
| mode, two runs byte-equal (tiled; split) | yes; yes | yes; yes | yes; yes | yes; yes |
| tiled vs split (both modes): argmax at every prefill and decode row; free-running ids | 4800/4800 each; 8/8 | the same | the same | the same |
| tiled vs split: prefill rows bit-equal; decode rows bit-equal; worst decode \|dlogit\| (over the range) | 4800/4800; 10/4800; 9.5e-7 (2.4e-7) | 4800/4800; 10/4800; 9.5e-7 (2.4e-7) | 4800/4800; 49/4800; 2.9e-6 (2.4e-7) | 4800/4800; 53/4800; 3.8e-6 (3.6e-7) |
| 39badc6 vs ec3201b under the mode | byte-equal | | byte-equal | |

**Speculative verification** (`EMBER_VERIFY_LINEAR=rows`) does not apply to these files: the verification graph runs
only for the fixed llama path (`have_picks = !ir_ && ...`, `api.cpp` line 1178 at ec3201b); an IR file's speculative
pass is the interpreter's all-rows forward (`ir_->forward(batch, pos, /*all_rows=*/true)`, line 1186), the prefill
kernels, whose rows are the "decode vs prefill" figures above (ROUNDING, at most 5.2e-7).

## Files

- `c12_inputs.py` (the earlier gates' own inputs through `../cuda_state_2026-10-25/make_inputs.py` and
  `../attnres_x5_2026-10-25/x5_inputs.py`, the manifests' sha256, the ctx-1024 documents and the long sequences),
  `c12_stage.sh`, `c12_vm.sh` (the VM: both trees from git archives, `../attnres_x5_2026-10-25/x5_device_logits.cpp`
  as a target beside `tools/state_device_gate` and `tools/ir_device_gate`; `LEGS` = `long` for the second session),
  `c12_drive.sh` (`../tensor_index_gate_2026-10-27/tig_drive.sh` with its upload list from the stage; microtorch/AGENTS.md
  path C: T4 first, the session stopped on every exit path), `c12_analyze.py` (`cuda_gates.json`).
- `cuda_gates.json`: every figure above, per file and configuration, with the nsys kernel counts and the earlier
  receipts' summaries. `receipts/s1/` (session `ember-c12g-1027`, 27 min: everything but the long leg) and
  `receipts/s2/` (`ember-c12g2-1027`, 23 min: the long leg and its nsys run): VM and driver logs, hardware, the
  binaries' and inputs' sha256, every `state_device_gate` and `ir_device_gate` record, the nsys kernel summaries, and
  the sha256 of every logits dump. The VM outputs are in `C:/ml_artifacts/ember_c12_gates/` (13 MB) without the
  logits dumps over 1 MB (1.2 GB in all; state4's are 48 MB each), which were deleted once `cuda_gates.json` was written,
  as were the build trees, the archives and the staging copies.

Rerun (WSL; archives made in Windows with `git -c core.autocrlf=false archive --format=tar.gz --prefix=ember/ -o
<dir>/ember_pin.tgz ec3201b` and the same for main as `ember_head.tgz`, plus `<dir>/COMMITS` "<pin> <head>"):
`bash c12_stage.sh <dir> ~/c12g/stage && bash c12_drive.sh <session> 100 ~/c12g/stage`; for the long leg alone,
`bash c12_stage.sh <dir> ~/c12g/stage2 long && bash c12_drive.sh <session> 60 ~/c12g/stage2`; then
`python3 c12_analyze.py <session 1>/out cuda_gates.json <session 2>/out`.
