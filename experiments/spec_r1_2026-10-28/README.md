# Speculative decoding, what remained (R1): the deterministic pass and draft quality (28 Oct 2026)

Research item R1 (root `LOCAL_LLM_WORK.md`) asked what remains of speculative decoding after the verify pass at
about one decode step (`../verify_2026-10-27`) and the corpus-table drafts (`../draft_2026-10-27`). Measured on a
Colab L4 at ember main (the tiled decode attention of `../decode_occupancy_2026-10-27` included), with the requests
and models of draft_2026-10-27:

- **The default mode** now runs table drafts at **2.35x plain decode on pick_e1_L16 and 2.71x on pk_base_v1**
  over eval100 (k = 8; pick_e1_L16 by set: chat40 1.99x, copy20 2.89x, carry40 2.33x, against 1.91x, 2.42x and 2.14x
  on 27 Oct), ids identical on every request.
- **Under `EMBER_DETERMINISTIC=1` the exact `rows` pass is 4.6-4.7 decode steps at k = 8** (n = 9 rows), 96% of
  it the n cuBLAS GEMV launches per layer; the tiled attention did not move it. Table drafts lose there: **0.90x and
  0.82x** the mode's plain decode.
- **`fixed_linear`** (new, `verify_kernels.cu`) computes every row of the pass in one launch per linear layer with
  each row's arithmetic independent of the row count. With the decode step on it too (`EMBER_DECODE_LINEAR=fixed`),
  the deterministic pass is **1.01-1.15 decode steps**, speculation under the mode stays bit-identical to the mode's
  plain decode (ids and every sampled row's logits, both models, k = 4 and 8, 3829/3829 and 4725/4725 rows), and the
  mode runs table drafts at **2.23x / 2.41x its plain decode** (2.34x / 2.72x main's deterministic plain decode).
  Its decode step is itself 5-8% shorter than cuBLAS's on pick_e1_L16 (the mode's plain decode 1.05x and 1.13x
  main's end to end). It is **opt-in**: the mode's plain logits move from
  cuBLAS's (ids equal on 260/260 requests, max |d| 4.6e-3 / 6.7e-3), and the ROUNDING gate against the CPU reference
  did not run (the second session was reclaimed before that leg); unset, every path is main's, bit for bit.
- **Draft quality**: the cheap variants (orders 2-6, prompt lookup and table mixed by order or by confidence,
  adaptive draft length, theta, k) move the estimate by -18% to +0.3% on the development sets. A table of the
  model's **own replies** to 3,991 user turns of its training dialogue, merged behind the corpus table, adds
  **+5.9% / +2.7%** (pick_e1_L16, chatdev / carrydev, estimate) and grows with the pool; its end-to-end leg is pending
  (GPU). No draft model: see section 4.

Two Colab L4 sessions: `ember-r1` (36 min, every pick_e1_L16 leg) and `ember-r1b` (40 min, reclaimed during
pk_base_v1's accuracy leg; its log holds pk_base_v1's end-to-end and bit results). No RunPod.

## 1. Re-measure at main (L4)

Builds: ember main 934393f (session 1) and 2fbf454 (session 2; no CUDA source or tuned-path file differs between
the two), Release for sm_89, each beside main plus this work (`files.txt`); every one of main's kernels has the same
SASS in the new build (108/108 at 934393f, 110/110 at 2fbf454; `receipts/cuda_s1/sass.json`, `cuda_s2/vm_log.txt`).
Requests (`../draft_2026-10-27/dr_prompts.py`): eval100 = chat40 + copy20 + carry40 (260 records), each sent twice
and the second recorded. Decode tokens per second after the first token, x the same binary and mode's plain decode.

| | pick_e1_L16 | | pk_base_v1 | |
|---|---|---|---|---|
| | tok/s (x plain) | accept / tokens per pass | tok/s (x plain) | accept / tokens per pass |
| plain decode (again at the end) | 744.1 (742.4) | | 380.1 (381.3) | |
| prompt lookup k = 8 | 1097.3 (**1.475**) | 0.298 / 3.18 | 540.8 (**1.423**) | 0.431 / 4.25 |
| table k = 8 | 1748.3 (**2.349**) | 0.264 / 2.02 | 1030.0 (**2.710**) | 0.296 / 2.35 |
| `EMBER_DETERMINISTIC=1`, main: plain | 745.6 | | 383.2 | |
| prompt lookup k = 4 / 8 | 806.0 (1.081) / 712.0 (0.955) | 0.447 / 0.298 | 414.6 (1.082) / 386.9 (1.010) | 0.580 / 0.431 |
| table k = 4 / 8 | 761.0 (1.021) / 672.7 (**0.902**) | 0.322 / 0.264 | 352.4 (0.920) / 314.3 (**0.820**) | 0.349 / 0.296 |

Ids, stop ids and prompt ids equal plain decode's on 260/260 requests in every configuration (`receipts/cuda_s1/e2e/`,
`cuda_s2/vm_log.txt`); the drafted, accepted and pass counts equal `r1_sim.py`'s replay of the CPU records on every
configuration (`receipts/sim/eval100_*.jsonl`).

**The pass, by its parts** (`r1_bench`, pick_e1_L16 d320 L16, a 512-position cache; p50 of 200 calls, host wall
clock; decode step 0.958 ms at a 32-token prefix and 0.973 ms at 384 on cuBLAS, 0.879 / 0.920 ms on `fixed_linear`;
the host sampler 0.22-0.23 ms per plain token on top). In decode steps, n = 1..9 rows, under the mode at P 384:

| linear layers | n = 1 | 2 | 3 | 5 | 9 |
|---|---|---|---|---|---|
| `rows` (main's exact pass: cuBLAS once per row) | 1.01 | 1.46 | 1.92 | 2.83 | **4.70** |
| `fixed` (`fixed_linear`, steps of the fixed decode step) | 1.01 | 1.01 | 1.02 | 1.04 | **1.15** |
| `smallm` (the default mode's pass, steps of the cuBLAS step) | 0.95 | 0.96 | 0.96 | 0.99 | 1.08 |

Kernel time inside the captured graphs (nsys `--cuda-graph-trace=node`, per call, µs; `receipts/cuda_s1/nsys/`):

| P 384 | kernels | linear | attention | rest | wall (bench) |
|---|---|---|---|---|---|
| decode step, cuBLAS | 371 | 680 | 98 | 175 | 973 |
| decode step, `fixed_linear` | 275 | 566 | 143 | 187 | 920 |
| pass `rows`, n = 5 / 9 | 1208 / 2044 | 2515 / 4380 | 126 / 175 | 191 / 192 | 2755 / 4572 |
| pass `fixed`, n = 5 / 9 | 276 / 276 | 593 / 655 | 146 / 179 | 200 / 199 | 958 / 1055 |

(At P 32 the same within 5%; `r1_summary.py nsys`.) cuBLAS's Sgemm with one column runs as `gemvx` plus a split-K
reduce kernel; the `rows` pass launches that pair once per row for each of the seven linear layers per layer and
the head, so its cost grows by about 0.46 steps per row. The tiled attention (all rows in one launch since
verify_2026-10-27) is 175 µs of 4572 here, so it did not shift the `rows` pass (table k = 8 on chat40 under the
mode: 0.871x on 27 Oct, 0.849x now); it did lower the default pass (n = 9 at 384 tokens: 1.335 steps on 27 Oct, 1.08
now), and the default mode's table speed-up rose on chat40 from 1.91x to 1.99x and on carry40 from 2.14x to 2.33x.

## 2. `fixed_linear`: a pass that computes the decode step's bits in one launch

`verify_kernels.cu`: `fixed_linear(X, W, Y, n, O, I)` is the `smallm` kernel (one warp per output, every row of X
at once, the weight read once) with a scalar twin for shapes its float4 loads refuse. Lane l accumulates each row's
dot product over the inputs 4 (l + 32 j) .. +3 in one fma chain and the warp adds the 32 lane sums in a fixed
butterfly, so a row's result depends on that row and W only: not on n, the other rows, the load path or the stream.
`EMBER_DECODE_LINEAR=fixed` runs the decode step's FP32 linear layers (`forward_device`, `forward_device_graph`) on it
with n = 1; under `EMBER_DETERMINISTIC=1` with `EMBER_VERIFY_LINEAR` unset, the pass then takes `fixed` (else `rows`,
main's). The attention rows were already one launch with the decode kernel's per-row arithmetic (the tiled kernel's
block bodies, `attn_tiled_rows_kernel`).

Gates:

- **Bit-exact speculation** (CUDA, L4; `dr_e2e.py --dump`, every sampled row's logits as float32 bit patterns,
  against the mode's plain decode on fixed_linear): prompt lookup k = 8, table k = 4 and k = 8, **3829/3829 rows on
  pick_e1_L16 and 4725/4725 on pk_base_v1**, ids 260/260 (`receipts/cuda_s1/bits/`, `cuda_s2/vm_log.txt`). One pass's
  rows against the decode step at P 32 and 384 (`r1_bench` bits_n9): `fixed` rows = fixed decode 9/9, `rows` =
  cuBLAS decode 9/9, across the two 0/9 (|d| 2.9e-6 to 4.8e-6).
- **Run to run** under the mode with fixed_linear: plain decode twice, 3829/3829 and 4725/4725 rows bit-identical.
- **Off, unchanged**: with `EMBER_DECODE_LINEAR` unset the code paths are main's (cuBLAS; the mode's pass `rows`);
  `EMBER_DECODE_LINEAR=cublas` under the mode gives main's deterministic plain decode bit for bit (3829/3829, 4725/4725);
  default-mode ids equal main's 260/260 on both models; main's kernels' SASS identical (16 functions added).
- **On the CPU** (`cpu_checks.sh`, `receipts/cpu/`): `r1_fixed_order_check.cpp` replays both kernels' statement order
  with `std::fmaf` and float adds on the host, for the picks' widths (320, 768, 864, 1280, 2048) and widths with
  I % 4 != 0, n = 1..16, the row first and last in the pass: **3328/3328 bit-equal** to the row alone (n = 1), and a
  single sequential fma chain differs from it in 62 of 64 controls, so the comparison sees order changes. The
  replay checks the order, not the GPU; the device measurements are the gates above. The CUDA tree compiles (sm_75
  check in WSL, `build.sh cuda`, at the committed state).
- **Not run (GPU leg pending)**: the mode's rows on fixed_linear against the CPU reference. They differ from the
  cuBLAS mode's by at most 4.6e-3 (pick_e1_L16) and 6.7e-3 (pk_base_v1) over the dumped rows, with ids equal on every
  request; a single pass row differs by under 5e-6, so the larger figure is what one generation accumulates
  somewhere in the 3829 / 4725 rows (the dumps stayed on the VMs). Until the CPU comparison places both modes within
  the ROUNDING tolerance, `fixed` stays opt-in.

In both sessions fixed_linear was the mode's default in the new build; the scripts now set
`EMBER_DECODE_LINEAR=fixed` for those runs (`DN` in `vm.sh`, `vm2.sh`). Speed with it (eval100, x the mode's plain
decode on fixed_linear, and x main's deterministic plain decode):

| `EMBER_DETERMINISTIC=1 EMBER_DECODE_LINEAR=fixed` | pick_e1_L16 | pk_base_v1 |
|---|---|---|
| plain decode (x main's) | 783.9 (1.051) | 432.7 (1.129) |
| prompt lookup k = 4 | 1122.2 (1.432; 1.505) | 592.1 (1.368; 1.545) |
| prompt lookup k = 8 | 1136.3 (1.450; 1.524) | 607.8 (1.405; 1.586) |
| table k = 4 | 1711.4 (2.183; 2.295) | 990.7 (2.290; 2.586) |
| table k = 8 | 1746.7 (**2.228**; 2.343) | 1040.8 (**2.405**; 2.716) |

## 3. Draft quality (`r1_sim.py`, development sets)

Greedy speculation is exact, so passes, drafts and acceptances follow from plain decode's ids; `r1_sim.py` reuses
draft_2026-10-27's replay and reproduces ember's counts on all 16 L4 configurations above. Estimates use session 1's
L4 costs (`receipts/sim/costs_pick_*_P384.json`: a plain step = decode step + host sampler, a pass of n rows its
bench time); on eval100 they come within 4% of the measured speed-ups (2.416 / 2.349 for table k = 8 default, 2.311 /
2.228 under the mode with fixed_linear). Settings are judged on chatdev and carrydev only (`receipts/sim/dev_*`).
pick_e1_L16, x plain decode (current table: 2.490 chatdev, 2.302 carrydev):

| change (k = 8 unless stated) | chatdev | carrydev |
|---|---|---|
| theta 0.01 / 0.05 (0.02 now) | 2.514 / 2.422 | 2.304 / 2.263 |
| k = 15 | 2.427 | 2.144 |
| orders 1-2 / 1-3 (1-4 now) | 2.199 / 2.360 | 2.125 / 2.250 |
| orders 1-5 / 1-6 (corpus re-counted, min count 8) | 2.484 / 2.456 | 2.294 / 2.288 |
| table first, prompt lookup where it has nothing | 2.144 | 2.017 |
| prompt lookup's draft continued by the table | 2.490 | 2.299 |
| table first when its share is at least 0.3 / 0.5 / 0.7 | 2.450 / 2.482 / 2.487 | 2.226 / 2.289 / 2.293 |
| draft length capped by the running acceptance (c = 1, 2, 4) | 2.042 / 2.184 / 2.289 | 2.022 / 2.072 / 2.149 |
| self-output table alone, 1,000 / 2,000 / 3,991 requests | 2.278 / 2.349 / 2.470 | 1.992 / 2.073 / 2.171 |
| **corpus table, then self-output (merged by order)**, 1,000 / 2,000 / 3,991 | 2.586 / 2.616 / **2.636** | 2.318 / 2.329 / **2.364** |

pk_base_v1 (its table, the earlier unit pass costs): orders 1-5 +2.0% chatdev and +0.3% carrydev, the rest as
pick_e1_L16's. A pass costs about one step at any n up to 9, so the lever is the first draft token's acceptance:
shorter or more selective drafts only lose tokens, and longer contexts in a 40M-token corpus add little.

**Self-output tables** (`r1_pool.py`, `r1_sim.py build --pool`, `merge`): the model's own greedy replies (60 new
tokens) to user turns of the dialogue it was trained on (pick_e1_L16: 3,991 requests over its 13 human sources,
session 1, 92,656 tokens in 64 s on the L4 with speculation on), counted at orders 1-6 over contexts seen at least
twice (shares of at least 0.10), and merged behind the corpus table at each order (`chain: order`). From 92k tokens
alone it nearly matches the 40M-token corpus table on chatdev, and the merged table gains 5.9% / 2.7%, still rising
at each doubling of the pool. `draft_table.h` now reads orders 1-8 (the order-4 hash continued; files of orders 1-4
unchanged), and its drafts equal `r1_sim.py`'s at every one of 5,015 development positions on the merged order-6 and
order-4 tables and 6,828 evaluation positions on the two corpus tables (`receipts/cpu/table_check.jsonl`). The
end-to-end leg with a 20,000-request pool (staged as `vm2.sh` (e)) did not run.

## 4. No draft model

At these sizes a decode step is latency-bound: 275-371 kernels for 0.9-1.0 ms on pick_e1_L16, of which the weights'
bandwidth is a small part. A draft model's own step runs the same chain of launches, so even a 1-2M-parameter
drafter would cost a large fraction of a target step per drafted token, against microseconds for the table (2-8 µs
per position), while the pass already costs about one step at any draft length. The table's acceptance (0.26-0.30)
leaves the gap; a drafter would have to accept far more often to pay for its own steps, and nothing at these sizes
suggests it can.

## Files

- `../../verify_kernels.cu`, `verify_kernels.h` (`fixed_linear`, `VerifyLinear::kFixed`), `../../model.cpp`,
  `model.h` (`decode_matvec`, `decode_linear_fixed`), `../../speculative_verify.cpp` (`EMBER_DECODE_LINEAR`, the mode's
  pass choice), `../../draft_table.h` (orders 5-8), `docs/USAGE.md`.
- `r1_bench.cpp` (built as `tools/r1_bench` by `build.sh`), `r1_summary.py`, `r1_sim.py`, `r1_pool.py`,
  `r1_rows.py`, `r1_table_check.cpp` / `.py`, `r1_fixed_order_check.cpp`, `cpu_checks.sh`, `r1_receipts.sh`.
- `build.sh`, `stage.sh`, `drive.sh`, `waitlaunch.sh`, `vm.sh` (session 1), `vm2.sh` (session 2: pk_base_v1's legs,
  the accuracy leg and the self-output tables; reclaimed after pk_base_v1's dumps), `r1_corpus.sh`.
- `receipts/cuda_s1/`: session 1's hardware, binaries, SASS, benches, nsys tables, end-to-end meta and comparisons,
  dump comparisons, the pool's meta, VM log; `receipts/cuda_s2/vm_log.txt`: session 2's log (pk_base_v1's end-to-end
  summaries, comparisons and dump comparisons); `receipts/cpu/`; `receipts/sim/`. Full outputs:
  `C:/ml_artifacts/ember_spec_r1/` (the pool's records in `ember-r1/out/pool/`).

Run (WSL): `bash build.sh cuda`; `bash stage.sh <ember_head.tgz> <commit> <stage> [vm2.sh]`; `bash waitlaunch.sh
<session> 115 <stage> <log>`; `python3 r1_summary.py bench|nsys|e2e|bits <out>`; `bash cpu_checks.sh`.

## E10.3 implementation started, 30 Oct 2026

`EMBER_DECODE_PICK=1` opts plain greedy decoding into a captured `select_rows` after the unchanged
LM-head kernels. The host reads one 20-byte RowPick. Greedy sampling uses the existing speculative
record decoder, including its RNG draw. Temperature/top-k modes needing a distribution, either
repetition penalty, and unsupported graph paths keep the existing full-row route. Near ties, NaNs,
infinities and diagnostic dumps read the already-computed logits without replaying the forward.
Switching between graph output modes recaptures; normal decoding remains the default.

Validation: g++ C++17 syntax checks of model.cpp and api.cpp both with and without HAS_CUDA
(CUDA 12.6 headers); `test_decode_pick.py` compares the production full sampler and record decoder
on random, tied and near-tied rows, vocabulary sizes through 16,384, seven temperatures and three
top-p settings: 19,830 accepted decisions have equal ids and RNG states; 5,370 conservatively
fall back. Invalid records are rejected without advancing RNG. This is a host-side differential
test, not a device kernel gate. No local GPU used.

Still required on the remote device: compare option off/on ids and every dumped logit row
bit-for-bit (the option with dumping fetches its selected graph's full row), compare ids without
dumping, run both default and deterministic modes, check graph fallback and penalties, and
measure at least 15% plain decode throughput gain on pick_e1_L16 before adopting the option.
No CUDA execution, speed claim or default change is established by the host checks.

### E10.3 receipt checker prepared, 30 Oct 2026

`decode_pick_gate.py` prepares and checks the next supervised remote campaign; it never starts
a server or GPU job itself. On the remote Linux host, generate a fresh plan with:

```sh
python3 decode_pick_gate.py plan /absolute/tinyllama_server /absolute/pick_e1_L16.gguf /absolute/eval100.jsonl /absolute/new-receipts
```

It prints 14 sequential `dr_e2e.py` commands, seven each in default and deterministic modes:
full-row option off, off repeated, on; then ABBA timing off/on/on/off without logit dumps.
Run them only under the campaign supervisor. No new driver or campaign is installed here.
The model, server and prompt hashes are frozen in `plan.json`; retain those inputs for checking.
The output directory must be new. The same frozen prompts are used in every leg; multiple-turn
prompts retain their ordinary generated histories.

`python3 decode_pick_gate.py analyze /absolute/new-receipts` checks every expected request,
warm-pass stability, server model/mode, explicit environment, lack of speculation, exact dump
lengths/offsets, full-row bits and compact-path token ids. Missing timing and empty/truncated
receipts fail closed. Both ABBA pairs must individually meet 1.15x in each mode for its
conservative throughput check. It compares raw dump bytes with SHA256 rather than parsing
float equality, so signed zeros and NaN payloads are not silently equated.

The extra off/off dump control matters: default CUDA arithmetic is not generally deterministic
(`determinism.h`). If its repeated baseline rows already differ, the bit gate is reported
`BASELINE_UNSTABLE`, not attributed to the new option. No relaxed numerical tolerance is substituted.
`PASS` covers only these measured rows/ids/timing checks; graph-refusal and penalty device checks
remain separate, and no default adoption follows automatically. These offline tests do not establish device execution.

`python3 test_decode_pick_gate.py`: 14 offline fixture regressions passed, including deliberately
corrupted rows, timing, ids, hashes, modes and baseline control. Ruff passed for both new scripts.

### Bounded priority-model device campaign

`decode_pick_vm.sh` packages the priority `pick_e1_L16` gate for the command-only supervised
RunPod boot. It first runs `decode_pick_rows`, which calls the actual CUDA selection kernel on
ordinary maxima, exact/near ties, mixed NaNs and infinities. `decode_pick_fallback.py` then checks
48 direct API requests: option off/on with graph enabled/refused, repetition penalty, top-k 4,
zero temperature and temperature 0.049, comparing reply/ids/prompt ids/stop/token count exactly.
The ordinary 100-prompt, 260-request row/ID/ABBA matrix follows. Five offline fallback regressions
and the fourteen existing receipt-checker regressions pass; the CUDA row harness passes a host
syntax check. These checks alone do not establish a device result.

The campaign preserves partial archives every two minutes. A complete valid `NO_GAIN` or
`BASELINE_UNSTABLE` result is an operationally completed job with a negative scientific outcome;
an incomplete receipt or failed CUDA/fallback test remains a failure. The fixed-order default,
self-output draft tables, secondary model and default adoption are outside this bounded campaign.

The 30 October L4 campaign has preliminary device evidence in `receipts/decode_pick_20261030_*.json`.
All six handcrafted CUDA cases and all 48 fallback requests passed. The frozen file has 100 items
(40 chat, 20 copy, 40 five-turn Carry conversations), yielding 260 actual requests per leg. All seven
default-mode legs have equal reply/ids/prompt ids/stop/token count, but the repeated option-off baseline's
row bytes differ: the default bit comparison is inconclusive. Deterministic off/repeat/on full row bytes
and ids agree exactly. Its four ABBA timing legs remain pending; these preliminary receipts have
`complete=false` and do not close E10.3 or establish default adoption.

Before the original driver deadline, only the orchestration parent was paused while its current
`dr_e2e.py` child finished and independently wrote valid metadata. This held all four deterministic
ABBA legs for one replacement L4, avoiding a pair split across pods. The runner then terminated its
own pod normally at 02:57 UTC, retaining all ten completed legs and the volume under the same $2
cumulative cap. No completed scientific leg is retried by the verified-resume wrapper. Full raw dumps
remain in the supervised campaign's volume until its canonical archive is home and checked.

The final 1,370,269,388-byte archive is now home and independently checked (streaming SHA256,
ZIP CRC, temporary extraction, full strict row/ID/timing analysis; one CPU budget slot). Final
receipt: `receipts/decode_pick_20261030_final.json`. Its classification is **BASELINE_UNSTABLE**:
default off/off row bits differ. Deterministic rows and IDs agree, but the paired throughput
ratios are 1.15924656 and 1.14146241; the latter misses the unchanged 1.15 threshold. The option
therefore remains experimental and the default is unchanged. The executable was not an archived
output: its SHA is consistent across the plan, binaries list and fallback receipt; local verification
recomputed model/prompt hashes and all raw row/ID/timing checks, not the absent executable's bytes.
Temporary extracted files were removed; the ZIP remains the canonical raw artifact outside OneDrive.

### Complete fixed-linear reference gate prepared

`fixed_reference_gate.py` plans six full-dump legs: CPU, deterministic cuBLAS and deterministic fixed
for each original GGUF, over all 260 frozen requests. Unlike the exploratory `r1_rows.py`, it never
drops requests with differing IDs. Complete inventories, modes/backends, warm-pass identity, frozen
inputs, row offsets/lengths and finite values are required. On every CPU-reference row, fixed must
meet ROUNDING (`max |difference| / max(1, max |CPU row|) <= 1e-4`) and have maximum absolute error no
larger than cuBLAS's. All per-row errors and their request/turn/position are retained. Fifteen offline
regressions pass; the bounded `fixed_reference_vm.sh` campaign uses exact original model hashes and
includes its executable in the final artifact. No engine default changes merely because preparation passes.

### Resumable self-output pool client prepared

`self_output_pool.py run BIN GGUF POOL.jsonl CORPUS_TABLE.bin OUT --max-new-requests 100` is the bounded
pilot form, to be run only through the supervised remote campaign. Repeating it on the same OUT without
the cap resumes the same frozen pool. Each completed response is appended and fsynced, with request
hash, source/binary/model/table/pool/environment provenance and a hash chain. Completed requests are
never re-sampled; only an interrupted final append can be truncated. A complete invalid response creates
HALTED and preserves its diagnostic. Manifests and summaries are atomic; final summaries are immutable.
The canonical journal is directly readable by the existing `r1_sim.py build --pool` recipe, avoiding
a second response corpus. The unchanged larger-pool recipe produces 19,994 pick requests (20,000 target
floored across thirteen sources) and 8,000 pk requests; these are not shortened evaluation subsets.

`self_output_pool.py analyze OUT [--input-map local-inputs.json]` independently audits partial or complete
journals without launching a server. Exact-hash local file mappings allow checking relayed artifacts;
wrong source, model, request, chain, environment, health or summary data are rejected. The exact
standalone client/auditor source is bundled under OUT/source for future checks after a checkout changes.
Twenty-one offline
regressions pass, including resume without repeat sampling, torn writes, corruption, HALTED preservation
and read-only relay checks. This is preparation, not a completed self-output experiment or speed claim.

The bounded `self_output_vm.sh` wrapper now prepares both original model pools, audits completion,
builds candidates with `self_output_candidate.py`, and evaluates them using `self_output_gate.py`.
The frozen builder uses six orders, minimum count 2, confidence 0.10 and corpus-first priority;
its recipe hashes the journal, model, tables and builder sources. An empty higher-order table is
valid and remains empty. Completed candidates reject source/input/output drift.

The evaluation retains every frozen request and checks exact IDs against plain decode. Two paired
corpus/merged timings must each improve chat40 and carry40 by at least 3%, in both explicitly selected
default-cuBLAS and deterministic-fixed modes. Copy20 still participates in the ID gate. Pick also
has complete deterministic row dumps and a repeated plain control. Nonfinite rows fail; unstable
controls and cross-device timing are classified explicitly. Complete negative or invalid legs are
never retried to obtain a pass. Forty-two client, candidate and evaluation regressions pass together.
The acceptance checks remain unchanged; the complete device outcome is recorded below. No default is changed.

### Fixed-linear reference gate: complete negative result

The complete six-leg L4/CPU matrix and 1,534,723,113-byte canonical archive have been independently
verified: streaming SHA256, ZIP CRC, actual archived executable and original input bytes, every
request/row, and exact agreement with the remote report. The final receipt is
`receipts/fixed_reference_20261030_final.json`; temporary extracted files were removed.

Both models meet ROUNDING on every row, with worst normalized fixed errors 1.08535e-6 (pick) and
7.61929e-7 (pk), against 1e-4. All 260 request identities align for each model. The stricter requirement
that fixed be no farther from CPU than cuBLAS fails on 1,223/3,829 pick rows and 1,105/4,725 pk rows.
The largest extra absolute errors are 4.05312e-6 and 4.91738e-6 respectively. Overall **FAIL** is a valid
negative result under the unchanged acceptance criterion. Fixed remains opt-in; no default was changed,
no tolerance was relaxed, and no completed scientific leg was retried.

### 30 October self-output pool operational failure

The bounded E10.2 device run stopped before its first generation request: the client required an explicit `deterministic: false` health field, whereas the server omits that field when off. No self-output corpus, candidate table, parity result or speed result was produced. At that point E10.2 remained open. The final archive passed streaming SHA256, ZIP CRC, archived executable and client-source hash checks; see `receipts/self_output_20261030_operational_failure.json`. All campaign resources were removed after outputs arrived home; displayed cumulative cost was $1.67. Commit `f8837f0` fixes the client schema assumption with 53 passing pool/gate/candidate tests, but was not deployed to this completed campaign. Defaults and scientific gates remain unchanged.


### Full self-output matrix: complete negative result

The complete original pools contain 19,994 pick responses and 8,000 pk responses. The immutable
100-response pilot prefixes were reused after exact manifest/hash checks; no completed request
was resampled. Both candidates use the frozen six-order, minimum-count-2, confidence-0.10,
corpus-first recipe. All 23 L4 evaluation legs completed with the original 260-request inventory
per leg. No gate, output budget, dataset or model was changed to obtain a result.

All evaluated IDs match plain decode in both modes. Pick's three deterministic files each contain
3,829 finite full-vocabulary rows (250,937,344 bytes) and are bit-identical, including the repeated
plain control. PK's frozen protocol has ten timing/ID legs and no row-dump requirement.

Paired corpus-to-merged throughput ratios (both pairs must reach 1.03x for each dataset and mode):

| Model | Mode | chat40 pairs | carry40 pairs | Throughput gate |
|---|---|---|---|---|
| pick | default | 1.0943x / 1.1089x | 1.0302x / 1.0454x | PASS |
| pick | det | 1.0690x / 1.0781x | 0.9763x / 1.0154x | NO_GAIN |
| pk | default | 1.0049x / 1.0088x | 1.0158x / 1.0081x | NO_GAIN |
| pk | det | 1.0002x / 1.0027x | 1.0178x / 1.0140x | NO_GAIN |

Both model verdicts are **NO_GAIN**, so the full result is **NOT_ADOPTED**. Pick's default-mode
subset passes the speed criterion, while deterministic carry40 does not. PK does not reach 3%
in either mode. The positive subset does not replace the registered whole-matrix criterion;
no default was changed and no completed scientific leg was retried.

The independent local audit streamed the archive SHA256 and ZIP CRC, verified the exact archived
executable, uploaded input manifest and original controller sources, replayed both complete raw
journals, rebuilt both self.npz and merged.bin candidates with exact hash agreement, and replayed
every ID, finite row, repeated control and paired timing through the archived unchanged analyzer.
The recomputed reports match the saved reports exactly under CPython 3.11 / NumPy 1.26.4.
See [final receipt](receipts/self_output_full_v3_20261030_final.json).

The canonical raw archive remains outside Git (728,426,832 bytes), SHA256
`3b8b8ff30049a1181aa68bc65d1680d994797794fea2aee2fd2a43737c18a633`.
Pilot plus full-continuation ledger cost was $1.219785 within their $2 envelope;
the prior failed v1 campaign is preserved separately. The owned pod and volume were removed
after outputs arrived home.

The running VM's initial ZIP checkpoint failure came from reproducible pre-1980 extracted file
mtimes. The approved repair recorded every affected original timestamp, size and SHA256, changed
only extracted timestamp metadata, and verified unchanged bytes plus a successful ZIP/CRC.
The frozen scientific script, input archive, models, controllers and gates were unchanged.
[Repair receipt](receipts/self_output_full_v3_timestamp_repair.json) preserves that history;
the future packer clamps ZIP metadata without altering raw timestamps, with an exact-code regression.

### Row-selection component follow-up, 31 October 2026

The measured avoidable cost is the existing selector's single 256-thread block and two
vocabulary scans. A **standalone experimental tool**, not a production kernel change, divides
that identical local reduction into 4, 16 or 64 partitions and merges the complete RowPick:
maximum, largest strictly smaller value, first/last maximum indices and NaN flag.

One supervised NVIDIA L4/SM89 run is fully home. The independent audit verifies the frozen
manifest, original kernel extraction, source/binary/generated-header hashes, raw trial identities,
units and GPU UUID. The original-only leg passed 2,268 CPU-oracle row checks before its timings;
the separate candidate leg passed **6,804 bitwise field comparisons** against the original before
any candidate timing. The two complete legs have identical generated-input hashes. The unchanged
host sampler differential also passed 19,830 accepted decisions and 5,370 conservative fallbacks,
including matching RNG state. No parity rule or signed-zero comparison was relaxed.

For the registered pick's actual vocabulary **16,384**, one row, ten-trial medians in microseconds:

| Selector | CUDA-event launch loop | Captured selector + 8-byte H2D + 20-byte D2H + stream synchronisation |
|---|---:|---:|
| Original, independent baseline leg | 16.15 | 24.70 |
| Original, candidate comparison leg | 16.12 | 24.64 |
| 4 partitions + merge | 7.97 | 16.39 |
| 16 partitions + merge | 5.95 | 16.41 |
| 64 partitions + merge | 8.44 | 16.41 |

The independent transfer-only baseline was 6.54 microseconds. These results identify about
**8.2 microseconds of component wall time** that may be avoidable for this shape. They are one
GPU's descriptive timings, not a significance test, full-model measurement or proof that one
partition count is best. Larger vocabularies and batches are retained in the raw results; they
are future geometries, not evidence about this 16,384-token pick. Host wall timing covers only
one row; batches four and sixteen have kernel timing only. Each kernel trial records 1,000
iterations and raw elapsed milliseconds; numerically that total is microseconds per iteration. The
CUDA-event interval includes gaps between host-submitted launches; it is not an isolated
kernel-body latency measurement. The captured host-wall measurements are reported separately.

The actual matrix includes seeded random rows, constant finite rows, all-negative-infinity and
all-NaN rows, positive infinities, an inserted NaN, both signed-zero layouts, separated equal
maxima and a guaranteed peak/nextafter runner-up. Two additional fixtures insert 0/-1e-5 or
positive/negative subnormal values among random values: those inserted values are **not guaranteed
to be the maximum/runner-up**. This is not a guaranteed precise threshold/subnormal-maximum gate
or a dedicated all-finite-negative fixture. The existing host sampler tests supplement this
component matrix; no new device rerun was used to expand coverage after seeing results.

For scale only, the retained historical deterministic pair b would need 7.992 microseconds less
per decode token to move from its recorded 1.14146x to 1.15x. That is comparable to this synthetic
component saving, with little margin; it does **not** predict an end-to-end gain across campaigns.
The original default-mode off/off instability is also unaffected. The unchanged full-model
repeatability and both-pair 15% gates remain unmet: **no default or production kernel changed**,
and no further paid full-model rerun was launched. A further full-model gate would require
separately reviewed production integration and a concrete case for sufficient overall margin,
together with addressing the existing default-control instability; this component result alone
does not justify an unchanged paid rerun.

Frozen sources: Ember `3ae7f47f4ed2d735333461eaa6e0be4212ce94ad`, command job
paperkiln `5de89be1d27db02f23c06213a802f20f5d6b3b2d`. The 756,113-byte canonical ZIP remains at
`C:/ml_artifacts/e10_component_20261031/e10_component_20261031/out.zip`; all public raw trials,
receipts, descriptive summaries and home proof are in
[`receipts/row_pick_component_20261031/`](receipts/row_pick_component_20261031/audit.json).
The pod and volume were independently absent from the provider after normal cleanup; rounded
ledger spend was **$0.07**, within the $1 cap. Newly generated input scratch was hashed and removed
before relay; exact generation is pinned in the archived source. The ZIP retains source and binaries.

Replay the independent audit without GPU or network calls:

```sh
python3 tools/row_pick_profile/audit.py /absolute/out.zip \
  --stage /absolute/original-frozen-stage --output /absolute/new-audit.json
```

The stage supplies the prelaunch `e10_component_manifest.json`; the public copy beside the raw
receipts has the same bytes. The output path must be new. Component tests and the independent
audit refusal tests are separate from the frozen GPU experiment.

### Experimental four-part selector integration, 31 October 2026

The standalone component candidate now has a separate, explicitly experimental integration:
`EMBER_DECODE_PICK=1 EMBER_DECODE_PICK_PARTS=4 EMBER_DETERMINISTIC=1`. Unset or `PARTS=0`
keeps the existing selector. A request that activates four partitions outside deterministic mode
throws an explicit configuration error; the integration never enables deterministic arithmetic
silently. This restriction leaves the historical default-mode instability unresolved.

`row_pick_partitioned.cuh` preserves the component experiment's partition and merge function bodies.
The model allocates four partial records before graph capture, retains them for the graph's lifetime,
and includes the partition mode in the capture cache key alongside KV identity and pick presence.
Its destruction synchronises the stream and destroys the graph before freeing those records.
The original selector and speculative-verification callers are unchanged. Only the new four-part
plain-decode route defers its redundant host embedding lookup. Fixed diagnostic positions still
fetch the current token, as does graph fallback; repeated requests for that token reuse the one
lookup. A near-tie or dump reads the already computed logits without another forward/KV write.

The bounded CPU and compile-only integration milestone is complete:

- 54 targeted tests pass, including 12 integration cases for strict option refusal, cache-mode
  transitions, allocation/destruction order, lazy lookup and actual generation-dispatch fragments
  with mock transport. The merge-body CPU checks include guaranteed negative, threshold and
  subnormal peak fixtures; these are not additional device checks or signed-zero bit proofs.
- The unchanged production sampler differential passes 19,830 compact decisions and 5,370
  conservative fallbacks with equal ids and RNG state. Ruff and both CUDA-host/CPU syntax checks pass.
- A fresh CPU build passes all three registered CTests, the native numerical and state-template
  checks, and Keramos conformance (554 PASS, no FAIL, 44 SKIP).
- Six actual CPU HTTP requests use the registered pick in the baseline binary, candidate with the
  option off and candidate with the experimental flags. Outputs match, and each run's 21 finite
  16,384-value logit rows have identical float32 bytes. This validates the CPU path, not CUDA dispatch.
- CUDA 12.6 compile-only SM89 disassembly preserves all 42 pre-existing verification kernel bodies,
  with exactly the partition and merge kernels added. Only compiler-generated anonymous namespace
  identifiers are normalised; instructions, addresses and encodings are compared unchanged.

The cross-toolchain comparison is deliberately retained as negative evidence: the new kernels'
CUDA 12.6 SASS is **not** identical to the archived component's CUDA 12.4 SASS, despite exact source
bodies. Its earlier device field checks therefore do not establish integrated-binary field parity.
The raw compiler objects, disassembly and CPU row dumps remain in the receipt's external artifact
home; source identities, logs, actual HTTP outputs and this limitation are in
[`partition_integration_20261031/validation.json`](receipts/partition_integration_20261031/validation.json).
An independent source review found no production blocker; unrelated Unicode comment changes found
in that review were restored before the final source-bound build and checks.

**No new GPU execution, paid calls, speed result, adoption or default change occurred.** The retained
full-model gate still has unstable default repeated rows and deterministic paired speedups 1.15925x /
1.14146x against the unchanged 1.15x requirement. A deterministic-only integration diagnostic could
investigate this new route, but cannot replace the original default-plus-deterministic adoption
matrix or close E10. No such diagnostic or paid rerun is queued by this change.

## Prepared synthetic integrated selector correctness packet (31 October 2026)

`integrated_gate_20261031/protocol.json` freezes a correctness-only L4 packet for production source `4836b7d`. It builds the unchanged production core and a diagnostic translation unit; no production flags or arithmetic are edited. A deterministic synthetic F32 Llama (16,384 vocabulary, two 32-wide layers, GQA 4/2) is generated from source, avoiding a registered-model transfer. The packet compares full raw rows and production sampler IDs/RNG states across no-pick, original selector, and explicit four-part selector; it also checks 96 adversarial production-kernel fixtures, pick-presence graph recapture/persistent scratch, and the existing HTTP fallback matrix.

Preparation passes 31 independent replay/preservation negatives, 12 existing source-bound CPU integration checks, a CUDA-header compile-only check, fresh CUDA CMake configuration, and actual CPU fixture load/forward. See `integrated_gate_20261031/preflight.json`. This is prepared evidence, **not an actual GPU pass**. Parent review of the exact source archive and runner job is required before any paid launch. Raw model rows and fields will remain in one canonical archive; the checker rejects shortened/substituted inventories and CPU/IR fallback.

This packet does not validate the registered `pick_e1_L16`, measure speed, change defaults, or close adoption. The historical default `BASELINE_UNSTABLE` result and deterministic speed pairs (1.15925x and 1.14146x, requiring both at least 1.15x) remain explicit failures of the full adoption requirement. Bounded synthetic correctness, if established, will be reported separately.

The first integrated attempt stopped before model-row forwards because the harness set `EMBER_IR=0`; production interprets that variable as an override pathname. The immutable V1 archive and exact failure are bound in `integrated_gate_20261031/environment_v2_repair.json`. V2 clears the variable in both controller and C++ probe. Actual CPU constructor controls reproduce the old error and accept the corrected environment; model/kernel/row/sampler criteria, all production code, and historical adoption failures are unchanged. V2 remains unlaunched pending parent review.

## Integrated synthetic correctness result, 31 October 2026

The separately bounded V2 L4 gate and independent raw-archive replay pass. See receipts/integrated_correctness_20261031/README.md for exact scope, retained V1 setup failure, complete counts and the distinction between compact public evidence and the canonical local raw archive. This closes the synthetic integrated correctness execution; registered-model speed, default-control stability and adoption remain open.
## Historical default repeatability: diagnosed, not repaired (1 November 2026)

The registered E10 off/off control failure is consistent with documented default-path nondeterminism, rather than an unexplained four-part-selector defect. `receipts/default_repeatability_20261101.json` binds an independent streamed replay of the original archive to its exact source and current public source `106fb0c`. The tuned decode RMSNorm kernel and its wrapper are body-identical between those revisions: eight warp leaders use unordered floating-point `atomicAdd`; deterministic mode selects the existing ordered kernel. The receipt includes a realizable positive-partial binary32 counterexample. This illustrates the mechanism, not a captured CUDA schedule or unique attribution of every historical logit difference.

On the actual registered off/off data, all 260 request ID/prompt/stop mappings and all 3,829 top-1 choices match, but only 101 full rows are bit-equal. All 60 turn-zero first rows match; 40 of 200 history-turn first rows match. The worst absolute difference is 0.0038707256, so this L4 result should not be described as merely a few ULPs. Raw row hashes reproduce the original receipt. Raw rows were streamed without writing extra copies or changing them.

The frozen bit-exact default criterion remains a valid strict adoption requirement and remains unmet. It cannot isolate a selector-induced change while unchanged off/off forwards already differ. The original `BASELINE_UNSTABLE` verdict and failed deterministic paired 1.15x speed requirement are preserved. Repeated attempts until scheduling happens to match would not establish reproducibility; neither a tolerance change nor silently enabling deterministic mode is a remedy. The current explicitly experimental four-part path requires deterministic mode, and its synthetic correctness result remains a separate scope.

If selector-only default correctness needs a prospective diagnostic, capture one registered-model forward row and run both selectors and the full sampler on that same immutable row. These would be diagnostic kernel calls, not enabling the unsupported four-part API route with deterministic mode off. That would isolate selector correctness, not close whole-model repeatability or speed. Unique RMSNorm attribution would additionally require actual first-divergence inputs/warp partials and an ordered control. No GPU diagnostic or production/default change was made for this source/receipt audit.
