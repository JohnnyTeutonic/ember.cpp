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
