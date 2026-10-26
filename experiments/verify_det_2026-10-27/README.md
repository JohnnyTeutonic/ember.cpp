# Deterministic mode selects the exact verify pass (27 Oct 2026)

The CUDA verification pass of `EMBER_SPEC=k` (`../verify_2026-10-27`) takes its linear layers from
`EMBER_VERIFY_LINEAR`. Its default, `smallm`, gives rows within ~5e-6 of the decode step's logits, so speculation's
ids equalled plain decode's only as measured. `rows`, the decode step's own GEMV once per row, is the decode step's
arithmetic, bit for bit under `EMBER_DETERMINISTIC=1`. Now:

- **`EMBER_DETERMINISTIC=1` selects `rows`** when `EMBER_VERIFY_LINEAR` is unset or empty
  (`speculative_verify.cpp`, `TinyLlamaModel::verify_options`). An explicit value always wins.
- **Default mode is unchanged:** with `EMBER_DETERMINISTIC` unset the unset variable still means `smallm`, the same
  expression as before.
- **`TINYLLAMA_LOGITS_DUMP` covers speculation** (`api.cpp`): the speculative loop dumped no row before. With the
  dump (or `TINYLLAMA_TOPK_DEBUG`) on, every verification row a token is taken from is fetched from the device and
  dumped, one line per generated token as in plain decode; the token is still the device record's pick. With both
  variables unset nothing is fetched and the loop is as before.
- `docs/USAGE.md`: the switch table and the deterministic-mode section.

## The CUDA gate (one Colab L4, session `ember-vd-1027`, 27 min)

Build: ember main 345d7f1 plus this change, Release for sm_89 (`receipts/cuda/bins.sha256`). Models: pick_e1_L16
and pk_base_v1 (`big.sha256`). Requests: `../verify_2026-10-27`'s all60 (the 40 frozen chat prompts and the 20
text-repeating copy20 requests), greedy, each sent twice and the second recorded (`vd_collect.py`). Every
generated token's logit row is dumped (prefill row, then decode step or verification row) and compared as float32
bit patterns (`vd_compare.py`), against plain decode in the same mode.

| model | run (`EMBER_DETERMINISTIC=1` unless noted) | ids, stop ids, replies | rows bit-identical | max \|d\| | drafted / accepted / passes |
|---|---|---|---|---|---|
| pick_e1_L16 | plain, second run | 60/60 | 1409/1409 | 0 | - |
| | k = 2 (unset: `rows`) | 60/60 | **1409/1409** | 0 | 537 / 358 / 269 |
| | k = 4 | 60/60 | **1409/1409** | 0 | 781 / 423 / 204 |
| | k = 8 | 60/60 | **1409/1409** | 0 | 1196 / 457 / 169 |
| | k = 4, `rows` set | 60/60 | 1409/1409 | 0 | 781 / 423 / 204 |
| | k = 4, `smallm` set (control) | 60/60 | 507/1409 (18 requests whole) | 7.2e-6 | 781 / 423 / 204 |
| | default mode: k = 4 against default plain | 60/60 | 100/1409 | 7.9e-6 | 781 / 423 / 204 |
| pk_base_v1 | plain, second run | 60/60 | 1997/1997 | 0 | - |
| | k = 2 (unset: `rows`) | 60/60 | **1997/1997** | 0 | 796 / 696 / 398 |
| | k = 4 | 60/60 | **1997/1997** | 0 | 1032 / 834 / 262 |
| | k = 8 | 60/60 | **1997/1997** | 0 | 1327 / 925 / 171 |
| | k = 4, `rows` set | 60/60 | 1997/1997 | 0 | 1032 / 834 / 262 |
| | k = 4, `smallm` set (control) | 60/60 | 650/1997 (24 requests whole) | 9.5e-6 | 1032 / 834 / 262 |
| | default mode: k = 4 against default plain | 60/60 | 62/1997 | 9.5e-6 | 1032 / 834 / 262 |

`receipts/cuda/cmp/*.json`. So under the mode with the variable unset, speculation gives plain decode's ids on
every request and every sampled row's logits bit for bit, at k = 2, 4 and 8 on both models, the same as `rows`
set by hand. The `smallm` control shows that the comparison does see the default kernel's ~1e-5 differences
(ids still equal). The default-mode line is the old behaviour: equal ids, rows apart by the decode rmsnorm's
run-to-run atomics and `smallm` (`../determinism_2026-10-26`); the mode's own plain decode repeats bit for bit.
`/health` reports `"deterministic": true` in the mode's runs and has no such field in default mode.

Without the dump (`vf_e2e.py`, `receipts/cuda/e2e/`): ids 60/60 at k = 2, 4, 8 on both models, and the mode's
speed-up over its plain decode: pick_e1_L16 1.224 / 1.205 / 1.101x, pk_base_v1 1.302 / 1.339 / 1.310x (the
earlier `smallm` pass in the mode: 1.40-1.55x on pick_e1_L16, `../verify_2026-10-27`).

The guarantee holds for one GPU and build, as deterministic mode's does (`../determinism_2026-10-26`): this is an
L4 measurement on two models; other GPUs, CUDA or cuBLAS versions are not covered.

## Files

- `vd_collect.py` (a server run with every sampled row dumped), `vd_compare.py` (request-by-request ids and bits).
- `stage.sh`, `drive.sh`, `vm.sh`: the Colab run (microtorch/AGENTS.md path C). The driver's exit stop did not take
  on this run (the session was still listed a minute later and was stopped by hand); `drive.sh` now retries the
  stop until the listing drops it.
- `receipts/cuda/`: VM log, hardware, binary and model sha256, the dumps' sha256 (`rows.sha256`; the dumps stayed on
  the VM), records and meta per run, comparisons, end-to-end records.

Run (WSL): `bash stage.sh ~/vd/stage && bash drive.sh <session> 110 ~/vd/stage`; results land in
`/mnt/c/ml_artifacts/ember_verify_det/<session>/`.
