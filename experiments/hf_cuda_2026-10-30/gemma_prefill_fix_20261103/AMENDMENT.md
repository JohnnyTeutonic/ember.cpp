# Gemma3-270M prefill after the attention fix: a re-measure on the frozen CUDA comparison

Claude Opus 5.5, 3 Nov 2026. Fixed before any GPU result for this engine source. It runs the frozen comparison
(`../README.md`, controller `../benchmark.py`) for Gemma3-270M again, exactly as `../gemma_colab_20261103/` ran it
(run `g6-20261103`), with one change: the engine is the prefill fix. Review: Codex methods review, kept in the
research workspace (`AI_ML/gemma_cuda_claude/reviews/`).

## Why

`g6-20261103` passed every parity gate. Ember's prefill was about 5x slower than HF's at 512 tokens (174.0 ms against
34.3 ms) while its decode was 4.5-5.5x faster, and the gap was left undiagnosed. The suspected cause, from the
source (a plausible bottleneck, not yet a measured contribution):
- Gemma 3's head_dim is 256. The tiled prefill kernels took only head_dim up to 128 (flash2 only 32 and 64), so
  Gemma's global layers ran the baseline kernel.
- Its sliding-window layers, five in six, ran the IR's `window_attention_kernel`. That kernel uses one block per
  (token, head), walks every visible key and accumulates values serially.

The fix, ember branch `claude/gemma-prefill-flash` at `5794343c` (tree `c4926553`), extends the tiled kernel to
head_dim 256, with 16 keys per tile so that its two shared tiles fit in 32.8 KB, and adds a window lower bound to it.
The IR now runs multi-token window attention with no sinks and no soft cap on it; decode and every other case keep
their kernels. This run measures what that does to the frozen comparison.

## What is unchanged from `../gemma_colab_20261103/AMENDMENT.md`

Everything except the engine source and the added gate:
- the controller (`benchmark.py`, SHA-256 `ddd3104489dd2935…`; byte-identical at the fix commit) and its frozen
  arguments;
- the parity gate (`max_abs_error / max(1, max_abs_HF_row) <= 1e-4` on every full-vocabulary row and greedy id
  before any timing) and the ABBA schedule;
- the toolkit (CUDA 12.6 pins), interpreter (CPython 3.11.15 via uv 0.11.29), packages, launch environment and
  contract;
- the inputs, weights and canonical GGUF (`0a174d42…`);
- the build and the driver's hardening, cleanup and receipt rules.

The VM script and driver are that packet's copies, with only these differences:
- the source pin;
- the folder, session name (`claude-gemmafix-RUN_ID`) and artifact directory
  (`/mnt/c/ml_artifacts/benchmarks/hf_cuda_gemma_prefill_fix_20261103/`);
- the added gate;
- the driver also checks `status.json`'s `source` against the pin.

## What differs

- **Engine source:** ember `5794343c`, tree `c4926553`, checked by commit and tree as before. It is main `c214802`
  plus the fix (two commits: the kernel change and its test). g6 ran the older frozen tree `ee183b61`, and the
  ember changes between that tree and main `c214802` are not shown here to be irrelevant to these timings. So this
  run is a re-measure on a new engine source: a change from g6 is that source's, not attributed to the patch alone.
- **`kernel_test` gate**, after the build and before every download and forward. It runs
  `tools/flash_window_attention_test.cu` at the engine source, compiled with the same toolkit for sm_89:
  - both new entry points and the existing window kernel, against an FP64 CPU reference;
  - head dims 32-256, heads (4,1), (4,2) and (8,8);
  - 2-600 tokens at offsets 0, 13 and 500, and windows 1-1024;
  - pass: `max |out - ref| / max(1, max |ref|) < 1e-5` everywhere, and the window entry point takes every shape.

  It also times Gemma 3 270M's attention shape (4 q heads over 1 kv head, head_dim 256, 512 tokens, window 512):
  the window kernel against the tiled window and causal kernels, median of 20 each. The three kernels' outputs at
  that shape are scored against the reference before they are timed.
  - A host self-check first confirms that a NaN output fails the score: any non-finite output or reference scores
    infinity (Codex round 1's blocking finding).
  - Every CUDA call is checked, and outputs are NaN-filled before each launch.
  - Its output (`kernel_test.txt`) and the test binary's hash (`kernel_test.sha256`) are retrieved with the receipts.

## Outcomes, stated before the run

The parity gate decides validity exactly as before. Timings are reported as the frozen README reports them, beside
`g6-20261103`'s. g6's figures are fixed in its receipts:
- ember prefill 7.0, 20.1 and 174.0 ms;
- HF 34.1-34.3 ms;
- HF/ember decode 5.52x, 5.26x and 4.48x at 16, 128 and 512 tokens.

- **Gap closed:** every gate passes, and ember's median prefill at 512 tokens is no slower than HF's in this run.
  As for the other outcomes, the change is the new source's (see Engine source), with the kernel timing as
  supporting evidence for where it comes from.
- **Gap narrowed:** every gate passes, and ember's 512-token prefill is faster than g6's but still slower than HF's.
  The remaining factor is reported, not explained.
- **No improvement:** every gate passes, and ember's 512-token prefill is not faster than g6's. The diagnosis above
  is then incomplete, and that is the finding.
- **Valid negative, operational failure, reruns:** as the parent packet defines them. A kernel_test failure is a
  defect in the fix and stops the run before any forward; it is not a numerical result of the comparison.
- **Limits:** the two runs are on different Colab VMs on the same day, so the comparison with g6 is
  across runs, not within one. HF's own figures in this run are the within-run reference. The run makes no claim
  about other models, decode, or other devices.

## Budget and procedure

- **Lane and time:** one Colab L4, Claude's lane. Expected about 40 minutes, as g6. Deadlines are the parent's:
  polling 60, stop and retrieval 75, shutdown 85, inventory 90.
- **Command:** `python3 experiments/hf_cuda_2026-10-30/gemma_prefill_fix_20261103/drive.py FULL_COMMIT_SHA RUN_ID`,
  from an ember checkout's root, after the reviewed packet commit is pushed. RUN_ID is `f1-YYYYMMDD`.
- **Tests:** 53 local tests, the parent's adapted:
  - `test_vm_gemma.py`, 22, including a failed kernel test that stops before any download or forward;
  - `test_drive.py`, 23, including a wrong engine source failing validation;
  - `test_launch_code.py` 1, and `test_apt_reconcile.py` 7.
- **Local evidence (no GPU):**
  - the kernel test compiles with CUDA 12.6 for sm_89; the head_dim-256 kernel uses 155 registers, no spills and
    32.8 KB of shared memory;
  - the full ember tree builds with CUDA, `ir_hf_cuda_bench` included;
  - the CPU ctest suite passes.

## Runs

- **`f1-20261103`:** stopped at the driver's local preflight, before any session existed. It had been started from an
  ember checkout, and the driver calls paperkiln's `tools/` helpers. No VM, nothing spent.
- **`f2-20261103`** (packet `0feeff1d`, engine `5794343c`): all 13 gates passed, about 23 L4-minutes. The outcome
  is gap narrowed: ember's prefill at 512 tokens was 47.23 ms against HF's 34.03 (g6: 174.0). The kernel test shows
  that on Gemma's window layers the tiled kernel is slower than the window kernel it replaced (1.79 against 0.59 ms
  per call). `RESULT.md` has the details and the implied next step.
