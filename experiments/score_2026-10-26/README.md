# `POST /score`: log-probabilities with `mtstudio score`'s semantics (26 Oct 2026)

ember's server gains `POST /score` (`score.h`, `score.cpp`, `server.cpp`; docs/USAGE.md), the engine for paperkiln's
`tools/bench_eval.py --scorer ember` (HellaSwag, ARC, PIQA, WinoGrande, LAMBADA, BLiMP by forced choice). The contract
is `mtstudio score`'s (`microtorch/tools/mtstudio.cpp` `score_cmd`) on token ids:

- request `{"requests": [{"key", "prompt_ids", "continuation_ids"}, ...]}`; response `{"results": [{"i", "key",
  "prompt_tokens", "continuation_ids", "over_ctx", "logprobs", "argmax", "sum"}], "stats"}`;
- each request one fresh forward over `prompt + continuation[:-1]` from position 0; per continuation token the
  log-softmax at temperature 1 in double from the float logits (mtstudio's loop: max, sum of exp(x - max) in vocabulary
  order, x_c - max - log(sum)); the argmax with `<unk>`, `<s>`, `</s>`, `<pad>` left out by name, first maximum
  winning; `sum` in order; `over_ctx` when prompt + continuation > ctx + 1;
- refusals where mtstudio fails (HTTP 400 for the whole call, naming the first bad request): not an object, ids not a
  list of integers (nlohmann's `get<std::vector<int>>`, as mtstudio reads them), empty prompt or continuation, prompt id
  outside [0, vocab + k), continuation id outside [0, vocab), state ids not in complete blocks after a reply's
  end-of-text (`nn::state_ids_error`, ported). Two documented differences: mtstudio's text fields are not served (ids
  only), and a call with a bad request is refused whole where mtstudio writes the records before the bad line.

Speed without changing a bit, CPU only: consecutive requests that share a prefix keep its KV rows (up to the prompt's
last position), and the final norm and LM head run on the continuation's rows only (`Interpreter::forward`'s new
`tail_rows`, the last-row trim generalised; the llama path passes those rows to `forward_cpu_logits_batch`). Every CPU
row is computed apart from the batch it is in (one dot product per row and output), so this gives the bits a fresh full
prefill gives; `EMBER_SCORE_FULL=1` is that reference, and the gate checks it. The device, recurrent state, eviction,
depth routing and state tokens always prefill afresh with every row. On CUDA the scoring forward runs FP32 GEMMs
(`TinyLlamaModel::set_gemm_fp32_only`: no BF16 tensor cores, no TF32, whatever `EMBER_GEMM_TF32` says); `/chat`'s
prefill is unchanged. `/score` is off (403) under `EMBER_PUBLIC=1`.

One fix on the way: a GGUF without `output.weight` (llama.cpp's converter omits it for a tied head, as for SmolLM2) now
uses `token_embd` as the head on the tuned llama path, as llama.cpp does; before, every forward of such a file failed
("No valid LM Head weights"). Files with `output.weight` are untouched.

## Gates (`score_gate.json`, `sc_analyze.py`; `pass: true`)

Requests (`sc_make_requests.py`, `sc_make_requests_smol.py`; sha256 in the receipt): 60 real items of `bench_v1.jsonl`
(10 HellaSwag, 8 ARC-Easy, 6 ARC-Challenge, 10 PIQA, 10 WinoGrande, 20 LAMBADA, 15 BLiMP pairs, seeded) tokenised and
truncated as `bench_eval.py` does (prompts open with end-of-text), 8 long prompts of 300-420 ids, and for paperkiln's
vocabulary (T = 512) the over_ctx edges P + C = 513 (scored), 514 and ~540 (not) and a 512-id prompt with one
continuation id: 198 requests on paperkiln's vocabulary, 194 on each third-party one; and 256 HellaSwag requests
(64 items) per vocabulary for throughput.

**CPU against `mtstudio score`** (WSL; paperkiln's CPU build `~/tpops_build/mtstudio`, its `score_cmd` identical to
paperkiln's working tree; both models F32; pick_e1_L16's weights rebuilt from its GGUF by `sc_gguf2st.py`, see below):

| model (tuned llama path) | over_ctx | argmax | logprob \|d\| median / p99 / max | ROUNDING | sum max \|d\| |
|---|---|---|---|---|---|
| pick_e1_L16 (20m, L16) | 198/198 agree (2 over) | 2732/2732 | 1.3e-6 / 6.9e-6 / 1.13e-5 | 4.3e-6 | 3.1e-5 |
| pk_base_v1 (d768 L12) | 198/198 agree (2 over) | 2732/2732 | 1.0e-6 / 5.8e-6 / 1.09e-5 | 2.6e-6 | 3.6e-5 |

ROUNDING (1e-5 of max(1, max |ref|), the tolerance of every earlier ember-vs-mtstudio log-prob gate) holds with room.
The plain |d| <= 1e-5 asked for misses on 8 and 1 of 2,732 tokens (worst 1.13e-5 and 1.09e-5): float32 rounding
between two engines, not a semantic difference. A float64 evaluation of the same graph (`sc_f64.py`: keramos's
ir_numpy with every float32 read as float64) on each model's 10 most-different requests puts ember nearer exact
arithmetic on 9 of 10 for both; mtstudio's own error there reaches 1.4e-5 (pick) and 2.0e-5 (pk_base_v1) against
ember's 1.0e-5 and 1.2e-5. Every record's `prompt_tokens` and `continuation_ids` equal mtstudio's.

**CPU against HF transformers float32** (`sc_hf.py` / `sc_hf_gguf.py`: one forward per request, no padding; Qwen2.5 and
SmolLM2 loaded from the same GGUF; Pythia from the hub checkpoint at the pinned revision, since transformers reads no
gptneox GGUF, the F32 GGUF being its conversion, sha256 97522db7...):

| model | path | argmax | logprob \|d\| / HF's logit range (real-model rule <= 1e-4) | max \|d\| |
|---|---|---|---|---|
| Pythia-160m | interpreter | 2408/2408 | 8.3e-6 | 6.9e-3 |
| Qwen2.5-0.5B-Instruct (F16 GGUF) | interpreter | 2436/2436 | 2.2e-6 | 5.8e-5 |
| SmolLM2-135M (F32 GGUF, sha256 a898c120...) | tuned llama (tied head) | 2435/2435 | 1.2e-6 | 2.8e-5 |

Pythia's absolute differences are its known float32 conditioning (`../thirdparty_2026-10-22`, `../wasm_families_2026-10-25`).
SmolLM2 through paperkiln's own harness (`sc_smol_bench.sh`: `bench_eval.py run` on HellaSwag, ARC-Easy, PIQA and
LAMBADA, every 200th item, 296 requests; `--scorer ember` against `--scorer hf` at revision 93efa2f0): `bench_eval.py
compare` max |d logprob| 3.1e-5, argmax 6097/6097, choices 72/72, and every metric identical (HellaSwag acc_norm 0.471,
ARC-E 0.667, PIQA 0.667, LAMBADA 0.308 on this slice; a 72-item slice, not the full-set reference numbers).

**The fast path is the reference, bit for bit**: default against `EMBER_SCORE_FULL=1` byte-identical on pick_e1_L16's
198 and 256 requests; on a 60-request subset (whole multiple-choice groups, the long prompts and the over_ctx edges) of
each other model, every value bit-equal (pk_base_v1, Pythia, Qwen, SmolLM2). One call of 198 against calls of 50:
byte-identical (pick_e1_L16).

**Refusals** (`sc_mts_errors.py` on the same cases): mtstudio exits 1 on all ten of not-JSON, request not an object,
empty continuation, empty prompt, negative id, prompt id = vocab, continuation id = vocab, ids not a list, a string
id, a bad request after a good one; ember answers 400 on each, naming the request. Text fields: mtstudio tokenises and
scores, ember refuses (ids only). An empty list: both succeed with nothing.

**CUDA** (Colab T4, sm_75, session `ember-score-1026-t4`; no L4: paperkiln's three L4 sessions held the slots; the VM
built this tree as paperkiln's `ensure_ember` does; Pythia and Qwen fetched on the VM, sha256 equal to the local files):

| model | against ember CPU: ROUNDING / max \|d\| | argmax | EMBER_DETERMINISTIC=1 twice |
|---|---|---|---|
| pick_e1_L16 | 4.0e-6 / 1.9e-5 | 2732/2732 | byte-identical |
| pk_base_v1 | 2.3e-6 / 8.1e-6 | 2732/2732 | byte-identical |
| Pythia-160m | 4.0e-4 / 2.5e-3 (against HF: 1.1e-5 of range) | 2408/2408 | byte-identical |
| Qwen2.5-0.5B | 8.2e-6 / 9.1e-5 (against HF: 2.2e-6 of range) | 2436/2436 | byte-identical |

Pythia's device leg is judged by the real-model rule, as on the CPU (its CPU-vs-CUDA spread is the one
`../wasm_families_2026-10-25` records). The default CUDA run equals the deterministic one byte for byte on all four:
the scoring path (batched prefill, the per-row FP32 head matvec, the interpreter's kernels) has no order-dependent
reduction. `/chat` gave the same reply, ids and stop before and after `/score` on every server, CPU and CUDA.

**Throughput** (requests per second, the 256-request HellaSwag batch unless noted; CPU = 2 cores under the budget on
WSL's 4 CPUs, shared with other sessions' jobs throughout, so these are lower bounds):

| | ember CPU default | ember CPU full reference | mtstudio CPU (198 requests, incl. load) | ember CUDA T4 |
|---|---|---|---|---|
| pick_e1_L16 | 0.81 | 0.43 | 0.22 | 53.2 |
| pk_base_v1 | 0.22 | 0.09 (60) | 0.16 | 30.2 |
| Pythia-160m | 0.27 | 0.14 (60) | | 34.4 |
| SmolLM2-135M | 0.27 | 0.10 (60) | | |
| Qwen2.5-0.5B | 0.09 (gate set) | 0.03 (60) | | 17.3 (gate set) |

The contention moves these: pick_e1_L16's gate set ran at 0.91 requests/s in the gate and at 1.3 in a quieter rerun
on the final build (same bytes). Prefix reuse kept 10.9k of 22.9k positions on the HellaSwag batch (four choices per
item). For CUDA serving at
benchmark scale, an L4 or T4 is the place: a full HellaSwag (40k requests) is ~13-22 minutes there.

**Regression** (`sc_regress.sh`, the final build): `../cuda_state_2026-10-25/cpu_regress.sh` against the committed
receipts (`git show HEAD` copies): recursion, MoR, MoR refusals, attnres, state4, regress_parity, G5 third party
byte-identical; state_tokens_gate equal to ember HEAD's own build byte for byte (both differ from the committed file in
the numpy-reference distances only, as in `../determinism_2026-10-26`); regress_bits with HEAD's `ir_logits` as the old
side 16/16 bit-equal; adapter self-test 156/156. pick_e1_L16's gate set rerun on the final build (after the tied-head
fallback landed): byte-identical to its first run. The gate ran on ember df43463 plus this work; the commit sits on
17b3999 (the WebGPU IR work, meanwhile committed: an `ir_document()` accessor and a re-vendored Ollama Modelfile
writer, nothing on a scoring path). That tree's build (`sc_final2.sh`, `receipts/commit_tree.txt`): pick_e1_L16's 198
requests byte-identical to the gate's run, Pythia's and SmolLM2's 60-request subsets equal value for value.

## Notes

- pick_e1_L16's `model.safetensors` in `C:/ml_artifacts/tiny_chat/e1/pick_e1_L16/` holds only AdamW state (146 m and
  146 v tensors), though its sha256 is the one `best.json` records as `weights_sha256`; mtstudio cannot load it. The
  gate's mtstudio side used weights rebuilt from the exported F32 GGUF (`sc_gguf2st.py`, the inverse of
  `export_gguf_llama`; checked bit for bit on pk_base_v1: 110/110 tensors equal its safetensors).
- The CUDA leg ran this work before the tied-head fallback (no model there lacks `output.weight`).
- Scripts are as run, from the session's scratch directory (`S=` at their top); results (every record) are in
  `C:/ml_artifacts/ember_score/` (`cpu/`, `ember-score-1026-t4/`).

## Files

`sc_make_requests*.py` (requests), `sc_client.py` (server driver and comparisons), `sc_hf.py`, `sc_hf_gguf.py`,
`sc_hf_pythia.sh` (HF references), `sc_specs.py`, `sc_gguf2st.py`, `sc_mts_errors.py` (mtstudio side),
`sc_gate_cpu.sh`, `sc_gate_cpu2.sh`, `sc_gate_cpu3.sh`, `sc_smol.sh`, `sc_smol_bench.sh` (CPU legs, run in that order;
the later parts trim the full-reference runs for time), `sc_f64.py`, `sc_f64.sh` (float64 arbitration), `sc_stage.sh`,
`sc_drive.sh`, `sc_vm.sh` (Colab), `sc_build.sh`, `sc_build_head.sh`, `sc_mkpatch.py` (builds), `sc_regress.sh`,
`sc_final_pick.sh`, `sc_final2.sh` (reruns on the final build and the commit's tree), `sc_analyze.py`, `sc_dist.py`
(receipt). `sc_strip.py` was the first attempt at pick_e1_L16's weights (the file holds no weights; superseded by
`sc_gguf2st.py`). `receipts/`: `score_gate.json`, `regress.txt`, `commit_tree.txt`, `f64_*.json`, `mts_errors.json`,
`smol_bench_compare.json`, `smol_bench_report.md`, the VM's log and hardware.
