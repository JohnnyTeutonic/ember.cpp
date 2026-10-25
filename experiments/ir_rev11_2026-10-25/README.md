# IR spec revision 1.1 in ember (25 October 2026)

The contract of keramos v1.1.0 (commit 322162aea13deba0039fc4c8e421fa72078c5b27; the text of paperkiln
7df7b10b with its paths changed and section 2.6 added): the digest (IR_SPEC 3.10, rule 30), stored-tensor values (4.7) and the
recurrence profile (6.10, rules 31 to 34). Code: ember a373179 (dequantisation) and 4c27bfa (digest,
recurrence, validator, adapter); gates 39dcce9. The digest and the recurrence op were written from the
specification text; paperkiln's `src/ir.cpp`, `recurrence.hpp` and `recurrence.cpp` were not read.
`tools/ir_validate.py` was run as a black-box oracle only.

## Results

Pinned build: `build_pinned.sh 39dcce9` (git archive, CPU only, Release, GCC 13.3, WSL ext4):
`~/ember_adapter_39dcce9/build/ir_adapter`.

**The suite** (`run_suite.sh`, keramos's runner at v1.1.0): 221 pass, 0 fail, 53 skip. Receipt:
keramos `conformance/receipts/ember_cpu.json` (keramos 178f2a2, not pushed).

| Kind | Pass | Fail | Skip |
|---|---:|---:|---:|
| model | 21 | 0 | 0 |
| trace | 37 | 0 | 0 |
| op | 7 | 0 | 46 |
| refuse | 72 | 0 | 0 |
| digest | 74 | 0 | 0 |
| dequant | 10 | 0 | 0 |
| grad | 0 | 0 | 7 |

The op skips are the fixtures of ops other than `recurrence` (the adapter's `run_node` runs recurrence
nodes; the other ops are checked through whole models and traces); the grad skips are `run_node_grad`
(ember has no backward).

- **Digest.** 74/74 vectors, the canonical form equal too (RFC 8785 Appendix B's numbers among them).
  `digest_stress.py`: 260,616 number spellings (60,000 random binary64 patterns, every decimal exponent
  from -330 to 308, four spellings each) and 1,600 objects of random strings and member names, every
  canonical form and digest equal to the reference's (`digest_stress_cpu.json`). Rule 30: the suite's
  pk_recurrence (which carries `paperkiln.ir_sha256`) loads; a wrong key, a key of another type and a
  wrong key under a substituted document are refused, a file without the key loads (self-test
  `rev11/rule30`).
- **Dequantisation.** All ten types bit-equal, through both of ember's routes (the interpreter's row route
  and the whole-tensor one), natively and in WebAssembly (`dequant_check.cpp`, `wasm_check_cpu.txt`).
  Before a373179 the same check misses 64 Q8_0 values (a negative scale made positive) and 17 Q5_0
  values (a -0 product written +0).
- **Recurrence.** op 7/7 (EXACT cases bit for bit: ssm_scan, conv1d_causal, ema_exact); traces
  pk_kimi, pk_srd, pk_mamba and tp_mamba1 with their rewritten documents, 9 steps each; refusals 12/12,
  each naming its rule (31 to 34); pk_recurrence's model case (greedy 4/4, log-probabilities within
  9.1e-8) and its decode and snapshot traces. Self-test: on all five recurrence models prefill =
  token-by-token decode = random chunks bit for bit (logits and recurrent tensors), truncate = a fresh
  run, snapshot and restore; the rewrites give the hand-written ops' logits and states bit for bit for
  kimi, srd and mamba, and within 1.4e-7 (logits) and 4.6e-7 (states) for tp_mamba1's selective scan; an
  init body `s[r] = B[r] - B[r]` equals the +0 start bit for bit, `s[r] = B[r]` changes the logits,
  decodes = prefill, and truncate 0 re-applies it. In WebAssembly pk_recurrence gives greedy 40/40 and
  top-1 320/320 against the native build, logits within 7.9e-8.
- **CUDA.** `build_cuda_check.sh`: graph_ir, graph_ir_cuda, the new units, api, weight_management and
  quantization compile with HAS_CUDA on. A recurrence document runs on the CPU in a CUDA build
  (`device_supported` is false: no device kernel), as MoR documents do.

## Regressions

- Adapter self-test (`../ir_adapter_2026-10-25`): 156/156; the 119 earlier checks give the same results,
  118 byte for byte and `skip/run_node` with a new skip message; 37 checks added; `hello` lists the new
  methods and profile.
- recursion_gate, mor_gate, attnres_gate: identical but for the binary's path; mor_refusals,
  state4_gate and ir_numerics_test (numerics_cpu.json): byte-identical.
- state_tokens_gate: every ember result identical; only the distances to paperkiln's numpy reference
  moved (in the seventh digit, as at d49c3e1).
- tuned_path: the 108 stored files of the receipt take the same path; 11 files added to
  /mnt/c/ml_artifacts since (pk_recurrence among them) take theirs as expected; 14/14 edits.
- agreement.py (run with paperkiln's tools at their new place): ember's verdict unchanged on all 181 load
  cases still in the corpus, and agrees with the validator on the 20 new ones and the 90 documents;
  one case reads DISAGREE because the validator now accepts the S = 1 pseudo-query that the script
  still expects it to refuse (a script expectation, not ember).

## IR_SPEC Appendix C (ember's statements)

- No ember path multiplies quantised blocks by quantised activations. The interpreter (CPU, CUDA,
  WebAssembly) and the tuned llama path dequantise stored blocks with the 4.7 routines and multiply the
  f32 values with f32 activations (the tuned CPU path dequantises a block at a time inside its matrix-vector
  product); CUDA dequantises on the host and uploads f32. quantization.cpp still defines Q4_K x Q8_K and Q6_K x Q8_K dot
  products and a Q8_K quantiser, but nothing calls them.
- The one quantised-activation path is opt-in and is not about stored tensors: the tuned llama path's
  CUDA INT8 KV cache (`use_kvcache_quantization`, off by default) quantises K and V rows to int8.
- The interpreter's recurrence op is CPU only; EXACT when its bodies use no exp, log, tanh, erf, sigmoid
  or ^, otherwise ROUNDING; compiled with `-ffp-contract=off`.

## Where the specification was silent (no paperkiln source read; settled from the vectors or the oracle)

1. 3.10: an integer token `-0` is the integer zero, not a negative zero (vector minus_zero_integer_form);
   the text says only "a negative zero".
2. 3.10: the reference checks non-finite numbers, integers beyond 2^53, lone surrogates and negative zeros
   only in what remains after the `_` members are removed (`{"_a": 1e400}` has a digest), and duplicate
   names before. The text puts every I-JSON condition in step 1, before the removal; ember follows the text
   and refuses all of them anywhere. No vector tells the two apart.
3. 6.10.1: whether a declaration's indices must be distinct (`W[c,c]`) is unstated; ember allows a repeated
   index in a declaration (only sizes matter) and refuses one in an assignment (rule 32).
4. 6.10.2: an unknown name read without brackets has no rule number; ember reports rule 32.
