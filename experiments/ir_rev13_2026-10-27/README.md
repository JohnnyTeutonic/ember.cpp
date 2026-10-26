# IR spec revisions 1.2 and 1.3 in ember (26-27 October 2026)

ember at keramos **v1.3.0** (da9ead6175925aa6b19758aa828cdc42236a45a6; paperkiln pins it at 68ab7a32), through
v1.2.0 (d4b880ae2a42865f4334b2c3ea27af6db02ffc84). Code: ember 1522936 (rules 35-37, citations), 66e648e
(adapter 3), a2340fa (rules 38-39, the builder's MoE llama, position_bias), 4e75c02 (adapter 4); schema
ea4a2a6 / 13cd2b4; gate scripts f4c26a2, 1ab6a25 and `../ir_rev12_2026-10-26/`. Everything was written from
IR_SPEC.md at the tags; paperkiln's `ir.cpp` and recurrence sources and keramos's `cleanroom/` were not read.
keramos's `tools/ir_validate.py` and `tools/ir_numpy.py` were used as black-box oracles only (their function
signatures and docstrings, their verdicts and values); `tools/ir_conformance.py` (the runner: its case
functions and `load_model`) and `tools/gguf_relabel.py` (run as a tool) were read for the protocol, and
`adapters/llamacpp/FINDINGS.txt` and `adapters/hf/FINDINGS.txt` for the outside engines' findings.

## Results

Pinned build: `../ir_rev12_2026-10-26/build_pinned.sh 1ab6a25` (git archive of
1ab6a252c8b11823412c3703b25c8b60b9bdc261, CPU only, Release, GCC 13.3, WSL ext4):
`~/ember_adapter_1ab6a25/build/ir_adapter`, sha256
a73412f63c90df51eb03e4fd780cd4486ef69e8b0e08a8ddeb68c1d53446a321. Later commits change no source.

**The suite** at v1.3.0 (`../ir_rev12_2026-10-26/run_suite.sh v1.3.0`, from a git archive of the tag, no
EMBER_* variable): **355 pass, 0 fail, 49 skip** (`suite_v1.3.0_cpu.json`; keramos
`conformance/receipts/ember_cpu.json`). At v1.2.0 a dev build of the same 1.2 code gave 318/0/52.

| Kind | Pass | Fail | Skip |
|---|---:|---:|---:|
| model | 21 | 0 | 0 |
| trace | 62 | 0 | 0 |
| op | 21 | 0 | 41 |
| refuse | 91 | 0 | 0 |
| digest | 84 | 0 | 0 |
| dequant | 10 | 0 | 0 |
| grad | 0 | 0 | 8 |
| accept | 34 | 0 | 0 |
| chat | 32 | 0 | 0 |

The op skips are the single-op fixtures of ops other than recurrence, expr and attention (whole models and traces
test them; rope's `first_head` among them, which ember refuses at load by name); the grad skips are
`run_node_grad` (ember has no backward).

- **Rule 35** (3.2): `parse_ir_document` refuses a member name twice in one object at any depth, `_` members
  included; every load goes through it. A file whose own document holds one loads with a valid document in its
  place, as the reference accepts it.
- **Rule 36** (3.5, 7.4(b)) and **rule 20**'s one run of steps (refused before, `mor_step_gap`).
- **Rule 37** when the rows arrive, CPU and device: scale, split, rmsnorm group, add/mul/sub, mul_col, rope's
  q and k widths (a new check: the kernels took the head count from the width), depth_attend, expr. The error
  names the rule and the adapter leaves the session as it was.
- **Citations**: every rule a message cites is section 10.1's (`rule_citations.py`, `rule_citations_cpu.json`):
  on 171 one-edit mutations and refusal documents ember and the reference accept or refuse alike, and each rule
  ember cites is one the reference cites.
- **Rule 38** (3.1): the embedded document of a file labelled with a section 9.3 architecture must be the graph
  `build_ir_from_gguf` makes from the file (up to 3.1's list); "paperkiln" is never compared; an unknown
  architecture, or metadata that builds no graph, is refused. The file's own document is checked also when
  another is read in place of it, as the reference does. The builder gains Mixtral's layout (a `moe_ffn`
  for `llama.expert_count` with `ffn_gate_inp`), so pk_moe is its architecture's graph.
- **Rule 39 and position_bias** (5.15): `attention_bias_batch_cpu` adds W[b(p - j), h] after the rounded
  scaled product on absolute positions, with windows, sinks, grouped-query attention and turn_state's compact
  rows. Op cases 3/3 (within 1.6e-7); in a whole model (pk_llama with a bias table, causal and windowed with a
  sink) within 1.6e-7 of the reference, prefill = decode bit for bit, and the bias moves the logits. CUDA builds
  run such a document on the CPU (`device_supported`); WebGPU leaves it to the CPU interpreter; depth_route with
  position_bias is refused by name.
- **The windowed rewind** (6.9): ember's cache holds every position up to max_len, so a cut always keeps the
  rows of positions < n: pk_gpt2_swa (14 tokens, cut to 7) and tp_gemma3 (48, cut to 10) continue bit for bit as
  a fresh run over the prefix (self-test `rev13/rewind_window`), including a cut into the region llama.cpp's
  windowed cache drops (its finding 4). No ember path drops a windowed row, so none needs to re-prefill or
  refuse.
- **gpt2-labelled IR-only files** (llama.cpp finding 7, HF H4): ember's loader reads `paperkiln.ir` before the
  architecture (`find_ir`), the tuned llama path takes only the llama graph, and rule 38 now refuses such a file
  at load: `architecture_kimi_as_gpt2` and `_swa_as_gpt2` are refused (also with their own document read in
  place), their relabelled twins pk_kimi and pk_gpt2_swa (the same document and digest) load and match the
  reference within 1e-5 (self-test `rev13/rule38`). `model_config` and `chat_tokenizer` decide word vocabularies
  by `tokenizer.ggml.model`, never by the architecture, so "paperkiln" and "gpt2" files read alike.
- **Adapter** (11.3): `check` with gguf null, `render_chat` (32/32 through the vendored `chat_template.h`),
  errors leave the session unchanged, empty forward and truncate past the end are errors, truncate n = len
  changes nothing, a reused snapshot tag replaces, depth `[]` without depth_route, `run_node` on expr and
  attention.

## Regressions

- Adapter self-test (`../ir_adapter_2026-10-25`, 189/189; `selftest_vs_rev11_receipt.txt`): of the revision 1.1
  receipt's 156 checks, 143 give the same results byte for byte; 12 differ only as revisions 1.2 and 1.3 require
  (seven schema messages now cite rules 1, 4 and 8 in place of "10.1 rule 1"; the two run_node skip messages
  name expr and attention; `rev11/digest_vectors` counts v1.3.0's 84 vectors where v1.1.0 had 74, and the next
  three checks' request ids move by those 10); `skip/depth_without_depth_route` is now
  `depth/without_depth_route_empty` (11.3: `[]`, not a skip). It runs on the relabelled gate models
  (`relabel_models.sh`) and on the suite at v1.3.0 for its rev11 and rev12 checks, whose pk_kimi and
  pk_recurrence carry "gpt2" before v1.3.0.
- recursion_gate, mor_gate, mor_refusals, attnres_gate, state4_gate byte-identical; state_tokens_gate identical
  but for the distances to paperkiln's numpy reference, as before (`cpu_regress.txt`).
- agreement.py at v1.3.0's tools (`agreement_v1.3.0_cpu.json`): 382 load cases, 380 agree and 2 expected
  capability differences (two kv slots; SmolLM2, a plain llama file the tuned path serves without a document);
  228 document cases agree; 0 disagreements. Two script fixes: the cited form of rule 24's message ("IR_SPEC 3.1
  (rule 24)") is dropped for third-party files as the uncited one was, and a plain llama or mistral file is
  labelled as a capability difference.
- CUDA compile check of the pinned sources: rc 0 (`cpu_regress.txt`).

## Findings

1. Rule 38 refuses three stored files that carry "gpt2" (the reference agrees): `/mnt/c/ml_artifacts`
   `kimi_tiny`, `swa_tiny` and `capacity/flex_plain/flex_plain_tiny`; tools/gguf_relabel.py relabels them (the
   self-test uses relabelled copies). The other 142 of the 145 stored files with a document pass, under the
   reference and ember alike.
2. No stored document is refused under rules 35, 36 or 37 (agreement.py's 382 load and 228 document cases).

## Where the specification was silent (settled from the reference as a black box, or by the text's wording)

1. Rule 35 with a document read in place of a file whose own holds a duplicate: the reference accepts; ember too.
2. Rule 38 with a document in place: 3.1 says the rule concerns the embedded document; the reference still checks
   the file's own document when another is read in its place, and ember does.
3. Rule 36 under depth_route: a block embedding reading `tokens` is refused (the text names x and pos only; the
   reference agrees). depth_route's own out is written before the step's nodes, so reading it is rule 17a's alone.
4. Rule 38's renaming: the text lists value and slot names; a block also carries values across applications, so
   ember's form records what each name the block reads before writing it holds after the block.
   `rmsnorm.group`'s default (the row width) needs widths, so a group is compared as written (no 9.3 graph writes
   one equal to the width).
5. position_bias under depth_route (positions or active rows, as Q16 asks of windows) is undefined; ember refuses it.
