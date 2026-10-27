# `recurrence` on the WebGPU path, 27-28 Oct 2026

`deploy/wasm/web/webgpu/ir_webgpu.js` ran every op of the core, state, recursion and thirdparty profiles
in WGSL except `recurrence` (IR_SPEC 6.10), so a document with a recurrence went to WebAssembly. It now
compiles each recurrence node to WGSL from its bodies, keeps the recurrent state on the GPU, and the page
sends recurrence documents to `ir_webgpu.js` like any other (`irSupport` passes them; `worker.js` and
`ember_ir_doc()` are unchanged).

## How it runs

At load, `compileRecurrence` parses and checks the node with the rules of ember's CPU interpreter
(`graph_ir_recurrence.cpp`, rules 31-33, its messages and its depth limit), and `recurrenceKernels`
compiles one WGSL kernel per assignment (each `init`, each `let`, each `update`, the `readout`), one
invocation per point of the assignment's index space. Per application the node owns a work buffer
`[inputs | state | new state | lets]` and a weight buffer (its weights' f32 values, `{i}` and
`tensor_index` resolved as for every other op). Per token: a copy kernel per input, the lets in order, the
updates into the new state, a commit kernel that copies the whole new state over the old (so no state
tensor changes before every update has read the state before the token, 6.10.3), the readout. A new
sequence clears the work buffer and runs the init kernels (every other state tensor starts at +0).

A body compiles to straight-line f32 code, one WGSL `let` per primitive, in the CPU's evaluation order:
reads outside a tensor's shape are +0 (decided at compile time where the index ranges allow, otherwise a
bounds test), a bare index is its value as f32, `sum(j, e)` is a loop from +0 in ascending j, `where`,
`max`, `min` and `relu` are selections with the CPU's NaN behaviour, `sigmoid` is 1 / (1 + exp(-x)),
`^` is `pow_s` (repeated products for a literal exponent 1 to 4), numbers are strtof's nearest f32 and
are bound through `let`s, so nothing is folded at shader creation (`exp(100)` is +inf there, as on the
CPU, not a shader error). WGSL's division, sqrt, exp and log have error bounds of their own, so the path
is ROUNDING against the CPU, as every other op on it.

## The gates

SwiftShader only (Dawn's node bindings, npm `webgpu` 0.6.2, through `tools/webgpu_parity/swiftshader_run.py`:
SwiftShader's ICD the only Vulkan driver, no layers, every script refuses an adapter that does not name
SwiftShader, and a monitor of Windows' per-process GPU counters kills the run on any context on a hardware
adapter). Adapter on every run: `google | swiftshader | swiftshader-device-subzero- | SwiftShader driver 5.0.0`.
References: ember CPU at HEAD 9d993f4 (`tools/ir_wasm_gate`'s native build and `tools/ir_adapter`, built in
`~/e6rec`), and keramos's numpy reference (`tools/ir_numpy.py`, `tools/ir_recurrence.py`) as a black-box
oracle. Two cores, under the CPU budget.

Across the gate runs below the monitor read the counters 302 times (120 in the recurrence gates, 165 in
the regression gates, 17 in the HEAD comparison) and found no context on a hardware adapter; the op gate
and the refusal check finish before the first read.

### The documents

- **pk_recurrence** (keramos suite, `models/pk_recurrence`): mtstudio's trained recurrence model, its own
  document (a decayed linear attention written as a recurrence).
- **pk_kimi_rec, pk_srd_rec, pk_mamba_rec, tp_mamba1_rec**: keramos's `conformance/recurrence/` documents,
  the suite models with their hand-written recurrent ops rewritten as recurrences (IR_SPEC 6.10.6), on the
  suite's GGUFs: linear attention, linear attention beside a kv slot, ssm_scan, and Mamba-1's
  conv1d_causal + selective_scan (two recurrence nodes per block).
- **kimi_tiny_rec, srd_tiny_rec, mamba_tiny_rec** (`make_docs.py`): the trained tiny families' own documents
  rewritten the same way (keramos `tools/ir_recurrence.py` `rewrite`).
- **kimi_tiny_custom** (`make_docs.py`): kimi_tiny with its linear attention replaced by a recurrence of
  this experiment: init bodies over per-application weights (`blk.{i}.attn_norm.*`), a decay through
  `sigmoid` of a weight, an offset read with zero padding (`k[h,e - 1]`), a `sum`, a state tensor of
  another shape (`c[h]`), a bare index. Not a trained model; it exists to put those forms in a whole model.
- **mamba-130m_rec**: the document ember builds for state-spaces/mamba-130m (f32), rewritten: conv1d_causal
  and selective_scan as recurrences on a real 24-layer checkpoint.
- **Op cases**: keramos's nine `conformance/ops/recurrence.json` cases and six of this experiment
  (`synthetic_cases.json`, `make_cases.py`, expected values from keramos `ir_recurrence.run`): every function
  and operator in one node with an init body, a 100 x 7 state (more than one workgroup), a 3-D state with
  offsets both ways and constant positions inside and outside it, numbers whose values overflow f32 inside
  a body (`exp(100)`, `3.4e38 * 0`), the `^` forms (literal and computed exponents, negative bases), and
  30 nested sums.

kimi_tiny's file is labelled `gpt2`, which ember HEAD refuses for a document that is not the gpt2 graph
(IR_SPEC 3.1, rule 38), so every run here reads a copy relabelled `paperkiln` by keramos's
`tools/gguf_relabel.py` (swa_tiny's likewise, in the regression); the relabel changes only metadata keys.

### Gate 1: logits against ember CPU and numpy (`gates.py`, `tools/webgpu_parity/ir_gate.mjs`)

Each document's prompt one token per step, every position's logits against the reference's prefill row
(max |dlogit| over the row's range), the argmax at every position, then the greedy continuation. ember CPU
references: `ref.sh` (40 seeded prompts of 24 ids and 32 greedy tokens, 8 + 8 for documents declaring under
64 positions, EMBER_IR = the document); numpy references on the same prompts: `np_ref.py`. Receipts:
`<document>/cpu/` and `<document>/numpy/` (summary.json, check.jsonl, run.log).

| Document | Reference | Prompts x ids + greedy | Greedy | Top-1 | Worst \|dlogit\| / range | Median | Dispatches per token | s per prompt |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pk_recurrence | ember CPU | 40 x 8 + 8 | 40/40 | 320/320 | 1.7e-07 | 1.1e-07 | 52 | 0.2 |
| pk_recurrence | numpy | 40 x 8 + 8 | 40/40 | 320/320 | 2.2e-07 | 1.0e-07 | 52 | 0.2 |
| pk_kimi_rec | ember CPU | 40 x 8 + 8 | 40/40 | 320/320 | 1.5e-07 | 1.1e-07 | 52 | 0.1 |
| pk_kimi_rec | numpy | 40 x 8 + 8 | 40/40 | 320/320 | 1.8e-07 | 1.1e-07 | 52 | 0.1 |
| pk_srd_rec | ember CPU | 40 x 8 + 8 | 40/40 | 320/320 | 1.6e-07 | 1.1e-07 | 82 | 0.2 |
| pk_srd_rec | numpy | 40 x 8 + 8 | 40/40 | 320/320 | 2.3e-07 | 1.1e-07 | 82 | 0.2 |
| pk_mamba_rec | ember CPU | 40 x 8 + 8 | 40/40 | 320/320 | 2.1e-07 | 1.0e-07 | 40 | 0.1 |
| pk_mamba_rec | numpy | 40 x 8 + 8 | 40/40 | 320/320 | 2.2e-07 | 1.0e-07 | 40 | 0.1 |
| tp_mamba1_rec | ember CPU | 40 x 24 + 32 | 40/40 | 960/960 | 2.5e-06 | 8.2e-07 | 61 | 0.4 |
| tp_mamba1_rec | numpy | 40 x 24 + 32 | 40/40 | 960/960 | 2.5e-06 | 7.8e-07 | 61 | 0.4 |
| kimi_tiny_rec | ember CPU | 40 x 24 + 32 | 40/40 | 960/960 | 6.2e-07 | 3.6e-07 | 52 | 1.7 |
| kimi_tiny_rec | numpy | 40 x 24 + 32 | 40/40 | 960/960 | 7.9e-07 | 4.0e-07 | 52 | 1.7 |
| srd_tiny_rec | ember CPU | 40 x 24 + 32 | 40/40 | 960/960 | 3.9e-07 | 3.3e-07 | 82 | 2.0 |
| srd_tiny_rec | numpy | 40 x 24 + 32 | 40/40 | 960/960 | 4.4e-07 | 3.8e-07 | 82 | 1.9 |
| mamba_tiny_rec | ember CPU | 40 x 24 + 32 | 40/40 | 960/960 | 6.8e-07 | 4.2e-07 | 40 | 1.2 |
| mamba_tiny_rec | numpy | 40 x 24 + 32 | 40/40 | 960/960 | 5.9e-07 | 4.8e-07 | 40 | 1.2 |
| kimi_tiny_custom | ember CPU | 40 x 24 + 32 | 40/40 | 960/960 | 4.6e-07 | 3.6e-07 | 56 | 1.6 |
| kimi_tiny_custom | numpy | 40 x 24 + 32 | 40/40 | 960/960 | 6.3e-07 | 4.2e-07 | 56 | 1.6 |
| mamba-130m_rec | ember CPU | 10 x 24 + 32 | 10/10 | 240/240 | 1.2e-04 | 2.5e-05 | 699 | 54.2 |
| mamba-130m_rec | numpy | 2 x 24 + 32 | 2/2 | 48/48 | 2.3e-05 | 2.3e-05 | 699 | 52.9 |

Every greedy continuation is identical and every argmax agrees, against both references, on every
document. The paperkiln and tiny documents agree within 2.5e-6 of the row range (tp_mamba1's random
weights; the rest under 8e-7), the level of the hand-written ops' WebGPU gate. mamba-130m_rec's 1.2e-4
against ember CPU is past ir_gate.mjs's flat 1e-4 bound (`summary.json` `pass` false on that bound alone,
greedy 10/10 and top-1 240/240): the float32 conditioning of a deep real checkpoint, below the 3.3e-4 of the
hand-written Mamba-130m kernels on the same adapter (../webgpu_ir_2026-10-26/). Against numpy it is 2.3e-5.
mamba-130m_rec ran 10 of the 40 prompts (the CPU interpreter's recurrence takes 200 s per prompt on one
core) and numpy 2 (10 minutes each); prefill on the GPU path is token by token, so 54 s per prompt on two
SwiftShader cores is not a GPU figure.

### Gate 2: single nodes (`tools/webgpu_parity/recurrence_gate.mjs`)

Each case's node run as application 0 from its slot's initial state, token by token, as ember's adapter's
`run_node` (`runRecurrenceNode`); the output and the slot's tensors after the last row against the case's
expected values (numpy) and against ember CPU's (`cpu_ops.py`: `tools/ir_adapter` run_node at HEAD,
`cpu_ops.json`). IR_SPEC 4.3's max |d| / max(1, max |ref|) and the bit-equal values. Receipt `op_gate.json`.

| Op case | Class | vs numpy | bit-equal | vs ember CPU | bit-equal |
|---|---|---:|---:|---:|---:|
| linear_attention_body | ROUNDING | 9.8e-08 | 31/64 | 9.8e-08 | 30/64 |
| ssm_scan_body | EXACT | 0 | 24/24 | 0 | 24/24 |
| conv1d_causal_body | EXACT | 0 | 24/24 | 0 | 24/24 |
| selective_scan_body | ROUNDING | 1.8e-07 | 9/21 | 1.8e-07 | 10/21 |
| rnn_init_offset | ROUNDING | 1.7e-07 | 4/21 | 1.2e-07 | 4/21 |
| ema_bias_corrected | ROUNDING | 8.7e-08 | 22/28 | 8.7e-08 | 22/28 |
| ema_exact | EXACT | 0 | 28/28 | 0 | 28/28 |
| square_weight_declaration | EXACT | 0 | 18/18 | 0 | 18/18 |
| read_beyond_declaration | EXACT | 0 | 18/18 | 0 | 18/18 |
| grammar_all | ROUNDING | 1.8e-07 | 22/36 | 1.8e-07 | 22/36 |
| wide_2d | EXACT | 0 | 1300/1300 | 0 | 1300/1300 |
| shift3d_init | EXACT | 0 | 72/72 | 0 | 72/72 |
| overflow_constants | ROUNDING | 0 | 21/21 | 0 | 21/21 |
| pow_forms | ROUNDING | 6.3e-08 | 15/35 | 6.3e-08 | 15/35 |
| deep_sums | EXACT | 0 | 14/14 | 0 | 14/14 |

15/15 within the cases' 1e-5. Every EXACT body is bit-equal to both references on SwiftShader (a GPU
driver may fuse a multiply and an add, so that is a SwiftShader figure, not a guarantee of the path).

### Gate 3: sequence state (`tools/webgpu_parity/recurrence_state_check.mjs`)

Per document, 5 seeded prompt pairs A, B (6 ids for the 16-position documents, 24 otherwise), every
comparison bit for bit on the logits and on every recurrent slot tensor of every application (`readState`):
the prompt in one call against one token per call (decode = prefill), in two calls at its middle, after an
unrelated sequence (a new sequence restarts from the initial state, init bodies included), a continued turn
on the GPU's state against the whole conversation prefilled at once, and a forward that skips positions on
recurrent state, which must throw. Then keramos's state traces for the suite's recurrence documents: each
forward's logits and each step's recurrent state against the trace (numpy, the original hand-written
document's values), snapshot and restore by re-prefilling the ids held at the snapshot (IR_SPEC 6.9).
Receipt `state_check.json`.

| Document | tokenwise | chunks | reset | turns | refusal | trace | trace logits | trace state |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| pk_recurrence | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | pk_recurrence_decode | 1.2e-07 | 2.0e-07 |
| pk_recurrence | | | | | | pk_recurrence_snapshot | 1.9e-07 | 2.2e-07 |
| pk_recurrence | | | | | | pk_recurrence_chunks | 1.2e-07 | 1.7e-07 |
| pk_kimi_rec | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | pk_kimi_recurrence | 2.4e-07 | 1.7e-07 |
| pk_srd_rec | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | pk_srd_recurrence | 2.2e-07 | 1.9e-07 |
| pk_mamba_rec | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | pk_mamba_recurrence | 1.4e-07 | 4.7e-09 |
| tp_mamba1_rec | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | tp_mamba1_recurrence | 6.5e-07 | 1.1e-06 |
| kimi_tiny_rec | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | | | |
| srd_tiny_rec | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | | | |
| mamba_tiny_rec | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | | | |
| kimi_tiny_custom | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | | | |
| mamba_tiny (hand-written ssm_scan) | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | | | |

Every trace step agrees with keramos within its 1e-5 with the argmax equal (7 traces, 41 forwards, 168
state tensors).

### Gate 4: the page's chat path (`chats.py`, `tools/webgpu_parity/chat_test.mjs`)

The worker's own steps outside a browser: ember.js built at HEAD by `deploy/wasm/build.sh` prepares each
request and samples, and the GPU path the worker picks computes the logits; the same request then goes
through `ember_chat` (WebAssembly, ember's CPU interpreter). The recurrence documents are embedded in copies
of their GGUFs (`gguf_doc.py put`), since the page's WebAssembly reads the file's own document. Greedy: the
40 page prompts (60 tokens) and 20 multi-turn requests with two or three earlier turns (`chat_multiturn.jsonl`,
40 tokens; every reply re-prefills the conversation from position 0). Fixtures: mtstudio's greedy replies
of the original files, which ember CPU equals 40/40 (../families_2026-10-20/). Receipts `chat/`.

| Model | Requests | GPU path | WebGPU = WASM | WebGPU = CPU fixture | WASM = CPU fixture | Reply tokens | s per reply (SwiftShader) |
|---|---|---|---:|---:|---:|---:|---:|
| kimi_tiny_rec | 40 page prompts | ir_webgpu | 40/40 | 40/40 | 40/40 | 2188 | 2.3 |
| kimi_tiny_rec | 20 multi-turn | ir_webgpu | 20/20 | | | 800 | 2.8 |
| srd_tiny_rec | 40 page prompts | ir_webgpu | 40/40 | 40/40 | 40/40 | 2367 | 2.4 |
| srd_tiny_rec | 20 multi-turn | ir_webgpu | 20/20 | | | 800 | 3.2 |
| mamba_tiny_rec | 40 page prompts | ir_webgpu | 40/40 | 40/40 | 40/40 | 2400 | 1.5 |
| mamba_tiny_rec | 20 multi-turn | ir_webgpu | 20/20 | | | 800 | 1.9 |
| kimi_tiny_custom | 40 page prompts | ir_webgpu | 40/40 | | | 2400 | 2.0 |
| kimi_tiny_custom | 20 multi-turn | ir_webgpu | 20/20 | | | 800 | 2.6 |

The rewritten documents' replies equal mtstudio's replies of the original files (the rewrite keeps the
model, IR_SPEC 6.10.6), on WebGPU and in WebAssembly; the custom document has no fixture beyond ember's
own WebAssembly.

### Regressions

- **ir_webgpu.js against itself at HEAD** (`same_as_head.mjs`): the working tree's module and HEAD's on the
  same device and prompts, the 23 documents of the 26 Oct gate list: logits bit-equal at every position on
  all 23 (2 prompts each, 3 for the 16-position models, 1 for Pythia-160m and Mamba-130m; 856 positions) and
  the same dispatches per token. Receipt `regress/same_as_head.json`.
- **The 26 Oct gate list again** (`regress_gates.py`, ember CPU references at HEAD): greedy N/N and every
  argmax agreeing on every family, and every per-prompt row (worst |dlogit| / range, top-1, greedy) equal to
  the 26 Oct receipt's to the last digit: 21 families on all 40 prompts, Pythia-160m on its first 10 and
  Mamba-130m on its first 5 (their CPU references take 40 s a prompt; the bit-equality above covers the
  module), 7.1e-5 and 8.0e-5 on those. Same dispatches per token everywhere. Receipts `regress/<family>/`.
- **The chat list again** (`regress/chat/`): every reply and token count equal the 26 Oct receipt's: chat_sftfinal_v3 on
  `llama_webgpu.js` 40/40 against WebAssembly and the CPU fixture (636 tokens), state4's multi-turn requests
  with state blocks 40/40 (200), kimi, srd, attnres, flex_highway, mamba and gpt2 tiny 40/40 against
  WebAssembly and mtstudio (2188, 2367, 2400, 2400, 2400, 1705), swa_tiny 40/40 against WebAssembly and 11/40
  against mtstudio, as before (that fixture is not this request set's reference; WebAssembly matches it on
  the same 11).
- The tuned llama path (`llama_webgpu.js`, `gguf.js`) and WebAssembly are unchanged: no C++ file and no
  other file of `deploy/wasm/web/` changes; chat_sftfinal_v3 reaches `llama_webgpu.js` (`ember_ir_doc()` "").

### Refusals (`refusal_check.mjs`, `refusals.json`)

No form IR_SPEC 6.10 allows is refused. `irSupport` refuses a recurrence outside the block (a recurrent
slot is a block slot) and one beside depth_route (ember's CPU interpreter refuses recurrent state under
depth_route), each with its reason, and every body the grammar or the read rules refuse with ember CPU's
message (rules 31-32); the worker then logs the reason and serves the document in WebAssembly. Sums nested
deeper than `REC_SUM_NEST` (100; WGSL guarantees 127 nested blocks) would be refused too, but no body gets
there: ember's parser (and this one) stops at depth 256, which holds 50 nested sums. At load, a work or
weight buffer past the device's storage binding size throws, and the worker falls back with a console
warning.

Depth: IR_SPEC 5.19 has every engine accept bodies to depth 200 and lets a deeper one be refused. keramos's
reference refuses past 200 (39 nested sums); ember's CPU interpreter, and so this path, accepts to 256. Both
conform; a document with 40 to 50 nested sums runs in ember and is refused by numpy.

## For a GPU run

The documents with their GGUFs, embedded so vm_run.sh's family mechanism takes them as they are, in
`C:/ml_artifacts/webgpu_recurrence/` (sha256 in `staged_SHA256SUMS`; each checked on ember CPU at HEAD:
the first prompt's greedy equals the EMBER_IR reference's): pk_recurrence, pk_kimi_rec, pk_srd_rec,
pk_mamba_rec, kimi_tiny_rec, srd_tiny_rec, mamba_tiny_rec, kimi_tiny_custom. Stage them as `$ST/m/<name>.gguf`
and run `E6_EXTRA_FAMILIES="pk_recurrence pk_kimi_rec pk_srd_rec pk_mamba_rec kimi_tiny_rec srd_tiny_rec
mamba_tiny_rec kimi_tiny_custom" bash vm_run.sh refs gates_ir` (../webgpu_gpu_2026-10-27/). tp_mamba1_rec and
mamba-130m_rec read a document in place of the file's (`EMBER_IR` for the reference, `doc=` for
gate_page.html): `docs/mamba-130m_rec.ir.json` on `mamba-130m-f32.gguf`, keramos's
`conformance/recurrence/tp_mamba1.ir.json` on `conformance/models/tp_mamba1/tp_mamba1.gguf`.

## Files

- `make_docs.py`, `docs/` (the rewritten and custom documents), `make_cases.py`, `synthetic_cases.json`,
  `make_chats.py`, `chat_multiturn.jsonl`, `page_prompts.json` (the 40 page prompts), `gguf_doc.py`.
- `ref.sh`, `np_ref.py` (WSL): the references; `cpu_ops.py`: ember CPU on the op cases.
- `gates.py`, `chats.py`, `regress_gates.py` (Windows, through swiftshader_run.py); `same_as_head.mjs`,
  `refusal_check.mjs`, `state_config.json`, `same_as_head_config.json`; `tables.py` prints this README's tables.
- `tools/webgpu_parity/recurrence_gate.mjs`, `recurrence_state_check.mjs` (new);
  `deploy/wasm/web/webgpu/ir_webgpu.js` (`compileRecurrence`, `recurrenceKernels`, `recurrenceDispatches`,
  `runRecurrenceNode`, `readState`).
