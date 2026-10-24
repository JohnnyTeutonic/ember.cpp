# One conversation schema: ember's side, 24 Oct 2026

LOCAL_LLM_STACK.md section 3, points 3-5 (the conversation schema and chat templates as data), with the
agreed amendments in paperkiln's frozen header. Everything here runs on the CPU in WSL.

## What ember does

- **The header.** `chat_template.h` is paperkiln's `include/microtorch/chat_template.hpp`, vendored verbatim
  at paperkiln 7a6c7659 (first at ae126f1e; 7a6c7659's content "trim" strips what Python's str.strip()
  strips, and its conformance file runs here).
- **Resolution at load** (`chat_gguf.h`, `chat_tokenizer.h` `make_chat_setup`): `chat::resolve` on the
  GGUF's `paperkiln.chat_template`, its tokens, `tokenizer.ggml.model`, `tokenizer.chat_template` and
  `general.architecture`. The log and `/health` (`chat_template` {name, version}, `chat_template_why`)
  say what was picked and why. A file that resolves to nothing serves raw completion, as before. A
  declared record ember cannot serve (an unknown field, a stop string that is not one token) also leaves
  chat mode off, and the reason is logged.
- **Rendering.** `ChatFormat` is the record plus its stop ids. Every prompt is `chat::render` of the
  conversation, and nothing about a format is hard-coded. One request path (`chat_request.h`) serves
  `/chat`, `/v1/chat/completions` and the WebAssembly `ember_chat` / `ember_prepare`. It parses the
  request into a conversation, applies the context rule (whole exchanges dropped oldest first, the
  system message pinned and folded into the first kept exchange) and renders it.
- **Vocabularies.** paperkiln "bpe" and "gpt2"/"llama-bpe" as before, and Qwen's "gpt2"/"qwen2" as
  before. New: SentencePiece ("llama", Gemma 3) through `spm_tokenizer.h`, a port of llama.cpp b6000's
  SPM session. It cuts out the special tokens (control, user-defined and unknown types, longest first),
  merges score-ordered bigrams and falls back to bytes.
- **/chat.** It takes `system`, or a leading `{"role": "system"}` history entry. Giving both, or a system
  entry anywhere else, is a 400. An empty system message counts as none. `thinking` is a boolean. A
  role-form history pairs strictly, as mtstudio's does (paperkiln 7a6c7659): after the optional system
  entry, user and assistant alternate, starting with user and ending with assistant. An assistant
  message without a user before it, two user messages in a row, a history ending on a user message and
  any other role are each a 400 naming the rule, in mtstudio's words. Pair-form entries are unchanged.
  Template refusals (a system message on a template without one, content holding the template's own
  markers) are 400s with the header's message. In public mode a leading system entry survives the
  history trim.
- **OpenAI endpoint.** Messages map one to one:
  - text parts are joined, and any other part is a 400;
  - system and developer messages before the first user message join with "\n\n", and an empty one is
    dropped;
  - a later system or developer message, or a tool message, is a 400 that names the rule;
  - user and assistant messages must alternate, and the last must be the user's;
  - `thinking` and `chat_template_kwargs.enable_thinking` are extension fields.
- **WebAssembly.** `ember_prepare` returns the record's `stop_ids`, and `ember_sample` / `ember_finish`
  and the worker stop on them (previously ids 1-3).
- **Word-level models** resolve to `paperkiln/legacy-word`. The server's word-level dialogue wrap renders
  it ("user: X assistant:", byte-identical to the old string).
- **State tokens** (`state_tokens.h`). The insertion rule stays over paperkiln/1's markers. A state-token
  model whose template does not write exactly those markers is refused at load.

## Results

| Gate | Result |
|---|---|
| (a) conformance file at paperkiln 7a6c7659, vendored header (`conformance_cases.json`) | 17 cases rendered and 15 refused as in Python; 5 records; Jinja (3) and Modelfile identical; 2 sha256 vectors; 7/7 resolutions; 30 legacy prompts, also through `/chat` in both history forms |
| (a) the 32 cases through ember's request paths, offline | 59 path results: 58 as the Python twin, 1 by design (below); 4 cases are not requests (training conversations, a loss flag) |
| (a) the same requests on live servers (`conformance_live.json`) | 57/57 match the offline path (prompt ids against `bpe.py` / HF encodings of the text, 400 where refused); legacy-word offline only |
| strict role-form pairing (`history_strict.json`; offline in `conformance_cases.json` `history_strict`) | live on chat_sftfinal_v3: role history = pair form (reply, prompt ids, ids, stop) 3/3 (plain, with a leading system entry, two exchanges); the four violations and a late system entry refused naming the rule 5/5; offline 4/4 and role = pair |
| G4 declared vs absent (`g4_declared.json`) | 40/40 prompt ids, reply ids, stop id, reply, stop reason, tokens, identical across 4 servers: this change and 69d7ca2, each on chat_sftfinal_v3 and on its copy with both template keys (tensor data byte-identical) |
| G5 ember vs mtstudio, chat_sftfinal_v3, 8k "bpe" (`g5_mtstudio_sftfinal_v3.json`) | frozen 40: prompt ids, replies, token counts, stop reasons, history_turns_used 40/40; system set 60: all 60/60 |
| G5 ember vs mtstudio, arxiv_2302_13971, 16k "llama-bpe" (`g5_mtstudio_arxiv_2302_13971.json`) | frozen 40: all 40/40; system set 60: all 60/60 (4 replies reach 60 tokens, same in both) |
| G5 third-party ids vs HF `apply_chat_template` (`g5_thirdparty.json`) | Qwen2.5 409/409, Qwen3 909/909, Gemma 3 409/409, text and ids (the 4 Unicode-edge cases included: Gemma 4/4 since 7a6c7659, 0/4 under ae126f1e's ASCII trim) |
| state-token refusal (`state_refusal.json`) | state4 declaring paperkiln/1: loads, 5/5 requests identical to the original; "custom/1" with v1's markers: refused at load; chatml-qwen3: refused at load |
| (d) gate scripts rerun in place | recursion_gate, mor_gate, mor_refusals, attnres_gate, state_tokens_gate (after make_state_gguf.py), state4_gate: every receipt rewritten byte-identical |
| (d) parity receipts replayed (`regress_parity.json`) | 10/10 byte-identical (chat_2026-10-18 n40; gate_llama3, pk_base_v1 f32 and Q8_0, both arXiv-chain models with their demos; repeat p1.1 and p1.3) |
| (d) ir_logits 69d7ca2 vs this change (`regress_bits.json`) | 16/16 bit-equal (bit_regression.py's list plus ar3, state4, the recursion and MoR gate models) |
| WebAssembly `ember_chat`, this change built by `deploy/wasm/build.sh` (`wasm_gate.sh`; one em++ process, about 2 min on one core) | chat_sftfinal_v3 against mtstudio's replies: frozen 40 and system set 60, reply, token count and stop reason 40/40 and 60/60 (`wasm_frozen40.json`, `wasm_system_set_v1.json`). The WebGPU path (`ember_prepare` / `ember_sample` with `stop_ids`, worker.js) needs a browser and is not run here |

The one by-design path difference: *two system messages* on the OpenAI path are joined with "\n\n" (the
agreed amendment); the header refuses a conversation with two. (Under the first, lenient `/chat` pairing,
*the assistant first* and *two user messages in a row* also rendered on `/chat`; both are refused since
strict pairing, as in Python.)

### G5 details

- **mtstudio.** `mtstudio chat` at paperkiln 28d53ca7 (`build_mtstudio.sh`, CPU, full recompute) on the
  run's own spec. Its prompt ids come from `mtstudio sample-set` on each conversation as `/chat` kept it
  (its own `history_turns_used`). That is its `chat::render` and its tokeniser; `/chat` returns no ids.
- **System set** (`system_set_v1.jsonl`, sha256 19d1c6bd..., `make_system_set.py`), built from the frozen
  prompts (tinychat_hf_prompts_v1, a906b25a...) and four system texts:
  - 40 with `system`;
  - 8 with a leading system entry and one exchange;
  - 8 with `system` and two exchanges;
  - 4 with six exchanges, where the 256-token context drops exchanges: 4 and 5 kept, the system message
    folded into the first kept one.
- **Third party.** ember's side is `chat_template_check render`: the session's load path and the
  request path, without a model. HF's side is transformers 4.57.0 tokenizers on the models' HF
  directories. Resolution:
  - Qwen2.5-0.5B-Instruct f16 (8e0ae260...) and Qwen3-0.6B Q8_0 (9465e63a...): by
    `tokenizer.chat_template` sha256;
  - Gemma-3-270m-it f32 (0a174d42..., no `tokenizer.chat_template`): by markers and architecture.

  Sets:
  - the 200 chat_prompts_v1 prompts (7272020c...);
  - 25 tokeniser hard cases;
  - 40 with a system message (as `system` and as a history entry);
  - 40 with two exchanges (half of Qwen3's with `<think>` reasoning in the replies, which strip_think
    removes);
  - 20 with a system message and two exchanges;
  - 20 with whitespace around every message (Gemma's trim);
  - 40 through the OpenAI mapping (developer messages included);
  - Qwen3's three sets again with thinking on and off (enable_thinking);
  - 20 with thinking false on Qwen2.5 and Gemma, which their templates ignore.
- **Unicode edge whitespace.** 4 messages with Unicode whitespace at their edges. HF's Jinja `trim` strips
  what Python's str.strip() strips. Under the header at ae126f1e (ASCII trim) Gemma gave 0/4 and the set
  was reported ungated; at 7a6c7659 the record strips the same, Gemma gives 4/4 and the set is gated.

## Files

- Drivers: `build_base.sh` (ember 69d7ca2, `~/ember_69d7ca2`), `build_cpu.sh`, `build_mtstudio.sh`
  (`~/pk_cpu_ct`), `common.py`, `add_template_keys.py` (a GGUF with string keys added, tensor data copied
  byte for byte and checked), `conformance.py`, `g4_declared.py`, `make_system_set.py`, `g5_mtstudio.py`,
  `g5_thirdparty.py`, `state_refusal.py`, `history_strict.py`, `regress_inplace.sh`, `regress.py`,
  `wasm_gate.sh`, `baseline_check.sh`, and `rerun_all.sh` (the reruns after the 7a6c7659 re-vendor and
  strict pairing: G5 third party, the in-place gates, parity replays from scratch, bits, WebAssembly; G4,
  the conformance run and `history_strict.py` ran beside it, all as in the table).
- Receipts: `conformance_cases.json`, `conformance_live.json`, `g4_declared.json` + `g4_rows.jsonl`,
  `g5_mtstudio_*.json` + `_rows.jsonl`, `g5_thirdparty.json`, `state_refusal.json`,
  `regress_parity.json`, `regress_bits.json`, `wasm_*.json`.

Commands, each from the workspace root inside WSL under
`scripts/cpu_budget.py run --session ember --cores 4` (1 for the tokenizer-only `g5_thirdparty.py` and
the WebAssembly build): `python3 ember.cpp/experiments/chat_template_2026-10-24/<driver>.py [args]`, with
`g5_mtstudio.py TAG GGUF RUN_DIR NAME` (`sftfinal_v3 .../chat_sftfinal_v3.gguf .../sftfinal chat_sftfinal`
and `arxiv_2302_13971 .../arxiv_2302_13971.gguf .../sft_2302_13971 arxiv_2302_13971`).
