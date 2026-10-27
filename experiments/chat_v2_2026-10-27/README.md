# Conversation schema v2 in ember (27 Oct 2026)

ember's side of paperkiln's conversation schema v2 (`LOCAL_LLM_WORK.md` D3; `LOCAL_LLM_STACK.md` section 3;
the record `paperkiln/2` of paperkiln 22f8fe6d's `include/microtorch/chat_template.hpp`): a system turn of its
own, tool definitions in the system turn, tool calls after `<|tool_call|>` and tool results behind `<|tool|>`.
The record is declared by `paperkiln.chat_template`, never inferred; behaviour follows the resolved record's
fields (`Template::has_tool`), and every other record's paths are v1's, unchanged. CPU only.

## The change

- `chat_template.h`: paperkiln 22f8fe6d's header, vendored verbatim (a five-line header naming the commit,
  then the file unchanged; the v1 parts are byte-identical to 33c45899's).
- Load (`chat_tokenizer.h`, `make_chat_setup`): a record with tool turns runs `chat::check_control_tokens`
  and then checks that the chat tokenizer encodes each reserved string as its one control id; either failure
  refuses the file at load (the server exits, the WebAssembly `ember_load` returns -1). `ChatFormat` gains
  `tool_call_id` (16385 on the 16k vocabulary; -1 under every other record). The stops are the record's:
  eos `<|endoftext|>`, turn `<|user|>`, `<|assistant|>`, `<|system|>`, `<|tool|>`; `<|tool_call|>` is not one.
- Requests (`chat_request.h`): `chat_conversation` and `openai_conversation` take the model's record; one
  with tool turns builds the whole message list (`Conversation::general`) and the tool definitions, read in
  the request's member order (`ordered_tools`: the body parsed as `ordered_json`, so a definition renders as
  Python's `json.dumps` text of what the client sent).
  - `/chat`: `tools`; role-form history entries with `tool_calls` (OpenAI form) and `tool_call_id`, the tool
    role, pair-form entries as before; `user_input` may be left out when the history ends on a tool message;
    `system` or a leading system entry as under v1 (empty means none).
  - `/v1/chat/completions`: system and developer messages join as under v1; tool messages and assistant
    `tool_calls` map one to one (an assistant's `content` may be null beside calls, a tool's may be text
    parts); `tools` passes through; `tool_choice` absent, null or `"auto"`; the last message is the user's or
    a tool's.
  - The record's structure rules (`chat::check_structure`) decide the order of user, assistant and tool
    messages; refusals are HTTP 400 naming the rule.
  - Under every other record a non-empty `tools`, a non-empty `tool_calls` and the tool role are HTTP 400 on
    both endpoints (`"tools": []` and `"tool_calls": null` are none).
- Context (`build_general_prompt`): exchanges are `chat::exchanges` (a user message through everything before
  the next one). The whole conversation renders first (so the record's refusals apply to all of it); then,
  while more than the newest exchange remains and prompt + max_new_tokens > ctx, the oldest whole exchange
  goes. The system turn with its tool definitions is pinned. When the system turn and the newest exchange
  are >= ctx tokens: HTTP 400 "the system turn and the newest exchange are N tokens (the system turn S); the
  model's context is C (an exchange is never cut)" ("the newest exchange is N tokens; ..." without a system
  turn). `history_turns_used` counts the past exchanges kept: on a user/assistant-only conversation without a
  system message the prompts, counts and turns are v1's at every context (checked at ctx 4-63).
- Replies (`chat_reply.h`, new): split on the generated ids with `chat::parse_reply`, never on decoded text.
  A reply that parses with at least one call is the content before the first `<|tool_call|>` (stripped of
  leading and trailing whitespace as every reply is) plus the calls (ids `call_0`, `call_1`, ...; name and
  arguments verbatim, made valid UTF-8); otherwise the whole decoded reply is the content, every token as
  its text (a spare token, or an unparsed `<|tool_call|>`, appears as itself). `/chat` adds `tool_calls` in
  OpenAI's form; `stop_reason` is unchanged. The OpenAI endpoint adds `message.tool_calls` and
  `finish_reason` "tool_calls" ("length" when the length limit ended the reply, calls or not).
- Streaming (`ReplyBuilder`, on `ReplyStream`): content pieces as generated (a cut character and trailing
  whitespace held, as since 7526469) until the first `<|tool_call|>` id, then nothing until the end; there the
  rest of the content (the content before the call, or the whole reply when it does not parse), then on the
  OpenAI endpoint one whole `delta.tool_calls` chunk per call (`index`, `id`, `type`, full name and arguments)
  before the finish chunk, and on `/chat` the final event with `tool_calls`.
- Under a v1 record the reply and stream are the code paths of before: the reply is `generate()`'s string,
  the stream the same `ReplyStream` calls.
- WebAssembly (`deploy/wasm/ember_wasm.cpp`): `ember_chat` and `ember_prepare` take the `/chat` v2 fields
  through the same parser and context rule (the page's caps: 600 bytes for the new message and the system
  message, the newest 16 past exchanges); `ember_chat` and `ember_finish` add `tool_calls`.
- Python bindings (`bindings.cpp`): `generate` and `generate_batch` take a rendered prompt string, not a
  conversation, so they have no v2 fields to take; nothing changed there. The browser pages (`www/`,
  `deploy/wasm/web/`) send plain user/assistant histories and show the reply text; they pass the reply object
  through and have no tool UI.
- Test hook: `TinyLlamaSession::set_scripted_replies` (the server's `EMBER_TEST_SCRIPTED_REPLIES=FILE`, a JSON
  list of id lists; refused under `EMBER_PUBLIC=1`): each chat request takes the next list in place of
  sampling, through the token callback, the stop rule and the reply decoding of sampled ids. No v2 model
  makes tool calls yet; this is how the HTTP check drives the handlers' tool-call paths.

## Receipts

All on the CPU: a Release build of ember bb472bf plus this change on ext4 (`~/cv2/ember/build_cpu`), and the
WebAssembly build of the same sources (`deploy/wasm/build.sh`, EMBER_WASM_MAX_MEMORY=4gb).

- `toy_models.txt`: the toy files from paperkiln's `gate_llama3.gguf` (sha256 f365148a): `toy_v2.gguf` (ids
  16384-16391 the control tokens, token type 3; 36b6fdc9), `toy_v2_nocontrol.gguf`, `toy_v2_badtype.gguf`. The
  appended tokens and types equal those of paperkiln's own `chat_v2_vocab.py` output
  (`/mnt/c/ml_artifacts/chat_v2/vocab_llama3_bpe_v2.gguf`, d44f27b2), and its first 16,384 tokens equal the base's.
- `chat_template_cases.json` (`chat_template_check cases`, paperkiln 22f8fe6d's `tests/chat_template_cases.json`):
  35 cases rendered and 39 refused as in Python, 6 records, Jinja and Modelfile identical (paperkiln/1, its
  `none` mode, legacy-word, the v1 Modelfile, the v2 Jinja and Modelfile), 7 resolutions, 12 parse_reply cases,
  check_control_tokens, 30 legacy prompts, 2 sha256: 0 failures. Through ember's request paths (`/chat` with the
  system message in either place, OpenAI messages, tools in the file's member order, a case ending on a tool
  message sent without user_input): 138 same, 8 not a request (training conversations, loss flags), 4 differ by
  the contract's rules: v1's "two system messages" on the OpenAI path (joined, as since chat_template_2026-10-24)
  and "v2 empty system message" on all three paths (an empty system message is none, point 4; the header renders
  an empty system turn).
- `chat_v2_test.json` (`tools/chat_v2_test`): 73 checks, 0 failed. Load: paperkiln/2 declared, tool_call_id
  16385, the stops, each control string one id each way, both fixtures refused, the v1 base unchanged. 25
  requests on both endpoints under paperkiln/2 and paperkiln/1 (rendered text or the refusal's rule), the
  control strings single ids in the prompt. The context rule at six contexts around the three exchanges'
  token counts (one token short drops the oldest exchange with its call; the newest alone; refused at
  system + newest), and v1-equal prompts, counts and turns at every ctx 4-63 for a user/assistant-only
  conversation. 14 replies on the toy vocabulary (calls after content, alone, two, arguments keeping their
  whitespace, unparsed calls, spare tokens in content and in arguments, a character cut by the call, split
  before it, a four-byte character over four tokens, arguments ending inside a character, no ids): content,
  calls, streamed = plain, every piece valid UTF-8, nothing sent after the call id before the end. Fuzz: 20,000
  random replies over text, whitespace, byte, spare and call tokens, streamed = plain 20,000/20,000 (1,547 with
  calls, 10,616 with a call id that does not parse), and under v1 (no call id) 20,000/20,000 equal to the v1
  reply.
- `server_check.json` (`tools/chat_v2_server_check.py`, HTTP, CPU): 0 failures. Load: both fixtures refused
  (exit 1, "chat template paperkiln/2 needs '<|system|>' as a control token of the vocabulary: the file is
  refused"). Render: 6 `/chat` requests (tools, a call answered, user_input left out, histories past the
  256-token context: 2 and 1 of 6 past exchanges kept) with prompt ids, counts and turns equal to `chat_template_check
  render`, and 3 OpenAI requests' usage.prompt_tokens equal to it. 11 refusals, HTTP 400 with the rule's words
  (the context refusal naming its counts on both endpoints). stream = plain on 4 `/chat` and 3 OpenAI greedy
  replies of the toy. Scripted (the test hook), each through `/chat` plain and streamed and the OpenAI endpoint
  plain and streamed: content then a call, two calls, an unparsed call (content "Sure <|tool_call|>get_weather"),
  a spare token as content, a turn stop (`<|tool|>`) after a call, the length limit inside a call's arguments
  (finish_reason "length", the partial arguments returned); content, tool_calls, stop_reason, finish_reason and
  ids as the rules say, streamed = plain, one whole delta per call. v1 server (gate_llama3): tools, tool_calls
  and tool messages HTTP 400 on both endpoints; `"tools": []` gives the reply without tools.
- `wasm_v2.json` (`tools/wasm_v2_check.mjs` on `wasm_v2_ref.jsonl`, the HTTP check's rows): ember_prepare's
  prompt ids, counts and turns and ember_chat's replies equal to the native server's on the 6 `/chat` requests;
  the 6 scripted replies through ember_prepare, one-hot ember_sample and ember_finish give the server's content,
  tool_calls and stop reasons; ember_load refuses both fixtures (-1).
- `v1_regression.txt`: the committed CPU receipts rerun unchanged, byte-identical: regress_parity (all 10
  receipts: chat_2026-10-18's chat40 greedy, the pk_base_v1 parity and demo receipts, the 1.1 and 1.3 penalty
  receipts, pk_base_v1 F32 and Q8_0) and g5_thirdparty; the IR adapter self-test 189/189 checks, 3,072
  requests (unchanged); `tools/utf8_server_check.py` (320 requests) equal record for record to
  `utf8_2026-10-27/receipts/server_check.json`; WebAssembly frozen 40 (40/40) and system set (60/60) equal to
  `chat_template_2026-10-24`'s receipts but for the timing.
- `gate_dryrun_ember_vs_ember.json`: `gate_v2.py`'s mechanics on the toy, ember against a second ember server
  in mtstudio's place, 4/4 on every count. Not the gate.

## The gate on paperkiln's gate_v2 (no tool calls): passed

Inputs: `/mnt/c/ml_artifacts/chat_v2/gate_v2/run/gate_v2.gguf` (sha256 598b2350; gate_llama3 with the 8 v2 rows,
declared paperkiln/2, context 512 in `llama.context_length` and the IR's `model.ctx`; no v2 training, so no tool
calls), `microtorch/corpus/eval/chat_v2_prompts_v1.jsonl` (40 rows: 10 each system, history, tool_user,
after_tool; sha256 254e8ee4), `mtstudio chat` at paperkiln 7673859f (`~/d3_build/mtstudio`, CPU, FP32,
OMP_NUM_THREADS=1) on the run's spec (T 512), and paperkiln's reference replies `mtstudio_greedy_v2.jsonl`
(sha256 b74e84d2, mtstudio 7673859f). Each request is the row's request plus temperature 1, top_k 1, top_p 1,
seed 7, max_new_tokens 60, return_ids. Both engines: the content before a call stripped of " \t\r\n" at both ends,
names and arguments verbatim; `/chat` `stop_reason` "eos", "turn" or "length" with calls; `history_turns_used` the
past exchanges kept; a leading tool message refused; `stop_id` -1 at a length stop (section 3, point 4).

    python3 gate_v2.py --ember ~/cv2/ember/build_cpu/tinyllama_server --check ~/cv2/ember/build_cpu/chat_template_check \
        --gguf GATE.gguf --prompts chat_v2_prompts_v1.jsonl --mts "OMP_NUM_THREADS=1 MTSTUDIO chat SPEC --port {port}" \
        --ref mtstudio_greedy_v2.jsonl --out receipts/gate_v2.json --n 40 --max-new 60 --seed 7

Result (`gate_v2.json`; ember b49eb5c's CPU build): **40/40 on every count.**
- Against the live mtstudio: HTTP status, reply, tool_calls, stop_reason, tokens, history_turns_used, prompt_ids,
  ids and stop_id equal on all 40 rows (25 eos, 8 turn, 6 length with `stop_id` -1; history_turns_used 0 on 28, 1
  on 10, 2 on 1). after_tool/5 is HTTP 400 in both (588 tokens against 512).
- Against paperkiln's file: all 40 rows equal, history_turns_used included; the live mtstudio equals the file on
  all 40 (`gate_v2_ref.json`).
- The rendering: ember's request path renders the whole conversation as every row's "prompt", 40/40, and the
  served prompt (the context rule applied at 512) is ember's prompt_ids, 40/40. At after_tool/9 the whole
  conversation is 534 tokens, and both engines drop the oldest exchange (485 tokens, history_turns_used 0)
  (`gate_v2_prompts.json`: the whole-conversation text 40/40, and its ids equal mtstudio's on the 38 other rows
  mtstudio serves).
- stream = plain on both endpoints, 40/40: `/chat` streamed (joined pieces, the final object) equals `/chat`
  plain, and the same conversations as OpenAI messages give the same content and tool_calls plain and streamed
  with finish_reason as the rule says; the refused row is refused on both endpoints, streamed or not.

## The tool-calling gate on pk_sft_v2: passed

Inputs: `/mnt/c/ml_artifacts/chat_v2/run/sft/pk_sft_v2.gguf` (sha256 2363aa42; F32, llama d768 L12 H12, 97.5M
parameters, paperkiln/2, 16,392 tokens, context 512) and its Q8_0 file (e88a77ce); the same 40 prompts and request
fields. The reference is paperkiln's `/mnt/c/ml_artifacts/chat_v2/run/mtstudio_greedy_v2_pksft2.jsonl` (sha256
79105a05; mtstudio 7673859f, CPU, full recompute, FP32), against which ember's replies are compared field by field
(no second live mtstudio: no row differed, so none was rerun with `gate_v2.py --rows`).

    python3 gate_v2.py ... --gguf pk_sft_v2.gguf --mts none --out receipts/gate_pksft2.json --seed 7
    python3 ref_compare.py receipts/gate_pksft2.json mtstudio_greedy_v2_pksft2.jsonl receipts/gate_pksft2_ref.json

ember (`gate_pksft2.json`, ember b49eb5c's CPU build):
- Tool calls on all 10 tool_user rows (one call each, content "", stop "eos"; search_hotel, generate_random_name,
  calculate_tip, get_movie_details, search_recipes, calculate_profit_margin, convert_currency, calculate_tax,
  calculate_bmi, calculate_fibonacci), none elsewhere; 33 eos and 6 length stops; after_tool/5 HTTP 400.
- stream = plain 40/40 on both endpoints: `/chat` streamed equals plain (pieces and the final object's
  tool_calls), the OpenAI endpoint gives the same content and tool_calls plain and streamed, each call one whole
  `delta.tool_calls` chunk (one call, its index, full name and arguments), finish_reason "tool_calls" on the 10
  calling rows, "length" or "stop" elsewhere as the rule says.
- Every row's "prompt" rendered exactly, and the served prompt ids equal the render, 40/40.
- Against the reference (`gate_pksft2_ref.json`): **every field equal on all 40 rows**: reply, tool_calls,
  stop_reason, tokens, history_turns_used, prompt_ids, ids and stop_id on the 39 served rows, and after_tool/5 HTTP
  400 in both. The file's tool calls are on tool_user/0-9 only, as ember's.

Q8_0 against F32, report only (`gate_pksft2_q8_0.json`, `q8_0_vs_f32.json`): the same ids on 35 of 40 rows, the
same first token on 39, the same tool_calls on 38 (the same 10 rows call; tool_user/0 and tool_user/4 differ late
in the arguments, from token 45 and 36 of 48), the same stop reason on 40; stream = plain 40/40 on both endpoints.

## Files

- `make_toy_v2.py`: the toy v2 GGUFs (gate_llama3 relabelled with the record and the control tokens, the two
  refusal fixtures), checked against paperkiln's `chat_v2_vocab.py` vocabulary.
- `gate_v2.py`: the final gate (ember against `mtstudio chat` on the v2 gate GGUF and prompt set, greedy, then
  stream = plain on both endpoints). `ref_compare.py`: ember and live mtstudio against paperkiln's reference file.
  `prompt_check.py`: every row's "prompt" against ember's rendering, and its ids against mtstudio's.
  `q8_agreement.py`: a Q8_0 file's greedy replies against the F32 file's (report only).
- The checks: `tools/chat_template_check.cpp` (cases), `tools/chat_v2_test.cpp` (unit),
  `tools/chat_v2_server_check.py` (HTTP), `tools/wasm_v2_check.mjs` (WebAssembly).
