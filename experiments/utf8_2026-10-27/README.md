# Valid UTF-8 on every text path (27 Oct 2026)

A reply decoded from BPE ids is bytes. When generation stops inside a multi-byte character (max_new_tokens, or a
byte token that forms no character), nlohmann::json's `dump()` throws type_error 316. pk-bench2's IF-tiny job
(paperkiln `experiments/benchmarks_2026-10-26`) stopped at its prompt 375 on pick_e1_60m_L32 for that reason:
greedy, max_new_tokens 64, the reply ending on the lone lead byte 0xD7 of a Hebrew letter. paperkiln's fix,
e57c0fb on branch `iftiny-utf8` (fe01b35 + `utf8_valid`), is cherry-picked onto main and extended to every
endpoint that returns text.

## The change

`utf8_text.h`:
- `utf8_valid`: each ill-formed subsequence becomes one U+FFFD per maximal subpart, as Python's
  `bytes.decode("utf-8", "replace")` (paperkiln's code, now on a shared `utf8_part` scanner).
- `ReplyStream`: a streamed reply. A character a token boundary cut (at most 3 bytes) is held until the next
  token completes it; bytes that can never become valid are replaced at once; at the end of the stream a
  character still cut becomes U+FFFD. Leading whitespace is skipped and trailing whitespace held until something
  else follows, as `generate()` strips both: the pieces join to the plain reply byte for byte.
- `dump_valid`: `dump()`, or, when a string is not valid UTF-8, the dump with `utf8_valid` on every string and
  key. The same bytes as `dump()` for valid text.

Where it applies:
- `server.cpp`: `/chat` (plain; streamed `{"token"}` pieces and the final object; the legacy non-chat path),
  `/v1/chat/completions` (plain; streamed deltas, now with the end flush the stream lacked), and every JSON body
  and server-sent event (`json_out`), error messages and `/health`'s paths included. The log lines that echo
  model or request text are valid UTF-8 too (also `api.cpp`'s "Generated response").
- `deploy/wasm/ember_wasm.cpp`: `ember_chat`'s and `ember_finish`'s reply, and every returned JSON. The worker
  (`web/worker.js`) only parses that JSON, so it needs no change.
- `bindings.cpp`: `generate` and `generate_batch` return valid UTF-8 (pybind11 decodes a `std::string` strictly
  and raised UnicodeDecodeError).
- `/completion` and `/v1/completions` do not exist in ember. `/score` returns ids, numbers and the request keys
  only (no decoded text).
- Generated ids, `return_ids` and the sampled tokens are unchanged.

## Before: main at bace837 (`receipts/before_probe_bace837.json`, `server_check_bace837.json`)

The reported request (IF-tiny's body: temperature 1, top_k 1, top_p 1, seed 7), each endpoint on a fresh server,
at max_new_tokens 12 (the first cut) and 64 (pk-bench2's):
- `/chat`: HTTP 500 (`incomplete UTF-8 string; last byte: 0xD7`);
- `/chat` streamed: the server process dies (the final object's `dump()` throws inside the stream provider);
- `/v1/chat/completions`: HTTP 500 with an empty body;
- `/v1/chat/completions` streamed: HTTP 200, the cut character silently missing from the deltas.

## After (`receipts/`)

- `unit.txt`: `tools/utf8_text_test` 27/27 (paperkiln's 18 `utf8_valid` cases, 8 `ReplyStream` cases, 1
  `dump_valid` case); `tools/utf8_text_check.py` 20,000/20,000 random byte strings equal to Python's
  decode(errors="replace"), and 20,000/20,000 random streamed replies (whitespace at the ends and inside, tokens
  of 0 to 4 bytes) equal after every token to a reference built on Python's decoder (the bytes so far less a
  trailing character some continuation completes, found by search and checked by Python's strict decoder), and
  joined equal to the plain reply.
- `server_check.json` (`tools/utf8_server_check.py`, CPU build, gpu_layers 0, pick_e1_60m_L32): the reported
  request at every max_new_tokens 1 to 80 on the four text endpoints, 320 requests, all HTTP 200, every
  server-sent event strict UTF-8 JSON; per k, the `/chat` reply, the joined `/chat` pieces, the stream's final
  reply, the `/v1` content and the joined `/v1` deltas equal byte for byte. 23 ks end inside a character
  (12, 14, ..., 64, ..., 77: the reply ends in U+FFFD, k = 64 as pk-bench2's), and at the 23 ks after them the
  next token completes that character across two streamed tokens (k = 13: the pieces `' "'`, then `'ש'`). Ids
  at every k are the first k of k = 80's. The Python bindings' `generate` equals the `/chat` reply at all 23
  cut ks.
- `wasm_chat.json` (`tools/wasm_chat_check.mjs`, the WebAssembly build with EMBER_WASM_MAX_MEMORY=4gb; the
  60M model does not load in the default 1 GB): the 37 cut and split ks within the page's 64-token cap, prompt
  ids, counts, replies, tokens and stop reasons 37/37 equal to the native server's. `wasm_finish.txt`
  (`wasm_finish_check.mjs`, rows from `wasm_ref.py`): the WebGPU page's path without a GPU, ember_prepare,
  ember_sample on one-hot logits for the native ids, ember_finish(-1): 37/37 replies equal to the native
  server's.
- `cpu_regress.txt` (`../cuda_state_2026-10-25/cpu_regress.sh` against the committed receipts as
  `git show HEAD:<path>` gives them): recursion, MoR, MoR refusals, attnres, state4, regress_parity and G5 third
  party byte-identical. state_tokens_gate equals bace837's own CPU build's receipt
  (`../draft_2026-10-27/receipts/cpu/state_tokens_gate_cpu_new.json`) byte for byte; both differ from the
  committed file in the numpy-reference distances only, as before. regress_bits (old side: bace837's CPU
  `ir_logits`): pass, the receipt differing from the committed one only in its `new` field, the binary's path.
  The IR adapter self-test: 189/189 checks pass, 3072 requests.

## Files

- `before_probe.py`: each endpoint alone against a fresh server (a build without the fix).
- `wasm_ref.py`, `wasm_finish_check.mjs`: the WebAssembly rows and the ember_finish check.
- The checks themselves: `tools/utf8_text_test.cpp`, `tools/utf8_text_check.py`, `tools/utf8_server_check.py`.
