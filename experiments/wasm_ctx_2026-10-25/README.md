# A context cap for the browser page, 25 Oct 2026

A third-party file declares its whole trained context: Qwen2.5-0.5B 32,768 positions (0.8 GB of f32
KV), Qwen3-0.6B 40,960 (9.4 GB, more than any 4 GB heap holds; `../wasm_families_2026-10-25/`). The
browser page (`deploy/wasm/`) caps the interpreter's KV state, as the native server does with
`EMBER_IR_CTX`.

## The change (ember 3b86e08)

- `ember_set_ctx(n)` before `ember_load` sets the cap. `ember_load` passes it to the loader as
  `EMBER_IR_CTX`, so the interpreter's KV state and the context the request rules use (`model.cpp`,
  `graph_ir.cpp`) are bounded exactly as on the native server. A file with a smaller context keeps its
  own. `n <= 0` lifts the cap. `ember_ctx()` gives the loaded model's context after the cap.
- The default is `EMBER_WASM_CTX`, set by `deploy/wasm/build.sh`: 2048, or 0 for no cap. At run time the
  page's `?ctx=N` sets it (`worker.js`, which also reports `ctx` when ready).
- Prompts past the cap follow `/chat`'s context rule: the oldest whole exchanges are dropped, the system
  message is kept.
- The cap applies to files the IR interpreter runs, which includes every third-party family. A
  llama-architecture file on the tuned path keeps its declared context, as natively.

## Gate (`run.sh`)

- Builds: `deploy/wasm/build.sh` at its defaults (1gb heap, cap 2048) and with
  `EMBER_WASM_MAX_MEMORY=4gb`, from ember f4fc5e0 with this change.
- Requests (`make_requests.py`):
  - 10 single-turn requests: distinct prompts of paperkiln's chat_prompts_v1, two with a system message,
    32 new tokens;
  - 3 long requests: a system message, 10 past exchanges and a new message, 24 new tokens.
  All greedy.
- Reference: ember's native CPU server (f4fc5e0, `collect_native.py`) with `EMBER_IR_CTX=2048`, and
  `EMBER_IR_CTX=256` for the long requests.
- Page side: `tools/wasm_chat_check.mjs` in node (emsdk's 24.19.0) on the 4gb build:
  - at its default cap for the 10 requests, and with `ember_set_ctx(256)` (the page's `?ctx=256`) for
    the long ones;
  - per request, `ember_prepare`'s prompt ids, token count and turns used, then `ember_chat`'s reply,
    token count, stop reason, token count and turns used, each against the server's.

## Results

| check | Qwen3-0.6B Q8_0 | Qwen2.5-0.5B-Instruct f16 |
|---|---|---|
| loads in the 4gb build at the default cap: context, WebAssembly memory after load | 2048; 1,952 MiB | 2048; 2,614 MiB |
| 10 requests: replies identical to the native server's (`EMBER_IR_CTX=2048`) | 10/10 | 10/10 |
| token counts, stop reasons, prompt token counts and turns used | 10/10 each | 10/10 each |
| `ember_prepare`'s prompt ids, token counts and turns used | 10/10 | 10/10 |
| 3 long requests at a cap of 256 (`ember_set_ctx(256)` / `EMBER_IR_CTX=256`): everything above | 3/3; turns kept 1, 1, 0 of 10, system message kept | 3/3; the same |
| reply time in node, one core | 22 s (32 tokens) | 26 s |

Without the cap (`ember_set_ctx(0)`, the file's 40,960 positions) Qwen3-0.6B does not load in the 4gb
build: `ember_load: std::bad_alloc` (`uncapped_qwen3.txt`).

The paperkiln chat models' WebAssembly gates on the default build (1gb, cap 2048) are unchanged
(`paperkiln_wasm.txt`). chat_sftfinal_v3 (context 256, below the cap) gives replies, token counts and
stop reasons equal to mtstudio's 40/40 on the frozen prompts and 60/60 on the system set. Both receipts
equal the committed `../chat_template_2026-10-24/wasm_*.json`, timing aside.

## Files

- `run.sh`, `make_requests.py`, `collect_native.py`; the page-side check is `tools/wasm_chat_check.mjs`.
- Receipts: `qwen3.json`, `qwen3_long.json`, `qwen25.json`, `qwen25_long.json` (summary and every reply),
  `uncapped_qwen3.txt`, `paperkiln_wasm.txt`.

Command (WSL; each job under `scripts/cpu_budget.py`, one core, session ember):

```bash
bash run.sh ~/wi/src ~/wi/build_cpu ~/wi/ctxwork   # steps: build qwen3 qwen25 uncapped paperkiln
```
