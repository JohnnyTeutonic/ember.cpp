# ember.cpp: full usage reference

The [README](../README.md) is the overview with quick starts. This page is the complete
reference: build dependencies, every command-line option, the Python package, the management
scripts, runtime switches and the source layout.

## Build dependencies

1. **CMake (>= 3.11)** and a **C++17 compiler** (g++, Clang or MSVC).
2. **Boost (Regex and Xpressive headers)**, used by the Llama 3 tokenizer.
   - Debian/Ubuntu: `sudo apt install build-essential cmake libboost-all-dev libomp-dev`
   - Fedora/RHEL: `sudo dnf install gcc-c++ cmake boost-devel libgomp`
   - macOS (Homebrew): `brew install cmake boost llvm` (llvm for OpenMP)
   - Windows (vcpkg): `vcpkg install boost-regex nlohmann-json cpp-httplib`
3. **nlohmann/json and cpp-httplib**: fetched by CMake when not installed system-wide.
4. **OpenMP** (optional, recommended): CPU multi-threading.
5. **CUDA Toolkit** (optional): GPU inference. `nvcc` and cuBLAS are needed; CMake detects them,
   or force with `-DHAS_CUDA=ON` / `-DHAS_CUDA=OFF`.
6. **Emscripten** (optional): the WebAssembly build (`deploy/wasm/build.sh`).

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release          # CUDA auto-detected; -DHAS_CUDA=ON/OFF to force
make -j$(nproc)                              # Windows/MSVC: cmake --build . --config Release
```

Executables: `tinyllama` (command line), `tinyllama_server` (HTTP server + web UI). The binary
names predate the ember.cpp rename and are kept for compatibility.

## Model files

- **GGUF**: one self-contained file with the tokenizer embedded. Tensor types F32, F16, BF16,
  Q8_0, Q8_K, Q4_K_M, Q6_K, and Q4_0 / Q4_1 / Q5_0 / Q5_1 (dequantised at load). Older Llama 2 /
  TinyLlama files without BPE merges get merges generated from vocabulary and scores.
  paperkiln GGUFs carry their byte-level BPE tokenizer and, from paperkiln master, the model's
  graph as the `paperkiln.ir` key.
- **SafeTensors**: a directory with `config.json`, `tokenizer.json` and `model.safetensors`
  (F32, BF16, F16; computed in F32).

Tested sources: [TinyLlama 1.1B Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
(SafeTensors) and [its GGUFs](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF),
[Llama-2 7B GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF),
[Llama-3 8B GGUF](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF).

## `tinyllama_server`

```
tinyllama_server <model> [port=8080] [host=localhost] [www=./www] [n_gpu_layers]
```

`model` is a `.gguf` file or a SafeTensors directory; `n_gpu_layers` is -1 for all layers on the
GPU, 0 for CPU only. Endpoints:

| Endpoint | What it does |
|---|---|
| `GET /` | the chat web UI (`www/`) |
| `POST /chat` | `{user_input, system, history, thinking, temperature, top_k, top_p, seed, max_new_tokens, repeat_penalty, repeat_last_n, stream, return_ids}` -> `{reply, stop_reason, tokens, ttft}`; `"stream": true` sends server-sent events |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions, with `stream` and `stream_options.include_usage`; `repeat_penalty` / `repeat_last_n`, `top_k`, `thinking` and `chat_template_kwargs.enable_thinking` as extension fields; `frequency_penalty` and `presence_penalty` accept only 0 |
| `GET /v1/models` | OpenAI-compatible model list |
| `GET /health` | model, tokenizer, context length and `chat_template` (`{name, version}`, null for raw completion; `chat_template_why` says how it was resolved) |
| `POST /score` | log-probabilities of continuations, paperkiln's `mtstudio score` on token ids (below): `{requests: [{key, prompt_ids, continuation_ids}]}` -> `{results: [{i, key, prompt_tokens, continuation_ids, over_ctx, logprobs, argmax, sum}], stats}` |

Chat template (LOCAL_LLM_STACK.md section 3, point 3; shared with paperkiln's `mtstudio chat`):
the prompt is the conversation rendered by a template record, never a format built into the
engine. At load the GGUF's template resolves in this order: its `paperkiln.chat_template` record;
`paperkiln/1` when ids 0-3 are paperkiln's chat specials (every paperkiln export before the key
existed); `paperkiln/legacy-word` for a word-level vocabulary; a third-party record
(`chatml-qwen2.5`, `chatml-qwen3`, `gemma3`) by the sha256 of `tokenizer.chat_template`, else by
its markers and `general.architecture`; otherwise none, and the model serves raw completion only.
The log and `/health` say which. Generation stops at the record's stop tokens.
- A role-form `history` (`[{"role", "content"}]`) pairs strictly, as mtstudio's: after an optional
  leading system entry, user and assistant alternate, starting with user and ending with assistant
  (the new message is `user_input`); anything else is a 400 naming the rule. Pair-form entries
  (`[{"user", "assistant"}]`) are whole exchanges.
- `system` (or a leading `{"role": "system"}` history entry, not both, nowhere else) is the system
  message, rendered by the template's system mode: `paperkiln/1` and Gemma 3 fold it into the first
  kept user message, ChatML writes a system turn. A template without a system role refuses it (400).
- `thinking` (Qwen3): `false` writes the empty think block after the generation prompt; absent is
  the template's default (on, as HF's). Earlier replies' reasoning (`</think>`) is stripped from the
  history as Qwen3's template does.
- The OpenAI endpoint maps messages one to one: system and developer messages before the first
  user message join with a blank line into the system message (an empty one is dropped); a later
  one, or a tool message, is a 400; user and assistant messages alternate and the last is the user's.
- Content holding one of the template's own markers is refused (400).

Context rule (shared with paperkiln's `mtstudio chat`): the oldest whole history turns are
dropped while prompt + `max_new_tokens` exceeds the context; the system message is never dropped
or cut; a message that alone fills the context (with the system message) is rejected with HTTP
400; a reply that reaches the limit ends with `stop_reason` "length".

Repetition penalty (shared with paperkiln's `mtstudio chat`): `repeat_penalty` p (default 1.0,
off) and `repeat_last_n` n (default 64; 0 is off). Before each token is sampled, every distinct
id among the last n ids of the prompt plus the reply so far, except the chat template's stop set
(`paperkiln/1`: end-of-text, `<|user|>` and `<|assistant|>`), has its logit l replaced by l / p
when l > 0 and l * p otherwise; greedy
decoding applies it too. p must be in [1.0, 2.0] and n in [0, context]: public mode and the
WebAssembly page clamp to those ranges, other requests get HTTP 400. Speculative decoding
(`EMBER_SPEC`) is off while the penalty is on. The command line's `--rep-penalty` is a different,
older penalty for word-level models.

### `POST /score`: log-probabilities (forced-choice benchmarks)

The semantics of paperkiln's `mtstudio score` (`tools/mtstudio.cpp` `score_cmd`, which its
`tools/bench_eval.py` uses for HellaSwag, ARC, PIQA, WinoGrande, LAMBADA and BLiMP), on token ids:

```bash
curl http://127.0.0.1:8080/score -H "Content-Type: application/json" \
  -d '{"requests": [{"key": "q1|0", "prompt_ids": [0, 412, 88], "continuation_ids": [93, 17]}]}'
```

- Each request is one fresh forward over `prompt_ids + continuation_ids[:-1]` from position 0; nothing
  is carried from one request to the next. For continuation token j, from the logits row before it:
  `logprobs[j]`, its log-probability under the softmax at temperature 1, in double from the float
  logits; `argmax[j]`, the top logit with `<unk>`, `<s>`, `</s>` and `<pad>` left out (by name, where
  the vocabulary has them; the first maximum wins); `sum`, the log-probabilities added in order.
- A request with more than context + 1 ids (`/health`'s `context`) comes back `"over_ctx": true`,
  unscored. Records carry mtstudio's fields: `i` (the index in this call), `key` (echoed, any JSON),
  `prompt_tokens`, `continuation_ids`.
- Refused with HTTP 400 before anything runs, naming the first bad request, where mtstudio score
  fails: a request that is not an object, ids that are not lists of integers, an empty prompt or
  continuation, a prompt id outside [0, vocab + k) or a continuation id outside [0, vocab) (k: the
  model's per-turn state tokens, 0 without), and, for state-token models, state ids that are not
  complete blocks after an end-of-text closing a reply. mtstudio's text fields (`prompt`,
  `continuation`, `messages`) are not served: ember scores ids. Off (403) under `EMBER_PUBLIC=1`.
- A call holds the session from its first request to its last, so a batch is not interleaved with
  chat requests; `/chat` after `/score` starts from a fresh cache, as it would anyway.
- Speed without changing a bit: on the CPU paths, consecutive requests sharing a prefix (the choices
  of one item) keep its KV rows, and only the continuation's rows go through the final norm and LM
  head; every CPU row is computed apart from the batch it is in, so the log-probabilities are those of
  a fresh full prefill, bit for bit (checked against `EMBER_SCORE_FULL=1`). The device, recurrent
  state, eviction, depth routing and state tokens always prefill afresh.
- On CUDA the scoring forward runs FP32 GEMMs whatever `EMBER_GEMM_TF32` says (no TF32, no BF16
  tensor cores; they move log-probabilities by ~1e-3); `/chat`'s prefill is unchanged.
  `EMBER_DETERMINISTIC=1` makes repeated calls bit-identical on one GPU and build.
- `stats`: requests, scored, over_ctx, positions run, positions reused, queue and scoring time.

## `tinyllama` (command line)

```
tinyllama <model_path> <tokenizer_path> <num_threads> <prompt|chat> [options] [initial prompt]
```

- `tokenizer_path`: `tokenizer.json` for SafeTensors; the model path again for GGUF.
- `--system-prompt "<text>"`, `--max-tokens <N>` (256), `--n-gpu-layers <N>` (-1),
  `--use-mmap <true|false>` (true), `--temperature <F>` (0.1), `--top-k <N>` (40),
  `--top-p <F>` (0.9), `--use-kv-quant <true|false>` (INT8 KV cache on GPU, false),
  `--use-batch-generation <true|false>`, `--max-batch-size <N>` (1),
  `--rep-penalty <F>` (1.3; 1.0 disables), `--raw-prompt` (no `Q: ... A:` wrap).

```bash
./build/tinyllama ./models/model.Q4_K_M.gguf ./models/model.Q4_K_M.gguf 4 chat \
    --system-prompt "You are a helpful AI." --n-gpu-layers -1 --temperature 0.7 "Tell me a joke."
```

Operational logs go to `debugging.log` in the working directory.

## Runtime switches

| Variable | Effect |
|---|---|
| `EMBER_PUBLIC=1` | public-serving limits: 600-character messages, 16 history turns, 64 new tokens, clamped sampling, 64 KB requests, bounded queue, warnings-only logging |
| `EMBER_PREFIX_REUSE=0` | disable KV/prefix reuse across turns |
| `EMBER_SCORE_REUSE=0` | `/score`: no prefix reuse between a call's requests (the bits do not change) |
| `EMBER_SCORE_FULL=1` | `/score`: every request a fresh prefill computing every row's logits (the reference the default path is checked against) |
| `EMBER_SPEC=k` | speculative decoding with prompt-lookup drafts of up to k tokens (greedy only); on CUDA a verification pass is one captured graph costing about one decode step (`experiments/verify_2026-10-27`) |
| `EMBER_DRAFT=table` | with `EMBER_SPEC`: where prompt lookup finds no draft, draft from the corpus table `EMBER_DRAFT_TABLE` (an n-gram table of the most frequent next token, built offline by `experiments/draft_2026-10-27/dr_table.py` from chat text in the model's vocabulary), cut where the product of the entries' shares falls below `EMBER_DRAFT_THETA` (default 0.02). Verification decides every token, so replies are plain decode's. Unset or `pld`: prompt lookup alone |
| `EMBER_VERIFY_LINEAR=rows` | with `EMBER_SPEC`: the verification rows through the decode step's own GEMV, bit-identical to decode under `EMBER_DETERMINISTIC=1` (slower). Unset, the mode decides: `rows` under `EMBER_DETERMINISTIC=1`, `smallm` otherwise; `smallm`, `gemm` or `gemvb` set explicitly override the mode |
| `EMBER_VERIFY=legacy` | with `EMBER_SPEC`: the per-launch verification pass of 20 Oct 2026 |
| `EMBER_DECODE_GRAPH=0` | disable the CUDA-graph decode on the llama path |
| `EMBER_DECODE_ATTN=split` | the split-K decode attention with 256-key chunks for the cache's capacity (the default until `experiments/decode_occupancy_2026-10-27`); `legacy`: the one-block-per-head kernel. Unset: the tiled split-K kernel, chunks of 32 keys chosen by the current length (at most 64 per head), 1.5x (20M) and 1.8x (60M) the decode tokens per second of `split` at 256-500 cached tokens on an L4 |
| `EMBER_DECODE_ATTN_TILE=m,x` | the tiled kernel's chunking: at least m keys per chunk, at most x chunks per head (x <= 64; default 32,64) |
| `EMBER_GEMM_TF32=0` | FP32 instead of TF32 GEMMs in the CUDA prefill |
| `EMBER_DETERMINISTIC=1` | deterministic CUDA mode: prefill and decode logits bit-identical run to run on the same GPU and build (below); `/health` then reports `"deterministic": true`. Off by default |
| `EMBER_DEQUANT_AT_LOAD=0` | keep small quantised models in their blocks (default: dequantise under 64M parameters) |
| `EMBER_MOE_DENSE=1` | run every MoE expert densely instead of the routed path |
| `EMBER_INTERPRETER=1/0` | force the IR interpreter on, or off, for a GGUF with `paperkiln.ir` |
| `EMBER_IR=path` | run the interpreter from a side-file IR |
| `EMBER_IR_ONLY=1` | load only through the interpreter (no llama weights) |
| `EMBER_IR_BUILD=1` | build the IR from the metadata for a plain llama or mistral GGUF too (third-party families and windowed llama/mistral always get one) |
| `EMBER_IR_CTX=N` | cap the interpreter's KV state at N positions (third-party files declare their whole trained context); the browser page's cap is `EMBER_WASM_CTX` / `?ctx=N` (below) |
| `EMBER_IR_DEVICE=cpu` | keep the interpreter on the CPU in a CUDA build |
| `EMBER_IR_GRAPH=0` | disable the interpreter's CUDA-graph decode |
| `EMBER_IR_QUANT=1` / `EMBER_IR_DEQUANT=1` | interpreter weights kept quantised / dequantised at load (default: quantised above 64M parameters, CPU only) |
| `EMBER_DECODE_TIMING=1`, `EMBER_OP_TIMING=1` | per-phase decode and per-op prefill timing |
| `EMBER_KERNEL_DEBUG=1` | the CUDA batch helpers' debug probes (each one synchronises) |
| `TINYLLAMA_LOGITS_DUMP=file` | append the full logit vector at every sampled position to a file, one line per generated token (under `EMBER_SPEC`, each verification row a token is taken from, fetched from the device for the dump) |
| `TINYLLAMA_TOPK_DEBUG=1` | top-10 candidates at every sampled position, on stderr |

### Deterministic CUDA mode (`EMBER_DETERMINISTIC=1`)

By default the CUDA decode is not bit-reproducible run to run: the decode rmsnorm adds its warp sums with
shared-memory atomics, so logits can move in the last bits (up to ~1e-5) between two runs of the same build, though
replies rarely change. With `EMBER_DETERMINISTIC=1` (any value but `0`) the decode rmsnorm and the one-block-per-head
decode attention reduce in a fixed order, and each cuBLAS handle gets `CUBLAS_WORKSPACE_CONFIG=:4096:8` (unless already
set) and no atomics; every logit at every position is then bit-identical run to run on the same GPU and build, on the
tuned llama path and the IR interpreter (whose device path was already fixed-order). It costs nothing measurable
(decode and prefill within 1-2% noise on a T4). Not promised across GPUs, CUDA or cuBLAS versions, or builds. Unset,
every path is exactly as before (same kernels; `/health` has no `deterministic` field). Receipts:
`experiments/determinism_2026-10-26/`.

With `EMBER_SPEC=k` the mode also selects `EMBER_VERIFY_LINEAR=rows` unless that variable is set: every row of the
CUDA verification pass is then the decode step's own arithmetic, so speculation's ids, and the logits of every token
it accepts, are bit-identical to plain decode in the mode (CUDA, L4: k = 2, 4, 8 on two models, 60 requests each,
every row of every generated token, `experiments/verify_det_2026-10-27/`). The `rows` pass costs n linear launches
per layer, so speculation gains less in the mode: 1.10-1.22x plain decode on pick_e1_L16 and 1.30-1.34x on
pk_base_v1 over those requests, against 1.40-1.55x for pick_e1_L16 with `smallm`
(`experiments/verify_2026-10-27/`). Setting `EMBER_VERIFY_LINEAR` explicitly overrides the mode, and gives up the
guarantee for any value but `rows`.

## The browser page (`deploy/wasm/`)

`deploy/wasm/build.sh OUT` builds `ember.js` + `ember.wasm`, and `tools/build_wasm_space.sh` adds
the page. The page follows `/chat`'s request rules, including the context rule, under public caps
(64 new tokens, 16 turns, 600-character messages).

Build-time variables:

| Variable | Effect |
|---|---|
| `EMBER_WASM_CTX=N` | the context cap (default 2048; 0: none): the KV state of a file the interpreter runs holds at most N positions, as `EMBER_IR_CTX` does natively; a file with a smaller context keeps its own |
| `EMBER_WASM_MAX_MEMORY=4gb` | the heap ceiling (default 1gb); every official third-party checkpoint gated in WebAssembly needs 4gb |
| `EMBER_WASM_SIMD=0` | build without WebAssembly SIMD |
| `EMBER_WASM_THREADS=N`, `EMBER_WASM_NAME=ember-mt` | a pthreads build for cross-origin-isolated pages |

The page's URL takes:
- `?ctx=N`: the context cap for this load, overriding the build's (`?ctx=0` gives the file's own
  context);
- `?backend=wasm`: the CPU even when WebGPU is available.

From JavaScript, `ember_set_ctx(n)` before `ember_load` sets the cap, and `ember_ctx()` gives the
loaded model's context after the cap.

Third-party files declare their whole trained context: Qwen3-0.6B declares 40,960 positions, 9.4 GB
of f32 KV. With the default cap Qwen3-0.6B and Qwen2.5-0.5B load in the 4gb build. Prompts past the
cap follow the context rule: the oldest exchanges are dropped and the system message is kept. The cap
applies to files the IR interpreter runs, which includes every third-party family. A llama-architecture
file on the tuned path keeps its declared context.

## Python package

```bash
pip install .                         # CPU
TINYLLAMA_CPP_BUILD_CUDA=1 pip install .   # CUDA (Windows: set / $env:)
pip install .[torch]                  # with the PyTorch extras
pip install -e .                      # editable
```

CUDA builds need the CUDA Toolkit (11.0+), CMake 3.18+ and a C++17 compiler.

```python
import tinyllama_cpp

session = tinyllama_cpp.TinyLlamaSession(
    model_path="path/to/model.gguf",
    tokenizer_path="path/to/model.gguf",   # GGUF: same path; SafeTensors: tokenizer.json
    threads=4,
    n_gpu_layers=-1,
)
print(session.generate("What is AI?", steps=64))
```

## Management scripts

`manage.sh` (Linux/macOS, `chmod +x manage.sh` first) and `manage.ps1` (Windows PowerShell) wrap
the common tasks; `./manage.sh help` lists every option.

- `build [--build-type Release|Debug] [--cuda ON|OFF]`
- `run-server [--model-dir <path>] [--threads <n>] [--host <h>] [--port <n>] [--n-gpu-layers <n>] [--mmap true|false] [--no-log]`
- `run-chat [--model-dir <path>] [--tokenizer <path>] [--threads <n>] [--system-prompt <text>] [--prompt <text>] [--steps <n>] [--n-gpu-layers <n>]`
- `run-prompt [--model-dir <path>] [--prompt <text>] [--steps <n>] [--temperature <f>]`
- `install [--gpu|--cpu]`
- `docs` (Doxygen API reference)

PowerShell takes the same commands with `-BuildType`, `-Cuda`, `-ModelDir`, `-TokenizerPath`,
`-Threads`, `-NGpuLayers`, `-Gpu` / `-Cpu`.

## Releasing a chat model

`tools/release_chat.py build` takes a paperkiln spec and checkpoint through five stages:
1. GGUF export (f32 and `q8_0_all`).
2. The greedy parity gate against `mtstudio chat` on CPU (and CUDA). The Q8 file is gated on
   first-token logits.
3. paperkiln's simple-QA and chat evals.
4. The WebAssembly build, with its own parity check.
5. The static Space folder.

Every stage writes its receipt to `OUT/release.json`, a finished stage is skipped on a rerun,
and every heavy stage takes the workspace's `.cpu_lock` first. `release_chat.py publish`
uploads the Space only with `--confirm ember-chat` and a build whose every stage passed. It
reads just `HF_TOKEN` from the env file named on the command line.

## PyTorch reference

`pytorch/` holds a plain PyTorch implementation of the Llama 2 / TinyLlama architecture for
SafeTensors models (`run_inference.py`, `tinyllama.py`, `utils.py`; `pip install -r
pytorch/requirements.txt`), for checking ember's outputs against an independent implementation.
See `pytorch/README.md`.

## Source layout

| Path | Contents |
|---|---|
| `api.cpp/.h` | `TinyLlamaSession`: load, tokenise, prefill, decode, sampling, prefix reuse, speculation |
| `server.cpp` | HTTP server: `/chat`, OpenAI endpoints, streaming, public mode |
| `main.cpp` | the `tinyllama` command line |
| `model.cpp/.h`, `cpu_batch_processor.cpp`, `cpu_attention.cpp` | the tuned llama path (CPU and CUDA forward, CUDA-graph decode, MoE) |
| `cuda_kernels.cu/.h` | CUDA kernels: flash prefill attention, split-K decode attention, MoE, RoPE, norms |
| `graph_ir.cpp/.h`, `graph_ir_ops.cpp/.h`, `graph_ir_cuda.cu/.h` | the paperkiln IR interpreter (CPU and CUDA) and `build_ir_from_gguf`, the IR of a third-party GGUF from its metadata |
| `graph_ir_expr.cpp/.h` | the `expr` op: paperkiln's parser (vendored), the CPU evaluator, the CUDA program it lowers to |
| `graph_ir_mamba.cpp/.h` | Mamba-1's `conv1d_causal` and `selective_scan` on the CPU |
| `graph_ir_recurrence.cpp/.h` | `recurrence`, a recurrent op the document defines (IR_SPEC 6.10), on the CPU |
| `ir_digest.cpp/.h` | the digest of an IR document (IR_SPEC 3.10: I-JSON, RFC 8785, SHA-256) |
| `gguf_parser.cpp`, `weight_management.cpp`, `safetensors_loader.cpp`, `quantization.cpp` | model loading and dequantisation |
| `tokenizer.cpp`, `bpe_tokenizer.h` | Llama SentencePiece / tiktoken tokenizers; paperkiln byte-level BPE |
| `parallel_for.h/.cpp` | OpenMP natively, a thread pool in WebAssembly pthread builds |
| `bindings.cpp`, `tinyllama_cpp/` | Python bindings (pybind11) |
| `deploy/wasm/` | WebAssembly build and the static web chat |
| `tools/` | parity gates, benchmarks, quantiser, kernel tests |
| `experiments/` | dated receipts for every measured claim |
| `www/` | the server's web UI |
