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
| `EMBER_SPEC=k` | speculative decoding with prompt-lookup drafts of up to k tokens (greedy only) |
| `EMBER_DECODE_GRAPH=0` | disable the CUDA-graph decode on the llama path |
| `EMBER_DECODE_ATTN=legacy` | the one-block-per-head decode attention kernel instead of split-K |
| `EMBER_GEMM_TF32=0` | FP32 instead of TF32 GEMMs in the CUDA prefill |
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
| `TINYLLAMA_LOGITS_DUMP=file` | append each prompt's next-token logits to a file |
| `TINYLLAMA_TOPK_DEBUG=1` | top-10 candidates at every sampled position, on stderr |

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
