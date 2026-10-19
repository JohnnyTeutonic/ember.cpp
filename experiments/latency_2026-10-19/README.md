# Chat latency, 19 Oct 2026

Model: paperkiln `chat_sftfinal.gguf` (TinyChat SFT on the 12,000-step base; d384,
6 layers, BPE 8192, ctx 256; 14.65M params), ember `tinyllama_server` on the local RTX
4060 laptop GPU (WSL2), private instance on port 8095. Requests are shaped like the chat
page's: t = 0.7, top_k 40, top_p 0.9, max_new_tokens 64, a seed each. Conversation length
is set by sending earlier turns as history (frozen TinyChat prompts and their reference
replies); the server drops the oldest turns once prompt + 64 would pass 256 tokens, so
past ~5 turns the prompt stays near 180 tokens. Tools: `tools/chat_latency.py`,
`tools/chat_idle_gap.py`, `tools/chat_concurrency_check.py`.

**Noise.** The laptop's speed drifted by about 2x within the day (GPU at 78 C, another
session compiling and running the Colab CLI on the same CPU in the later sittings).
The first sitting is the clean one; later sittings are recorded but only their
within-sitting comparisons are read.

## One request at a time, GPU (first sitting, 50 requests per row; `cuda_a.json`)

| Earlier turns | Prompt tokens | TTFT p50 / p90 / p99 (ms) | Decode p50 (ms/token) | Reply (tokens) | Request wall p50 / p90 / p99 (ms) |
|---|---|---|---|---|---|
| 0 | 18 | 21 / 29 / 35 | 4.1 | 15 | 86 / 105 / 128 |
| 1 | 52 | 35 / 50 / 98 | 6.8 | 16 | 150 / 252 / 343 |
| 2 | 87 | 42 / 65 / 210 | 7.2 | 17 | 164 / 274 / 552 |
| 4 | 155 | 52 / 81 / 94 | 8.8 | 17 | 202 / 334 / 659 |
| 6 | 179 | 39 / 65 / 89 | 7.0 | 17 | 159 / 270 / 359 |
| 8 | 181 | 48 / 91 / 207 | 8.9 | 17 | 198 / 360 / 488 |

A typical reply is 15-17 tokens, so a chat turn takes 0.1-0.2 s at p50 and stays
under 0.7 s at p99 on the GPU. A second sitting (`cuda_b.json`) gave the same picture
within its noise.

CPU build for comparison (15 requests per row, `cpu.json`): TTFT 404 / 1604 / 3182 ms
p50 at 19 / 89 / 165 prompt tokens, decode 30-36 ms/token, request wall 0.9 / 2.1 / 3.8 s.
CPU prefill grows linearly at about 19 ms per prompt token; it is the CPU build's
bottleneck.

Cold start: the GGUF loads and `/health` answers in 1-2 s; the first request costs
0.2-0.6 s (TTFT 150-410 ms), then warm numbers apply.

## Concurrent clients: a correctness bug, fixed

The server held one `TinyLlamaSession` (one KV cache, one set of device buffers, one
RNG) and served `/chat` from httplib's thread pool with no lock. With 4 parallel clients
the first sitting measured 1.4 requests/s and 3.2 s p50 per request, and
`chat_concurrency_check.py` got HTTP 500s. `/chat` now takes a session mutex (the
response's `ttft.queue_ms` reports the wait). After the fix, 24 greedy requests from 4
clients give replies identical to the sequential ones (0/24 differ), at 14.8 requests/s
against 13.0 sequential, with p50 252 ms (queueing behind ~64 ms requests).

## Pauses between messages

The same 2-turn requests, 20 each (`chat_idle_gap.py`, clean sitting):

| | TTFT p50 / p90 (ms) | Request wall p50 / p90 (ms) |
|---|---|---|
| back to back | 20 / 25 | 83 / 97 |
| after a 2 s pause | 46 / 73 | 257 / 350 |

A person reading and typing always pauses, so this is the latency a user sees: about
3x the back-to-back figure. An opt-in keep-warm (a 1 MiB memset on its own stream every
50 ms for 60 s after each reply, to hold the GPU clocks up) did not remove it across two
A/B pairs, so the penalty is not only GPU clock ramp; the host side (CPU power state and
WSL scheduling under the many small launches and syncs of each decode step) is the
likely remainder. The keep-warm was not kept.

## Levers, in order

1. Capture the per-token decode step as a CUDA graph: one launch per token instead of
   ~60 kernel launches and cuBLAS calls plus syncs, which cuts exactly the host work that
   pauses and contention amplify.
2. Stream tokens to the page (server-sent events) so the first words show at TTFT
   (20-50 ms) instead of after the whole reply.
3. Decode cost rises with context (4 -> 9 ms/token from 18 to 180 prompt tokens);
   look at the attention kernel (one block per head, 6 blocks on a 24-SM GPU) after (1).
