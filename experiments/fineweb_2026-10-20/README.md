# FineWeb chat models, 20 Oct 2026

paperkiln's FineWeb base (47,500 steps, best val 3.437), fine-tuned two ways, re-exported with
paperkiln master (eps 1e-5, ggml dims, embedded IR). Their tokenizer is the FineWeb+TinyChat
BPE (vocab_fineweb_tinychat_bpe8k_v1), not tinychat_bpe8k; ember's prompt token counts match
paperkiln's bpe.py on every prompt. References: mtstudio /chat (paperkiln de33d142, context
contract) on each model's spec.

| Model (sha256) | CPU | CUDA | WASM | context harness (CPU, CUDA) |
|---|---|---|---|---|
| chat_fwsft, TinyChat SFT 500 steps (aa3f3144) | 40/40 | 40/40 | 40/40 | 13/13, 13/13 |
| chat_fwsft_q8_0_all (eb02e629) | 35/40 | 35/40 | 35/40 | - |
| chat_fwmix, QA+chat SFT 1500 steps (d92a4528) | 40/40 | 40/40 | 40/40 | 13/13, 13/13 |
| chat_fwmix_q8_0_all (59c7e018) | 36/40 | 36/40 | 36/40 | - |

40 frozen prompts, greedy, up to 60 tokens; the 8-bit files differ from f32 on the usual few
near-tie prompts and agree across all three backends.
