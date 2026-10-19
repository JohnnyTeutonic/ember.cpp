---
title: ember chat
emoji: 🔥
colorFrom: gray
colorTo: red
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: A 14.6M chat model, built from scratch, in your browser
---

# ember chat

A small conversational model trained from scratch with
[paperkiln](https://github.com/JohnnyTeutonic/paperkiln) (a C++ deep-learning framework
with its own autograd and CUDA kernels) and run by
[ember.cpp](https://github.com/JohnnyTeutonic/ember.cpp) (a C++ inference engine), both
written by Jonathan Reich. ember.cpp is compiled to WebAssembly, so the model runs in your
browser: the 19 MB model (8-bit) downloads once, and nothing you type is sent anywhere.

- Llama-style decoder: 6 layers, width 384, 6 heads, RoPE, SwiGLU, RMSNorm; 14.65M
  parameters; byte-level BPE, 8,192 tokens; 256-token context.
- Pretrained on TinyChat dialogue (49M tokens), then supervised fine-tuning on 24,000
  examples in the chat template.
- The WebAssembly build gives the same greedy replies as ember.cpp's native server, word
  for word, on the prompts checked.

It makes small talk. It has not seen general text, so it does not know facts or do
arithmetic.

Training data: [TinyChat](https://huggingface.co/datasets/starhopp3r/TinyChat)
(starhopp3r), CC BY-NC 4.0; the model inherits that licence.
