---
title: ember chat
emoji: 🔥
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
short_description: A 14.6M-parameter chat model, trained and served from scratch
---

# ember chat

A small conversational model trained from scratch with
[paperkiln](https://github.com/JohnnyTeutonic/paperkiln) (a C++ deep-learning framework
with its own autograd and CUDA kernels) and served by
[ember.cpp](https://github.com/JohnnyTeutonic/ember.cpp) (a C++ inference engine), both
written by Jonathan Reich.

- Llama-style decoder: 6 layers, width 384, 6 heads, RoPE, SwiGLU, RMSNorm; 14.65M
  parameters; byte-level BPE, 8,192 tokens; 256-token context.
- Pretrained on TinyChat dialogue (49M tokens), then supervised fine-tuning on 24,000
  examples in the chat template.
- Served here on CPU by ember.cpp; greedy output matches paperkiln's own sampler token
  for token.

It makes small talk. It has not seen general text, so it does not know facts or do
arithmetic. Conversations are not stored.

Training data: [TinyChat](https://huggingface.co/datasets/starhopp3r/TinyChat)
(starhopp3r), CC BY-NC 4.0; the model inherits that licence.
