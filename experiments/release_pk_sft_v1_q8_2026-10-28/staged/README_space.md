---
title: ember chat
emoji: 🔥
colorFrom: gray
colorTo: red
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: A 97.5M chat model, built from scratch, in your browser
---

# ember chat

A conversational model trained from scratch with
[paperkiln](https://github.com/JohnnyTeutonic/paperkiln) (a C++ deep-learning framework
with its own autograd and CUDA kernels) and run by
[ember.cpp](https://github.com/JohnnyTeutonic/ember.cpp) (a C++ inference engine), both
written by Jonathan Reich. ember.cpp is compiled to WebAssembly, so the model runs in your
browser: the 221 MB model (16-bit) downloads once, and nothing you type is sent anywhere. It needs
about 1.5 GB of the browser's memory, so a desktop browser is the safer choice.

- Model: pk_sft_v1. Llama-style decoder: 12 layers, width 768, 12 heads, feed-forward width
  2,048, RoPE, SwiGLU, RMSNorm; 97.5M parameters; byte-level BPE in the Llama 3 scheme,
  16,384 tokens; 512-token context.
- Pretrained (pk_base_v1) for 120,000 steps, 0.98B tokens, on a mix of FineWeb-Edu and
  dialogue (SODA, UltraChat 200k, OpenAssistant, everyday conversations): 27% dialogue in the
  main part, 47% in the last 15% of the run.
- Supervised fine-tuning: 18,620 steps on 313,606 conversations in the chat template (463,695
  replies, every assistant turn in the loss), from smol-smoltalk, TinyChat and everyday
  conversations, decontaminated against the evaluation sets.
- Weights: F16 (`chat.gguf`; the tied embedding table is stored twice, as input embedding
  and output head). On the 40 frozen prompts checked, the F16 file gives the same greedy replies
  as the F32 one, and the WebAssembly build (and the WebGPU path, where the browser has it) the
  same as ember.cpp's native server, token for token.

It chats and answers some simple questions; it gets many facts wrong and does no arithmetic
(21 of 60 on paperkiln's simple-QA probe). Check anything it tells you.

Training data and licences: [TinyChat](https://huggingface.co/datasets/starhopp3r/TinyChat)
(starhopp3r), CC BY-NC 4.0, so the model is CC BY-NC 4.0;
[smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), Apache-2.0;
[everyday-conversations-llama3.1-2k](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k),
Apache-2.0; [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), ODC-By 1.0;
[SODA](https://huggingface.co/datasets/allenai/soda) (Kim et al., 2023, "SODA: Million-scale
Dialogue Distillation with Social Commonsense Contextualization"), CC BY 4.0;
[UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), MIT;
[OpenAssistant oasst1 and oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2),
Apache-2.0.
