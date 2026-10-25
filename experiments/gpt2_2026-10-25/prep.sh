#!/bin/bash
# Copy the official GPT-2 checkpoint and paperkiln's reference tools to ext4 (~/gpt2_gate), verify sha256s.
set -e
A=/mnt/c/ml_artifacts/thirdparty/gpt2-124m
PK=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
W=~/gpt2_gate
mkdir -p $W/model $W/pk/tools
for f in gpt2-124m-f32.gguf config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt SOURCES.json SHA256SUMS; do
  cmp -s $A/$f $W/model/$f 2>/dev/null || cp $A/$f $W/model/
done
(cd $W/model && sha256sum -c SHA256SUMS)
cp $PK/tools/*.py $W/pk/tools/
cp $PK/experiments/engine_2026-10-25/gpt2/gate_n40.jsonl $PK/experiments/engine_2026-10-25/gpt2/gate_n40.jsonl.ir.json $PK/experiments/engine_2026-10-25/gpt2/gate_n40.jsonl.summary.json $W/
mkdir -p $W/pk/corpus/eval
cp $PK/corpus/eval/chat_prompts_v1.jsonl $W/pk/corpus/eval/
(cd $PK && git rev-parse HEAD && git status --short -- tools/gguf_ir.py tools/ir_numpy.py)
sha256sum $W/pk/tools/gguf_ir.py $W/pk/tools/ir_numpy.py
du -sh $W
