#!/bin/bash
# Copy the official GPT-2 checkpoint and paperkiln's reference tools to ext4 (~/gpt2_gate), verify sha256s.
# The IR tools (gguf_ir, ir_numpy: keramos) come from $KERAMOS_TOOLS, else paperkiln's vendored spec/tools,
# else paperkiln's tools/ (before keramos v1.1.1); they land in $W/pk/spec/tools, which the scripts read first.
set -e
A=/mnt/c/ml_artifacts/thirdparty/gpt2-124m
PK=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
if [ -n "$KERAMOS_TOOLS" ]; then IRT=$KERAMOS_TOOLS
elif [ -f $PK/spec/tools/gguf_ir.py ]; then IRT=$PK/spec/tools
else IRT=$PK/tools; fi
W=~/gpt2_gate
mkdir -p $W/model $W/pk/tools $W/pk/spec/tools
for f in gpt2-124m-f32.gguf config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt SOURCES.json SHA256SUMS; do
  cmp -s $A/$f $W/model/$f 2>/dev/null || cp $A/$f $W/model/
done
(cd $W/model && sha256sum -c SHA256SUMS)
cp $PK/tools/*.py $W/pk/tools/
cp $IRT/*.py $W/pk/spec/tools/
cp $PK/experiments/engine_2026-10-25/gpt2/gate_n40.jsonl $PK/experiments/engine_2026-10-25/gpt2/gate_n40.jsonl.ir.json $PK/experiments/engine_2026-10-25/gpt2/gate_n40.jsonl.summary.json $W/
mkdir -p $W/pk/corpus/eval
cp $PK/corpus/eval/chat_prompts_v1.jsonl $W/pk/corpus/eval/
echo "IR tools: $IRT"
(cd $IRT && git rev-parse HEAD && git status --short -- gguf_ir.py ir_numpy.py)
sha256sum $W/pk/spec/tools/gguf_ir.py $W/pk/spec/tools/ir_numpy.py
du -sh $W
