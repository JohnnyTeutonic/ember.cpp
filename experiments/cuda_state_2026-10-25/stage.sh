#!/bin/bash
# Stage the VM inputs on ext4 (run in WSL): the two ember trees (git archives made with Windows git, so WSL
# reads no OneDrive repository), the gate models, the inputs of tools/state_device_gate (make_inputs.py),
# the long chat prompts of the third-party device gate, the L32 prompts; files over 36 MB in parts.
#   bash stage.sh <dir holding ember_new.tgz, ember_head.tgz and COMMITS> <stage dir>
set -e
IN=${1:?archives}; ST=${2:?stage}
H=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/cuda_state_2026-10-25
A=/mnt/c/ml_artifacts
rm -rf "$ST"; mkdir -p "$ST/small/models" "$ST/small/inputs"
cp "$IN/ember_new.tgz" "$IN/ember_head.tgz" "$IN/COMMITS" "$ST/"
cp $H/vm.sh "$ST/"
python3 $H/make_inputs.py "$ST/small/inputs"
M=$ST/small/models
mkdir -p $M/state4 $M/ar3 $M/synth $M/gpt2_tiny
cp $A/state_gate/state4/state4.gguf $A/state_gate/state4/state4_q8_0.gguf $M/state4/
cp $A/attnres_gate/ar3/ar3.gguf $M/ar3/
for f in mistral_swa gptneox gpt2 mamba gemma3; do cp ~/ember_wt/bcheck_$f/$f.gguf $M/synth/; done
cp ~/ember_wt/bcheck_gemma3/gemma3.per_layer_freq_scale_0.25.ir.json ~/ember_wt/bcheck_gemma3/gemma3.weight_offset_0.25.ir.json $M/synth/
cp $A/gpt2_tiny/gpt2_tiny.gguf $M/gpt2_tiny/
cp ~/ember_wt/expr_gate_gpt2/ir_expr.json $M/gpt2_tiny/gpt2_tiny_expr.ir.json
cp $A/devgate/stage-ember-gate-1022d/ids.tgz "$ST/small/"
head -40 $A/chat_base/sft/tinychat_hf_prompts_v1.jsonl > "$ST/small/prompts.jsonl"
cp /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/rmsnorm_2026-10-24/sp_compare.py "$ST/small/"
(cd "$ST/small" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
cd "$ST"
split -b 36M -d ~/ember_wt/state_tokens/chat_state_k4.gguf k4.part
split -b 36M -d $A/tiny_chat/e1/post_e1_L32_r1/sft/tc_sft_L32_r1.gguf l32.part
(cd ~/ember_wt/state_tokens && sha256sum chat_state_k4.gguf; cd $A/tiny_chat/e1/post_e1_L32_r1/sft && sha256sum tc_sft_L32_r1.gguf) > BIG_SHA256SUMS
rm -rf "$ST/small"
ls -la "$ST"; cat BIG_SHA256SUMS
