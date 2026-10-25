#!/bin/bash
# Stage the VM inputs on ext4 (run in WSL): the two ember trees (git archives made with Windows git, so WSL
# reads no OneDrive repository), the gate models, the inputs of tools/mor_device_gate (make_inputs.py), the long
# chat prompts of the third-party device gate; the tiny-chat MoR3 model in parts of 36 MB.
#   bash stage.sh <dir holding ember_new.tgz, ember_head.tgz and COMMITS> <stage dir>
set -e
IN=${1:?archives}; ST=${2:?stage}
H=$(cd "$(dirname "$0")" && pwd)
A=/mnt/c/ml_artifacts
TC=$A/tiny_chat/e1_mor/pt_e1_mor_MoR3_r1/tc_e1_mor_MoR3_r1.gguf
rm -rf "$ST"; mkdir -p "$ST/small/models" "$ST/small/inputs"
cp "$IN/ember_new.tgz" "$IN/ember_head.tgz" "$IN/COMMITS" "$ST/"
cp $H/vm.sh "$ST/"
python3 $H/make_inputs.py "$ST/small/inputs"
M=$ST/small/models
mkdir -p $M/mor $M/state4 $M/ar3 $M/synth $M/gpt2_tiny
for m in mor3 mor2p0; do cp $A/mor_gate/$m/$m.gguf $A/mor_gate/$m/${m}_q8_0.gguf $M/mor/; done
mkdir -p $M/recursion
for m in p1_b2x3_c1 p1_b2x3_c1_inject b2x4; do cp $A/recursion_gate/$m/r.gguf $M/recursion/$m.gguf; done
cp $A/state_gate/state4/state4.gguf $A/state_gate/state4/state4_q8_0.gguf $M/state4/
cp $A/attnres_gate/ar3/ar3.gguf $M/ar3/
for f in mistral_swa gptneox gpt2 mamba gemma3; do cp ~/ember_wt/bcheck_$f/$f.gguf $M/synth/; done
cp ~/ember_wt/bcheck_gemma3/gemma3.per_layer_freq_scale_0.25.ir.json ~/ember_wt/bcheck_gemma3/gemma3.weight_offset_0.25.ir.json $M/synth/
cp $A/gpt2_tiny/gpt2_tiny.gguf $M/gpt2_tiny/
cp ~/ember_wt/expr_gate_gpt2/ir_expr.json $M/gpt2_tiny/gpt2_tiny_expr.ir.json
cp $A/devgate/stage-ember-gate-1022d/ids.tgz "$ST/small/"
(cd "$ST/small" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
cd "$ST"
split -b 36M -d $TC tc.part
(cd $(dirname $TC) && sha256sum $(basename $TC)) > BIG_SHA256SUMS
rm -rf "$ST/small"
ls -la "$ST"; cat BIG_SHA256SUMS
