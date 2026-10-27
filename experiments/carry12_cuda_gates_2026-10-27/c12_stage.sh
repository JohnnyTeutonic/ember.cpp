#!/bin/bash
# Stage the VM inputs of the Carry-12 CUDA gates on ext4 (WSL):
#   bash c12_stage.sh ARCHIVE_DIR STAGE_DIR
# ARCHIVE_DIR holds ember_pin.tgz and ember_head.tgz (git archives, prefix ember/, made with Windows git and
# core.autocrlf=false) and COMMITS ("<pin> <head>"). small.tgz: x5_device_logits.cpp (../attnres_x5_2026-10-25),
# state4, state4_q8_0, ar3, ar3_q8_0 and ar_x5_q8_0, c12_inputs.py's inputs and SHA256SUMS. ar_x5.gguf (79 MB)
# goes in 36 MB parts (x5.partNN, BIG_SHA256SUMS). UPLOADS lists what c12_drive.sh sends.
set -e
IN=${1:?archives}; ST=${2:?stage}; LEGS=${3:-}
E=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments
S=$E/carry12_cuda_gates_2026-10-27
A=/mnt/c/ml_artifacts
rm -rf "$ST"; mkdir -p "$ST/small/models" "$ST/small/inputs"
cp "$IN/ember_pin.tgz" "$IN/ember_head.tgz" "$IN/COMMITS" "$ST/"
cp $S/c12_vm.sh "$ST/" && sed -i 's/\r$//' "$ST/c12_vm.sh"
cp $E/attnres_x5_2026-10-25/x5_device_logits.cpp "$ST/small/"
python3 $S/c12_inputs.py "$ST/small/inputs"
M=$ST/small/models
cp $A/state_gate/state4/state4.gguf $A/state_gate/state4/state4_q8_0.gguf $M/
cp $A/attnres_gate/ar3/ar3.gguf $A/attnres_gate/ar3/ar3_q8_0.gguf $M/
cp $A/attnres_gate_x5/ar_x5/ar_x5_q8_0.gguf $M/
(cd "$ST/small" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
rm -rf "$ST/small"
cd "$ST"
split -b 36M -d $A/attnres_gate_x5/ar_x5/ar_x5.gguf x5.part
(cd $A/attnres_gate_x5/ar_x5 && sha256sum ar_x5.gguf) > BIG_SHA256SUMS
[ -n "$LEGS" ] && echo "$LEGS" > LEGS
{ echo c12_vm.sh; [ -n "$LEGS" ] && echo LEGS; echo COMMITS; echo BIG_SHA256SUMS; echo small.tgz; ls x5.part*; echo ember_pin.tgz; echo ember_head.tgz; } > UPLOADS
ls -la "$ST"; cat BIG_SHA256SUMS COMMITS
