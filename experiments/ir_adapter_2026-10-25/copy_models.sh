#!/bin/bash
# Copy the gate models the adapter self-test reads to ext4 (~/ember_wt/adapter_models; scratch, a working
# copy only while the tests need it). Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
set -e
A=/mnt/c/ml_artifacts
D=~/ember_wt/adapter_models
mkdir -p $D
for m in state_gate/state4 attnres_gate/ar3 mor_gate/mor3 mor_gate/mor2p0 recursion_gate/b2x4 kimi_tiny mamba_tiny swa_tiny; do
  n=$(basename $m)
  mkdir -p $D/$n
  for f in $A/$m/*.gguf $A/$m/*.ir.json $A/$m/score_in.jsonl $A/$m/score_out.jsonl; do
    [ -f "$f" ] || continue
    cmp -s "$f" $D/$n/$(basename $f) || cp "$f" $D/$n/
  done
done
du -sh $D
