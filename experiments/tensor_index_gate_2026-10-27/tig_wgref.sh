#!/bin/bash
# ember CPU references for the WebGPU leg (WSL), as ../webgpu_ir_2026-10-26/ref.sh makes them: tools/ir_wasm_gate's
# native build (tig_wasm.sh's ~/tig/wasm/bin) per run into ~/tig/wgate/NAME/ref (ids.json, ref.jsonl, ref_<i>.bin),
# EMBER_IR_CTX 1024, 40 prompts of 8 ids and 8 greedy tokens (the IR holds 16 positions, as ar3's legs). The untagged
# references serve tig_gate_cpu.py's untagged document through EMBER_IR. Under the CPU budget (tig_job.sh).
set -u
BIN=$HOME/tig/wasm/bin
WORK=$HOME/tig/wgate
D=/mnt/c/ml_artifacts/attnres_gate_rec/ar_rec
UT=$HOME/tig/work/ar_rec_untagged.ir.json
for run in ar_rec ar_rec_untagged ar_rec_q8_0 ar_rec_q8_0_untagged; do
  g=$D/ar_rec.gguf; case $run in ar_rec_q8_0*) g=$D/ar_rec_q8_0.gguf;; esac
  irenv=""; case $run in *_untagged) irenv="EMBER_IR=$UT";; esac
  d=$WORK/$run; mkdir -p $d/ref
  env $irenv $BIN/ir_wasm_gate ref $g $d/ref --ctx 1024 --first 40 --len 8 --gen 8 > $d/ref.out 2> $d/ref.log
  echo "$run ref rc=$? ($(wc -l < $d/ref/ref.jsonl) prompts)"
done
cp $UT $WORK/
