#!/bin/bash
# The receipts of spec_r1_2026-10-28 from the full VM outputs (run in WSL):  bash r1_receipts.sh
# Session 1 (ember-r1, /mnt/c/ml_artifacts/ember_spec_r1/ember-r1/out, pick_e1_L16's legs) into receipts/cuda_s1/:
# hardware, binaries, SASS, the benches, nsys kernel tables, end-to-end meta and comparisons (plain runs' records
# gzipped), dump comparisons, the pool's meta; session 2's log (ember-r1b, reclaimed before its first pack) into
# receipts/cuda_s2/vm_log.txt.
set -e
A=/mnt/c/ml_artifacts/ember_spec_r1
X=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/spec_r1_2026-10-28
S1=$A/ember-r1/out
R=$X/receipts/cuda_s1
mkdir -p $R/bench $R/nsys $R/e2e $R/bits $R/pool $X/receipts/cuda_s2
cp $S1/hardware.txt $S1/gpu.txt $S1/bins.sha256 $S1/big.sha256 $S1/sass.json $R/
cp $A/ember-r1/vm.log $R/vm_log.txt
cp $S1/bench/*.json $R/bench/; for f in $S1/bench/*.log; do cp $f $R/bench/$(basename $f .log)_log.txt; done
cp $S1/nsys/*.json $S1/nsys/*.csv $R/nsys/
cp $S1/e2e/*.meta.json $S1/e2e/*_vs_*.json $S1/e2e/*.txt $R/e2e/
for f in $S1/e2e/pick_h_off.jsonl $S1/e2e/pick_h_det_off.jsonl $S1/e2e/pick_n_det_off.jsonl; do gzip -c $f > $R/e2e/$(basename $f).gz; done
cp $S1/bits/*_vs_*.json $S1/bits/*.meta.json $S1/rows.sha256 $R/bits/ 2>/dev/null || true
cp $S1/pool/pick_pool.jsonl.meta.json $S1/pool/pick_pool.txt $R/pool/
cp $A/ember-r1b/vm.log $X/receipts/cuda_s2/vm_log.txt
du -sh $X/receipts
