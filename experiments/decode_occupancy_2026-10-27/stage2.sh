#!/bin/bash
# Stage the second session's VM inputs (run in WSL) from the first session's stage dir:
#   bash stage2.sh <first stage dir> <stage dir>
# The same COMMITS, archives' layout, model parts and checksums; vm.sh is vm2.sh, and ember_new.tgz is rebuilt from
# the working tree (build.sh tree, plus tools/occ_bench.cu) so it carries the current decode_attention_test.
set -e
S1=${1:?first stage}; ST=${2:?stage}
O=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/decode_occupancy_2026-10-27
SRC=~/occ/src
bash $O/build.sh tree
cp $O/occ_bench.cu $SRC/tools/occ_bench.cu
rm -rf "$ST"; mkdir -p "$ST"
cp $S1/COMMITS $S1/BIG_SHA256SUMS $S1/ember_head.tgz $S1/small.tgz $S1/pick.part* $S1/m60.part* "$ST/"
tar czf "$ST/ember_new.tgz" -C "$(dirname $SRC)" --transform "s,^$(basename $SRC),ember," "$(basename $SRC)"
cp $O/vm2.sh "$ST/vm.sh" && sed -i 's/\r$//' "$ST/vm.sh"
ls -la "$ST"
