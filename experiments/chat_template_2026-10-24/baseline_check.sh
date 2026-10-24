#!/bin/bash
# Is ~/ember_wt/src (the ext4 build tree's sources) ember main 69d7ca2, the pre-change engine?
# Exports 69d7ca2 with git archive into ~/ember_69d7ca2/src and compares the files the CPU build
# syncs (top-level *.cpp *.h *.cu CMakeLists.txt, tools/*.cpp). Prints the differing files.
set -e
SRC=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp
OUT=~/ember_69d7ca2/src
rm -rf "$OUT"
mkdir -p "$OUT"
git -C "$SRC" archive 69d7ca2 | tar -x -C "$OUT"
cd "$OUT"
n=0
for f in *.cpp *.h *.cu CMakeLists.txt tools/*.cpp; do
  [ -f "$f" ] || continue
  if ! cmp -s "$f" ~/ember_wt/src/"$f"; then echo "differs: $f"; n=$((n+1)); fi
done
echo "files differing from 69d7ca2: $n"
ls -la ~/ember_wt/build_cpu/tinyllama_server ~/ember_wt/build_cpu/ir_logits ~/ember_wt/build_cpu/state_tokens_test
