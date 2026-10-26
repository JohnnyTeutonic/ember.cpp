#!/bin/bash
# The adapter self-test's gate models for keramos v1.3.0 (rule 38): ~/ember_wt/adapter_models_r13 holds every model of
# ~/ember_wt/adapter_models (copy_models.sh's; symlinks), except kimi_tiny and swa_tiny, which carry
# general.architecture "gpt2" and get relabelled copies (keramos tools/gguf_relabel.py: architecture "paperkiln",
# the gpt2.* keys renamed paperkiln.*, the document, its digest and the tensors unchanged).
# Usage (WSL): bash relabel_models.sh [KERAMOS archive, default ~/keramos_v1.3.0]
set -e
K=${1:-$HOME/keramos_v1.3.0}
S=~/ember_wt/adapter_models
D=~/ember_wt/adapter_models_r13
mkdir -p $D
for m in $S/*/; do
  n=$(basename $m)
  if [ "$n" = kimi_tiny ] || [ "$n" = swa_tiny ]; then
    mkdir -p $D/$n
    for f in $S/$n/*; do
      b=$(basename $f)
      case "$b" in
        *.gguf) [ -f $D/$n/$b ] || python3 $K/tools/gguf_relabel.py $f $D/$n/$b ;;
        *) ln -sfn $f $D/$n/$b ;;
      esac
    done
  else
    ln -sfn $S/$n $D/$n
  fi
done
ls -la $D | tail -n +2
