#!/bin/bash
# The adapter self-test's gate models for keramos v1.3.0 (rule 38): ~/ember_wt/adapter_models_r13 holds every model of
# ~/ember_wt/adapter_models (copy_models.sh's; symlinks), except kimi_tiny and swa_tiny, which carry
# general.architecture "gpt2" and get relabelled copies (keramos tools/gguf_relabel.py: architecture "paperkiln",
# the gpt2.* keys renamed paperkiln.*, the document, its digest and the tensors unchanged). The originals in
# /mnt/c/ml_artifacts are only read. adapter_selftest.py and regress.py bits run this script themselves when the cache
# is missing; a model already relabelled is kept.
# Usage (WSL): bash relabel_models.sh [KERAMOS archive, default ~/keramos_v1.3.0]
# The default archive is made when missing, as ../ir_rev12_2026-10-26/run_suite.sh makes it (git archive of the tag,
# keramos's checkout untouched).
set -e
E=$(cd "$(dirname "$0")/.." && pwd)
if [ -n "${1:-}" ]; then K=$1
else
  K=~/keramos_v1.3.0
  KR=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos
  C=$(git -C "$KR" rev-parse "v1.3.0^{commit}")
  if [ ! -f "$K/COMMIT" ] || [ "$(cat "$K/COMMIT")" != "$C" ]; then
    rm -rf "$K" && mkdir -p "$K" && git -C "$KR" archive "$C" | tar -x -C "$K" && echo "$C" > "$K/COMMIT"
  fi
fi
[ -f $K/tools/gguf_relabel.py ] || { echo "no $K/tools/gguf_relabel.py" >&2; exit 1; }
S=~/ember_wt/adapter_models
D=~/ember_wt/adapter_models_r13
[ -d $S/kimi_tiny ] && [ -d $S/swa_tiny ] || bash $E/ir_adapter_2026-10-25/copy_models.sh > /dev/null
mkdir -p $D
for m in $S/*/; do
  n=$(basename $m)
  if [ "$n" = kimi_tiny ] || [ "$n" = swa_tiny ]; then
    mkdir -p $D/$n
    for f in $S/$n/*; do
      b=$(basename $f)
      case "$b" in
        *.gguf)  # written as tmp_<name>.gguf, then renamed (the tool validates its output by the .gguf suffix)
          if [ ! -f $D/$n/$b ]; then
            python3 $K/tools/gguf_relabel.py $f $D/$n/tmp_$b
            mv $D/$n/tmp_$b $D/$n/$b
          fi ;;
        *) ln -sfn $f $D/$n/$b ;;
      esac
    done
  else
    ln -sfn $S/$n $D/$n
  fi
done
ls -la $D | tail -n +2
