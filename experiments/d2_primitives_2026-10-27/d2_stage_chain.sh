#!/bin/bash
# Stage the chain model's CUDA leg (WSL):  bash d2_stage_chain.sh COMMIT STAGE_DIR
# ember.tgz (WSL git's archive at COMMIT), small.tgz (the chain's GGUF, chain_parity.sh's 42 prompts, SHA256SUMS),
# COMMITS, d2_vm_chain.sh.
set -e
C=${1:?commit}; ST=${2:?stage}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
H=$R/ember.cpp/experiments/d2_primitives_2026-10-27
rm -rf "$ST"; mkdir -p "$ST/small"
git -C $R/ember.cpp archive --prefix=ember/ -o "$ST/ember.tgz" "$C"
git -C $R/ember.cpp rev-parse "$C" > "$ST/COMMITS"
cp /mnt/c/ml_artifacts/arxiv_chain_2026-10-27_d2/arxiv_1803_02155.gguf "$ST/small/"
cp $HOME/d2e/chain/prompts.jsonl "$ST/small/"
cp $H/d2_vm_chain.sh "$ST/" && sed -i 's/\r$//' "$ST/d2_vm_chain.sh"
(cd "$ST/small" && sha256sum * > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
rm -rf "$ST/small"
ls -la "$ST"
