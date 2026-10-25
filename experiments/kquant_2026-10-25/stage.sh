#!/bin/bash
# Stage the VM inputs on ext4 (WSL): the gate's ember tree (~/kq/src: ember main plus this change), gate.py,
# the 40 prompts, the CPU receipts' reply files and the mtstudio receipt, and the two k-quant GGUFs in 36 MB
# parts.   bash stage.sh <stage dir>
set -e
ST=${1:?stage}
H=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments
B=/mnt/c/ml_artifacts/next_base/base
rm -rf "$ST"; mkdir -p "$ST/small/kquant_2026-10-25" "$ST/small/pk_base_v1_2026-10-24" "$ST/small/receipts"
tar czf "$ST/ember_kq.tgz" -C ~/kq src
echo "$(cat ~/kq/src/BASE_COMMIT) + kquant working tree" > "$ST/COMMITS"
cp $H/kquant_2026-10-25/vm.sh "$ST/"
cp $H/kquant_2026-10-25/gate.py "$ST/small/kquant_2026-10-25/"
cp $H/pk_base_v1_2026-10-24/pk_base_v1_parity.jsonl "$ST/small/pk_base_v1_2026-10-24/"
cp $H/kquant_2026-10-25/q4_k_?_replies.jsonl $H/kquant_2026-10-25/q4_k_?_ref_replies.jsonl "$ST/small/receipts/" 2>/dev/null \
  || echo "CPU reply receipts not staged (compared locally afterwards)"
head -40 /mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl > "$ST/small/prompts.jsonl"
(cd "$ST/small" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
rm -rf "$ST/small"
cd "$ST"
split -b 36M -d $B/pk_base_v1_q4_k_m.gguf qm.part
split -b 36M -d $B/pk_base_v1_q4_k_s.gguf qs.part
(cd $B && sha256sum pk_base_v1_q4_k_m.gguf pk_base_v1_q4_k_s.gguf) > BIG_SHA256SUMS
ls -la "$ST"; cat BIG_SHA256SUMS
