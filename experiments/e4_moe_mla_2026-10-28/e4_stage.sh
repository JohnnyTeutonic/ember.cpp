#!/bin/bash
# Stage the VM inputs of E4's CUDA leg on ext4 (WSL):  bash e4_stage.sh REPO COMMIT STAGE_DIR
# ember.tgz: git's archive of ember at COMMIT from REPO (prefix ember/, LF line ends); small.tgz: the five synthetic
# files and their cases (C:/ml_artifacts/thirdparty/e4_synth, prefix e4_synth/), keramos v1.6.1's git archive (prefix
# keramos/) and SHA256SUMS; COMMITS; e4_vm.sh.
set -e
REPO=${1:?repo}; C=${2:?commit}; ST=${3:?stage}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
rm -rf "$ST"; mkdir -p "$ST/small/keramos" "$ST/small/e4_synth"
git -C "$REPO" archive --prefix=ember/ -o "$ST/ember.tgz" "$C"
git -C "$REPO" rev-parse "$C" > "$ST/COMMITS"
git -C $R/keramos archive "v1.6.1^{commit}" | tar -x -C "$ST/small/keramos"
for f in mixtral qwen2moe qwen3moe deepseek2 deepseek2_qlora deepseek2_dense_yarn deepseek3; do
  cp /mnt/c/ml_artifacts/thirdparty/e4_synth/$f.gguf /mnt/c/ml_artifacts/thirdparty/e4_synth/$f.cases.json "$ST/small/e4_synth/"
done
git -C "$REPO" show "$C:experiments/e4_moe_mla_2026-10-28/e4_vm.sh" > "$ST/e4_vm.sh"
(cd "$ST/small" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
rm -rf "$ST/small"
ls -la "$ST"
