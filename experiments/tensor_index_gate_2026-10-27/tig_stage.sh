#!/bin/bash
# Stage the VM inputs of the tensor_index gate's CUDA leg on ext4 (WSL):
#   bash tig_stage.sh ARCHIVE COMMIT STAGE_DIR
# ARCHIVE is a git archive of ember at COMMIT (prefix ember/). small.tgz holds tig_vm.sh's tools and inputs:
# x5_device_logits.cpp (../attnres_x5_2026-10-25), ar_rec.gguf and ar_rec_q8_0.gguf, their copies with
# rec_app.{4,5}.attnres_{attn,ffn}_q set to NaN (tig_nan.py), the untagged document (tig_gate_cpu.py's), the 40
# score_in sequences (tig_inputs.py) and SHA256SUMS.
set -e
ARC=${1:?archive}; C=${2:?commit}; ST=${3:?stage}
S=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/tensor_index_gate_2026-10-27
X=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/attnres_x5_2026-10-25
D=/mnt/c/ml_artifacts/attnres_gate_rec/ar_rec
rm -rf "$ST"; mkdir -p "$ST/small/models"
cp "$ARC" "$ST/ember.tgz"
echo "$C" > "$ST/COMMITS"
cp $S/tig_vm.sh "$ST/" && sed -i 's/\r$//' "$ST/tig_vm.sh"
cp $X/x5_device_logits.cpp "$ST/small/"
cp $D/ar_rec.gguf $D/ar_rec_q8_0.gguf "$ST/small/models/"
for v in ar_rec ar_rec_q8_0; do
  python3 $S/tig_nan.py $D/$v.gguf "$ST/small/models/${v}_nan_app45.gguf" rec_app.4.attnres_attn_q.weight \
    rec_app.4.attnres_ffn_q.weight rec_app.5.attnres_attn_q.weight rec_app.5.attnres_ffn_q.weight
done
cp ~/tig/work/ar_rec_untagged.ir.json "$ST/small/"
python3 $S/tig_inputs.py "$ST/small"
(cd "$ST/small" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
rm -rf "$ST/small"
ls -la "$ST"
