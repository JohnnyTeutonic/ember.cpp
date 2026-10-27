#!/bin/bash
# Stage the VM inputs of the D2 CUDA leg on ext4 (WSL):  bash d2_stage.sh COMMIT STAGE_DIR
# ember.tgz: WSL git's archive of ember at COMMIT (prefix ember/, LF line ends); small.tgz: keramos v1.4.0's git
# archive (prefix keramos/) and SHA256SUMS; COMMITS; d2_vm.sh.
set -e
C=${1:?commit}; ST=${2:?stage}
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
H=$R/ember.cpp/experiments/d2_primitives_2026-10-27
rm -rf "$ST"; mkdir -p "$ST/small/keramos"
git -C $R/ember.cpp archive --prefix=ember/ -o "$ST/ember.tgz" "$C"
git -C $R/ember.cpp rev-parse "$C" > "$ST/COMMITS"
git -C $R/keramos archive "v1.4.0^{commit}" | tar -x -C "$ST/small/keramos"
cp $H/d2_vm.sh "$ST/" && sed -i 's/\r$//' "$ST/d2_vm.sh"
(cd "$ST/small" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
tar czf "$ST/small.tgz" -C "$ST" small
rm -rf "$ST/small"
ls -la "$ST"
