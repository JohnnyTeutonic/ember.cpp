#!/usr/bin/env bash
# E6 on a Colab GPU VM (VM side; drive.sh launches it detached): unpack the ember tree, vm_probe.sh (stop
# unless Chrome has a hardware WebGPU adapter), vm_setup.sh, wait for the staged models (drive.sh uploads
# them while the VM builds: /content/stage/UPLOAD_DONE), check their sha256, then vm_run.sh.
#   bash vm_colab.sh [PART ...]       (PARTs: vm_run.sh's; default all)
# /content/E6_DONE or /content/E6_FAILED at the end; the log is /content/e6_vm.log; results in /content/e6out.
set -uo pipefail
C=/content
export E6_W=$C/w E6_OUT=$C/e6out E6_STAGE=$C/stage E6_SRC=$C/w/ember
mkdir -p "$E6_W" "$E6_OUT" "$E6_STAGE"
exec >> $C/e6_vm.log 2>&1
fail() { echo "FAILED: $*"; echo "$*" > $C/E6_FAILED; exit 1; }
echo "=== vm_colab $(date -u) parts: ${*:-all}"
if [ ! -d "$E6_SRC" ]; then
  mkdir -p "$E6_SRC" && tar --no-same-owner -xzf $C/ember.tgz -C "$E6_SRC" || fail "untar ember"
  find "$E6_SRC" -name "*.sh" -exec sed -i "s/\r$//" {} +
fi
cat $C/COMMIT 2>/dev/null > "$E6_OUT/COMMIT"
H=$E6_SRC/experiments/webgpu_gpu_2026-10-27
if [ ! -f "$E6_OUT/probe/PASS" ]; then
  bash "$H/vm_probe.sh"; rc=$?
  [ $rc = 0 ] || fail "probe rc=$rc (4: Vulkan sees the GPU but Chrome does not; 5: no NVIDIA Vulkan device)"
  touch "$E6_OUT/probe/PASS"
fi
if [ ! -f "$E6_OUT/setup/BUILT" ]; then
  bash "$H/vm_setup.sh" || fail "setup"
  touch "$E6_OUT/setup/BUILT"
fi
echo "--- waiting for the staged models $(date -u +%T)"
for i in $(seq 1 360); do [ -f $E6_STAGE/UPLOAD_DONE ] && break; sleep 10; done
[ -f $E6_STAGE/UPLOAD_DONE ] || fail "no UPLOAD_DONE"
(cd $E6_STAGE && for p in m/*.part.00; do b=${p%.part.00}; [ -f "$b" ] || cat "$b".part.* > "$b"; done
 sha256sum -c --quiet SHA256SUMS) || fail "stage sha256"
rm -f $E6_STAGE/m/*.part.*
echo "--- waiting for the official checkpoints $(date -u +%T)"
for i in $(seq 1 180); do [ -f $E6_W/TP_DONE ] && break; sleep 10; done
cat "$E6_OUT/setup/thirdparty.sha256"
bash "$H/vm_run.sh" "$@"
echo "vm_colab done $(date -u)"
touch $C/E6_DONE
