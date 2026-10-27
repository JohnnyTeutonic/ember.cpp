#!/usr/bin/env bash
# E6 on a GPU VM (VM side): does headless Chrome see the GPU as a WebGPU adapter? Installs Chrome stable,
# the Vulkan loader, vulkaninfo and glvnd's EGL, Playwright (driver only), then tries at most two designs
# for the NVIDIA Vulkan driver and runs gpu_probe.html through remote_chrome.py on the first that works.
#   E6_SRC=ember tree E6_OUT=out root bash vm_probe.sh      (exit 0: Chrome has a hardware adapter;
#   4: Vulkan sees the GPU but Chrome does not; 5: no NVIDIA Vulkan device with either design)
# Design 1: the driver's user-space graphics libraries as the platform mounts them (RunPod with
#   NVIDIA_DRIVER_CAPABILITIES=all mounts libGLX_nvidia and nvidia_icd.json) plus glvnd's libEGL.so.1, which
#   the ICD dlopens before it creates an instance (headless device enumeration goes through EGL): without
#   it vkCreateInstance fails (ERROR_INCOMPATIBLE_DRIVER) and Chrome falls back to llvmpipe (RunPod, 27 Oct).
# Design 2: the graphics libraries of the exact running driver version from NVIDIA's own installer
#   (NVIDIA-Linux-x86_64-<version>.run --extract-only; only the GLX/EGL/Vulkan user-space files are copied,
#   the CUDA libraries the platform provides are left alone), with the ICD, EGL vendor and layer manifests.
# Outputs: $E6_OUT/probe/ (hardware.txt, icd.txt, vulkaninfo_*.txt, probe.json, probe.log).
set -uo pipefail
SRC=${E6_SRC:-/root/e6/ember}
O=${E6_OUT:-/workspace/e6/out}/probe
mkdir -p "$O"
exec > >(tee -a "$O/probe.log") 2>&1
echo "=== probe $(date -u) ==="
export DEBIAN_FRONTEND=noninteractive
{ nvidia-smi; nvidia-smi -q | sed -n '1,40p'; lscpu; free -g; cat /etc/os-release; } > "$O/hardware.txt" 2>&1
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
echo "driver $DRV; NVIDIA_DRIVER_CAPABILITIES (pid 1): $(tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep NVIDIA_DRIVER_CAPABILITIES)"
if ! command -v google-chrome > /dev/null || ! command -v vulkaninfo > /dev/null || [ ! -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 ]; then
  apt-get update -qq
  apt-get install -y -q wget ca-certificates libvulkan1 vulkan-tools libegl1 libglvnd0 libgl1 libgles2 strace > /dev/null 2>&1
  [ -f /tmp/chrome.deb ] || wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  apt-get install -y -q /tmp/chrome.deb > /dev/null 2>&1
fi
python3 -c "import playwright" 2> /dev/null || python3 -m pip -q install --root-user-action=ignore playwright 2>&1 | grep -v notice
google-chrome --version | tee "$O/chrome_version.txt"

nv_ok() {   # vulkaninfo, NVIDIA's ICD only: a discrete NVIDIA device?
  local f
  f=$(grep -ls libGLX_nvidia /etc/vulkan/icd.d/*.json /usr/share/vulkan/icd.d/*.json 2>/dev/null | head -1)
  [ -n "$f" ] || return 1
  VK_ICD_FILENAMES=$f vulkaninfo --summary > "$O/vulkaninfo_$1.txt" 2>&1
  grep -E "deviceName|driverName|driverInfo|apiVersion|deviceType" "$O/vulkaninfo_$1.txt"
  grep -q "deviceName *= NVIDIA" "$O/vulkaninfo_$1.txt"
}
{ echo "--- ICD files"; ls -la /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1
  echo "--- NVIDIA graphics libraries"; ldconfig -p | grep -Ei "libGLX_nvidia|libnvidia-glvkspirv|libnvidia-gpucomp|libEGL_nvidia|libnvidia-glcore" 2>&1
} > "$O/icd.txt"
cat "$O/icd.txt"
design=0
if ! grep -qs libGLX_nvidia /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json && ldconfig -p | grep -q libGLX_nvidia.so.0; then
  mkdir -p /etc/vulkan/icd.d
  printf '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}\n' > /etc/vulkan/icd.d/nvidia_icd.json
  echo "design 1: wrote /etc/vulkan/icd.d/nvidia_icd.json"
fi
if nv_ok design1; then design=1; else
  echo "--- design 1 found no NVIDIA Vulkan device; design 2: NVIDIA's $DRV installer, graphics libraries only"
  N=/tmp/nv_$DRV
  if [ ! -d "$N" ]; then
    for u in https://us.download.nvidia.com/tesla/$DRV/NVIDIA-Linux-x86_64-$DRV.run \
             https://us.download.nvidia.com/XFree86/Linux-x86_64/$DRV/NVIDIA-Linux-x86_64-$DRV.run; do
      wget -q -O /tmp/nv.run "$u" && break
    done
    sh /tmp/nv.run --extract-only --target "$N" > /dev/null 2>&1 || echo "extract failed"
  fi
  L=/usr/lib/x86_64-linux-gnu
  for lib in libGLX_nvidia libEGL_nvidia libnvidia-glcore libnvidia-eglcore libnvidia-glsi libnvidia-tls libnvidia-glvkspirv \
             libnvidia-gpucomp libnvidia-present; do
    f=$(ls "$N"/$lib.so.$DRV 2>/dev/null | head -1)
    [ -n "$f" ] && cp -n "$f" $L/ && echo "copied $(basename $f)"
  done
  ln -sf libGLX_nvidia.so.$DRV $L/libGLX_nvidia.so.0
  ln -sf libEGL_nvidia.so.$DRV $L/libEGL_nvidia.so.0
  mkdir -p /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d /usr/share/glvnd/egl_vendor.d
  [ -f "$N/nvidia_icd.json" ] && cp "$N/nvidia_icd.json" /etc/vulkan/icd.d/nvidia_icd.json
  [ -f /etc/vulkan/icd.d/nvidia_icd.json ] || sed "s/__NV_VK_ICD__/libGLX_nvidia.so.0/" "$N/nvidia_icd.json.template" > /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null
  [ -f "$N/nvidia_layers.json" ] && cp "$N/nvidia_layers.json" /etc/vulkan/implicit_layer.d/
  [ -f "$N/10_nvidia.json" ] && cp "$N/10_nvidia.json" /usr/share/glvnd/egl_vendor.d/
  ldconfig
  cat /etc/vulkan/icd.d/nvidia_icd.json
  if nv_ok design2; then design=2; fi
fi
echo "design $design" | tee "$O/design.txt"
[ "$design" = 0 ] && { echo "no NVIDIA Vulkan device with either design"; exit 5; }
cd "$SRC/tools/webgpu_parity"
EMBER_WEBGPU_REMOTE=1 python3 remote_chrome.py --root . --page gpu_probe.html --out "$O/probe.json" --timeout 180
rc=$?
echo "probe rc=$rc"
python3 - "$O/probe.json" << 'PY'
import json, sys
r = json.load(open(sys.argv[1]))
res = r.get("result") or {}
print("adapter", json.dumps(res.get("adapter"))[:400])
print("fallback", res.get("fallback"), "compute", res.get("compute"))
g = (r.get("chrome") or {}).get("gpu") or {}
print("chrome gpu devices", json.dumps(g.get("devices"))[:500])
fs = g.get("featureStatus") or {}
print("webgpu feature status", fs.get("webgpu"), "| vulkan", fs.get("vulkan"))
print("error", r.get("error"))
PY
[ $rc = 0 ] || exit 4
