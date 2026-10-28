#!/bin/bash
# Stage the ember-chat Space for pk_sft_v1 F16 in ~/e22rel/space (nothing is uploaded here):
# tools/build_wasm_space.sh's layout, from the ember 9d993f4 tree in ~/e22rel/src, with the 2gb page
# build (~/e22rel/page2gb), the F16 file as chat.gguf, the staged card and index.html, SOURCE.txt and
# MANIFEST.sha256 (every file and its sha256, checked by publish_space.py before any upload).
set -euo pipefail
SP=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/e22rel
W=$HOME/e22rel
S=$W/space
G=/mnt/c/ml_artifacts/general_chat/sft/pk_sft_v1_f16.gguf
# the F16 file's home is beside the F32 and Q8_0 files; the staging copy is a working copy for the upload
if [ ! -f $G ]; then cp $W/variants/pk_sft_v1_f16.gguf $G.tmp && mv $G.tmp $G; fi
[ "$(sha256sum < $G | cut -d' ' -f1)" = 3916b3ae617f01b6610e8c6a8ebfab301227c83cad2d40b33d90d548a639b957 ] || { echo "F16 sha256 mismatch"; exit 1; }
rm -rf $S && mkdir -p $S
cp $SP/README_space_pk_sft_v1.md $S/README.md
cp $SP/staged_src/index.html $S/index.html
cp $W/src/deploy/wasm/web/chat.js $W/src/deploy/wasm/web/worker.js $S/
cp -r $W/src/deploy/wasm/web/webgpu $S/webgpu
cp $W/src/www/style.css $S/
cp $W/page2gb/ember.js $W/page2gb/ember.wasm $S/
cp $G $S/chat.gguf
printf '*.gguf filter=lfs diff=lfs merge=lfs -text\n*.wasm filter=lfs diff=lfs merge=lfs -text\n' > $S/.gitattributes
{
  echo "ember.cpp commit: 9d993f48de2cd976a7708d90767df3a58107baf8 (deploy/wasm/build.sh with EMBER_WASM_MAX_MEMORY=2gb; README.md and index.html from experiments/release_pk_sft_v1_q8_2026-10-28/staged)"
  echo "model: pk_sft_v1_f16.gguf (F16; llama.cpp b6000 llama-quantize from pk_sft_v1.gguf 4d53c7ff) sha256 $(sha256sum $S/chat.gguf | cut -d' ' -f1)"
  echo "ember.wasm sha256 $(sha256sum $S/ember.wasm | cut -d' ' -f1)"
  echo "gate: greedy F16 = F32 40/40 (ids), WebAssembly = native 40/40 (ids), WebGPU (SwiftShader, llama_webgpu.js) = WebAssembly = native 40/40"
} > $S/SOURCE.txt
( cd $S && find . -type f ! -name MANIFEST.sha256 | sed 's|^./||' | sort | xargs sha256sum > MANIFEST.sha256 )
cat $S/SOURCE.txt
cat $S/MANIFEST.sha256
du -sh $S
