#!/usr/bin/env bash
# Assemble the static Hugging Face Space for ember chat in the browser.
# Usage: tools/build_wasm_space.sh <model.gguf> <wasm_build_dir> <out_dir>
#   wasm_build_dir holds ember.js + ember.wasm from deploy/wasm/build.sh.
# out_dir gets: README.md (the Space card, sdk: static), index.html, chat.js,
# worker.js, style.css, ember.js, ember.wasm, chat.gguf, .gitattributes (LFS
# for the model and the wasm) and SOURCE.txt (commit and sha256s).
set -euo pipefail
GGUF=${1:?model.gguf}
WASM=${2:?wasm_build_dir}
OUT=${3:?out_dir}
REPO=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$OUT"
mkdir -p "$OUT"
cp "$REPO/deploy/wasm/README_space.md" "$OUT/README.md"
cp "$REPO/deploy/wasm/web/index.html" "$REPO/deploy/wasm/web/chat.js" "$REPO/deploy/wasm/web/worker.js" "$OUT/"
cp "$REPO/www/style.css" "$OUT/"
cp "$WASM/ember.js" "$WASM/ember.wasm" "$OUT/"
cp "$GGUF" "$OUT/chat.gguf"
printf '*.gguf filter=lfs diff=lfs merge=lfs -text\n*.wasm filter=lfs diff=lfs merge=lfs -text\n' > "$OUT/.gitattributes"
{
  echo "ember.cpp commit: $(git -C "$REPO" rev-parse HEAD) (+ deploy/wasm working tree)"
  echo "model: $(basename "$GGUF") sha256 $(sha256sum "$GGUF" | cut -d' ' -f1)"
  echo "ember.wasm sha256 $(sha256sum "$WASM/ember.wasm" | cut -d' ' -f1)"
} > "$OUT/SOURCE.txt"
cat "$OUT/SOURCE.txt"
du -sh "$OUT"
