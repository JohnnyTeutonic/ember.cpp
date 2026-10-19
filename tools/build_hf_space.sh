#!/usr/bin/env bash
# Assemble the Hugging Face Space folder for ember chat (Docker SDK).
# Usage: tools/build_hf_space.sh <model.gguf> <out_dir>
# out_dir gets: Dockerfile, README.md (the Space card), src/ (an ember.cpp
# source snapshot of HEAD via git archive), model/chat.gguf, www/ (the chat
# page with the public index.html), and SOURCE.txt (commit and sha256s).
set -euo pipefail
GGUF=${1:?model.gguf}
OUT=${2:?out_dir}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
rm -rf "$OUT/src" "$OUT/www" "$OUT/model"
mkdir -p "$OUT/src" "$OUT/www" "$OUT/model"
git -C "$REPO" archive HEAD --format=tar \
    ':(exclude)experiments' ':(exclude)docs' ':(exclude)*.gguf' ':(exclude)tests' | tar -x -C "$OUT/src"
cp "$REPO/deploy/hf_space/Dockerfile" "$REPO/deploy/hf_space/README.md" "$REPO/deploy/hf_space/start.sh" "$OUT/"
cp "$REPO/www/script.js" "$REPO/www/style.css" "$OUT/www/"
cp "$REPO/deploy/hf_space/index.html" "$OUT/www/index.html"
cp "$GGUF" "$OUT/model/chat.gguf"
{
  echo "ember.cpp commit: $(git -C "$REPO" rev-parse HEAD)"
  echo "model: $(basename "$GGUF") sha256 $(sha256sum "$GGUF" | cut -d' ' -f1)"
} > "$OUT/SOURCE.txt"
cat "$OUT/SOURCE.txt"
du -sh "$OUT"
