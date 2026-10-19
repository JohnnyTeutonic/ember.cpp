#!/bin/sh
# Start ember chat: the AVX2/FMA build when the CPU has both, else portable.
BIN=/app/tinyllama_server
if grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo; then
  BIN=/app/tinyllama_server_v3
fi
echo "ember chat: starting $BIN"
exec "$BIN" /app/model.gguf 7860 0.0.0.0 /app/www 0
