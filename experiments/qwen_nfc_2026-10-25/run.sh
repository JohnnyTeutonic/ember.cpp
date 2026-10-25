#!/bin/bash
# The Qwen NFC gate end to end (WSL; ember sources on ext4 at SRC, default ~/ember_wt/src, built CPU tree
# at ~/ember_wt/build_cpu for nlohmann/json; OLD = an ember aa50a61 tree for the before-NFC column).
# Each job runs under the CPU budget. Receipts are written to OUT (default ~/qnfc) and then copied here.
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
set -e
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
H=$R/ember.cpp/experiments/qwen_nfc_2026-10-25
SRC=${SRC:-$HOME/ember_wt/src}
OLD=${OLD:-$HOME/ember_cpu/head}
OUT=${OUT:-$HOME/qnfc}
J=$HOME/ember_wt/build_cpu/_deps/nlohmann_json-src/include
B="python3 $R/scripts/cpu_budget.py run --session ember --cores 1"
mkdir -p "$OUT/bin"
$B --job "Qwen NFC: tools" --resume "rerun $H/run.sh" -- bash -c "
  cd $SRC && g++ -O2 -std=c++17 -I. tools/nfc_check.cpp -o $OUT/bin/nfc_check &&
  g++ -O2 -std=c++17 -I. -I$J tools/llama3_vocab_check.cpp -o $OUT/bin/llama3_vocab_check &&
  cd $OLD && g++ -O2 -std=c++17 -I. -I$J tools/llama3_vocab_check.cpp -o $OUT/bin/llama3_vocab_check_old"
# The tables are what the committed generator writes.
python3 $SRC/tools/gen_nfc_tables.py "$OUT/unicode_nfc_tables.h"
cmp "$OUT/unicode_nfc_tables.h" "$SRC/unicode_nfc_tables.h" && echo "unicode_nfc_tables.h regenerates byte-identical"
python3 $H/make_nfc_set.py "$OUT/nfc_set_v1.jsonl"
cmp "$OUT/nfc_set_v1.jsonl" "$H/nfc_set_v1.jsonl" && echo "nfc_set_v1.jsonl regenerates byte-identical"
$B --job "Qwen NFC: nfc engine vs HF" --resume "rerun $H/run.sh" -- \
  python3 $H/nfc_gate.py nfc "$OUT/bin/nfc_check" "$OUT/nfc_unit.json"
$B --job "Qwen NFC: token ids vs HF" --resume "rerun $H/run.sh" -- \
  python3 $H/nfc_gate.py tokens "$OUT/bin/llama3_vocab_check" "$OUT/bin/llama3_vocab_check_old" "$OUT" 2>&1 | grep -v -i warn
cp "$OUT/nfc_unit.json" "$OUT"/tokens_qwen*.json "$H/"
