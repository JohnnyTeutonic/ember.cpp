#!/bin/bash
# The follow-up reruns after re-vendoring at paperkiln 7a6c7659 and strict role-form pairing: G5 third
# party (tokenizers only, 1 core), the in-place gate scripts, the parity replays from scratch, the
# ir_logits bit check, and the WebAssembly build and parity. Each job runs under the CPU budget.
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
H=$R/ember.cpp/experiments/chat_template_2026-10-24
B="python3 $R/scripts/cpu_budget.py run --session ember"
cd ~/ct24
$B --job "chat template G5 third-party (7a6c7659)" --cores 1 --resume "rerun $H/rerun_all.sh" -- \
  python3 $H/g5_thirdparty.py 2>&1 | grep -v -i warning | tail -4
bash $H/regress_inplace.sh 2>&1 | tail -10
rm -f $H/regress_parity.json
$B --job "chat template regression: parity replays (7a6c7659)" --cores 4 --resume "rerun $H/regress.py parity" -- \
  python3 $H/regress.py parity 2>&1 | grep -o '^[^ ]* .*byte_identical": [a-z]*\|^pass.*'
$B --job "chat template regression: ir_logits bits (7a6c7659)" --cores 4 --resume "rerun $H/regress.py bits" -- \
  python3 $H/regress.py bits 2>&1 | tail -1
bash $H/wasm_gate.sh 2>&1 | grep -E "build rc|n=|error"
echo all done
