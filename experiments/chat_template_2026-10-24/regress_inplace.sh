#!/bin/bash
# Gate (d), part 1: the existing CPU gate scripts rerun in place with this change's build
# (~/ember_wt/build_cpu); each rewrites its own receipt, and git then shows whether any receipt
# changed (no change = byte-identical). Servers and scratch run from ~/ct24 (ext4).
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
# no set -e: a failing gate is reported, the rest still run
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp/experiments
mkdir -p ~/ct24/regress && cd ~/ct24/regress
run() {
  python3 $R/scripts/cpu_budget.py run --session ember --job "chat template regression: $1" --cores 4 \
    --resume "rerun ember.cpp/experiments/chat_template_2026-10-24/regress_inplace.sh" -- "${@:2}" > ~/ct24/regress/$1.out 2>&1
  echo "$1 rc=$?"
}
run recursion_gate python3 $E/recursion_2026-10-23/recursion_gate.py
run mor_gate python3 $E/recursion_2026-10-23/mor_gate.py
run mor_refusals python3 $E/recursion_2026-10-23/mor_refusals.py
run attnres_gate python3 $E/attnres_2026-10-24/attnres_gate.py
run make_state_gguf python3 $E/state_tokens_2026-10-24/make_state_gguf.py
run state_tokens_gate python3 $E/state_tokens_2026-10-24/state_tokens_gate.py
run state4_gate python3 $E/state_tokens_2026-10-24/state4_gate.py
cd $R/ember.cpp
git status --short -- experiments/recursion_2026-10-23 experiments/attnres_2026-10-24 experiments/state_tokens_2026-10-24
git diff --stat -- experiments/recursion_2026-10-23 experiments/attnres_2026-10-24 experiments/state_tokens_2026-10-24
echo "receipts checked"
