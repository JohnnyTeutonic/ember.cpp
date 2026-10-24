#!/bin/bash
# IR_SPEC 4.2: NaN and signed zero in ember's EXACT CPU ops (tools/ir_numerics_test.cpp), receipt numerics_cpu.json.
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
H=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/ir_spec_2026-10-25
~/ember_wt/build_cpu/ir_numerics_test > $H/numerics_cpu.json
echo "rc=$?"
grep -B2 -A4 '"pass": false' $H/numerics_cpu.json | head -40
tail -3 $H/numerics_cpu.json
