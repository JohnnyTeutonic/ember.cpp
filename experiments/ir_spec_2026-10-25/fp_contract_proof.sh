#!/bin/bash
# IR_SPEC A.6 / Q4: -ffp-contract=off on graph_ir.cpp and utils.cpp changes no x86-64 instruction. Each file is
# compiled twice with the core library's own flags (read from the CMake build tree: Release -O3, no -march),
# without and with the flag, and the disassembly of the two objects is compared (objdump -d, every section,
# addresses and bytes included). Identical code computes identical values on every input, so no x86-64 output
# can move. Writes fp_contract_proof.txt beside this file.
# Usage (from Windows): wsl.exe --cd ~ -e bash -lc "bash <this file>"
set -e
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
H=$R/ember.cpp/experiments/ir_spec_2026-10-25
B=~/ember_wt/build_cpu
W=~/ember_wt/fp_contract_proof
mkdir -p $W
F=$B/CMakeFiles/tinyllama_core.dir/flags.make
DEFS=$(sed -n 's/^CXX_DEFINES = //p' $F)
INCS=$(sed -n 's/^CXX_INCLUDES = //p' $F)
FLAGS=$(sed -n 's/^CXX_FLAGS = //p' $F)
# compile.sh SOURCE OBJECT [EXTRA_FLAG]
echo '#!/bin/bash' > $W/compile.sh
echo "/usr/bin/c++ $DEFS $INCS $FLAGS "'$3 -o $2 -c $1' >> $W/compile.sh
chmod +x $W/compile.sh
cd $R
python3 scripts/cpu_budget.py run --session ember --job "ember fp-contract proof: 4 compiles" --cores 4 \
  --resume "rerun ember.cpp/experiments/ir_spec_2026-10-25/fp_contract_proof.sh" -- bash -c "
    for f in graph_ir utils; do
      $W/compile.sh ~/ember_wt/src/\$f.cpp $W/\$f.default.o '' &
      $W/compile.sh ~/ember_wt/src/\$f.cpp $W/\$f.nocontract.o -ffp-contract=off &
    done
    wait"
{
  echo "fp-contract proof, $(date -u +%Y-%m-%dT%H:%M:%SZ), $(/usr/bin/c++ --version | head -1), $(uname -m)"
  echo "flags: $FLAGS"
  for f in graph_ir utils; do
    for v in default nocontract; do
      objdump -d $W/$f.$v.o | tail -n +3 > $W/$f.$v.dis
    done
    a=$(sha256sum < $W/$f.default.dis | cut -c1-64)
    b=$(sha256sum < $W/$f.nocontract.dis | cut -c1-64)
    n=$(grep -c . $W/$f.default.dis)
    fma=$(grep -c -E "vfmadd|vfmsub|vfnmadd|vfnmsub" $W/$f.default.dis || true)
    if [ "$a" = "$b" ]; then s=identical; else s=DIFFERENT; fi
    echo "$f.cpp: disassembly $s ($n lines; default $a, -ffp-contract=off $b; FMA instructions in the default build: $fma)"
  done
} | tee $H/fp_contract_proof.txt
