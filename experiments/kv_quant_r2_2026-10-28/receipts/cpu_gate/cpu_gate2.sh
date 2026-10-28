#!/usr/bin/env bash
# The two synthetic E4 files (small vocabularies) with ids below 200; deepseek2 under EMBER_KV=f16 must refuse (MLA).
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/r2kv
G=~/r2kv/gate3; T=/mnt/c/ml_artifacts/thirdparty/e4_synth; out=$R/cpu_gate3
python3 -c "
import json, random
r = random.Random(7)
json.dump([[r.randrange(4, 200) for _ in range(24)] for _ in range(4)], open('$G/seqs_small.json', 'w'))"
same() { if [ -s "$1" ] && cmp -s "$1" "$2"; then r=IDENTICAL; else r=DIFFER; fi
         echo "$3 $r $(sha256sum < "$1" | cut -c1-16) $(sha256sum < "$2" | cut -c1-16)" | tee -a $out/gate.txt; }
for n in deepseek2 qwen3moe gemma2 phi3_longrope; do
  for t in base new; do EMBER_IR_CTX=256 $G/$t/build/ir_logits $T/$n.gguf < $G/seqs_small.json > $G/${n}_$t.json 2> $G/${n}_$t.log; done
  same $G/${n}_base.json $G/${n}_new.json "$n default (ids < 200): HEAD vs HEAD+R2"
  EMBER_KV=f32 EMBER_IR_CTX=256 $G/new/build/ir_logits $T/$n.gguf < $G/seqs_small.json > $G/${n}_f32.json 2> $G/${n}_f32.log
  same $G/${n}_base.json $G/${n}_f32.json "$n EMBER_KV=f32 vs HEAD"
done
EMBER_KV=f16 EMBER_IR_CTX=256 $G/new/build/ir_logits $T/deepseek2.gguf < $G/seqs_small.json > /dev/null 2> $G/deepseek2_f16.log
echo "deepseek2 EMBER_KV=f16: $(grep -o 'EMBER_KV=f16: [^\"]*' $G/deepseek2_f16.log | head -1)" | tee -a $out/gate.txt
EMBER_KV=q8 EMBER_IR_CTX=256 $G/new/build/ir_logits $T/qwen3moe.gguf < $G/seqs_small.json > $G/qwen3moe_q8.json 2> $G/qwen3moe_q8.log
if [ -s $G/qwen3moe_q8.json ]; then
  cmp -s $G/qwen3moe_base.json $G/qwen3moe_q8.json && r="equal to f32" || r="differs from f32"
else r="refused: $(grep -o 'EMBER_KV=q8: .*' $G/qwen3moe_q8.log | head -1)"; fi
echo "qwen3moe EMBER_KV=q8: $r" | tee -a $out/gate.txt
echo CPU_GATE2_DONE
