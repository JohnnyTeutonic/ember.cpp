#!/usr/bin/env bash
# WSL, 1 thread: the default mode unchanged on the CPU. ir_logits (prefill rows and token-by-token decode) from
# ember HEAD and from HEAD + R2, byte for byte, on five IR families; EMBER_KV=f32 equal to unset; the formats
# differ from f32 (they are in force). Then the CUDA compile check of HEAD + R2 (compilation only).
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/r2kv
G=~/r2kv/gate3
T=/mnt/c/ml_artifacts/thirdparty
mkdir -p $G
for t in base new; do
  if [ ! -x $G/$t/build/ir_logits ]; then
    rm -rf $G/$t; mkdir -p $G/$t/src $G/$t/build
    tar xzf $R/stage2/$( [ $t = base ] && echo ember_base.tgz || echo ember.tgz ) -C $G/$t/src
    cd $G/$t/build
    cmake $G/$t/src/ember -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF > cmake.log 2>&1 || { tail -20 cmake.log; exit 1; }
    make ir_logits -j1 > make.log 2>&1 || { tail -30 make.log; exit 1; }
  fi
  echo "$t build ok"
done
python3 -c "
import json, random
r = random.Random(7)
json.dump([[r.randrange(100, 50000) for _ in range(24)] for _ in range(4)], open('$G/seqs.json', 'w'))"
out=$R/cpu_gate3; mkdir -p $out; : > $out/gate.txt
same() { if [ -s "$1" ] && cmp -s "$1" "$2"; then r=IDENTICAL; else r=DIFFER; fi
         echo "$3 $r $(sha256sum < "$1" | cut -c1-16) $(sha256sum < "$2" | cut -c1-16)" | tee -a $out/gate.txt; }
for m in qwen25:qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-fp16.gguf qwen3:qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf \
         gemma3:gemma-3-270m-it/gemma-3-270m-it-f32.gguf pythia:pythia-160m/pythia-160m-f32.gguf \
         mamba:mamba-130m/mamba-130m-f32.gguf; do
  n=${m%%:*}; g=$T/${m#*:}

  [ -s "$g" ] || { echo "$n: no gguf" | tee -a $out/gate.txt; continue; }
  for t in base new; do
    [ -s $G/${n}_$t.json ] || EMBER_IR_CTX=256 $G/$t/build/ir_logits $g < $G/seqs.json > $G/${n}_$t.json 2> $G/${n}_$t.log
  done
  same $G/${n}_base.json $G/${n}_new.json "$n default: HEAD vs HEAD+R2"
  [ -s $G/${n}_f32.json ] || EMBER_KV=f32 EMBER_IR_CTX=256 $G/new/build/ir_logits $g < $G/seqs.json > $G/${n}_f32.json 2> $G/${n}_f32.log
  same $G/${n}_base.json $G/${n}_f32.json "$n EMBER_KV=f32 vs HEAD"
  [ -s $G/${n}_q8.json ] || EMBER_KV=q8 EMBER_IR_CTX=256 $G/new/build/ir_logits $g < $G/seqs.json > $G/${n}_q8.json 2> $G/${n}_q8.log
  if cmp -s $G/${n}_base.json $G/${n}_q8.json; then echo "$n EMBER_KV=q8: equal to f32 (no kv cache, or q8 not in force)"; else echo "$n EMBER_KV=q8: differs from f32 ($(tail -c 160 $G/${n}_q8.log | tr '\n' ' '))"; fi | tee -a $out/gate.txt
done
echo "--- CUDA compile check of HEAD + R2 (nvcc, no GPU use)" | tee -a $out/gate.txt
cd $G/new/src/ember
JI=$(dirname "$(dirname "$(find $G/new/build /usr/include /usr/local/include -name json.hpp -path '*nlohmann*' 2>/dev/null | head -1)")")
/usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_89 -DHAS_CUDA -I. -c kv_quant.cu -o $G/kv_quant.o && echo "nvcc kv_quant.cu OK" | tee -a $out/gate.txt
g++ -std=c++17 -fsyntax-only -ffp-contract=off -DHAS_CUDA -I. -I"$JI" -I/usr/local/cuda/include graph_ir.cpp && echo "g++ graph_ir.cpp (HAS_CUDA) OK" | tee -a $out/gate.txt
g++ -std=c++17 -fsyntax-only -DHAS_CUDA -I. -I"$JI" -I/usr/local/cuda/include tools/kv_quant_eval.cpp && echo "g++ kv_quant_eval.cpp (HAS_CUDA) OK" | tee -a $out/gate.txt
bash /mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/1aba05f9-b164-4547-9536-e677c2310edf/scratchpad/r2kv/r2_cpu_gate3b.sh
echo CPU_GATE_DONE
