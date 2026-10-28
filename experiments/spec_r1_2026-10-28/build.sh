#!/bin/bash
# The tree of this work on ext4 (~/r1/src) and its builds (run in WSL, under the CPU budget, one core):
#   bash build.sh tree         ~/r1/src only
#   bash build.sh cpu [dir]    the CPU server (default ~/r1/build_cpu)
#   bash build.sh cuda         a CUDA compile check of tinyllama_server and r1_bench (sm_75; no GPU is used)
# ~/r1/src is ember main (the archive ~/r1/in/ember_head.tgz, `git archive HEAD` without experiments, deploy and
# docs) with this work's engine files (files.txt) copied from the working tree, plus tools/r1_bench.cpp and a CMake
# target for it. Other sessions' uncommitted work in the shared tree stays out. Unchanged files keep their times,
# so builds stay incremental.
set -e
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp
X=$E/experiments/spec_r1_2026-10-28
W=~/r1
HT=${R1_HEAD_TGZ:-$W/in/ember_head.tgz}
rm -rf $W/tree.tmp && mkdir -p $W/tree.tmp $W/src
tar xzf $HT -C $W/tree.tmp
for f in $(sed 's/\r$//' $X/files.txt); do cp $E/$f $W/tree.tmp/$f; done
cp $X/r1_bench.cpp $W/tree.tmp/tools/r1_bench.cpp
printf '\nif(TARGET tinyllama_core AND CUDA_ENABLED)\n  add_executable(r1_bench tools/r1_bench.cpp)\n  target_link_libraries(r1_bench PRIVATE tinyllama_core)\nendif()\n' >> $W/tree.tmp/CMakeLists.txt
rsync -a --checksum --delete $W/tree.tmp/ $W/src/
rm -rf $W/tree.tmp
export PATH=/usr/local/cuda/bin:$PATH
case "${1:-cpu}" in
  tree) ;;
  cpu)
    mkdir -p $W/${2:-build_cpu} && cd $W/${2:-build_cpu}
    [ -f CMakeCache.txt ] || cmake $W/src -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF > cmake.log 2>&1
    python3 $R/scripts/cpu_budget.py run --session ember --job "spec R1: CPU build" --cores 1 \
      --resume "bash build.sh cpu ${2:-build_cpu} (incremental)" -- make -j1 tinyllama_server > build.log 2>&1 \
      || { tail -40 build.log; exit 1; } ;;
  cuda)
    mkdir -p $W/build_cuda && cd $W/build_cuda
    [ -f CMakeCache.txt ] || cmake $W/src -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 > cmake.log 2>&1
    grep -q "CUDA toolkit and cuBLAS found" cmake.log || { tail -20 cmake.log; exit 1; }
    python3 $R/scripts/cpu_budget.py run --session ember --job "spec R1: CUDA compile check" --cores 1 \
      --resume "bash build.sh cuda (incremental)" -- make -j1 tinyllama_server r1_bench > build.log 2>&1 \
      || { grep -n "error" build.log | head -40; exit 1; } ;;
esac
echo "built ${1:-cpu}"
