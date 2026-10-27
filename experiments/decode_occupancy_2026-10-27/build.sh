#!/bin/bash
# The source tree of decode_occupancy_2026-10-27 on ext4, and its WSL builds (run in WSL, under the CPU budget):
#   bash build.sh tree|cpu|cuda|bench
# ~/occ/src is the ember working tree (no .git, experiments, deploy, docs) plus tools/occ_step.cpp and a CMake
# target for it; stage.sh packs that tree as the VM's ember_new.tgz. cpu: tinyllama_server (HAS_CUDA=OFF).
# cuda: a compile check for sm_89 (no GPU runs on the laptop), with occ_step. bench: occ_bench and
# decode_attention_test with nvcc for sm_89 (compile only).
set -e
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp
X=$E/experiments/decode_occupancy_2026-10-27
W=~/occ
tree() {
  mkdir -p $W/src
  rsync -a --delete --exclude .git --exclude experiments --exclude 'build*' --exclude deploy --exclude docs $E/ $W/src/
  mkdir -p $W/src/tools
  cp $X/occ_step.cpp $W/src/tools/occ_step.cpp
  grep -q occ_step $W/src/CMakeLists.txt || printf '\nif(TARGET tinyllama_core AND CUDA_ENABLED)\n  add_executable(occ_step tools/occ_step.cpp)\n  target_link_libraries(occ_step PRIVATE tinyllama_core)\nendif()\n' >> $W/src/CMakeLists.txt
}
export PATH=/usr/local/cuda/bin:$PATH
case "$1" in
  tree) tree ;;
  cpu)
    tree
    mkdir -p $W/build_cpu && cd $W/build_cpu
    [ -f CMakeCache.txt ] || cmake $W/src -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF > cmake.log 2>&1
    make -j4 tinyllama_server > build.log 2>&1 || { tail -40 build.log; exit 1; } ;;
  cuda)
    tree
    mkdir -p $W/build_cuda && cd $W/build_cuda
    [ -f CMakeCache.txt ] || cmake $W/src -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 > cmake.log 2>&1
    grep -q "CUDA toolkit and cuBLAS found" cmake.log || { tail -20 cmake.log; exit 1; }
    make -j4 tinyllama_server occ_step > build.log 2>&1 || { grep -n "error" build.log | head -40; exit 1; } ;;
  bench)
    tree
    mkdir -p $W/bench && cd $W/src
    nvcc -O3 -std=c++17 -arch=sm_89 -I. $X/occ_bench.cu -o $W/bench/occ_bench > $W/bench/occ_bench.log 2>&1 \
      || { cat $W/bench/occ_bench.log | head -40; exit 1; }
    nvcc -O2 -std=c++17 -DHAS_CUDA -arch=sm_89 -I. tools/decode_attention_test.cu cuda_kernels.cu logger.cpp -lcublas \
      -o $W/bench/decode_attention_test > $W/bench/dat.log 2>&1 || { grep -n "error" $W/bench/dat.log | head -40; exit 1; }
    ls -la $W/bench ;;
esac
