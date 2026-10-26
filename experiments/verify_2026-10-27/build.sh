#!/bin/bash
# The new source tree of verify_2026-10-27 on ext4, and its WSL builds (run in WSL, under the CPU budget):
#   bash build.sh cpu|cuda
# ~/vf/src is the ember working tree (no .git, experiments, deploy, docs) plus tools/vf_bench.cpp and a CMake
# target for it; stage.sh packs that tree as the VM's ember_new.tgz. cpu: tinyllama_server and the CPU gates'
# binaries (HAS_CUDA=OFF). cuda: a compile check for sm_75 (no GPU runs on the laptop).
set -e
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp
W=~/vf
mkdir -p $W/src
rsync -a --delete --exclude .git --exclude experiments --exclude 'build*' --exclude deploy --exclude docs $E/ $W/src/
mkdir -p $W/src/tools
cp $E/experiments/verify_2026-10-27/vf_bench.cpp $W/src/tools/vf_bench.cpp
grep -q vf_bench $W/src/CMakeLists.txt || printf '\nif(TARGET tinyllama_core AND CUDA_ENABLED)\n  add_executable(vf_bench tools/vf_bench.cpp)\n  target_link_libraries(vf_bench PRIVATE tinyllama_core)\nendif()\n' >> $W/src/CMakeLists.txt
export PATH=/usr/local/cuda/bin:$PATH
case "$1" in
  cpu)
    mkdir -p $W/build_cpu && cd $W/build_cpu
    [ -f CMakeCache.txt ] || cmake $W/src -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF > cmake.log 2>&1
    make -j4 tinyllama_server ir_logits ir_adapter ir_path_check state_tokens_test chat_template_check > build.log 2>&1 \
      || { tail -40 build.log; exit 1; } ;;
  cuda)
    mkdir -p $W/build_cuda && cd $W/build_cuda
    [ -f CMakeCache.txt ] || cmake $W/src -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 > cmake.log 2>&1
    grep -q "CUDA toolkit and cuBLAS found" cmake.log || { tail -20 cmake.log; exit 1; }
    make -j4 tinyllama_server vf_bench > build.log 2>&1 || { grep -n "error" build.log | head -40; exit 1; } ;;
esac
