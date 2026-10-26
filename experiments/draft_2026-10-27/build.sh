#!/bin/bash
# The ember working tree on ext4 (~/dr/src) and its builds (run in WSL, under the CPU budget): cpu into ~/dr/<dir>
# (default build_cpu; this work used build_cpu for HEAD 08ead73 and build_new for the change), cuda a compile check.
#   bash build.sh cpu [dir]    or    bash build.sh cuda
set -e
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
E=$R/ember.cpp
W=~/dr
mkdir -p $W/src
rsync -a --delete --exclude .git --exclude experiments --exclude 'build*' --exclude deploy --exclude docs $E/ $W/src/
export PATH=/usr/local/cuda/bin:$PATH
case "${1:-cpu}" in
  cpu)
    mkdir -p $W/${2:-build_cpu} && cd $W/${2:-build_cpu}
    [ -f CMakeCache.txt ] || cmake $W/src -DCMAKE_BUILD_TYPE=Release -DHAS_CUDA=OFF > cmake.log 2>&1
    python3 $R/scripts/cpu_budget.py run --session ember --job "draft: CPU build" --cores 4 \
      --resume "bash build.sh cpu ${2:-build_cpu} (incremental)" -- make -j4 tinyllama_server ir_logits ir_adapter ir_path_check state_tokens_test chat_template_check > build.log 2>&1 \
      || { tail -40 build.log; exit 1; } ;;
  cuda)
    mkdir -p $W/build_cuda && cd $W/build_cuda
    [ -f CMakeCache.txt ] || cmake $W/src -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 > cmake.log 2>&1
    grep -q "CUDA toolkit and cuBLAS found" cmake.log || { tail -20 cmake.log; exit 1; }
    python3 $R/scripts/cpu_budget.py run --session ember --job "draft: CUDA compile check" --cores 4 \
      --resume "bash build.sh cuda (incremental)" -- make -j4 tinyllama_server > build.log 2>&1 \
      || { grep -n "error" build.log | head -40; exit 1; } ;;
esac
echo "built $1"
