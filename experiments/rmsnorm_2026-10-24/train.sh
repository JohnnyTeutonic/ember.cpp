#!/bin/bash
# Briefly train sp_L$1 on CPU with paperkiln's mtstudio (exports safetensors + gguf as the jobs do).
set -e
W=~/ember_wt/shape_parity
MT=~/pk_state_build/mtstudio
cd $W
$MT run $W/sp_L$1.json > $W/sp_L$1.train.log 2>&1
echo "train rc=$?"
ls -la $W/sp_L$1
