#!/bin/bash
# paperkiln's tools/bench_eval.py on a slice (HellaSwag, ARC-Easy, PIQA, LAMBADA, every 200th item) of bench_v1.jsonl:
# SmolLM2-135M through ember's /score (CPU, the F32 GGUF) and through HF transformers float32 (CPU, the hub's
# pinned revision), then bench_eval compare and report. Runs inside one budget job (sc_smol.sh).
#   bash sc_smol_bench.sh <ember bin> <gguf> <port> <work>
BIN=$1; G=$2; PORT=$3; W=$4
PK=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
HF=smol=HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2
SL="--items /mnt/c/ml_artifacts/benchmarks/bench_v1.jsonl --sets hellaswag,arc_easy,piqa,lambada --every 200"
mkdir -p $W/bench_cwd
if [ ! -d $W/bench_ember/smol ] || [ ! -s $W/bench_ember.done ]; then
  (cd $W/bench_cwd && exec $BIN $G $PORT 127.0.0.1 /nonexistent_www 0 > $W/bench_server.log 2>&1) &
  SRV=$!
  for i in $(seq 1 300); do curl -s http://127.0.0.1:$PORT/health > /dev/null && break; sleep 1; done
  python3 $PK/tools/bench_eval.py run $SL --out $W/bench_ember --scorer ember --ember-url http://127.0.0.1:$PORT --hf $HF \
    > $W/bench_ember.log 2>&1 && touch $W/bench_ember.done
  kill $SRV; wait $SRV 2> /dev/null
fi
[ -s $W/bench_hf.done ] || { python3 $PK/tools/bench_eval.py run $SL --out $W/bench_hf --scorer hf --hf $HF --device cpu \
  > $W/bench_hf.log 2>&1 && touch $W/bench_hf.done; }
python3 $PK/tools/bench_eval.py compare $W/bench_hf/smol $W/bench_ember/smol --json $W/bench_compare.json
python3 $PK/tools/bench_eval.py report $W/bench_hf $W/bench_ember --md $W/bench_report.md > /dev/null; cat $W/bench_report.md
