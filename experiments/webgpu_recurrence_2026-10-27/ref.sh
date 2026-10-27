#!/bin/bash
# The references of the WebGPU recurrence gate (WSL), per document into $WORK/<name>/:
#   ref/  ember CPU: tools/ir_wasm_gate's native build at ember HEAD (BIN), EMBER_IR = the document, --ctx 1024:
#         ids.json (ir_device_gate's seeded prompts), ref_<i>.bin (prefill logits at every position), ref.jsonl
#         (argmax, token-by-token decode against prefill, greedy tokens). Documents declaring a context under 64
#         positions take 8 ids and 8 greedy tokens, as their WASM legs.
#   np/   keramos's numpy reference (tools/ir_numpy.py, a black-box oracle) on the first NPK of the same prompts:
#         np_ref.py. Rows: NAME|GGUF|DOC (- for the file's own)|K prompts|NPK numpy prompts.
# Resumes per prompt. Under the CPU budget, one core:
#   python3 scripts/cpu_budget.py run --session ember --cores 1 -- bash ref.sh [NAME ...]
set -u
H=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HOME/e6rec/bin}
WORK=${WORK:-$HOME/e6rec/gate}
K=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos/conformance
A=/mnt/c/ml_artifacts
D=$H/docs
M=$HOME/e6rec/models   # kimi_tiny relabelled "paperkiln" (rule 38; gguf_relabel.py): ember HEAD refuses the gpt2-labelled file
DOCS="
pk_recurrence|$K/models/pk_recurrence/pk_recurrence.gguf|-|40|40
pk_kimi_rec|$K/models/pk_kimi/pk_kimi.gguf|$K/recurrence/pk_kimi.ir.json|40|40
pk_srd_rec|$K/models/pk_srd/pk_srd.gguf|$K/recurrence/pk_srd.ir.json|40|40
pk_mamba_rec|$K/models/pk_mamba/pk_mamba.gguf|$K/recurrence/pk_mamba.ir.json|40|40
tp_mamba1_rec|$K/models/tp_mamba1/tp_mamba1.gguf|$K/recurrence/tp_mamba1.ir.json|40|40
kimi_tiny_rec|$M/kimi_tiny.gguf|$D/kimi_tiny_rec.ir.json|40|40
srd_tiny_rec|$A/srd_tiny/srd_tiny.gguf|$D/srd_tiny_rec.ir.json|40|40
mamba_tiny_rec|$A/mamba_tiny/mamba_tiny.gguf|$D/mamba_tiny_rec.ir.json|40|40
kimi_tiny_custom|$M/kimi_tiny.gguf|$D/kimi_tiny_custom.ir.json|40|40
mamba-130m_rec|$A/thirdparty/mamba-130m/mamba-130m-f32.gguf|$D/mamba-130m_rec.ir.json|10|2
"
sel=" $* "
echo "$DOCS" | grep -v '^$' | while IFS='|' read -r name gguf doc k npk; do
  [ "$sel" = "  " ] || [[ "$sel" == *" $name "* ]] || continue
  d=$WORK/$name
  mkdir -p "$d/ref" "$d/np"
  if [ "$doc" = "-" ]; then
    ctx=$(python3 "$H/gguf_doc.py" get "$gguf" | python3 -c 'import json,sys; print(json.load(sys.stdin)["model"]["ctx"])')
    irenv=""
  else
    ctx=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["ctx"])' "$doc")
    irenv="EMBER_IR=$doc"
  fi
  extra=""
  [ "$ctx" -lt 64 ] && extra="--len 8 --gen 8"
  env $irenv "$BIN/ir_wasm_gate" ref "$gguf" "$d/ref" --ctx 1024 --first "$k" $extra > "$d/ref.out" 2> "$d/ref.log"
  echo "$name ember CPU ref rc=$? ($(wc -l < "$d/ref/ref.jsonl") prompts)"
  python3 "$H/np_ref.py" "$gguf" "$doc" "$d/ref" "$d/np" "$npk" > "$d/np.log" 2>&1
  echo "$name numpy ref rc=$? ($(wc -l < "$d/np/ref.jsonl") prompts)"
done
