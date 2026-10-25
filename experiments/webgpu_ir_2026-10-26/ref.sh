#!/bin/bash
# ember CPU references for the WebGPU IR gate (WSL): tools/ir_wasm_gate's native build (BIN, default
# ~/wasmfam/bin, built by ../wasm_families_2026-10-25/build_gate.sh) per family into
# $WORK/FAM/ref (ids.json, ref.jsonl, ref_<i>.bin), with EMBER_IR_CTX 1024. Families whose IR declares a
# context under 64 positions (recursion, MoR, ar3) take 8 ids and 8 greedy tokens, as their WASM legs.
# expr: gpt2_tiny with its GELU as an expr formula, and with its final layernorm as an expr norm + scale
# (expr_doc.py), served through EMBER_IR. Resumes per prompt. Run under the CPU budget, one core:
#   python3 scripts/cpu_budget.py run --session ember --cores 1 -- bash ref.sh [FAM ...]
set -u
H=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HOME/wasmfam/bin}
WORK=${WORK:-$HOME/wgate}
A=/mnt/c/ml_artifacts
T=$A/thirdparty
FAMILIES="
kimi_tiny|$A/kimi_tiny/kimi_tiny.gguf|40
srd_tiny|$A/srd_tiny/srd_tiny.gguf|40
attnres_tiny|$A/attnres_tiny/attnres_tiny.gguf|40
flex_highway_tiny|$A/flex_highway_tiny/flex_highway_tiny.gguf|40
mamba_tiny|$A/mamba_tiny/mamba_tiny.gguf|40
gpt2_tiny|$A/gpt2_tiny/gpt2_tiny.gguf|40
swa_tiny|$A/swa_tiny/swa_tiny.gguf|40
moe_tiny_v3|$A/moe_tiny/moe_tiny_v3.gguf|40
rec_b2x4|$A/recursion_gate/b2x4/r.gguf|40
rec_p1_b2x3_c1|$A/recursion_gate/p1_b2x3_c1/r.gguf|40
rec_p1_b2x3_c1_inject|$A/recursion_gate/p1_b2x3_c1_inject/r.gguf|40
mor3|$A/mor_gate/mor3/mor3.gguf|40
mor2p0|$A/mor_gate/mor2p0/mor2p0.gguf|40
state4|$A/state_gate/state4/state4.gguf|40
state4_q8_0|$A/state_gate/state4/state4_q8_0.gguf|40
ar3|$A/attnres_gate/ar3/ar3.gguf|40
ar3_q8_0|$A/attnres_gate/ar3/ar3_q8_0.gguf|40
synthetic-gpt2|$HOME/ember_wt/bcheck_gpt2/gpt2.gguf|40
synthetic-mistral-swa|$HOME/ember_wt/bcheck_mistral_swa/mistral_swa.gguf|40
gpt2_tiny_expr|$A/gpt2_tiny/gpt2_tiny.gguf|40
gpt2_tiny_exprnorm|$A/gpt2_tiny/gpt2_tiny.gguf|40
pythia-160m|$T/pythia-160m/pythia-160m-f32.gguf|40
mamba-130m|$T/mamba-130m/mamba-130m-f32.gguf|40
qwen3-0.6b|$T/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf|10
"
sel=" $* "
python3 "$H/expr_doc.py" "$WORK/docs" > /dev/null
echo "$FAMILIES" | grep -v '^$' | while IFS='|' read -r fam gguf k; do
  [ "$sel" = "  " ] || [[ "$sel" == *" $fam "* ]] || continue
  d=$WORK/$fam
  mkdir -p "$d/ref"
  extra=""
  ctx=$(python3 "$H/ir_ctx.py" "$gguf")
  [ "$ctx" -lt 64 ] && extra="--len 8 --gen 8"
  irenv=""
  case $fam in gpt2_tiny_expr*) irenv="EMBER_IR=$WORK/docs/${fam}.ir.json";; esac
  env $irenv "$BIN/ir_wasm_gate" ref "$gguf" "$d/ref" --ctx 1024 --first "$k" $extra > "$d/ref.out" 2> "$d/ref.log"
  echo "$fam ref rc=$? ($(wc -l < "$d/ref/ref.jsonl") prompts)"
done
