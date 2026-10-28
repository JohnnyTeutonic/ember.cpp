#!/bin/bash
# The E4 CPU gates at one commit (WSL, under scripts/cpu_budget.py, 1 core; ember.cpp experiments/e4_gemma2_phi3_2026-10-28):
#   bash e4_cpu_gates.sh COMMIT BASE_BUILD OUT
# 1. A pinned CPU-only build from a git archive of COMMIT (../ir_rev12_2026-10-26/build_pinned.sh: ir_adapter,
#    ir_logits, tinyllama_core).
# 2. keramos v1.5.0's suite against it (../ir_rev12_2026-10-26/run_suite.sh): OUT/suite_v1.5.0_cpu.json.
# 3. HF on the synthetic files (e4_hf_gate.py): OUT/hf_gate_cpu.json.
# 4. The C++ builder against gguf_ir.py (e4_builder.py): OUT/builder_equality.json.
# 5. Regression bits: ir_logits of BASE_BUILD (the commit before this work) and of the pinned build on the suite's
#    earlier third-party models (tools/ir_logits aborts on the suite's paperkiln files in both builds; the suite covers them), the same seeded ids: OUT/regress_bits.txt (equal or DIFFER per file).
# Every step resumes by rerunning (the build is incremental, the rest is minutes).
set -u
C=${1:?commit}; BASE=${2:?base build dir}; OUT=${3:?out dir}
H=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments
E=$(cd "$(dirname "$0")" && pwd)  # this folder (any checkout)
K=~/keramos_v1.5.0
mkdir -p "$OUT"
NJ=1 bash $H/ir_rev12_2026-10-26/build_pinned.sh "$C" ir_adapter ir_logits tinyllama_core | tee "$OUT/build.txt"
FULL=$(git -C /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp rev-parse "$C")
B=~/ember_adapter_${FULL:0:7}/build
bash $H/ir_rev12_2026-10-26/run_suite.sh v1.5.0 $B/ir_adapter "$OUT/suite_v1.5.0_cpu.json" | tee "$OUT/suite.txt"
python3 $E/e4_hf_gate.py $B/ir_adapter /mnt/c/ml_artifacts/thirdparty/e4_synth "$OUT/hf_gate_cpu.json" | tee "$OUT/hf_gate.txt"
python3 $E/e4_builder.py $K $B ~/e4a_work/builder_$C /mnt/c/ml_artifacts/thirdparty/e4_synth/gemma2.gguf \
  /mnt/c/ml_artifacts/thirdparty/e4_synth/phi3_longrope.gguf /mnt/c/ml_artifacts/thirdparty/e4_synth/phi3_swa.gguf \
  $K/conformance/models/tp_gemma2/tp_gemma2.gguf $K/conformance/models/tp_phi3/tp_phi3.gguf | tee "$OUT/builder.txt"
cp ~/e4a_work/builder_$C/builder_equality.json "$OUT/"
: > "$OUT/regress_bits.txt"
python3 -c "import json, random; r = random.Random(0); print(json.dumps([[r.randrange(24) for _ in range(20)] for _ in range(4)]))" > ~/e4a_work/regress_ids.json
for m in tp_gemma3 tp_qwen3 tp_qwen3_hd16 tp_llama_hd16 tp_mistral tp_mistral_hd16 tp_gptneox tp_mamba1; do
  G=$K/conformance/models/$m/$m.gguf
  env -u EMBER_IR $BASE/ir_logits $G < ~/e4a_work/regress_ids.json > ~/e4a_work/rb_base_$m.json 2> /dev/null
  env -u EMBER_IR $B/ir_logits $G < ~/e4a_work/regress_ids.json > ~/e4a_work/rb_head_$m.json 2> /dev/null
  echo "$m $(cmp -s ~/e4a_work/rb_base_$m.json ~/e4a_work/rb_head_$m.json && echo equal || echo DIFFER) $(wc -c < ~/e4a_work/rb_head_$m.json) bytes" | tee -a "$OUT/regress_bits.txt"
done
