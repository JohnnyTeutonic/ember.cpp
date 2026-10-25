"""mtstudio specs for the gate, re-pointed at the local files (out_dir, name, data.vocab).
    sc_specs.py OUT_DIR"""
import json
import os
import sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
E = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments"
pick = "/mnt/c/ml_artifacts/tiny_chat/e1/pick_e1_L16"
s = json.load(open(pick + "/spec.json"))
# model.safetensors carries the AdamW state as well: sc_strip.py's weights-only copy (sc_gate_cpu.sh).
s["out_dir"], s["name"] = os.path.expanduser("~/ember_score/work/pick_w"), "pick_e1_L16"
s["data"]["vocab"] = pick + "/model.gguf"
json.dump(s, open(out + "/pick_e1_L16.json", "w"), indent=1)
s = json.load(open(E + "/pk_base_v1_2026-10-24/specs/pk_base_v1.json"))
json.dump(s, open(out + "/pk_base_v1.json", "w"), indent=1)
print(sorted(os.listdir(out)))
