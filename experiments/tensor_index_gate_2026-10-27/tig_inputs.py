"""ar_rec's 40 score_in sequences (prompt + numpy's greedy continuation) and prompt lengths, in the input form of
tools/state_device_gate, tools/state_wasm_gate and ../attnres_x5_2026-10-25/x5_device_logits.cpp: OUT/ar_rec_in.json.
    python3 tig_inputs.py OUT_DIR"""
import json
import os
import sys

D = "/mnt/c/ml_artifacts/attnres_gate_rec/ar_rec"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
ins = [json.loads(x) for x in open(os.path.join(D, "score_in.jsonl")) if x.strip()]
json.dump({"seqs": [r["prompt_ids"] + r["continuation_ids"] for r in ins], "prompt_len": [len(r["prompt_ids"]) for r in ins]},
          open(os.path.join(out, "ar_rec_in.json"), "w"))
print(len(ins), "sequences")
