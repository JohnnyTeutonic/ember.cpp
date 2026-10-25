"""score_in.jsonl -> {"seqs": [prompt + continuation, ...], "prompt_len": [P, ...]} (tp_session_logits and
tools/state_device_gate input).    python3 tp_inputs.py score_in.jsonl OUT.json"""
import json
import sys

ins = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
json.dump({"seqs": [r["prompt_ids"] + r["continuation_ids"] for r in ins], "prompt_len": [len(r["prompt_ids"]) for r in ins]},
          open(sys.argv[2], "w"))
print(len(ins), "sequences ->", sys.argv[2])
