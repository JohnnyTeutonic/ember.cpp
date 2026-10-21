"""WebGPU prototype fixture: ember CPU first-token logits (p = 1 rows of the repetition-penalty rule
check, chat_sftfinal_v3) with their exact prompt ids. Writes fixture.json beside this script."""
import json
import os

src = "/home/jonat/ember_ttft/repeat"
rows = [list(map(float, line.split())) for line in open(f"{src}/rule_logits.txt") if line.strip()]
meta = [json.loads(line) for line in open(f"{src}/rule_ids.jsonl")]
out = [{"i": m["i"], "prompt_ids": m["prompt_ids"], "logits": r} for m, r in zip(meta, rows) if m["p"] == 1.0]
here = os.path.dirname(os.path.abspath(__file__))
json.dump({"model": "chat_sftfinal_v3.gguf", "prompts": out}, open(os.path.join(here, "fixture.json"), "w"))
print(len(out), "prompts,", len(out[0]["logits"]), "logits each")
