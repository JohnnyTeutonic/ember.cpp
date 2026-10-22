#!/usr/bin/env python3
"""Token ids of chat prompts through a Hugging Face tokenizer's chat template, for the long-prompt
gates (tools/thirdparty_ember_check.py --ids, tools/ir_device_gate --ids).

    python3 tools/chat_prompt_ids.py HF_DIR PROMPTS.jsonl OUT.json [--start 100] [--count 40] [--max-tokens 4096]
        [--min-tokens N]

PROMPTS.jsonl holds one {"prompt": ...} per line (paperkiln corpus/eval/chat_prompts_v1.jsonl: lines
100-139 by index are its UltraChat prompts, up to ~1,500 tokens). Each prompt becomes the ids of
[{"role": "user", "content": prompt}] with the generation prompt added, as paperkiln's
tools/thirdparty_parity.py forms them; prompts longer than --max-tokens, or shorter than --min-tokens
(e.g. 513 to keep only prompts past a 512-token window), are dropped. OUT.json is a
list of id lists; the lengths are printed.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--min-tokens", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf_dir)
    lines = open(a.prompts, encoding="utf-8").read().splitlines()[a.start:a.start + a.count]
    seqs = []
    for line in lines:
        p = json.loads(line)["prompt"]
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        ids = list(ids["input_ids"] if isinstance(ids, dict) else ids)
        if a.min_tokens <= len(ids) <= a.max_tokens:
            seqs.append(ids)
    json.dump(seqs, open(a.out, "w"))
    lens = sorted(len(s) for s in seqs)
    print(json.dumps({"prompts": len(seqs), "min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1]}))


if __name__ == "__main__":
    main()
