"""The request sets of draft_2026-10-27 (deterministic; rerun to rebuild them byte for byte).

    python3 dr_prompts.py OUTDIR

Evaluation sets (the measured numbers):
  chat40.jsonl, copy20.jsonl: verify_2026-10-27's two sets (vf_prompts.py, unchanged).
  carry40.jsonl: multi-turn conversations whose history repeats: the 40 Carry-5 items of the spot-check
    (carry5_v1.jsonl items 0, 26, ..., 1014), each served as carry5_eval serves its self turns: five turns, each
    request carrying the earlier user turns and the model's own replies so far as history, 24 new tokens. One line
    per item ({"set", "item", "turns": [user...], "max_new_tokens"}); the runner builds the history from its replies.
Development sets (the drafter's settings were chosen on these only):
  chatdev.jsonl: tinychat_hf_prompts_v1 prompts 40-199 (the 160 not in chat40), first turn, 60 new tokens.
  carrydev.jsonl: Carry-5 items 13, 39, ..., 1027 (40 items, none in carry40), as carry40."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verify_2026-10-27"))
R = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
CHAT = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"
CARRY5 = R + "/microtorch/corpus/eval/carry5_v1.jsonl"


def carry(items, name):
    return [{"set": name, "item": it["id"], "turns": [t["user"] for t in it["turns"]], "max_new_tokens": 24}
            for it in items]


def write(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(out):
    import vf_prompts
    vf_prompts.main(out)
    items = [json.loads(x) for x in open(CARRY5, encoding="utf-8") if x.strip()]
    write(os.path.join(out, "carry40.jsonl"), carry(items[::26][:40], "carry40"))
    write(os.path.join(out, "carrydev.jsonl"), carry(items[13::26][:40], "carrydev"))
    rows = [json.loads(x) for x in open(CHAT, encoding="utf-8")][40:200]
    write(os.path.join(out, "chatdev.jsonl"),
          [{"set": "chatdev", "user_input": r["prompt"], "max_new_tokens": 60} for r in rows])
    with open(os.path.join(out, "eval100.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for name in ("chat40.jsonl", "copy20.jsonl", "carry40.jsonl"):
            f.write(open(os.path.join(out, name), encoding="utf-8").read())
    with open(os.path.join(out, "dev.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for name in ("chatdev.jsonl", "carrydev.jsonl"):
            f.write(open(os.path.join(out, name), encoding="utf-8").read())


if __name__ == "__main__":
    main(sys.argv[1])
