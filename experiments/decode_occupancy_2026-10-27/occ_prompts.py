"""The request sets of decode_occupancy_2026-10-27 (deterministic; rerun to rebuild them byte for byte).

    python3 occ_prompts.py OUTDIR

all60.jsonl: ../verify_2026-10-27's chat40 + copy20 (vf_prompts.py), short contexts (73-225 tokens with the reply).
c12_40.jsonl: 40 Carry-12 v1 items (microtorch/corpus/eval/carry12_v1.jsonl, rows 0, 36, ..., 1404): the first 11
  exchanges (user, gold reply) as history and the 12th user turn, 60 new tokens; 245-295 prompt tokens (305-355 with the reply, llama3_bpe count).
long40.jsonl: the same 40 items, each followed by the first 3 exchanges of the next item (row + 1) as more history
  and that item's 4th user turn, 48 new tokens; 320-430 prompt tokens (368-478 with the reply), inside the picks' 512.
"""
import json
import os
import subprocess
import sys

R = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
C12 = R + "/microtorch/corpus/eval/carry12_v1.jsonl"
HERE = os.path.dirname(os.path.abspath(__file__))


def main(out):
    os.makedirs(out, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(HERE, "..", "verify_2026-10-27", "vf_prompts.py"), out], check=True)
    rows = [json.loads(l) for l in open(C12, encoding="utf-8")]
    picks = list(range(0, 40 * 36, 36))
    with open(os.path.join(out, "c12_40.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for i in picks:
            t = rows[i]["turns"]
            hist = [{"user": x["user"], "assistant": x["gold"]} for x in t[:11]]
            f.write(json.dumps({"set": "c12_40", "id": rows[i]["id"], "history": hist, "user_input": t[11]["user"],
                                "max_new_tokens": 60}) + "\n")
    with open(os.path.join(out, "long40.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for i in picks:
            a, b = rows[i]["turns"], rows[i + 1]["turns"]
            hist = [{"user": x["user"], "assistant": x["gold"]} for x in a + b[:3]]
            f.write(json.dumps({"set": "long40", "id": rows[i]["id"] + "+" + rows[i + 1]["id"], "history": hist,
                                "user_input": b[3]["user"], "max_new_tokens": 48}) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
