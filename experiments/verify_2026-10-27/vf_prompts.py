"""The two request sets of verify_2026-10-27 (deterministic; rerun to rebuild them byte for byte).

    python3 vf_prompts.py OUTDIR

chat40.jsonl: the 40 frozen chat prompts (the first 40 of tinychat_hf_prompts_v1.jsonl), first turn, 60 new tokens.
copy20.jsonl: 20 text-repeating requests built from the held-out TinyStories panel
  (microtorch/corpus/eval/tinystories_heldout_v1.txt, stories 0-19, each cut to its first three sentences):
  a conversation in which the assistant has told the passage twice (asked for it, then asked for it again word for
  word) and the user asks a third time, 80 new tokens. The reply can copy from its context, which is the case
  prompt-lookup drafting is built for (quoting, editing, extraction). The tiny models follow no bare "repeat this"
  instruction: in CPU trials on five of these passages (EMBER_SPEC=4) with the passage in the user turn, or told
  once by the assistant, both models wrote a story of their own; told twice, both copy it (draft acceptance
  pk_base_v1 89%, pick_e1_L16 65%).
all60.jsonl: chat40 then copy20 (one server run serves both; records carry their set)."""
import json
import os
import re
import sys

R = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
CHAT = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"
STORIES = R + "/microtorch/corpus/eval/tinystories_heldout_v1.txt"


def main(out):
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(CHAT, encoding="utf-8")][:40]
    with open(os.path.join(out, "chat40.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"set": "chat40", "user_input": r["prompt"], "max_new_tokens": 60}) + "\n")
    stories = [l.strip() for l in open(STORIES, encoding="utf-8") if l.strip()][:20]
    with open(os.path.join(out, "copy20.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for s in stories:
            sent = re.split(r"(?<=[.!?])\s+", s)
            passage = " ".join(sent[:3])
            again = "Please tell me that story again, word for word."
            f.write(json.dumps({"set": "copy20", "history": [{"user": "Tell me a short story.", "assistant": passage},
                                            {"user": again, "assistant": passage}],
                                "user_input": again,
                                "max_new_tokens": 80}) + "\n")
    with open(os.path.join(out, "all60.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for name in ("chat40.jsonl", "copy20.jsonl"):
            f.write(open(os.path.join(out, name), encoding="utf-8").read())


if __name__ == "__main__":
    main(sys.argv[1])
