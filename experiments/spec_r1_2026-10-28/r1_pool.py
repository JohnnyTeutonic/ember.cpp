"""Request pools for the self-output draft tables of spec_r1_2026-10-28 (deterministic; rerun to rebuild byte for byte).

    python3 r1_pool.py OUTDIR [N] [N_PK]

Each model's pool is N requests (default 4000; N_PK for pk_base_v1's, default N) drawn from the dialogue text it was trained on, so a table built
from the model's own replies to them can be built at export time from the training data alone:
  pick_pool.jsonl: pick_e1_L16's 13 human dialogue sources (draft_2026-10-27's dr_tables.sh list, dialogues.jsonl),
    N / 13 dialogues from each (seeded), in source order;
  pk_pool.jsonl:   pk_base_v1's dialogue sources, lever2's SODA (its first 100,000 dialogues) and OASST, N / 2 each.
A request is one user turn of a dialogue: the first, or (every second request) the second with the dialogue's first
exchange as history (when that exchange is under 600 characters); user turns over 400 characters are skipped;
60 new tokens. Requests whose user text equals any chat40, chatdev, copy20, carry40 or
carrydev user turn are skipped (none is expected: those come from TinyChat, TinyStories and Carry-5)."""
import json
import os
import random
import sys

A = "/mnt/c/ml_artifacts"
H = A + "/human_dialogue"
HUMAN = [f"{H}/hd_default/human_{s}_v1/dialogues.jsonl" for s in
         ("oasst", "convai2", "wow", "bst", "taskmaster", "multiwoz", "self_dialogue")] + \
        [f"{H}/hd_optin/human_{s}_v1/dialogues.jsonl" for s in
         ("topical_chat", "dolly", "quac", "empathetic", "dailydialog", "no_robots")]
PK = [(A + "/capacity/lever2/dialogue_soda_v1/dialogues.jsonl", 100000),
      (A + "/capacity/lever2/dialogue_oasst_v1/dialogues.jsonl", None)]


def read(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def turns(d):
    """[(user, assistant), ...] from a dialogues.jsonl row ({"messages": [{"role", "content"}...]})."""
    m = [x for x in d.get("messages", []) if x.get("role") in ("user", "assistant")]
    out = []
    for i in range(0, len(m) - 1, 2):
        if m[i]["role"] == "user" and m[i + 1]["role"] == "assistant":
            out.append((m[i]["content"].strip(), m[i + 1]["content"].strip()))
        else:
            break
    return out


def pool(sources, n, seed, exclude):
    rng = random.Random(seed)
    per = n // len(sources)
    rows = []
    for k, (path, limit) in enumerate(sources):
        ds = [t for t in (turns(d) for d in read(path, limit)) if t]
        rng.shuffle(ds)
        got = 0
        for t in ds:
            if got >= per:
                break
            second = got % 2 == 1 and len(t) >= 2 and len(t[0][0]) + len(t[0][1]) < 600
            user = t[1][0] if second else t[0][0]
            if not user or len(user) > 400 or user in exclude:
                continue
            hist = [{"user": t[0][0], "assistant": t[0][1]}] if second else []
            rows.append({"set": "pool", "src": os.path.basename(os.path.dirname(path)), "user_input": user,
                         "history": hist, "max_new_tokens": 60})
            got += 1
    return rows


def main(out, n, n_pk=None):
    here = os.path.dirname(os.path.abspath(__file__))
    prompts = os.path.join(out, "_eval")
    os.makedirs(prompts, exist_ok=True)
    sys.path.insert(0, os.path.join(here, "..", "draft_2026-10-27"))
    import dr_prompts
    dr_prompts.main(prompts)
    exclude = set()
    for name in ("chat40", "chatdev", "copy20", "carry40", "carrydev"):
        for line in open(os.path.join(prompts, name + ".jsonl"), encoding="utf-8"):
            r = json.loads(line)
            exclude.update(r.get("turns", []))
            if "user_input" in r:
                exclude.add(r["user_input"])
    for name, sources, seed in (("pick", [(p, None) for p in HUMAN], 1), ("pk", PK, 2)):
        rows = pool(sources, n if name == "pick" or n_pk is None else n_pk, seed, exclude)
        with open(os.path.join(out, name + "_pool.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(name, len(rows))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4000, int(sys.argv[3]) if len(sys.argv) > 3 else None)
