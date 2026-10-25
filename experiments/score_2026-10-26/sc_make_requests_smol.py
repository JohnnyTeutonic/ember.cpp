"""The /score gate's request sets (run in WSL): real benchmark items (bench_v1.jsonl, tokenised and
truncated as paperkiln's tools/bench_eval.py does), long prompts of 300-420 ids, and for the paperkiln
models over_ctx edges around T + 1. One set per tokeniser:
    python3 sc_make_requests.py OUT_DIR
writes OUT_DIR/{pk,pythia,qwen}_requests.jsonl (the gate set, ~200 each) and OUT_DIR/{pk,pythia}_bench256.jsonl
(256 HellaSwag requests, the throughput batch)."""
import json
import os
import random
import sys

R = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
sys.path.insert(0, R + "/microtorch/tools")
import bench_eval  # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
items = bench_eval.read_jsonl("/mnt/c/ml_artifacts/benchmarks/bench_v1.jsonl")
by = {}
for it in items:
    by.setdefault(it["set"], []).append(it)
rng = random.Random(20261026)
PICK = {"hellaswag": 10, "arc_easy": 8, "arc_challenge": 6, "piqa": 10, "winogrande": 10, "lambada": 20, "blimp": 15}
picked = []
for s, k in PICK.items():
    picked += rng.sample(by[s], k)
bench256 = rng.sample([x for x in by["hellaswag"] if x not in picked], 64)


class PkT:  # paperkiln's tokeniser (the 16k Llama 3-scheme vocabulary both paperkiln models use)
    def __init__(self, gguf, T):
        import carry5_eval
        t = carry5_eval.Tok(gguf)
        self.encode, self.eot, self.ctx = t.encode, t.eot, T


class HfT:
    def __init__(self, d, ctx, gguf=None):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(d, gguf_file=gguf) if gguf else AutoTokenizer.from_pretrained(d)
        self.eot = self.t.convert_tokens_to_ids("<|endoftext|>")
        self.ctx = ctx

    def encode(self, s):
        return self.t(s, add_special_tokens=False)["input_ids"]


def long_prompts(tok, n, lo, hi):
    """Prompts of lo..hi ids: HellaSwag contexts joined, then one of the last item's choices."""
    res = []
    pool = by["hellaswag"]
    for q in range(n):
        target = rng.randint(lo, hi)
        ids = [tok.eot]
        while True:
            it = rng.choice(pool)
            more = tok.encode(" " + it["context"])
            if len(ids) + len(more) > target:
                break
            ids += more
        c = tok.encode(rng.choice(it["choices"]))
        res.append((f"long/{q}", ids, c))
    return res


def write(path, reqs):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k, p, c in reqs:
            f.write(json.dumps({"key": k, "prompt_ids": p, "continuation_ids": c}) + "\n")
    lens = [len(p) + len(c) for _, p, c in reqs]
    print(f"{path}: {len(reqs)} requests, ids {min(lens)}..{max(lens)}, mean {sum(lens) / len(lens):.0f}")


sets = {  # the same draws as sc_make_requests.py (the rng runs through every set in this order), then SmolLM2
    "pk": PkT("/mnt/c/ml_artifacts/next_base/base/pk_base_v1.gguf", 512),
    "pythia": HfT("/mnt/c/ml_artifacts/thirdparty/pythia-160m", 2048),
    "qwen": HfT("/mnt/c/ml_artifacts/thirdparty/qwen2.5-0.5b-instruct", 2048),
    "smol": HfT("/mnt/c/ml_artifacts/thirdparty/smollm2-135m", 2048, gguf="SmolLM2-135M-f32.gguf"),
}
for name, tok in sets.items():
    if name == "smol":
        assert tok.eot == 0, tok.eot
    reqs, trunc = bench_eval.requests(picked, tok)
    reqs += long_prompts(tok, 8, 300, 420)
    if name == "pk":  # over_ctx edges at T = 512: P + C = T + 1 is scored, T + 2 and beyond are not
        T = tok.ctx
        base = long_prompts(tok, 1, 600, 700)[0][1]
        c = tok.encode(" and then the story went on for a while.")
        for tag, total in (("edge_T+1", T + 1), ("edge_T+2", T + 2), ("over_700", None)):
            p = base[:total - len(c)] if total else base
            reqs.append((f"ctx/{tag}", p, c))
        reqs.append(("ctx/edge_T+1_c1", base[:T], c[:1]))  # one continuation id, 512 prompt ids
    write(f"{out}/{name}_requests.jsonl", reqs)
    if name in ("pk", "pythia", "smol"):
        b, _ = bench_eval.requests(bench256, tok)
        write(f"{out}/{name}_bench256.jsonl", b)
