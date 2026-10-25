"""Compare two x5_plain_collect.py runs (two builds, or two backends) request by request.
    python3 x5_plain_compare.py A_DIR B_DIR      (prints JSON; exit 0 when every field is bit-equal)
Per set (tinychat 40, carry5 320): replies + stop equal, ids equal, prompt ids equal, the dump of every sampled
position's logits equal (sha256 of the nine-digit text, so bit for bit), the first sampled position's logits
bit-equal (from first_logits.f32) and their worst |d|; the first divergent requests."""
import json
import os
import sys

import numpy as np


def load(d):
    recs = [json.loads(x) for x in open(d + "/records.jsonl", encoding="utf-8") if x.strip()]
    meta = json.load(open(d + "/meta.json"))
    fl = np.fromfile(d + "/first_logits.f32", dtype=np.float32).reshape(len(recs), meta["vocab"])
    return recs, fl, meta


A, B = sys.argv[1], sys.argv[2]
ra, fa, ma = load(A)
rb, fb, mb = load(B)
assert [r["key"] for r in ra] == [r["key"] for r in rb]
rep = {"a": A, "b": B, "a_bin": ma["bin"], "b_bin": mb["bin"], "gguf": ma["gguf"], "gpu_layers": [ma["gpu_layers"], mb["gpu_layers"]],
       "sets": {}}
ok = True
for s in ("tinychat", "carry5"):
    idx = [i for i, r in enumerate(ra) if r["set"] == s]
    n = len(idx)
    d = {"n": n}
    d["replies_identical"] = sum(ra[i]["reply"] == rb[i]["reply"] and ra[i]["stop"] == rb[i]["stop"] for i in idx)
    d["ids_identical"] = sum(ra[i]["ids"] == rb[i]["ids"] for i in idx)
    d["prompt_ids_identical"] = sum(ra[i]["prompt_ids"] == rb[i]["prompt_ids"] for i in idx)
    d["all_positions_logits_bit_equal"] = sum(ra[i]["dump_sha256"] == rb[i]["dump_sha256"] for i in idx)
    d["positions"] = sum(ra[i]["positions"] for i in idx)
    eq = [bool(np.array_equal(fa[i].view(np.uint32), fb[i].view(np.uint32))) for i in idx]
    d["first_token_logits_bit_equal"] = sum(eq)
    d["first_token_logits_worst_abs"] = float(max(np.abs(fa[i].astype(np.float64) - fb[i]).max() for i in idx))
    d["divergent"] = [{"key": ra[i]["key"], "a": ra[i]["reply"][:80], "b": rb[i]["reply"][:80]}
                      for i in idx if ra[i]["dump_sha256"] != rb[i]["dump_sha256"] or ra[i]["ids"] != rb[i]["ids"]][:10]
    if os.path.exists(A + "/all_logits.f32") and os.path.exists(B + "/all_logits.f32"):
        xa = np.memmap(A + "/all_logits.f32", dtype=np.float32, mode="r").reshape(-1, ma["vocab"])
        xb = np.memmap(B + "/all_logits.f32", dtype=np.float32, mode="r").reshape(-1, mb["vocab"])
        beq, worst, wrel, top1, rows = 0, 0.0, 0.0, 0, 0
        for i in idx:
            if ra[i]["ids"] != rb[i]["ids"]:
                continue  # positions line up only while the ids agree
            pa = np.asarray(xa[ra[i]["row0"]:ra[i]["row0"] + ra[i]["positions"]], dtype=np.float64)
            pb = np.asarray(xb[rb[i]["row0"]:rb[i]["row0"] + rb[i]["positions"]], dtype=np.float64)
            beq += int(np.all(pa == pb, axis=1).sum())
            diff = np.abs(pa - pb).max(1)
            worst = max(worst, float(diff.max()))
            wrel = max(wrel, float((diff / (pb.max(1) - pb.min(1))).max()))
            top1 += int((pa.argmax(1) == pb.argmax(1)).sum())
            rows += len(pa)
        d["every_position"] = {"rows": rows, "bit_equal_rows": beq, "worst_abs": worst, "worst_rel_to_range": wrel,
                               "top1_agree": f"{top1}/{rows}"}
    rep["sets"][s] = d
    ok = ok and all(d[k] == n for k in ("replies_identical", "ids_identical", "prompt_ids_identical",
                                        "all_positions_logits_bit_equal", "first_token_logits_bit_equal"))
rep["bit_identical"] = bool(ok)
print(json.dumps(rep, indent=1))
sys.exit(0 if ok else 1)
