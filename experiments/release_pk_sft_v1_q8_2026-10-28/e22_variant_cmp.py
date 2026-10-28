"""One variant against F32: greedy (reply, tokens, stop, ids) out of 40, the status of the three rows
where Q8_0-all flipped (7, 10, 14), and the teacher-forced logit error on F32's greedy ids.

    python3 e22_variant_cmp.py RUN_DIR VARIANT_DIR TAG > TAG.summary.json
"""
import json
import os
import sys

import numpy as np

R, V, tag = sys.argv[1:4]
rows = lambda p: [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]  # noqa: E731
f32, var = rows(os.path.join(R, "f32_server.jsonl")), rows(os.path.join(V, f"{tag}_server.jsonl"))
g = [n for n in os.listdir(V) if n == f"pk_sft_v1_{tag}.gguf"][0]
out = {"tag": tag, "file": g, "bytes": os.path.getsize(os.path.join(V, g)),
       "sha256": open(os.path.join(V, f"{tag}.sha256")).read().split()[0], "n": len(var)}
for f in ("reply", "tokens", "stop_reason", "prompt_ids", "ids"):
    out["greedy_" + f] = sum(a[f] == b[f] for a, b in zip(var, f32))
out["differing_rows"] = [a["i"] for a, b in zip(var, f32) if a["ids"] != b["ids"] or a["reply"] != b["reply"]]
meta = json.load(open(os.path.join(V, f"tf_{tag}.f32.json")))
Vn = meta[0]["vocab"]
A = np.fromfile(os.path.join(R, "tf_f32.f32"), dtype=np.float32).reshape(-1, Vn)
B = np.fromfile(os.path.join(V, f"tf_{tag}.f32"), dtype=np.float32).reshape(-1, Vn)
assert A.shape == B.shape, (A.shape, B.shape)
d = np.abs(A.astype(np.float64) - B)
rng = (A.max(1) - A.min(1)).astype(np.float64)
srt = np.sort(A, 1)
margin = srt[:, -1] - srt[:, -2]
top1 = A.argmax(1) == B.argmax(1)
off, o = {}, 0
for m in meta:
    off[m["i"]] = o
    o += m["rows"]
flips = {}
for i, j in ((7, 10), (10, 11), (14, 13)):   # Q8_0-all's first differing token per row
    r = off[i] + j
    fa = f32[i]["ids"][j]
    vb = int(B[r].argmax())
    flips[str(i)] = {"same_reply_as_f32": var[i]["ids"] == f32[i]["ids"], "token": j, "f32_pick": fa,
                     "variant_pick": vb, "f32_margin": float(margin[r]),
                     "variant_margin_f32pick_minus_best_other": float(B[r, fa] - np.max(np.delete(B[r], fa)))}
out.update({
    "positions": int(A.shape[0]), "max_abs_dlogit": float(d.max()), "mean_abs_dlogit": float(d.mean()),
    "max_abs_over_row_range": float((d.max(1) / rng).max()), "mean_abs_over_row_range": float((d.mean(1) / rng).mean()),
    "top1_agree": f"{int(top1.sum())}/{len(top1)}",
    "top1_disagree_at": [[k for k in off if off[k] <= r < off[k] + [m["rows"] for m in meta if m["i"] == k][0]][0]
                         for r in np.nonzero(~top1)[0]],
    "flip_rows": flips,
})
print(json.dumps(out, indent=1))
