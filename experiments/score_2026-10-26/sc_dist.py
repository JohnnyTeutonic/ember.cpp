"""Distribution of |d logprob| between two score files, and the worst positions with their logprob and position.
    sc_dist.py REF GOT"""
import json
import sys

import numpy as np

ref = {json.loads(x)["key"]: json.loads(x) for x in open(sys.argv[1])}
got = {json.loads(x)["key"]: json.loads(x) for x in open(sys.argv[2])}
rows = []
for k, r in ref.items():
    g = got.get(k)
    if not g or r.get("over_ctx") or g.get("over_ctx"):
        continue
    for j, (a, b) in enumerate(zip(r["logprobs"], g["logprobs"])):
        rows.append((abs(a - b), a, k, j, r["prompt_tokens"] + j))
d = np.array([x[0] for x in rows])
print(f"tokens {len(d)}: median {np.median(d):.2e}, p90 {np.percentile(d, 90):.2e}, p99 {np.percentile(d, 99):.2e}, max {d.max():.2e}; "
      f"> 1e-5: {(d > 1e-5).sum()}, > 5e-6: {(d > 5e-6).sum()}")
for x in sorted(rows, reverse=True)[:6]:
    print(f"  |d| {x[0]:.2e} at logprob {x[1]:.4f} {x[2]}@{x[3]} (position {x[4]})")
