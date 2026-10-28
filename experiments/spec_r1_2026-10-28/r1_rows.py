"""Distances between the dumped rows of dr_e2e.py --dump runs (spec_r1_2026-10-28).

    python3 r1_rows.py REF.jsonl RUN.jsonl [RUN.jsonl ...]       (prints one JSON object per run)

Each run against REF (CPU plain decode, say): over the requests whose ids, stop ids, prompt ids and row counts are
equal, every dumped row's max |d| and max |d| / max(1, max |REF row|) (IR_SPEC 4.3's ROUNDING scale), as quantiles
(50%, 90%, 99%, max), the worst row (request, turn, its index within the request: 0 is the prefill row), and the
mean per row index within the request (does the distance grow along a reply?)."""
import json
import sys

import numpy as np


def load(p):
    recs = [json.loads(l) for l in open(p, encoding="utf-8")]
    meta = json.load(open(p + ".meta.json"))
    rows = np.memmap(p + ".rows.f32", dtype=np.float32, mode="r").reshape(-1, meta["vocab"])
    return recs, rows


def compare(ref, run):
    (ra, xa), (rb, xb) = ref, run
    d_abs, d_rel, where, by_idx = [], [], [], {}
    same = 0
    for x, y in zip(ra, rb):
        if (x["ids"], x["stop_id"], x["prompt_ids"], x["rows"]) != (y["ids"], y["stop_id"], y["prompt_ids"], y["rows"]):
            continue
        same += 1
        pa = xa[x["row0"]:x["row0"] + x["rows"]].astype(np.float64)
        pb = xb[y["row0"]:y["row0"] + y["rows"]].astype(np.float64)
        da = np.abs(pa - pb).max(axis=1)
        dr = da / np.maximum(1.0, np.abs(pa).max(axis=1))
        for j in range(da.size):
            d_abs.append(da[j])
            d_rel.append(dr[j])
            where.append((x["i"], x["turn"], j))
            by_idx.setdefault(min(j, 60), []).append(da[j])
    d_abs, d_rel = np.array(d_abs), np.array(d_rel)
    q = lambda a: {f"p{int(p)}": float(np.percentile(a, p)) for p in (50, 90, 99)} | {"max": float(a.max())}
    w = int(d_abs.argmax())
    return {"requests_compared": same, "of": len(ra), "rows": int(d_abs.size), "abs": q(d_abs), "rel": q(d_rel),
            "rows_bit_equal": int((d_abs == 0).sum()), "worst_row": where[w],
            "mean_abs_by_row_index": {str(k): float(np.mean(v)) for k, v in sorted(by_idx.items()) if k in (0, 1, 5, 10, 20, 40, 60)}}


if __name__ == "__main__":
    ref = load(sys.argv[1])
    for p in sys.argv[2:]:
        r = compare(ref, load(p))
        r.update({"ref": sys.argv[1], "run": p})
        print(json.dumps(r), flush=True)
