"""Compare two vd_collect.py runs request by request (verify_det_2026-10-27).

    python3 vd_compare.py BASE_DIR OTHER_DIR

Per request: reply, ids and stop id equal; the same number of dumped rows; every row bit-identical (float32 bit
patterns). Prints one JSON object: counts over requests and rows, the largest |d| over rows that differ, and the
requests that differ."""
import json
import os
import sys

import numpy as np


def load(d):
    recs = [json.loads(x) for x in open(os.path.join(d, "records.jsonl"), encoding="utf-8")]
    meta = json.load(open(os.path.join(d, "meta.json")))
    v = meta["summary"]["vocab"]
    rows = np.fromfile(os.path.join(d, "rows.f32"), dtype=np.float32).reshape(-1, v)
    return recs, meta, rows


def main(a, b):
    ra, ma, xa = load(a)
    rb, mb, xb = load(b)
    res = {"base": a, "other": b, "base_env": ma["env"], "other_env": mb["env"], "requests": len(ra),
           "replies_identical": 0, "ids_identical": 0, "row_counts_equal": 0, "requests_all_rows_bit_identical": 0,
           "rows_compared": 0, "rows_bit_identical": 0, "max_abs_diff": 0.0, "argmax_differs": 0,
           "differ_ids": [], "differ_rows": [],
           "other_spec": {k: mb["summary"][k] for k in ("drafted", "accepted", "passes")},
           "warm_pass_same": [ma["summary"]["warm_pass_same"], mb["summary"]["warm_pass_same"]]}
    if len(ra) != len(rb) or ma["summary"]["vocab"] != mb["summary"]["vocab"]:
        res["error"] = "request counts or vocabularies differ"
        print(json.dumps(res))
        return
    for x, y in zip(ra, rb):
        res["replies_identical"] += x["reply"] == y["reply"]
        same_ids = x["ids"] == y["ids"] and x["stop_id"] == y["stop_id"]
        res["ids_identical"] += same_ids
        if not same_ids:
            res["differ_ids"].append(x["i"])
        if x["rows"] != y["rows"]:
            res["differ_rows"].append(x["i"])
            continue
        res["row_counts_equal"] += 1
        pa = xa[x["row0"]:x["row0"] + x["rows"]]
        pb = xb[y["row0"]:y["row0"] + y["rows"]]
        eq = np.all(pa.view(np.uint32) == pb.view(np.uint32), axis=1)
        res["rows_compared"] += int(eq.size)
        res["rows_bit_identical"] += int(eq.sum())
        if eq.all():
            res["requests_all_rows_bit_identical"] += 1
        else:
            res["differ_rows"].append(x["i"])
            d = np.abs(pa[~eq].astype(np.float64) - pb[~eq].astype(np.float64))
            res["max_abs_diff"] = max(res["max_abs_diff"], float(np.nanmax(d)))
            res["argmax_differs"] += int(np.sum(pa[~eq].argmax(axis=1) != pb[~eq].argmax(axis=1)))
    res["bit_identical"] = (res["ids_identical"] == len(ra) and res["requests_all_rows_bit_identical"] == len(ra))
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
