"""relative_kv through ember's adapter against the suite's stored values (keramos's numpy reference), bit by bit, and
the refusal documents' full error lists under ember's document and container rules.

    python3 relkv_bits.py KERAMOS_TREE ADAPTER [OUT.json]

(a) Each op case attention/relative_kv*: ember's run_node output against the case's expected tensor: the count of
    elements equal bit for bit, the worst |d| in ulps and IR_SPEC 4.3's measure. Run through the runner's CmdAdapter.
(b) Each refuse/relative_kv_* case: ember's check (the case's gguf and document) and every error it reports, with
    the rule the case cites and whether ember's first error cites it.
"""
import json
import sys
from pathlib import Path

import numpy as np

tree, adapter_bin = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(tree / "tools"))
import ir_conformance as rc  # noqa: E402  (the runner)

suite = tree / "conformance"
ad = rc.CmdAdapter(f"{adapter_bin} --conformance")
out = {"op": [], "refuse": []}


def ulps(a, b):
    ia = a.astype(np.float32).view(np.int32).astype(np.int64)
    ib = b.astype(np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib)


ops = json.loads((suite / "ops/attention.json").read_text())
for c in ops["cases"]:
    if not c["name"].startswith("relative_kv"):
        continue
    weights = {k: rc.dec(v) for k, v in c.get("weights", {}).items()}
    inputs = {k: rc.dec(v) for k, v in c["inputs"].items()}
    res = ad.run_node(c["node"], inputs, weights, None)
    for name, want in c["expect"].items():
        g = np.asarray(res["outputs"][name], dtype=np.float32)
        w = np.asarray(rc.dec(want), dtype=np.float32)
        u = ulps(g, w)
        row = {"case": c["name"], "out": name, "elements": int(w.size),
               "bit_equal": int(np.sum(g.view(np.uint32) == w.view(np.uint32))), "worst_ulp": int(u.max()),
               "worst_of_max1": float(np.max(np.abs(g.astype(np.float64) - w)) / max(1.0, float(np.max(np.abs(w)))))}
        out["op"].append(row)
        print("op", json.dumps(row))

for f in sorted((suite / "refuse").glob("relative_kv_*.json")):
    case = json.loads(f.read_text())
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ir = Path(tmp) / "doc.ir.json"
        ir.write_text(json.dumps(case["doc"]))
        r = ad.check(suite / case["gguf"], ir)
        r0 = ad.check(None, ir)  # the document rules alone
    tag = f"IR_SPEC {case['section']} (rule {case['rule']})"
    errs = r.get("errors") or [r.get("message", "")]
    row = {"case": f.stem, "cited": tag, "accepted": r["accepted"], "message": r.get("message"),
           "doc_only_accepted": r0["accepted"], "doc_only_message": r0.get("message"),
           "cites_rule": tag in (r.get("message") or "") or tag in (r0.get("message") or "")}
    out["refuse"].append(row)
    print("refuse", json.dumps(row))
ad.close_all() if hasattr(ad, "close_all") else None
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
