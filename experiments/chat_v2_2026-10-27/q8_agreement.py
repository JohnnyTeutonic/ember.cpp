#!/usr/bin/env python3
"""Report only: the Q8_0 file's greedy replies against the F32 file's, row by row (two gate_v2.py receipts).

    python3 q8_agreement.py F32.json Q8_0.json OUT.json

Per row: the same ids (whole reply), the same first id, the same tool_calls, the same stop reason, and the length
of the common id prefix."""
import json
import sys

f, q = (json.load(open(p, encoding="utf-8"))["records"] for p in sys.argv[1:3])
rows = []
for a, b in zip(f, q):
    ea, eb = a["ember"], b["ember"]
    ia, ib = ea.get("ids") or [], eb.get("ids") or []
    k = 0
    while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
        k += 1
    rows.append({"id": a["id"], "status": [a["status"][0], b["status"][0]], "ids_equal": ia == ib and a["status"][0] == b["status"][0],
                 "first_id_equal": ia[:1] == ib[:1], "tool_calls_equal": ea.get("tool_calls") == eb.get("tool_calls"),
                 "stop_equal": ea.get("stop_reason") == eb.get("stop_reason"), "common_prefix": k, "f32_tokens": len(ia),
                 "f32_calls": bool(ea.get("tool_calls")), "q8_calls": bool(eb.get("tool_calls"))})
summary = {k: sum(r[k] for r in rows) for k in ("ids_equal", "first_id_equal", "tool_calls_equal", "stop_equal")}
summary.update({"rows": len(rows), "f32_rows_with_calls": sum(r["f32_calls"] for r in rows),
                "q8_rows_with_calls": sum(r["q8_calls"] for r in rows)})
json.dump({"summary": summary, "rows": rows}, open(sys.argv[3], "w", encoding="utf-8"), indent=1)
print(json.dumps(summary))
