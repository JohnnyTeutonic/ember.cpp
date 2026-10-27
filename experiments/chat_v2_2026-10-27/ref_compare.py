#!/usr/bin/env python3
"""gate_v2.json's ember and live mtstudio replies against paperkiln's mtstudio reference file.

    python3 ref_compare.py GATE.json MTSTUDIO_GREEDY.jsonl OUT.json

Per row (by id): ember vs the file and live mtstudio vs the file on reply, tool_calls, stop_reason, tokens,
prompt_ids, ids and stop_id, and history_turns_used where the file has it (a row the file records as
{"id", "error"} must be HTTP 400)."""
import hashlib
import json
import sys

FIELDS = ("reply", "tool_calls", "stop_reason", "tokens", "prompt_ids", "ids", "stop_id")
gate, refp, out = sys.argv[1:4]
g = json.load(open(gate, encoding="utf-8"))
ref = {r["id"]: r for r in (json.loads(x) for x in open(refp, encoding="utf-8") if x.strip())}


def same(status, resp, r):
    if "error" in r:
        return status == 400
    keys = FIELDS + (("history_turns_used",) if "history_turns_used" in r else ())
    return status == 200 and all(resp.get(k) == r.get(k) for k in keys)


rows, ember_ok, mts_ok, mts_vs_file = [], 0, 0, []
for rec in g["records"]:
    r = ref.get(rec["id"])
    if r is None:
        rows.append({"id": rec["id"], "in_file": False})
        continue
    e = same(rec["status"][0], rec["ember"], r)
    m = None if rec["status"][1] is None else same(rec["status"][1], rec["mtstudio"], r)  # None: ember alone
    ember_ok += e
    mts_ok += bool(m)
    if m is False:
        mts_vs_file.append(rec["id"])
    rows.append({"id": rec["id"], "ember_equal_file": e, "live_mtstudio_equal_file": m, "status": rec["status"],
                 "file": "error" if "error" in r else r.get("stop_reason")})
summary = {"file": refp, "file_sha256": hashlib.sha256(open(refp, "rb").read()).hexdigest(), "file_rows": len(ref),
           "ember_equal_file": ember_ok, "live_mtstudio_equal_file": mts_ok, "mtstudio_sources_disagree": mts_vs_file,
           "ember_differs": [x["id"] for x in rows if not x.get("ember_equal_file")]}
json.dump({"summary": summary, "rows": rows}, open(out, "w", encoding="utf-8"), indent=1)
print(json.dumps(summary))
