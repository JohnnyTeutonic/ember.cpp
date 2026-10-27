"""Compare a suite run (the runner's --json) with ember's v1.3.5 receipt: every earlier case's verdict and detail,
and the cases the run adds.

    python3 compare_receipt.py RECEIPT.json RUN.json [OUT.json]
"""
import json
import sys

old = {r["id"]: r for r in json.load(open(sys.argv[1]))["results"]}
run = json.load(open(sys.argv[2]))
new = {r["id"]: r for r in run["results"]}
changed = [{"id": i, "was": [o["verdict"], o.get("detail")], "now": [new[i]["verdict"], new[i].get("detail")]}
           for i, o in old.items() if i in new and (new[i]["verdict"], new[i].get("detail")) != (o["verdict"], o.get("detail"))]
missing = sorted(i for i in old if i not in new)
added = [{"id": i, "verdict": r["verdict"], "detail": r.get("detail")} for i, r in new.items() if i not in old]
out = {"receipt_cases": len(old), "run_cases": len(new), "run_counts": run.get("counts"),
       "earlier_cases_changed": changed, "earlier_cases_missing": missing, "added": added}
print(json.dumps({k: v for k, v in out.items() if k != "added"}, indent=1))
for a in added:
    print(f"{a['verdict']:5} {a['id']}: {a['detail']}")
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
