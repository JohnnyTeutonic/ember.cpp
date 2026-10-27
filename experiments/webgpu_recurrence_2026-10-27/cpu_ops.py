"""ember CPU's outputs of recurrence op cases (WSL): every case of the given case files through ember's
conformance adapter (tools/ir_adapter.cpp, run_node: the node as application 0, token by token from the slot's
initial state), into one JSON {case name: {"outputs": {...}, "state": {...}}} in the cases' tensor format.

    python3 cpu_ops.py IR_ADAPTER OUT.json CASES.json [CASES.json ...]
"""
import json
import subprocess
import sys

adapter, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
reqs, names = [], []
for f in files:
    for c in json.load(open(f, encoding="utf-8"))["cases"]:
        names.append(c["name"])
        reqs.append({"id": len(reqs), "method": "run_node",
                     "params": {"node": c["node"], "inputs": c["inputs"], "weights": c.get("weights", {})}})
p = subprocess.run([adapter], input="".join(json.dumps(r) + "\n" for r in reqs), capture_output=True, text=True, timeout=600)
rows = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
res = {}
for r in rows:
    if "result" not in r:
        sys.exit(f"case {names[r['id']]}: {r}")
    res[names[r["id"]]] = r["result"]
if len(res) != len(reqs):
    sys.exit(f"{len(res)} replies for {len(reqs)} cases; stderr: {p.stderr[-2000:]}")
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
print(out, len(res), "cases")
