"""ember refuses malformed Mixture-of-Recursions IRs at load: mor3's IR (paperkiln's gate model) altered one way
at a time, through EMBER_IR on the same GGUF."""
import copy
import json
import os
import subprocess

D = "/mnt/c/ml_artifacts/mor_gate/mor3"
BIN = os.path.expanduser("~/ember_wt/build_cpu/ir_logits")
base = json.load(open(os.path.join(D, "mor3.ir.json")))


def node(doc):
    return doc["block"]["nodes"][0]


def edit(f):
    doc = copy.deepcopy(base)
    f(doc)
    return doc


cases = {
    "route_not_first": edit(lambda d: d["block"]["nodes"].insert(0, d["block"]["nodes"].pop(1))),
    "expert_choice": edit(lambda d: node(d).update(route="expert_choice")),
    "unequal_steps": edit(lambda d: node(d).update(step={"per_layer": [-1, 0, 0, 0, 1, 2, 2, -1]})),
    "steps_descending": edit(lambda d: node(d).update(step={"per_layer": [-1, 2, 2, 1, 1, 0, 0, -1]})),
    "step_length_wrong": edit(lambda d: node(d).update(step={"per_layer": [-1, 0, 0, 1, 1, 2, 2]})),
    "no_kv_rows_active": edit(lambda d: d["block"]["state"][0].pop("kv_rows")),
    "kv_from": edit(lambda d: d["block"]["state"][0].update(kv_from={"per_layer": list(range(8))})),
    "gate_unknown": edit(lambda d: node(d).update(gate="sigmoid")),
    "router_with_i": edit(lambda d: node(d).update(weight="rec.{i}.router.weight")),
}
out = {}
for name, doc in cases.items():
    path = os.path.expanduser("~/ember_wt/mor_refusal.ir.json")
    json.dump(doc, open(path, "w"))
    r = subprocess.run([BIN, os.path.join(D, "mor3.gguf")], input="[[1, 2, 3]]", capture_output=True, text=True,
                       env={**os.environ, "EMBER_IR": path})
    msg = [ln for ln in r.stderr.splitlines() if "what()" in ln]
    out[name] = {"refused": r.returncode != 0, "message": msg[-1].split("what():", 1)[1].strip() if msg else r.stderr[-200:]}
out["all_refused"] = all(v["refused"] for v in out.values())
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mor_refusals.json"), "w"), indent=1)
