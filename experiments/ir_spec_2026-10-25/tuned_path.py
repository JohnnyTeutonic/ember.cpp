#!/usr/bin/env python3
"""IR_SPEC 9.1: ember's tuned llama path by structural match (graph_ir.h tuned_llama_match), not by family_hint.

  stored    every GGUF under /mnt/c/ml_artifacts and ~/ember_ttft with paperkiln.ir, and the third-party files:
            the path tools/ir_path_check.cpp reports against the rule before this change (tuned iff family_hint
            "llama", no window and no state tokens): every stored file keeps its path;
  edits     chat_sftfinal_v3's and moe_tiny_v3's documents changed one way each that the tuned path does not
            compute (or that the validator refuses), through EMBER_IR: each is served by the interpreter; the
            documents as they are match.
Writes tuned_path_cpu.json beside this file; exits 1 on any difference. Usage (WSL): python3 tuned_path.py"""
import copy
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agreement import A, SCR, embedded, hdr, third_party_ggufs  # noqa: E402

BIN = os.path.expanduser("~/ember_wt/build_cpu/ir_path_check")


def check(paths, ir=None):
    env = {k: v for k, v in os.environ.items() if k != "EMBER_IR"}
    if ir:
        env["EMBER_IR"] = ir
    out = []
    for i in range(0, len(paths), 50):
        r = subprocess.run([BIN] + paths[i:i + 50], capture_output=True, text=True, env=env)
        out += [json.loads(x) for x in r.stdout.splitlines()]
    return out


def old_rule(doc):
    if doc is None:
        return None
    blk = doc["block"]
    windowed = any(isinstance(s.get("window"), dict) or s.get("window", 0) for s in blk.get("state", [])) or any(
        n["op"] == "attention" and (isinstance(n.get("window"), dict) or n.get("window", 0)) for n in blk["nodes"])
    state = any(n["op"] == "embedding" and n.get("state_weight") for n in doc["embed"]["nodes"]) or any(
        n["op"] == "attention" and n.get("mask", "causal") != "causal" for n in blk["nodes"]) or any(
        "evict" in s for s in blk.get("state", []))
    return "tuned" if doc.get("family_hint") == "llama" and not windowed and not state else "interpreter"


report = {"stored": [], "edits": []}
paths = []
for p in sorted(set(glob.glob(A + "/**/*.gguf", recursive=True) + glob.glob(os.path.expanduser("~/ember_ttft/**/*.gguf"),
                                                                          recursive=True))):
    try:
        if "paperkiln.ir" in hdr(p)[0]:
            paths.append(p)
    except Exception:  # noqa: BLE001 (not a GGUF)
        pass
paths += third_party_ggufs()
for r in check(paths):
    doc = embedded(r["gguf"])
    want = old_rule(doc) if doc is not None else r.get("path")  # third-party: the IR ember builds
    if doc is None and "paperkiln.ir" not in hdr(r["gguf"])[0]:
        want = "interpreter" if r.get("path") != "llama (no IR)" else "llama (no IR)"
    r["before"] = want
    r["same"] = r.get("path") == want
    report["stored"].append(r)
    if not r["same"]:
        print("DIFFERS", json.dumps(r))

SFT = A + "/chat_base/sftfinal/chat_sftfinal_v3.gguf"
MOE = A + "/moe_tiny/moe_tiny_v3.gguf"


def blocknode(d, op):
    return [n for n in d["block"]["nodes"] if n["op"] == op][0]


def put(o, **kw):
    o.update(kw)


EDITS = [
    ("as_is", SFT, lambda d: None, "tuned"),
    ("theta_differs_from_metadata", SFT, lambda d: put(blocknode(d, "rope"), theta=500000.0), "interpreter"),
    ("eps_differs_from_metadata", SFT, lambda d: put(blocknode(d, "rmsnorm"), eps=1e-6), "interpreter"),
    ("scale_not_inverse_sqrt", SFT, lambda d: put(blocknode(d, "attention"), scale=0.1), "interpreter"),
    ("neox_pairing", SFT, lambda d: put(blocknode(d, "rope"), pairing="neox"), "interpreter"),
    ("gelu_ffn", SFT, lambda d: put(blocknode(d, "silu"), op="gelu", approx="tanh"), "interpreter"),
    ("extra_affine", SFT, lambda d: d["block"]["nodes"].insert(0, {"op": "affine", "in": ["x"], "scale": 1.0, "shift": 0.0,
                                                                   "out": "x"}), "interpreter"),
    ("no_residual_add", SFT, lambda d: put(d["block"]["nodes"][7], **{"in": ["o", "o"]}), "interpreter"),
    ("q_from_wrong_value", SFT, lambda d: put(d["block"]["nodes"][1], **{"in": ["x"]}), "interpreter"),
    ("renamed_values", SFT, lambda d: [put(n, **{"in": ["hh" if v == "h" else v for v in n["in"]]}) for n in d["block"]["nodes"]]
     + [put(d["block"]["nodes"][0], out="hh")], "tuned"),
    ("invalid_document", SFT, lambda d: put(blocknode(d, "attention"), flavour="vanilla"), "interpreter"),
    ("windowed", SFT, lambda d: (put(blocknode(d, "attention"), window=16), put(d["block"]["state"][0], window=16)), "interpreter"),
    ("moe_as_is", MOE, lambda d: None, "tuned"),
    ("moe_raw_gates", MOE, lambda d: put(blocknode(d, "moe_ffn"), gate_norm="raw"), "interpreter"),
]
for name, g, edit, want in EDITS:
    d = copy.deepcopy(embedded(g))
    edit(d)
    p = os.path.join(SCR, f"tuned_{name}.ir.json")
    json.dump(d, open(p, "w"))
    r = check([g], p)[0]
    r.update(edit=name, want=want, same=r.get("path") == want)
    report["edits"].append(r)
    print(("ok   " if r["same"] else "FAIL ") + name, r.get("path"), r.get("why", "")[:120])
report["summary"] = {"stored": len(report["stored"]), "stored_same_path": sum(r["same"] for r in report["stored"]),
                     "tuned": sum(r.get("path") == "tuned" for r in report["stored"]),
                     "edits": len(report["edits"]), "edits_as_expected": sum(r["same"] for r in report["edits"])}
report["pass"] = all(r["same"] for r in report["stored"] + report["edits"])
print(json.dumps(report["summary"]), "pass", report["pass"])
json.dump(report, open(os.path.join(HERE, "tuned_path_cpu.json"), "w"), indent=1)
sys.exit(0 if report["pass"] else 1)
