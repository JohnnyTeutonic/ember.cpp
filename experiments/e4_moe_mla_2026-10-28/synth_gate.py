#!/usr/bin/env python3
"""ember's CPU interpreter on paperkiln's E4 synthetic files (C:/ml_artifacts/thirdparty/e4_synth/<fam>.gguf), against
HF's recorded greedy continuations and last-position logits (<fam>.cases.json), and ember's C++ builder against
keramos's gguf_ir.py (v1.6.1's, run as a black box) on each file: field by field except _comment, and in Appendix D.7's
normal form.

    python3 synth_gate.py --adapter BIN --ir-logits BIN --keramos DIR --dir SYNTH --out OUT.json fam ...
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


def dec(t):
    h = t["f32"]
    v = np.array(struct.unpack(">%dI" % (len(h) // 8), bytes.fromhex(h)), dtype=np.uint32).view(np.float32)
    return v.reshape(t["shape"])


class Adapter:
    def __init__(self, cmd, extra=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("EMBER_")}
        env.update(extra or {})
        self.p = subprocess.Popen([cmd, "--conformance"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                  bufsize=1, env=env)
        self.n = 0

    def call(self, method, params):
        self.n += 1
        self.p.stdin.write(json.dumps({"id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        if "result" not in r:
            raise RuntimeError(f"{method}: {r}")
        return r["result"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---- Appendix D.7's normal form (written from IR_SPEC.md's text at v1.6.1, revisions 1.5 and 1.6 included)
DEFAULTS = {  # op: {attr: default}, step 2 (rmsnorm.group excepted)
    "linear": {"bias": None, "tied_to": None}, "scale": {"bias": None}, "attention": {"mask": "causal"},
    "rmsnorm": {"weight_offset": 0}, "depth_attend": {"eps": 1e-5}, "conv1d_causal": {"bias": None},
    "selective_scan": {"d": None, "dt_softplus": True}, "split": {"groups": 1}, "rope": {"freq_scale": 1, "attn_factor": 1},
    "moe_ffn": {"scoring": "softmax"},  # revision 1.6
}


def f32(x):
    return float(np.float32(x))


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if not k.startswith("_")}
    if isinstance(v, list):
        return [strip(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    return f32(v)


def normal_form(doc, tensors):
    d = strip(doc)
    d = {k: d[k] for k in ("model", "embed", "block", "head") if k in d}
    slot_names = {}
    for k, s in enumerate(d["block"].get("state", [])):
        slot_names.setdefault(s.get("name"), "slot%d" % k)
        s["name"] = "slot%d" % k
        if s.get("kind") == "kv":
            s.setdefault("dtype", "f32")
    cur, nxt = {}, [0]

    def node(n, in_block=False):
        n = dict(n)
        op = n.get("op")
        n.pop("softmax", None) if op == "attention" else None
        n.pop("routing", None) if op == "moe_ffn" else None
        for a, v in DEFAULTS.get(op, {}).items():
            n.setdefault(a, strip(v))
        if in_block:  # revision 1.6: a block node without apply gets true
            n.setdefault("apply", True)
        if op == "rope":
            n.setdefault("kv_heads", n.get("heads"))
            n.setdefault("n_rot", n.get("head_dim"))
        if op == "linear" and n.get("tied_to") is not None:
            if n.get("weight") not in tensors:
                n["weight"] = n["tied_to"]
            n["tied_to"] = None
        if "state" in n and n["state"] in slot_names:
            n["state"] = slot_names[n["state"]]
        ins = [x if x in ("tokens", "pos") else cur.get(x, "unwritten:" + x) for x in n.get("in", [])]
        if op in ("add", "mul"):
            ins = sorted(ins)
        n["in"] = ins
        outs = n["out"] if isinstance(n["out"], list) else [n["out"]]
        new = []
        for o in outs:
            cur[o] = "v%d" % nxt[0]
            new.append(cur[o])
            nxt[0] += 1
        n["out"] = new if isinstance(n["out"], list) else new[0]
        return n

    for sect in ("embed", "block", "head"):
        d[sect]["nodes"] = [node(n, sect == "block") for n in d[sect]["nodes"]]
    return d


def first_diff(a, b, path="doc"):
    if type(a) != type(b):
        return f"{path}: {a!r} against {b!r}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: present in one only"
            r = first_diff(a[k], b[k], path + "." + k)
            if r:
                return r
        return ""
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} entries against {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = first_diff(x, y, f"{path}[{i}]")
            if r:
                return r
        return ""
    return "" if a == b else f"{path}: {a!r} against {b!r}"


def gguf_tensor_names(path, keramos):
    sys.path.insert(0, os.path.join(keramos, "tools"))
    import gguf  # the gguf package gguf_ir.py reads with
    return {t.name for t in gguf.GGUFReader(path).tensors}


def run_family(fam, a, args):
    g = os.path.join(args.dir, fam + ".gguf")
    cases = json.load(open(os.path.join(args.dir, fam + ".cases.json")))
    res = {"family": fam, "gguf": os.path.basename(g), "gguf_sha256": sha(g),
           "cases_sha256": sha(os.path.join(args.dir, fam + ".cases.json"))}
    # builder: ember's (ir_logits, as every loader builds it) against gguf_ir.py
    with tempfile.TemporaryDirectory() as tmp:
        mine = os.path.join(tmp, "ember.ir.json")
        env = {k: v for k, v in os.environ.items() if not k.startswith("EMBER_")}
        env["EMBER_IR_PRINT"] = mine
        if fam == "mixtral":  # a llama file: the IR is built on request (EMBER_IR_BUILD=1); served on the llama path
            env["EMBER_IR_BUILD"] = "1"
        subprocess.run([args.ir_logits, g], input="[]", text=True, capture_output=True, env=env, check=True)
        ember_doc = json.load(open(mine))
    with tempfile.TemporaryDirectory() as tmp:
        refp = os.path.join(tmp, "gguf_ir.json")
        subprocess.run([sys.executable, os.path.join(args.keramos, "tools", "gguf_ir.py"), g, refp], capture_output=True,
                       text=True, check=True)
        ref_doc = json.load(open(refp))
    raw_e = {k: v for k, v in ember_doc.items() if k != "_comment"}
    raw_r = {k: v for k, v in ref_doc.items() if k != "_comment"}
    res["builder_equal_field_by_field"] = raw_e == raw_r
    res["builder_first_raw_difference"] = first_diff(raw_e, raw_r)
    names = gguf_tensor_names(g, args.keramos)
    d7 = first_diff(normal_form(ember_doc, names), normal_form(ref_doc, names))
    res["builder_equal_d7"] = d7 == ""
    res["builder_first_d7_difference"] = d7
    if args.save_ir:
        os.makedirs(args.save_ir, exist_ok=True)
        json.dump(ember_doc, open(os.path.join(args.save_ir, fam + ".ember.ir.json"), "w"), indent=1)
        json.dump(ref_doc, open(os.path.join(args.save_ir, fam + ".gguf_ir.json"), "w"), indent=1)
    # greedy and first-token logits
    if fam == "mixtral":
        a = Adapter(args.adapter, {"EMBER_IR_BUILD": "1"})
    rows = []
    for ci, c in enumerate(cases["cases"]):
        s = a.call("open", {"gguf": g, "ir": None, "evict": None})["session"]
        r = a.call("forward", {"session": s, "ids": c["ids"], "report": ["logits"]})
        lg = dec(r["logits"])
        last = lg[-1]
        hf = dec(c["hf_last_logits"]).reshape(-1)
        rng = float(hf.max() - hf.min())
        dist = float(np.max(np.abs(last.astype(np.float64) - hf.astype(np.float64))))
        out = []
        cur = last
        for k in range(cases["max_new"]):
            t = int(np.argmax(cur))
            out.append(t)
            if k + 1 < cases["max_new"]:
                cur = dec(a.call("forward", {"session": s, "ids": [t], "report": ["logits"]})["logits"])[-1]
        a.call("close", {"session": s})
        rows.append({"case": ci, "len": len(c["ids"]), "greedy_equal": out == c["hf_greedy"], "ember": out,
                     "first_logit_max_abs": dist, "row_range": rng, "rel": dist / rng,
                     "top1_equal": int(np.argmax(last)) == int(np.argmax(hf))})
    res["greedy"] = f"{sum(r['greedy_equal'] for r in rows)}/{len(rows)}"
    res["top1_first"] = f"{sum(r['top1_equal'] for r in rows)}/{len(rows)}"
    res["worst_rel"] = max(r["rel"] for r in rows)
    res["median_rel"] = float(np.median([r["rel"] for r in rows]))
    res["rows"] = rows
    if args.llama_path and fam == "mixtral":  # the path ember serves a llama file on, by fresh prefills
        env = {k: v for k, v in os.environ.items() if not k.startswith("EMBER_")}
        lp = subprocess.run([args.llama_path, g, os.path.join(args.dir, fam + ".cases.json")], capture_output=True,
                            text=True, env=env, check=True).stdout.splitlines()
        lrows = []
        for c, line in zip(cases["cases"], lp):
            r = json.loads(line)
            hf = dec(c["hf_last_logits"]).reshape(-1).astype(np.float64)
            first = np.array(r["first"], dtype=np.float64)
            lrows.append({"greedy_equal": r["greedy"] == c["hf_greedy"],
                          "rel": float(np.max(np.abs(first - hf)) / (hf.max() - hf.min())),
                          "top1_equal": int(np.argmax(first)) == int(np.argmax(hf))})
        res["llama_path"] = {"greedy": f"{sum(x['greedy_equal'] for x in lrows)}/{len(lrows)}",
                             "top1_first": f"{sum(x['top1_equal'] for x in lrows)}/{len(lrows)}",
                             "worst_rel": max(x["rel"] for x in lrows),
                             "median_rel": float(np.median([x["rel"] for x in lrows]))}
        print(fam, "llama path", res["llama_path"], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--ir-logits", required=True)
    ap.add_argument("--keramos", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-ir")
    ap.add_argument("--llama-path")
    ap.add_argument("families", nargs="+")
    args = ap.parse_args()
    a = Adapter(args.adapter)
    out = []
    for fam in args.families:
        r = run_family(fam, a, args)
        print(fam, "greedy", r["greedy"], "top1", r["top1_first"], "worst rel %.3g" % r["worst_rel"],
              "builder raw", r["builder_equal_field_by_field"], "D.7", r["builder_equal_d7"],
              r["builder_first_raw_difference"][:160], r["builder_first_d7_difference"][:160], flush=True)
        out.append(r)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
