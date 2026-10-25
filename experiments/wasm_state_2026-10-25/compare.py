"""Compare tools/state_wasm_gate's WebAssembly run with its native CPU run (one model).

    python3 compare.py NAME INPUT.json NATIVE_DIR WASM_DIR [--manifest manifest.json] [--tol 1e-4]

Both runs made the same calls in the same order, so every logits block lines up. Reported (max |dlogit|
over the native row's range, worst over rows; argmax agreement at every row):
  - WebAssembly against native on every block: prefill on and off, decode, turn by turn, the rewinds'
    re-prefills, the session script's calls (on, off and fresh);
  - inside each engine: eviction on against off (prefill and calls), decode and turn by turn against
    prefill, calls against a fresh prefill;
  - greedy on each engine: the argmax at every scored position (P-1 .. len-2) equal to the next id, and
    free-running greedy equal to the continuation; WebAssembly's greedy ids equal to native's;
  - every integer result equal (rows held on and off, resume points, refused direct rewinds, the block
    layout), and with --manifest rows held equal to mtstudio's (kv_rows_evict, kv_rows_keep).
Prints a JSON summary; exit 1 unless every check passes within --tol."""
import json
import sys

import numpy as np

a = sys.argv[1:]
tol = float(a[a.index("--tol") + 1]) if "--tol" in a else 1e-4
manifest = json.load(open(a[a.index("--manifest") + 1])) if "--manifest" in a else None
name, inp, nd, wd = a[:4]
inp = json.load(open(inp))


def load(d):
    run = json.load(open(d + "/run.json"))
    lg = np.fromfile(d + "/logits.f32", dtype=np.float32).reshape(-1, run["vocab"])
    assert lg.shape[0] == run["logit_rows"], d
    return run, lg


nr, nl = load(nd)
wr, wl = load(wd)


def block(lg, ref):
    return lg[ref[0]:ref[0] + ref[1]]


def rel(x, ref):
    """Worst over rows of max |x - ref| / (max ref - min ref)."""
    x, ref = x.astype(np.float64), ref.astype(np.float64)
    rng = ref.max(1) - ref.min(1)
    d = np.abs(x - ref).max(1)
    return float(np.where(rng > 0, d / np.where(rng > 0, rng, 1), d).max()) if len(ref) else 0.0


def top1(x, ref):
    return int((x.argmax(1) == ref.argmax(1)).sum()), len(ref)


class W:
    def __init__(self):
        self.v, self.agree, self.rows, self.bit = 0.0, 0, 0, True

    def add(self, x, ref, argmax=True):
        self.v = max(self.v, rel(x, ref))
        self.bit = self.bit and bool(np.array_equal(x, ref))
        if argmax:
            g, n = top1(x, ref)
            self.agree += g
            self.rows += n

    def out(self, argmax=True):
        o = {"worst_rel": self.v, "bit_equal": self.bit}
        if argmax:
            o["top1"] = f"{self.agree}/{self.rows}"
        return o


def strip(run):
    return {k: v for k, v in run.items() if k not in ("engine", "ms", "gguf")}


vs = {}  # WebAssembly against native, per kind of block
inside = {e: {} for e in ("native", "wasm")}


def w(d, k):
    return d.setdefault(k, W())


tf = {"native": 0, "wasm": 0}
free = {"native": 0, "wasm": 0}
seqs = inp.get("seqs", [])
for rn, rw_ in zip(nr["rows"], wr["rows"]):
    for kind in ("prefill_on", "prefill_off", "decode_on", "turns_on"):
        if kind in rn:
            w(vs, kind).add(block(wl, rw_[kind]), block(nl, rn[kind]))
    for x, y in zip(rn.get("rewinds", []), rw_.get("rewinds", [])):
        w(vs, "rewind_tail").add(block(wl, y["tail"]), block(nl, x["tail"]))
    for eng, run, lg in (("native", rn, nl), ("wasm", rw_, wl)):
        pre = block(lg, run["prefill_on"])
        w(inside[eng], "on_vs_off").add(pre, block(lg, run["prefill_off"]))
        w(inside[eng], "decode_vs_prefill").add(block(lg, run["decode_on"]), pre)
        if "turns_on" in run:
            w(inside[eng], "turns_vs_prefill").add(block(lg, run["turns_on"]), pre)
        for x in run.get("rewinds", []):
            w(inside[eng], "rewind_vs_prefill").add(block(lg, x["tail"]), pre[x["n"]:])
        if "prompt_len" in run:
            s, P = seqs[run["i"]], run["prompt_len"]
            tf[eng] += all(int(pre[t].argmax()) == s[t + 1] for t in range(P - 1, len(s) - 1))
            free[eng] += run["free_greedy"] == s[P:]
for cn, cw in zip(nr.get("calls", []), wr.get("calls", [])):
    for kind in ("on", "off", "fresh"):
        w(vs, "call_" + kind).add(block(wl, cw[kind]), block(nl, cn[kind]))
    for eng, c, lg in (("native", cn, nl), ("wasm", cw, wl)):
        w(inside[eng], "call_on_vs_off").add(block(lg, c["on"]), block(lg, c["off"]))
        w(inside[eng], "call_on_vs_fresh").add(block(lg, c["on"]), block(lg, c["fresh"]))

n = len(nr["rows"])
m = sum("prompt_len" in r for r in nr["rows"])
rep = {"model": name, "native_gguf": nr["gguf"], "sequences": n, "calls": len(nr.get("calls", [])),
       "logit_rows": nr["logit_rows"], "tol": tol, "integer_results_equal": strip(nr) == strip(wr),
       "wasm_vs_native": {k: v.out() for k, v in vs.items()},
       "inside": {e: {k: v.out() for k, v in d.items()} for e, d in inside.items()},
       "ms": {"native": nr["ms"], "wasm": wr["ms"]}}
if m:
    rep["greedy"] = {e: {"teacher_forced": f"{tf[e]}/{m}", "free_running": f"{free[e]}/{m}"} for e in tf}
    rep["greedy_ids_equal"] = all(x.get("free_greedy") == y.get("free_greedy") for x, y in zip(nr["rows"], wr["rows"]))
ts = wr["turn_state"]  # kv_rows_held counts turn_state's rows; it is 0 for any other IR
rows_on = [r["kv_rows_held"]["on"] for r in wr["rows"]] if ts else []
rows_off = [r["kv_rows_held"]["off"] for r in wr["rows"]] if ts else []
if rows_on:
    rep["kv_rows_held"] = {"wasm_equal_native": rows_on == [r["kv_rows_held"]["on"] for r in nr["rows"]] and
                           rows_off == [r["kv_rows_held"]["off"] for r in nr["rows"]],
                           "mean_on": float(np.mean(rows_on)), "mean_off": float(np.mean(rows_off))}
    if manifest:
        mv = manifest["numpy_vs_mtstudio"]
        rep["kv_rows_held"]["on_equal_mtstudio"] = rows_on == mv["kv_rows_evict"]
        rep["kv_rows_held"]["off_equal_mtstudio"] = rows_off == mv["kv_rows_keep"]
rw = [x for r in wr["rows"] for x in r.get("rewinds", [])]
if rw:
    exp = [x for x in rw if x["resume"] != x["n"]]
    rep["rewinds"] = {"n": len(rw), "resume_moved": len(exp),
                      "direct_refused": f"{sum(x['direct_refused'] for x in exp)}/{len(exp)}"}
cw = wr.get("calls", [])
if cw:
    exp = [c for c in cw if "direct_refused" in c]
    rep["calls_direct_refused"] = f"{sum(c['direct_refused'] for c in exp)}/{len(exp)}"
ok = rep["integer_results_equal"]
ok = ok and all(v.v <= tol and v.agree == v.rows for v in vs.values())
ok = ok and all(v.v <= tol for v in inside["wasm"].values())
if m:
    ok = ok and tf["wasm"] == m and free["wasm"] == m and rep["greedy_ids_equal"]
if rows_on and manifest:
    ok = ok and rep["kv_rows_held"]["on_equal_mtstudio"] and rep["kv_rows_held"]["off_equal_mtstudio"]
if rw:
    ok = ok and all(x["direct_refused"] for x in rw if x["resume"] != x["n"])
if cw:
    ok = ok and all(c["direct_refused"] for c in cw if "direct_refused" in c)
rep["pass"] = bool(ok)
print(json.dumps(rep, indent=1))
sys.exit(0 if ok else 1)
