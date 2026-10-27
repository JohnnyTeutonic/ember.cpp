"""Does ember's relative_kv path follow IR_SPEC 5.15's stated order? An f32 statement of the text written for this
check (not the reference's code): per head, s = ((q . k_j) + (q . rk[n])) * scale with each dot product summed from
+0 over the columns in ascending order, each operation rounded to f32; then ember's softmax (softmax_vector_cpu: the
max, exp(s - max) rounded, their sum in ascending order, then each times 1 / (sum + 1e-9)); then
out = (sum a v_j) + (sum a rv[n]), each sum from +0 in ascending key order. exp is glibc's expf (through ctypes),
the function ember's std::exp(float) calls.

    python3 relkv_order.py KERAMOS_TREE ADAPTER

Prints, per op case attention/relative_kv*: elements bit-equal between ember and this statement, and between
ember and the suite's stored (numpy reference) values.
"""
import ctypes
import json
import sys
from pathlib import Path

import numpy as np

tree, adapter_bin = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(tree / "tools"))
import ir_conformance as rc  # noqa: E402  (the runner: dec, CmdAdapter)

f32 = np.float32
_libm = ctypes.CDLL("libm.so.6")
_libm.expf.restype, _libm.expf.argtypes = ctypes.c_float, [ctypes.c_float]


def stated(node, q, k, v, rk, rv):
    H, hd = node["heads"], node["head_dim"]
    KV = node.get("kv_heads", H)
    W, S = node.get("window", 0), node.get("sinks", 0)
    K = node["relative_kv"]["max_distance"]
    scale = f32(node["scale"])
    T = q.shape[0]
    out = np.zeros((T, H * hd), dtype=f32)
    for p in range(T):
        if W > 0:
            lo = max(p + 1 - W, 0)
            vis = list(range(min(S, p + 1, lo))) + list(range(lo, p + 1))
        else:
            vis = list(range(p + 1))
        for h in range(H):
            g = h // (H // KV)
            qh = q[p, h * hd:(h + 1) * hd]
            cols = lambda tab: (slice(h * hd, (h + 1) * hd) if tab.shape[1] != hd else slice(0, hd))
            sc = []
            for j in vis:
                n = min(p - j, K)
                dot, rel = f32(0), f32(0)
                kj = k[j, g * hd:(g + 1) * hd]
                rr = rk[n, cols(rk)]
                for d in range(hd):
                    dot = f32(dot + f32(qh[d] * kj[d]))
                for d in range(hd):
                    rel = f32(rel + f32(qh[d] * rr[d]))
                sc.append(f32(f32(dot + rel) * scale))
            mx = max(sc)
            e = [f32(_libm.expf(float(f32(s - mx)))) for s in sc]
            z = f32(0)
            for x in e:
                z = f32(z + x)
            inv = f32(f32(1) / f32(z + f32(1e-9)))
            a = [f32(x * inv) for x in e]
            for d in range(hd):
                acc = f32(0)
                for i, j in enumerate(vis):
                    if a[i] != 0:
                        acc = f32(acc + f32(a[i] * v[j, g * hd + d]))
                if rv is not None:
                    racc = f32(0)
                    for i, j in enumerate(vis):
                        if a[i] != 0:
                            n = min(p - j, K)
                            racc = f32(racc + f32(a[i] * rv[n, cols(rv)][d]))
                    acc = f32(acc + racc)
                out[p, h * hd + d] = acc
    return out


ad = rc.CmdAdapter(f"{adapter_bin} --conformance")
ops = json.loads((tree / "conformance/ops/attention.json").read_text())
for c in ops["cases"]:
    if not c["name"].startswith("relative_kv"):
        continue
    node = c["node"]
    weights = {kk: rc.dec(vv) for kk, vv in c.get("weights", {}).items()}
    inputs = {kk: rc.dec(vv) for kk, vv in c["inputs"].items()}
    q, k, v = (np.asarray(inputs[x], dtype=f32) for x in node["in"])
    rel = node["relative_kv"]
    rk = np.asarray(weights[rel["key"]], dtype=f32)
    rv = np.asarray(weights[rel["value"]], dtype=f32) if rel.get("value") else None
    mine = stated(node, q, k, v, rk, rv)
    got = np.asarray(ad.run_node(node, inputs, weights, None)["outputs"][node["out"]], dtype=f32)
    ref = np.asarray(rc.dec(c["expect"][node["out"]]), dtype=f32)
    eq = lambda a, b: int(np.sum(a.view(np.uint32) == b.view(np.uint32)))
    print(json.dumps({"case": c["name"], "elements": int(ref.size), "ember_eq_stated": eq(got, mine),
                      "ember_eq_reference": eq(got, ref), "stated_eq_reference": eq(mine, ref),
                      "ember_vs_stated_differ": [[int(i), float(got.flat[i]), float(mine.flat[i])]
                                                 for i in np.flatnonzero(got.view(np.uint32) != mine.view(np.uint32))]}))
