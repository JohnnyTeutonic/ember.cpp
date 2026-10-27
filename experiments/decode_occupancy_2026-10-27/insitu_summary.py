"""The decode attention kernels inside the captured decode graph (vm2.sh leg (c): occ_step at one prefix per run under
nsys --cuda-graph-trace=node): per model, kernel and cached length, the split-K kernel's and its merge's median time,
and their sum per layer.

    python3 insitu_summary.py OUT
"""
import csv
import glob
import os
import sys

out = sys.argv[1]
rows = {}
for f in glob.glob(os.path.join(out, "insitu", "*_cuda_gpu_kern_sum.csv")):
    m, k, p = os.path.basename(f).split("_")[:3]
    d = {}
    for r in csv.DictReader(open(f)):
        n = r["Name"]
        if "decode_attn" not in n:
            continue
        part = "merge" if "merge" in n else "split"
        d[part] = (float(r["Med (ns)"]) / 1000, float(r["Avg (ns)"]) / 1000, int(r["Instances"]))
    rows[(m, k, int(p))] = d
for m in sorted({k[0] for k in rows}):
    print(f"{m}: keys | split kernel + merge, median us (instances) | tiled kernel + merge | per layer split -> tiled")
    for p in sorted({k[2] for k in rows if k[0] == m}):
        s, t = rows.get((m, "split", p), {}), rows.get((m, "tiled", p), {})
        if not s or not t:
            continue
        ss = s["split"][0] + s["merge"][0]
        ts = t["split"][0] + t["merge"][0]
        print(f"  {p + 1:4d} | {s['split'][0]:6.2f} + {s['merge'][0]:5.2f} ({s['split'][2]}) | "
              f"{t['split'][0]:6.2f} + {t['merge'][0]:5.2f} ({t['split'][2]}) | {ss:6.2f} -> {ts:5.2f} ({ss / ts:.1f}x)")
