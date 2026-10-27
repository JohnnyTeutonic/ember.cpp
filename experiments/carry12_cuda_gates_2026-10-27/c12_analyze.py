"""The Carry-12 CUDA gates' numbers from c12_vm.sh's outputs (DIR = /mnt/c/ml_artifacts/ember_c12_gates/<session>/out),
per file and configuration ({def, det} = default mode or EMBER_DETERMINISTIC=1; {tiled, split} = the decode attention
default or EMBER_DECODE_ATTN=split):
  against mtstudio score (the file's score_out.jsonl): log-probs of the 40 continuations, worst |d| / max(1, max |ref|)
    (gate 1e-5 on F32; the Q8_0 files' distance is the quantisation's, reported), the argmax at every scored position
    = continuation, free-running greedy = continuation; decode vs prefill on the device (max |d| over the prefill
    row's range); run a vs run b byte for byte;
  state_device_gate (device against the CPU in one process) and ir_device_gate (random ids): summaries and verdicts;
  tiled vs split in each mode: byte-equal outputs, argmax equal at every prefill and decode row, free-running ids
    equal, max |dlogit| (absolute and over the row's range), prefill rows and decode rows apart;
  the EMBER_IR control (the unchanged document = the embedded one, byte for byte);
  the long leg (ctx-1024 document, 8 x 600 random ids, 300 free-running): the same comparisons without a reference,
    and ir_device_gate (4 x 600, 64 greedy) against the CPU;
  nsys: every attention kernel launched (instances), per profiled run;
  the head build against the pin, byte for byte (the mode, default kernel), and its state_device_gate.
Gate: every check above (Q8_0 log-probs reported only). Earlier receipts' figures are carried beside for the files
the earlier gates ran.
    python3 c12_analyze.py DIR OUT.json [LONG_DIR]   (LONG_DIR: a session that ran the long leg alone)"""
import csv
import json
import os
import sys

import numpy as np

src, out_path = sys.argv[1], sys.argv[2]
lsrc = sys.argv[3] if len(sys.argv) > 3 else src
H = os.path.dirname(os.path.abspath(__file__))
A = "/mnt/c/ml_artifacts"
REF = {"state4": f"{A}/state_gate/state4", "ar3": f"{A}/attnres_gate/ar3", "ar_x5": f"{A}/attnres_gate_x5/ar_x5"}
FILES = ["state4", "state4_q8_0", "ar3", "ar3_q8_0", "ar_x5", "ar_x5_q8_0"]
LONG = ["ar3", "ar3_q8_0", "ar_x5", "ar_x5_q8_0"]
CFGS = ["def_tiled", "def_split", "det_tiled", "det_split"]
EARLIER = {  # (receipt, file) of the earlier CUDA gates
    "state4": ("../cuda_state_2026-10-25/receipts/sdg_state4.json", "../cuda_state_2026-10-25/receipts/dg_state4.json"),
    "state4_q8_0": ("../cuda_state_2026-10-25/receipts/sdg_state4_q8_0.json", None),
    "ar3": ("../cuda_state_2026-10-25/receipts/sdg_ar3.json", "../cuda_state_2026-10-25/receipts/dg_ar3.json"),
    "ar_x5": ("../attnres_x5_2026-10-25/receipts/s1_sdg_ar_x5.json", "../attnres_x5_2026-10-25/receipts/s1_dg_new_ar_x5.json"),
    "ar_x5_q8_0": ("../attnres_x5_2026-10-25/receipts/s1_sdg_ar_x5_q8_0.json", None),
}


def P(*a):
    return os.path.join(src, *a)


def PL(*a):
    return os.path.join(lsrc, *a)


def rows(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def load(p):
    d = json.load(open(p))
    f = lambda m: np.array([[np.nan if v is None else v for v in r] for r in m], dtype=np.float64)  # noqa: E731
    return [{"prefill": f(e["prefill"]), "decode": f(e["decode"]), "free": e["free"]} for e in d]


def same(a, b):
    return os.path.exists(a) and os.path.exists(b) and open(a, "rb").read() == open(b, "rb").read()


def rel(a, ref):
    rng = ref.max(1) - ref.min(1)
    return float((np.abs(a - ref).max(1) / np.where(rng > 0, rng, 1.0)).max())


def vs_ref(dev, base):
    ins, outs = rows(f"{REF[base]}/score_in.jsonl"), rows(f"{REF[base]}/score_out.jsonl")
    worst, greedy, free, dec = 0.0, 0, 0, 0.0
    for k, (a, b) in enumerate(zip(ins, outs)):
        pre = dev[k]["prefill"]
        lse = np.log(np.exp(pre - pre.max(1, keepdims=True)).sum(1)) + pre.max(1)
        n, cont = len(a["prompt_ids"]), a["continuation_ids"]
        lp = np.array([pre[n - 1 + j, t] - lse[n - 1 + j] for j, t in enumerate(cont)])
        ref = np.array(b["logprobs"])
        worst = max(worst, float(np.abs(lp - ref).max() / max(1.0, np.abs(ref).max())))
        greedy += [int(pre[n - 1 + j].argmax()) for j in range(len(cont))] == cont
        free += dev[k]["free"] == cont
        dec = max(dec, rel(dev[k]["decode"], pre))
    N = len(ins)
    return {"logprob_worst_rel_vs_mtstudio": worst, "greedy": f"{greedy}/{N}", "free_running_greedy": f"{free}/{N}",
            "decode_vs_prefill_rel": dec, "_ok": greedy == N and free == N and dec <= 1e-4, "_lp": worst}


def self_only(dev):
    return {"decode_vs_prefill_rel": max(rel(e["decode"], e["prefill"]) for e in dev),
            "logits_finite": bool(all(np.isfinite(e["prefill"]).all() and np.isfinite(e["decode"]).all() for e in dev))}


def tiled_vs_split(t, s, bt, bs):
    r = {"byte_equal": same(bt, bs)}
    for part in ("prefill", "decode"):
        n = sum(len(e[part]) for e in t)
        top = sum(int((a[part].argmax(1) == b[part].argmax(1)).sum()) for a, b in zip(t, s))
        r[f"{part}_argmax_equal"] = f"{top}/{n}"
        r[f"{part}_bit_equal_rows"] = f"{sum(int((a[part] == b[part]).all(1).sum()) for a, b in zip(t, s))}/{n}"
        r[f"{part}_max_abs_d"] = max(float(np.abs(a[part] - b[part]).max()) for a, b in zip(t, s))
        r[f"{part}_max_rel_d"] = max(rel(a[part], b[part]) for a, b in zip(t, s))
        r["_ok_" + part] = top == n
    r["free_ids_equal"] = f"{sum(a['free'] == b['free'] for a, b in zip(t, s))}/{len(t)}"
    r["_ok"] = r.pop("_ok_prefill") and r.pop("_ok_decode") and all(a["free"] == b["free"] for a, b in zip(t, s))
    return r


def gate_summary(p, keys):
    if not os.path.exists(p):
        return {"missing": p}
    try:
        d = json.load(open(p))
    except Exception as e:  # noqa: BLE001
        return {"unreadable": str(e)[:200]}
    if "summary" in d:
        return {"summary": d["summary"], "pass": d["pass"]}
    return {k: d.get(k) for k in keys}


DGK = ("n", "len", "gen", "greedy_identical", "top1_agree", "worst_prefill_rel", "worst_decode_rel", "pass")


def strip(d):
    return {k: (strip(v) if isinstance(v, dict) else v) for k, v in d.items() if not k.startswith("_")}


rep = {"dir": src, "long_dir": lsrc, "gpu": open(os.path.join(os.path.dirname(src), "GPU")).read().strip(),
       "nvidia_smi": [ln.strip() for ln in open(P("hardware.txt")) if "Tesla" in ln or "NVIDIA L4" in ln
                      or "Driver Version" in ln or "release" in ln][:4], "files": {}, "long": {}, "nsys": {}}
ok = True
fails = []
for v in FILES:
    base = v.replace("_q8_0", "")
    q8 = v.endswith("_q8_0")
    r = {"configs": {}}
    devs = {}
    for c in CFGS:
        a, b = P("pin", f"dev_{v}_{c}_a.json"), P("pin", f"dev_{v}_{c}_b.json")
        devs[c] = load(a)
        x = vs_ref(devs[c], base)
        x["run_a_vs_b_byte_equal"] = same(a, b)
        x["state_device_gate"] = gate_summary(P("pin", f"sdg_{v}_{c}.json"), ())
        x["ir_device_gate"] = gate_summary(P("pin", f"dg_{v}_{c}.json"), DGK)
        good = (x["_ok"] and (q8 or x["_lp"] <= 1e-5) and x["state_device_gate"].get("pass") is True
                and x["ir_device_gate"].get("pass") is True and (c.startswith("def") or x["run_a_vs_b_byte_equal"]))
        if not good:
            fails.append(f"{v} {c}")
        ok = ok and good
        r["configs"][c] = x
    for mode in ("def", "det"):
        t = tiled_vs_split(devs[f"{mode}_tiled"], devs[f"{mode}_split"], P("pin", f"dev_{v}_{mode}_tiled_a.json"),
                           P("pin", f"dev_{v}_{mode}_split_a.json"))
        r[f"tiled_vs_split_{mode}"] = t
        if not t["_ok"]:
            fails.append(f"{v} tiled vs split {mode}")
        ok = ok and t["_ok"]
    if base != "state4":
        r["ember_ir_unchanged_document_byte_equal"] = same(P("pin", f"dev_{v}_det_tiled_irdoc.json"),
                                                          P("pin", f"dev_{v}_det_tiled_a.json"))
        ok = ok and r["ember_ir_unchanged_document_byte_equal"]
    r["head_vs_pin_det_tiled_byte_equal"] = same(P("head", f"dev_{v}_det_tiled_a.json"), P("pin", f"dev_{v}_det_tiled_a.json"))
    r["head_state_device_gate"] = gate_summary(P("head", f"sdg_{v}_det_tiled.json"), ())
    ok = ok and r["head_vs_pin_det_tiled_byte_equal"] and r["head_state_device_gate"].get("pass") is True
    if v in EARLIER:
        s, d = EARLIER[v]
        r["earlier"] = {"state_device_gate": gate_summary(os.path.join(H, s), ()), "receipt": s}
        if d:
            r["earlier"]["ir_device_gate"] = gate_summary(os.path.join(H, d), DGK)
            r["earlier"]["ir_device_gate_receipt"] = d
    rep["files"][v] = r
for v in LONG:
    base = v.replace("_q8_0", "")
    r = {"configs": {}}
    devs = {}
    for c in CFGS:
        devs[c] = load(PL("pin", f"long_{v}_{c}_a.json"))
        x = self_only(devs[c])
        if c.startswith("det"):
            x["run_a_vs_b_byte_equal"] = same(PL("pin", f"long_{v}_{c}_a.json"), PL("pin", f"long_{v}_{c}_b.json"))
        good = x["logits_finite"] and x["decode_vs_prefill_rel"] <= 1e-4 and x.get("run_a_vs_b_byte_equal", True)
        if not good:
            fails.append(f"long {v} {c}")
        ok = ok and good
        r["configs"][c] = x
    for mode in ("def", "det"):
        t = tiled_vs_split(devs[f"{mode}_tiled"], devs[f"{mode}_split"], PL("pin", f"long_{v}_{mode}_tiled_a.json"),
                           PL("pin", f"long_{v}_{mode}_split_a.json"))
        r[f"tiled_vs_split_{mode}"] = t
        if not t["_ok"]:
            fails.append(f"long {v} tiled vs split {mode}")
        ok = ok and t["_ok"]
    for mk in ("det_tiled", "def_tiled", "det_split"):
        g = gate_summary(PL("pin", f"dgl_{v}_{mk}.json"), DGK)
        r[f"ir_device_gate_{mk}"] = g
        if g.get("pass") is not True:
            fails.append(f"long {v} ir_device_gate {mk}")
        ok = ok and g.get("pass") is True
    if v in ("ar3", "ar_x5"):
        r["head_vs_pin_det_tiled_byte_equal"] = same(PL("head", f"long_{v}_det_tiled_a.json"),
                                                    PL("pin", f"long_{v}_det_tiled_a.json"))
        ok = ok and r["head_vs_pin_det_tiled_byte_equal"]
    rep["long"][v] = r
nd = P("nsys")
if os.path.isdir(nd):
    for f in sorted(os.listdir(nd)):
        if not f.endswith("_cuda_gpu_kern_sum.csv"):
            continue
        k = f[: -len("_cuda_gpu_kern_sum.csv")]
        kern = {}
        with open(os.path.join(nd, f)) as fh:
            for row in csv.DictReader(fh):
                name = row.get("Name", "")
                short = name.split("(")[0].split("::")[-1].split("<")[0].strip()
                if "attn" in short or "attention" in short:
                    kern[short] = kern.get(short, 0) + int(row.get("Instances", 0) or 0)
        out = {"attention_kernels": kern, "tiled_launched": any(n.startswith("decode_attn_tiled") for n in kern)}
        j = os.path.join(nd, k + ".json")
        if os.path.exists(j):
            out["output_byte_equal_to_unprofiled_run"] = same(j, P("pin", f"dev_{k}_a.json"))
        rep["nsys"][k] = out
ndl = PL("nsys")
if lsrc != src and os.path.isdir(ndl):
    for f in sorted(os.listdir(ndl)):
        if f.endswith("_cuda_gpu_kern_sum.csv"):
            k = f[: -len("_cuda_gpu_kern_sum.csv")]
            kern = {}
            with open(os.path.join(ndl, f)) as fh:
                for row in csv.DictReader(fh):
                    short = row.get("Name", "").split("(")[0].split("::")[-1].split("<")[0].strip()
                    if "attn" in short or "attention" in short:
                        kern[short] = kern.get(short, 0) + int(row.get("Instances", 0) or 0)
            rep["nsys"][k] = {"attention_kernels": kern, "tiled_launched": any(n.startswith("decode_attn_tiled") for n in kern),
                              "session": lsrc}
rep["fails"] = fails
rep["pass"] = bool(ok)
json.dump(strip(rep), open(out_path, "w"), indent=1)
print(json.dumps(strip(rep), indent=1)[:20000])
sys.exit(0 if ok else 1)
