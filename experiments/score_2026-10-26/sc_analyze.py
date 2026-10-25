"""The /score gate's receipt from the legs' outputs (run in WSL).
    sc_analyze.py CPU_OUT_DIR REQ_DIR OUT.json [CUDA_OUT_DIR]
CPU: ember (default and EMBER_SCORE_FULL=1) against mtstudio score on pick_e1_L16 and pk_base_v1 (ROUNDING 1e-5 of
max(1, max |ref|), and the plain |d| <= 1e-5 asked for), against HF float32 on Pythia-160m and Qwen2.5-0.5B (|d| /
HF's logit range <= 1e-4, the real-model rule), default = full bit for bit, one call = calls of 50 bit for bit,
/chat unchanged by /score, refusals against mtstudio's, throughput. CUDA (optional): each model's device run against
ember's CPU (ROUNDING, argmax), EMBER_DETERMINISTIC=1 run to run bit for bit."""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
cpu, reqd, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
cuda = sys.argv[4] if len(sys.argv) > 4 else None


def cmp(ref, got, *extra):
    r = subprocess.run([sys.executable, HERE + "/sc_client.py", "compare", ref, got, *extra], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    return json.loads(r.stdout)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def same_bytes(a, b):
    return os.path.exists(a) and os.path.exists(b) and open(a, "rb").read() == open(b, "rb").read()


def short(c, hf=False):
    keys = ["scored", "over_ctx_ref", "over_ctx_same", "fields_same", "requests_bit_equal", "logprob_worst_abs",
            "logprob_worst_rounding", "sum_worst_abs", "argmax_same", "argmax_positions", "argmax_flips"]
    if hf:
        keys.append("logprob_worst_over_range")
    return {k: c[k] for k in keys} | {"keys_same": c["keys_same"]}


R = {"requests": {f: {"sha256": sha(f"{reqd}/{f}"), "n": sum(1 for _ in open(f"{reqd}/{f}"))}
                  for f in sorted(os.listdir(reqd)) if f.endswith(".jsonl")}, "cpu": {}, "cuda": {}}
ok = True
for m, r in (("pick", "pk"), ("pkbase", "pk"), ("pythia", "pythia"), ("qwen", "qwen"), ("smol", "smol")):
    d = {}
    if not os.path.exists(f"{cpu}/{m}_def.stats.json"):
        R["cpu"][m] = "not run"
        ok = False
        continue
    st_def = json.load(open(f"{cpu}/{m}_def.stats.json"))
    st_full = json.load(open(f"{cpu}/{m}_full.stats.json"))
    d["health"] = {k: st_def["health"].get(k) for k in ("context", "vocab_size", "layers", "hidden_size", "tokenizer")}
    log = f"{cpu}/{m}_def.cwd/debugging.log"
    if os.path.exists(log):
        d["path"] = "interpreter" if "through the paperkiln IR interpreter" in open(log, errors="replace").read() else "tuned llama"
    else:
        d["path"] = "interpreter" if m in ("pythia", "qwen") else "tuned llama"
    req = f"{r}_requests"
    e_def, e_full = f"{cpu}/{m}_def.{req}.jsonl", f"{cpu}/{m}_full.{req}.jsonl"
    if os.path.exists(e_full):
        d["full_reference_on"] = f"{req} (all)"
        d["default_eq_full_bytes"] = same_bytes(e_def, e_full)
    else:  # the 60-request subset (sc_gate_cpu3.sh): every record's values bit for bit
        sub = f"{cpu}/{m}_full.{r}_sub60.jsonl"
        c = cmp(sub, e_def, "--label", f"{m} full subset vs default")
        d["full_reference_on"] = f"{r}_sub60 ({c['keys_ref']} requests)"
        d["default_eq_full_bytes"] = (c["over_ctx_same"] == c["keys_ref"] and c["requests_bit_equal"] == c["scored"]
                                      and c["argmax_same"] == c["argmax_positions"])
    ch = f"{cpu}/{m}_def.{req}.chunk.jsonl"
    d["one_call_eq_chunks_of_50_bytes"] = same_bytes(e_def, ch) if os.path.exists(ch) else "not run"
    if m in ("pick", "pkbase"):
        c = cmp(f"{cpu}/mts_{m}.jsonl", e_def, "--label", f"{m} ember vs mtstudio")
        d["vs_mtstudio"] = short(c)
        # The pass is ROUNDING (1e-5 of max(1, max |ref|)); the plain |d| <= 1e-5 asked for is recorded beside it,
        # with the float64 arbitration (sc_f64.py) of the largest differences.
        d["plain_abs_1e-5_met"] = c["logprob_worst_abs"] <= 1e-5
        f64 = f"{cpu}/f64_{m}.json"
        if os.path.exists(f64):
            F = json.load(open(f64))
            d["float64_on_the_10_largest"] = {k: F[k] for k in ("ember_closer", "worst_ember_vs_f64", "worst_mtstudio_vs_f64")}
        g = (c["keys_same"] and c["over_ctx_same"] == c["keys_ref"] and c["logprob_worst_rounding"] <= 1e-5
             and c["argmax_same"] == c["argmax_positions"] and c["sum_worst_abs"] <= 1e-4 and c["fields_same"] == c["scored"])
        d["pass"] = g and d["default_eq_full_bytes"] and d["one_call_eq_chunks_of_50_bytes"] in (True, "not run")
        b = f"{cpu}/mts_{m}_bench256.jsonl"
        if os.path.exists(b):
            d["bench256_vs_mtstudio"] = short(cmp(b, f"{cpu}/{m}_def.{r}_bench256.jsonl"))
    else:
        c = cmp(f"{cpu}/hf_{m}.jsonl", e_def, "--hf", "--label", f"{m} ember vs HF")
        d["vs_hf"] = short(c, hf=True)
        g = c["keys_same"] and c["logprob_worst_over_range"] <= 1e-4 and c["argmax_same"] == c["argmax_positions"]
        d["pass"] = g and d["default_eq_full_bytes"] and d["one_call_eq_chunks_of_50_bytes"] in (True, "not run")
    d["chat_unchanged_by_score"] = st_def.get("chat_same")
    d["pass"] = d["pass"] and bool(d["chat_unchanged_by_score"])
    thr = {}
    for tag, st in (("default", st_def), ("full", st_full)):
        for call in st["calls"]:
            if "chunk" in call["req"]:
                continue
            thr[f"{tag}:{call['req']}"] = {"n": call["n"], "req_per_s": call["req_per_s"], "wall_s": call["wall_s"],
                                          "positions": call["stats"]["positions"],
                                          "reused_positions": call["stats"]["reused_positions"]}
    b256 = f"{r}_bench256"
    if os.path.exists(f"{cpu}/{m}_def.{b256}.jsonl") and os.path.exists(f"{cpu}/{m}_full.{b256}.jsonl"):
        d["bench256_default_eq_full_bytes"] = same_bytes(f"{cpu}/{m}_def.{b256}.jsonl", f"{cpu}/{m}_full.{b256}.jsonl")
        d["pass"] = d["pass"] and d["bench256_default_eq_full_bytes"]
    tm = f"{cpu}/mts_{m}.time"
    if os.path.exists(tm):
        thr["mtstudio:gate set (process incl. load)"] = open(tm).read().strip()
    d["throughput"] = thr
    d["errors"] = {k: v["status"] for k, v in st_def.get("errors", {}).items()}
    R["cpu"][m] = d
    ok = ok and d["pass"]
me = f"{cpu}/mts_errors.json"
if os.path.exists(me):
    mts = json.load(open(me))
    ember_err = R["cpu"]["pick"]["errors"]
    rows = {}
    for k, v in mts.items():
        rows[k] = {"ember_status": ember_err.get(k), "mtstudio_rc": v.get("rc"), "mtstudio": v.get("stderr") or v.get("n/a")}
    rows["no_requests"] = {"ember_status": ember_err.get("no_requests"), "mtstudio": "n/a (one request per line)"}
    R["refusals"] = rows
    agree = all((r["ember_status"] == 400) == (r.get("mtstudio_rc") not in (0, None)) for k, r in rows.items()
                if k not in ("text_fields_only", "no_requests", "not_json", "empty_list"))
    R["refusals_agree"] = agree
    ok = ok and agree
if cuda:
    for m, r in (("pick", "pk"), ("pkbase", "pk"), ("pythia", "pythia"), ("qwen", "qwen")):
        f = f"{cuda}/{m}_cuda_def.{r}_requests.jsonl"
        if not os.path.exists(f):
            R["cuda"][m] = "not run"
            continue
        d = {"vs_cpu": short(cmp(f"{cpu}/{m}_def.{r}_requests.jsonl", f, "--label", f"{m} CUDA vs CPU")),
             "det_vs_cpu": short(cmp(f"{cpu}/{m}_def.{r}_requests.jsonl", f"{cuda}/{m}_cuda_det1.{r}_requests.jsonl")),
             "det_run_to_run_bytes": same_bytes(f"{cuda}/{m}_cuda_det1.{r}_requests.jsonl", f"{cuda}/{m}_cuda_det2.{r}_requests.jsonl")}
        c = d["vs_cpu"]
        if os.path.exists(f"{cpu}/hf_{m}.jsonl"):  # the interpreter models: the real-model rule against HF as well
            d["vs_hf"] = short(cmp(f"{cpu}/hf_{m}.jsonl", f, "--hf"), hf=True)
        close = c["logprob_worst_rounding"] <= 1e-5
        if not close and "vs_hf" in d:  # float32 conditioning (Pythia): judged by the real-model rule, as on the CPU
            close = d["vs_hf"]["logprob_worst_over_range"] <= 1e-4 and d["vs_hf"]["argmax_same"] == d["vs_hf"]["argmax_positions"]
            d["judged_by"] = "real-model rule against HF (ROUNDING against the CPU not met)"
        d["pass"] = close and c["argmax_same"] == c["argmax_positions"] and d["det_run_to_run_bytes"]
        st = json.load(open(f"{cuda}/{m}_cuda_def.stats.json"))
        d["throughput"] = {c2["req"]: {"n": c2["n"], "req_per_s": c2["req_per_s"]} for c2 in st["calls"]}
        d["chat_unchanged_by_score"] = st.get("chat_same")
        d["errors"] = {k: v["status"] for k, v in st.get("errors", {}).items()}
        R["cuda"][m] = d
        ok = ok and d["pass"]
R["pass"] = ok
json.dump(R, open(out_p, "w"), indent=1)
print(json.dumps({"pass": ok, **{m: (R["cpu"][m]["pass"] if isinstance(R["cpu"][m], dict) else R["cpu"][m]) for m in R["cpu"]},
                  **{"cuda_" + m: (v["pass"] if isinstance(v, dict) else v) for m, v in R["cuda"].items()}}))
