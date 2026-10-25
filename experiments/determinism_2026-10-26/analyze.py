"""The deterministic-mode gate's receipt (run in WSL after drive.sh and cpu_legs.sh):

    python3 analyze.py VM_OUT CPU_WORK OUT.json

VM_OUT is the VM's out/ (vm.sh); CPU_WORK is cpu_legs.sh's work directory (plain/<model>_cpu: the CPU build's
360-request records). Writes OUT.json:
  - gpu, builds (sha256 of the binaries), sass (cuobjdump -sass, head d74e052 against this work, kernel by kernel);
  - plain: every compared pair of 360-request runs (x5_plain_compare.py), per set and in total;
  - on_vs_cpu: each model's first on-mode run against the CPU build's (replies, stop, ids, prompt ids; first-token
    logits) via x5_plain_compare.py;
  - cost: bench.py summaries, on against off;
  - interpreter: ir_device_dump comparisons (head vs new off, new on vs new on, new off vs new on);
  - gates: the pass/fail lines this folder's README reports."""
import json
import os
import statistics
import subprocess
import sys

VM, CPU, OUT = sys.argv[1:4]
X = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments/attnres_x5_2026-10-25"
rep = {"vm_out": VM, "cpu_work": CPU}
rep["gpu"] = open(os.path.join(VM, "gpu.txt")).read().split("\n")[0]
rep["builds"] = [x.split() for x in open(os.path.join(VM, "bins.sha256")) if x.strip()]
s = json.load(open(os.path.join(VM, "sass.json")))
rep["sass"] = {k: s[k] for k in ("arches", "head_functions", "new_functions", "identical", "differ", "only_in_head",
                                 "only_in_new", "differing", "only_in_head_names", "only_in_new_names",
                                 "every_head_kernel_identical")}


def pair_summary(d):
    out = {"bit_identical": d["bit_identical"]}
    tot = {"requests": 0, "replies_identical": 0, "ids_identical": 0, "first_token_logits_bit_equal": 0,
           "requests_all_positions_bit_equal": 0, "positions": 0, "positions_bit_equal": 0, "worst_abs": 0.0}
    for name, v in d["sets"].items():
        ep = v.get("every_position", {})
        out[name] = {"n": v["n"], "replies_identical": v["replies_identical"], "ids_identical": v["ids_identical"],
                     "prompt_ids_identical": v["prompt_ids_identical"],
                     "first_token_logits_bit_equal": v["first_token_logits_bit_equal"],
                     "first_token_logits_worst_abs": v["first_token_logits_worst_abs"],
                     "requests_all_positions_bit_equal": v["all_positions_logits_bit_equal"],
                     "positions": ep.get("rows"), "positions_bit_equal": ep.get("bit_equal_rows"),
                     "positions_worst_abs": ep.get("worst_abs"), "positions_top1": ep.get("top1_agree")}
        tot["requests"] += v["n"]
        tot["replies_identical"] += v["replies_identical"]
        tot["ids_identical"] += v["ids_identical"]
        tot["first_token_logits_bit_equal"] += v["first_token_logits_bit_equal"]
        tot["requests_all_positions_bit_equal"] += v["all_positions_logits_bit_equal"]
        if ep:
            tot["positions"] += ep["rows"]
            tot["positions_bit_equal"] += ep["bit_equal_rows"]
            tot["worst_abs"] = max(tot["worst_abs"], ep["worst_abs"])
    out["total"] = tot
    return out


rep["plain"] = {}
for f in sorted(os.listdir(VM)):
    if f.startswith("cmp_") and f.endswith(".json"):
        rep["plain"][f[4:-5]] = pair_summary(json.load(open(os.path.join(VM, f))))

rep["health"] = {}
for run in sorted(os.listdir(os.path.join(VM, "plain"))):
    m = json.load(open(os.path.join(VM, "plain", run, "meta.json")))
    rep["health"][run] = m["health"].get("deterministic")

rep["on_vs_cpu"] = {}
for model in ("pick", "sft"):
    a, b = os.path.join(CPU, "plain", model + "_cpu"), os.path.join(VM, "plain", model + "_on_a")
    if not (os.path.exists(a + "/records.jsonl") and os.path.exists(b + "/first_logits.f32")):
        continue
    r = subprocess.run([sys.executable, X + "/x5_plain_compare.py", a, b], capture_output=True, text=True)
    d = json.loads(r.stdout)
    rep["on_vs_cpu"][model] = pair_summary(d)
    # The on-mode second run's replies against the CPU's (records only).
    ra = [json.loads(x) for x in open(a + "/records.jsonl", encoding="utf-8") if x.strip()]
    rb = [json.loads(x) for x in open(os.path.join(VM, "plain", model + "_on_b", "records.jsonl"), encoding="utf-8") if x.strip()]
    rep["on_vs_cpu"][model]["on_b_replies_and_ids_equal_cpu"] = sum(
        x["reply"] == y["reply"] and x["stop"] == y["stop"] and x["ids"] == y["ids"] for x, y in zip(ra, rb))

tmp = OUT + ".vs_cpu.tmp"
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vs_cpu.py"), VM, CPU, tmp],
               capture_output=True, check=True)
rep["replies_vs_cpu"] = json.load(open(tmp))
os.remove(tmp)

rep["cost"] = {}
for f in sorted(os.listdir(VM)):
    if f.startswith("bench_") and f.endswith(".json"):
        rep["cost"][f[6:-5]] = json.load(open(os.path.join(VM, f)))["summary"]
for model in ("pick", "sft"):
    off = [v for k, v in rep["cost"].items() if k.startswith(model + "_off")]
    on = [v for k, v in rep["cost"].items() if k.startswith(model + "_on")]
    if off and on:
        def m(xs, k):
            return statistics.fmean(x[k] for x in xs)
        rep["cost"][model + "_on_over_off"] = {
            k: round(m(on, k) / m(off, k), 4) for k in ("decode_ms_per_token_p50", "decode_ms_per_token_mean",
                                                       "prefill_wall_ms_p50", "prefill_ttft_ms_p50")}

rep["interpreter"] = {}
if os.path.exists(os.path.join(VM, "dumps.txt")):
    for line in open(os.path.join(VM, "dumps.txt")):
        parts = line.split()
        if parts:
            rep["interpreter"][parts[0]] = dict(p.split("=", 1) for p in parts[1:])

P = rep["plain"]
g = {}
g["sass: every d74e052 kernel's SASS identical in this build"] = rep["sass"]["every_head_kernel_identical"]
for model in ("pick", "sft"):
    k = f"{model}_on_a_vs_{model}_on_b"
    if k in P:
        t = P[k]["total"]
        g[f"{model}: on vs on, every sampled position bit-identical, replies {t['requests']}/{t['requests']}"] = (
            P[k]["bit_identical"] and t["positions"] == t["positions_bit_equal"] and t["replies_identical"] == t["requests"])
    k = f"{model}_off_a_vs_{model}_head_a"
    if k in P:
        t = P[k]["total"]
        g[f"{model}: off vs d74e052, replies and ids {t['requests']}/{t['requests']}"] = (
            t["replies_identical"] == t["requests"] and t["ids_identical"] == t["requests"])
        tc = P[k]["tinychat"]
        g[f"{model}: off vs d74e052, fresh-prompt first-token logits bit-equal {tc['n']}/{tc['n']}"] = (
            tc["first_token_logits_bit_equal"] == tc["n"])
    if model in rep["on_vs_cpu"]:
        t = rep["on_vs_cpu"][model]["total"]
        if model == "pick":
            g[f"{model}: on replies = CPU replies {t['requests']}/{t['requests']}"] = (
                t["replies_identical"] == t["requests"] and t["ids_identical"] == t["requests"])
        # Every run's agreement with the CPU, request by request: on mode agrees exactly where d74e052 does.
        rv = rep["replies_vs_cpu"]
        heads = [k for k in rv if k.startswith(model + "_head")]
        ons = [k for k in rv if k.startswith(model + "_on")]
        if heads and ons:
            g[f"{model}: on vs CPU = d74e052 vs CPU, request by request ({rv[ons[0]]['equal']}/{rv[ons[0]]['n']})"] = all(
                rv[k].get("differing_keys", []) == rv[heads[0]].get("differing_keys", []) for k in ons)
g["/health reports deterministic only on the on runs"] = all(
    (v is True) == ("_on_" in k) for k, v in rep["health"].items())
interp = rep["interpreter"]
if interp:
    g["interpreter: head = new off, every dump"] = all(v.get("head_vs_off") == "identical" for v in interp.values())
    g["interpreter: new on = new on, every dump"] = all(v.get("on_a_vs_on_b") == "identical" for v in interp.values())
rep["gates"] = g
rep["pass"] = all(g.values())
json.dump(rep, open(OUT, "w"), indent=1)
print(json.dumps(g, indent=1))
print("pass", rep["pass"])
