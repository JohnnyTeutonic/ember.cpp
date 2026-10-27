"""The CUDA leg's numbers from tig_vm.sh's outputs (DIR = /mnt/c/ml_artifacts/ember_tig_gate/<session>/out), against
mtstudio score and ember's CPU (tig_gate_cpu.py's ~/tig/work/cpu_<variant>_{tagged,untagged}.json), per variant:
  tagged: log-probs vs mtstudio, greedy (argmax at every scored position) and free-running greedy n/40, logits vs the
    CPU (max |dlogit| / the CPU row's range) and top-1, decode vs prefill on the device;
  run to run: the default mode's two runs, and EMBER_DETERMINISTIC=1's two runs, byte-equal outputs;
  untagged (EMBER_IR): logits vs the CPU's untagged run, greedy; the device's tagged-untagged gap in paperkiln's
    measure (max |d| / max(1, max |tagged|), first 8 prompts' rows; and all 40 sequences);
  rec_app.{4,5} NaN copy: tagged logits all NaN (null in the JSON), untagged bit-equal to the untagged run;
  state_device_gate and ir_device_gate summaries and verdicts.
Gate: tagged greedy and free greedy 40/40, logits within 1e-4 of the CPU's row range (the device gates' ROUNDING
bound), state_device_gate and ir_device_gate pass, EMBER_DETERMINISTIC=1 byte-equal run to run, the untagged run
within 1e-4 of the CPU's untagged run and its gap above 1e-3, the NaN checks.
    python3 tig_analyze_cuda.py DIR OUT.json"""
import json
import os
import subprocess
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
W = os.path.expanduser("~/tig/work")
D = "/mnt/c/ml_artifacts/attnres_gate_rec/ar_rec"
src, out_path = sys.argv[1], sys.argv[2]
ins = [json.loads(x) for x in open(D + "/score_in.jsonl") if x.strip()]
N = len(ins)


def lp(kind, a, b):
    r = subprocess.run([sys.executable, os.path.join(H, "tig_logprobs.py"), kind, a, b], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr[-1000:])
    return json.loads(r.stdout)


def load(p):
    return [np.array([[np.nan if v is None else v for v in row] for row in e["prefill"]], dtype=np.float64)
            for e in json.load(open(p))]


def same(a, b):
    return open(a, "rb").read() == open(b, "rb").read()


rep = {"dir": src, "gpu": open(os.path.join(os.path.dirname(src), "GPU")).read().strip(),
       "nvidia_smi": [ln.strip() for ln in open(os.path.join(src, "hardware.txt")) if "Tesla" in ln or "NVIDIA L4" in ln
                      or "Driver Version" in ln or "release" in ln][:4]}
ok = True
for v, cv in (("ar_rec", "f32"), ("ar_rec_q8_0", "q8_0")):
    f = lambda n: os.path.join(src, n)
    r = {"tagged": lp("device", f(f"dev_{v}_a.json"), f"{W}/cpu_{cv}_tagged.json"),
         "untagged": lp("device", f(f"dev_{v}_untagged.json"), f"{W}/cpu_{cv}_untagged.json")}
    r["run_to_run"] = {"default_mode_byte_equal": same(f(f"dev_{v}_a.json"), f(f"dev_{v}_b.json")),
                       "deterministic_byte_equal": same(f(f"dev_{v}_det_a.json"), f(f"dev_{v}_det_b.json")),
                       "deterministic_vs_default_byte_equal": same(f(f"dev_{v}_a.json"), f(f"dev_{v}_det_a.json"))}
    t, u = load(f(f"dev_{v}_a.json")), load(f(f"dev_{v}_untagged.json"))
    g8 = max(float(np.abs(t[k][:len(ins[k]["prompt_ids"])] - u[k][:len(ins[k]["prompt_ids"])]).max()
                   / max(1.0, float(np.abs(t[k][:len(ins[k]["prompt_ids"])]).max()))) for k in range(8))
    g40 = max(float(np.abs(t[k] - u[k]).max() / max(1.0, float(np.abs(t[k]).max()))) for k in range(N))
    r["untagged_gap"] = {"device_first8_prompts": g8, "device_all40_sequences": g40}
    n45 = load(f(f"dev_{v}_nan45.json"))
    r["nan_rec_app_4_5"] = {"tagged_all_nan": bool(all(np.isnan(x).all() for x in n45)),
                            "untagged_bit_equal": same(f(f"dev_{v}_nan45_untagged.json"), f(f"dev_{v}_untagged.json"))}
    sdg = json.load(open(f(f"sdg_{v}.json")))
    r["state_device_gate"] = {"summary": sdg["summary"], "pass": sdg["pass"]}
    dg = json.load(open(f(f"dg_{v}.json")))
    r["ir_device_gate"] = {k: dg.get(k) for k in ("n", "len", "gen", "greedy_identical", "top1_agree",
                                                    "worst_prefill_rel", "worst_decode_rel", "pass")}
    T, U = r["tagged"], r["untagged"]
    r["pass"] = bool(T["greedy"] == f"{N}/{N}" and T["free_running_greedy"] == f"{N}/{N}"
                     and T["logits_worst_rel_vs_cpu"] <= 1e-4 and U["logits_worst_rel_vs_cpu"] <= 1e-4
                     and r["run_to_run"]["deterministic_byte_equal"] and g8 > 1e-3
                     and r["nan_rec_app_4_5"]["tagged_all_nan"] and r["nan_rec_app_4_5"]["untagged_bit_equal"]
                     and sdg["pass"] and dg.get("pass"))
    ok = ok and r["pass"]
    rep[v] = r
rep["pass"] = ok
json.dump(rep, open(out_path, "w"), indent=1)
print(json.dumps(rep, indent=1))
sys.exit(0 if ok else 1)
