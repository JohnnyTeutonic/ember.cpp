"""ember's own logits before and after the state-token change, bit for bit: ir_logits built at 7d2133d (the
commit before it; ~/ember_old/build) against the current build (~/ember_wt/build_cpu), on the models and
IRs of the existing gates and families, prefill and decode, compared as ir_logits' JSON text (equal text
means equal floats). Writes bit_regression_cpu.json beside this file."""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.expanduser("~/ember_old/build/ir_logits")
NEW = os.path.expanduser("~/ember_wt/build_cpu/ir_logits")
A = "/mnt/c/ml_artifacts"


def rows(path, n=8):
    out = []
    for x in open(path):
        if x.strip():
            r = json.loads(x)
            out.append(r["prompt_ids"] + r["continuation_ids"])
        if len(out) == n:
            break
    return out


short = [[5, 17, 300, 42, 9, 77, 128, 3], [11, 2, 250, 99, 1]]
# gpt2_tiny's 2 stored layers applied 4 times (recursion_smoke.py's "twice" schedule).
rec = json.load(open(f"{A}/gpt2_tiny/gpt2_tiny.ir.json"))
rec["block"]["repeat"] = 4
rec["block"]["weight_index"] = {"per_layer": [0, 1, 0, 1]}
rec_path = os.path.expanduser("~/ember_wt/state_tokens/bitreg_recursion.ir.json")
json.dump(rec, open(rec_path, "w"))
chat = [[2, 400, 51, 77, 1203, 3, 88, 901, 15, 1, 2, 64, 3000, 3], [2, 17, 18, 19, 3]]
cases = {
    "gpt2_tiny": (f"{A}/gpt2_tiny/gpt2_tiny.gguf", None, short),
    "gpt2_tiny_scale_smoke_ir": (f"{A}/gpt2_tiny/gpt2_tiny.gguf", os.path.expanduser("~/ember_wt/scale_smoke.ir.json"), short),
    "gpt2_tiny_recursion_0101": (f"{A}/gpt2_tiny/gpt2_tiny.gguf", rec_path, short),
    "chat_sftfinal_v3": (f"{A}/chat_base/sftfinal/chat_sftfinal_v3.gguf", None, chat),
    "swa_tiny": (f"{A}/swa_tiny/swa_tiny.gguf", None, short),
    "kimi_tiny": (f"{A}/kimi_tiny/kimi_tiny.gguf", None, short),
    "mamba_tiny": (f"{A}/mamba_tiny/mamba_tiny.gguf", None, short),
    "moe_tiny_v3": (f"{A}/moe_tiny/moe_tiny_v3.gguf", None, short),
    "attnres_tiny": (f"{A}/attnres_tiny/attnres_tiny.gguf", None, short),
}
for m in ("p1_b2x3_c1", "p1_b2x3_c1_inject", "b2x4"):
    cases["recursion_gate_" + m] = (f"{A}/recursion_gate/{m}/r.gguf", None, rows(f"{A}/recursion_gate/{m}/score_in.jsonl"))
for m in ("mor3", "mor2p0"):
    cases["mor_gate_" + m] = (f"{A}/mor_gate/{m}/{m}.gguf", None, rows(f"{A}/mor_gate/{m}/score_in.jsonl"))

report = {"old": OLD, "new": NEW, "cases": {}}
for name, (gguf, ir, seqs) in cases.items():
    env = {k: v for k, v in os.environ.items() if k != "EMBER_IR"}
    if ir:
        env["EMBER_IR"] = ir
    outs = []
    for b in (OLD, NEW):
        r = subprocess.run([b, gguf], input=json.dumps(seqs), capture_output=True, text=True, env=env)
        outs.append((r.returncode, r.stdout))
    report["cases"][name] = {"sequences": len(seqs), "positions": sum(map(len, seqs)),
                             "rc": [outs[0][0], outs[1][0]], "bit_equal": outs[0] == outs[1] and outs[0][0] == 0}
report["pass"] = all(c["bit_equal"] for c in report["cases"].values())
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(HERE, "bit_regression_cpu.json"), "w"), indent=1)
