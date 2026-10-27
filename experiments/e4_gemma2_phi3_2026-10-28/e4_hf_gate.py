#!/usr/bin/env python3
"""E4 gate: ember's CPU interpreter (through ir_adapter, IR_SPEC 11.3) against HF on the synthetic files.

For each family F in e4_synth/: every case's ids are prefilled in one forward, then 16 greedy tokens are decoded one
forward of one id at a time (ember's decode path; under LongRoPE a step that takes the length past original_ctx
recomputes the slot); one session per family, truncated to 0 between sequences. Reports greedy agreement with HF's continuation, the first token's logit distance from HF's
f32 last-position row (max |d| over the row range), and for each case the final decode step's logits against a
fresh prefill of the same ids (decode = fresh run).

    python3 e4_hf_gate.py ADAPTER SYNTH_DIR OUT.json [family ...]
"""
import json
import os
import struct
import subprocess
import sys


def dec(t):
    h = t["f32"]
    n = len(h) // 8
    vals = struct.unpack(">%dI" % n, bytes.fromhex(h))
    fl = [struct.unpack("<f", struct.pack("<I", v))[0] for v in vals]
    return t["shape"], fl


class Adapter:
    def __init__(self, cmd):
        env = {k: v for k, v in os.environ.items() if not k.startswith("EMBER_")}
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.n = 0

    def call(self, method, params):
        self.n += 1
        self.p.stdin.write(json.dumps({"id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        if "error" in r or "skip" in r:
            raise RuntimeError(f"{method}: {r}")
        return r["result"]

    def rows(self, sid, ids):
        r = self.call("forward", {"session": sid, "ids": ids, "report": ["logits"]})
        shape, fl = dec(r["logits"])
        V = shape[1]
        return [fl[i * V:(i + 1) * V] for i in range(shape[0])]


def argmax(row):
    best = 0
    for i, v in enumerate(row):
        if v > row[best]:
            best = i
    return best


def main():
    adapter, synth, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fams = sys.argv[4:] or ["gemma2", "phi3_longrope", "phi3_swa"]
    a = Adapter([adapter, "--conformance"])
    report = {}
    for fam in fams:
        cases = json.load(open(os.path.join(synth, fam + ".cases.json")))
        gguf = os.path.join(synth, fam + ".gguf")
        res = []
        s = a.call("open", {"gguf": gguf, "ir": None, "evict": None})["session"]
        for ci, c in enumerate(cases["cases"]):
            ids = c["ids"]
            a.call("truncate", {"session": s, "n": 0, "report": []})  # a fresh sequence (IR_SPEC 6.9)
            rows = a.rows(s, ids)
            last = rows[-1]
            _, hf = dec(c["hf_last_logits"])
            rng = max(hf) - min(hf)
            dist = max(abs(x - y) for x, y in zip(last, hf))
            toks = [argmax(last)]
            final = None
            while len(toks) < len(c["hf_greedy"]):
                final = a.rows(s, [toks[-1]])[-1]
                toks.append(argmax(final))
            # the last decode step's row against a fresh prefill of the same ids
            a.call("truncate", {"session": s, "n": 0, "report": []})
            fresh = a.rows(s, ids + toks[:-1])[-1]
            fd = max(abs(x - y) for x, y in zip(final, fresh)) if final else 0.0
            res.append({"case": ci, "len": len(ids), "greedy_equal": toks == c["hf_greedy"],
                        "first_equal": toks[0] == c["hf_greedy"][0], "first_dist": dist, "row_range": rng,
                        "first_dist_frac": dist / rng if rng else 0.0, "decode_vs_fresh": fd,
                        "ember": toks, "hf": c["hf_greedy"]})
        a.call("close", {"session": s})
        g = sum(r["greedy_equal"] for r in res)
        worst = max(r["first_dist_frac"] for r in res)
        report[fam] = {"greedy": f"{g}/{len(res)}", "first": f"{sum(r['first_equal'] for r in res)}/{len(res)}",
                       "worst_first_dist_frac": worst, "worst_first_dist": max(r["first_dist"] for r in res),
                       "worst_decode_vs_fresh": max(r["decode_vs_fresh"] for r in res), "cases": res}
        print(fam, report[fam]["greedy"], "greedy; first", report[fam]["first"], "; worst first-token distance",
              "%.3g of the row range (%.3g)" % (worst, report[fam]["worst_first_dist"]),
              "; decode vs fresh %.3g" % report[fam]["worst_decode_vs_fresh"], flush=True)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
