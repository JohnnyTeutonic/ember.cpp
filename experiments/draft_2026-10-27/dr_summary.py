"""Summary of draft_2026-10-27's end-to-end runs: per model, configuration and request set, decode tokens per
second, the ratio to plain decode of the same mode (off / det_off), acceptance (accepted / drafted), tokens per
pass ((accepted + passes) / passes), steps (plain steps + passes) per decode token, the drafter's own time per
call and per pass, and ids against plain decode.

    python3 dr_summary.py OUTDIR           (OUTDIR/e2e/<model>_<tag>.jsonl from vm.sh)"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dr_e2e import totals  # noqa: E402

SETS = ("chat40", "copy20", "carry40")


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main(out):
    files = sorted(glob.glob(os.path.join(out, "e2e", "*.jsonl")))
    runs = {os.path.basename(f)[:-6]: load(f) for f in files}
    for model in ("pick", "pk"):
        tags = sorted(t[len(model) + 1:] for t in runs if t.startswith(model + "_"))
        if not tags:
            continue
        print(f"== {model}")
        for tag in tags:
            recs = runs[f"{model}_{tag}"]
            base = runs.get(f"{model}_{'det_off' if tag.startswith('det_') else 'off'}")
            for s in SETS + ("all",):
                rs = recs if s == "all" else [r for r in recs if r["set"] == s]
                bs = base if s == "all" else [r for r in base if r["set"] == s]
                t, tb = totals(rs), totals(bs)
                ids = sum(x["ids"] == y["ids"] and x["stop_id"] == y["stop_id"] for x, y in zip(rs, bs))
                tpp = (t["accepted"] + t["passes"]) / t["passes"] if t["passes"] else None
                calls = sum(r.get("draft_calls", 0) for r in rs)
                dms = sum(r.get("draft_ms", 0.0) for r in rs)
                steps = sum(max(0, (r["tokens"] or 0) - 1) for r in rs) - t["accepted"]
                print(f"  {tag:14s} {s:8s} {t['decode_tok_per_s']:8.1f} tok/s  x{t['decode_tok_per_s'] / tb['decode_tok_per_s']:.3f}"
                      f"  acc {t['acceptance'] if t['acceptance'] is None else round(t['acceptance'], 2)}"
                      f"  tok/pass {tpp if tpp is None else round(tpp, 2)}  passes {t['passes']}  steps {steps}"
                      f"  draft {dms * 1000 / calls if calls else 0:.1f} us/call ({dms * 1000 / t['passes'] if t['passes'] else 0:.1f} us/pass)"
                      f"  ids {ids}/{len(rs)}  warm {t['warm_pass_same']}")


if __name__ == "__main__":
    main(sys.argv[1])
