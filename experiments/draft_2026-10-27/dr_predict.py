"""The simulator's per-request counts against a run's (draft_2026-10-27): every request's drafted, accepted and
passes from dr_sim.py's replay of the CPU records (HEAD's plain decode, dr_collect.sh) against the counts ember
reported, and the run's ids against the records'.

    python3 dr_predict.py RECORDS_DIR OUT_DIR TABLES_DIR     (OUT_DIR/e2e/<model>_<tag>.jsonl from vm.sh)"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dr_sim  # noqa: E402

TABLE = {"pick": "h_c8_p10.bin", "pk": "s_c8_p10.bin"}
RUNS = {"pld_k4": (4, None), "pld_k8": (8, None), "pld_k15": (15, None), "table_k4": (4, 0.02), "table_k8": (8, 0.02),
        "table_k15": (15, 0.02), "table_k8_t10": (8, 0.1), "det_pld_k4": (4, None), "det_pld_k8": (8, None),
        "det_table_k4": (4, 0.02), "det_table_k8": (8, 0.02)}


def main(rec_dir, out, tdir):
    res = {}
    for m in ("pick", "pk"):
        recs = []
        for s in ("chat40", "copy20", "carry40"):
            recs += [json.loads(l) for l in open(os.path.join(rec_dir, f"{m}_{s}.jsonl"), encoding="utf-8")]
        dr_sim.attach_max_new(recs, None)
        tab = dr_sim.Table(os.path.join(tdir, TABLE[m]))
        for tag, (k, th) in RUNS.items():
            p = os.path.join(out, "e2e", f"{m}_{tag}.jsonl")
            if not os.path.exists(p):
                continue
            run = [json.loads(l) for l in open(p, encoding="utf-8")]
            drafter = dr_sim.pld if th is None else (lambda h, kk, th=th: dr_sim.pld(h, kk) or tab.draft(h, kk, th))
            ids = same = 0
            for r, g in zip(recs, run):
                ids += r["ids"] == g["ids"] and r["stop_id"] == g["stop_id"] and r["prompt_ids"] == g["prompt_ids"]
                o = dr_sim.replay(r, k, drafter)
                same += (o["drafted"], o["accepted"], o["passes"]) == (g["drafted"], g["accepted"], g["passes"])
            res[f"{m}_{tag}"] = {"requests": len(run), "ids_equal_cpu_head": ids, "counts_equal_simulator": same}
            print(f"{m}_{tag}", json.dumps(res[f"{m}_{tag}"]), flush=True)
    return res


if __name__ == "__main__":
    main(*sys.argv[1:4])
