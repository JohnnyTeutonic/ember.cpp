"""ember's drafter (draft_table.h through r1_table_check) against r1_sim.py's on the same table and histories.

    python3 r1_table_check.py CHECK_BIN TABLE.bin RECORDS.jsonl[,...] [K] [THETA]     (prints one JSON object)

For every decode position of every record (its prompt ids and the plain decode's ids so far), both drafters draft
up to K tokens (default 8) at THETA (default 0.02); the drafts must be equal at every position. r1_sim reads the
same EMBERDT1 file, so this checks draft_table.h's keys (orders 5-8 included), lookups and cut against r1_sim's."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1_sim  # noqa: E402


def main(binp, table, records, k=8, theta=0.02):
    tab = r1_sim.Table(table)
    qs, want = [], []
    for p in records.split(","):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            seq = list(r["prompt_ids"]) + list(r["ids"] or [])
            for i in range(len(r["prompt_ids"]), len(seq) + 1):
                h = seq[:i]
                qs.append(f"{k} {theta} {len(h)} " + " ".join(map(str, h)))
                want.append(r1_sim.table_draft([tab], h, k, theta))
    out = subprocess.run([binp, table, "16384"], input="\n".join(qs) + "\n", capture_output=True, text=True, check=True)
    got = [[int(x) for x in l.split()] for l in out.stdout.split("\n")[:len(qs)]]
    bad = [i for i, (a, b) in enumerate(zip(want, got)) if a != b]
    res = {"table": table, "orders": tab.orders, "positions": len(qs), "equal": len(qs) - len(bad),
           "nonempty": sum(1 for w in want if w), "drafted_tokens": sum(len(w) for w in want),
           "first_differences": [[i, want[i], got[i]] for i in bad[:5]], "stderr": out.stderr.strip()}
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 8, float(a[4]) if len(a) > 4 else 0.02)
