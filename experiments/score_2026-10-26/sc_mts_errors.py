"""mtstudio score on the refusal cases of sc_client.ERRORS (each request list as an input file): rc and stderr.
    sc_mts_errors.py MTSTUDIO SPEC VOCAB WORK OUT.json"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc_client  # noqa: E402

mts, spec, vocab, work, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
os.makedirs(work, exist_ok=True)
res = {}
for name, body in sc_client.ERRORS:
    inp, o = f"{work}/err_{name}.in.jsonl", f"{work}/err_{name}.out.jsonl"
    if isinstance(body, bytes):
        text = body.decode() + "\n"
    elif "requests" not in body:
        res[name] = {"n/a": "mtstudio reads one request per line; no body object"}
        continue
    else:
        text = "".join(json.dumps(r) + "\n" for r in sc_client.subst(body, vocab)["requests"])
    open(inp, "w").write(text)
    r = subprocess.run([mts, "score", spec, "--in", inp, "--out", o], capture_output=True, text=True)
    lines = open(o).read().count("\n") if os.path.exists(o) else None
    res[name] = {"rc": r.returncode, "stderr": r.stderr.strip().splitlines()[-1:] if r.stderr.strip() else [],
                 "records_written": lines}
    print(name, res[name], flush=True)
json.dump(res, open(out, "w"), indent=1)
