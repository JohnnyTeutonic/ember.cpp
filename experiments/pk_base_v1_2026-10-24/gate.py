#!/usr/bin/env python3
"""Parity gates: start ember's CPU tinyllama_server on GGUF and mtstudio chat on the run's spec,
check /health on both, run ember.cpp/tools/chat_parity.py once per JOB, stop both servers.

    gate.py TAG RUN_DIR NAME GGUF JOB [JOB ...]
    JOB = N,OUT[,PROMPTS[,MAX_NEW]]       chat_parity on the first N prompts (frozen file by default)
    JOB = ref,OUT,REF_JSONL                ember only: replay REF's prompts (a chat_parity record
                                           file) and compare with its recorded mtstudio replies

The spec is the run's own <dir>_spec.json re-pointed at RUN_DIR (out_dir) with data.vocab = the
run's f32 GGUF (the vocabulary the export embedded). mtstudio is not started for ref-only runs."""
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
EMBER = os.path.expanduser("~/ember_wt/build_cpu/tinyllama_server")
MTS = os.path.expanduser("~/pk_cpu_1024/build/mtstudio")
PK_TOOLS = os.path.expanduser("~/pk_cpu_1024/src/tools")
WORK = os.path.expanduser("~/e24")
PE, PM = 8097, 8098

tag, run_dir, name, gguf = sys.argv[1:5]
jobs = [j.split(",") for j in sys.argv[5:]]
need_mts = any(j[0] != "ref" for j in jobs)
os.makedirs(WORK + "/specs", exist_ok=True)
os.makedirs(WORK + "/logs", exist_ok=True)


def health(port):
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2))
    except Exception:
        return None


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def start(argv, port, log):
    if health(port) is not None:
        sys.exit(f"port {port} already answers /health; refusing to share it")
    p = subprocess.Popen(argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    for _ in range(900):
        if p.poll() is not None:
            sys.exit(f"{argv[0]} on {port} exited {p.returncode}; see {log}")
        h = health(port)
        if h is not None:
            return p, h
        time.sleep(1)
    p.kill()
    sys.exit(f"{argv[0]} on {port} did not come up")


def stop(p):
    p.terminate()
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()


spec_src = next(c for c in (f"{run_dir}/{os.path.basename(run_dir)}_spec.json", f"{run_dir}/base_spec.json",
                            f"{run_dir}/gate_spec.json") if os.path.exists(c))
spec = json.load(open(spec_src))
spec["out_dir"], spec["name"] = run_dir, name
spec.setdefault("data", {})["vocab"] = os.path.join(run_dir, name + ".gguf")
spec_path = f"{WORK}/specs/{tag}.json"
json.dump(spec, open(spec_path, "w"), indent=1)

procs = []
try:
    pe, he = start([EMBER, gguf, str(PE), "127.0.0.1", "-", "0"], PE, f"{WORK}/logs/{tag}_ember.log")
    procs.append(pe)
    print("ember /health:", json.dumps(he), flush=True)
    if need_mts:
        pm, hm = start([MTS, "chat", spec_path, "--port", str(PM)], PM, f"{WORK}/logs/{tag}_mtstudio.log")
        procs.append(pm)
        print("mtstudio /health:", json.dumps(hm), flush=True)
        if hm.get("model") != name:
            sys.exit(f"mtstudio /health names {hm.get('model')!r}, not {name!r}")
    for j in jobs:
        t0 = time.time()
        if j[0] == "ref":
            out, ref = j[1], j[2]
            rows = [json.loads(l) for l in open(ref, encoding="utf-8")]
            same = same_f32 = 0
            with open(out, "w", encoding="utf-8") as f:
                for r in rows:
                    body = {"user_input": r["prompt"], "max_new_tokens": 60, "temperature": 1.0,
                            "top_k": 1, "top_p": 1.0, "seed": 0}
                    e = post(PE, body)
                    rec = {"i": r["i"], "prompt": r["prompt"], "ember": e["reply"], "mtstudio": r["mtstudio"],
                           "ember_tokens": e["tokens"], "mts_tokens": r["mts_tokens"],
                           "ember_stop": e["stop_reason"], "mts_stop": r["mts_stop"],
                           "ember_prompt_tokens": e["prompt_tokens"],
                           "bpe_py_prompt_tokens": r["bpe_py_prompt_tokens"], "ember_f32": r["ember"]}
                    same += e["reply"] == r["mtstudio"]
                    same_f32 += e["reply"] == r["ember"]
                    f.write(json.dumps(rec) + "\n")
            print(f"ref replay {out}: n={len(rows)} reply identical to mtstudio f32 {same}, "
                  f"to ember f32 {same_f32}; wall {time.time() - t0:.0f} s", flush=True)
            continue
        n, out = j[0], j[1]
        env = {**os.environ, "PARITY_EMBER_PORT": str(PE), "PARITY_MTS_PORT": str(PM), "PARITY_GGUF": gguf,
               "PAPERKILN_TOOLS": PK_TOOLS}
        if len(j) > 2:
            env["PARITY_PROMPTS"] = j[2]
        if len(j) > 3:
            env["PARITY_MAX_NEW"] = j[3]
        r = subprocess.run([sys.executable, f"{ROOT}/ember.cpp/tools/chat_parity.py", n, out], env=env)
        print(f"chat_parity {out} rc={r.returncode} wall {time.time() - t0:.0f} s", flush=True)
finally:
    for p in procs:
        stop(p)
