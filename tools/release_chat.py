#!/usr/bin/env python3
"""One command from a paperkiln checkpoint to a chat release (LOCAL_LLM_STACK.md section 7,
"Release"): GGUF export, ember's parity gate, evals, the WebAssembly build and the Space folder.
Publishing is a separate subcommand that runs only on Jonathan's explicit go for that release.

    python3 tools/release_chat.py build --spec SPEC.json --ckpt MODEL.safetensors --out DIR \\
        --mtstudio BIN --ember-cpu BIN [--ember-cuda BIN] --paperkiln DIR \\
        [--reference REF.jsonl] [--mts-env K=V ...] [--space-model q8|f32] \\
        [--json-inc DIR] [--emsdk DIR] [--stages export,gate,evals,wasm,space] [--redo]
    python3 tools/release_chat.py publish --out DIR --env-file FILE --confirm ember-chat

Stages (each writes its receipt into DIR/release.json; a stage already "ok" is skipped
unless --redo; every stage that runs servers, builds or evals takes the workspace's
.cpu_lock first, waits while another session holds it, and removes only its own lock):

- export: `mtstudio export SPEC --from CKPT --formats gguf,gguf_q8_0_all`; the GGUFs land in
  the spec's out_dir as NAME.gguf and NAME_q8_0_all.gguf and are copied to DIR/model/.
- gate:   greedy parity of the f32 GGUF on ember's CPU (and CUDA with --ember-cuda) against
  mtstudio on the 40 frozen prompts (tools/chat_parity.py against a live `mtstudio chat`, or
  --reference, a recorded mtstudio greedy file): every reply, token count and stop reason must
  match. The Q8 GGUF passes on first-token logits (experiments/quant_2026-10-19 criterion):
  top-1 40/40 and max |dlogit| <= 1% of the logit range against f32 on the CPU.
- evals:  paperkiln's tools/qa_eval.py (simple-QA probe, greedy) and tools/chat_eval.py
  (held-out chat prompts) through ember; numbers are recorded, not gated.
- wasm:   deploy/wasm/build.sh (heap ceiling from the model's size), then tools/wasm_parity.mjs:
  the WASM build's greedy replies for the Space model must equal ember CPU's, 40/40.
- space:  tools/build_wasm_space.sh with the Space model (q8 by default; it needs the Q8 gate).

publish reads only HF_TOKEN from --env-file (never printed), checks the account is
johnnyteutonic, needs --confirm ember-chat and a complete build whose every stage passed, and
uploads DIR/space to the ember-chat Space.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/.cpu_lock"
STAGES = ["export", "gate", "evals", "wasm", "space"]
PORT_EMBER, PORT_MTS = 8095, 8096
OWNER = "johnnyteutonic"


# ---- receipts, hashing, the CPU lock --------------------------------------
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


class Receipt:
    def __init__(self, out):
        self.path = os.path.join(out, "release.json")
        self.data = json.load(open(self.path)) if os.path.exists(self.path) else {"stages": {}}

    def done(self, stage):
        return self.data["stages"].get(stage, {}).get("ok") is True

    def put(self, stage, ok, **info):
        self.data["stages"][stage] = {"ok": ok, "at": now(), **info}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)


class CpuLock:
    """The workspace's one-heavy-job rule (AGENTS.md, "Shared CPU budget")."""
    def __init__(self, job, path=LOCK, wait_s=6 * 3600, poll_s=60):
        self.job, self.path, self.wait_s, self.poll_s = job, path, wait_s, poll_s
        self.line = f"ember release_chat.py: {job} (pid {os.getpid()}), started {now()}\n"

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, self.line.encode())
                os.close(fd)
                return self
            except FileExistsError:
                if time.time() - t0 > self.wait_s:
                    raise SystemExit(f"{self.path} still held after {self.wait_s} s: {open(self.path).read().strip()}")
                print(f"waiting for the CPU lock: {open(self.path).read().strip()}", flush=True)
                time.sleep(self.poll_s)

    def __exit__(self, *exc):
        try:
            if open(self.path).read() == self.line:
                os.remove(self.path)
        except FileNotFoundError:
            pass


# ---- servers ----------------------------------------------------------------
def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


class Server:
    def __init__(self, argv, port, env=None, log=None):
        if not port_free(port):
            raise SystemExit(f"port {port} is busy")
        self.port = port
        self.log = open(log or os.devnull, "w")
        self.p = subprocess.Popen(argv, env={**os.environ, **(env or {})}, stdout=self.log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL)

    def __enter__(self):
        for _ in range(600):
            if self.p.poll() is not None:
                raise SystemExit(f"server on {self.port} exited ({self.p.returncode})")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)
                return self
            except Exception:
                time.sleep(1)
        raise SystemExit(f"server on {self.port} did not start")

    def __exit__(self, *exc):
        self.p.terminate()
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()


def ember(binary, gguf, port, ngl, env=None, log=None):
    return Server([binary, gguf, str(port), "127.0.0.1", os.path.join(REPO, "www"), str(ngl)], port, env, log)


def run(argv, env=None, cwd=None):
    print("+", " ".join(argv), flush=True)
    r = subprocess.run(argv, env={**os.environ, **(env or {})}, cwd=cwd, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-4000:])
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
    return r


# ---- comparison helpers -----------------------------------------------------
def compare(records, ref_field="mtstudio"):
    n = len(records)
    same = sum(r["ember"] == r[ref_field] for r in records)
    tok = sum(r.get("ember_tokens") == r.get("mts_tokens") for r in records)
    stop = sum(r.get("ember_stop") == r.get("mts_stop") for r in records)
    return {"n": n, "reply_identical": same, "tokens_identical": tok, "stop_identical": stop,
            "pass": n > 0 and same == tok == stop == n}


def greedy_vs(port, ref_rows, max_new=60):
    """ember's greedy replies for recorded reference rows (fields i, prompt, mtstudio, mts_tokens, mts_stop)."""
    out = []
    for r in ref_rows:
        body = {"user_input": r["prompt"], "max_new_tokens": max_new, "temperature": 1.0, "top_k": 1, "top_p": 1.0,
                "seed": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=600))
        out.append({**r, "ember": d["reply"], "ember_tokens": d["tokens"], "ember_stop": d["stop_reason"]})
    return out


def logit_rows(path):
    return [[float(x) for x in line.split()] for line in open(path) if line.strip()]


def q8_check(f32_dump, q8_dump):
    f, q = logit_rows(f32_dump), logit_rows(q8_dump)
    if len(f) != len(q) or not f:
        return {"pass": False, "error": f"{len(f)} f32 rows vs {len(q)} q8 rows"}
    lo = min(min(r) for r in f)
    hi = max(max(r) for r in f)
    worst = max(abs(a - b) for rf, rq in zip(f, q) for a, b in zip(rf, rq))
    top1 = sum(max(range(len(rf)), key=rf.__getitem__) == max(range(len(rq)), key=rq.__getitem__) for rf, rq in zip(f, q))
    frac = worst / (hi - lo)
    return {"max_abs_dlogit": worst, "max_over_range": frac, "top1": f"{top1}/{len(f)}",
            "pass": top1 == len(f) and frac <= 0.01}


# ---- stages ----------------------------------------------------------------
def stage_export(a, rc):
    spec = json.load(open(a.spec))
    name, out_dir = spec["name"], spec.get("out_dir", "")
    with CpuLock("export"):
        r = run([a.mtstudio, "export", a.spec, "--from", a.ckpt, "--formats", "gguf,gguf_q8_0_all"])
    if r.returncode != 0:
        return rc.put("export", False, error="mtstudio export failed")
    model = os.path.join(a.out, "model")
    os.makedirs(model, exist_ok=True)
    files = {}
    for kind, fn in (("f32", f"{name}.gguf"), ("q8", f"{name}_q8_0_all.gguf")):
        src = os.path.join(out_dir, fn)
        if not os.path.exists(src):
            return rc.put("export", False, error=f"{src} missing; out_dir holds {sorted(glob.glob(os.path.join(out_dir, '*.gguf')))}")
        dst = os.path.join(model, fn)
        shutil.copyfile(src, dst)
        files[kind] = {"path": dst, "sha256": sha256(dst), "bytes": os.path.getsize(dst)}
    rc.put("export", True, spec=a.spec, spec_sha256=sha256(a.spec), ckpt=a.ckpt, ckpt_sha256=sha256(a.ckpt),
           files=files, ember_commit=git_head())


def stage_gate(a, rc):
    files = rc.data["stages"]["export"]["files"]
    f32, q8 = files["f32"]["path"], files["q8"]["path"]
    work = os.path.join(a.out, "gate")
    os.makedirs(work, exist_ok=True)
    info = {}
    with CpuLock("parity gate"):
        if a.reference:
            ref = [json.loads(line) for line in open(a.reference) if line.strip()]
            info["reference"] = {"path": a.reference, "sha256": sha256(a.reference)}
        else:   # record mtstudio live, against ember CPU, with tools/chat_parity.py
            mts_env = dict(kv.split("=", 1) for kv in a.mts_env)
            with Server([a.mtstudio, "chat", a.spec, "--port", str(PORT_MTS)], PORT_MTS, mts_env,
                        os.path.join(work, "mtstudio.log")), \
                 ember(a.ember_cpu, f32, PORT_EMBER, 0, log=os.path.join(work, "ember_cpu.log")):
                rec = os.path.join(work, "live_cpu.jsonl")
                r = run([sys.executable, os.path.join(REPO, "tools", "chat_parity.py"), "40", rec],
                        env={"PARITY_EMBER_PORT": str(PORT_EMBER), "PARITY_MTS_PORT": str(PORT_MTS),
                             "PARITY_GGUF": f32, "PAPERKILN_TOOLS": os.path.join(a.paperkiln, "tools")})
                if r.returncode != 0:
                    return rc.put("gate", False, error="chat_parity.py failed")
            ref = [{k: x[k] for k in ("i", "prompt", "mtstudio", "mts_tokens", "mts_stop")}
                   for x in map(json.loads, open(rec))]
            info["reference"] = {"live": True, "mts_env": mts_env}
        backends = [("cpu", a.ember_cpu, 0)] + ([("cuda", a.ember_cuda, -1)] if a.ember_cuda else [])
        for tag, binary, ngl in backends:
            with ember(binary, f32, PORT_EMBER, ngl, {"EMBER_GEMM_TF32": "0"}, os.path.join(work, f"ember_{tag}.log")):
                recs = greedy_vs(PORT_EMBER, ref)
            with open(os.path.join(work, f"f32_{tag}.jsonl"), "w") as f:
                f.writelines(json.dumps(x) + "\n" for x in recs)
            info[f"f32_{tag}"] = compare(recs)
        dumps = {}
        for tag, g in (("f32", f32), ("q8", q8)):
            d = os.path.join(work, f"logits_{tag}.txt")
            if os.path.exists(d):
                os.remove(d)
            with ember(a.ember_cpu, g, PORT_EMBER, 0, {"EMBER_PREFIX_REUSE": "0", "TINYLLAMA_LOGITS_DUMP": d}):
                run([sys.executable, os.path.join(REPO, "tools", "quant_gate.py"), "probe", str(PORT_EMBER), "40"])
            dumps[tag] = d
        info["q8"] = q8_check(dumps["f32"], dumps["q8"])
        with ember(a.ember_cpu, f32, PORT_EMBER, 0):
            cpu_ref = greedy_vs(PORT_EMBER, ref)
        with open(os.path.join(work, "ember_cpu_replies.jsonl"), "w") as f:
            f.writelines(json.dumps({"i": x["i"], "prompt": x["prompt"], "ember": x["ember"]}) + "\n" for x in cpu_ref)
        if a.space_model == "q8":
            with ember(a.ember_cpu, q8, PORT_EMBER, 0):
                q8_ref = greedy_vs(PORT_EMBER, ref)
            with open(os.path.join(work, "ember_cpu_q8_replies.jsonl"), "w") as f:
                f.writelines(json.dumps({"i": x["i"], "prompt": x["prompt"], "ember": x["ember"]}) + "\n" for x in q8_ref)
    ok = all(info[k]["pass"] for k in info if k.startswith("f32_")) and (a.space_model == "f32" or info["q8"]["pass"])
    rc.put("gate", ok, **info)


def stage_evals(a, rc):
    spec_tag = f"release={a.spec}"
    work = os.path.join(a.out, "evals")
    os.makedirs(work, exist_ok=True)
    info = {}
    with CpuLock("evals"):
        r = run([sys.executable, os.path.join(a.paperkiln, "tools", "qa_eval.py"), "--ember-bin", a.ember_cuda or a.ember_cpu,
                 "--model", spec_tag, "--qa", os.path.join(a.paperkiln, "corpus", "eval", "simple_qa_v1.jsonl"),
                 "--out", os.path.join(work, "qa"), "--port", str(PORT_EMBER)], cwd=a.paperkiln)
        info["qa_eval"] = {"exit": r.returncode, "summary": read_json(os.path.join(work, "qa", "qa_summary.json"))}
        r = run([sys.executable, os.path.join(a.paperkiln, "tools", "chat_eval.py"), "--mtstudio", a.mtstudio,
                 "--ember-bin", a.ember_cuda or a.ember_cpu, "--model", spec_tag,
                 "--prompts", os.path.join(a.paperkiln, "corpus", "eval", "tinychat_hf_prompts_v1.jsonl"),
                 "--out", os.path.join(work, "chat"), "--port", str(PORT_EMBER)], cwd=a.paperkiln)
        info["chat_eval"] = {"exit": r.returncode, "summary": read_json(os.path.join(work, "chat", "summary.json"))}
    rc.put("evals", all(v["exit"] == 0 for v in info.values()), **info)


def stage_wasm(a, rc):
    files = rc.data["stages"]["export"]["files"]
    model = files[a.space_model]
    size = model["bytes"]
    heap = "1gb" if size < 250e6 else ("2gb" if size < 600e6 else "4gb")
    work = os.path.join(a.out, "wasm")
    if os.path.exists(work):
        shutil.rmtree(work)
    env = {"JSON_INC": a.json_inc, "EMBER_WASM_MAX_MEMORY": heap}
    with CpuLock("wasm build + parity"):
        r = run(["bash", "-c", f'source "{a.emsdk}/emsdk_env.sh" > /dev/null 2>&1; '
                               f'bash "{REPO}/deploy/wasm/build.sh" "{work}"'], env=env)
        if r.returncode != 0:
            return rc.put("wasm", False, error="deploy/wasm/build.sh failed", heap=heap)
        ref = os.path.join(a.out, "gate", "ember_cpu_q8_replies.jsonl" if a.space_model == "q8" else "ember_cpu_replies.jsonl")
        res = os.path.join(work, "parity.json")
        r = run(["bash", "-c", f'source "{a.emsdk}/emsdk_env.sh" > /dev/null 2>&1; '
                               f'node "{REPO}/tools/wasm_parity.mjs" "{work}" "{model["path"]}" "{ref}" ember.js "{res}"'],
                env={"REF_FIELD": "ember"})
    parity = read_json(res) or {}
    parity.pop("replies", None)
    rc.put("wasm", r.returncode == 0, heap=heap, parity=parity, ember_wasm_sha256=sha256(os.path.join(work, "ember.wasm")))


def stage_space(a, rc):
    model = rc.data["stages"]["export"]["files"][a.space_model]["path"]
    space = os.path.join(a.out, "space")
    r = run(["bash", os.path.join(REPO, "tools", "build_wasm_space.sh"), model, os.path.join(a.out, "wasm"), space])
    rc.put("space", r.returncode == 0, dir=space, model=os.path.basename(model),
           files=sorted(os.listdir(space)) if os.path.isdir(space) else [])


def read_json(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def git_head():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def build(a):
    os.makedirs(a.out, exist_ok=True)
    rc = Receipt(a.out)
    rc.data["args"] = {k: v for k, v in vars(a).items() if k != "func"}
    for s in a.stages.split(","):
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s}")
        if rc.done(s) and not a.redo:
            print(f"{s}: done ({rc.data['stages'][s]['at']}), skipped")
            continue
        print(f"== {s}", flush=True)
        globals()[f"stage_{s}"](a, rc)
        st = rc.data["stages"].get(s, {})
        print(f"{s}: {'ok' if st.get('ok') else 'FAILED'}", flush=True)
        if not st.get("ok"):
            return 1
    return 0


def publish(a):
    rc = Receipt(a.out)
    if a.confirm != "ember-chat":
        raise SystemExit("publish needs --confirm ember-chat (Jonathan's go for this release)")
    missing = [s for s in STAGES if not rc.done(s)]
    if missing:
        raise SystemExit(f"stages not passed: {missing}")
    from huggingface_hub import HfApi
    token = None
    with open(a.env_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not token:
        raise SystemExit(f"no HF_TOKEN line in {a.env_file}")
    api = HfApi(token=token)
    who = api.whoami()["name"]
    if who.lower() != OWNER:
        raise SystemExit(f"logged in as {who}, not {OWNER}: stopping")
    exp = rc.data["stages"]["export"]
    kind = rc.data.get("args", {}).get("space_model", "q8")   # the model the build gated and assembled
    msg = (f"model {rc.data['stages']['space']['model']} (sha256 {exp['files'][kind]['sha256'][:12]}), "
           f"ember.cpp {exp.get('ember_commit', '')[:10]}, parity gate passed")
    info = api.upload_folder(folder_path=rc.data["stages"]["space"]["dir"], repo_id=f"{who}/ember-chat",
                             repo_type="space", commit_message=msg)
    rc.put("publish", True, commit=str(getattr(info, "oid", info)), message=msg)
    print("published:", getattr(info, "oid", info))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(required=True)
    b = sub.add_parser("build")
    b.add_argument("--spec", required=True)
    b.add_argument("--ckpt", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--mtstudio", required=True)
    b.add_argument("--ember-cpu", required=True)
    b.add_argument("--ember-cuda", default="")
    b.add_argument("--paperkiln", required=True)
    b.add_argument("--reference", default="", help="recorded mtstudio greedy jsonl instead of a live mtstudio chat")
    b.add_argument("--mts-env", action="append", default=[], help="K=V for the live mtstudio chat (repeat)")
    b.add_argument("--space-model", choices=["q8", "f32"], default="q8")
    b.add_argument("--json-inc", default=os.path.expanduser("~/ember_ttft/build_cpu/_deps/nlohmann_json-src/single_include"))
    b.add_argument("--emsdk", default=os.path.expanduser("~/emsdk"))
    b.add_argument("--stages", default=",".join(STAGES))
    b.add_argument("--redo", action="store_true")
    b.set_defaults(func=build)
    p = sub.add_parser("publish")
    p.add_argument("--out", required=True)
    p.add_argument("--env-file", required=True)
    p.add_argument("--confirm", default="")
    p.set_defaults(func=publish)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
