"""Shared paths and server helpers for the chat template gates (run in WSL, under the CPU budget)."""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
HERE = os.path.dirname(os.path.abspath(__file__))
PK = ROOT + "/microtorch"
NEW = os.path.expanduser("~/ember_wt/build_cpu")          # this change
BASE = os.path.expanduser("~/ember_69d7ca2/build")        # ember main 69d7ca2, the pre-change engine
MTS = os.path.expanduser("~/pk_cpu_ct/build/mtstudio")     # paperkiln HEAD (build_mtstudio.sh)
CHECK = NEW + "/chat_template_check"
WORK = os.path.expanduser("~/ct24")                       # ext4 scratch (copies, logs)
A = "/mnt/c/ml_artifacts"
SFT = A + "/chat_base/sftfinal/chat_sftfinal_v3.gguf"
FROZEN = A + "/chat_base/sft/tinychat_hf_prompts_v1.jsonl"   # the frozen parity prompts (first 40)
CHAT_PROMPTS = PK + "/corpus/eval/chat_prompts_v1.jsonl"    # the third-party gate's 200 test prompts
os.makedirs(WORK + "/logs", exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path, n=None):
    out = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    return out[:n] if n else out


def health(port):
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2))
    except Exception:
        return None


def post(port, body, path="/chat"):
    """(status, json) for a POST; HTTP errors return their status and body."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=1800))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw}


def start(argv, port, log, env=None):
    if health(port) is not None:
        sys.exit(f"port {port} already answers /health; refusing to share it")
    # cwd: ember's logger writes debugging.log into the working directory (ext4 scratch, not OneDrive).
    p = subprocess.Popen(argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env={**os.environ, **(env or {})}, cwd=WORK)
    for _ in range(1800):
        if p.poll() is not None:
            sys.exit(f"{argv[0]} on {port} exited {p.returncode}; see {log}")
        h = health(port)
        if h is not None:
            return p, h
        time.sleep(1)
    p.kill()
    sys.exit(f"{argv[0]} on {port} did not come up")


def ember(binary_dir, gguf, port, tag, env=None):
    return start([binary_dir + "/tinyllama_server", gguf, str(port), "127.0.0.1", "-", "0"], port,
                 f"{WORK}/logs/{tag}.log", env)


def stop(p):
    p.terminate()
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()


def greedy(body, max_new=60):
    """The parity request (tools/chat_parity.py): greedy in both samplers, with the exact ids."""
    return {**body, "max_new_tokens": max_new, "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 0,
            "return_ids": True}


def git_rev(path):
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
