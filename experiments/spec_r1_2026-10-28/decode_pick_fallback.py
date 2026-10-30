"""Direct API off/on gate for paths which cannot use compact greedy selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def request(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


PROMPTS = ["Repeat the word blue three times.", "What is two plus two?", "Tell me about a garden."]


def compare(records):
    required = {(graph, pick) for graph in (0, 1) for pick in (0, 1)}
    if set(records) != required:
        raise ValueError("missing fallback legs")
    for graph in (0, 1):
        a, b = records[(graph, 0)], records[(graph, 1)]
        inventory = [(i, prompt) for i in range(4) for prompt in PROMPTS]
        if ([(r.get("config"), r.get("prompt")) for r in a] != inventory
                or [(r.get("config"), r.get("prompt")) for r in b] != inventory):
            raise ValueError("incomplete fallback requests")
        for x, y in zip(a, b):
            if x != y:
                raise ValueError(f"graph={graph}: fallback ids/stop/prompt differ")
    return {"pass": True, "requests_per_leg": len(records[(0, 0)]), "legs": 4}


def run(binary, model, out):
    out = Path(out)
    out.mkdir(exist_ok=False)
    prompts = PROMPTS
    configs = [{"repeat_penalty": 1.2, "repeat_last_n": 64},
               {"top_k": 4}, {"temperature": 0.0}, {"temperature": 0.049}]
    collected = {}
    for graph in (0, 1):
        for pick in (0, 1):
            name = f"graph{graph}_pick{pick}"
            work = out / name
            (work / "www").mkdir(parents=True)
            env = dict(os.environ, EMBER_DECODE_GRAPH=str(graph), EMBER_DECODE_PICK=str(pick),
                       EMBER_DETERMINISTIC="1", EMBER_SPEC="0", EMBER_GEMM_TF32="0")
            for k in ("TINYLLAMA_LOGITS_DUMP", "TINYLLAMA_TOPK_DEBUG", "EMBER_DECODE_TIMING"):
                env.pop(k, None)
            with (work / "server.log").open("w") as log:
                process = subprocess.Popen([binary, model, "8097", "127.0.0.1", str(work / "www"), "999"],
                                           cwd=work, env=env, stdout=subprocess.DEVNULL, stderr=log)
                try:
                    for _ in range(120):
                        try:
                            health = request("http://127.0.0.1:8097/health")
                            break
                        except OSError:
                            if process.poll() is not None:
                                raise RuntimeError("fallback server exited")
                            time.sleep(1)
                    else:
                        raise RuntimeError("fallback server unavailable")
                    if health.get("model") != model or not health.get("deterministic"):
                        raise ValueError("wrong fallback model or mode")
                    rows = []
                    for i, config in enumerate(configs):
                        for prompt in prompts:
                            body = {"user_input": prompt, "temperature": 1, "top_k": 1,
                                    "top_p": 1, "seed": 13, "return_ids": True, "max_new_tokens": 16, **config}
                            r = request("http://127.0.0.1:8097/chat", body)
                            if not r.get("prompt_ids") or not isinstance(r.get("ids"), list):
                                raise ValueError("missing fallback ids")
                            rows.append({"config": i, "prompt": prompt, **{k: r[k] for k in
                                         ("reply", "ids", "prompt_ids", "stop_id", "tokens")}})
                    collected[(graph, pick)] = rows
                    (out / (name + ".json")).write_text(json.dumps({"health": health, "rows": rows, "env": {k: env[k] for k in
                        ("EMBER_DECODE_GRAPH", "EMBER_DECODE_PICK", "EMBER_DETERMINISTIC", "EMBER_SPEC", "EMBER_GEMM_TF32")}}, indent=1))
                finally:
                    process.terminate()
                    try:
                        process.wait(20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    result = compare(collected)
    result["binary_sha256"] = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    result["model_sha256"] = hashlib.sha256(Path(model).read_bytes()).hexdigest()
    (out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "out"):
        parser.add_argument(name)
    args = parser.parse_args()
    print(json.dumps(run(str(Path(args.binary).resolve()), str(Path(args.model).resolve()), args.out)))
