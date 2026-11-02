"""PARTS4 route matrix: API exclusions and the graph-off fallback, traced, for all three E10 arms.

  python3 parts4_route_matrix.py run BIN GGUF OUT
  python3 parts4_route_matrix.py analyze OUT

decode_pick_fallback.py's requests (its prompts, seed 13, 16 new tokens) under five configurations: the
registered greedy request, repeat penalty, top_k 4, temperature 0 and 0.049; each served by host, PARTS0 and
PARTS4 servers (EMBER_DETERMINISTIC=1) with the decode graph on and off, every server traced by
EMBER_ROUTE_TRACE (decode_route_trace.h). Analyze requires, per graph mode, identical ids across the arms, and
per sampled position the route each configuration must take: exclusions named, graph off reported as the
per-launch fallback, and compact records checked against their replay's row. Untimed; changes no default.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parts4_gate as gate  # noqa: E402

PROMPTS = ["Repeat the word blue three times.", "What is two plus two?", "Tell me about a garden."]
CONFIGS = [{}, {"repeat_penalty": 1.2, "repeat_last_n": 64}, {"top_k": 4}, {"temperature": 0.0},
           {"temperature": 0.049}]
# The exclusions each configuration adds (decode_route_trace.h names), and whether a compact pick applies.
EXCLUDED = ["", "repeat_penalty", "top_k", "", ""]
ARMS = ("host", "parts0", "parts4")
PORT = 8098


def request(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def legs():
    return [(graph, arm) for graph in (1, 0) for arm in ARMS]


def leg_name(graph, arm):
    return f"graph{graph}_{arm}"


def leg_env(graph, arm, trace):
    env = dict(e.split("=", 1) for e in gate.ARMS[arm])
    env.update(EMBER_DECODE_GRAPH=str(graph), EMBER_DETERMINISTIC="1", EMBER_SPEC="0", EMBER_GEMM_TF32="0",
               EMBER_ROUTE_TRACE=str(trace))
    return env


def run(binary, model, out):
    out = Path(out)
    out.mkdir(exist_ok=False)
    summary = {"binary_sha256": gate.sha(binary), "model_sha256": gate.sha(model), "legs": []}
    for graph, arm in legs():
        name = leg_name(graph, arm)
        work = out / name
        (work / "www").mkdir(parents=True)
        trace = out / (name + ".trace.jsonl")
        env = dict(os.environ)
        for k in gate.CLEARED:
            env.pop(k, None)
        env.update(leg_env(graph, arm, trace))
        with (work / "server.log").open("w") as log:
            process = subprocess.Popen([binary, model, str(PORT), "127.0.0.1", str(work / "www"), "999"],
                                       cwd=work, env=env, stdout=subprocess.DEVNULL, stderr=log)
            try:
                for _ in range(300):
                    try:
                        health = request(f"http://127.0.0.1:{PORT}/health")
                        break
                    except OSError:
                        if process.poll() is not None:
                            raise RuntimeError(name + ": server exited")
                        time.sleep(1)
                else:
                    raise RuntimeError(name + ": server unavailable")
                if health.get("model") != model or not health.get("deterministic"):
                    raise ValueError(name + ": wrong model or mode")
                rows = []
                for i, config in enumerate(CONFIGS):
                    for prompt in PROMPTS:
                        body = {"user_input": prompt, "temperature": 1, "top_k": 1, "top_p": 1, "seed": 13,
                                "return_ids": True, "max_new_tokens": 16, **config}
                        r = request(f"http://127.0.0.1:{PORT}/chat", body)
                        if not r.get("prompt_ids") or not isinstance(r.get("ids"), list):
                            raise ValueError(name + ": missing ids")
                        rows.append({"config": i, "prompt": prompt,
                                     **{k: r.get(k) for k in ("reply", "ids", "prompt_ids", "stop_id", "tokens")}})
                (out / (name + ".json")).write_text(json.dumps(
                    {"health": health, "rows": rows, "env": leg_env(graph, arm, trace)}, indent=1) + "\n")
            finally:
                process.terminate()
                try:
                    process.wait(20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        summary["legs"].append(name)
    (out / "run.json").write_text(json.dumps(summary, indent=1) + "\n")
    return analyze(out)


def expected(graph, arm, config):
    """(eligible, exclusions, route set) for one sampled decode position."""
    excl = ",".join(x for x in (("pick_off" if arm == "host" else ""), EXCLUDED[config]) if x)
    eligible = excl == ""
    if not eligible:
        return False, excl, {"host_ineligible"}
    if graph == 0:
        return True, "", {"per_launch_fallback"}
    return True, "", {"compact", "host_after_near_tie"}


def check_position(ev, graph, arm, config, temperature):
    eligible, excl, routes = expected(graph, arm, config)
    bad = []
    if ev.get("eligible") is not eligible or ev.get("exclusions") != excl or ev.get("route") not in routes:
        bad.append(f"route {ev.get('route')} exclusions '{ev.get('exclusions')}' (wanted {sorted(routes)} '{excl}')")
    if not ev.get("eligibility_consistent"):
        bad.append("eligibility mask disagrees with want_device_pick")
    want_graph = "replay" if graph else "off"
    if ev.get("graph") != want_graph:
        bad.append(f"graph {ev.get('graph')} (wanted {want_graph})")
    if graph and eligible:
        if not ev.get("graph_has_pick") or ev.get("graph_parts") != gate.PARTS[arm]:
            bad.append("replayed graph lacks the arm's selector")
        check = ev.get("check") or {}
        if not (check.get("record_written") and check.get("record_value")):
            bad.append("record not written by this replay or differs from its row")
        if gate.PARTS[arm] == 4 and (check.get("parts_written") != 4 or check.get("parts_value") != 4
                                    or not check.get("merge_value")):
            bad.append("partition records not written by this replay or differ from their oracles")
        if ev.get("route") == "compact" and ev.get("decision") != ev.get("token"):
            bad.append("compact token is not the record's decision")
        if temperature < 0.05 and ev.get("route") != "compact":
            bad.append("below temperature 0.05 the record always decides")
    elif ev.get("record_returned") or ev.get("picked_on_device"):
        bad.append("record consumed where no compact pick applies")
    if ev.get("captured"):
        inventory = ev.get("inventory") or {}
        want = gate.expected_inventory(arm if eligible else "host")
        if any(inventory.get(k) != v for k, v in want.items()):
            bad.append("captured graph's selector nodes differ: " + json.dumps(inventory))
    return bad


def analyze(out):
    out = Path(out)
    records, problems, counts = {}, [], {}
    for graph, arm in legs():
        name = leg_name(graph, arm)
        leg = json.loads((out / (name + ".json")).read_text())
        if leg["env"] != leg_env(graph, arm, out / (name + ".trace.jsonl")) or not leg["health"].get("deterministic"):
            raise ValueError(name + ": environment or mode differs")
        rows = leg["rows"]
        if [(r["config"], r["prompt"]) for r in rows] != [(i, p) for i in range(len(CONFIGS)) for p in PROMPTS]:
            raise ValueError(name + ": incomplete request inventory")
        records[(graph, arm)] = rows
        traced = gate.read_trace(out / (name + ".trace.jsonl"))
        if len(traced) != len(rows):
            raise ValueError(f"{name}: {len(traced)} traced requests for {len(rows)} requests")
        c = counts.setdefault(name, {"positions": 0, "routes": {}})
        for row, t in zip(rows, traced):
            config = row["config"]
            temperature = CONFIGS[config].get("temperature", 1.0)
            q = t["request"]
            where = f"{name} config {config} '{row['prompt']}'"
            if q.get("prompt_ids") != row["prompt_ids"] or not gate.generated_matches(
                    t["end"].get("generated"), row["ids"], row["stop_id"]):
                problems.append(where + ": trace differs from the response")
            if q.get("spec") or q.get("cpu_offload_layers") != 0 or q.get("ir") or not q.get("deterministic"):
                problems.append(where + ": not a plain all-GPU deterministic request")
            for ev in t["tokens"]:
                if ev.get("phase") != "decode":
                    continue
                c["positions"] += 1
                c["routes"][ev.get("route")] = c["routes"].get(ev.get("route"), 0) + 1
                problems.extend(f"{where} pos {ev.get('pos')}: {b}"
                                for b in check_position(ev, graph, arm, config, temperature))
    for graph in (0, 1):
        base = records[(graph, "host")]
        for arm in ("parts0", "parts4"):
            for x, y in zip(base, records[(graph, arm)]):
                if x != y:
                    problems.append(f"graph{graph} {arm} config {x['config']} '{x['prompt']}': response differs from host")
    for name, c in counts.items():
        if c["positions"] == 0:
            problems.append(name + ": no sampled decode positions")
    for arm in ("parts0", "parts4"):
        if counts[leg_name(1, arm)]["routes"].get("compact", 0) == 0:
            problems.append(arm + ": no compact route with the graph on")
    return {"pass": not problems, "counts": counts, "problems": problems[:50], "problem_count": len(problems),
            "scope": "exclusion and graph-off routes for the three arms; IR and CPU offload are not executed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("run")
    for name in ("binary", "model", "out"):
        p.add_argument(name)
    p = sub.add_parser("analyze")
    p.add_argument("out")
    args = parser.parse_args()
    if args.action == "run":
        result = run(str(Path(args.binary).resolve()), str(Path(args.model).resolve()), args.out)
    else:
        result = analyze(args.out)
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["pass"] else 1)
