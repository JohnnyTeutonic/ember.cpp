"""PARTS4 route matrix: API exclusions and the graph-off fallback, traced, for all three E10 arms.

  python3 parts4_route_matrix.py run BIN GGUF OUT
  python3 parts4_route_matrix.py analyze OUT

decode_pick_fallback.py's requests (its prompts, seed 13, 16 new tokens) under five configurations: the
registered greedy request, repeat penalty, top_k 4, temperature 0 and 0.049; each served by host, PARTS0 and
PARTS4 servers (EMBER_DETERMINISTIC=1) with the decode graph on and off, every server traced by
EMBER_ROUTE_TRACE (decode_route_trace.h). Analyze binds the receipt to one binary and model, then requires per
request the configuration it was sent, a complete trace (prefill token plus one event per decode position),
capture/replay continuity, and per position the route its configuration requires: exclusions named, graph off
reported as the per-launch fallback, and compact records checked against their replay's row and recomputed
decision. Responses must agree across arms in each graph mode; differences are reported separately, since a
PARTS4 difference is a correctness failure. Untimed; changes no default.
"""
import argparse
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
# The exclusion each configuration adds (decode_route_trace.h names).
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


def body(config, prompt):
    return {"user_input": prompt, "temperature": 1, "top_k": 1, "top_p": 1, "seed": 13, "return_ids": True,
            "max_new_tokens": 16, **CONFIGS[config]}


def run(binary, model, out):
    out = Path(out)
    out.mkdir(exist_ok=False)
    receipt = {"binary": binary, "model": model, "binary_sha256": gate.sha(binary), "model_sha256": gate.sha(model),
               "legs": [leg_name(g, a) for g, a in legs()], "completed": []}
    for graph, arm in legs():
        name = leg_name(graph, arm)
        work = out / name
        (work / "www").mkdir(parents=True)
        trace = out / (name + ".trace.jsonl")
        env = dict(os.environ)
        for k in list(env):
            if k.startswith(("EMBER_", "TINYLLAMA_")):
                env.pop(k)
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
                for i in range(len(CONFIGS)):
                    for prompt in PROMPTS:
                        sent = body(i, prompt)
                        r = request(f"http://127.0.0.1:{PORT}/chat", sent)
                        if not r.get("prompt_ids") or not isinstance(r.get("ids"), list):
                            raise ValueError(name + ": missing ids")
                        rows.append({"config": i, "prompt": prompt, "body": sent,
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
        receipt["completed"].append(name)
        (out / "run.json").write_text(json.dumps(receipt, indent=1) + "\n")
    return analyze(out)


def expected(graph, arm, config):
    """(eligible, exclusions, allowed routes) for one sampled decode position."""
    excl = ",".join(x for x in (("pick_off" if arm == "host" else ""), EXCLUDED[config]) if x)
    if excl:
        return False, excl, {"host_ineligible"}
    if graph == 0:
        return True, "", {"per_launch_fallback"}
    return True, "", {"compact", "host_after_near_tie", "host_after_nonfinite"}


def request_problems(q, graph, arm, config):
    """The traced request is the configuration the matrix sent, on the leg's server."""
    b = body(config, "")
    want_env = {k: None for k in gate.TRACE_ENV_KEYS}
    want_env.update({k: v for k, v in leg_env(graph, arm, "").items() if k in want_env})
    bad = []
    if q.get("env") != want_env:
        bad.append("server environment differs from the leg")
    penalty = b.get("repeat_penalty")
    # The trace serializes the server's float32 values (0.049 arrives as 0.04899999871850014).
    if (q.get("temperature") != gate.f32(gate_bits(b["temperature"])) or q.get("top_k") != b["top_k"]
            or q.get("top_p") != 1.0
            or q.get("repeat_on") is not (penalty is not None) or q.get("max_new_tokens") != b["max_new_tokens"]
            or (penalty is not None and (q.get("repeat_penalty") != gate.f32(gate_bits(penalty))
                                         or q.get("repeat_last_n") != b["repeat_last_n"]))
            or q.get("debug_rows") is not False or q.get("forced") is not False or q.get("spec") is not False
            or q.get("ir") is not False or q.get("cpu_offload_layers") != 0 or q.get("deterministic") is not True):
        bad.append("traced request is not the configuration sent")
    return bad


def gate_bits(x):
    """The float32 bits of a JSON number, as the server stores it."""
    import struct
    return struct.unpack("<I", struct.pack("<f", x))[0]


def check_position(ev, graph, arm, config, temperature):
    """(proof problems, demonstrated-wrong selector output) for one sampled decode position."""
    eligible, excl, routes = expected(graph, arm, config)
    bad, wrong = [], []
    if ev.get("eligible") is not eligible or ev.get("exclusions") != excl or ev.get("route") not in routes:
        bad.append(f"route {ev.get('route')} exclusions '{ev.get('exclusions')}' (wanted {sorted(routes)} '{excl}')")
    if not ev.get("eligibility_consistent") or ev.get("forced"):
        bad.append("eligibility mask disagrees with want_device_pick, or a forced position")
    want_graph = "replay" if graph else "off"
    if ev.get("graph") != want_graph:
        bad.append(f"graph {ev.get('graph')} (wanted {want_graph})")
    if graph and eligible:
        if not ev.get("graph_has_pick") or ev.get("graph_parts") != gate.PARTS[arm]:
            bad.append("replayed graph lacks the arm's selector")
        check = ev.get("check") or {}
        if not check.get("record_written"):
            bad.append("record not written by this replay")
        elif not check.get("record_value"):
            wrong.append("written record differs from its replay's row")
        if gate.PARTS[arm] == 4:
            if check.get("parts_written") != 4:
                bad.append("partition records not written by this replay")
            elif check.get("parts_value") != 4 or not check.get("merge_value"):
                wrong.append("written partition records differ from their oracles or do not merge to the record")
        try:
            decided, why = gate.recompute(ev["record"], temperature)
        except (KeyError, TypeError):
            return bad + ["record missing from the event"], wrong
        want = {"none": "compact", "near_tie": "host_after_near_tie", "nonfinite": "host_after_nonfinite"}[why]
        if (ev.get("decision"), ev.get("undecided")) != (decided, why) or ev.get("route") != want \
                or bool(ev.get("picked_on_device")) is not (want == "compact"):
            bad.append(f"route {ev.get('route')} or decision is not the record's ({want}, {decided})")
        elif want == "compact" and ev.get("token") != decided:
            wrong.append("compact token is not the record's decision")
        if temperature < 0.05 and why != "none":
            bad.append("below temperature 0.05 the record always decides")
    elif ev.get("record_returned") or ev.get("picked_on_device"):
        bad.append("record consumed where no compact pick applies")
    if ev.get("captured"):
        bad.extend(gate.inventory_problems(ev.get("inventory") or {}, arm if (graph and eligible) else "host"))
    return bad, wrong


def analyze(out, expect=None):
    """expect: {"binary": SHA, "model": SHA} from the gate's plan; the matrix must come from the same ones."""
    out = Path(out)
    receipt = json.loads((out / "run.json").read_text())
    names = [leg_name(g, a) for g, a in legs()]
    if receipt.get("legs") != names or receipt.get("completed") != names:
        raise ValueError("matrix leg inventory incomplete or altered")
    if expect and (receipt.get("binary_sha256") != expect["binary"] or receipt.get("model_sha256") != expect["model"]):
        raise ValueError("matrix receipt is from another binary or model")
    records, problems, differences, candidate_wrong, counts = {}, [], [], [], {}
    # The responses first, from the receipts alone, each leg on its own: a PARTS4 difference from host is reported
    # whatever the traces or the other legs hold.
    for graph, arm in legs():
        name = leg_name(graph, arm)
        try:
            leg = json.loads((out / (name + ".json")).read_text())
            # The trace is read beside the leg (an archive replays wherever it is unpacked); its recorded path must
            # name that file.
            env = dict(leg["env"])
            traced = Path(env.pop("EMBER_ROUTE_TRACE", "")).name
            want = leg_env(graph, arm, "")
            want.pop("EMBER_ROUTE_TRACE")
            if (env != want or traced != name + ".trace.jsonl" or not leg["health"].get("deterministic")
                    or leg["health"].get("model") != receipt.get("model")):
                raise ValueError("environment, mode or model differs")
            rows = leg["rows"]
            if [(r["config"], r["prompt"], r.get("body")) for r in rows] != [
                    (i, p, body(i, p)) for i in range(len(CONFIGS)) for p in PROMPTS]:
                raise ValueError("incomplete request inventory, or a request body other than the one specified")
            records[(graph, arm)] = rows
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{name}: receipt unusable: {exc}")
    for graph in (0, 1):
        base = records.get((graph, "host"))
        for arm in ("parts0", "parts4"):
            if base is None or (graph, arm) not in records:
                continue
            for x, y in zip(base, records[(graph, arm)]):
                if x != y:
                    differences.append(f"{arm} graph{graph} config {x['config']} '{x['prompt']}': response differs from host")
    for graph, arm in legs():
        name = leg_name(graph, arm)
        c = counts.setdefault(name, {"positions": 0, "per_config": [0] * len(CONFIGS), "routes": {}, "captures": 0,
                                     "immediate_stops": 0})
        if (graph, arm) not in records:
            continue
        rows = records[(graph, arm)]
        try:
            traced_requests, error = gate.read_trace_partial(out / (name + ".trace.jsonl"))
            if error:
                problems.append(name + ": " + error)
            elif len(traced_requests) != len(rows):
                problems.append(f"{name}: {len(traced_requests)} traced requests for {len(rows)} requests")
            continuity = gate.Continuity()
            for row, t in zip(rows, traced_requests):
                config = row["config"]
                q = t["request"]
                where = f"{name} config {config} '{row['prompt']}'"
                problems.extend(where + ": " + b for b in request_problems(q, graph, arm, config))
                if q.get("prompt_ids") != row["prompt_ids"] or not gate.generated_matches(
                        t["end"].get("generated"), row["ids"], row["stop_id"]):
                    problems.append(where + ": trace differs from the response")
                bad = gate.composition_problems(t)
                problems.extend(where + ": " + b for b in bad)
                if bad:
                    continue
                if not t["tokens"]:
                    # Only a reply that stopped on its first (prefill) token has no decode position.
                    if t["end"].get("generated") == [row["stop_id"]] and row["ids"] == []:
                        c["immediate_stops"] += 1
                    else:
                        problems.append(where + ": no decode position traced")
                for ev in t["tokens"]:
                    c["positions"] += 1
                    c["per_config"][config] += 1
                    c["routes"][ev.get("route")] = c["routes"].get(ev.get("route"), 0) + 1
                    c["captures"] += bool(ev.get("captured"))
                    at = f"{where} pos {ev.get('pos')}: "
                    if graph:
                        problems.extend(at + b for b in continuity.step(ev))
                    bad, wrong = check_position(ev, graph, arm, config, q["temperature"])
                    problems.extend(at + b for b in bad)
                    (candidate_wrong if arm == "parts4" else problems).extend(at + w for w in wrong)
        except (OSError, ValueError, KeyError, TypeError, gate.Inconclusive) as exc:
            problems.append(name + ": trace unusable: " + repr(exc))
    for name, c in counts.items():
        if any(n == 0 for n in c["per_config"]):
            problems.append(name + ": a configuration has no sampled decode position")
    for arm in ("parts0", "parts4"):
        if counts[leg_name(1, arm)]["routes"].get("compact", 0) == 0:
            problems.append(arm + ": no compact route with the graph on")
    return {"pass": not (problems or differences or candidate_wrong), "counts": counts, "problems": problems[:50],
            "problem_count": len(problems), "response_differences": differences,
            "candidate_wrong": candidate_wrong[:50], "candidate_wrong_count": len(candidate_wrong),
            "binary_sha256": receipt.get("binary_sha256"), "model_sha256": receipt.get("model_sha256"),
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
    try:
        if args.action == "run":
            result = run(str(Path(args.binary).resolve()), str(Path(args.model).resolve()), args.out)
        else:
            result = analyze(args.out)
    except Exception as exc:  # an execution failure, not a failed route check: exit 3
        print(json.dumps({"pass": False, "error": repr(exc)}))
        sys.exit(3)
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["pass"] else 1)
