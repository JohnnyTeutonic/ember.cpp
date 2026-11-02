"""Plan and check the E10 PARTS4 deterministic-greedy gate; this tool never launches a GPU job.

  python3 parts4_gate.py plan BIN GGUF PROMPTS.jsonl OUTPUT_DIR [--port 8080]
  python3 parts4_gate.py analyze OUTPUT_DIR [--matrix MATRIX_DIR]

Protocol: ../parts4_full_model_protocol_20261102.json (public at 41b6cd5). The requests, the client and every
validity predicate of a leg are the registered R1 ones: dr_e2e.py run, and decode_pick_gate.load_run / ids /
speed, imported unchanged and hash-bound in the plan. Three arms, all under EMBER_DETERMINISTIC=1:
host (compact picking off, PARTS0), parts0 (on, PARTS0), parts4 (on, PARTS4).

Legs, in order: five untimed route-proof legs with EMBER_ROUTE_TRACE (decode_route_trace.h): one per arm, PARTS4
with fault injection (every record handed to the API undecidable), and PARTS4 with row dumps; then the protocol's
six row legs (dumps, no trace) and its two ordered timing triples (no dumps, no trace). A route leg's server
traces every request; analyze aligns the trace with the client's records, checks every sampled position's route,
and checks each replay's record (and PARTS4's partition records) against the row the same replay produced.

Analyze is fail-closed. Verdicts: PASS (deterministic-greedy, model-specific opt-in eligibility only),
PERFORMANCE_MISS, CORRECTNESS_FAIL, INCONCLUSIVE (operational, control, inventory or route-evidence failure).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
R1 = HERE.parent
CLIENT = R1.parent / "draft_2026-10-27" / "dr_e2e.py"
REGISTERED_GATE = R1 / "decode_pick_gate.py"
PROTOCOL = R1 / "parts4_full_model_protocol_20261102.json"
MATRIX = HERE / "parts4_route_matrix.py"
SCHEMA = "ember.e10.parts4-gate.v1"
THRESHOLDS = {"parts4_over_host": 1.15, "parts4_over_parts0": 1.0}
# The registered environment of decode_pick_gate's det_ legs, plus the explicit partition count.
BASE_ENV = ["EMBER_SPEC=0", "EMBER_DECODE_GRAPH=1", "EMBER_DECODE_OPT=1", "EMBER_DECODE_LINEAR=cublas",
            "EMBER_DETERMINISTIC=1"]
ARMS = {"host": ["EMBER_DECODE_PICK=0", "EMBER_DECODE_PICK_PARTS=0"],
        "parts0": ["EMBER_DECODE_PICK=1", "EMBER_DECODE_PICK_PARTS=0"],
        "parts4": ["EMBER_DECODE_PICK=1", "EMBER_DECODE_PICK_PARTS=4"]}
PARTS = {"host": 0, "parts0": 0, "parts4": 4}
# name, arm, kind, dump, trace, forced. Names start with det_: the registered load_run ties that prefix to the
# server's reported deterministic mode.
LEGS = (("det_route_host", "host", "route", False, True, False),
        ("det_route_parts0", "parts0", "route", False, True, False),
        ("det_route_parts4", "parts4", "route", False, True, False),
        ("det_route_parts4_forced", "parts4", "route", False, True, True),
        ("det_route_parts4_rows", "parts4", "route", True, True, False),
        ("det_rows_host_a", "host", "rows", True, False, False),
        ("det_rows_parts0_a", "parts0", "rows", True, False, False),
        ("det_rows_parts4_a", "parts4", "rows", True, False, False),
        ("det_rows_parts4_b", "parts4", "rows", True, False, False),
        ("det_rows_parts0_b", "parts0", "rows", True, False, False),
        ("det_rows_host_b", "host", "rows", True, False, False),
        ("det_time_host_a", "host", "timing", False, False, False),
        ("det_time_parts0_a", "parts0", "timing", False, False, False),
        ("det_time_parts4_a", "parts4", "timing", False, False, False),
        ("det_time_parts4_b", "parts4", "timing", False, False, False),
        ("det_time_parts0_b", "parts0", "timing", False, False, False),
        ("det_time_host_b", "host", "timing", False, False, False))
PAIRS = ({"host": "det_time_host_a", "parts0": "det_time_parts0_a", "parts4": "det_time_parts4_a"},
         {"host": "det_time_host_b", "parts0": "det_time_parts0_b", "parts4": "det_time_parts4_b"})
# The routing variables a trace's request event reports from the server's environment.
TRACE_ENV_KEYS = ("EMBER_DECODE_PICK", "EMBER_DECODE_PICK_PARTS", "EMBER_DECODE_GRAPH", "EMBER_DECODE_OPT",
                  "EMBER_DETERMINISTIC", "EMBER_SPEC")
CLEARED = ("TINYLLAMA_LOGITS_DUMP", "TINYLLAMA_TOPK_DEBUG", "EMBER_DECODE_TIMING", "EMBER_ROUTE_TRACE",
           "EMBER_ROUTE_TRACE_FORCE")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def registered_gate():
    spec = importlib.util.spec_from_file_location("decode_pick_gate_registered", REGISTERED_GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


def frozen_files():
    return {"dr_e2e.py": CLIENT, "decode_pick_gate.py": REGISTERED_GATE, "protocol.json": PROTOCOL,
            "parts4_gate.py": Path(__file__).resolve(), "parts4_route_matrix.py": MATRIX}


def plan(binary, model, prompts, output, port=8080):
    gate = registered_gate()
    paths = {k: str(Path(v).resolve()) for k, v in (("binary", binary), ("model", model), ("prompts", prompts))}
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "inputs": paths, "sha256": {k: sha(v) for k, v in paths.items()},
                "frozen_sha256": {k: sha(v) for k, v in frozen_files().items()},
                "request_keys": gate.request_keys(paths["prompts"]), "thresholds": THRESHOLDS, "runs": []}
    for name, arm, kind, dump, trace, forced in LEGS:
        env = BASE_ENV + ARMS[arm]
        if trace:
            env = env + ["EMBER_ROUTE_TRACE=" + str(out / (name + ".trace.jsonl"))]
        if forced:
            env = env + ["EMBER_ROUTE_TRACE_FORCE=undecidable"]
        cmd = ["env"] + [x for k in CLEARED for x in ("-u", k)] + [
            "python3", str(CLIENT), "run", paths["binary"], paths["model"], "999", str(port), paths["prompts"],
            str(out / (name + ".jsonl"))]
        if dump:
            cmd.append("--dump")
        cmd.extend(env)
        manifest["runs"].append({"name": name, "arm": arm, "kind": kind, "dump": dump, "trace": trace,
                                 "forced": forced, "env": env, "command": cmd})
    (out / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for run in manifest["runs"]:
        print(shlex.join(run["command"]))
    return manifest


class Inconclusive(Exception):
    """Operational, control, inventory or evidence failure: no scientific verdict."""


def read_trace(path):
    requests, current = [], None
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        ev = json.loads(line)
        kind = ev.get("ev")
        if kind == "request":
            if current is not None:
                raise Inconclusive(f"{path}:{n}: request without end")
            current = {"request": ev, "tokens": []}
        elif kind in ("token", "end"):
            if current is None or ev.get("request") != current["request"]["request"]:
                raise Inconclusive(f"{path}:{n}: event outside its request")
            if kind == "token":
                current["tokens"].append(ev)
            else:
                current["end"] = ev
                requests.append(current)
                current = None
        else:
            raise Inconclusive(f"{path}:{n}: unknown event")
    if current is not None:
        raise Inconclusive(f"{path}: truncated final request")
    if [r["request"]["request"] for r in requests] != list(range(1, len(requests) + 1)):
        raise Inconclusive(f"{path}: request sequence is not 1..n")
    return requests


def generated_matches(generated, ids, stop_id):
    """The trace's generated ids are the reply's ids, plus the stop id when generation stopped on one."""
    if generated[:len(ids)] != ids:
        return False
    return len(generated) == len(ids) or (len(generated) == len(ids) + 1 and generated[-1] == stop_id)


def expected_inventory(arm):
    if arm == "host":
        return {"select": 0, "partition": 0, "merge": 0}
    if arm == "parts0":
        return {"select": 1, "partition": 0, "merge": 0}
    return {"select": 0, "partition": 1, "partition_grid": [1, 4, 1], "merge": 1, "merge_grid": 1}


def check_position(ev, arm, dump, forced, counts):
    """One sampled decode position of a registered greedy request; returns the problems found."""
    bad = []
    parts = PARTS[arm]
    if not ev.get("eligibility_consistent"):
        bad.append("eligibility mask disagrees with want_device_pick")
    if ev.get("graph") != "replay":
        bad.append("decode position not served by a graph replay: " + str(ev.get("graph")))
    if ev.get("captured"):
        inventory = ev.get("inventory") or {}
        want = expected_inventory(arm)
        if any(inventory.get(k) != v for k, v in want.items()) or not inventory.get("kernel", 0) > 0:
            bad.append("captured graph's selector nodes differ from the arm: " + json.dumps(inventory))
        counts["captures"] += 1
    else:
        counts["replays_not_capturing"] += 1
    route = ev.get("route")
    counts["routes"][route] = counts["routes"].get(route, 0) + 1
    if arm == "host":
        if (ev.get("eligible") or ev.get("exclusions") != "pick_off" or route != "host_ineligible"
                or ev.get("graph_has_pick") or ev.get("graph_parts") != 0 or ev.get("record_returned")
                or ev.get("picked_on_device")):
            bad.append("host arm position took a compact-pick route")
        return bad
    if not ev.get("eligible") or ev.get("exclusions") != "" or not ev.get("record_returned"):
        bad.append("pick arm position not eligible or no record returned (silent fallback)")
        return bad
    if not ev.get("graph_has_pick") or ev.get("graph_parts") != parts:
        bad.append(f"replayed graph has pick={ev.get('graph_has_pick')} parts={ev.get('graph_parts')}, arm wants {parts}")
    check = ev.get("check") or {}
    if not check.get("record_written"):
        bad.append("record still the sentinel after replay: selector did not run on this replay")
    if not check.get("record_value"):
        bad.append("record differs from the whole-row oracle of the same replay")
    counts["record_bits_equal"] += bool(check.get("record_bits"))
    if parts == 4:
        if check.get("parts_written") != 4 or check.get("parts_value") != 4 or not check.get("merge_value"):
            bad.append("partition records not all written by this replay or differ from partition oracles")
        if len(ev.get("scratch") or []) != 4:
            bad.append("partition records missing from the event")
    elif ev.get("scratch") is not None:
        bad.append("PARTS0 replay reports partition records")
    if ev.get("undecided") == "nonfinite":
        counts["nonfinite_records"] += 1
    if dump:
        if route != "host_after_debug_rows" or ev.get("picked_on_device"):
            bad.append("debug-row leg consumed a compact record")
    elif forced:
        if route != "host_after_forced_undecidable" or ev.get("picked_on_device"):
            bad.append("forced leg consumed a compact record")
        if ev.get("undecided") == "none":
            counts["forced_decidable"] += 1
            if ev.get("decision") != ev.get("token"):
                bad.append("compact decision differs from the host sampler's token")
            else:
                counts["forced_decision_agrees"] += 1
        elif ev.get("undecided") == "near_tie":
            counts["near_ties"] += 1
    else:
        if route == "compact":
            if not ev.get("picked_on_device") or ev.get("decision") != ev.get("token"):
                bad.append("compact route without device pick, or token is not the record's decision")
            counts["compact"] += 1
        elif route == "host_after_near_tie":
            counts["near_ties"] += 1
            if ev.get("picked_on_device"):
                bad.append("near tie consumed on device")
        elif route == "host_after_nonfinite":
            if ev.get("picked_on_device"):
                bad.append("nonfinite record consumed on device")
        else:
            bad.append("unexpected route " + str(route))
    return bad


def check_route(run, records, manifest):
    """The route evidence of one route leg against its client records."""
    path = Path(dict(e.split("=", 1) for e in run["env"])["EMBER_ROUTE_TRACE"])
    if not path.is_file():
        raise Inconclusive(run["name"] + ": no route trace")
    requests = read_trace(path)
    if len(requests) != 2 * len(records):
        raise Inconclusive(f"{run['name']}: {len(requests)} traced requests for {len(records)} records (warm + recorded)")
    arm, dump, forced = run["arm"], run["dump"], run["forced"]
    leg_env = dict(e.split("=", 1) for e in BASE_ENV + ARMS[arm])
    want_env = {k: leg_env.get(k) for k in TRACE_ENV_KEYS}
    counts = {"requests": len(requests), "prefill_host_tokens": 0, "decode_positions": 0, "captures": 0,
              "replays_not_capturing": 0, "compact": 0, "near_ties": 0, "nonfinite_records": 0,
              "forced_decidable": 0, "forced_decision_agrees": 0, "record_bits_equal": 0, "routes": {}}
    problems = []
    for k, rec in enumerate(records):
        for t in (requests[2 * k], requests[2 * k + 1]):  # the warm pass, then the recorded request
            q = t["request"]
            where = f"{run['name']} record {k} request {q['request']}"
            if q.get("env") != want_env:
                problems.append(where + ": server environment differs from the arm")
            if (q.get("spec") or q.get("ir") or q.get("cpu_offload_layers") != 0 or q.get("deterministic") is not True
                    or q.get("debug_rows") is not dump or q.get("forced") is not forced or q.get("top_k") != 1
                    or q.get("temperature") != 1.0 or q.get("top_p") != 1.0 or q.get("repeat_on")
                    or not q.get("repetition_penalty", 1.0) <= 1.0):
                problems.append(where + ": request is not a registered plain greedy all-GPU request")
            if q.get("prompt_ids") != rec["prompt_ids"]:
                problems.append(where + ": traced prompt differs from the client's prompt ids")
            if not generated_matches(t["end"].get("generated"), rec["ids"], rec["stop_id"]):
                problems.append(where + ": traced tokens differ from the client's ids")
            if not q.get("prefill") or len(q.get("prefill_sampled") or []) != 1:
                problems.append(where + ": first token not sampled by the batch prefill")
                continue
            counts["prefill_host_tokens"] += 1
            positions = [ev.get("pos") for ev in t["tokens"]]
            if positions != list(range(q["num_prompt_tokens"], q["num_prompt_tokens"] + len(positions))):
                problems.append(where + ": decode positions not consecutive from the prompt length")
            if q["prefill_sampled"] + [ev.get("token") for ev in t["tokens"]] != t["end"].get("generated"):
                problems.append(where + ": per-position tokens do not compose the generated ids")
            for ev in t["tokens"]:
                if ev.get("phase") != "decode":
                    problems.append(where + ": unexpected phase " + str(ev.get("phase")))
                    continue
                counts["decode_positions"] += 1
                problems.extend(where + f" pos {ev.get('pos')}: " + b for b in check_position(ev, arm, dump, forced, counts))
    if counts["decode_positions"] == 0 or counts["captures"] < 1 or counts["replays_not_capturing"] < 1:
        problems.append(run["name"] + ": no capture or no replay after capture in the trace")
    if forced and counts["forced_decidable"] == 0:
        problems.append(run["name"] + ": fault injection exercised no decidable record")
    return {"counts": counts, "problems": problems[:50], "problem_count": len(problems)}


def rows_finite(path, vocab):
    import numpy as np
    data = np.memmap(path, dtype=np.float32, mode="r")
    if data.size % vocab:
        return False
    step = vocab * 4096
    return all(bool(np.isfinite(data[i:i + step]).all()) for i in range(0, data.size, step))


def analyze(output, matrix=None):
    out = Path(output)
    manifest = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    result = {"status": None, "scope": ("Registered pick_e1_L16 eval100 greedy requests on one GPU, "
                                         "EMBER_DETERMINISTIC=1; model-specific deterministic-greedy opt-in only; "
                                         "no global/default eligibility; no sampler RNG-state equivalence"),
              "reasons": []}
    try:
        if manifest.get("schema") != SCHEMA or [r["name"] for r in manifest["runs"]] != [leg[0] for leg in LEGS]:
            raise Inconclusive("incomplete or unknown plan")
        if manifest.get("thresholds") != THRESHOLDS or not manifest.get("request_keys"):
            raise Inconclusive("altered thresholds or empty request inventory")
        for name, path in frozen_files().items():
            if sha(path) != manifest["frozen_sha256"].get(name):
                raise Inconclusive("frozen collector file changed: " + name)
        for key, path in manifest["inputs"].items():
            if sha(path) != manifest["sha256"][key]:
                raise Inconclusive("frozen input changed: " + key)
        for run, leg in zip(manifest["runs"], LEGS):
            if (run["arm"], run["kind"], run["dump"], run["trace"], run["forced"]) != leg[1:]:
                raise Inconclusive("plan leg altered: " + run["name"])
            want = BASE_ENV + ARMS[run["arm"]]
            if run["env"][:len(want)] != want:
                raise Inconclusive("leg environment altered: " + run["name"])
        gate = registered_gate()
        loaded = {}
        for run in manifest["runs"]:
            try:
                loaded[run["name"]] = gate.load_run(out, run, manifest)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise Inconclusive(run["name"] + ": " + str(exc)) from exc
            if not run["trace"] and any(e.startswith("EMBER_ROUTE_TRACE") for e in loaded[run["name"]][1]["env"]):
                raise Inconclusive(run["name"] + ": tracing in a row or timing leg")
            if not run["trace"] and (out / (run["name"] + ".trace.jsonl")).exists():
                raise Inconclusive(run["name"] + ": a trace exists for an untraced leg")

        # Correctness: every leg's replies, ids, stops, prompt ids and token counts; every row leg's bits.
        rows = [r["name"] for r in manifest["runs"] if r["kind"] == "rows"]
        vocab = loaded[rows[0]][1]["vocab"]
        if any(loaded[n][1]["vocab"] != vocab for n in rows) or any(
                [r["rows"] for r in loaded[n][0]] != [r["rows"] for r in loaded[rows[0]][0]] for n in rows):
            raise Inconclusive("row-leg inventories differ in shape")
        digest = {n: sha(out / (n + ".jsonl.rows.f32")) for n in rows + ["det_route_parts4_rows"]}
        finite = {n: rows_finite(out / (n + ".jsonl.rows.f32"), vocab) for n in rows}
        reference = gate.ids(loaded["det_rows_host_a"][0])
        controls = ["det_rows_host_a", "det_rows_host_b", "det_rows_parts0_a", "det_rows_parts0_b"]
        candidates = ["det_rows_parts4_a", "det_rows_parts4_b"]
        result["rows"] = {"vocab": vocab, "rows_per_leg": sum(r["rows"] for r in loaded[rows[0]][0]),
                          "sha256": digest, "finite": finite}
        if not all(finite[n] for n in controls):
            raise Inconclusive("nonfinite control logits")
        if len({digest[n] for n in controls}) != 1 or any(gate.ids(loaded[n][0]) != reference for n in controls):
            raise Inconclusive("control legs (host and PARTS0 repeats) differ: comparison inconclusive")
        correctness = []
        for n in candidates:
            if not finite[n]:
                correctness.append(n + ": nonfinite candidate logits with finite controls")
            if digest[n] != digest["det_rows_host_a"]:
                correctness.append(n + ": logit row bits differ from the controls")
        for name, (records, _) in loaded.items():
            if gate.ids(records) != reference:
                arm = next(r["arm"] for r in manifest["runs"] if r["name"] == name)
                if arm == "parts4":
                    correctness.append(name + ": replies, ids, stops, prompt ids or token counts differ")
                else:
                    raise Inconclusive(name + ": control leg ids differ")
        result["correctness"] = {"pass": not correctness, "failures": correctness}

        # Route evidence: every route leg proves its arm; the dumping route leg reproduces the untraced rows.
        routes, route_problems = {}, []
        for run in manifest["runs"]:
            if run["kind"] == "route":
                routes[run["name"]] = check_route(run, loaded[run["name"]][0], manifest)
                if routes[run["name"]]["problem_count"]:
                    route_problems.append(run["name"])
        if digest["det_route_parts4_rows"] != digest["det_rows_parts4_a"]:
            route_problems.append("det_route_parts4_rows: traced rows differ from untraced PARTS4 rows")
        if matrix is not None:
            spec = importlib.util.spec_from_file_location("parts4_route_matrix", MATRIX)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            try:
                routes["matrix"] = mod.analyze(matrix)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                routes["matrix"] = {"pass": False, "error": str(exc)}
            if not routes["matrix"].get("pass"):
                route_problems.append("route matrix (exclusions and graph-off fallback) failed")
        else:
            route_problems.append("route matrix not supplied")
        result["routes"] = routes
        result["route_evidence"] = {"pass": not route_problems, "failures": route_problems}

        # Timing: the two ordered triples, each pair judged alone; trace-on throughput is disclosure only.
        speed = {r["name"]: gate.speed(loaded[r["name"]][0]) for r in manifest["runs"]
                 if r["kind"] in ("timing", "route") and not r["dump"]}
        ratios = [{"parts4_over_host": speed[p["parts4"]] / speed[p["host"]],
                   "parts4_over_parts0": speed[p["parts4"]] / speed[p["parts0"]]} for p in PAIRS]
        performance = all(r[k] >= THRESHOLDS[k] for r in ratios for k in THRESHOLDS)
        result["timing"] = {"decode_tokens_per_second": speed, "pairs": ratios, "pass": performance,
                            "trace_on_over_off_disclosure": {
                                arm: speed["det_route_" + arm] / ((speed[PAIRS[0][arm]] + speed[PAIRS[1][arm]]) / 2)
                                for arm in ARMS},
                            "note": "Route legs are untimed evidence; their throughput is disclosure, never a ratio input."}
        if correctness:
            result["status"] = "CORRECTNESS_FAIL"
        elif route_problems:
            result["status"] = "INCONCLUSIVE"
            result["reasons"].append("route evidence incomplete: ineligible for adoption")
        elif not performance:
            result["status"] = "PERFORMANCE_MISS"
        else:
            result["status"] = "PASS"
    except Inconclusive as exc:
        result["status"] = "INCONCLUSIVE"
        result["reasons"].append(str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan")
    for name in ("binary", "model", "prompts", "output"):
        p.add_argument(name)
    p.add_argument("--port", type=int, default=8080)
    p = sub.add_parser("analyze")
    p.add_argument("output")
    p.add_argument("--matrix")
    args = parser.parse_args()
    try:
        if args.action == "plan":
            plan(args.binary, args.model, args.prompts, args.output, args.port)
            return 0
        result = analyze(args.output, args.matrix)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"status": "INCONCLUSIVE", "reasons": ["analysis error: " + str(exc)]}
    print(json.dumps(result, indent=2))
    return {"PASS": 0, "PERFORMANCE_MISS": 1, "CORRECTNESS_FAIL": 1}.get(result["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
