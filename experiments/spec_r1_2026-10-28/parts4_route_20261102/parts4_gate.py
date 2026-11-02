"""Plan and check the E10 PARTS4 deterministic-greedy gate; this tool never launches a GPU job.

  python3 parts4_gate.py plan BIN GGUF PROMPTS.jsonl OUTPUT_DIR [--port 8080]
  python3 parts4_gate.py analyze OUTPUT_DIR --matrix MATRIX_DIR [--relocate KEY=PATH ...]

Protocol: ../parts4_full_model_protocol_20261102.json (public at 41b6cd5). The requests, the client and every
validity predicate of a leg are the registered R1 ones: dr_e2e.py run, and decode_pick_gate.load_run / ids /
speed, imported unchanged and hash-bound in the plan. Three arms, all under EMBER_DETERMINISTIC=1:
host (compact picking off, PARTS0), parts0 (on, PARTS0), parts4 (on, PARTS4).

Legs, in order: five untimed route-proof legs with EMBER_ROUTE_TRACE (decode_route_trace.h): one per arm, PARTS4
with fault injection (every record handed to the API undecidable: host-row plumbing, not a real near tie), and
PARTS4 with row dumps; then the protocol's six row legs (dumps, no trace) and its two ordered timing triples (no
dumps, no trace). A route leg's server traces every request; analyze aligns the trace with the client's records,
recomputes every position's decision and route from the serialized record, checks capture and replay continuity
and the captured selector nodes, and, for the dumping PARTS4 route leg, recomputes every record and partition
record from the dumped rows.

Analyze is fail-closed and keeps two tracks. A candidate failure demonstrated against valid controls (PARTS4 row
bits, ids, row inventory or finiteness; a written PARTS4 record that disagrees with its row; a PARTS4 response
that differs from host in the route matrix) is CORRECTNESS_FAIL, and later missing evidence never downgrades it.
Otherwise any operational, control, plan or route-evidence failure is INCONCLUSIVE (ineligible for adoption);
then PERFORMANCE_MISS if either pair misses a threshold; else PASS (deterministic-greedy, model-specific opt-in
eligibility only).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import struct
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
CONTROL_ARMS = ("host", "parts0")
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
CONTROL_ROWS = ("det_rows_host_a", "det_rows_host_b", "det_rows_parts0_a", "det_rows_parts0_b")
CANDIDATE_ROWS = ("det_rows_parts4_a", "det_rows_parts4_b")
# The routing variables a trace's request event reports from the server's environment.
TRACE_ENV_KEYS = ("EMBER_DECODE_PICK", "EMBER_DECODE_PICK_PARTS", "EMBER_DECODE_GRAPH", "EMBER_DECODE_OPT",
                  "EMBER_DETERMINISTIC", "EMBER_SPEC")
CLEARED = ("TINYLLAMA_LOGITS_DUMP", "TINYLLAMA_TOPK_DEBUG", "EMBER_DECODE_TIMING", "EMBER_ROUTE_TRACE",
           "EMBER_ROUTE_TRACE_FORCE")
INT_MAX = 2147483647


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


def leg_env(name, arm, trace, forced, out):
    env = BASE_ENV + ARMS[arm]
    if trace:
        env = env + ["EMBER_ROUTE_TRACE=" + str(Path(out) / (name + ".trace.jsonl"))]
    if forced:
        env = env + ["EMBER_ROUTE_TRACE_FORCE=undecidable"]
    return env


def leg_command(name, dump, env, inputs, out, port, client=CLIENT):
    cmd = ["env"] + [x for k in CLEARED for x in ("-u", k)] + [
        "python3", str(client), "run", inputs["binary"], inputs["model"], "999", str(port), inputs["prompts"],
        str(Path(out) / (name + ".jsonl"))]
    return cmd + (["--dump"] if dump else []) + env


def plan(binary, model, prompts, output, port=8080):
    gate = registered_gate()
    paths = {k: str(Path(v).resolve()) for k, v in (("binary", binary), ("model", model), ("prompts", prompts))}
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "inputs": paths, "sha256": {k: sha(v) for k, v in paths.items()},
                "frozen_sha256": {k: sha(v) for k, v in frozen_files().items()},
                "request_keys": gate.request_keys(paths["prompts"]), "thresholds": THRESHOLDS, "port": port,
                "client": str(CLIENT), "runs": []}
    for name, arm, kind, dump, trace, forced in LEGS:
        env = leg_env(name, arm, trace, forced, out)
        manifest["runs"].append({"name": name, "arm": arm, "kind": kind, "dump": dump, "trace": trace,
                                 "forced": forced, "env": env,
                                 "command": leg_command(name, dump, env, paths, out, port)})
    (out / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for run in manifest["runs"]:
        print(shlex.join(run["command"]))
    return manifest


class Inconclusive(Exception):
    """Operational, control, inventory or evidence failure: no scientific verdict."""


def check_plan(manifest, relocate=None):
    """Every leg's environment and command are exactly the canonical ones: no appended override (a later
    NAME=VALUE wins in the client), no duplicate, no other client, inputs or port. The executed client is the
    plan's recorded client, whose bytes must be the frozen client's: hashed where it was recorded (on the VM), or,
    offline, at relocate["client"]."""
    if manifest.get("schema") != SCHEMA or [r["name"] for r in manifest["runs"]] != [leg[0] for leg in LEGS]:
        raise Inconclusive("incomplete or unknown plan")
    if manifest.get("thresholds") != THRESHOLDS or not manifest.get("request_keys"):
        raise Inconclusive("altered thresholds or empty request inventory")
    first = manifest["runs"][0]["command"]
    recorded_out = Path(first[first.index("run") + 6]).parent  # run BIN GGUF 999 PORT PROMPTS OUT.jsonl
    client = manifest.get("client")
    if not isinstance(client, str) or Path(client).name != "dr_e2e.py" or not isinstance(manifest.get("port"), int):
        raise Inconclusive("plan client or port altered")
    where = (relocate or {}).get("client", client)
    if not Path(where).is_file() or sha(where) != manifest["frozen_sha256"].get("dr_e2e.py"):
        raise Inconclusive("the executed client is not the frozen client (offline: relocate client=PATH)")
    for run, leg in zip(manifest["runs"], LEGS):
        if (run["arm"], run["kind"], run["dump"], run["trace"], run["forced"]) != leg[1:]:
            raise Inconclusive("plan leg altered: " + run["name"])
        env = leg_env(run["name"], run["arm"], run["trace"], run["forced"], recorded_out)
        if run["env"] != env or run["command"] != leg_command(run["name"], run["dump"], env, manifest["inputs"],
                                                               recorded_out, manifest["port"], client):
            raise Inconclusive("leg environment or command is not canonical: " + run["name"])


def read_trace_partial(path):
    """The trace's complete requests, in order, up to the first malformed line: (requests, error or None). The
    requests before an error remain checkable, so a demonstrated failure in them is never lost to a bad tail."""
    requests, current = [], None
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return requests, f"{path}:{n}: unreadable event"
        kind = ev.get("ev") if isinstance(ev, dict) else None
        if kind == "request":
            if current is not None:
                return requests, f"{path}:{n}: request without end"
            if ev.get("request") != len(requests) + 1:
                return requests, f"{path}:{n}: request sequence is not 1..n"
            current = {"request": ev, "tokens": []}
        elif kind in ("token", "end"):
            if current is None or ev.get("request") != current["request"]["request"]:
                return requests, f"{path}:{n}: event outside its request"
            if kind == "token":
                current["tokens"].append(ev)
            else:
                current["end"] = ev
                requests.append(current)
                current = None
        else:
            return requests, f"{path}:{n}: unknown event"
    if current is not None:
        return requests, f"{path}: truncated final request"
    return requests, None


def read_trace(path):
    requests, error = read_trace_partial(path)
    if error:
        raise Inconclusive(error)
    return requests


def generated_matches(generated, ids, stop_id):
    """The trace's generated ids are the reply's ids, plus the stop id when generation stopped on one."""
    if not isinstance(generated, list) or generated[:len(ids)] != ids:
        return False
    return len(generated) == len(ids) or (len(generated) == len(ids) + 1 and generated[-1] == stop_id)


def composition_problems(t):
    """A traced request's first token comes from the batch prefill; one decode event per later token, at
    consecutive positions from the prompt length; together they are the generated ids."""
    q, bad = t["request"], []
    if not q.get("prefill") or len(q.get("prefill_sampled") or []) != 1:
        return ["first token not sampled by the batch prefill"]
    positions = [ev.get("pos") for ev in t["tokens"]]
    if positions != list(range(q["num_prompt_tokens"], q["num_prompt_tokens"] + len(positions))):
        bad.append("decode positions not consecutive from the prompt length")
    if q["prefill_sampled"] + [ev.get("token") for ev in t["tokens"]] != t["end"].get("generated"):
        bad.append("prefill token plus per-position tokens are not the generated ids")
    if any(ev.get("phase") != "decode" for ev in t["tokens"]):
        bad.append("a non-decode phase in a prefilled request")
    return bad


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def recompute(rec, temperature):
    """api.cpp greedy_from_pick's decision from the serialized record's bits, in float32: (token or -1, reason)."""
    import numpy as np
    mx, second = np.float32(f32(rec["max_bits"])), np.float32(f32(rec["second_bits"]))
    if rec["nan"] or not np.isfinite(mx) or rec["first"] < 0 or rec["last"] < 0:
        return -1, "nonfinite"
    t = np.float32(temperature)
    if t < np.float32(0.05):
        return rec["first"], "none"
    with np.errstate(all="ignore"):
        scaled = (second - mx) * (np.float32(1.0) / t)
    if not scaled < np.float32(-1e-5):
        return -1, "near_tie"
    return rec["last"], "none"


def expected_inventory(arm, vocab=None):
    """The selector nodes a captured graph must hold for an arm, with their captured arguments."""
    if arm == "host":
        want = {"select": 0, "partition": 0, "merge": 0}
    elif arm == "parts0":
        want = {"select": 1, "select_buffers": True, "partition": 0, "merge": 0}
    else:
        want = {"select": 0, "partition": 1, "partition_grid": [1, 4, 1], "partition_parts": 4,
                "partition_buffers": True, "merge": 1, "merge_grid": [1, 1, 1], "merge_count": 4,
                "merge_buffers": True}
    if vocab is not None and arm == "parts0":
        want["select_vocab"] = vocab
    if vocab is not None and arm == "parts4":
        want["partition_vocab"] = vocab
    return want


def inventory_problems(inventory, arm, vocab=None):
    want = expected_inventory(arm, vocab)
    if any(inventory.get(k) != v for k, v in want.items()) or not inventory.get("kernel", 0) > 0 \
            or not isinstance(inventory.get("unidentified"), int) or inventory["unidentified"] < 0:
        return ["captured graph's selector nodes or arguments differ from the arm: " + json.dumps(inventory)]
    return []


class Continuity:
    """Capture ids rise by one at each capture, whose replay is the first; every later position replays the
    current capture once."""

    def __init__(self):
        self.capture, self.replay = 0, 0

    def step(self, ev):
        if ev.get("captured"):
            ok = ev.get("capture_id") == self.capture + 1 and ev.get("replay_index") == 1
            self.capture, self.replay = self.capture + 1, 1
        else:
            ok = self.capture > 0 and ev.get("capture_id") == self.capture and ev.get("replay_index") == self.replay + 1
            self.replay += 1
        return [] if ok else [f"capture/replay counters break continuity (capture {ev.get('capture_id')}, "
                              f"replay {ev.get('replay_index')})"]


def check_position(ev, arm, dump, forced, counts, vocab):
    """One sampled decode position of a registered greedy request: (proof problems, demonstrated-wrong problems)."""
    proof, wrong = [], []
    parts = PARTS[arm]
    if not ev.get("eligibility_consistent"):
        proof.append("eligibility mask disagrees with want_device_pick")
    if ev.get("graph") != "replay":
        proof.append("decode position not served by a graph replay: " + str(ev.get("graph")))
    if ev.get("captured"):
        proof.extend(inventory_problems(ev.get("inventory") or {}, arm, vocab))
        counts["captures"] += 1
        counts["inventories"].append(ev.get("inventory"))
    else:
        counts["replays_not_capturing"] += 1
    route = ev.get("route")
    counts["routes"][route] = counts["routes"].get(route, 0) + 1
    if arm == "host":
        if (ev.get("eligible") or ev.get("exclusions") != "pick_off" or route != "host_ineligible"
                or ev.get("graph_has_pick") or ev.get("graph_parts") != 0 or ev.get("record_returned")
                or ev.get("picked_on_device") or ev.get("forced")):
            proof.append("host arm position took a compact-pick route")
        return proof, wrong
    if not ev.get("eligible") or ev.get("exclusions") != "" or not ev.get("record_returned"):
        proof.append("pick arm position not eligible or no record returned (silent fallback)")
        return proof, wrong
    if not ev.get("graph_has_pick") or ev.get("graph_parts") != parts:
        proof.append(f"replayed graph has pick={ev.get('graph_has_pick')} parts={ev.get('graph_parts')}, arm wants {parts}")
    check = ev.get("check") or {}
    if not check.get("record_written"):
        proof.append("record still the sentinel after replay: selector did not run on this replay")
    elif not check.get("record_value"):
        wrong.append("written record differs from the whole-row oracle of the same replay")
    counts["record_bits_equal"] += bool(check.get("record_bits"))
    if parts == 4:
        if check.get("parts_written") != 4 or len(ev.get("scratch") or []) != 4:
            proof.append("partition records not all written by this replay, or missing from the event")
        elif check.get("parts_value") != 4 or not check.get("merge_value"):
            wrong.append("written partition records differ from their partitions' oracles or do not merge to the record")
    elif ev.get("scratch") is not None:
        proof.append("PARTS0 replay reports partition records")
    try:
        decided, why = recompute(ev["record"], ev["temperature_used"])
    except (KeyError, TypeError):
        return proof + ["record or temperature missing from the event"], wrong
    if (ev.get("decision"), ev.get("undecided")) != (decided, why):
        proof.append(f"reported decision {ev.get('decision')}/{ev.get('undecided')} is not the record's {decided}/{why}")
    if why == "nonfinite":
        counts["nonfinite_records"] += 1
    if why == "near_tie":
        counts["near_ties"] += 1
    if dump:
        if route != "host_after_debug_rows" or ev.get("picked_on_device") or ev.get("forced"):
            proof.append("debug-row leg consumed a compact record")
    elif forced:
        if route != "host_after_forced_undecidable" or ev.get("picked_on_device") or ev.get("forced") is not True:
            proof.append("forced leg consumed a compact record, or the position is not labelled forced")
        if why == "none":
            counts["forced_decidable"] += 1
            if decided != ev.get("token"):
                wrong.append("record's decision differs from the host sampler's token for the same row")
            else:
                counts["forced_decision_agrees"] += 1
    else:
        want = {"none": "compact", "near_tie": "host_after_near_tie", "nonfinite": "host_after_nonfinite"}[why]
        if route != want or bool(ev.get("picked_on_device")) is not (want == "compact") or ev.get("forced"):
            proof.append(f"route {route} is not the record's route {want}")
        elif want == "compact":
            counts["compact"] += 1
            if ev.get("token") != decided:
                wrong.append("compact token is not the record's decision")
    return proof, wrong


def check_route(out, run, records, vocab=None, wrong=None):
    """The route evidence of one route leg against its client records. The trace is read beside the leg's
    records, so an archived output directory replays wherever it is unpacked. Demonstrated-wrong selector output
    is appended to `wrong` as it is found, so it survives an exception raised by a later event."""
    path = Path(out) / (run["name"] + ".trace.jsonl")
    if not path.is_file():
        raise Inconclusive(run["name"] + ": no route trace")
    requests, error = read_trace_partial(path)
    arm, dump, forced = run["arm"], run["dump"], run["forced"]
    canonical = dict(e.split("=", 1) for e in BASE_ENV + ARMS[arm])
    want_env = {k: canonical.get(k) for k in TRACE_ENV_KEYS}
    counts = {"requests": len(requests), "prefill_host_tokens": 0, "decode_positions": 0, "captures": 0,
              "replays_not_capturing": 0, "compact": 0, "near_ties": 0, "nonfinite_records": 0,
              "forced_decidable": 0, "forced_decision_agrees": 0, "record_bits_equal": 0, "routes": {},
              "inventories": []}
    proof = []
    wrong = [] if wrong is None else wrong
    if error:
        proof.append(run["name"] + ": " + error)
    if len(requests) != 2 * len(records) and not error:
        proof.append(f"{run['name']}: {len(requests)} traced requests for {len(records)} records (warm + recorded)")
    continuity = Continuity()
    # Every complete warm/recorded pair is checked, even when the trace is short or has a malformed tail.
    for k, rec in enumerate(records[:len(requests) // 2]):
        for t in (requests[2 * k], requests[2 * k + 1]):  # the warm pass, then the recorded request
            q = t["request"]
            where = f"{run['name']} record {k} request {q['request']}"
            if q.get("env") != want_env:
                proof.append(where + ": server environment differs from the arm")
            if (q.get("spec") is not False or q.get("ir") is not False or q.get("cpu_offload_layers") != 0
                    or q.get("deterministic") is not True or q.get("debug_rows") is not dump
                    or q.get("forced") is not forced or q.get("top_k") != 1 or q.get("temperature") != 1.0
                    or q.get("top_p") != 1.0 or q.get("repeat_on") is not False
                    or not q.get("repetition_penalty", 2.0) <= 1.0):
                proof.append(where + ": request is not a registered plain greedy all-GPU request")
            if q.get("prompt_ids") != rec["prompt_ids"]:
                proof.append(where + ": traced prompt differs from the client's prompt ids")
            if not generated_matches(t["end"].get("generated"), rec["ids"], rec["stop_id"]):
                proof.append(where + ": traced tokens differ from the client's ids")
            bad = composition_problems(t)
            proof.extend(where + ": " + b for b in bad)
            if bad:
                continue
            counts["prefill_host_tokens"] += 1
            for ev in t["tokens"]:
                counts["decode_positions"] += 1
                ev = dict(ev, temperature_used=q.get("temperature"))
                at = where + f" pos {ev.get('pos')}: "
                proof.extend(at + b for b in continuity.step(ev))
                p, w = check_position(ev, arm, dump, forced, counts, vocab)
                proof.extend(at + b for b in p)
                wrong.extend(at + b for b in w)
    if counts["decode_positions"] == 0 or counts["captures"] < 1 or counts["replays_not_capturing"] < 1:
        proof.append(run["name"] + ": no capture or no replay after capture in the trace")
    if arm != "host" and not dump and not forced and counts["compact"] == 0:
        proof.append(run["name"] + ": no position consumed a compact record")
    if forced and counts["forced_decidable"] == 0:
        proof.append(run["name"] + ": fault injection exercised no decidable record")
    return {"counts": counts, "problems": (wrong + proof)[:50], "problem_count": len(proof) + len(wrong),
            "wrong": wrong[:50], "wrong_count": len(wrong), "proof_count": len(proof)}


def row_oracle(row, begin, end):
    """select_rows_kernel / partition_rows_kernel's record of row[begin:end], independently in NumPy."""
    import numpy as np
    seg = row[begin:end]
    nan = bool(np.isnan(seg).any())
    finite = seg[~np.isnan(seg)]
    mx = finite.max() if finite.size else np.float32(-np.inf)
    hits = np.nonzero(seg == mx)[0]
    below = seg[seg < mx]
    return {"max": float(mx), "second": float(below.max()) if below.size else float("-inf"),
            "first": int(begin + hits[0]) if hits.size else INT_MAX, "last": int(begin + hits[-1]) if hits.size else -1,
            "nan": int(nan)}


def same_record(rec, oracle):
    return (f32(rec["max_bits"]) == oracle["max"] and f32(rec["second_bits"]) == oracle["second"]
            and (rec["first"], rec["last"], rec["nan"]) == (oracle["first"], oracle["last"], oracle["nan"]))


def check_dumped_records(out, run, records, vocab, wrong):
    """For the dumping PARTS4 route leg: every recorded request's records and partition records, recomputed in
    NumPy from the rows the client dumped (row 0 of a request is the prefill's, then one per decode position).
    Independent of the C++ oracle the trace reports. Appends demonstrated-wrong problems to `wrong` as found;
    returns the number of positions checked."""
    import numpy as np
    rows = np.memmap(Path(out) / (run["name"] + ".jsonl.rows.f32"), dtype=np.float32, mode="r").reshape(-1, vocab)
    requests, error = read_trace_partial(Path(out) / (run["name"] + ".trace.jsonl"))
    checked = 0
    for k, rec in enumerate(records[:len(requests) // 2]):
        tokens = requests[2 * k + 1]["tokens"]
        if rec["rows"] != len(tokens) + 1:
            raise Inconclusive(f"{run['name']} record {k}: {rec['rows']} dumped rows for {len(tokens)} decode positions")
        for j, ev in enumerate(tokens):
            row = np.asarray(rows[rec["row0"] + 1 + j])
            if not same_record(ev["record"], row_oracle(row, 0, vocab)):
                wrong.append(f"record {k} pos {ev['pos']}: record differs from its dumped row")
            bounds = [vocab * p // 4 for p in range(5)]
            for p in range(4):
                if not same_record(ev["scratch"][p], row_oracle(row, bounds[p], bounds[p + 1])):
                    wrong.append(f"record {k} pos {ev['pos']}: partition {p} differs from its dumped row")
            checked += 1
    if error or len(requests) != 2 * len(records):
        raise Inconclusive(f"{run['name']}: trace incomplete after {checked} checked positions: {error or len(requests)}")
    return checked


def rows_finite(path, vocab):
    import numpy as np
    data = np.memmap(path, dtype=np.float32, mode="r")
    if data.size % vocab:
        return False
    step = vocab * 4096
    return all(bool(np.isfinite(data[i:i + step]).all()) for i in range(0, data.size, step))


def analyze(output, matrix=None, relocate=None):
    """relocate: offline replay only. {"model": PATH, "prompts": PATH} hashes the frozen inputs where they now
    live; {"binary": "sha256:HEX"} accepts the executed binary by its recorded hash (binary bytes are not archived),
    which must equal the plan's. The plan's recorded paths are never rewritten."""
    out = Path(output)
    relocate = relocate or {}
    result = {"status": None, "scope": ("Registered pick_e1_L16 eval100 greedy requests on one GPU, "
                                         "EMBER_DETERMINISTIC=1; model-specific deterministic-greedy opt-in only; "
                                         "no global/default eligibility; no sampler RNG-state equivalence"),
              "correctness_failures": [], "evidence_failures": []}
    sci, op = result["correctness_failures"], result["evidence_failures"]
    loaded, perf = {}, None

    def section(name, fn):
        try:
            fn()
        except Inconclusive as exc:
            op.append(name + ": " + str(exc))
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            op.append(name + ": analysis error: " + repr(exc))

    try:
        manifest = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        check_plan(manifest, relocate)
        for name, path in frozen_files().items():
            if sha(path) != manifest["frozen_sha256"].get(name):
                raise Inconclusive("frozen collector file changed: " + name)
        for key, path in manifest["inputs"].items():
            where = relocate.get(key, path)
            digest = where[len("sha256:"):] if str(where).startswith("sha256:") else sha(where)
            if digest != manifest["sha256"][key]:
                raise Inconclusive("frozen input changed: " + key)
        gate = registered_gate()
    except Inconclusive as exc:
        op.append("plan: " + str(exc))
        result["status"] = "INCONCLUSIVE"
        return result
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        op.append("plan: analysis error: " + repr(exc))
        result["status"] = "INCONCLUSIVE"
        return result
    runs = {r["name"]: r for r in manifest["runs"]}

    for run in manifest["runs"]:
        def load(run=run):
            try:
                loaded[run["name"]] = gate.load_run(out, run, manifest)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise Inconclusive(str(exc)) from exc
            if not run["trace"] and (out / (run["name"] + ".trace.jsonl")).exists():
                raise Inconclusive("a trace exists for an untraced leg")
        section(run["name"], load)

    vocab = {}

    def correctness():
        """Controls first, on their own; then each loaded candidate leg separately against them. A missing or
        invalid candidate leg is an evidence failure that hides nothing about the others."""
        missing = [n for n in CONTROL_ROWS if n not in loaded]
        if missing:
            raise Inconclusive("control row legs not loaded: " + ", ".join(missing))
        gate_ids = gate.ids
        v = loaded["det_rows_host_a"][1]["vocab"]
        shape = [r["rows"] for r in loaded["det_rows_host_a"][0]]
        if any(loaded[n][1]["vocab"] != v or [r["rows"] for r in loaded[n][0]] != shape for n in CONTROL_ROWS):
            raise Inconclusive("control row inventories differ")
        digest = {n: sha(out / (n + ".jsonl.rows.f32")) for n in CONTROL_ROWS}
        finite = {n: rows_finite(out / (n + ".jsonl.rows.f32"), v) for n in CONTROL_ROWS}
        result["rows"] = {"vocab": v, "rows_per_leg": sum(shape), "sha256": digest, "finite": finite}
        if not all(finite.values()):
            raise Inconclusive("nonfinite control logits")
        reference = gate_ids(loaded["det_rows_host_a"][0])
        if len(set(digest.values())) != 1 or any(gate_ids(loaded[n][0]) != reference for n in CONTROL_ROWS):
            raise Inconclusive("control legs (host and PARTS0 repeats) differ: comparison inconclusive")
        vocab.update(v=v, reference=True)
        result["reference_ids_sha256"] = hashlib.sha256(json.dumps(reference).encode()).hexdigest()
        for n in CANDIDATE_ROWS:
            if n not in loaded:
                op.append(n + ": candidate row leg not loaded")
                continue
            rows = out / (n + ".jsonl.rows.f32")
            digest[n] = sha(rows)
            if loaded[n][1]["vocab"] != v or [r["rows"] for r in loaded[n][0]] != shape:
                sci.append(n + ": row inventory differs from the controls (generation changed)")
                continue
            finite[n] = rows_finite(rows, v)
            if digest[n] != digest["det_rows_host_a"]:
                sci.append(n + ": logit row bits differ from the controls")
            if not finite[n]:
                sci.append(n + ": nonfinite candidate logits with finite controls")
        for name, (records, _) in loaded.items():
            if gate_ids(records) != reference:
                if runs[name]["arm"] == "parts4":
                    sci.append(name + ": replies, ids, stops, prompt ids or token counts differ from the controls")
                else:
                    op.append(name + ": control leg ids differ")
    section("correctness", correctness)

    routes = result.setdefault("routes", {})

    def demonstrated(name, arm, wrong, what):
        """Demonstrated-wrong selector output is a candidate failure against valid controls, else evidence."""
        if wrong:
            (sci if arm == "parts4" and "reference" in vocab else op).append(f"{name}: {len(wrong)} {what}")

    def route_legs():
        for run in manifest["runs"]:
            if run["kind"] != "route":
                continue
            if run["name"] not in loaded:
                op.append(run["name"] + ": route leg not loaded")
                continue
            sink = []  # filled as positions are checked, so a later malformed event cannot discard it
            try:
                r = check_route(out, run, loaded[run["name"]][0], vocab.get("v"), sink)
                routes[run["name"]] = r
                if r["proof_count"]:
                    op.append(run["name"] + f": {r['proof_count']} route-evidence problems")
            except Exception as exc:
                op.append(run["name"] + ": " + (str(exc) if isinstance(exc, Inconclusive) else repr(exc)))
            demonstrated(run["name"], run["arm"], sink, "selector records demonstrably wrong")
        name = "det_route_parts4_rows"
        if name in loaded and "v" in vocab:
            sink = []
            try:
                checked = check_dumped_records(out, runs[name], loaded[name][0], vocab["v"], sink)
                routes["dumped_record_oracle"] = {"positions": checked, "wrong": sink[:50], "wrong_count": len(sink)}
                if checked == 0:
                    op.append(name + ": no dumped decode position to check")
            except Exception as exc:
                op.append(name + ": dumped-row oracle: " + (str(exc) if isinstance(exc, Inconclusive) else repr(exc)))
            demonstrated(name, "parts4", sink, "records differ from their dumped rows")
            ref = result["rows"]["sha256"].get("det_rows_parts4_a")
            if ref is None or sha(out / (name + ".jsonl.rows.f32")) != ref:
                op.append(name + ": traced rows differ from untraced PARTS4 rows (or no untraced rows to compare)")
        # One binary: graphs differ only by the selector nodes, whatever cuBLAS nodes the runtime cannot name.
        firsts = {a: (routes.get("det_route_" + a) or {}).get("counts", {}).get("inventories", [None])[0]
                  for a in ARMS}
        if all(firsts.values()):
            k, u = {a: firsts[a]["kernel"] for a in ARMS}, {a: firsts[a]["unidentified"] for a in ARMS}
            if not (k["parts0"] == k["host"] + 1 and k["parts4"] == k["host"] + 2 and len(set(u.values())) == 1):
                op.append("captured graphs differ beyond the selector nodes: " + json.dumps({"kernel": k, "unidentified": u}))
        else:
            op.append("a route leg has no captured-graph inventory")
    section("routes", route_legs)

    def matrix_check():
        if matrix is None:
            raise Inconclusive("route matrix not supplied")
        spec = importlib.util.spec_from_file_location("parts4_route_matrix", MATRIX)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        routes["matrix"] = mod.analyze(matrix, {"binary": manifest["sha256"]["binary"],
                                                "model": manifest["sha256"]["model"]})
        for d in routes["matrix"].get("response_differences", []):
            (sci if d.startswith("parts4") and "reference" in vocab else op).append("matrix: " + d)
        n = routes["matrix"].get("candidate_wrong_count", 0)
        if n:
            (sci if "reference" in vocab else op).append(f"matrix: {n} PARTS4 selector records demonstrably wrong")
        if routes["matrix"].get("problem_count"):
            op.append("matrix: exclusion and graph-off route evidence failed")
    section("matrix", matrix_check)

    def timing():
        nonlocal perf
        speed = {n: gate.speed(loaded[n][0]) for n in [p[a] for p in PAIRS for a in ARMS]}
        ratios = [{"parts4_over_host": speed[p["parts4"]] / speed[p["host"]],
                   "parts4_over_parts0": speed[p["parts4"]] / speed[p["parts0"]]} for p in PAIRS]
        perf = all(r[k] >= THRESHOLDS[k] for r in ratios for k in THRESHOLDS)
        traced = {a: gate.speed(loaded["det_route_" + a][0]) for a in ARMS if "det_route_" + a in loaded}
        result["timing"] = {"decode_tokens_per_second": speed, "pairs": ratios, "pass": perf,
                            "trace_on_over_off_disclosure": {
                                a: traced[a] / ((speed[PAIRS[0][a]] + speed[PAIRS[1][a]]) / 2) for a in traced},
                            "note": "Route legs are untimed evidence; their throughput is disclosure, never a ratio input."}
    section("timing", timing)

    if sci:
        result["status"] = "CORRECTNESS_FAIL"
    elif op:
        result["status"] = "INCONCLUSIVE"
    elif not perf:
        result["status"] = "PERFORMANCE_MISS"
    else:
        result["status"] = "PASS"
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
    p.add_argument("--relocate", action="append", default=[], metavar="KEY=PATH",
                   help="offline replay: model=PATH, prompts=PATH, binary=sha256:HEX")
    args = parser.parse_args()
    try:
        if args.action == "plan":
            plan(args.binary, args.model, args.prompts, args.output, args.port)
            return 0
        result = analyze(args.output, args.matrix, dict(r.split("=", 1) for r in args.relocate))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"status": "INCONCLUSIVE", "evidence_failures": ["analysis error: " + str(exc)]}
    print(json.dumps(result, indent=2))
    return {"PASS": 0, "PERFORMANCE_MISS": 1, "CORRECTNESS_FAIL": 1}.get(result["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
