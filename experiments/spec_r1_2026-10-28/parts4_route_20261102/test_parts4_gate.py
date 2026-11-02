"""Synthetic receipts for parts4_gate.py and parts4_route_matrix.py: every verdict and every refusal path.

The fixtures satisfy the registered decode_pick_gate.load_run predicates unchanged; each test then breaks one
thing and checks the verdict. No binary, model or GPU is used.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import parts4_gate as g
import parts4_route_matrix as m

V = 8
PROMPTS = [{"user_input": "a"}, {"user_input": "b"}, {"turns": ["x", "y"]}]
WALL = {"host": 11.0, "parts0": 9.5, "parts4": 9.0}  # 200, 235.3 and 250 decode tokens per second


def record(k, key, wall):
    return {"i": key[0], "set": None, "turn": key[1], "reply": f"r{k}", "ids": [10 + k, 20 + k], "stop_id": 2,
            "prompt_ids": [1, 100 + k], "tokens": 3, "prompt_tokens": 2, "wall_ms": wall, "first_token_ms": 1.0,
            "drafted": 0, "accepted": 0, "passes": 0, "draft_ms": 0.0, "draft_calls": 0, "warm_pass_same": True}


def inventory(arm):
    base = {"kernel": 120, "unidentified": 0, "select": 0, "partition": 0, "partition_grid": [0, 0, 0], "merge": 0,
            "merge_grid": 0}
    if arm == "parts0":
        base["select"] = 1
    elif arm == "parts4":
        base.update(partition=1, partition_grid=[1, 4, 1], merge=1, merge_grid=1)
    return base


def pick_record(tok):
    return {"max": 5.0, "max_bits": 1084227584, "second": 1.0, "second_bits": 1065353216, "first": tok, "last": tok,
            "nan": 0}


def token_event(n, pos, tok, arm, dump, forced, captured, replay):
    ev = {"ev": "token", "request": n, "pos": pos, "phase": "decode", "token": tok, "eligibility_consistent": True,
          "debug_rows": dump, "forced": False, "graph": "replay", "captured": captured, "capture_id": 1,
          "replay_index": replay, "graph_parts": g.PARTS[arm]}
    if captured:
        ev["inventory"] = inventory(arm)
    if arm == "host":
        ev.update(eligible=False, exclusions="pick_off", record_returned=False, picked_on_device=False,
                  route="host_ineligible", graph_has_pick=False)
        return ev
    ev.update(eligible=True, exclusions="", record_returned=True, graph_has_pick=True, record=pick_record(tok),
              decision=tok, undecided="none",
              check={"record_written": True, "record_value": True, "record_bits": True,
                     "parts_written": 4 if arm == "parts4" else 0, "parts_value": 4 if arm == "parts4" else 0,
                     "merge_value": arm == "parts4"})
    if arm == "parts4":
        ev["scratch"] = [pick_record(tok)] * 4
    if dump:
        ev.update(route="host_after_debug_rows", picked_on_device=False)
    elif forced:
        ev.update(route="host_after_forced_undecidable", picked_on_device=False, forced=True)
    else:
        ev.update(route="compact", picked_on_device=True)
    return ev


def trace_lines(run, records):
    arm, dump, forced = run["arm"], run["dump"], run["forced"]
    leg_env = dict(e.split("=", 1) for e in g.BASE_ENV + g.ARMS[arm])
    env = {k: leg_env.get(k) for k in g.TRACE_ENV_KEYS}
    lines, n, replay = [], 0, 0
    for rec in records:
        for _ in range(2):
            n += 1
            np_ = len(rec["prompt_ids"])
            generated = rec["ids"] + [rec["stop_id"]]
            lines.append({"ev": "request", "request": n, "prompt_ids": rec["prompt_ids"], "num_prompt_tokens": np_,
                          "temperature": 1.0, "top_k": 1, "top_p": 1.0, "repetition_penalty": 1.0,
                          "repeat_on": False, "ir": False, "cpu_offload_layers": 0, "spec": False, "prefill": True,
                          "prefix_reused": 0, "start_pos": np_, "total_steps": np_ + 59,
                          "prefill_sampled": generated[:1], "deterministic": True, "debug_rows": dump,
                          "forced": forced, "env": env})
            for j, tok in enumerate(generated[1:]):
                replay += 1
                lines.append(token_event(n, np_ + j, tok, arm, dump, forced, replay == 1, replay))
            lines.append({"ev": "end", "request": n, "generated": generated, "decode_steps": len(generated) - 1})
    return lines


def write_trace(path, lines):
    Path(path).write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")


def build(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name, data in (("bin", b"binary"), ("model.gguf", b"model")):
        (inputs / name).write_bytes(data)
    (inputs / "eval.jsonl").write_text("".join(json.dumps(p) + "\n" for p in PROMPTS), encoding="utf-8")
    out = tmp_path / "gate"
    manifest = g.plan(inputs / "bin", inputs / "model.gguf", inputs / "eval.jsonl", out)
    keys = manifest["request_keys"]
    for run in manifest["runs"]:
        records = [record(k, key, WALL[run["arm"]]) for k, key in enumerate(keys)]
        meta = {"bin": manifest["inputs"]["binary"], "gguf": manifest["inputs"]["model"], "gpu_layers": 999,
                "env": run["env"], "once": False, "health": {"model": manifest["inputs"]["model"],
                                                             "deterministic": True},
                "prompts": manifest["inputs"]["prompts"], "summary": {}, "dump_rows": None, "vocab": None}
        if run["dump"]:
            for k, r in enumerate(records):
                r.update(rows=3, row0=3 * k, dump_sha256="x")
            rows = np.arange(3 * len(records) * V, dtype=np.float32).reshape(-1, V) * 0.25
            rows.tofile(out / (run["name"] + ".jsonl.rows.f32"))
            meta.update(dump_rows=3 * len(records), vocab=V)
        (out / (run["name"] + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        (out / (run["name"] + ".jsonl.meta.json")).write_text(json.dumps(meta), encoding="utf-8")
        if run["trace"]:
            write_trace(out / (run["name"] + ".trace.jsonl"), trace_lines(run, records))
    matrix = tmp_path / "matrix"
    build_matrix(matrix)
    return out, matrix


def matrix_event(n, pos, tok, graph, arm, config, captured):
    eligible, excl, routes = m.expected(graph, arm, config)
    ev = {"ev": "token", "request": n, "pos": pos, "phase": "decode", "token": tok, "eligible": eligible,
          "exclusions": excl, "eligibility_consistent": True, "graph": "replay" if graph else "off",
          "captured": bool(graph) and captured, "graph_parts": g.PARTS[arm] if eligible else 0,
          "graph_has_pick": bool(graph) and eligible, "record_returned": False, "picked_on_device": False,
          "route": sorted(routes)[0]}
    if ev["captured"]:
        ev["inventory"] = inventory(arm if eligible else "host")
    if graph and eligible:
        ev.update(record_returned=True, picked_on_device=True, route="compact", decision=tok, undecided="none",
                  check={"record_written": True, "record_value": True, "record_bits": True,
                         "parts_written": 4 if arm == "parts4" else 0, "parts_value": 4 if arm == "parts4" else 0,
                         "merge_value": arm == "parts4"})
    return ev


def build_matrix(out):
    out.mkdir()
    for graph, arm in m.legs():
        name = m.leg_name(graph, arm)
        rows, lines, n = [], [], 0
        for i in range(len(m.CONFIGS)):
            for p, prompt in enumerate(m.PROMPTS):
                n += 1
                ids = [30 + i, 40 + p]
                rows.append({"config": i, "prompt": prompt, "reply": "x", "ids": ids, "prompt_ids": [1, 2, 3 + p],
                             "stop_id": 2, "tokens": 3})
                generated = ids + [2]
                lines.append({"ev": "request", "request": n, "prompt_ids": [1, 2, 3 + p], "spec": False,
                              "cpu_offload_layers": 0, "ir": False, "deterministic": True,
                              "prefill_sampled": generated[:1]})
                for j, tok in enumerate(generated[1:]):
                    lines.append(matrix_event(n, 3 + j, tok, graph, arm, i, j == 0))
                lines.append({"ev": "end", "request": n, "generated": generated})
        (out / (name + ".json")).write_text(json.dumps(
            {"health": {"deterministic": True}, "rows": rows,
             "env": m.leg_env(graph, arm, out / (name + ".trace.jsonl"))}), encoding="utf-8")
        write_trace(out / (name + ".trace.jsonl"), lines)


def edit_trace(out, leg, change):
    path = out / (leg + ".trace.jsonl")
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    change(lines)
    write_trace(path, lines)


def edit_tokens(out, leg, change, which=lambda i, ev: True):
    def apply(lines):
        tokens = [ev for ev in lines if ev["ev"] == "token"]
        for i, ev in enumerate(tokens):
            if which(i, ev):
                change(ev)
    edit_trace(out, leg, apply)


@pytest.fixture
def receipts(tmp_path):
    return build(tmp_path)


def test_complete_receipts_pass(receipts):
    out, matrix = receipts
    result = g.analyze(out, matrix)
    assert result["status"] == "PASS", result["reasons"] + result.get("route_evidence", {}).get("failures", [])
    assert result["correctness"]["pass"] and result["route_evidence"]["pass"] and result["timing"]["pass"]
    counts = result["routes"]["det_route_parts4"]["counts"]
    assert counts["requests"] == 8 and counts["prefill_host_tokens"] == 8 and counts["decode_positions"] == 16
    assert counts["compact"] == 16 and counts["captures"] == 1 and counts["replays_not_capturing"] == 15
    forced = result["routes"]["det_route_parts4_forced"]["counts"]
    assert forced["forced_decidable"] == forced["forced_decision_agrees"] == 16
    pair = result["timing"]["pairs"][0]
    assert pair["parts4_over_host"] == pytest.approx(1.25) and pair["parts4_over_parts0"] == pytest.approx(8.5 / 8)


def test_plan_commands(receipts):
    out, _ = receipts
    manifest = json.loads((out / "plan.json").read_text())
    names = [r["name"] for r in manifest["runs"]]
    assert names[-6:] == [g.PAIRS[0][a] for a in ("host", "parts0", "parts4")] + [
        g.PAIRS[1][a] for a in ("parts4", "parts0", "host")]
    for run in manifest["runs"]:
        cmd = run["command"]
        for var in g.CLEARED:
            assert ["-u", var] == cmd[cmd.index(var) - 1:cmd.index(var) + 1]
        assert ("--dump" in cmd) is run["dump"]
        traced = any(e.startswith("EMBER_ROUTE_TRACE=") for e in run["env"])
        assert traced is run["trace"] and (run["kind"] == "route") is run["trace"]
        assert ("EMBER_ROUTE_TRACE_FORCE=undecidable" in run["env"]) is run["forced"]
        assert run["env"][:7] == g.BASE_ENV + g.ARMS[run["arm"]] and "EMBER_DETERMINISTIC=1" in run["env"]
    with pytest.raises(FileExistsError):
        g.plan(manifest["inputs"]["binary"], manifest["inputs"]["model"], manifest["inputs"]["prompts"], out)


@pytest.mark.parametrize("pair,ratio,status", [
    (0, "host", "PERFORMANCE_MISS"), (1, "host", "PERFORMANCE_MISS"), (1, "parts0", "PERFORMANCE_MISS")])
def test_each_pair_judged_alone(receipts, pair, ratio, status):
    out, matrix = receipts
    # Slow one comparator leg's opposite: parts4 in one pair just below the bound, the other pair unchanged.
    leg = g.PAIRS[pair]["parts4"]
    path = out / (leg + ".jsonl")
    records = [json.loads(x) for x in path.read_text().splitlines()]
    wall = 1 + (WALL["host"] - 1) / 1.149 if ratio == "host" else 1 + (WALL["parts0"] - 1) / 0.999
    for r in records:
        r["wall_ms"] = wall
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    result = g.analyze(out, matrix)
    assert result["status"] == status and not result["timing"]["pass"]
    assert result["timing"]["pairs"][1 - pair]["parts4_over_host"] >= 1.15


def test_candidate_row_bits_differ(receipts):
    out, matrix = receipts
    path = out / "det_rows_parts4_b.jsonl.rows.f32"
    rows = np.fromfile(path, dtype=np.float32)
    rows[5] = np.nextafter(rows[5], np.float32(1e9))
    rows.tofile(path)
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL"
    assert any("bits differ" in f for f in result["correctness"]["failures"])


def test_candidate_only_nonfinite_is_a_correctness_failure(receipts):
    out, matrix = receipts
    path = out / "det_rows_parts4_a.jsonl.rows.f32"
    rows = np.fromfile(path, dtype=np.float32)
    rows[3] = np.nan
    rows.tofile(path)
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL"
    assert any("nonfinite candidate" in f for f in result["correctness"]["failures"])


def test_nonfinite_controls_are_inconclusive(receipts):
    out, matrix = receipts
    for leg in ("det_rows_host_a", "det_rows_host_b", "det_rows_parts0_a", "det_rows_parts0_b"):
        path = out / (leg + ".jsonl.rows.f32")
        rows = np.fromfile(path, dtype=np.float32)
        rows[3] = np.inf
        rows.tofile(path)
    assert g.analyze(out, matrix)["status"] == "INCONCLUSIVE"


def test_unstable_controls_are_inconclusive(receipts):
    out, matrix = receipts
    path = out / "det_rows_host_b.jsonl.rows.f32"
    rows = np.fromfile(path, dtype=np.float32)
    rows[0] += 1
    rows.tofile(path)
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "control" in result["reasons"][0]


def test_candidate_ids_differ(receipts):
    out, matrix = receipts
    path = out / "det_time_parts4_b.jsonl"
    records = [json.loads(x) for x in path.read_text().splitlines()]
    records[1]["ids"] = [99, 98]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert g.analyze(out, matrix)["status"] == "CORRECTNESS_FAIL"


@pytest.mark.parametrize("leg,change,expect", [
    ("det_route_parts4", lambda ev: ev.update(route="per_launch_fallback", record_returned=False,
                                             graph="disabled"), "silent fallback"),
    ("det_route_parts4", lambda ev: ev["check"].update(record_written=False), "sentinel"),
    ("det_route_parts4", lambda ev: ev["check"].update(parts_written=3), "partition records"),
    ("det_route_parts4", lambda ev: ev.update(graph_parts=0), "arm wants 4"),
    ("det_route_parts4", lambda ev: ev["check"].update(record_value=False), "whole-row oracle"),
    ("det_route_parts4", lambda ev: ev.update(decision=-5), "decision"),
    ("det_route_parts0", lambda ev: ev.update(scratch=[{}] * 4), "PARTS0 replay"),
    ("det_route_parts4_forced", lambda ev: ev.update(decision=-5), "host sampler"),
    ("det_route_parts4_forced", lambda ev: ev.update(route="compact", picked_on_device=True), "forced leg consumed"),
    ("det_route_parts4_rows", lambda ev: ev.update(route="compact", picked_on_device=True), "debug-row leg"),
    ("det_route_host", lambda ev: ev.update(route="compact", picked_on_device=True), "host arm"),
    ("det_route_parts4", lambda ev: ev.update(eligibility_consistent=False), "eligibility mask"),
])
def test_route_failures_are_inconclusive(receipts, leg, change, expect):
    out, matrix = receipts
    edit_tokens(out, leg, change, which=lambda i, ev: i == 3)
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE", result["status"]
    assert leg in result["route_evidence"]["failures"]
    assert any(expect in p for p in result["routes"][leg]["problems"]), result["routes"][leg]["problems"]


def test_capture_inventory_must_name_the_arms_kernels(receipts):
    out, matrix = receipts
    edit_tokens(out, "det_route_parts4", lambda ev: ev["inventory"].update(select=1, partition=0),
                which=lambda i, ev: ev["captured"])
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE"
    assert any("selector nodes" in p for p in result["routes"]["det_route_parts4"]["problems"])


def test_capture_without_replay_is_not_proof(receipts):
    out, matrix = receipts
    edit_tokens(out, "det_route_parts4", lambda ev: ev.update(captured=True, inventory=inventory("parts4")))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE"
    assert any("no replay after capture" in p for p in result["routes"]["det_route_parts4"]["problems"])


def test_near_tie_route_is_accepted_and_counted(receipts):
    out, matrix = receipts
    edit_tokens(out, "det_route_parts4", lambda ev: ev.update(route="host_after_near_tie", picked_on_device=False,
                                                              decision=-1, undecided="near_tie"),
                which=lambda i, ev: i == 4)
    result = g.analyze(out, matrix)
    assert result["status"] == "PASS"
    assert result["routes"]["det_route_parts4"]["counts"]["near_ties"] == 1


def test_trace_must_cover_warm_and_recorded_requests(receipts):
    out, matrix = receipts
    edit_trace(out, "det_route_parts0", lambda lines: [lines.pop() for _ in range(4)])
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE"


def test_trace_prompt_must_match_the_client(receipts):
    out, matrix = receipts
    edit_trace(out, "det_route_parts4", lambda lines: lines[0].update(prompt_ids=[7, 7]))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE"
    assert any("traced prompt" in p for p in result["routes"]["det_route_parts4"]["problems"])


def test_trace_environment_must_be_the_arm(receipts):
    out, matrix = receipts
    edit_trace(out, "det_route_parts4", lambda lines: lines[0]["env"].update(EMBER_DECODE_PICK_PARTS="0"))
    result = g.analyze(out, matrix)
    assert any("environment" in p for p in result["routes"]["det_route_parts4"]["problems"])


def test_missing_trace_is_inconclusive(receipts):
    out, matrix = receipts
    (out / "det_route_parts4.trace.jsonl").unlink()
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "no route trace" in result["reasons"][0]


def test_traced_rows_must_equal_untraced_rows(receipts):
    out, matrix = receipts
    path = out / "det_route_parts4_rows.jsonl.rows.f32"
    rows = np.fromfile(path, dtype=np.float32)
    rows[1] += 1
    rows.tofile(path)
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE"
    assert any("traced rows differ" in f for f in result["route_evidence"]["failures"])


def test_trace_in_a_timing_leg_is_refused(receipts):
    out, matrix = receipts
    (out / "det_time_parts4_a.trace.jsonl").write_text("")
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "untraced leg" in result["reasons"][0]


@pytest.mark.parametrize("edit", [
    lambda p: p.update(thresholds={"parts4_over_host": 1.1, "parts4_over_parts0": 1.0}),
    lambda p: p["runs"].pop(),
    lambda p: p["runs"][12].update(dump=True),
    lambda p: p["runs"][12]["env"].__setitem__(0, "EMBER_SPEC=4"),
])
def test_altered_plan_is_refused(receipts, edit):
    out, matrix = receipts
    plan = json.loads((out / "plan.json").read_text())
    edit(plan)
    (out / "plan.json").write_text(json.dumps(plan))
    assert g.analyze(out, matrix)["status"] == "INCONCLUSIVE"


def test_changed_input_is_refused(receipts):
    out, matrix = receipts
    plan = json.loads((out / "plan.json").read_text())
    Path(plan["inputs"]["model"]).write_bytes(b"other model")
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "frozen input" in result["reasons"][0]


def test_registered_predicates_still_apply(receipts):
    out, matrix = receipts
    meta = out / "det_time_host_a.jsonl.meta.json"
    data = json.loads(meta.read_text())
    data["health"]["deterministic"] = False
    meta.write_text(json.dumps(data))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "deterministic mode" in result["reasons"][0]


def test_matrix_is_required(receipts):
    out, _ = receipts
    result = g.analyze(out, None)
    assert result["status"] == "INCONCLUSIVE"
    assert "route matrix not supplied" in result["route_evidence"]["failures"]


def test_matrix_passes(receipts):
    _, matrix = receipts
    result = m.analyze(matrix)
    assert result["pass"], result["problems"]
    # Registered greedy and both low temperatures: 3 configurations x 3 prompts x 2 decode positions.
    assert result["counts"]["graph1_parts4"]["routes"]["compact"] == 18
    assert result["counts"]["graph0_parts4"]["routes"]["per_launch_fallback"] == 18


@pytest.mark.parametrize("leg,change,which,expect", [
    ("graph1_parts4", lambda ev: ev.update(exclusions=""), lambda i, ev: ev["exclusions"] == "top_k", "exclusions"),
    ("graph0_parts4", lambda ev: ev.update(route="compact"), lambda i, ev: ev["eligible"], "route compact"),
    ("graph1_parts0", lambda ev: ev.update(graph_parts=4), lambda i, ev: ev["eligible"], "selector"),
    ("graph1_host", lambda ev: ev.update(record_returned=True), lambda i, ev: i == 0, "record consumed"),
    ("graph1_parts4", lambda ev: ev.update(route="host_after_near_tie", picked_on_device=False),
     lambda i, ev: ev["eligible"] and ev["request"] == 10, "always decides"),
])
def test_matrix_route_failures(receipts, leg, change, which, expect):
    _, matrix = receipts
    edit_tokens(matrix, leg, change, which)
    result = m.analyze(matrix)
    assert not result["pass"] and any(expect in p for p in result["problems"]), result["problems"]


def test_matrix_responses_must_agree(receipts):
    _, matrix = receipts
    path = matrix / "graph0_parts4.json"
    leg = json.loads(path.read_text())
    leg["rows"][4]["ids"] = [1, 1]
    path.write_text(json.dumps(leg))
    result = m.analyze(matrix)
    assert not result["pass"] and any("differs from host" in p for p in result["problems"])
