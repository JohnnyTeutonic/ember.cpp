"""Synthetic receipts for parts4_gate.py and parts4_route_matrix.py: every verdict and every refusal path.

The fixtures satisfy the registered decode_pick_gate.load_run predicates unchanged. Their rows are real: every
traced record and partition record is computed from the row it describes (here, by a plain loop independent of
the checker's NumPy oracle), so the dumped-row oracle and the float32 decision recomputation are exercised.
Each test then breaks one thing and checks the verdict. No binary, model or GPU is used.
"""
import json
from pathlib import Path
import struct

import numpy as np
import pytest

import parts4_gate as g
import parts4_route_matrix as m

V = 64
PROMPTS = [{"user_input": "a"}, {"user_input": "b"}, {"turns": ["x", "y"]}]
WALL = {"host": 11.0, "parts0": 9.5, "parts4": 9.0}  # 200, 235.3 and 250 decode tokens per second
KERNELS = {"host": 120, "parts0": 121, "parts4": 122}


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def row_for(tok):
    row = np.full(V, -1.0, dtype=np.float32)
    row[tok] = 5.0
    row[(tok + 7) % V] = 1.0
    return row


def plain_record(row, a, b):
    vals = [float(v) for v in row[a:b]]
    mx = max(vals) if vals else float("-inf")
    hits = [a + i for i, v in enumerate(vals) if v == mx]
    below = [v for v in vals if v < mx]
    second = max(below) if below else float("-inf")
    return {"max": mx, "max_bits": bits(mx), "second": second, "second_bits": bits(second),
            "first": hits[0] if hits else g.INT_MAX, "last": hits[-1] if hits else -1, "nan": 0}


def record(k, key, wall):
    return {"i": key[0], "set": None, "turn": key[1], "reply": f"r{k}", "ids": [10 + k, 20 + k], "stop_id": 2,
            "prompt_ids": [1, 100 + k], "tokens": 3, "prompt_tokens": 2, "wall_ms": wall, "first_token_ms": 1.0,
            "drafted": 0, "accepted": 0, "passes": 0, "draft_ms": 0.0, "draft_calls": 0, "warm_pass_same": True}


def inventory(arm, vocab=V):
    inv = {"kernel": KERNELS[arm], "unidentified": 3, "select": 0, "partition": 0, "partition_grid": [0, 0, 0],
           "merge": 0, "merge_grid": [0, 0, 0], "select_vocab": -1, "select_buffers": False, "partition_vocab": -1,
           "partition_parts": -1, "partition_buffers": False, "merge_count": -1, "merge_buffers": False}
    if arm == "parts0":
        inv.update(select=1, select_vocab=vocab, select_buffers=True)
    elif arm == "parts4":
        inv.update(partition=1, partition_grid=[1, 4, 1], partition_vocab=vocab, partition_parts=4,
                   partition_buffers=True, merge=1, merge_grid=[1, 1, 1], merge_count=4, merge_buffers=True)
    return inv


def pick_fields(arm, tok, temperature=1.0):
    row = row_for(tok)
    rec = plain_record(row, 0, V)
    decided, why = g.recompute(rec, temperature)
    ev = {"record": rec, "decision": decided, "undecided": why,
          "check": {"record_written": True, "record_value": True, "record_bits": True,
                    "parts_written": 4 if arm == "parts4" else 0, "parts_value": 4 if arm == "parts4" else 0,
                    "merge_value": arm == "parts4"}}
    if arm == "parts4":
        ev["scratch"] = [plain_record(row, V * p // 4, V * (p + 1) // 4) for p in range(4)]
    return ev


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
    ev.update(eligible=True, exclusions="", record_returned=True, graph_has_pick=True, **pick_fields(arm, tok))
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
            rows = []
            for k, r in enumerate(records):
                r.update(rows=3, row0=3 * k, dump_sha256="x")
                rows.extend(row_for(t) for t in r["ids"] + [r["stop_id"]])
            np.stack(rows).tofile(out / (run["name"] + ".jsonl.rows.f32"))
            meta.update(dump_rows=3 * len(records), vocab=V)
        (out / (run["name"] + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        (out / (run["name"] + ".jsonl.meta.json")).write_text(json.dumps(meta), encoding="utf-8")
        if run["trace"]:
            write_trace(out / (run["name"] + ".trace.jsonl"), trace_lines(run, records))
    matrix = tmp_path / "matrix"
    build_matrix(matrix, manifest)
    return out, matrix


def matrix_event(n, pos, tok, graph, arm, config, capture, replay, captured):
    eligible, excl, routes = m.expected(graph, arm, config)
    picks = bool(graph) and eligible
    ev = {"ev": "token", "request": n, "pos": pos, "phase": "decode", "token": tok, "eligible": eligible,
          "exclusions": excl, "eligibility_consistent": True, "graph": "replay" if graph else "off",
          "captured": captured, "capture_id": capture, "replay_index": replay,
          "graph_parts": g.PARTS[arm] if picks else 0, "graph_has_pick": picks, "record_returned": False,
          "picked_on_device": False, "forced": False, "route": sorted(routes)[0] if not picks else "compact"}
    if captured:
        ev["inventory"] = inventory(arm if picks else "host")
    if picks:
        temperature = m.CONFIGS[config].get("temperature", 1.0)
        ev.update(record_returned=True, picked_on_device=True, **pick_fields(arm, tok, temperature))
    return ev


def build_matrix(out, manifest):
    out.mkdir()
    receipt = {"binary": manifest["inputs"]["binary"], "model": manifest["inputs"]["model"],
               "binary_sha256": manifest["sha256"]["binary"], "model_sha256": manifest["sha256"]["model"],
               "legs": [m.leg_name(gr, a) for gr, a in m.legs()], "completed": [m.leg_name(gr, a) for gr, a in m.legs()]}
    (out / "run.json").write_text(json.dumps(receipt))
    for graph, arm in m.legs():
        name = m.leg_name(graph, arm)
        rows, lines, n, capture, replay, state = [], [], 0, 0, 0, None
        for i in range(len(m.CONFIGS)):
            b = m.body(i, "")
            for p, prompt in enumerate(m.PROMPTS):
                n += 1
                ids = [30 + i, 40 + p]
                rows.append({"config": i, "prompt": prompt, "body": m.body(i, prompt), "reply": "x", "ids": ids,
                             "prompt_ids": [1, 2, 3 + p], "stop_id": 2, "tokens": 3})
                generated = ids + [2]
                env = {k: None for k in g.TRACE_ENV_KEYS}
                env.update({k: v for k, v in m.leg_env(graph, arm, "").items() if k in env})
                penalty = b.get("repeat_penalty")
                lines.append({"ev": "request", "request": n, "prompt_ids": [1, 2, 3 + p], "num_prompt_tokens": 3,
                              "temperature": float(b["temperature"]), "top_k": b["top_k"], "top_p": 1.0,
                              "repeat_on": penalty is not None,
                              "repeat_penalty": g.f32(m.gate_bits(penalty)) if penalty else 1.0,
                              "repeat_last_n": b.get("repeat_last_n", 64), "max_new_tokens": b["max_new_tokens"],
                              "debug_rows": False, "forced": False,
                              "spec": False, "ir": False, "cpu_offload_layers": 0, "deterministic": True,
                              "prefill": True, "prefill_sampled": generated[:1], "env": env})
                for j, tok in enumerate(generated[1:]):
                    picks = m.expected(graph, arm, i)[0]
                    captured = bool(graph) and picks != state
                    if graph:
                        if captured:
                            capture, replay, state = capture + 1, 1, picks
                        else:
                            replay += 1
                    lines.append(matrix_event(n, 3 + j, tok, graph, arm, i, capture, replay, captured))
                lines.append({"ev": "end", "request": n, "generated": generated})
        (out / (name + ".json")).write_text(json.dumps(
            {"health": {"deterministic": True, "model": manifest["inputs"]["model"]}, "rows": rows,
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


def failures(result):
    return " | ".join(result["correctness_failures"] + result["evidence_failures"])


@pytest.fixture
def receipts(tmp_path):
    return build(tmp_path)


def test_complete_receipts_pass(receipts):
    out, matrix = receipts
    result = g.analyze(out, matrix)
    assert result["status"] == "PASS", failures(result)
    counts = result["routes"]["det_route_parts4"]["counts"]
    assert counts["requests"] == 8 and counts["prefill_host_tokens"] == 8 and counts["decode_positions"] == 16
    assert counts["compact"] == 16 and counts["captures"] == 1 and counts["replays_not_capturing"] == 15
    forced = result["routes"]["det_route_parts4_forced"]["counts"]
    assert forced["forced_decidable"] == forced["forced_decision_agrees"] == 16
    assert result["routes"]["dumped_record_oracle"] == {"positions": 8, "wrong": [], "wrong_count": 0}
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


@pytest.mark.parametrize("pair,ratio", [(0, "host"), (1, "host"), (1, "parts0")])
def test_each_pair_judged_alone(receipts, pair, ratio):
    out, matrix = receipts
    leg = g.PAIRS[pair]["parts4"]
    path = out / (leg + ".jsonl")
    records = [json.loads(x) for x in path.read_text().splitlines()]
    wall = 1 + (WALL["host"] - 1) / 1.149 if ratio == "host" else 1 + (WALL["parts0"] - 1) / 0.999
    for r in records:
        r["wall_ms"] = wall
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    result = g.analyze(out, matrix)
    assert result["status"] == "PERFORMANCE_MISS" and not result["timing"]["pass"]
    assert result["timing"]["pairs"][1 - pair]["parts4_over_host"] >= 1.15


def rows_edit(out, leg, change):
    path = out / (leg + ".jsonl.rows.f32")
    rows = np.fromfile(path, dtype=np.float32)
    change(rows)
    rows.tofile(path)


def test_candidate_row_bits_differ(receipts):
    out, matrix = receipts
    rows_edit(out, "det_rows_parts4_b", lambda r: r.__setitem__(5, np.nextafter(r[5], np.float32(1e9))))
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL" and "bits differ" in failures(result)


def test_candidate_only_nonfinite_is_a_correctness_failure(receipts):
    out, matrix = receipts
    rows_edit(out, "det_rows_parts4_a", lambda r: r.__setitem__(3, np.nan))
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL" and "nonfinite candidate" in failures(result)


def test_nonfinite_controls_are_inconclusive(receipts):
    out, matrix = receipts
    for leg in g.CONTROL_ROWS:
        rows_edit(out, leg, lambda r: r.__setitem__(3, np.inf))
    assert g.analyze(out, matrix)["status"] == "INCONCLUSIVE"


def test_unstable_controls_are_inconclusive(receipts):
    out, matrix = receipts
    rows_edit(out, "det_rows_host_b", lambda r: r.__setitem__(0, r[0] + 1))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "control" in failures(result)


def test_candidate_ids_differ(receipts):
    out, matrix = receipts
    path = out / "det_time_parts4_b.jsonl"
    records = [json.loads(x) for x in path.read_text().splitlines()]
    records[1]["ids"] = [99, 98]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert g.analyze(out, matrix)["status"] == "CORRECTNESS_FAIL"


def test_candidate_generation_length_change_is_a_correctness_failure(receipts):
    # Valid controls, PARTS4 rows differ in count: a candidate behaviour change, not an inventory error.
    out, matrix = receipts
    for leg in g.CANDIDATE_ROWS:
        path, meta = out / (leg + ".jsonl"), out / (leg + ".jsonl.meta.json")
        records = [json.loads(x) for x in path.read_text().splitlines()]
        records[-1]["rows"] = 2
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        data = json.loads(meta.read_text())
        data["dump_rows"] -= 1
        meta.write_text(json.dumps(data))
        rows_edit(out, leg, lambda r: None)
        rows = np.fromfile(out / (leg + ".jsonl.rows.f32"), dtype=np.float32)[:-V]
        rows.tofile(out / (leg + ".jsonl.rows.f32"))
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL" and "row inventory differs" in failures(result)


def test_correctness_failure_survives_missing_route_evidence(receipts):
    out, matrix = receipts
    rows_edit(out, "det_rows_parts4_b", lambda r: r.__setitem__(5, r[5] + 1))
    (out / "det_route_parts4.trace.jsonl").unlink()
    (matrix / "run.json").unlink()
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL"
    assert "no route trace" in failures(result) and "bits differ" in failures(result)


def test_demonstrated_wrong_record_is_a_correctness_failure(receipts):
    out, matrix = receipts
    edit_tokens(out, "det_route_parts4", lambda ev: ev["check"].update(record_value=False), lambda i, ev: i == 3)
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL" and "demonstrably wrong" in failures(result)


def test_dumped_rows_are_checked_independently_of_the_reported_booleans(receipts):
    # The C++ check says all is well, but the serialized partition record is not its dumped row's.
    out, matrix = receipts
    edit_tokens(out, "det_route_parts4_rows", lambda ev: ev["scratch"][2].update(first=0),
                lambda i, ev: ev["request"] == 2 and i % 2 == 0)
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL"
    assert result["routes"]["dumped_record_oracle"]["wrong_count"] >= 1


@pytest.mark.parametrize("leg,change,expect", [
    ("det_route_parts4", lambda ev: ev.update(route="per_launch_fallback", record_returned=False,
                                             graph="disabled"), "silent fallback"),
    ("det_route_parts4", lambda ev: ev["check"].update(record_written=False), "sentinel"),
    ("det_route_parts4", lambda ev: ev["check"].update(parts_written=3), "partition records not all written"),
    ("det_route_parts4", lambda ev: ev.update(graph_parts=0), "arm wants 4"),
    ("det_route_parts4", lambda ev: ev.update(decision=-5), "reported decision"),
    ("det_route_parts4", lambda ev: ev.update(route="host_after_near_tie", picked_on_device=False), "record's route"),
    ("det_route_parts4", lambda ev: ev.update(capture_id=7), "continuity"),
    ("det_route_parts4", lambda ev: ev.update(replay_index=ev["replay_index"] + 2), "continuity"),
    ("det_route_parts0", lambda ev: ev.update(scratch=[{}] * 4), "PARTS0 replay"),
    ("det_route_parts4_forced", lambda ev: ev.update(route="compact", picked_on_device=True), "forced leg"),
    ("det_route_parts4_forced", lambda ev: ev.update(forced=False), "labelled forced"),
    ("det_route_parts4_rows", lambda ev: ev.update(route="compact", picked_on_device=True), "debug-row leg"),
    ("det_route_host", lambda ev: ev.update(route="compact", picked_on_device=True), "host arm"),
    ("det_route_parts4", lambda ev: ev.update(eligibility_consistent=False), "eligibility mask"),
])
def test_route_failures_are_inconclusive(receipts, leg, change, expect):
    out, matrix = receipts
    edit_tokens(out, leg, change, which=lambda i, ev: i == 3)
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE", (result["status"], failures(result))
    assert any(expect in p for p in result["routes"][leg]["problems"]), result["routes"][leg]["problems"]


def test_record_that_really_is_a_near_tie_routes_to_the_host(receipts):
    out, matrix = receipts

    def tie(ev):
        ev["record"].update(second=5.0 - 2e-6, second_bits=bits(np.float32(5.0 - 2e-6)))
        ev.update(route="host_after_near_tie", picked_on_device=False, decision=-1, undecided="near_tie")
    edit_tokens(out, "det_route_parts4", tie, which=lambda i, ev: i == 4)
    result = g.analyze(out, matrix)
    assert result["status"] == "PASS", failures(result)
    assert result["routes"]["det_route_parts4"]["counts"]["near_ties"] == 1


def test_a_pick_leg_with_no_compact_position_is_not_proof(receipts):
    out, matrix = receipts

    def tie(ev):
        ev["record"].update(second_bits=ev["record"]["max_bits"])
        ev.update(route="host_after_near_tie", picked_on_device=False, decision=-1, undecided="near_tie")
    edit_tokens(out, "det_route_parts4", tie)
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE"
    assert any("no position consumed" in p for p in result["routes"]["det_route_parts4"]["problems"])


@pytest.mark.parametrize("change,expect", [
    (lambda inv: inv.update(select=1, partition=0), "selector nodes"),
    (lambda inv: inv.update(partition_parts=2), "selector nodes"),
    (lambda inv: inv.update(merge_buffers=False), "selector nodes"),
    (lambda inv: inv.update(partition_vocab=V - 1), "selector nodes"),
])
def test_capture_inventory_must_name_the_arms_kernels_and_arguments(receipts, change, expect):
    out, matrix = receipts
    edit_tokens(out, "det_route_parts4", lambda ev: change(ev["inventory"]), which=lambda i, ev: ev["captured"])
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE"
    assert any(expect in p for p in result["routes"]["det_route_parts4"]["problems"])


def test_graphs_must_differ_only_by_the_selector_nodes(receipts):
    out, matrix = receipts
    edit_tokens(out, "det_route_parts0", lambda ev: ev["inventory"].update(kernel=140), lambda i, ev: ev["captured"])
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "beyond the selector nodes" in failures(result)


def test_capture_without_replay_is_not_proof(receipts):
    out, matrix = receipts
    counter = iter(range(1, 1000))
    edit_tokens(out, "det_route_parts4", lambda ev: ev.update(captured=True, capture_id=next(counter), replay_index=1,
                                                              inventory=inventory("parts4")))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE"
    assert any("no replay after capture" in p for p in result["routes"]["det_route_parts4"]["problems"])


def test_trace_must_cover_warm_and_recorded_requests(receipts):
    out, matrix = receipts
    edit_trace(out, "det_route_parts0", lambda lines: [lines.pop() for _ in range(4)])
    assert g.analyze(out, matrix)["status"] == "INCONCLUSIVE"


def test_trace_must_hold_every_position(receipts):
    out, matrix = receipts
    edit_trace(out, "det_route_parts4", lambda lines: lines.pop(2))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE"
    assert any("generated ids" in p for p in result["routes"]["det_route_parts4"]["problems"])


def test_trace_prompt_must_match_the_client(receipts):
    out, matrix = receipts
    edit_trace(out, "det_route_parts4", lambda lines: lines[0].update(prompt_ids=[7, 7]))
    result = g.analyze(out, matrix)
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
    assert result["status"] == "INCONCLUSIVE" and "no route trace" in failures(result)


def test_traced_rows_must_equal_untraced_rows(receipts):
    out, matrix = receipts
    rows_edit(out, "det_route_parts4_rows", lambda r: r.__setitem__(1, r[1] + 0.5))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "traced rows differ" in failures(result)


def test_trace_in_a_timing_leg_is_refused(receipts):
    out, matrix = receipts
    (out / "det_time_parts4_a.trace.jsonl").write_text("")
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "untraced leg" in failures(result)


def set_env(plan, name, env):
    run = next(r for r in plan["runs"] if r["name"] == name)
    k = len(run["command"]) - len(run["env"])
    run["env"] = env
    run["command"] = run["command"][:k] + env


@pytest.mark.parametrize("edit", [
    lambda p: p.update(thresholds={"parts4_over_host": 1.1, "parts4_over_parts0": 1.0}),
    lambda p: p["runs"].pop(),
    lambda p: p["runs"][12].update(dump=True),
    lambda p: set_env(p, "det_time_parts4_a", p["runs"][13]["env"] + ["EMBER_DECODE_PICK_PARTS=0"]),
    lambda p: set_env(p, "det_time_parts4_b", ["EMBER_SPEC=4"] + p["runs"][14]["env"][1:]),
    lambda p: set_env(p, "det_route_parts4", [e.replace("/gate/", "/elsewhere/") for e in p["runs"][2]["env"]]),
    lambda p: p["runs"][11]["command"].__setitem__(-8, "/other/model.gguf"),
])
def test_altered_plan_is_refused(receipts, edit):
    out, matrix = receipts
    plan = json.loads((out / "plan.json").read_text())
    edit(plan)
    (out / "plan.json").write_text(json.dumps(plan))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and failures(result).startswith("plan:")


def test_appended_override_is_refused_even_with_matching_metadata(receipts):
    out, matrix = receipts
    plan = json.loads((out / "plan.json").read_text())
    env = plan["runs"][13]["env"] + ["EMBER_DECODE_PICK_PARTS=0"]
    set_env(plan, "det_time_parts4_a", env)
    (out / "plan.json").write_text(json.dumps(plan))
    meta = out / "det_time_parts4_a.jsonl.meta.json"
    data = json.loads(meta.read_text())
    data["env"] = env
    meta.write_text(json.dumps(data))
    assert g.analyze(out, matrix)["status"] == "INCONCLUSIVE"


def test_changed_input_is_refused(receipts):
    out, matrix = receipts
    plan = json.loads((out / "plan.json").read_text())
    Path(plan["inputs"]["model"]).write_bytes(b"other model")
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "frozen input" in failures(result)


def test_registered_predicates_still_apply(receipts):
    out, matrix = receipts
    meta = out / "det_time_host_a.jsonl.meta.json"
    data = json.loads(meta.read_text())
    data["health"]["deterministic"] = False
    meta.write_text(json.dumps(data))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "deterministic mode" in failures(result)


def test_archived_output_replays_where_it_is_unpacked(receipts, tmp_path):
    out, matrix = receipts
    inputs = out.parent / "inputs"
    binary_sha = "sha256:" + g.sha(inputs / "bin")
    home = tmp_path / "home"
    home.mkdir()
    for p in (out, matrix, inputs):
        p.rename(home / p.name)
    moved = {"model": str(home / "inputs" / "model.gguf"), "prompts": str(home / "inputs" / "eval.jsonl")}
    result = g.analyze(home / "gate", home / "matrix", {**moved, "binary": binary_sha})
    assert result["status"] == "PASS", failures(result)
    refused = g.analyze(home / "gate", home / "matrix", {**moved, "binary": "sha256:" + "0" * 64})
    assert refused["status"] == "INCONCLUSIVE" and "binary" in failures(refused)
    assert g.analyze(home / "gate", home / "matrix")["status"] == "INCONCLUSIVE"  # inputs not where recorded


def test_matrix_is_required(receipts):
    out, _ = receipts
    result = g.analyze(out, None)
    assert result["status"] == "INCONCLUSIVE" and "route matrix not supplied" in failures(result)


def test_matrix_passes(receipts):
    out, matrix = receipts
    plan = json.loads((out / "plan.json").read_text())
    result = m.analyze(matrix, {"binary": plan["sha256"]["binary"], "model": plan["sha256"]["model"]})
    assert result["pass"], result["problems"]
    # Registered greedy and both low temperatures: 3 configurations x 3 prompts x 2 decode positions.
    assert result["counts"]["graph1_parts4"]["routes"]["compact"] == 18
    assert result["counts"]["graph0_parts4"]["routes"]["per_launch_fallback"] == 18


def test_matrix_from_another_binary_is_refused(receipts):
    out, matrix = receipts
    receipt = json.loads((matrix / "run.json").read_text())
    receipt["binary_sha256"] = "0" * 64
    (matrix / "run.json").write_text(json.dumps(receipt))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "another binary" in failures(result)


@pytest.mark.parametrize("leg,change,which,expect", [
    ("graph1_parts4", lambda ev: ev.update(exclusions=""), lambda i, ev: ev["exclusions"] == "top_k", "exclusions"),
    ("graph0_parts4", lambda ev: ev.update(route="compact"), lambda i, ev: ev["eligible"], "route compact"),
    ("graph1_parts0", lambda ev: ev.update(graph_parts=4), lambda i, ev: ev["eligible"], "selector"),
    ("graph1_host", lambda ev: ev.update(record_returned=True), lambda i, ev: i == 0, "record consumed"),
    ("graph1_parts4", lambda ev: ev.update(replay_index=99), lambda i, ev: i == 2, "continuity"),
    ("graph1_parts4", lambda ev: ev.update(decision=-1, undecided="near_tie"),
     lambda i, ev: ev["eligible"] and ev["request"] == 10, "decision"),
])
def test_matrix_route_failures(receipts, leg, change, which, expect):
    _, matrix = receipts
    edit_tokens(matrix, leg, change, which)
    result = m.analyze(matrix)
    assert not result["pass"] and any(expect in p for p in result["problems"]), result["problems"]


def test_matrix_requests_must_be_the_configuration_sent(receipts):
    _, matrix = receipts
    edit_trace(matrix, "graph1_parts4", lambda lines: next(x for x in lines if x["ev"] == "request" and x["top_k"] == 4)
               .update(top_k=1))
    result = m.analyze(matrix)
    assert not result["pass"] and any("configuration sent" in p for p in result["problems"])


def test_matrix_positions_cannot_be_dropped(receipts):
    _, matrix = receipts
    edit_trace(matrix, "graph1_parts4", lambda lines: [lines.remove(x) for x in [
        x for x in lines if x["ev"] == "token" and x["exclusions"] in ("top_k", "repeat_penalty")]])
    result = m.analyze(matrix)
    assert not result["pass"] and any("generated ids" in p for p in result["problems"])


def test_candidate_failure_survives_a_missing_sibling_leg(receipts):
    out, matrix = receipts
    rows_edit(out, "det_rows_parts4_a", lambda r: r.__setitem__(5, r[5] + 1))
    (out / "det_rows_parts4_b.jsonl").unlink()
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL", failures(result)
    assert "det_rows_parts4_a: logit row bits differ" in failures(result) and "parts4_b" in failures(result)


def test_wrong_record_survives_a_later_malformed_event(receipts):
    out, matrix = receipts

    def change(lines):
        tokens = [x for x in lines if x["ev"] == "token"]
        tokens[0]["check"]["record_value"] = False
        next(x for x in lines if x["ev"] == "request" and x["request"] == 5)["num_prompt_tokens"] = None
    edit_trace(out, "det_route_parts4", change)
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL", failures(result)
    assert "demonstrably wrong" in failures(result) and "TypeError" in failures(result)


def test_matrix_wrong_parts4_record_is_a_correctness_failure(receipts):
    out, matrix = receipts
    edit_tokens(matrix, "graph1_parts4", lambda ev: ev["check"].update(record_value=False),
                lambda i, ev: ev["eligible"] and i == 0)
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL" and "matrix: 1 PARTS4 selector records" in failures(result)


def test_matrix_wrong_parts0_record_is_evidence_failure(receipts):
    out, matrix = receipts
    edit_tokens(matrix, "graph1_parts0", lambda ev: ev["check"].update(record_value=False),
                lambda i, ev: ev["eligible"] and i == 0)
    assert g.analyze(out, matrix)["status"] == "INCONCLUSIVE"


def test_executed_client_must_be_the_frozen_client(receipts, tmp_path):
    out, matrix = receipts
    other = tmp_path / "other" / "dr_e2e.py"
    other.parent.mkdir()
    other.write_text("print('not the registered client')\n")
    plan = json.loads((out / "plan.json").read_text())
    plan["client"] = str(other)
    for run in plan["runs"]:
        run["command"][run["command"].index("run") - 1] = str(other)
    (out / "plan.json").write_text(json.dumps(plan))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "frozen client" in failures(result)
    # A client at a path that no longer exists replays only with an explicit relocation to the frozen bytes.
    other.unlink()
    assert "frozen client" in failures(g.analyze(out, matrix))
    assert g.analyze(out, matrix, {"client": str(g.CLIENT)})["status"] == "PASS"


def test_matrix_request_parameters_must_be_the_ones_sent(receipts):
    _, matrix = receipts
    edit_trace(matrix, "graph1_parts4", lambda lines: next(
        x for x in lines if x["ev"] == "request" and x["repeat_on"]).update(repeat_last_n=8))
    result = m.analyze(matrix)
    assert not result["pass"] and any("configuration sent" in p for p in result["problems"])


def test_matrix_request_body_must_be_the_one_specified(receipts):
    out, matrix = receipts
    path = matrix / "graph1_parts4.json"
    leg = json.loads(path.read_text())
    leg["rows"][3]["body"]["repeat_penalty"] = 1.05
    path.write_text(json.dumps(leg))
    result = g.analyze(out, matrix)
    assert result["status"] == "INCONCLUSIVE" and "request body" in failures(result)


def test_matrix_allows_only_true_immediate_stops_without_positions(receipts):
    _, matrix = receipts
    for name in [m.leg_name(gr, a) for gr, a in m.legs()]:
        path = matrix / (name + ".json")
        leg = json.loads(path.read_text())
        leg["rows"][0]["ids"] = []
        path.write_text(json.dumps(leg))

        def stop_first(lines):
            req = next(x for x in lines if x["ev"] == "request" and x["request"] == 1)
            req["prefill_sampled"] = [2]
            for x in [x for x in lines if x["ev"] == "token" and x["request"] == 1]:
                lines.remove(x)
            next(x for x in lines if x["ev"] == "end" and x["request"] == 1)["generated"] = [2]
            # the next capture-holding position now opens the leg's first capture
            first = next(x for x in lines if x["ev"] == "token")
            if first["graph"] == "replay" and not first["captured"]:
                first.update(captured=True, replay_index=1, inventory=inventory(
                    first["graph_parts"] == 4 and "parts4" or first["graph_has_pick"] and "parts0" or "host"))
                for x in [x for x in lines if x["ev"] == "token" and x["capture_id"] == first["capture_id"]][1:]:
                    x["replay_index"] -= 2
        edit_trace(matrix, name, stop_first)
    result = m.analyze(matrix)
    assert result["pass"], result["problems"]
    assert result["counts"]["graph1_parts4"]["immediate_stops"] == 1


def test_matrix_parts4_response_difference_is_a_correctness_failure(receipts):
    out, matrix = receipts
    path = matrix / "graph0_parts4.json"
    leg = json.loads(path.read_text())
    leg["rows"][4]["ids"] = [1, 1]
    path.write_text(json.dumps(leg))
    result = g.analyze(out, matrix)
    assert result["status"] == "CORRECTNESS_FAIL" and "differs from host" in failures(result)
