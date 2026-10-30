"""Plan and check E10.3 device receipts; this tool never launches a GPU job.

  python decode_pick_gate.py plan BIN GGUF PROMPTS.jsonl OUTPUT_DIR --port 8080
  python decode_pick_gate.py analyze OUTPUT_DIR

The plan prints sequential dr_e2e commands for the supervised remote campaign.
It records frozen input hashes and refuses to overwrite an existing directory.
Analyze is fail-closed: incomplete receipts are not a pass. Dumps check logits;
separate ABBA runs without dumps check compact-path ids and throughput.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
CLIENT = HERE.parent / "draft_2026-10-27" / "dr_e2e.py"
MODES = ("default", "det")
LEGS = (("rows_off", 0, True), ("rows_off_repeat", 0, True), ("rows_on", 1, True),
        ("off_a", 0, False), ("on_a", 1, False),
        ("on_b", 1, False), ("off_b", 0, False))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def request_keys(prompts):
    result = []
    lines = [line for line in Path(prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, line in enumerate(lines):
        r = json.loads(line)
        result.extend([[i, k + 1] for k in range(len(r["turns"]))]
                      if "turns" in r else [[i, 0]])
    if not result:
        raise ValueError("empty prompt set")
    return result


def plan(binary, model, prompts, output, port):
    paths = {k: str(Path(v).resolve()) for k, v in
             (("binary", binary), ("model", model), ("prompts", prompts))}
    # dr_e2e enumerates nonempty prompt lines.
    keys = request_keys(paths["prompts"])
    hashes = {k: sha(v) for k, v in paths.items()}
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "inputs": paths,
                "sha256": hashes, "request_keys": keys,
                "minimum_speedup": 1.15, "runs": []}
    for mode in MODES:
        for leg, pick, dump in LEGS:
            env = ["EMBER_DECODE_PICK=" + str(pick), "EMBER_SPEC=0", "EMBER_DECODE_GRAPH=1",
                   "EMBER_DECODE_OPT=1", "EMBER_DECODE_LINEAR=cublas",
                   "EMBER_DETERMINISTIC=" + ("1" if mode == "det" else "0")]
            name = mode + "_" + leg
            cmd = ["env", "-u", "TINYLLAMA_LOGITS_DUMP", "-u", "TINYLLAMA_TOPK_DEBUG", "-u", "EMBER_DECODE_TIMING",
                   "python3", str(CLIENT), "run", paths["binary"], paths["model"], "999",
                   str(port), paths["prompts"], str(out / (name + ".jsonl"))]
            if dump:
                cmd.append("--dump")
            cmd.extend(env)
            manifest["runs"].append({"name": name, "env": env, "dump": dump, "command": cmd})
    (out / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for run in manifest["runs"]:
        print(shlex.join(run["command"]))
    return manifest


def load_run(out, run, manifest):
    path = out / (run["name"] + ".jsonl")
    records = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
    if [[r["i"], r["turn"]] for r in records] != manifest["request_keys"]:
        raise ValueError(run["name"] + ": missing, extra, or reordered requests")
    if meta.get("env") != run["env"] or meta.get("once") is not False:
        raise ValueError(run["name"] + ": unexpected environment or no warm pass")
    if meta.get("gpu_layers") not in (999, "999"):
        raise ValueError(run["name"] + ": not the all-GPU run")
    for key, field in (("binary", "bin"), ("model", "gguf"), ("prompts", "prompts")):
        if meta.get(field) != manifest["inputs"][key]:
            raise ValueError(run["name"] + ": input path mismatch")
    if (meta.get("health") or {}).get("model") != manifest["inputs"]["model"]:
        raise ValueError(run["name"] + ": server reports a different model")
    if bool((meta.get("health") or {}).get("deterministic")) != run["name"].startswith("det_"):
        raise ValueError(run["name"] + ": server deterministic mode mismatch")
    if not run["dump"] and meta.get("dump_rows") is not None:
        raise ValueError(run["name"] + ": timing contaminated by logit dumping")
    for r in records:
        if (r.get("warm_pass_same") is not True or not isinstance(r.get("ids"), list)
                or not isinstance(r.get("prompt_ids"), list) or not r["prompt_ids"]):
            raise ValueError(run["name"] + ": invalid ids or unstable warm pass")
        if any(r.get(k, 0) != 0 for k in ("passes", "accepted", "drafted")):
            raise ValueError(run["name"] + ": speculation contaminated plain decode")
    if run["dump"]:
        vocab, offset = meta.get("vocab"), 0
        if not isinstance(vocab, int) or vocab <= 0:
            raise ValueError("invalid dump vocabulary")
        for r in records:
            if r.get("row0") != offset or not isinstance(r.get("rows"), int) or r["rows"] <= 0:
                raise ValueError("missing, empty, or noncontiguous dump rows")
            offset += r["rows"]
        if meta.get("dump_rows") != offset or Path(str(path) + ".rows.f32").stat().st_size != offset * vocab * 4:
            raise ValueError("truncated or extra dump data")
    return records, meta


def ids(records):
    return [(r["reply"], r["ids"], r["stop_id"], r["prompt_ids"], r["tokens"]) for r in records]


def speed(records):
    tokens, elapsed = 0, 0.0
    for r in records:
        wall, first = r.get("wall_ms"), r.get("first_token_ms")
        if (not isinstance(wall, (int, float)) or not isinstance(first, (int, float))
                or not math.isfinite(wall) or not math.isfinite(first) or not 0 <= first < wall):
            raise ValueError("invalid or missing first-token timing")
        tokens += max(0, r["tokens"] - 1)
        elapsed += wall - first
    if tokens <= 0 or elapsed <= 0:
        raise ValueError("no decode work to time")
    return tokens * 1000 / elapsed


def analyze(output):
    out = Path(output)
    manifest = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    expected = [mode + "_" + leg for mode in MODES for leg, _, _ in LEGS]
    if manifest.get("schema") != 1 or [r["name"] for r in manifest["runs"]] != expected:
        raise ValueError("incomplete or unknown plan")
    if not manifest.get("request_keys") or manifest.get("minimum_speedup") != 1.15:
        raise ValueError("empty requests or altered threshold")
    for key, path in manifest["inputs"].items():
        if sha(path) != manifest["sha256"][key]:
            raise ValueError("frozen input changed: " + key)
    loaded = {run["name"]: load_run(out, run, manifest) for run in manifest["runs"]}
    result = {"status": "PASS", "modes": {}, "scope": "E10.3 rows, greedy ids and paired throughput only; fallback/penalty device checks remain separate"}
    for mode in MODES:
        base, bm = loaded[mode + "_rows_off"]
        if any(ids(loaded[mode + "_" + leg][0]) != ids(base) for leg, _, _ in LEGS):
            raise ValueError(mode + ": reply, ids, stop, prompt ids or token count differ")
        repeat, rm = loaded[mode + "_rows_off_repeat"]
        baseline_dumps = [out / (mode + "_" + tag + ".jsonl.rows.f32")
                          for tag in ("rows_off", "rows_off_repeat")]
        if (bm["vocab"] != rm["vocab"] or [r["rows"] for r in base] != [r["rows"] for r in repeat]
                or sha(baseline_dumps[0]) != sha(baseline_dumps[1])):
            result["status"] = "BASELINE_UNSTABLE"
            result["modes"][mode] = {"baseline_repeat_bits_equal": False,
                                     "reason": "off/off logit bits differ; off/on bit gate is inconclusive"}
            continue
        other, om = loaded[mode + "_rows_on"]
        if bm["vocab"] != om["vocab"] or [r["rows"] for r in base] != [r["rows"] for r in other]:
            raise ValueError(mode + ": dump shapes differ")
        dumps = [out / (mode + "_rows_" + tag + ".jsonl.rows.f32") for tag in ("off", "on")]
        if sha(dumps[0]) != sha(dumps[1]):
            raise ValueError(mode + ": logit bits differ")
        ratios = [speed(loaded[mode + "_on_" + pair][0]) / speed(loaded[mode + "_off_" + pair][0])
                  for pair in ("a", "b")]
        passed = min(ratios) >= manifest["minimum_speedup"]
        result["modes"][mode] = {"requests": len(base), "logit_bits_equal": True,
                                 "paired_speedups": ratios, "throughput_pass": passed}
        if not passed and result["status"] == "PASS":
            result["status"] = "NO_GAIN"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan")
    for name in ("binary", "model", "prompts", "output"):
        p.add_argument(name)
    p.add_argument("--port", type=int, default=8080)
    p = sub.add_parser("analyze")
    p.add_argument("output")
    args = parser.parse_args()
    try:
        if args.action == "plan":
            plan(args.binary, args.model, args.prompts, args.output, args.port)
        else:
            result = analyze(args.output)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "INCOMPLETE_OR_INVALID", "error": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
