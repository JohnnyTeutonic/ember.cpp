"""Fail-closed E9 receipt checks; no GPU or model calls are made by this tool."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

MODELS = ("qwen3", "qwen25")
EXPECTED = {
    "Qwen3-0.6B-Q8_0.gguf": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
    "qwen2.5-0.5b-instruct-fp16.gguf": "8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc",
    "inputs_qwen25.json": "64b85f2eecd7db09ba323f23935e8c1a4ddff79fa0045bb67e290ff4fcdc9ea4",
    "inputs_qwen3.json": "de22edf77fd41e10e11e6f5549600fb027d09cbd1f1329b981b66e586354a24c",
    "long_qwen25.json": "1f67e810f56cc6c877778ee9e604f544855f0c1523183d66fa24ceb68b4b5812",
    "long_qwen3.json": "ad8be540f48fed37ee41e848164a63d3c2936a2ea933f7ebad776efebfa3728c",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_dump(path, n=40, length=24, gen=32, ids=None, seed=0):
    del seed  # generator is fixed in ir_device_dump; this only checks receipt integrity
    path = Path(path)
    meta = load(str(path) + ".json")
    prompts = load(ids) if ids else None
    count = len(prompts) if ids else n
    if count <= 0 or length <= 0 or gen <= 0 or (ids and any(not x for x in prompts)):
        raise ValueError("empty or invalid dump request")
    vocab = meta.get("vocab")
    if type(vocab) is not int or vocab <= 0 or meta.get("prompts") != count:
        raise ValueError("dump metadata differs from requested prompt inventory")
    rows = 2 if ids else 2 * length
    expected_bytes = count * (rows * vocab + gen) * 4
    if path.stat().st_size != expected_bytes:
        raise ValueError("dump truncated or has extra rows")
    return {"pass": True, "bytes": expected_bytes, "sha256": sha(path), "prompts": count,
            "vocab": vocab, "kind": "long" if ids else "short"}


def gate_labels(phases, formats):
    labels = []
    for model in MODELS:
        if "g1" in phases:
            labels.extend(f"g1 {model} {kind} det={det} base-vs-new"
                          for det in (0, 1) for kind in ("short", "long"))
            labels.append(f"g1 {model} long det=1 new run-to-run")
        if "g2" in phases:
            labels.extend(f"g2 {model} {fmt} {kind} det=1 run-to-run"
                          for fmt in formats for kind in ("long", "short"))
            if "q8" in formats:
                labels.append(f"g2 {model} q8 long det=0 run-to-run")
    return labels


def check_device(data):
    if not (data.get("pass") is True and data.get("on_gpu") is True
            and data.get("device_supported") is True and data.get("prompts") == 12
            and data.get("len") == 24 and data.get("gen") == 32 and len(data.get("rows", [])) == 12
            and data.get("top1_agree") == "288/288" and data.get("greedy_identical") == "12/12"):
        raise ValueError("incomplete or failed CPU/device gate")
    for key in ("worst_prefill_rel", "worst_decode_rel"):
        value = data.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1e-4:
            raise ValueError("device error exceeds fixed 1e-4 gate")
    if {row.get("i") for row in data["rows"]} != set(range(12)):
        raise ValueError("device prompt rows are duplicated or missing")
    for row in data["rows"]:
        if row.get("top1") != 24 or row.get("greedy_first_diff") != -1:
            raise ValueError("device row fails top-1 or greedy equality")
    if data.get("tol") != 1e-4:
        raise ValueError("device tolerance changed")


def compare_teacher(decode, prefill):
    if {r.get("name") for r in decode} != {"book", "chat"} or len(decode) != 2:
        raise ValueError("teacher-forcing input inventory differs")
    prior = {r["name"]: r for r in prefill}
    comparisons = []
    for row in decode:
        if row.get("positions") != 8192:
            raise ValueError("teacher forcing is incomplete")
        contexts = row["formats"]["f16"]["contexts"]
        if set(contexts) != {"2048", "4096", "8192"}:
            raise ValueError("teacher-forcing contexts differ")
        for context, result in contexts.items():
            measured = result["all"]
            ref = prior[row["name"]]["formats"]["f16"]["contexts"][context]["all"]
            if measured.get("positions") != int(context):
                raise ValueError("teacher-forcing rows are missing")
            record = {"input": row["name"], "context": int(context)}
            for key in ("max_rel", "mean_rel", "ppl_rel_diff"):
                value, before = measured[key], ref[key]
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (value, before)):
                    raise ValueError("nonfinite teacher-forcing metric")
                if key != "ppl_rel_diff" and min(value, before) < 0:
                    raise ValueError("negative error metric")
                record[key] = {"decode": value, "prefill": before,
                               "within_prefill_observed": abs(value) <= abs(before)}
            comparisons.append(record)
    return comparisons


def summary(out, phases, formats):
    out = Path(out)
    errors, teacher = [], {}
    try:
        allowed_phases = {"g1", "g2", "g3", "eval_qwen3", "eval_qwen25", "tfd_qwen3", "tfd_qwen25"}
        if (not phases or set(phases) - allowed_phases or len(set(phases)) != len(phases)
                or not formats or set(formats) - {"f16", "q8", "q4"} or len(set(formats)) != len(formats)):
            raise ValueError("empty, duplicate or unknown campaign scope")
        actual = {Path(parts[1]).name: parts[0] for line in (out / "models.sha256").read_text().splitlines()
                  if (parts := line.split())}
        if actual != EXPECTED:
            raise ValueError("model/input hashes differ from recorded experiment")
        for name, digest in EXPECTED.items():
            if name.endswith(".json") and sha(out / name) != digest:
                raise ValueError("input bytes differ from recorded experiment")
        labels = gate_labels(phases, formats)
        lines = (out / "gates.txt").read_text().splitlines()
        found = {}
        for line in lines:
            if not line.startswith(("g1 ", "g2 ")):
                continue
            parts = line.rsplit(" ", 3)
            if len(parts) != 4 or parts[0] in found or parts[1] != "IDENTICAL" or parts[2] != parts[3]:
                raise ValueError("failed, duplicate or ambiguous byte comparison")
            if not re.fullmatch(r"[0-9a-f]{16}", parts[2]):
                raise ValueError("invalid comparison hash")
            found[parts[0]] = parts[2]
        if set(found) != set(labels):
            raise ValueError("missing or extra byte comparison")
        dump_checks = []
        for phase, count in (("g1", 18), ("g2", len(MODELS) * (4 * len(formats) + (2 if "q8" in formats else 0)))):
            checks = list((out / phase).glob("*.check.json"))
            if phase in phases and len(checks) != count:
                raise ValueError("missing dump integrity receipts")
            dump_checks.extend(checks)
        for path in dump_checks:
            check = load(path)
            if check.get("pass") is not True or check.get("bytes", 0) <= 0:
                raise ValueError("invalid dump integrity receipt")
        if "g3" in phases:
            for model in MODELS:
                for fmt in ["f32", *formats]:
                    check_device(load(out / "g3" / f"{model}_{fmt}.json"))
        reference = Path(__file__).resolve().parents[1] / "experiments/kv_quant_r2_2026-10-28/receipts/colab_l4_eval"
        for model in MODELS:
            if f"tfd_{model}" in phases or f"eval_{model}" in phases:
                if load(out / f"eval_{model}/meta.json").get("device") != "cuda":
                    raise ValueError("teacher-forcing/evaluation did not run on CUDA")
            if f"tfd_{model}" in phases:
                teacher[model] = compare_teacher(load(out / f"eval_{model}/tf_decode.json"),
                                                load(reference / f"eval_{model}/tf_prefill.json"))
                if any(not metric["within_prefill_observed"] for row in teacher[model]
                       for key, metric in row.items() if key not in ("input", "context")):
                    errors.append(f"{model}: decode metric exceeds corresponding prefill observation")
            if f"eval_{model}" in phases:
                for phase in ("bench", "greedy", "tf_prefill"):
                    if not load(out / f"eval_{model}/{phase}.json"):
                        raise ValueError("empty evaluation receipt")
        if "tfd_qwen25" in phases:
            bench = load(out / "eval_qwen25_8064/bench.json")
            if len(bench) != 1 or bench[0].get("ctx") != 8064 or bench[0].get("steps") != 128:
                raise ValueError("missing Qwen2.5 8064-position bench")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(str(exc))
    return {"pass": not errors, "errors": errors, "formats": formats, "phases": phases,
            "teacher_forcing_comparison": teacher}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dump = sub.add_parser("dump")
    dump.add_argument("path")
    dump.add_argument("--n", type=int, default=40)
    dump.add_argument("--len", dest="length", type=int, default=24)
    dump.add_argument("--gen", type=int, default=32)
    dump.add_argument("--seed", type=int, default=0)
    dump.add_argument("--ids")
    report = sub.add_parser("summary")
    report.add_argument("path")
    report.add_argument("--phases", required=True)
    report.add_argument("--formats", required=True)
    args = vars(parser.parse_args())
    command, path = args.pop("command"), args.pop("path")
    result = check_dump(path, **args) if command == "dump" else summary(
        path, args["phases"].split(","), args["formats"].split(","))
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
