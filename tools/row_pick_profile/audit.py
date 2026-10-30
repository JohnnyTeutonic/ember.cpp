"""Independent local audit of the supervised E10 component archive; no GPU/API calls."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(archive, name):
    return json.loads(archive.read(name))


def audit_trials(rows, baseline):
    expected = {
        "correctness": {
            (v, n)
            for v in [1, 31, 257, 4096, 16384, 32000, 32768, 65536, 128256]
            for n in [1, 4, 16]
        },
        "kernel_trial": {
            (v, n, p, b)
            for v in [4096, 16384, 32000, 32768, 65536, 128256]
            for n in [1, 4, 16]
            for p in ([0] if baseline else [0, 4, 16, 64])
            for b in range(10)
        },
        "host_trial": {
            (v, 1, p, b, t)
            for v in [4096, 16384, 32000, 32768, 65536, 128256]
            for p in ([0] if baseline else [0, 4, 16, 64])
            for b in range(10)
            for t in [0, 1]
        },
    }
    fields = {
        "correctness": ["vocab", "n"],
        "kernel_trial": ["vocab", "n", "parts", "batch"],
        "host_trial": ["vocab", "n", "parts", "batch", "transfer_only"],
    }
    found = {kind: set() for kind in expected}
    for row in rows:
        kind = row["kind"]
        if kind in found:
            key = tuple(row[f] for f in fields[kind])
            if key not in expected[kind] or key in found[kind]:
                raise ValueError("unexpected/duplicate raw trial identity")
            found[kind].add(key)
        if kind in {"kernel_trial", "host_trial"}:
            if not math.isfinite(row["microseconds"]) or row["microseconds"] <= 0:
                raise ValueError("invalid raw timing")
        if kind == "kernel_trial" and (
            row["iterations"] != 1000 or row["raw_elapsed_ms"] != row["microseconds"]
        ):
            raise ValueError("raw kernel timing units inconsistent")
    complete = bool(rows and rows[-1].get("kind") == "COMPLETE")
    if complete:
        if found != expected or rows[-1]["checked_rows"] != (
            2268 if baseline else 6804
        ):
            raise ValueError("incomplete full inventory")
        allowed = set(expected) | {"device", "COMPLETE"}
        if any(r["kind"] not in allowed for r in rows):
            raise ValueError("unexpected record in complete leg")
        if (
            sum(r["kind"] == "device" for r in rows) != 1
            or sum(r["kind"] == "COMPLETE" for r in rows) != 1
        ):
            raise ValueError("bad boundary record count")
    elif any(found[k] for k in ["kernel_trial", "host_trial"]):
        raise ValueError("candidate timing accepted before complete parity matrix")
    device = rows[0]
    if (
        device.get("kind") != "device"
        or device.get("name") != "NVIDIA L4"
        or device.get("sm") != 89
    ):
        raise ValueError("raw device identity mismatch")
    return {
        "complete": complete,
        "counts": {k: len(v) for k, v in found.items()},
        "first_mismatch": next((r for r in rows if r["kind"] == "MISMATCH"), None),
    }


def summarize(rows):
    groups = {}
    for row in rows:
        if row["kind"] in {"kernel_trial", "host_trial"}:
            key = (
                row["kind"],
                row["vocab"],
                row["n"],
                row["parts"],
                row.get("transfer_only"),
            )
            groups.setdefault(key, []).append(row["microseconds"])
    return [
        {
            "kind": k[0],
            "vocab": k[1],
            "n": k[2],
            "parts": k[3],
            "transfer_only": k[4],
            "trials": len(v),
            "median_us": statistics.median(v),
            "min_us": min(v),
            "max_us": max(v),
        }
        for k, v in sorted(groups.items())
    ]


def audit(path, stage):
    trusted = json.loads((stage / "e10_component_manifest.json").read_text())
    report = {
        "archive_sha256": digest(path.read_bytes()),
        "archive_bytes": path.stat().st_size,
        "manifest_sha256": digest((stage / "e10_component_manifest.json").read_bytes()),
        "scope": "component only; no full-model speed/adoption inference",
        "legs": {},
    }
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive CRC failure")
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            n.startswith("/") or ".." in Path(n).parts for n in names
        ):
            raise ValueError("unsafe/duplicate archive path")
        if read_json(archive, "manifest.json") != trusted:
            raise ValueError("manifest differs from prelaunch frozen stage")
        if digest(archive.read("vm.py")) != trusted["vm_sha256"]:
            raise ValueError("VM wrapper differs from frozen stage")
        for relative, expected in trusted["source_sha256"].items():
            if digest(archive.read("source_tree/" + relative)) != expected:
                raise ValueError("scientific source differs: " + relative)
        summary = read_json(archive, "summary.json")
        report["summary"] = summary
        for leg in ["baseline", "candidate"]:
            if leg + "/receipt.json" not in names:
                if leg == "baseline":
                    raise ValueError("original baseline absent")
                continue
            receipt = read_json(archive, leg + "/receipt.json")
            if bool(receipt["baseline_only"]) != (leg == "baseline"):
                raise ValueError("wrong leg mode")
            for relative, expected in receipt["artifacts"].items():
                if relative == "inputs.bin":
                    continue  # Explicitly hashed generated scratch, not retained/relayed.
                data = archive.read(leg + "/" + relative)
                if digest(data) != expected["sha256"] or len(data) != expected["bytes"]:
                    raise ValueError("artifact hash mismatch: " + leg + "/" + relative)
            if receipt["source_sha256"] != {
                p: h
                for p, h in trusted["source_sha256"].items()
                if not p.endswith("test_profile.py")
            }:
                raise ValueError("leg source inventory differs")
            for relative, expected in receipt["source_sha256"].items():
                if digest(archive.read(leg + "/source/" + relative)) != expected:
                    raise ValueError("archived leg source differs")
            extraction = read_json(archive, leg + "/extraction.json")
            header = archive.read(leg + "/selector.cuh")
            original_source = archive.read("source_tree/verify_kernels.cu").decode()
            start = original_source.index("__global__ void select_rows_kernel(")
            opening = original_source.index("{", start)
            end, depth = opening + 1, 1
            while depth:
                depth += (original_source[end] == "{") - (original_source[end] == "}")
                end += 1
            body = original_source[start:end].encode()
            if (
                extraction["production_source_sha256"]
                != trusted["source_sha256"]["verify_kernels.cu"]
                or extraction["original_kernel_sha256"] != digest(body)
                or extraction["generated_header_sha256"] != digest(header)
                or body not in header
            ):
                raise ValueError("original kernel extraction differs")
            rows = [
                json.loads(line)
                for line in archive.read(leg + "/trials.jsonl").decode().splitlines()
            ]
            checks = audit_trials(rows, leg == "baseline")
            if checks["complete"] != (receipt["status"] == "COMPLETE"):
                raise ValueError("receipt/raw completion mismatch")
            if summary.get(leg) != receipt["status"]:
                raise ValueError("VM/leg status mismatch")
            if receipt["gpu"]["name"] != "NVIDIA L4" or not receipt["gpu"][
                "uuid"
            ].startswith("GPU-"):
                raise ValueError("GPU UUID missing")
            report["legs"][leg] = {
                "status": receipt["status"],
                "gpu": receipt["gpu"],
                "checks": checks,
                "timings": summarize(rows),
                "input_capture": receipt["artifacts"].get("inputs.bin"),
                "error": receipt.get("error"),
            }
        if all(
            report["legs"].get(leg, {}).get("status") == "COMPLETE"
            for leg in ["baseline", "candidate"]
        ):
            if (
                report["legs"]["baseline"]["input_capture"]
                != report["legs"]["candidate"]["input_capture"]
            ):
                raise ValueError("complete baseline/candidate synthetic inputs differ")
            if report["legs"]["baseline"]["gpu"] != report["legs"]["candidate"]["gpu"]:
                raise ValueError("baseline/candidate GPU identity differs")
        if report["legs"].get("baseline", {}).get("status") != "COMPLETE":
            report["interpretation"] = (
                "Original baseline did not complete; no performance conclusion."
            )
        elif report["legs"].get("candidate", {}).get("status") == "HALTED":
            report["interpretation"] = (
                "Original baseline measured; candidate halted; no valid candidate speed comparison/adoption."
            )
        else:
            report["interpretation"] = (
                "Component measurements only; unchanged full-model gates remain required."
            )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.archive, args.stage)
    report["audit_source_sha256"] = digest(Path(__file__).read_bytes())
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "legs": {k: v["status"] for k, v in report["legs"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
