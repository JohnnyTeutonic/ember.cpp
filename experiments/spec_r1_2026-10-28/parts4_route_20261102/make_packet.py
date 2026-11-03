"""Write packet.json: the source, collector and input pins parts4_vm.py enforces. Run from this folder after any
change to the instrumented source or the collector, then commit both.

  python3 make_packet.py [--check]
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKET = HERE.relative_to(ROOT).as_posix()
SOURCE = ["api.cpp", "model.cpp", "model.h", "verify_kernels.cu", "verify_kernels.h", "decode_route_trace.h",
          "decode_pick_policy.h", "row_pick_partitioned.cuh", "determinism.h", "speculative_verify.cpp", "server.cpp",
          "CMakeLists.txt"]
COLLECTOR = [PACKET + "/" + n for n in ("parts4_vm.py", "timing_gate.py", "parts4_gate.py", "parts4_route_matrix.py",
                                         "test_parts4_gate.py", "test_source_reduction.py", "test_route_trace.cpp")] + [
    "experiments/spec_r1_2026-10-28/decode_pick_gate.py", "experiments/spec_r1_2026-10-28/decode_pick_fallback.py",
    "experiments/spec_r1_2026-10-28/decode_pick_rows.cpp", "experiments/draft_2026-10-27/dr_e2e.py",
    "experiments/spec_r1_2026-10-28/parts4_full_model_protocol_20261102.json",
    "tools/row_pick_profile/test_integration.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet():
    sys.path.insert(0, str(HERE))
    import parts4_gate
    protocol = json.loads((ROOT / "experiments/spec_r1_2026-10-28/parts4_full_model_protocol_20261102.json").read_text())
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider",
                            "test_source_reduction.py", "test_parts4_gate.py"], cwd=HERE, capture_output=True,
                           text=True, check=True).stdout
    count = int(re.search(r"(\d+) tests? collected", tests).group(1))
    return {
        "kind": "E10_PARTS4_ROUTE_EVIDENCE_AND_FROZEN_COLLECTOR_V1",
        "status": "PREPARED_UNADMITTED",
        "protocol": {"path": "experiments/spec_r1_2026-10-28/parts4_full_model_protocol_20261102.json",
                     "commit": "41b6cd5a006ecb828523cee43756f05af9d0835f",
                     "sha256": sha(ROOT / "experiments/spec_r1_2026-10-28/parts4_full_model_protocol_20261102.json"),
                     "candidate_source_design_basis": protocol["candidate_source"],
                     "candidate_source_files_sha256": protocol["source_files_sha256"]},
        "instrumentation": ("Trace-only additions behind EMBER_ROUTE_TRACE; removing them reproduces the protocol "
                            "source byte for byte (test_source_reduction.py). The selector kernels, the policy header "
                            "and api.cpp's dispatch are unchanged."),
        "source_files_sha256": {n: sha(ROOT / n) for n in SOURCE},
        "frozen_collector_sha256": {n: sha(ROOT / n) for n in COLLECTOR},
        "input_pins": {"model_sha256": protocol["input_pins"]["model_sha256"],
                       "prompts_sha256": protocol["input_pins"]["prompts_sha256"],
                       "items": protocol["input_pins"]["items"], "requests": protocol["input_pins"]["requests"]},
        "schedule": [leg[0] for leg in parts4_gate.LEGS],
        "timing_pairs": list(parts4_gate.PAIRS),
        "thresholds": parts4_gate.THRESHOLDS,
        "packet_test_count": count,
        "admission": {"authorized": False, "gpu_or_provider_calls": 0, "gpu": "one L4",
                      "owner": "Codex (serial GPU queue, financial cap, deadline, watchdog, home replay, cleanup)"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if packet.json is stale")
    value = json.dumps(packet(), indent=2) + "\n"
    path = HERE / "packet.json"
    if ap.parse_args().check:
        if not path.exists() or path.read_text() != value:
            sys.exit("packet.json is stale: run make_packet.py")
        print("packet.json current")
        return
    path.write_text(value)
    print(sha(path), path.name)


if __name__ == "__main__":
    main()
