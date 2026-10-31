"""Record the completed CPU/compile-only integration validation, never launch work.

The canonical raw compiler objects and row dumps remain in the external check
folder. This records their identities and checks the actual finite dump shapes.
No device result or adoption claim is inferred from these checks.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = Path("/home/jonat/ember_e10_integration_checks_20261031")
BASE = Path("/home/jonat/ember_publication_20261030")
KERAMOS = Path("/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def command(args):
    return subprocess.check_output(args, text=True).strip()


def main():
    expected_base = "c95c557a8408c01c7b2aaa8c7ee63af3874f310b"
    assert command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]) == expected_base
    cases = ET.parse(OUT / "pytest_final.xml").getroot().findall("testsuite")
    assert sum(int(c.attrib["tests"]) for c in cases) == 54
    assert all(int(c.attrib[k]) == 0 for c in cases for k in ("errors", "failures", "skipped"))
    conform = load(OUT / "conformance_final.json")
    assert conform["counts"] == {"PASS": 554, "FAIL": 0, "SKIP": 44, "XFAIL": 0, "XPASS": 0}
    numerical = load(OUT / "ir_numerics_final.json")
    assert numerical["pass"] and numerical["failures"] == 0
    state = load(OUT / "state_templates_final.log")
    assert state["pass"] and all(case["pass"] for case in state.values() if isinstance(case, dict))
    http = load(OUT / "cpu_http_final/result.json")
    assert http["status"] == "PASS_CPU_ONLY"
    for name in http["runs"]:
        health = load(OUT / "cpu_http_final" / name / "health.json")
        rows = (OUT / "cpu_http_final" / name / "rows.txt").read_text().splitlines()
        assert len(rows) == 21
        assert all(len(row.split()) == health["vocab_size"] == 16384 for row in rows)
        assert sha(OUT / "cpu_http_final" / name / "rows.f32") == http["row_evidence"][name]["sha256"]
    sass = load(OUT / "sass_identity.json")
    assert sass["existing_functions_equal"] == 42 and len(sass["added_functions"]) == 2
    cross = load(OUT / "component_integration_sass.json")
    assert not any(v["instructions_identical"] for v in cross["kernels"].values())
    sources = ["api.cpp", "model.cpp", "model.h", "verify_kernels.cu", "verify_kernels.h",
               "decode_pick_policy.h", "row_pick_partitioned.cuh", "tools/row_pick_profile/test_integration.py",
               "tools/row_pick_profile/check_integration_sass.py", "tools/row_pick_profile/profile.cu",
               "tools/row_pick_profile/prepare.py", "experiments/spec_r1_2026-10-28/test_decode_pick.py",
               "experiments/spec_r1_2026-10-28/cpu_pick_integration_check.py"]
    logs = ["cmake_cpu.log", "build_cpu_final.log", "build_baseline_cpu.log", "ctest_final.log",
            "pytest_final.log", "pytest_final.xml", "ruff_final.log", "sampler_final.log",
            "ir_numerics_final.json", "state_templates_final.log", "conformance_final.json",
            "conformance_final.log", "conformance_invocation_path_failure.log", "nvcc_baseline.log",
            "nvcc_candidate.log", "sass_identity.json", "component_integration_sass.json",
            "archived_component_identity.json", "cuda_host_final_model.cpp.log", "cuda_host_final_api.cpp.log",
            "cpu_host_final_model.cpp.log", "cpu_host_final_api.cpp.log"]
    for name in logs:
        shutil.copyfile(OUT / name, HERE / name)
    shutil.copyfile(OUT / "cpu_http_final/result.json", HERE / "cpu_http_result.json")
    for name in http["runs"]:
        shutil.copyfile(OUT / "cpu_http_final" / name / "requests.json", HERE / (name + "_requests.json"))
    receipt = {
      "status": "CPU_AND_COMPILE_ONLY_INTEGRATION_COMPLETE",
      "base_commit": expected_base,
      "source_sha256": {name: sha(ROOT / name) for name in sources},
      "commands_source_sha256": sha(__file__),
      "raw_artifact_home": str(OUT),
      "checks": {"pytest": {"pass": 54, "fail": 0}, "ctest": {"pass": 3, "fail": 0},
                 "ir_conformance": conform["counts"], "native_numerical_cases": len(numerical["cases"]),
                 "state_template_cases": sum(isinstance(case, dict) for case in state.values()), "cpu_http": http,
                 "cuda_host_syntax": ["model.cpp", "api.cpp"], "cpu_host_syntax": ["model.cpp", "api.cpp"],
                 "cuda_compile_only": "CUDA 12.6 SM89, verify_kernels.cu baseline and candidate",
                 "existing_kernel_sass": sass, "cross_toolchain_candidate_comparison": cross,
                 "sampler": "19,830 compact decisions and 5,370 fallbacks; ids and RNG state equal"},
      "build_environment": {"cxx": command(["g++", "--version"]),
                            "nvcc": command(["/usr/local/cuda/bin/nvcc", "--version"]),
                            "python": sys.version,
                            "keramos_commit": command(["git", "-C", str(KERAMOS), "rev-parse", "HEAD"]),
                            "keramos_runner_sha256": sha(KERAMOS / "tools/ir_conformance.py"),
                            "keramos_manifest_sha256": sha(KERAMOS / "conformance/manifest.json")},
      "review": {"independent_reviewer": "extension_run agent", "production_blocker": False,
                 "action": "restored five Unicode comment lines; retained explicit diagnostic lookups at positions 14-16"},
      "limits": {"new_gpu_execution": False, "new_paid_calls": False, "default_changed": False,
                 "adoption": "NOT_ADOPTED", "historical_default_off_off_stable": False,
                 "historical_paired_speedups": [1.15925, 1.14146], "unchanged_speed_threshold": 1.15,
                 "new_full_model_gain": None,
                 "experimental_route_requires_deterministic_mode": True,
                 "original_default_and_deterministic_matrix_replaced": False,
                 "next_device_work": "separately reviewed integration diagnostic only; original full adoption gate remains unresolved"},
      "technical_history": "Initial conformance invocation used a nonexistent Keramos publication path. Its error log is retained; corrected and final invocations both passed the complete current suite.",
      "artifacts": {name: sha(HERE / name) for name in logs}}
    receipt["artifacts"]["cpu_http_result.json"] = sha(HERE / "cpu_http_result.json")
    with (HERE / "validation.json").open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(HERE / "validation.json")}))


if __name__ == "__main__":
    main()
