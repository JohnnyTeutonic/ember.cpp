"""vm_gemma.sh's gate bookkeeping with stub tools on PATH (no GPU, network or real conversion).
python3 -m unittest test_vm_gemma -v"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCE = "5bfda89345b3bcf1f0352eee555f0fd4eb73ac7d"
CANONICAL = "0a174d42ccd7de15688e50dc044ea8006db31ee3bc6dc7e838f8a360ca46cdaa"
CONTROLLER = "ddd3104489dd2935187271d48dfabe77501a2b3c5fa02350aad1ca5e54a2f04c"
REQUIRED = {"analyze", "benchmark", "build", "checkout", "contract", "inputs", "packages", "provenance", "regenerate",
            "weights"}

STUBS = {
    "git": r'''#!/usr/bin/env bash
if [ "$1" = clone ]; then mkdir -p "${@: -1}/experiments/hf_cuda_2026-10-30/gemma_colab_20261103"; [ "$FAIL_AT" = clone ] && exit 7; exit 0; fi
dir=.; if [ "$1" = -C ]; then dir=$2; shift 2; fi
case "$1" in
  worktree) mkdir -p "${@: -2:1}/experiments/hf_cuda_2026-10-30"; echo x > "${@: -2:1}/experiments/hf_cuda_2026-10-30/benchmark.py";;
  rev-parse) case "$dir" in *src) echo 5bfda89345b3bcf1f0352eee555f0fd4eb73ac7d;; *) echo "${STUB_HEAD:-0123456789abcdef0123456789abcdef01234567}";; esac;;
  status) true;;
  *) [ "$FAIL_AT" = checkout ] && exit 5; true;;
esac''',
    "sha256sum": r'''#!/usr/bin/env bash
if [ "$1" = -c ]; then [ "$FAIL_AT" = "manifest_$(basename $2 .sha256)" ] && exit 1; exit 0; fi
for f in "$@"; do
  case "$f" in
    *benchmark.py) echo "ddd3104489dd2935187271d48dfabe77501a2b3c5fa02350aad1ca5e54a2f04c  $f";;
    *.gguf) [ "$FAIL_AT" = regenerate ] && echo "ffff$(printf '%060d' 0)  $f" || echo "0a174d42ccd7de15688e50dc044ea8006db31ee3bc6dc7e838f8a360ca46cdaa  $f";;
    *) echo "1111111111111111111111111111111111111111111111111111111111111111  $f";;
  esac
done''',
    "nvidia-smi": "#!/usr/bin/env bash\necho 'NVIDIA L4'",
    "apt-get": "#!/usr/bin/env bash\nexit 0",
    "nproc": "#!/usr/bin/env bash\necho 2",
    "cmake": r'''#!/usr/bin/env bash
[ "$FAIL_AT" = build ] && exit 4
echo "-- CUDA toolkit and cuBLAS found"''',
    "make": "#!/usr/bin/env bash\nmkdir -p build && echo bin > build/ir_hf_cuda_bench",
    "tar": "#!/usr/bin/env bash\nmkdir -p conv/converter && echo x > conv/converter/convert_hf_to_gguf.py",
    "cp": "#!/usr/bin/env bash\n[ \"$FAIL_AT\" = copy ] && exit 3; exit 0",
    "python3": r'''#!/usr/bin/env bash
case "$*" in
  *"pip install"*) exit 0;;
  *"pip list"*) echo '[]'; exit 0;;
  *convert_hf_to_gguf.py*) for ((i=1; i<=$#; i++)); do [ "${!i}" = --outfile ] && { j=$((i+1)); echo gguf > "${!j}"; }; done; exit 0;;
  *launch_contract_gemma.py*) [ "$FAIL_AT" = contract ] && exit 1; exit 0;;
  *benchmark.py*--analyze*) echo '{"pass": true}'; exit 0;;
  *benchmark.py*) [ "$FAIL_AT" = benchmark ] && exit 1; exit 0;;
  "- "*) cat > /dev/null; [ "$FAIL_AT" = weights ] && [ "$2" != *provenance* ] && grep -q . /dev/null; exit 0;;
esac
exit 0''',
}


class VmGemmaTests(unittest.TestCase):
    def run_vm(self, fail_at="", run="g1", head=None):
        tmp = Path(tempfile.mkdtemp())
        b = tmp / "bin"
        b.mkdir()
        for name, body in STUBS.items():
            (b / name).write_text(body + "\n")
            (b / name).chmod(0o755)
        (tmp / f"upload_{run}").mkdir()  # the driver creates it before launch
        env = dict(os.environ, PATH=f"{b}:/usr/bin:/bin", VM_TEST_HARNESS="1", VM_ROOT=str(tmp), FAIL_AT=fail_at)
        if head:
            env["STUB_HEAD"] = head
        r = subprocess.run(["bash", str(HERE / "vm_gemma.sh"), COMMIT, run], env=env, capture_output=True, text=True,
                           timeout=60)
        out = tmp / f"run_{run}" / "out"
        status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else None
        done = (out / "DONE").read_text().strip() if (out / "DONE").exists() else None
        return r.returncode, status, done, r.stderr

    def test_success_records_exactly_the_required_gates(self):
        rc, s, done, err = self.run_vm()
        self.assertEqual(rc, 0, (s, err))
        self.assertEqual(done, "g1")
        self.assertTrue(s["pass"])
        self.assertEqual(set(s["gates"]), REQUIRED)
        self.assertEqual((s["commit"], s["head"], s["source"], s["exit_status"]), (COMMIT, COMMIT, SOURCE, 0))

    def test_no_forward_on_a_non_canonical_gguf(self):
        rc, s, done, _ = self.run_vm("regenerate")
        self.assertEqual(rc, 1)
        self.assertEqual(s["gates"]["regenerate"], 96)
        self.assertNotIn("contract", s["gates"])
        self.assertNotIn("benchmark", s["gates"])
        self.assertEqual(done, "g1")

    def test_prerequisite_failures_stop_with_their_own_status(self):
        for fail_at, gate, code in (("checkout", "checkout", 5), ("build", "build", 4), ("manifest_inputs", "inputs", 1),
                                    ("manifest_weights", "weights", 1), ("contract", "contract", 1)):
            rc, s, done, _ = self.run_vm(fail_at)
            self.assertEqual(rc, 1, fail_at)
            self.assertEqual(s["gates"][gate], code, fail_at)
            self.assertNotIn("benchmark", s["gates"], fail_at)
            self.assertFalse(s["pass"], fail_at)

    def test_benchmark_failure_is_recorded_and_analysis_still_runs(self):
        rc, s, _, _ = self.run_vm("benchmark")
        self.assertEqual(rc, 1)
        self.assertNotEqual(s["gates"]["benchmark"], 0)
        self.assertEqual(s["gates"]["analyze"], 0)
        self.assertFalse(s["pass"])

    def test_wrong_head_fails_checkout(self):
        rc, s, _, _ = self.run_vm(head="f" * 40)
        self.assertEqual((rc, s["gates"]["checkout"]), (1, 99))

    def test_bad_arguments_refused(self):
        r = subprocess.run(["bash", str(HERE / "vm_gemma.sh"), "main", "g1"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
