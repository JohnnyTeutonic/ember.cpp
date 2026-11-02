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
SOURCE = "4d276ff91ab9b191be8556010b05d74cd4cb2520"
CANONICAL = "0a174d42ccd7de15688e50dc044ea8006db31ee3bc6dc7e838f8a360ca46cdaa"
CONTROLLER = "ddd3104489dd2935187271d48dfabe77501a2b3c5fa02350aad1ca5e54a2f04c"
REQUIRED = {"analyze", "benchmark", "build", "checkout", "contract", "environment", "inputs", "packages", "provenance",
            "regenerate", "toolkit", "weights"}

STUBS = {
    "git": r'''#!/usr/bin/env bash
if [ "$1" = clone ]; then d="${@: -1}/experiments/hf_cuda_2026-10-30/gemma_colab_20261103"; mkdir -p "$d"; cat "$PACKET_DIR/apt_reconcile.py" > "$d/apt_reconcile.py"; [ "$FAIL_AT" = clone ] && exit 7; exit 0; fi
dir=.; if [ "$1" = -C ]; then dir=$2; shift 2; fi
case "$1" in
  worktree) mkdir -p "${@: -2:1}/experiments/hf_cuda_2026-10-30"; echo x > "${@: -2:1}/experiments/hf_cuda_2026-10-30/benchmark.py";;
  rev-parse) case "$dir" in *src) case "$2" in *tree*) echo "${STUB_SOURCE_TREE:-ee183b61772737b9b6f659ddc25f34d1a75ba332}";; *) echo 4d276ff91ab9b191be8556010b05d74cd4cb2520;; esac;; *) echo "${STUB_HEAD:-0123456789abcdef0123456789abcdef01234567}";; esac;;
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
    "apt-get": r'''#!/usr/bin/env bash
echo "apt-get $*" >> "$VM_ROOT/apt.calls"
if [ "$1" = update ] && grep -qs '^deb ' "$VM_APT_ETC/sources.list.d/cuda-old.list"; then
  echo "E: Conflicting values set for option Signed-By"; exit 100; fi
exit 0''',
    # apt-cache: the pinned versions are offered by the image's configuration unless a scenario says otherwise;
    # in the reconciliation scenarios only after the keyring package is installed
    "apt-cache": r'''#!/usr/bin/env bash
case "$APT_SCENARIO" in
  conflict|commented|unoffered)
    [ "$APT_SCENARIO" != unoffered ] && grep -qs "^deb " "$VM_APT_ETC/sources.list.d/vm-gemma-cuda-12-6.list" || exit 0;;
esac
[ "$FAIL_AT" = toolkit ] && exit 0
[ "$APT_SCENARIO" = nvtx_missing ] && [ "$2" = cuda-nvtx-12-6 ] && exit 0
echo " $2 | 12.6.3-1 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 Packages"
echo " $2 | 12.6.4.1-1 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 Packages"
echo " $2 | 12.6.77-1 | https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 Packages"''',
    "curl": "#!/usr/bin/env bash\n[ \"$FAIL_AT\" = toolkit ] && exit 22; while [ $# -gt 0 ]; do [ \"$1\" = -o ] && echo deb > \"$2\"; shift; done; exit 0",
    "dpkg": r'''#!/usr/bin/env bash
[ "$1" = -l ] && { echo 'ii  cuda-minimal-build-12-6 12.6.3-1'; exit 0; }
echo "dpkg $*" >> "$VM_ROOT/apt.calls"; exit 0''',
    "dpkg-deb": r'''#!/usr/bin/env bash
echo "dpkg-deb $*" >> "$VM_ROOT/apt.calls"
[ "$1" = -x ] && mkdir -p "$3/usr/share/keyrings" && echo key > "$3/usr/share/keyrings/cuda-archive-keyring.gpg"''',
    "ldd": "#!/usr/bin/env bash\necho '\tlibcublas.so.12 => /usr/local/cuda-12.6/lib64/libcublas.so.12'",
    "nproc": "#!/usr/bin/env bash\necho 2",
    "cmake": r'''#!/usr/bin/env bash
[ "$FAIL_AT" = build ] && exit 4
root=""; for a in "$@"; do case "$a" in -DCUDA_TOOLKIT_ROOT_DIR=*) root=${a#*=};; esac; done
# FindCUDA reports the root it used; without CUDA_TOOLKIT_ROOT_DIR it falls back to /usr/local/cuda (CUDA 13 on Colab)
[ -n "$root" ] && echo "-- Found CUDA: $root (found version \"${STUB_FINDCUDA_VERSION:-12.6}\")" \
  || echo '-- Found CUDA: /usr/local/cuda (found version "13.0")'
echo "-- CUDA toolkit and cuBLAS found"''',
    "make": "#!/usr/bin/env bash\nmkdir -p build && echo bin > build/ir_hf_cuda_bench",
    "tar": "#!/usr/bin/env bash\nmkdir -p conv/converter && echo x > conv/converter/convert_hf_to_gguf.py",
    "cp": "#!/usr/bin/env bash\n[ \"$FAIL_AT\" = copy ] && exit 3; exit 0",
    "python3": r'''#!/usr/bin/env bash
case "$*" in
  *"pip install"*) exit 0;;
  *"pip list"*) echo '[]'; exit 0;;
  *convert_hf_to_gguf.py*) for ((i=1; i<=$#; i++)); do [ "${!i}" = --outfile ] && { j=$((i+1)); echo gguf > "${!j}"; }; done; exit 0;;
  *launch_contract_gemma.py*--binary*) [ "$FAIL_AT" = contract ] && exit 1; exit 0;;
  *launch_contract_gemma.py*) [ "$FAIL_AT" = environment ] && exit 1; exit 0;;
  "-c "*) [ "$FAIL_AT" = imports ] && exit 1; exit 0;;
  *benchmark.py*--analyze*) echo '{"pass": true}'; exit 0;;
  *benchmark.py*) [ "$FAIL_AT" = benchmark ] && exit 1
    # a native-like writer: stamps the time (ns) every 50 ms until it is killed
    case "$FAIL_AT" in slow_controller|slow_setsid|leftover_writer)
      trap "" INT
      (while :; do date +%s%N > "$VM_ROOT/hb.tmp" && mv "$VM_ROOT/hb.tmp" "$VM_ROOT/heartbeat"; sleep 0.05; done) & echo $! > "$VM_ROOT/native_child.pid";;
    esac
    case "$FAIL_AT" in slow_controller|slow_setsid) wait;; esac
    exit 0;;
  "- "*) cat > /dev/null; [ "$FAIL_AT" = weights ] && [ "$2" != *provenance* ] && grep -q . /dev/null; exit 0;;
esac
exit 0''',
    # the real setsid, optionally slow to start: the window in which the controller's group does not exist yet
    "setsid": r'''#!/usr/bin/env bash
[ "$FAIL_AT" = slow_setsid ] && sleep 3
exec /usr/bin/setsid "$@"''',
}


def make_tools(tmp):
    """Stub tools on PATH, and a fake CUDA 12.6 toolkit for the build (VM_CUDA_HOME, test harness only)."""
    b = tmp / "bin"
    b.mkdir()
    for name, body in STUBS.items():
        (b / name).write_text(body + "\n")
        (b / name).chmod(0o755)
    nvcc = tmp / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text('#!/usr/bin/env bash\necho "Cuda compilation tools, release ${STUB_NVCC_RELEASE:-12.6}, V12.6.85"\n')
    nvcc.chmod(0o755)
    (tmp / "apt" / "sources.list.d").mkdir(parents=True)
    (tmp / "apt" / "sources.list").write_text("deb http://archive.ubuntu.com/ubuntu noble main\n")
    keyrings = tmp / "keyrings"
    keyrings.mkdir()
    return dict(os.environ, PATH=f"{b}:/usr/bin:/bin", VM_TEST_HARNESS="1", VM_ROOT=str(tmp),
                VM_CUDA_HOME=str(tmp / "cuda"), VM_APT_ETC=str(tmp / "apt"), VM_KEYRINGS=str(keyrings),
                VM_SYS_PYTHON="/usr/bin/python3", PACKET_DIR=str(HERE))


def alive(pid):
    """True for a live, non-zombie process."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


class VmGemmaTests(unittest.TestCase):
    def run_vm(self, fail_at="", run="g1", head=None, extra=None, prepare=None, with_tmp=False):
        tmp = Path(tempfile.mkdtemp())
        (tmp / f"upload_{run}").mkdir()  # the driver creates it before launch
        env = dict(make_tools(tmp), FAIL_AT=fail_at, **(extra or {}))
        if prepare:
            prepare(tmp)
        if head:
            env["STUB_HEAD"] = head
        r = subprocess.run(["bash", str(HERE / "vm_gemma.sh"), COMMIT, run], env=env, capture_output=True, text=True,
                           timeout=60)
        out = tmp / f"run_{run}" / "out"
        status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else None
        done = (out / "DONE").read_text().strip() if (out / "DONE").exists() else None
        if with_tmp:
            return r.returncode, status, done, tmp
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
                                    ("manifest_weights", "weights", 1), ("contract", "contract", 1),
                                    ("environment", "environment", 1), ("imports", "packages", 1)):
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

    def test_science_marker_only_once_the_controller_starts(self):
        rc, s, _, _ = self.run_vm("contract")
        self.assertFalse(s["science_started"])
        rc, s, _, _ = self.run_vm("benchmark")
        self.assertTrue(s["science_started"])

    def start_vm(self, fail_at, signal_at=""):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "upload_s1").mkdir()
        env = dict(make_tools(tmp), FAIL_AT=fail_at, VM_TEST_SIGNAL_AT=signal_at)
        p = subprocess.Popen(["bash", str(HERE / "vm_gemma.sh"), COMMIT, "s1"], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return p, tmp, tmp / "run_s1" / "out"

    def assert_stopped_before_publication(self, p, tmp, out, child_expected):
        p.wait(timeout=90)
        stop = json.loads((out / "stop.json").read_text())
        self.assertTrue(stop["controller_started"])
        self.assertTrue(stop["controller_group_gone"])
        s = json.loads((out / "status.json").read_text())
        self.assertEqual(s["gates"]["benchmark"], 143)
        self.assertTrue(s["science_started"])
        self.assertFalse(s["pass"])
        self.assertEqual((out / "DONE").read_text().strip(), "s1")
        self.assertEqual(p.returncode, 1)
        child_file = tmp / "native_child.pid"
        if child_expected:
            self.assertTrue(child_file.exists())
        if child_file.exists():
            self.assertFalse(alive(int(child_file.read_text())))  # the native-like writer is gone
        if (tmp / "heartbeat").exists():  # it may have been stopped before its first write
            # it had stopped writing before status.json was published, not merely by the time we looked
            self.assertLess(int((tmp / "heartbeat").read_text()), (out / "status.json").stat().st_mtime_ns)

    def test_stop_during_the_controller_takes_down_its_group_before_status(self):
        import signal
        import time
        p, tmp, out = self.start_vm("slow_controller")
        for _ in range(100):  # wait until the controller (and its child) are running
            if (out / "controller.pid").exists() and (tmp / "heartbeat").exists():
                break
            time.sleep(0.1)
        os.kill(int((tmp / "vm_s1.pid").read_text()), signal.SIGTERM)
        self.assert_stopped_before_publication(p, tmp, out, child_expected=True)

    def test_stop_between_fork_and_registration_is_held_then_delivered(self):
        p, tmp, out = self.start_vm("slow_controller", signal_at="after_fork")
        self.assert_stopped_before_publication(p, tmp, out, child_expected=False)

    def test_stop_before_the_controller_group_exists(self):
        # setsid is slow to start, the signal arrives right after registration: the group does not exist yet
        p, tmp, out = self.start_vm("slow_setsid", signal_at="after_register")
        self.assert_stopped_before_publication(p, tmp, out, child_expected=False)
        self.assertFalse((tmp / "heartbeat").exists())  # the controller never ran

    def test_a_writer_left_behind_by_the_controller_is_killed_and_fails_the_gate(self):
        p, tmp, out = self.start_vm("leftover_writer")
        p.wait(timeout=90)
        s = json.loads((out / "status.json").read_text())
        self.assertEqual(s["gates"]["benchmark"], 97)
        self.assertFalse(s["pass"])
        self.assertTrue((out / "leftover_processes.txt").read_text().strip())
        self.assertFalse(alive(int((tmp / "native_child.pid").read_text())))
        self.assertLess(int((tmp / "heartbeat").read_text()), (out / "status.json").stat().st_mtime_ns)
        self.assertEqual((out / "DONE").read_text().strip(), "s1")

    def test_toolkit_failure_stops_before_the_packages(self):
        # the pinned versions are not offered and the keyring download fails: nothing large was installed
        rc, s, done, _ = self.run_vm("toolkit")
        self.assertEqual((rc, s["gates"]["toolkit"]), (1, 22))
        self.assertNotIn("packages", s["gates"])
        self.assertNotIn("build", s["gates"])
        self.assertFalse(s["science_started"])
        self.assertEqual(done, "g1")

    def test_toolkit_uses_the_image_apt_configuration_when_it_offers_the_pins(self):
        rc, s, _, tmp = self.run_vm(with_tmp=True)
        self.assertEqual((rc, s["gates"]["toolkit"]), (0, 0))
        calls = (tmp / "apt.calls").read_text()
        self.assertNotIn("dpkg-deb", calls)  # no keyring, no change to the sources
        self.assertIn("apt-get install --simulate", calls)
        self.assertFalse((tmp / "apt" / "sources.list.d" / "vm-gemma-cuda-12-6.list").exists())
        self.assertIn("cuda-minimal-build-12-6=12.6.3-1 libcublas-dev-12-6=12.6.4.1-1 cuda-nvtx-12-6=12.6.77-1", calls)
        log = (tmp / "run_g1" / "out" / "toolkit.log").read_text()
        self.assertIn("image's own APT configuration", log)

    def test_toolkit_sets_aside_a_conflicting_nvidia_entry(self):
        repo = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
        other = "deb http://archive.ubuntu.com/ubuntu noble universe\n"

        def prepare(tmp):
            d = tmp / "apt" / "sources.list.d"
            (d / "cuda-old.list").write_text(f"deb [signed-by=/usr/share/keyrings/other.gpg] {repo} /\n")
            (d / "cuda-x.sources").write_text(f"Types: deb\nURIs: {repo}\nSuites: /\nSigned-By: /x.gpg\n")
            (d / "other.list").write_text(other)
        rc, s, _, tmp = self.run_vm(extra={"APT_SCENARIO": "conflict"}, prepare=prepare, with_tmp=True)
        self.assertEqual((rc, s["gates"]["toolkit"]), (0, 0))
        d = tmp / "apt" / "sources.list.d"
        self.assertTrue((d / "cuda-old.list").read_text().startswith("# set aside by vm_gemma.sh: deb "))
        self.assertIn("Enabled: no", (d / "cuda-x.sources").read_text())
        self.assertEqual((d / "other.list").read_text(), other)  # unrelated repositories untouched
        ours = (d / "vm-gemma-cuda-12-6.list").read_text()
        self.assertEqual(ours, f"deb [signed-by={tmp / 'keyrings' / 'vm-gemma-cuda-archive-keyring.gpg'}] {repo} /\n")
        self.assertEqual((tmp / "keyrings" / "vm-gemma-cuda-archive-keyring.gpg").read_text(), "key\n")
        kept = tmp / "run_g1" / "out" / "apt_disabled"  # retrieved with the evidence
        self.assertEqual(sorted(x.name for x in kept.iterdir()), ["cuda-old.list", "cuda-x.sources"])
        calls = (tmp / "apt.calls").read_text()
        self.assertNotIn("dpkg -i", calls)  # no package manages the entry
        self.assertLess(calls.index("dpkg-deb -x"), calls.index("apt-get install -y -q --no-install-recommends"))
        before = (tmp / "run_g1" / "out" / "apt_nvidia_sources_before.txt").read_text()
        self.assertIn("cuda-old.list", before)

    def test_toolkit_restores_an_entry_when_the_package_owned_list_is_commented_out(self):
        # round 6: cuda-keyring already installed, its own list commented out, nothing active for the repository
        repo = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
        own = f"# deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] {repo} /\n"

        def prepare(tmp):
            (tmp / "apt" / "sources.list.d" / "cuda-ubuntu2404-x86_64.list").write_text(own)
        rc, s, _, tmp = self.run_vm(extra={"APT_SCENARIO": "commented"}, prepare=prepare, with_tmp=True)
        self.assertEqual((rc, s["gates"]["toolkit"]), (0, 0))
        d = tmp / "apt" / "sources.list.d"
        self.assertEqual((d / "cuda-ubuntu2404-x86_64.list").read_text(), own)
        self.assertTrue((d / "vm-gemma-cuda-12-6.list").read_text().startswith("deb [signed-by="))

    def test_toolkit_fails_closed_when_the_pins_are_not_offered_after_reconciliation(self):
        rc, s, _, tmp = self.run_vm(extra={"APT_SCENARIO": "unoffered"}, with_tmp=True)
        self.assertEqual((rc, s["gates"]["toolkit"]), (1, 98))
        self.assertNotIn("packages", s["gates"])
        self.assertNotIn("install -y", (tmp / "apt.calls").read_text())

    def test_toolkit_fails_closed_when_only_nvtx_is_not_offered(self):
        rc, s, _, tmp = self.run_vm(extra={"APT_SCENARIO": "nvtx_missing"}, with_tmp=True)
        self.assertEqual((rc, s["gates"]["toolkit"]), (1, 98))
        self.assertNotIn("install -y", (tmp / "apt.calls").read_text())

    def test_a_toolkit_other_than_12_6_fails_the_build(self):
        rc, s, _, _ = self.run_vm(extra={"STUB_NVCC_RELEASE": "13.0"})
        self.assertEqual(rc, 1)
        self.assertNotEqual(s["gates"]["build"], 0)
        self.assertNotIn("benchmark", s["gates"])

    def test_findcuda_on_another_toolkit_fails_the_build(self):
        # g2-20261103: the legacy FindCUDA found /usr/local/cuda (13.0) and its headers broke the 12.6 compile
        rc, s, _, _ = self.run_vm(extra={"STUB_FINDCUDA_VERSION": "13.0"})
        self.assertEqual(rc, 1)
        self.assertNotEqual(s["gates"]["build"], 0)
        self.assertNotIn("benchmark", s["gates"])

    def test_a_source_tree_other_than_the_frozen_one_fails_checkout(self):
        rc, s, _, _ = self.run_vm(extra={"STUB_SOURCE_TREE": "f" * 40})
        self.assertEqual((rc, s["gates"]["checkout"]), (1, 99))

    def test_a_second_launch_of_the_same_run_exits_before_doing_anything(self):
        # the driver may retry a launch it found absent; a delayed first call must then do nothing
        def prepare(tmp):
            (tmp / "run_g1" / "out").mkdir(parents=True)
            (tmp / "run_g1" / "out" / "marker").write_text("first")
        rc, s, done, tmp = self.run_vm(prepare=prepare, with_tmp=True)
        self.assertEqual(rc, 2)
        self.assertIsNone(s)
        self.assertIsNone(done)
        self.assertFalse((tmp / "vm_g1.pid").exists())
        self.assertEqual(sorted(p.name for p in (tmp / "run_g1" / "out").iterdir()), ["marker"])

    def test_wrong_head_fails_checkout(self):
        rc, s, _, _ = self.run_vm(head="f" * 40)
        self.assertEqual((rc, s["gates"]["checkout"]), (1, 99))

    def test_bad_arguments_refused(self):
        r = subprocess.run(["bash", str(HERE / "vm_gemma.sh"), "main", "g1"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
