"""Offline contract checks for the wheel bootstrap (no installs or network)."""
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/linux-wheel-bootstrap.sh"


class WheelWorkflowTests(unittest.TestCase):
    def test_matrix_versions_and_container_compiler(self):
        from packaging.version import Version
        text = (ROOT / ".github/workflows/build-wheels.yml").read_text()
        self.assertEqual(text.count("pypa/cibuildwheel@v2.23.4"), 2)
        self.assertEqual(text.count("quay.io/pypa/manylinux2014_x86_64:2025.04.19-1"), 2)
        self.assertEqual(text.count("bash {project}/.github/scripts/linux-wheel-bootstrap.sh '${{ matrix.cuda_version }}'"), 2)
        self.assertEqual(text.count("CUDAHOSTCXX=/opt/rh/devtoolset-11/root/usr/bin/g++"), 2)
        self.assertEqual(text.count("CMAKE_CUDA_HOST_COMPILER=/opt/rh/devtoolset-11/root/usr/bin/g++"), 2)
        for suffix in ("+cu118", "+cu121"):
            self.assertEqual(text.count('wheel_suffix: "' + suffix + '"'), 2)
            self.assertEqual(str(Version("0.1.0" + suffix)), "0.1.0" + suffix)
        self.assertNotIn("CUDA_VERSION_MAJOR=", text)
        self.assertIn("cp38-*", text)

    def test_native_smoke_rejects_package_that_swallows_extension_failure(self):
        import sys
        import shlex
        text = (ROOT / ".github/workflows/build-wheels.yml").read_text()
        commands = [line.split("CIBW_TEST_COMMAND: ", 1)[1] for line in text.splitlines() if "CIBW_TEST_COMMAND:" in line]
        self.assertEqual(len(commands), 2)
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "tinyllama_cpp"
            package.mkdir()
            # Same suppression mechanism as the actual package initializer.
            (package / "__init__.py").write_text("try:\n    from .tinyllama_bindings import *\nexcept ImportError:\n    pass\n")
            old = subprocess.run([sys.executable, "-c", "import tinyllama_cpp"], cwd=tmp, capture_output=True)
            self.assertEqual(old.returncode, 0)
            def check(expected, cuda="0"):
                import os
                for command in commands:
                    args = shlex.split(command)
                    args[0] = sys.executable
                    proc = subprocess.run(args, cwd=tmp, capture_output=True, env=dict(os.environ, TINYLLAMA_CPP_BUILD_CUDA=cuda))
                    self.assertEqual(proc.returncode == 0, expected, proc.stderr)
            check(False)  # Missing extension must fail, despite successful package import.
            (package / "tinyllama_bindings.py").write_text("raise ImportError('unresolved native symbol')\n")
            check(False)  # An installed but unloadable extension must also fail.
            (package / "tinyllama_bindings.py").write_text("class TinyLlamaSession: pass\nclass ModelConfig: pass\ncuda_compiled = False\n")
            check(True)
            check(False, "1")  # GPU-labelled CPU fallback must fail.
            check(False, "invalid")
            (package / "tinyllama_bindings.py").write_text("class TinyLlamaSession: pass\nclass ModelConfig: pass\ncuda_compiled = True\n")
            check(True, "1")
            check(False, "0")
            (package / "tinyllama_bindings.py").write_text("TinyLlamaSession = None\nModelConfig = None\ncuda_compiled = False\n")
            check(False)  # Required APIs must actually be callable.

    def test_actual_version_step_changes_only_project_version(self):
        import tomllib
        import sys
        import shlex
        text = (ROOT / ".github/workflows/build-wheels.yml").read_text()
        block = text.split("    - name: Modify package version for CUDA builds\n", 1)[1]
        block = block.split("      run: |\n", 1)[1].split("\n    # Install build dependencies", 1)[0]
        command = "\n".join(line[8:] if line.startswith("        ") else line for line in block.splitlines())
        original = (ROOT / "pyproject.toml").read_text()
        before = tomllib.loads(original)
        for version, suffix in (("11.8", "+cu118"), ("12.1", "+cu121")):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "pyproject.toml"
                path.write_text(original)
                shell = command.replace("${{ matrix.wheel_suffix }}", suffix).replace("${{ matrix.cuda_version }}", version)
                shell = shell.replace("python -c", shlex.quote(sys.executable) + " -c", 1)
                subprocess.run(["bash", "-eu", "-c", shell], cwd=tmp, check=True, capture_output=True)
                after = tomllib.loads(path.read_text())
                self.assertEqual(after["project"]["version"], before["project"]["version"] + suffix)
                after["project"]["version"] = before["project"]["version"]
                self.assertEqual(after, before)

    def test_shell_syntax_and_invalid_selector_refuses_before_install(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        for selector in ("", "12.2", "75", "cpu; false"):
            proc = subprocess.run(["bash", str(SCRIPT), selector], capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertNotIn("Installing", proc.stdout)

    def run_bootstrap(self, selector, nvcc_version=None):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            source = SCRIPT.read_text()
            # Relocate every absolute system path into this private fixture.
            source = re.sub(r"(?<![a-zA-Z0-9/])/(etc|opt|usr)/", lambda m: str(home) + m.group(), source)
            script = home / "bootstrap.sh"
            script.write_text(source)
            (home / "etc/yum.repos.d").mkdir(parents=True)
            (home / "etc/yum.repos.d/CentOS-Base.repo").write_text("old mirrorlist")
            compiler = home / "opt/rh/devtoolset-11/root/usr/bin/g++"
            compiler.parent.mkdir(parents=True)
            compiler.write_text('#!/bin/sh\nif [ "$1" = "-dumpversion" ]; then echo 11; else while [ "$1" != "-o" ]; do shift; done; shift; printf "#!/bin/sh\\nexit 0\\n" > "$1"; chmod +x "$1"; fi\n')
            compiler.chmod(0o755)
            if nvcc_version is not None:
                nvcc = home / "usr/local/cuda/bin/nvcc"
                nvcc.parent.mkdir(parents=True)
                nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release " + nvcc_version + ", Vfixture'\n")
                nvcc.chmod(0o755)
            fake = home / "bin"
            fake.mkdir()
            # Never install, download or extract anything. Record the selected commands.
            programs = {
                "yum": 'printf "%s\n" "$*" >> "$TEST_LOG"',
                "curl": 'printf "%s\n" "$*" >> "$TEST_LOG"',
                "sha256sum": 'printf "%s\n" "$*" >> "$TEST_LOG"',
                "sh": 'printf "%s\n" "$*" >> "$TEST_LOG"',
                "tar": 'while [ "$1" != "-C" ]; do shift; done; shift; mkdir -p "$1/boost_1_82_0"; printf "#!/bin/sh\nexit 0\n" > "$1/boost_1_82_0/bootstrap.sh"; cp "$1/boost_1_82_0/bootstrap.sh" "$1/boost_1_82_0/b2"; chmod +x "$1/boost_1_82_0/"*',
            }
            for name, body in programs.items():
                path = fake / name
                path.write_text("#!/bin/bash\n" + body + "\n")
                path.chmod(0o755)
            import os
            env = dict(os.environ, PATH=str(fake) + ":" + os.environ["PATH"], TEST_LOG=str(home / "calls"))
            proc = subprocess.run(["bash", str(script), selector], capture_output=True, text=True, env=env)
            calls = (home / "calls").read_text()
            repo = (home / "etc/yum.repos.d/wheel-vault.repo").read_text()
            return proc, calls, repo

    def test_cpu_bootstrap_never_installs_cuda_and_checks_repositories(self):
        proc, calls, repo = self.run_bootstrap("cpu")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("local_installers", calls)
        self.assertIn("devtoolset-11-gcc-11.2.1-9.1.el7", calls)
        self.assertEqual(repo.count("gpgcheck=1"), 3)
        self.assertNotIn("mirrorlist=", repo)
        self.assertIn("sha256sum", SCRIPT.read_text())

    def test_existing_cuda_must_match_explicit_matrix_version(self):
        for version in ("11.8", "12.1"):
            proc, calls, _ = self.run_bootstrap(version, version)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotIn("local_installers", calls)
            wrong = "12.1" if version == "11.8" else "11.8"
            proc, _, _ = self.run_bootstrap(version, wrong)
            self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
