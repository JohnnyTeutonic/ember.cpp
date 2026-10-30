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
        self.assertEqual(text.count("PIP_PREFER_BINARY=1"), 2)

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

    def test_graph_instantiation_calls_compile_with_cuda_118_and_121_contracts(self):
        import shutil
        compiler = shutil.which("c++") or shutil.which("g++")
        self.assertIsNotNone(compiler, "C++ compiler required for CUDA API compatibility contract")
        calls = []
        for name in ("model.cpp", "speculative_verify.cpp", "graph_ir.cpp"):
            source = (ROOT / name).read_text()
            found = re.findall(r"cudaGraphInstantiate(?:WithFlags)?\([^;]+\)", source)
            self.assertEqual(len(found), 1)
            calls += found
        # Check every production translation unit/header at the repository root;
        # historical experimental snapshots deliberately remain unchanged.
        for pattern in ("*.cpp", "*.cu", "*.h"):
            for path in ROOT.glob(pattern):
                self.assertNotIn("cudaGraphInstantiate(", path.read_text(), str(path))
        # Official runtime headers: 11.8's base API has five parameters; 12.1's has
        # three. WithFlags has the same three-parameter signature in both:
        # docs.nvidia.com/cuda/archive/{11.8.0,12.1.1}/cuda-runtime-api/group__CUDART__GRAPH.html
        common = "using cudaGraphExec_t = void*; using cudaGraph_t = void*; using cudaGraphNode_t = void*; using cudaError_t = int;\n"
        common += "cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t*, cudaGraph_t, unsigned long long);\n"
        legacy = "cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, cudaGraphNode_t*, char*, unsigned long);\n"
        modern = "cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned long long);\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph_contract.cpp"
            def compile_call(declaration, call):
                path.write_text(common + declaration + "int main() { cudaGraphExec_t exec{}, decode_graph_exec_{}, graph_exec_{}; cudaGraph_t graph{}; return " + call + "; }\n")
                return subprocess.run([compiler, "-std=c++17", "-fsyntax-only", "-Werror", str(path)], capture_output=True)
            for call in calls:
                for declaration in (legacy, modern):
                    proc = compile_call(declaration, call)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                old_call = call.replace("cudaGraphInstantiateWithFlags", "cudaGraphInstantiate")
                self.assertNotEqual(compile_call(legacy, old_call).returncode, 0)
                self.assertEqual(compile_call(modern, old_call).returncode, 0)

    def test_cuda_headers_are_vendor_types_without_compatibility_stubs(self):
        header = (ROOT / "cuda_safe_headers.h").read_text()
        for name in ("cuda_runtime.h", "cublas_v2.h", "cuda_fp16.h", "cuda_bf16.h"):
            self.assertIn("#include <" + name + ">", header)
        for name in ("CMakeLists.txt", "cuda_safe_headers.h", "cuda_kernels.h", "model.h"):
            source = (ROOT / name).read_text()
            for forbidden in ("WINDOWS_CUDA_12_1_WORKAROUND", "CUDA_NO_HALF", "__CUDA_FP16_H__", "__CUDA_BF16_H__", "allow-unsupported-compiler", "ALLOW_COMPILER_AND_STL_VERSION_MISMATCH", "typedef struct", "static inline __host__"):
                self.assertNotIn(forbidden, source)

    def test_windows_cuda_uses_supported_toolset_and_complete_headers(self):
        text = (ROOT / ".github/workflows/build-wheels.yml").read_text()
        for version in ("11.8", "12.1"):
            self.assertIn('- os: windows-2022\n            cuda_version: "' + version + '"', text)
        self.assertIn('- os: windows-latest\n            cuda_version: "cpu"', text)
        self.assertIn('"thrust"', text)
        self.assertIn('"nvtx"', text)
        self.assertIn("CUDA NVTX development headers are missing", text)
        self.assertIn("vsversion: '2022'", text)
        self.assertIn("toolset: '14.34'", text)
        self.assertIn("Microsoft.VisualStudio.Component.VC.14.34.17.4.x86.x64", text)
        self.assertLess(text.index("Setup Visual Studio Environment (Windows CUDA)"), text.index("Install build dependencies (Windows)"))
        self.assertIn("--overlay-triplets=.github/triplets/cuda-msvc1434", text)
        self.assertIn("DISTUTILS_USE_SDK=1 CC=cl CXX=cl CUDAHOSTCXX=cl", text)
        triplet = (ROOT / ".github/triplets/cuda-msvc1434/x64-windows.cmake").read_text()
        self.assertIn("VCPKG_PLATFORM_TOOLSET_VERSION 14.34", triplet)
        self.assertNotIn("VCPKG_VISUAL_STUDIO_PATH", triplet)
        self.assertNotIn("VCPKG_VISUAL_STUDIO_PATH", text)
        self.assertIn("VCPKG_OVERLAY_TRIPLETS=$env:GITHUB_WORKSPACE", text)
        gpu = text.split("    - name: Build wheels (CUDA)", 1)[1]
        hook = gpu.split("CIBW_BEFORE_BUILD_WINDOWS: >", 1)[1].split("CIBW_BEFORE_BUILD_MACOS:", 1)[0]
        self.assertNotIn("vcpkg install", hook)
        self.assertNotIn("DETECTED_CUDA_PATH", hook)
        self.assertIn("@('CUDA_PATH', 'CUDA_HOME', 'CUDA_ROOT', 'CUDAToolkit_ROOT')", text)
        self.assertIn("CMAKE_CUDA_COMPILER=$cudaRoot", text)

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
