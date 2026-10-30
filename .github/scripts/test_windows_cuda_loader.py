"""Exercise the real package initializer with a simulated dependent DLL loader."""
import contextlib
import gc
import importlib
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class WindowsCudaLoaderTests(unittest.TestCase):
    def load(self, platform="win32", configured=True, existing=True, fail=False, cpu=False, relative=False):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "loader_fixture"
            package.mkdir()
            shutil.copyfile(ROOT / "tinyllama_cpp/__init__.py", package / "__init__.py")
            (package / "tinyllama_bindings.py").write_text(
                "import os\nos._fixture_attempts += 1\n"
                + ("" if cpu else "if not os._fixture_registered: raise ImportError('missing dependent DLL')\n")
                + "class TinyLlamaSession: pass\nclass ModelConfig: pass\ncuda_compiled = " + str(not cpu) + "\n")
            toolkit = Path(tmp) / "explicit-toolkit"
            if existing:
                (toolkit / "bin").mkdir(parents=True)
            calls = []
            handle = types.SimpleNamespace(closed=False)
            def register(path):
                calls.append(path)
                if fail:
                    raise OSError("registration denied")
                os._fixture_registered = True
                return handle
            environment = dict(os.environ)
            environment.pop("CUDA_PATH", None)
            if configured:
                environment["CUDA_PATH"] = "relative-toolkit" if relative else str(toolkit)
            old_path = environment.get("PATH")
            sys.path.insert(0, tmp)
            try:
                with patch.dict(os.environ, environment, clear=True), patch.object(sys, "platform", platform), patch.object(os, "add_dll_directory", register, create=True), patch.object(os, "_fixture_registered", False, create=True), patch.object(os, "_fixture_attempts", 0, create=True), contextlib.redirect_stderr(io.StringIO()):
                    module = importlib.import_module("loader_fixture")
                    gc.collect()
                    self.assertEqual(os.environ.get("PATH"), old_path)
                    module._fixture_attempts = os._fixture_attempts
                    return module, calls, handle, str(toolkit / "bin")
            finally:
                sys.path.remove(tmp)
                for name in list(sys.modules):
                    if name == "loader_fixture" or name.startswith("loader_fixture."):
                        del sys.modules[name]

    def test_registers_explicit_sdk_before_retry_and_keeps_handle(self):
        module, calls, handle, directory = self.load()
        self.assertEqual(calls, [directory])
        self.assertIs(module._cuda_dll_directory_handles[0], handle)
        self.assertEqual(module._fixture_attempts, 2)
        self.assertTrue(module.cuda_compiled)
        self.assertTrue(callable(module.TinyLlamaSession))

    def test_cpu_success_does_not_register_cuda(self):
        for configured in (False, True):
            module, calls, _, _ = self.load(cpu=True, configured=configured)
            self.assertEqual(calls, [])
            self.assertFalse(module.cuda_compiled)

    def test_absent_stale_relative_nonwindows_or_denied_does_not_fake_success(self):
        for options in ({"configured": False}, {"existing": False}, {"relative": True}, {"platform": "linux"}, {"fail": True}):
            with self.subTest(options=options):
                module, calls, _, _ = self.load(**options)
                self.assertEqual(module._fixture_attempts, 1)
                self.assertFalse(hasattr(module, "TinyLlamaSession"))
                self.assertEqual(module.__all__, ["__version__"])
                self.assertEqual(len(calls), int(options.get("fail", False)))


if __name__ == "__main__":
    unittest.main()
