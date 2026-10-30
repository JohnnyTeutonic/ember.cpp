"""CI diagnostics only: show installed PE dependencies without importing them."""
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig

roots = {Path(sysconfig.get_path(name)) / "tinyllama_cpp" for name in ("purelib", "platlib")}
for root in sorted(roots):
    for extension in sorted(root.glob("*.pyd")):
        print("Native extension dependency inventory:", extension, flush=True)
        dumpbin = shutil.which("dumpbin")
        if dumpbin:
            subprocess.run([dumpbin, "/DEPENDENTS", str(extension)], check=True)
        else:
            print("dumpbin unavailable; PE dependency inventory not captured", flush=True)
toolkit = os.environ.get("CUDA_PATH")
if toolkit:
    directory = Path(toolkit) / "bin"
    print("Explicit CUDA runtime directory:", directory, flush=True)
    print("Available CUDA runtime DLLs:", [p.name for p in sorted(directory.glob("*.dll"))], flush=True)
