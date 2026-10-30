try:
    from ._version import __version__
except ImportError:
    # Fallback version if _version.py doesn't exist (development mode)
    __version__ = "0.1.0+dev"

# Python 3.8+ resolves extension dependencies through AddDllDirectory, not
# ordinary PATH. Retry only a failed native import, so working CPU wheels do
# not load or register an installed CUDA toolkit. Keep the handle alive for
# transitive libraries that the CUDA runtime may load later.
_cuda_dll_directory_handles = []


def _register_cuda_dll_directory():
    import os
    import sys

    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return False
    toolkit = os.environ.get("CUDA_PATH")
    if not toolkit or not os.path.isabs(toolkit):
        return False
    directory = os.path.join(toolkit, "bin")
    if not os.path.isdir(directory):
        return False
    try:
        _cuda_dll_directory_handles.append(os.add_dll_directory(directory))
        return True
    except OSError:
        # Preserve the native import error if a configured toolkit is stale or
        # inaccessible; never substitute a different SDK or CPU implementation.
        return False


try:
    import importlib as _importlib

    try:
        _importlib.import_module(".tinyllama_bindings", __name__)
    except ImportError:
        if not _register_cuda_dll_directory():
            raise
        _importlib.import_module(".tinyllama_bindings", __name__)
    from .tinyllama_bindings import *

    _all_symbols = [s for s in dir(tinyllama_bindings) if not s.startswith('_')]
    __all__ = _all_symbols + ['__version__']

except ImportError as e:
    import sys
    print("Could not import 'tinyllama_bindings' C++ extension. "
          "If you are building from source, please make sure the C++ code is compiled.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)

    # Still export __version__ even if bindings fail to import
    __all__ = ['__version__']
