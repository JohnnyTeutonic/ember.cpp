"""Run an existing CPU gate script against another build, writing its receipt elsewhere (run in WSL).

    python3 rungate.py BUILD_DIR OUT_DIR SCRIPT [ARGS...]

The script runs unchanged as __main__ (runpy), with these substitutions: os.path.expanduser maps
~/ember_wt/build_cpu to BUILD_DIR (and ~/ct24, the chat-template gates' scratch, to OUT_DIR/ct24, so a
concurrent run elsewhere cannot rewrite this run's scratch files), and a file under ember.cpp/experiments
opened for writing, or renamed into place, goes to OUT_DIR under its own name. So a gate rerun with this
work's build leaves the committed receipts untouched, and `cmp` against them decides byte identity.
Reading is unchanged (the committed receipts are read where the gates compare against them), except for
the basenames in RUNGATE_OUTPUTS (comma-separated): a gate that resumes from its own output file
(regress.py parity) reads and tests for it in OUT_DIR only."""
import builtins
import os
import runpy
import sys

build, outdir, script = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)
EXP = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/experiments"
_expand = os.path.expanduser


def expanduser(p):
    p = os.fspath(p)
    if p.startswith("~/ember_wt/build_cpu"):
        return build + p[len("~/ember_wt/build_cpu"):]
    if p == "~/ct24" or p.startswith("~/ct24/"):
        return os.path.join(outdir, "ct24") + p[len("~/ct24"):]
    return _expand(p)


os.path.expanduser = expanduser
_open = builtins.open


OWN = set(x for x in os.environ.get("RUNGATE_OUTPUTS", "").split(",") if x)


def redirected(file, mode="r", *a, **k):
    if isinstance(file, (str, os.PathLike)):
        f = os.path.abspath(os.fspath(file))
        if f.startswith(EXP + "/") and (any(c in mode for c in "wax+") or os.path.basename(f) in OWN):
            file = os.path.join(outdir, os.path.basename(f))
    return _open(file, mode, *a, **k)


_exists = os.path.exists


def exists(p):
    f = os.path.abspath(os.fspath(p))
    if f.startswith(EXP + "/") and os.path.basename(f) in OWN:
        return _exists(os.path.join(outdir, os.path.basename(f)))
    return _exists(p)


os.path.exists = exists


builtins.open = redirected


def moved(f):
    f = os.path.abspath(os.fspath(f))
    return os.path.join(outdir, os.path.basename(f)) if f.startswith(EXP + "/") else f


_replace, _rename = os.replace, os.rename
os.replace = lambda a, b, *x, **k: _replace(moved(a), moved(b), *x, **k)
os.rename = lambda a, b, *x, **k: _rename(moved(a), moved(b), *x, **k)
sys.argv = [script] + sys.argv[4:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
