"""The route-evidence patch is a pure insertion of trace-guarded code: removing it reproduces the protocol's
candidate source byte for byte.

Every insertion is whole lines. Removing (a) each statement guarded by the trace flag, (b) the flag definitions
and includes, and (c) the named trace helpers and declarations yields files whose SHA-256 equals the protocol
pins (api.cpp, model.cpp) or the protocol commit 41b6cd5 (the others). A removable guard must be one of five audited
conditions (the bare flag, or the flag conjoined with reads of locals and members); any other guard, negated,
disjunctive, calling or otherwise, is refused rather than deleted.

This is a source-difference proof, not a semantic one. With EMBER_ROUTE_TRACE unset the patched build also runs
the cached flag reads, the trace_request initialisation and the untaken branches' conditions, and TinyLlamaModel
carries route-state members (constructed once). All three arms of the gate run this same binary; it is not timed
against an uninstrumented build.
"""
import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PINS = {  # parts4_full_model_protocol_20261102.json source_files_sha256, and git show 41b6cd5:<file>
    "api.cpp": "43e66be5dfaa369a165af8746b6de516b0559f9168a7d9379e019ed16a76b903",
    "model.cpp": "6c23856da6015306abeefd59af5c6044fc6790779d93bd672614c51b02a0fd6f",
    "model.h": "974aa758b8eaaf570359d72a1549bcaf5f1495e08f07a25b80cf60a7e684204d",
    "verify_kernels.cu": "89507d2459f75336be9d453d85df60cf9355702add11a71613223ae754436e3f",
    "verify_kernels.h": "de5b1edc70bd4ef6349d9203b4d408d6efd3023a6de5ddd23e6f28a05ba4aee1",
}
UNCHANGED = {
    "decode_pick_policy.h": "53cd249e1c9ebaf8166a067eacb12a18ed09f0acb2a51cf3bdd028779d169d84",
    "row_pick_partitioned.cuh": "a9500fcbdcaa793b12ecbd5d529c6826c669ea4b4ba4630f82a51c1737d27346",
    "determinism.h": "67c359489a0482885dfe38347a81a0d470794b2f125321e8ec9274bf1cf8d944",
    "CMakeLists.txt": "339857345162138b5aec821adef589e3987ea1b10ac21b4cf442d04579d7f6b1",
}
INCLUDES = {"api.cpp": {'#include "decode_route_trace.h"', '#include "determinism.h"'},
            "model.cpp": {'#include "decode_route_trace.h"'}}
FLAG_LINES = {
    "const bool trace = ember::route_trace::on();",
    "const bool tracing = ember::route_trace::on();",
    "const long long trace_request = tracing ? next_trace_request() : 0;",
    "DecodeRoute decode_route_;",
    "long long decode_capture_count_ = 0, decode_replay_count_ = 0;",
    "void trace_graph_inventory(cudaGraph_t graph);",
}
# (file, first line, last line, also remove the blank line before, after): named trace helpers.
RANGES = [
    ("api.cpp", "// Route tracing (decode_route_trace.h): one event per sampled position of the plain decode loop. The forward's",
     ("          {\"second_bits\", ember::route_trace::bits(r.second)}, {\"first\", r.first}, {\"last\", r.last}, {\"nan\", r.nan}};",
      "}"), False, True),
    ("model.cpp", "// Route tracing's inventory of a captured decode graph: its kernel nodes, and which of them are the selector",
     "}", False, True),
    ("model.h", "  // What the last forward_device_graph call did, filled only while route tracing is on (decode_route_trace.h).",
     "  const DecodeRoute& decode_route() const { return decode_route_; }", False, False),
    ("verify_kernels.h", "// The selector kernels' host stubs, which a captured graph's kernel nodes name (decode route tracing,",
     "SelectorKernels selector_kernels();", False, False),
    ("verify_kernels.cu", "SelectorKernels selector_kernels() {", "}", True, False),
]
TRACE_IF = re.compile(r"^\s*if \((?=[^\n]*\b(trace|tracing)\b)")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def statement_end(lines, i):
    """Index of the last line of the statement starting at lines[i] (balanced parentheses and braces)."""
    paren = brace = 0
    for j in range(i, len(lines)):
        code = re.sub(r'"(\\.|[^"\\])*"', '""', lines[j]).split("//")[0]
        paren += code.count("(") - code.count(")")
        brace += code.count("{") - code.count("}")
        stripped = lines[j].rstrip()
        if paren == 0 and brace == 0 and (stripped.endswith(";") or stripped.endswith("}")):
            return j
    raise AssertionError("unterminated statement at line %d" % (i + 1))


def guard(lines, i):
    """The condition of the `if` starting at lines[i], balanced across lines."""
    text = "\n".join(lines[i:i + 8])
    start = text.index("if (") + 3
    depth = 0
    for j in range(start, len(text)):
        depth += (text[j] == "(") - (text[j] == ")")
        if depth == 0:
            return text[start + 1:j]
    raise ValueError("unterminated condition at line %d" % (i + 1))


def conjuncts(cond):
    parts, depth, cur, j = [], 0, "", 0
    while j < len(cond):
        c = cond[j]
        depth += (c == "(") - (c == ")")
        if depth == 0 and cond.startswith("&&", j):
            parts.append(cur.strip())
            cur, j = "", j + 2
            continue
        if depth == 0 and cond.startswith("||", j):
            raise ValueError("top-level disjunction in a trace guard: " + cond)
        cur += c
        j += 1
    return parts + [cur.strip()]


# The audited guards, exactly: the bare flag, or the flag conjoined with reads of locals and members. A syntactic
# side-effect test can be bypassed (a template call, an overloaded operator), so nothing else is accepted.
AUDITED_GUARDS = {
    "trace",
    "tracing",
    "trace && pick",
    "trace && (!enabled || decode_graph_disabled_)",
    "e == cudaSuccess && trace",
}


def check_guard(cond):
    """A removable guard is one of the audited guards; any other condition is refused, not deleted."""
    normal = " ".join(cond.split())
    conjuncts(normal)  # a top-level disjunction is reported as such
    if normal not in AUDITED_GUARDS:
        raise ValueError("unaudited trace guard: " + normal)


def reduce(name, text):
    lines = text.split("\n")
    for file, first, last, before, after in RANGES:
        if file != name:
            continue
        a = lines.index(first)
        tail = last if isinstance(last, tuple) else (last,)
        b = next(j for j in range(a, len(lines)) if tuple(lines[j:j + len(tail)]) == tail) + len(tail) - 1
        if before:
            assert lines[a - 1] == ""
            a -= 1
        if after:
            assert lines[b + 1] == ""
            b += 1
        del lines[a:b + 1]
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() in FLAG_LINES or lines[i] in INCLUDES.get(name, ()):
            i += 1
        elif TRACE_IF.match(lines[i]):
            check_guard(guard(lines, i))
            i = statement_end(lines, i) + 1
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def test_reduction_reproduces_the_protocol_source():
    for name, pin in PINS.items():
        current = (ROOT / name).read_text(encoding="utf-8")
        assert sha(current) != pin, name + " carries no patch"
        assert sha(reduce(name, current)) == pin, name + ": trace-only reduction differs from the protocol source"


def test_selector_and_policy_sources_unchanged():
    for name, pin in UNCHANGED.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == pin, name


@pytest.mark.parametrize("statement", [
    "  if (!trace) logits_dev_ = nullptr;",
    "  if (trace || refill_cache()) x = 1;",
    "  if (trace && refill_cache()) x = 1;",
    "  if (trace && (n = 4)) x = 1;",
    "  if (trace && i++) x = 1;",
    "  if (pick) record(trace);",
    "  if (refill_cache<int>() && trace) x = 1;",
    "  if ((*hook)() && trace) x = 1;",
    "  if (trace && pick && refill) x = 1;",
])
def test_unsafe_guards_are_refused_not_deleted(statement):
    source = (ROOT / "model.cpp").read_text(encoding="utf-8")
    anchor = "  if (!enabled || decode_graph_disabled_) return {};"
    assert source.count(anchor) == 1
    with pytest.raises(ValueError):
        reduce("model.cpp", source.replace(anchor, statement + "\n" + anchor))


def test_audited_guard_forms_are_accepted():
    for cond in ("trace", "tracing", "trace && pick", "trace && (!enabled || decode_graph_disabled_)",
                 "e == cudaSuccess && trace"):
        check_guard(cond)


def test_no_trace_statement_outside_a_trace_branch():
    # Every inserted statement that touches route state is inside a removed construct: after the reduction no
    # trace identifier is left in the patched files.
    for name in PINS:
        reduced = reduce(name, (ROOT / name).read_text(encoding="utf-8"))
        for word in ("route_trace", "decode_route_", "trace_graph_inventory", "selector_kernels", "tracing"):
            assert word not in reduced, (name, word)
