#!/usr/bin/env python3
"""E4 gate 3: ember's C++ IR builder against keramos tools/gguf_ir.py (an oracle, run, not read).

Per GGUF: gguf_ir.py's document (run as a program) and ember's (tools/ir_logits with EMBER_IR_PRINT, i.e.
build_ir_from_gguf through find_ir, the file having no paperkiln.ir), compared
  (a) as JSON values, every key but _comment;
  (b) under Appendix D.7 by the oracle: the file rewritten with ember's document embedded as paperkiln.ir and
      ir_validate.py run on it (rule 38 compares the embedded document with the graph the metadata builds);
  (c) under ember's own D.7 (graph_ir_validate.cpp rule38_form): the file with gguf_ir.py's document embedded,
      through ir_adapter's check.
Variants with odd metadata values (gemma2's final_logit_softcapping 0.1, attn cap 0.3, no sliding_window key;
phi3's attn_factor 0 and 1) are written by --variants.

    python3 e4_builder.py KERAMOS EMBER_BUILD WORK GGUF ...
"""
import json
import os
import subprocess
import sys

import gguf as G


def gguf_ir_doc(keramos, path, out):
    subprocess.run([sys.executable, os.path.join(keramos, "tools", "gguf_ir.py"), path, out], check=True,
                   capture_output=True)
    return json.load(open(out))


def ember_doc(build, path, out):
    env = {k: v for k, v in os.environ.items() if not k.startswith("EMBER_")}
    env["EMBER_IR_PRINT"] = out
    r = subprocess.run([os.path.join(build, "ir_logits"), path], input="[[1, 2]]", capture_output=True, text=True,
                       env=env)
    if not os.path.exists(out):
        raise RuntimeError(f"ir_logits wrote no IR: {r.stderr[-1500:]}")
    return json.load(open(out))


def rewrite(src, dst, set_kv=None, drop=(), embed=None):
    """src's metadata and tensors into dst, with keys changed, dropped, and paperkiln.ir embedded."""
    r = G.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    w = G.GGUFWriter(dst, arch, use_temp_file=False)
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for k, f in r.fields.items():
        if k in skip or k in drop or (set_kv and k in set_kv) or k in ("paperkiln.ir", "paperkiln.ir_version"):
            continue
        t = f.types[0]
        v = f.contents()
        if t == G.GGUFValueType.ARRAY:
            w.add_array(k, v)
        elif t == G.GGUFValueType.STRING:
            w.add_string(k, v)
        else:
            w.add_key_value(k, v, t)
    for k, (v, t) in (set_kv or {}).items():
        w.add_key_value(k, v, t)
    if embed is not None:
        w.add_string("paperkiln.ir", json.dumps(embed))
        w.add_uint32("paperkiln.ir_version", int(embed["ir_version"]))
    for t in r.tensors:
        w.add_tensor(t.name, t.data, raw_shape=list(reversed([int(x) for x in t.shape])), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def oracle_check(keramos, path):
    r = subprocess.run([sys.executable, os.path.join(keramos, "tools", "ir_validate.py"), path], capture_output=True,
                       text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-600:]


def ember_check(build, path):
    p = subprocess.run([os.path.join(build, "ir_adapter"), "--conformance"],
                       input=json.dumps({"id": 1, "method": "check", "params": {"gguf": path, "ir": None}}) + "\n",
                       capture_output=True, text=True)
    r = json.loads(p.stdout.splitlines()[0])["result"]
    return r["accepted"], r.get("message", "")[:600]


def compare(keramos, build, work, path, tag):
    a = gguf_ir_doc(keramos, path, os.path.join(work, tag + ".gguf_ir.json"))
    b = ember_doc(build, path, os.path.join(work, tag + ".ember.json"))
    strip = lambda d: {k: v for k, v in d.items() if k != "_comment"}  # noqa: E731
    equal = strip(a) == strip(b)
    diffs = [k for k in sorted(set(a) | set(b)) if k != "_comment" and a.get(k) != b.get(k)]
    with_ember = os.path.join(work, tag + ".with_ember.gguf")
    rewrite(path, with_ember, embed=b)
    oracle_ok, oracle_msg = oracle_check(keramos, with_ember)
    with_ref = os.path.join(work, tag + ".with_gguf_ir.gguf")
    rewrite(path, with_ref, embed=a)
    ember_ok, ember_msg = ember_check(build, with_ref)
    os.remove(with_ember)
    os.remove(with_ref)
    row = {"file": tag, "json_equal": equal, "differing_keys": diffs, "oracle_rule38_accepts_ember_doc": oracle_ok,
           "ember_rule38_accepts_gguf_ir_doc": ember_ok}
    if not oracle_ok:
        row["oracle_message"] = oracle_msg
    if not ember_ok:
        row["ember_message"] = ember_msg
    if diffs:
        row["diff"] = {k: {"gguf_ir": a.get(k), "ember": b.get(k)} for k in diffs}
    print(json.dumps({k: v for k, v in row.items() if k != "diff"}), flush=True)
    return row


def main():
    keramos, build, work = sys.argv[1:4]
    files = sys.argv[4:]
    os.makedirs(work, exist_ok=True)
    rows = []
    for f in files:
        tag = os.path.splitext(os.path.basename(f))[0]
        rows.append(compare(keramos, build, work, f, tag))
        arch = G.GGUFReader(f).fields["general.architecture"].contents()
        variants = []
        if arch == "gemma2":
            F32 = G.GGUFValueType.FLOAT32
            variants = [("fcap0.1_cap0.3", {"gemma2.final_logit_softcapping": (0.1, F32),
                                             "gemma2.attn_logit_softcapping": (0.3, F32)}, ()),
                        ("fcap30_cap50_nowindow", {"gemma2.final_logit_softcapping": (30.0, F32),
                                                   "gemma2.attn_logit_softcapping": (50.0, F32)},
                         ("gemma2.attention.sliding_window",)),
                        ("caps_zero", {"gemma2.final_logit_softcapping": (0.0, F32),
                                       "gemma2.attn_logit_softcapping": (0.0, F32)}, ())]
        if arch == "phi3" and "phi3.rope.scaling.attn_factor" in G.GGUFReader(f).fields:
            F32 = G.GGUFValueType.FLOAT32
            variants = [("attn_factor0", {"phi3.rope.scaling.attn_factor": (0.0, F32)}, ()),
                        ("attn_factor1", {"phi3.rope.scaling.attn_factor": (1.0, F32)}, ()),
                        ("attn_factor_absent", {}, ("phi3.rope.scaling.attn_factor",))]
        for name, kv, drop in variants:
            v = os.path.join(work, f"{tag}.{name}.gguf")
            rewrite(f, v, set_kv=kv, drop=drop)
            rows.append(compare(keramos, build, work, v, f"{tag}.{name}"))
            os.remove(v)
    json.dump(rows, open(os.path.join(work, "builder_equality.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
