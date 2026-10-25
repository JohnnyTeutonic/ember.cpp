#!/usr/bin/env python3
"""ember's load against paperkiln's shared validator (spec/IR_SPEC.md section 10.1, tools/ir_validate.py),
case by case: both accept or both refuse.

  load      ember's adapter `check` (tools/ir_adapter.cpp: the interpreter's whole load) against the validator's
            document and container rules on the same (GGUF, document) pair:
              - every GGUF under /mnt/c/ml_artifacts with paperkiln.ir, as embedded;
              - the third-party GGUFs ember builds a document for (/mnt/c/ml_artifacts/thirdparty and the suite's
                spec/conformance/models), against paperkiln's tools/gguf_ir.py document for the same file;
              - every stored document (ir_validate.corpus()) on a GGUF whose tensors it names (same directory
                first);
              - must-refuse documents and GGUFs, one or more per rule of 10.1, each made from a valid one by one
                edit (GGUF edits write copies under ~/ember_wt/spec_scratch);
  doc       ember's document rules without a GGUF (tools/ir_doc_check.cpp) against the validator's doc_errors,
            on every stored document and every must-refuse document.

Expected differences are labelled, not hidden: "capability" (a valid document ember does not run, section 10.3:
the validator accepts, ember refuses). A missing S = 1 depth_attend pseudo-query (section 5.18) is accepted by
both: the validator (revision 1.1) reports it as an exporter-rule warning, not a refusal. Writes
agreement_cpu.json beside this file; exits 1 on any unexpected disagreement.

The IR tools (ir_validate, gguf_ir: keramos) come from $KERAMOS_TOOLS, else paperkiln's vendored spec/tools,
else paperkiln's tools/ (before keramos v1.1.1). keramos's corpus() holds only its own documents, so the stores
paperkiln's corpus() used to glob (paperkiln's and ember's experiments, /mnt/c/ml_artifacts) are passed to it.
--build DIR is the build holding ir_adapter and ir_doc_check (default ~/ember_wt/build_cpu).
Usage (WSL): python3 agreement.py [--build DIR] [--only SUBSTRING]
"""
import copy
import glob
import json
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete"
PK = ROOT + "/microtorch"
IRT = os.path.expanduser(os.environ.get("KERAMOS_TOOLS") or (
    PK + "/spec/tools" if os.path.isfile(PK + "/spec/tools/ir_validate.py") else PK + "/tools"))
sys.path.insert(0, PK + "/tools")
sys.path.insert(0, IRT)
import ir_validate as V  # noqa: E402
import gguf_ir  # noqa: E402

A = "/mnt/c/ml_artifacts"
M = os.path.expanduser("~/ember_wt/adapter_models")
B = os.path.expanduser(sys.argv[sys.argv.index("--build") + 1] if "--build" in sys.argv else "~/ember_wt/build_cpu")
STORES = [PK + "/experiments/**/*.ir.json", ROOT + "/ember.cpp/experiments/**/*.ir.json", A + "/**/*.ir.json",
          A + "/*gate*/**/*.gguf"]


def corpus():
    """keramos's corpus() with the workspace's stores; paperkiln's former corpus() globbed them itself."""
    try:
        return V.corpus(STORES)
    except TypeError:
        return V.corpus()
SCR = os.path.expanduser("~/ember_wt/spec_scratch")
os.makedirs(SCR, exist_ok=True)
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
_S = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def header(path):
    """(meta {key: (type, value)}, tensors [(name, dims, type, offset)]) in ir_validate.read_gguf's form, reading
    the header only (tokenizer arrays other than the tokens are skipped)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF")
        f.read(4)
        n_t, n_kv = struct.unpack("<QQ", f.read(16))

        def take(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def string():
            return f.read(take("<Q")).decode("utf-8", "surrogateescape")

        def value(t, keep):
            if t == 8:
                return string()
            if t == 9:
                et, n = take("<I"), take("<Q")
                if not keep and et != 8:
                    f.seek(n * struct.calcsize(_S[et]), 1)
                    return None
                return [value(et, keep) for _ in range(n)]
            return take(_S[t])

        meta = {}
        for _ in range(n_kv):
            k = string()
            t = take("<I")
            meta[k] = (t, value(t, k == "tokenizer.ggml.tokens"))
        tensors = []
        for _ in range(n_t):
            name = string()
            nd = take("<I")
            dims = [take("<Q") for _ in range(nd)]
            tensors.append((name, dims, take("<I"), take("<Q")))
    return meta, tensors


_HDR = {}


def hdr(path):
    if path not in _HDR:
        _HDR[path] = header(path)
    return _HDR[path]


def embedded(path):
    m, _ = hdr(path)
    return json.loads(m["paperkiln.ir"][1]) if "paperkiln.ir" in m else None


def validator(doc, gguf):
    """The validator's messages for doc loaded from gguf (doc None: the GGUF's own, or paperkiln's gguf_ir
    document for a third-party file)."""
    meta, tensors = hdr(gguf)
    third = "paperkiln.ir" not in meta
    if doc is None:
        doc = embedded(gguf) if not third else gguf_ir.build(*gguf_ir.read_meta(gguf))
    errs = V.doc_errors(doc) + V.gguf_errors(doc, meta, tensors)
    if third:  # rule 24 is about paperkiln.ir's companion key, which a third-party file has no reason to carry
        errs = [e for e in errs if not e.startswith("IR_SPEC 3.1: paperkiln.ir_version")]
    return errs


class Adapter:
    def __init__(self):
        self.log = open(os.path.join(SCR, "adapter_stderr.log"), "w")
        # EMBER_IR_CTX caps the kv cache a load allocates (a third-party document declares the full trained
        # context, 40960 positions for Qwen3): loading is what is compared, and WSL has 8 GB.
        self.p = subprocess.Popen([B + "/ir_adapter"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, text=True, bufsize=1, env={**os.environ, "EMBER_IR_CTX": "1024"})
        self.n = 0

    def check(self, gguf, ir):
        self.n += 1
        self.p.stdin.write(json.dumps({"id": self.n, "method": "check", "params": {"gguf": gguf, "ir": ir}}) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit(f"ir_adapter died on {gguf} {ir}; see {SCR}/adapter_stderr.log")
        r = json.loads(line)
        if "result" not in r:
            return False, "adapter error: " + json.dumps(r)
        return r["result"]["accepted"], r["result"].get("message", "")


def doc_check(paths):
    out = {}
    for i in range(0, len(paths), 200):
        r = subprocess.run([B + "/ir_doc_check"] + paths[i:i + 200], capture_output=True, text=True)
        for ln in r.stdout.splitlines():
            x = json.loads(ln)
            out[x["path"]] = x
    return out


# ---------------------------------------------------------------- GGUF copies with one edit

def copy_gguf(src, dst, drop_keys=(), set_keys=None, drop_tensors=(), arch=None):
    """src rewritten to dst with metadata keys dropped or set ({key: (GGUFValueType, value)}) and tensors dropped;
    every other key and tensor byte for byte (F32 sources only)."""
    import gguf
    r = gguf.GGUFReader(src)
    a = arch or bytes(r.fields["general.architecture"].parts[-1]).decode()
    w = gguf.GGUFWriter(dst, a)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name in ("general.architecture", "general.alignment") or name in drop_keys \
                or (set_keys and name in set_keys):
            continue
        vt = f.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            sub = f.types[-1]
            if sub == gguf.GGUFValueType.STRING:
                val = [bytes(f.parts[i]) for i in f.data]
            else:
                val = [f.parts[i].tolist()[0] for i in f.data]
            w.add_key_value(name, val, vt, sub_type=sub)
        elif vt == gguf.GGUFValueType.STRING:
            w.add_key_value(name, bytes(f.parts[-1]), vt)
        else:
            w.add_key_value(name, f.parts[-1].tolist()[0], vt)
    for name, (vt, val) in (set_keys or {}).items():
        w.add_key_value(name, val, getattr(gguf.GGUFValueType, vt))
    for t in r.tensors:
        if t.name in drop_tensors:
            continue
        assert t.tensor_type == gguf.GGMLQuantizationType.F32, (src, t.name)
        shape = [int(x) for x in reversed(t.shape.tolist())]
        w.add_tensor(t.name, np.asarray(t.data).reshape(shape).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    _HDR.pop(dst, None)
    return dst


# ---------------------------------------------------------------- the must-refuse set

BASES = {
    "S4": (f"{M}/state4/state4.gguf", None),      # llama_state: turn_state, evict, state_weight (paperkiln)
    "MR": (f"{M}/mor3/mor3.gguf", None),          # llama_mor: depth_route, kv_rows active, weight_index
    "AR": (f"{M}/ar3/ar3.gguf", None),            # llama_attnres: list, append, depth_attend
    "B2": (f"{M}/b2x4/r.gguf", None),             # llama_recursive: weight_index
    "KI": (f"{M}/kimi_tiny/kimi_tiny.gguf", None),  # linear_attention, a recurrent slot
    "MA": (f"{M}/mamba_tiny/mamba_tiny.gguf", None),  # ssm_scan
    "SW": (f"{M}/swa_tiny/swa_tiny.gguf", None),  # gpt2_swa: window and sinks
    "GP": (f"{A}/gpt2_tiny/gpt2_tiny.gguf", None),  # gpt2: split, pos_embedding, gelu tanh
    "MO": (f"{A}/moe_tiny/moe_tiny_v3.gguf", None),  # llama + moe_ffn
    "M1": (f"{PK}/spec/conformance/models/tp_mamba1/tp_mamba1.gguf", "built"),  # conv1d_causal, selective_scan
}


def base_doc(key):
    g, kind = BASES[key]
    return copy.deepcopy(gguf_ir.build(*gguf_ir.read_meta(g)) if kind == "built" else embedded(g))


def nodes(d, op, sect="block"):
    return [n for n in d[sect]["nodes"] if n["op"] == op]


def kv(d):
    return [s for s in d["block"]["state"] if s["kind"] == "kv"][0]


def first(d, op, sect="block"):
    return nodes(d, op, sect)[0]


def put(obj, **kw):
    obj.update(kw)


def pop(obj, key):
    obj.pop(key)


def insert(lst, i, x):
    lst.insert(i, x)


# (name, rule, base, edit of the document, expected) with expected "refuse", "capability" or "accept"
DOC_CASES = [
    ("r1_unknown_top_level_key", "1", "S4", lambda d: put(d, extras={})),
    ("r1_model_d_string", "1", "S4", lambda d: put(d["model"], d="64")),
    ("r1_slot_kind_unknown", "1", "S4", lambda d: put(kv(d), kind="paged")),
    ("r2_ir_version_2", "2", "S4", lambda d: put(d, ir_version=2)),
    ("r3_n_layers_ne_repeat", "3", "S4", lambda d: put(d["model"], n_layers=3)),
    ("r3_ctx_zero", "3", "S4", lambda d: put(d["model"], ctx=0)),
    ("r4_unknown_attribute", "4", "S4", lambda d: put(first(d, "attention"), flavour="vanilla")),
    ("r4_unknown_op", "4", "S4", lambda d: put(first(d, "silu"), op="swish")),
    ("r4_ssm_weights_object", "4", "MA", lambda d: put(first(d, "ssm_scan"), weights={"a": first(d, "ssm_scan")["a"]})),
    ("r5_gelu_approx", "5", "GP", lambda d: put(first(d, "gelu"), approx="erf")),
    ("r5_rope_pairing", "5", "S4", lambda d: put(first(d, "rope"), pairing="spiral")),
    ("r5_mask", "5", "S4", lambda d: put(first(d, "attention"), mask="bidirectional")),
    ("r5_moe_gate_norm", "5", "MO", lambda d: put(first(d, "moe_ffn"), gate_norm="softmax")),
    ("r6_per_layer_outside_block", "6", "S4", lambda d: put(first(d, "rmsnorm", "head"), eps={"per_layer": [1e-5]})),
    ("r6_per_layer_length", "6", "AR", lambda d: put(first(d, "attention"), window={"per_layer": [0]})),
    ("r6_per_layer_structural", "6", "S4",
     lambda d: put(first(d, "rmsnorm"), weight={"per_layer": ["blk.0.attn_norm.weight", "blk.1.attn_norm.weight"]})),
    ("r7_add_one_input", "7", "S4", lambda d: put(first(d, "add"), **{"in": ["x"]})),
    ("r7_rope_three_outputs", "7", "S4", lambda d: put(first(d, "rope"), out=["q", "k", "z"])),
    ("r8_rope_theta_missing", "8", "S4", lambda d: pop(first(d, "rope"), "theta")),
    ("r8_attention_scale_missing", "8", "AR", lambda d: pop(first(d, "attention"), "scale")),
    ("r8_rmsnorm_eps_missing", "8", "S4", lambda d: pop(first(d, "rmsnorm"), "eps")),
    ("r8_gelu_approx_missing", "8", "GP", lambda d: pop(first(d, "gelu"), "approx")),
    ("r9_expr_does_not_parse", "9", "GP",
     lambda d: d["block"]["nodes"].__setitem__(d["block"]["nodes"].index(first(d, "gelu")),
                                               {"op": "expr", "in": first(d, "gelu")["in"], "expr": "x + * 2",
                                                "out": first(d, "gelu")["out"]})),
    ("r9_expr_parses", "9 (valid)", "GP",
     lambda d: d["block"]["nodes"].__setitem__(d["block"]["nodes"].index(first(d, "gelu")),
                                               {"op": "expr", "in": first(d, "gelu")["in"], "expr": "x * sigmoid(1.702 * x)",
                                                "out": first(d, "gelu")["out"]})),
    ("r10_split_count", "10", "GP", lambda d: put(first(d, "split"), out=["q", "k"])),
    ("r10_rope_n_rot_odd", "10", "S4", lambda d: put(first(d, "rope"), n_rot=31)),
    ("r10_heads_not_multiple", "10", "AR", lambda d: put(first(d, "attention"), heads=3)),
    ("r10_window_negative", "10", "AR", lambda d: put(first(d, "attention"), window={"per_layer": [0, -1, 0]})),
    ("r10_top_k_past_n_experts", "10", "MO", lambda d: put(first(d, "moe_ffn"), top_k=first(d, "moe_ffn")["n_experts"] + 1)),
    ("r10_conv_kernel_1", "10", "M1", lambda d: put(first(d, "conv1d_causal"), kernel=1)),
    ("r11_slot_declared_twice", "11", "AR", lambda d: d["block"]["state"].append(copy.deepcopy(kv(d)))),
    ("r11_slot_read_by_no_node", "11", "AR", lambda d: d["block"]["state"].append(dict(copy.deepcopy(kv(d)), name="kv2"))),
    ("r11_attention_names_no_slot", "11", "AR", lambda d: put(first(d, "attention"), state="nope")),
    ("r11_stateful_op_outside_block", "11", "AR",
     lambda d: insert(d["head"]["nodes"], 0, dict(copy.deepcopy(first(d, "attention")), **{"in": ["x", "x", "x"], "out": "junk"}))),
    ("r12_kv_lacks_max_len", "12", "AR", lambda d: pop(kv(d), "max_len")),
    ("r12_window_on_unwindowed_slot", "12", "AR", lambda d: put(first(d, "attention"), window=4)),
    ("r12_kv_heads_differs", "12", "AR", lambda d: put(first(d, "attention"), kv_heads=1)),
    ("r12_head_dim_differs", "12", "AR", lambda d: put(kv(d), head_dim=kv(d)["head_dim"] * 2)),
    ("r13_recurrent_init", "13", "KI", lambda d: put(d["block"]["state"][-1]["tensors"][0], init=1)),
    ("r13_recurrent_dtype", "13", "KI", lambda d: put(d["block"]["state"][-1], dtype="f16")),
    ("r13_recurrent_op_differs", "13", "KI", lambda d: put(d["block"]["state"][-1], op="ssm_scan")),
    ("r13_recurrent_shape_misfit", "13", "KI",
     lambda d: put(d["block"]["state"][-1]["tensors"][0], shape=[d["block"]["state"][-1]["tensors"][0]["shape"][0], 7])),
    ("r14_causal_false", "14", "AR", lambda d: put(first(d, "attention"), causal=False)),
    ("r15_turn_state_window", "15", "S4", lambda d: (put(first(d, "attention"), window=8), put(kv(d), window=8))),
    ("r15_turn_state_without_state_weight", "15", "AR", lambda d: put(first(d, "attention"), mask="turn_state")),
    ("r16_evict_with_causal_mask", "16", "S4", lambda d: pop(first(d, "attention"), "mask")),
    ("r17_state_weight_with_i", "17", "S4", lambda d: put(first(d, "embedding", "embed"), state_weight="state_embd.{i}.weight")),
    ("r17_two_state_weights", "17", "S4",
     lambda d: d["embed"]["nodes"].append(dict(copy.deepcopy(first(d, "embedding", "embed")), out="e"))),
    ("r17a_i_outside_block", "17a", "S4", lambda d: put(first(d, "rmsnorm", "head"), weight="output_norm.{i}.weight")),
    ("r17a_i_twice", "17a", "S4", lambda d: put(first(d, "rmsnorm"), weight="blk.{i}.attn_norm.{i}.weight")),
    ("r17a_reads_tokens", "17a", "S4", lambda d: put(first(d, "silu"), **{"in": ["tokens"]})),
    ("r17a_writes_pos", "17a", "S4", lambda d: d["block"]["nodes"].append({"op": "silu", "in": ["x"], "out": "pos"})),
    ("r17a_rope_position_not_pos", "17a", "S4", lambda d: put(first(d, "rope"), **{"in": ["q", "k", "x"]})),
    ("r17a_two_attention_nodes_one_slot", "17a", "AR",
     lambda d: insert(d["block"]["nodes"], d["block"]["nodes"].index(first(d, "attention")) + 1,
                      dict(copy.deepcopy(first(d, "attention")), out="a2"))),
    ("r17a_reads_depth_route_out", "17a", "MR",
     lambda d: d["block"]["nodes"].append({"op": "silu", "in": [d["block"]["nodes"][0]["out"]], "out": "junk"})),
    ("r18_weight_index_length", "18", "B2", lambda d: d["block"]["weight_index"]["per_layer"].pop()),
    ("r18_weight_index_unused_layer", "18", "B2",
     lambda d: put(d["block"]["weight_index"], per_layer=[max(1, m) for m in d["block"]["weight_index"]["per_layer"]])),
    ("r19_tensor_index_without_weight_index", "19", "AR", lambda d: put(first(d, "rmsnorm"), tensor_index="application")),
    ("r19_tensor_index_value", "19", "B2", lambda d: put(first(d, "rmsnorm"), tensor_index="stored")),
    ("r20_depth_route_not_first", "20", "MR", lambda d: d["block"]["nodes"].insert(0, d["block"]["nodes"].pop(1))),
    ("r20_n_rec_zero", "20", "MR", lambda d: put(d["block"]["nodes"][0], n_rec=0)),
    ("r20_steps_unequal", "20", "MR",
     lambda d: put(d["block"]["nodes"][0], step={"per_layer": [-1, 0, 0, 0, 1, 2, 2, -1]})),
    ("r20_route_expert_choice", "20", "MR", lambda d: put(d["block"]["nodes"][0], route="expert_choice")),
    ("r20_gate_unknown", "20", "MR", lambda d: put(d["block"]["nodes"][0], gate="sigmoid")),
    ("r20_router_name_braced", "20", "MR", lambda d: put(d["block"]["nodes"][0], weight="rec.{k}.router.weight")),
    ("r21_no_kv_rows_active", "21", "MR", lambda d: pop(kv(d), "kv_rows")),
    ("r22_kv_from", "22", "MR", lambda d: put(kv(d), kv_from={"per_layer": list(range(8))})),
    ("r23_depth_route_turn_state", "23", "MR", lambda d: put(first(d, "attention"), mask="turn_state")),
    ("r23_depth_route_lists", "23", "MR",
     lambda d: d["block"]["nodes"].append({"op": "list", "in": ["x"], "out": "bank"})),
    ("r25_missing_tensor", "25", "S4", lambda d: put(first(d, "rmsnorm"), weight="blk.{i}.nope.weight")),
    ("r26_router_ne", "26", "MR", lambda d: put(d["block"]["nodes"][0], weight="blk.0.attn_norm.weight")),
    ("r27_weight_index_not_paperkiln", "27", "GP",
     lambda d: put(d["block"], weight_index={"per_layer": list(range(d["block"]["repeat"]))})),
    ("r28_specials_lack_user", "28", "S4", lambda d: d["tokenizer"]["specials"].pop("user")),
    ("r28_specials_name_other_ids", "28", "S4", lambda d: put(d["tokenizer"]["specials"], eot=5)),
    ("r28_state_weight_without_k_metadata_doc", "28", "AR",
     lambda d: put(first(d, "embedding", "embed"), state_weight="blk.0.attnres_attn_q.weight")),
    ("v_tensor_index_under_weight_index", "7.2 (valid)", "B2",
     lambda d: put(first(d, "rmsnorm"), weight=first(d, "rmsnorm")["weight"].replace("{i}", "0"), tensor_index="application")),
    ("cap_two_kv_slots", "6.1", "AR",
     lambda d: (d["block"]["state"].append(dict(copy.deepcopy(kv(d)), name="kv2")),
                insert(d["block"]["nodes"], d["block"]["nodes"].index(first(d, "attention")) + 1,
                       dict(copy.deepcopy(first(d, "attention")), state="kv2", out="a2")))),
]
EXPECT = {"v_tensor_index_under_weight_index": "accept", "cap_two_kv_slots": "capability", "r9_expr_parses": "accept",
          "s1_missing_pseudo_query": "accept"}  # 5.18: an exporter-rule warning in revision 1.1, not a refusal


def gguf_cases():
    """(name, rule, gguf, doc or None, expected): the container rules, on edited copies of gate GGUFs."""
    s4, b2, ar = BASES["S4"][0], BASES["B2"][0], BASES["AR"][0]
    out = [
        ("r24_ir_version_key_missing", "24", copy_gguf(s4, SCR + "/r24_missing.gguf", drop_keys=("paperkiln.ir_version",))),
        ("r24_ir_version_key_differs", "24",
         copy_gguf(s4, SCR + "/r24_differs.gguf", set_keys={"paperkiln.ir_version": ("UINT32", 1)})),
        ("r24_ir_version_key_not_uint32", "24",
         copy_gguf(s4, SCR + "/r24_int32.gguf", set_keys={"paperkiln.ir_version": ("INT32", 0)})),
        ("r27_block_count_missing", "27", copy_gguf(b2, SCR + "/r27_missing.gguf", drop_keys=("paperkiln.block_count",))),
        ("r27_block_count_differs", "27",
         copy_gguf(b2, SCR + "/r27_differs.gguf", set_keys={"paperkiln.block_count": ("UINT32", 3)})),
        ("r27_architecture_not_paperkiln", "27", copy_gguf(b2, SCR + "/r27_llama.gguf", arch="llama")),
        ("r28_state_weight_without_k", "28",
         copy_gguf(s4, SCR + "/r28_no_k.gguf", drop_keys=("paperkiln.state_tokens.k",))),
        ("r28_k_differs_from_rows", "28",
         copy_gguf(s4, SCR + "/r28_k5.gguf", set_keys={"paperkiln.state_tokens.k": ("UINT32", 5)})),
        ("r28_architecture_not_paperkiln", "28", copy_gguf(s4, SCR + "/r28_llama.gguf", arch="llama")),
        ("r28_template_not_paperkiln1", "28",
         copy_gguf(s4, SCR + "/r28_chatml.gguf", set_keys={"paperkiln.chat_template": (
             "STRING", json.dumps({"name": "chatml", "version": 1}))})),
        ("s1_missing_pseudo_query", "25",
         copy_gguf(ar, SCR + "/s1_no_q0.gguf", drop_tensors=("blk.0.attnres_attn_q.weight",))),
        ("r25_missing_s2_pseudo_query", "25",
         copy_gguf(ar, SCR + "/s2_no_q1.gguf", drop_tensors=("blk.1.attnres_attn_q.weight",))),
    ]
    return [(n, r, g, None, EXPECT.get(n, "refuse")) for n, r, g in out]


# ---------------------------------------------------------------- the valid set

def paperkiln_ggufs():
    out = []
    for p in sorted(glob.glob(A + "/**/*.gguf", recursive=True)):
        try:
            m, _ = hdr(p)
        except (OSError, ValueError, struct.error):
            continue
        if "paperkiln.ir" in m:
            out.append(p)
    return out


def third_party_ggufs():
    return sorted(glob.glob(A + "/thirdparty/*/*.gguf") + glob.glob(PK + "/spec/conformance/models/*/*.gguf"))


def pair(doc_path, ggufs):
    """A GGUF the document can load from: the file's own document has the same model fields and the validator
    finds every tensor the document names there (the GGUF embedding this very document first, then the same
    directory)."""
    doc = json.load(open(doc_path, encoding="utf-8"))
    d = os.path.dirname(doc_path)
    order = sorted(ggufs, key=lambda g: (embedded(g) != doc, os.path.dirname(g) != d, g))
    for g in order:
        meta, tensors = hdr(g)
        if "paperkiln.ir" in meta and embedded(g).get("model") == doc.get("model") and V.weight_refs(doc) and not any(
                "is not a tensor of the file" in e for e in V.gguf_errors(doc, meta, tensors)):
            return g
    return None


def verdict(v_errs, e_ok, expected):
    v_ok = not v_errs
    if expected == "capability":
        return "expected (capability)" if v_ok and not e_ok else "DISAGREE"
    if v_ok == e_ok:
        return "agree"
    return "DISAGREE"


def main():
    ad = Adapter()
    report = {"validator": os.path.abspath(V.__file__),"adapter": B + "/ir_adapter", "load": [], "doc": []}
    pk = paperkiln_ggufs()

    def run(name, kind, rule, gguf, doc, expected):
        if ONLY and ONLY not in name:
            return
        ir_path = None
        if doc is not None:
            ir_path = os.path.join(SCR, name.replace("/", "_") + ".ir.json")
            if isinstance(doc, str):
                ir_path = doc
                doc = json.load(open(doc, encoding="utf-8"))
            else:
                json.dump(doc, open(ir_path, "w"))
        v = validator(doc, gguf)
        ok, msg = ad.check(gguf, ir_path)
        row = {"case": name, "kind": kind, "rule": rule, "gguf": gguf, "ir": ir_path, "expected": expected,
               "validator": "accept" if not v else "refuse", "validator_messages": v[:4],
               "ember": "accept" if ok else "refuse", "ember_message": msg[:400],
               "result": verdict(v, ok, expected)}
        report["load"].append(row)
        print(f"{row['result']:22s} {name}: validator {row['validator']}, ember {row['ember']}"
              + ("" if row["result"] == "agree" and ok else f" | {msg[:150]}"), flush=True)

    for g in pk:
        run("gguf:" + os.path.relpath(g, A), "valid", "-", g, None, "accept")
    for g in third_party_ggufs():
        tag = "suite:" if g.startswith(PK) else "thirdparty:"
        run(tag + os.path.basename(g), "valid", "-", g, None, "accept")
    stored = [p for p in corpus() if p.endswith(".ir.json")]
    unpaired = []
    for p in stored:
        g = pair(p, pk)
        if g is None:
            unpaired.append(p)
            continue
        run("stored:" + p.replace(ROOT + "/", "").replace(A + "/", "ml_artifacts/"), "valid", "-", g, p, "accept")
    report["stored_without_a_gguf"] = unpaired
    must = []
    for name, rule, base, edit in DOC_CASES:
        d = base_doc(base)
        edit(d)
        must.append((name, rule, BASES[base][0], d, EXPECT.get(name, "refuse")))
    for name, rule, g, d, exp in must + gguf_cases():
        run(name, "must-refuse" if exp == "refuse" else exp, rule, g, d, exp)
    # Document tier: every stored document and every must-refuse document, without a GGUF.
    docs = {p: json.load(open(p, encoding="utf-8")) for p in stored}
    for name, rule, g, d, exp in must:
        if ONLY and ONLY not in name:
            continue
        p = os.path.join(SCR, "doc_" + name + ".ir.json")
        json.dump(d, open(p, "w"))
        docs[p] = d
    em = doc_check(list(docs))
    for p, d in docs.items():
        v = V.doc_errors(d)
        e = em[p]
        name = os.path.basename(p)
        exp = "accept" if p in stored else EXPECT.get(name[4:-8], "refuse")
        if exp == "capability":  # a capability refusal is the interpreter's, not a document rule
            exp = "accept"
        res = "agree" if (not v) == e["accepted"] else "DISAGREE"
        report["doc"].append({"doc": p, "expected": exp, "validator": "accept" if not v else "refuse",
                              "validator_messages": v[:4], "ember": "accept" if e["accepted"] else "refuse",
                              "ember_errors": e["errors"][:4], "result": res})
        if res != "agree":
            print(f"doc DISAGREE {p}: validator {v[:2]} ember {e['errors'][:2]}")
    bad = [r for r in report["load"] + report["doc"] if r["result"] == "DISAGREE"]
    report["summary"] = {
        "load_cases": len(report["load"]), "doc_cases": len(report["doc"]),
        "load_agree": sum(r["result"] == "agree" for r in report["load"]),
        "load_expected_differences": sum(r["result"].startswith("expected") for r in report["load"]),
        "doc_agree": sum(r["result"] == "agree" for r in report["doc"]),
        "disagree": len(bad)}
    report["pass"] = not bad
    print(json.dumps(report["summary"]))
    if not ONLY:
        json.dump(report, open(os.path.join(HERE, "agreement_cpu.json"), "w"), indent=1)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
