#!/usr/bin/env python3
"""Rule citations (IR_SPEC 10.1, revision 1.2), ember against the reference validator: keramos's
tools/ir_validate.py doc_errors at the tag, called as a black box. One-edit mutations of the suite's documents (the
version, the model's fields, sections, ops, attributes and their closed sets, arity, slots, recurrences, weight_index,
depth_route, reads of unwritten values) and every refusal document of the suite, each checked by ember's
ir_doc_check (the document rules, with rule 35's parse) and by doc_errors. A case agrees when both accept or both
refuse and every rule number ember's messages cite is one the reference cites (ember names the rule in its schema
failures and its rule 7, 13, 20-37 messages; others carry the section alone).
Usage (WSL): python3 rule_citations.py BUILD_DIR [OUT.json]   (KERAMOS: the tag's archive, default ~/keramos_v1.2.0)"""
import copy
import json
import os
import re
import subprocess
import sys
import tempfile

K = os.path.expanduser(os.environ.get("KERAMOS", "~/keramos_v1.2.0"))
sys.path.insert(0, os.path.join(K, "tools"))
import ir_validate as V  # noqa: E402

B = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
SUITE = os.path.join(K, "conformance")
RX = re.compile(r"rule (\d+)")


def rules(msgs):
    return sorted({int(x) for m in msgs for x in RX.findall(m)})


def doc_of(model):
    from pathlib import Path
    from ir_conformance import load_model
    return load_model(Path(os.path.join(SUITE, "models", model)))[0]


def node_index(d, op, sect="block"):
    for j, n in enumerate(d[sect]["nodes"]):
        if n.get("op") == op:
            return j
    raise KeyError(op)


def mut(d, f):
    d = copy.deepcopy(d)
    f(d)
    return d


def setk(o, k, v):
    o[k] = v


def delk(o, k):
    o.pop(k)


docs = {m: doc_of(m) for m in ("pk_llama", "pk_mor", "pk_recurrence", "pk_state", "pk_attnres", "pk_kimi", "pk_moe")}
L = docs["pk_llama"]
cases = []


def add(name, d):
    cases.append((name, d))


def blk(d, op):
    return d["block"]["nodes"][node_index(d, op)]


add("top_unknown", mut(L, lambda d: setk(d, "extras", 1)))
add("ir_version_missing", mut(L, lambda d: delk(d, "ir_version")))
add("ir_version_string", mut(L, lambda d: setk(d, "ir_version", "1")))
add("ir_version_negative", mut(L, lambda d: setk(d, "ir_version", -1)))
add("family_hint_missing", mut(L, lambda d: delk(d, "family_hint")))
add("family_hint_int", mut(L, lambda d: setk(d, "family_hint", 3)))
for k in ("vocab", "d", "n_layers", "ctx"):
    add(f"model_{k}_missing", mut(L, lambda d, k=k: delk(d["model"], k)))
    add(f"model_{k}_zero", mut(L, lambda d, k=k: setk(d["model"], k, 0)))
    add(f"model_{k}_string", mut(L, lambda d, k=k: setk(d["model"], k, "x")))
add("model_unknown", mut(L, lambda d: setk(d["model"], "hidden", 1)))
add("model_dtype_f16", mut(L, lambda d: setk(d["model"], "dtype", "f16")))
add("block_unknown", mut(L, lambda d: setk(d["block"], "loop", 1)))
add("block_repeat_string", mut(L, lambda d: setk(d["block"], "repeat", "2")))
add("block_repeat_zero", mut(L, lambda d: setk(d["block"], "repeat", 0)))
add("block_nodes_missing", mut(L, lambda d: delk(d["block"], "nodes")))
add("embed_unknown", mut(L, lambda d: setk(d["embed"], "loop", 1)))
add("head_missing", mut(L, lambda d: delk(d, "head")))
add("embed_missing", mut(L, lambda d: delk(d, "embed")))
add("tokenizer_unknown", mut(L, lambda d: setk(d.setdefault("tokenizer", {}), "zz", 1)))
add("op_unknown", mut(L, lambda d: setk(blk(d, "rmsnorm"), "op", "warp")))
add("op_int", mut(L, lambda d: setk(blk(d, "rmsnorm"), "op", 3)))
add("attr_unknown", mut(L, lambda d: setk(blk(d, "attention"), "flavour", 1)))
add("attr_of_other_op", mut(L, lambda d: setk(blk(d, "rmsnorm"), "theta", 1)))
add("rmsnorm_eps_missing", mut(L, lambda d: delk(blk(d, "rmsnorm"), "eps")))
add("rmsnorm_eps_string", mut(L, lambda d: setk(blk(d, "rmsnorm"), "eps", "x")))
add("rope_theta_missing", mut(L, lambda d: delk(blk(d, "rope"), "theta")))
add("rope_pairing_bad", mut(L, lambda d: setk(blk(d, "rope"), "pairing", "x")))
add("rope_heads_bad", mut(L, lambda d: setk(blk(d, "rope"), "rope_heads", "some")))
add("rope_heads_string", mut(L, lambda d: setk(blk(d, "rope"), "heads", "x")))
add("rope_n_rot_odd", mut(L, lambda d: setk(blk(d, "rope"), "n_rot", 3)))
add("attention_mask_bad", mut(L, lambda d: setk(blk(d, "attention"), "mask", "bidirectional")))
add("attention_causal_false", mut(L, lambda d: setk(blk(d, "attention"), "causal", False)))
add("attention_scale_missing", mut(L, lambda d: delk(blk(d, "attention"), "scale")))
add("attention_state_missing", mut(L, lambda d: delk(blk(d, "attention"), "state")))
add("attention_window_negative", mut(L, lambda d: setk(blk(d, "attention"), "window", -1)))
add("node_in_missing", mut(L, lambda d: delk(blk(d, "rmsnorm"), "in")))
add("node_out_missing", mut(L, lambda d: delk(blk(d, "rmsnorm"), "out")))
add("node_in_string", mut(L, lambda d: setk(blk(d, "rmsnorm"), "in", "x")))
add("node_out_list", mut(L, lambda d: setk(blk(d, "rmsnorm"), "out", ["h"])))
add("add_one_input", mut(L, lambda d: setk(blk(d, "add"), "in", ["x"])))
add("tensor_index_bad", mut(L, lambda d: setk(blk(d, "rmsnorm"), "tensor_index", "x")))
add("slot_kind_bad", mut(L, lambda d: setk(d["block"]["state"][0], "kind", "paged")))
add("slot_kind_missing", mut(L, lambda d: delk(d["block"]["state"][0], "kind")))
add("slot_unknown", mut(L, lambda d: setk(d["block"]["state"][0], "pages", 1)))
for k in ("heads", "max_len", "window"):
    add(f"kv_{k}_missing", mut(L, lambda d, k=k: delk(d["block"]["state"][0], k)))
add("kv_heads_zero", mut(L, lambda d: setk(d["block"]["state"][0], "heads", 0)))
add("kv_dtype_f16", mut(L, lambda d: setk(d["block"]["state"][0], "dtype", "f16")))
add("kv_evict_lru", mut(L, lambda d: setk(d["block"]["state"][0], "evict", "lru")))
add("kv_rows_all", mut(L, lambda d: setk(d["block"]["state"][0], "kv_rows", "all")))
add("kv_from", mut(L, lambda d: setk(d["block"]["state"][0], "kv_from", {"per_layer": [-1] * d["block"]["repeat"]})))
add("per_layer_head", mut(L, lambda d: setk(d["head"]["nodes"][0], "eps", {"per_layer": [1e-5]})))
add("per_layer_length", mut(L, lambda d: setk(blk(d, "rmsnorm"), "eps", {"per_layer": [1e-5]})))
add("weight_index_string", mut(L, lambda d: setk(d["block"], "weight_index", {"per_layer": "x"})))
add("weight_index_strings", mut(L, lambda d: setk(d["block"], "weight_index", {"per_layer": ["a"] * d["block"]["repeat"]})))
add("weight_index_negative", mut(L, lambda d: setk(d["block"], "weight_index", {"per_layer": [-1] * d["block"]["repeat"]})))
add("reads_unwritten", mut(L, lambda d: setk(blk(d, "add"), "in", ["x", "nowhere"])))
add("head_no_logits", mut(L, lambda d: setk(d["head"]["nodes"][-1], "out", "lg")))
add("embed_no_x", mut(L, lambda d: setk(d["embed"]["nodes"][-1], "out", "e0")))
R = docs["pk_recurrence"]
try:
    rj = node_index(R, "recurrence")
    rec = lambda d: d["block"]["nodes"][rj]  # noqa: E731
    add("rec_index_zero", mut(R, lambda d: setk(rec(d), "index", {k: 0 for k in rec(d)["index"]})))
    add("rec_inputs_empty", mut(R, lambda d: setk(rec(d), "inputs", [])))
    add("rec_update_empty", mut(R, lambda d: setk(rec(d), "update", [])))
    add("rec_readout_int", mut(R, lambda d: setk(rec(d), "readout", 5)))
    add("rec_weights_int", mut(R, lambda d: setk(rec(d), "weights", {"W[c]": 3})))
    add("rec_index_missing", mut(R, lambda d: delk(rec(d), "index")))
    rs = [i for i, s in enumerate(R["block"]["state"]) if s["kind"] == "recurrent"][0]
    add("rec_slot_dtype", mut(R, lambda d: setk(d["block"]["state"][rs], "dtype", "f16")))
    add("rec_slot_op_missing", mut(R, lambda d: delk(d["block"]["state"][rs], "op")))
    add("rec_slot_tensors_empty", mut(R, lambda d: setk(d["block"]["state"][rs], "tensors", [])))
    add("rec_slot_shape_zero", mut(R, lambda d: setk(d["block"]["state"][rs]["tensors"][0], "shape", [0])))
    add("rec_slot_init_one", mut(R, lambda d: setk(d["block"]["state"][rs]["tensors"][0], "init", 1)))
    add("rec_slot_unknown", mut(R, lambda d: setk(d["block"]["state"][rs], "pages", 1)))
except KeyError:
    pass
M_ = docs["pk_mor"]
dr = M_["block"]["nodes"][0]
add("mor_route_bad", mut(M_, lambda d: setk(d["block"]["nodes"][0], "route", "expert_choice")))
add("mor_gate_bad", mut(M_, lambda d: setk(d["block"]["nodes"][0], "gate", "soft")))
add("mor_n_rec_zero", mut(M_, lambda d: setk(d["block"]["nodes"][0], "n_rec", 0)))
add("mor_step_gap", mut(M_, lambda d: setk(d["block"]["nodes"][0], "step",
                                          {"per_layer": [-1, 0, -1, 1] + [1] * (d["block"]["repeat"] - 4)})))
add("mor_out_missing", mut(M_, lambda d: delk(d["block"]["nodes"][0], "out")))
add("mor_reads_embed_value", mut(M_, lambda d: d["block"]["nodes"][-1]["in"].__setitem__(-1, d["embed"]["nodes"][0]["out"])))
A_ = docs["pk_attnres"]
try:
    j = node_index(A_, "gelu")
    add("gelu_approx_bad", mut(A_, lambda d: setk(d["block"]["nodes"][j], "approx", "fast")))
except KeyError:
    pass
MO = docs["pk_moe"]
try:
    j = node_index(MO, "moe_ffn")
    add("moe_gate_norm_bad", mut(MO, lambda d: setk(d["block"]["nodes"][j], "gate_norm", "x")))
    add("moe_activation_bad", mut(MO, lambda d: setk(d["block"]["nodes"][j], "activation", "x")))
    add("moe_top_k_big", mut(MO, lambda d: setk(d["block"]["nodes"][j], "top_k", 99)))
except KeyError:
    pass
KI = docs["pk_kimi"]
try:
    j = node_index(KI, "linear_attention")
    add("linear_attention_causal_false", mut(KI, lambda d: setk(d["block"]["nodes"][j], "causal", False)))
    add("linear_attention_feature_map_bad", mut(KI, lambda d: setk(d["block"]["nodes"][j], "feature_map", "x")))
except KeyError:
    pass
for name in sorted(os.listdir(os.path.join(SUITE, "refuse"))):
    if name.endswith(".json"):
        c = json.load(open(os.path.join(SUITE, "refuse", name)))
        if "doc" in c:
            add("refuse/" + name[:-5], c["doc"])

results, agree = [], 0
with tempfile.TemporaryDirectory() as tmp:
    paths = []
    for i, (name, d) in enumerate(cases):
        p = os.path.join(tmp, f"{i:03d}.ir.json")
        json.dump(d, open(p, "w"))
        paths.append(p)
    out = subprocess.run([os.path.join(B, "ir_doc_check")] + paths, capture_output=True, text=True).stdout.splitlines()
    ember = [json.loads(x) for x in out]
for (name, d), e in zip(cases, ember):
    ref = V.doc_errors(d)
    re_, rr = rules(e["errors"]), rules(ref)
    # ember need not name a rule in every message; a rule it names must be one the reference names
    ok = (bool(e["errors"]) == bool(ref)) and set(re_) <= set(rr)
    agree += ok
    results.append({"case": name, "agree": ok, "ember_rules": re_, "ref_rules": rr, "ember": e["errors"][:3],
                    "ref": ref[:3]})
    if not ok:
        print(f"DIFF {name}: ember {re_} {e['errors'][:2]}\n      ref {rr} {ref[:2]}")
print(f"{agree}/{len(cases)} mutations name the same rules (and accept or refuse alike)")
if OUT:
    json.dump({"keramos": open(os.path.join(K, "COMMIT")).read().strip() if os.path.isfile(os.path.join(K, "COMMIT"))
               else None, "cases": len(cases), "agree": agree, "results": results}, open(OUT, "w"), indent=1)
sys.exit(0 if agree == len(cases) else 1)
