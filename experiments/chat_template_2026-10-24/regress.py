#!/usr/bin/env python3
"""Gate (d): receipts that involve no system message stay byte-identical under this change.

    regress.py parity    the committed CPU parity receipts, replayed through this change's server
    regress.py bits      ir_logits, 69d7ca2 against this change, bit for bit

parity: each receipt's prompts go to this change's CPU tinyllama_server with the request its harness sent
(tools/chat_parity.py: greedy, seed 0, 60 new tokens or the receipt's cap, the repetition penalty where the
receipt used one). Each record is rebuilt with ember's fields from the new run and every other field (the
recorded mtstudio side, paperkiln's prompt-token count) from the receipt, serialised as the harness did
(json.dumps, one line per record), and the file is compared with the committed one byte for byte.
bits: ../state_tokens_2026-10-24/bit_regression.py's model and IR list, run by ember main 69d7ca2's ir_logits
(build_base.sh) and by this change's; equal JSON text means equal floats, prefill and decode. swa_tiny and
kimi_tiny are read as relabelled copies (rule 38; ../ir_rev13_2026-10-27/relabel_models.sh, run when missing).
Writes regress_parity.json / regress_bits.json beside this file."""
import json
import os
import subprocess
import sys

from common import A, BASE, FROZEN, HERE, NEW, ember, post, rows, sha256, stop

EXP = os.path.dirname(HERE)
PKB = EXP + "/pk_base_v1_2026-10-24"
NB = A + "/next_base"
DEMO = PKB + "/demo_prompts.jsonl"


def full(i, prompt, e, ref):
    rec = {"i": i, "prompt": prompt, "ember": e["reply"], "mtstudio": ref["mtstudio"], "ember_tokens": e["tokens"],
           "mts_tokens": ref["mts_tokens"], "ember_stop": e["stop_reason"], "mts_stop": ref["mts_stop"],
           "ember_prompt_tokens": e["prompt_tokens"], "bpe_py_prompt_tokens": ref["bpe_py_prompt_tokens"]}
    if "ember_f32" in ref:
        rec["ember_f32"] = ref["ember_f32"]
    return rec


def short(i, prompt, e, ref):
    return {"i": i, "ember": e["reply"], "mtstudio": ref["mtstudio"], "ember_tokens": e["tokens"],
            "ember_stop": e["stop_reason"]}


# receipt, gguf, prompts, max_new, extra request fields, record builder
PARITY = [
    (EXP + "/chat_2026-10-18/parity_cpu_greedy_n40.jsonl", os.path.expanduser("~/ember_ttft/chat/chat_sft_bpe.gguf"),
     FROZEN, 60, {}, full),
    (PKB + "/gate_llama3_parity.jsonl", NB + "/gate/gate_llama3.gguf", FROZEN, 60, {}, full),
    (PKB + "/arxiv_2302_13971_parity.jsonl", NB + "/sft_2302_13971/arxiv_2302_13971.gguf", FROZEN, 60, {}, full),
    (PKB + "/arxiv_2302_13971_demo.jsonl", NB + "/sft_2302_13971/arxiv_2302_13971.gguf", DEMO, 120, {}, full),
    (PKB + "/arxiv_2304_03208_parity.jsonl", NB + "/sft_2304_03208/arxiv_2304_03208.gguf", FROZEN, 60, {}, full),
    (PKB + "/arxiv_2304_03208_demo.jsonl", NB + "/sft_2304_03208/arxiv_2304_03208.gguf", DEMO, 120, {}, full),
    (EXP + "/repeat_2026-10-21/p1.1_cpu.jsonl", A + "/chat_base/sftfinal/chat_sftfinal_v3.gguf", FROZEN, 60,
     {"repeat_penalty": 1.1, "repeat_last_n": 64}, short),
    (EXP + "/repeat_2026-10-21/p1.3_cpu.jsonl", A + "/chat_base/sftfinal/chat_sftfinal_v3.gguf", FROZEN, 60,
     {"repeat_penalty": 1.3, "repeat_last_n": 64}, short),
    (PKB + "/pk_base_v1_q8_0_all_parity.jsonl", NB + "/base/pk_base_v1_q8_0_all.gguf", FROZEN, 60, {}, full),
    (PKB + "/pk_base_v1_parity.jsonl", NB + "/base/pk_base_v1.gguf", FROZEN, 60, {}, full),
]


def parity(only=None):
    out_path = os.path.join(HERE, "regress_parity.json")
    report = json.load(open(out_path)) if os.path.exists(out_path) else {"receipts": {}}  # resumable per receipt
    port = 8141
    for receipt, gguf, prompts_path, max_new, extra, build in PARITY:
        name = os.path.relpath(receipt, EXP)
        if only and only not in name:
            continue
        if name in report["receipts"]:
            continue
        refs = rows(receipt)
        prompts = rows(prompts_path)
        p, h = ember(NEW, gguf, port, "regress_" + os.path.basename(receipt))
        lines = []
        try:
            for ref in refs:
                i = ref["i"]
                body = {"user_input": prompts[i]["prompt"], "max_new_tokens": max_new, "temperature": 1.0, "top_k": 1,
                        "top_p": 1.0, "seed": 0, **extra}
                status, e = post(port, body)
                assert status == 200, (receipt, i, status, e)
                lines.append(json.dumps(build(i, prompts[i]["prompt"], e, ref)) + "\n")
        finally:
            stop(p)
        port += 1
        new = "".join(lines).encode("utf-8")
        old = open(receipt, "rb").read()
        report["receipts"][name] = {"gguf": gguf, "gguf_sha256": sha256(gguf), "records": len(refs),
                                    "receipt_sha256": sha256(receipt), "byte_identical": new == old,
                                    "lines_identical": sum(a == b for a, b in zip(new.split(b"\n"), old.split(b"\n")))}
        print(name, json.dumps(report["receipts"][name]), flush=True)
        json.dump(report, open(out_path + ".tmp", "w"), indent=1)
        os.replace(out_path + ".tmp", out_path)
    report["pass"] = len(report["receipts"]) == len(PARITY) and all(
        r["byte_identical"] for r in report["receipts"].values())
    json.dump(report, open(out_path, "w"), indent=1)
    print("pass", report["pass"])


def bits():
    short_ = [[5, 17, 300, 42, 9, 77, 128, 3], [11, 2, 250, 99, 1]]
    # A recursion document read in place of a file's own (EMBER_IR): the recursion gate's b2x4 (two stored
    # paperkiln blocks, applied 8 times in its own document) applied as 0, 1, 0, 1 (n_layers 4). It conforms to
    # IR_SPEC; the hand-made gpt2_tiny version it replaces broke rules 3 and 27 and is refused at load (README).
    rec = json.load(open(f"{A}/recursion_gate/b2x4/r.ir.json"))
    rec["model"]["n_layers"] = rec["block"]["repeat"] = 4
    rec["block"]["weight_index"] = {"per_layer": [0, 1, 0, 1]}
    rec["block"]["_recursion"]["repeats"] = 2  # the prose annotation, kept true
    rec_path = os.path.expanduser("~/ct24/bitreg_recursion_b2x4_0101.ir.json")
    json.dump(rec, open(rec_path, "w"))
    chat = [[2, 400, 51, 77, 1203, 3, 88, 901, 15, 1, 2, 64, 3000, 3], [2, 17, 18, 19, 3]]

    def seqs(path, n=8):
        return [r["prompt_ids"] + r["continuation_ids"] for r in rows(path)][:n]

    # swa_tiny and kimi_tiny carry general.architecture "gpt2", which rule 38 (keramos v1.3.0) refuses for these
    # IR-only files in every build since 70ed063; both builds read their relabelled copies (architecture
    # "paperkiln", the gpt2.* keys renamed, document and tensors unchanged) from the ext4 cache that
    # ../ir_rev13_2026-10-27/relabel_models.sh makes, as the adapter self-test does. The originals are only read.
    r13 = os.path.expanduser("~/ember_wt/adapter_models_r13")
    if not all(os.path.exists(f"{r13}/{n}/{n}.gguf") for n in ("swa_tiny", "kimi_tiny")):
        subprocess.run(["bash", EXP + "/ir_rev13_2026-10-27/relabel_models.sh"], check=True, stdout=subprocess.DEVNULL)

    cases = {
        "gpt2_tiny": (f"{A}/gpt2_tiny/gpt2_tiny.gguf", None, short_),
        "gpt2_tiny_scale_smoke_ir": (f"{A}/gpt2_tiny/gpt2_tiny.gguf", os.path.expanduser("~/ember_wt/scale_smoke.ir.json"), short_),
        "recursion_b2x4_0101": (f"{A}/recursion_gate/b2x4/r.gguf", rec_path, seqs(f"{A}/recursion_gate/b2x4/score_in.jsonl")),
        "chat_sftfinal_v3": (f"{A}/chat_base/sftfinal/chat_sftfinal_v3.gguf", None, chat),
        "swa_tiny": (f"{r13}/swa_tiny/swa_tiny.gguf", None, short_),
        "kimi_tiny": (f"{r13}/kimi_tiny/kimi_tiny.gguf", None, short_),
        "mamba_tiny": (f"{A}/mamba_tiny/mamba_tiny.gguf", None, short_),
        "moe_tiny_v3": (f"{A}/moe_tiny/moe_tiny_v3.gguf", None, short_),
        "attnres_tiny": (f"{A}/attnres_tiny/attnres_tiny.gguf", None, short_),
        "attnres_gate_ar3": (f"{A}/attnres_gate/ar3/ar3.gguf", None, seqs(f"{A}/attnres_gate/ar3/score_in.jsonl")),
        "state_gate_state4": (f"{A}/state_gate/state4/state4.gguf", None, seqs(f"{A}/state_gate/state4/score_in.jsonl")),
    }
    for m in ("p1_b2x3_c1", "p1_b2x3_c1_inject", "b2x4"):
        cases["recursion_gate_" + m] = (f"{A}/recursion_gate/{m}/r.gguf", None, seqs(f"{A}/recursion_gate/{m}/score_in.jsonl"))
    for m in ("mor3", "mor2p0"):
        cases["mor_gate_" + m] = (f"{A}/mor_gate/{m}/{m}.gguf", None, seqs(f"{A}/mor_gate/{m}/score_in.jsonl"))
    report = {"old": BASE + "/ir_logits (ember 69d7ca2)", "new": NEW + "/ir_logits", "cases": {}}
    for name, (gguf, ir, ss) in cases.items():
        env = {k: v for k, v in os.environ.items() if k != "EMBER_IR"}
        if ir:
            env["EMBER_IR"] = ir
        outs = []
        for b in (BASE, NEW):
            r = subprocess.run([b + "/ir_logits", gguf], input=json.dumps(ss), capture_output=True, text=True, env=env)
            outs.append((r.returncode, r.stdout))
        report["cases"][name] = {"sequences": len(ss), "positions": sum(map(len, ss)), "rc": [outs[0][0], outs[1][0]],
                                 "bit_equal": outs[0] == outs[1] and outs[0][0] == 0}
        print(name, json.dumps(report["cases"][name]), flush=True)
    report["pass"] = all(c["bit_equal"] for c in report["cases"].values())
    json.dump(report, open(os.path.join(HERE, "regress_bits.json"), "w"), indent=1)
    print("pass", report["pass"])


if __name__ == "__main__":
    {"parity": lambda: parity(sys.argv[2] if len(sys.argv) > 2 else None), "bits": bits}[sys.argv[1]]()
