#!/usr/bin/env python3
"""G5, cross-engine template parity, ember against mtstudio (LOCAL_LLM_STACK.md section 3, gates):

    g5_mtstudio.py TAG GGUF RUN_DIR NAME

- ember: this change's CPU tinyllama_server on GGUF; mtstudio: `mtstudio chat SPEC` at paperkiln HEAD
  (build_mtstudio.sh; full recompute), both /health checked (the template both report). SPEC is the
  run's own spec re-pointed at RUN_DIR (out_dir, name NAME, data.vocab = RUN_DIR/NAME.gguf), as
  ../pk_base_v1_2026-10-24/gate.py writes it.
- Two prompt sets: the 40 frozen parity prompts, and system_set_v1.jsonl (make_system_set.py: 60
  requests, every one with a system message: 40 frozen prompts with one of four system texts in the
  "system" field, 8 with a leading {"role": "system"} history entry and one exchange, 8 with "system"
  and two exchanges, 4 with "system" and six exchanges, long enough for the context rule to drop
  exchanges under the pinned system message).
- Prompt ids: ember's /chat return_ids against mtstudio's own rendering and tokeniser
  (`mtstudio sample-set` on the same conversation as a {"messages"} row: chat::render of the model's
  template, then mtstudio's tokenize), 1 sample of 1 token; the /chat replies of both (greedy, 60 new
  tokens): reply text, token count, stop reason and history_turns_used.
Writes g5_mtstudio_TAG.json and g5_mtstudio_TAG_rows.jsonl beside this file."""
import json
import os
import subprocess
import sys

from common import FROZEN, HERE, MTS, NEW, PK, WORK, ember, git_rev, greedy, post, rows, sha256, start, stop

tag, gguf, run_dir, name = sys.argv[1:5]
spec_src = next(c for c in (f"{run_dir}/{os.path.basename(run_dir)}_spec.json", f"{run_dir}/sft_2302_13971_spec.json")
                if os.path.exists(c))
sp = json.load(open(spec_src))
sp["out_dir"], sp["name"] = run_dir, name
sp.setdefault("data", {})["vocab"] = os.path.join(run_dir, name + ".gguf")
os.makedirs(WORK + "/specs", exist_ok=True)
spec = f"{WORK}/specs/{tag}.json"
json.dump(sp, open(spec, "w"), indent=1)
sets ={"frozen40": [{"user_input": r["prompt"]} for r in rows(FROZEN, 40)],
        "system_set_v1": rows(os.path.join(HERE, "system_set_v1.jsonl"))}


def messages(req, kept):
    """The /chat request as the prompt conversation both engines render (chat_request.h, mtstudio chat),
    with only the newest `kept` exchanges of its history: the ones mtstudio's own context rule kept (its
    history_turns_used), so the system message folds into the first kept exchange as it did in /chat."""
    m, hist = [], list(req.get("history", []))
    if req.get("system"):
        m.append({"role": "system", "content": req["system"]})
    if hist and hist[0].get("role") == "system":
        m.append({"role": "system", "content": hist.pop(0)["content"]})
    hist = hist[len(hist) - 2 * kept:]
    for h in hist:
        m.append({"role": h["role"], "content": h["content"]})
    m.append({"role": "user", "content": req["user_input"]})
    return m


report = {"tag": tag, "gguf": gguf, "gguf_sha256": sha256(gguf), "spec": sp, "spec_source": spec_src,
          "paperkiln_mtstudio_build": open(os.path.expanduser("~/pk_cpu_ct/REV")).read().strip(),
          "paperkiln_head_now": git_rev(PK),
          "mtstudio": MTS, "ember": NEW, "sets": {}}
allreq = [(s, i, r) for s, rs in sets.items() for i, r in enumerate(rs)]
rows_path = os.path.join(HERE, f"g5_mtstudio_{tag}_rows.jsonl")
health_path = f"{WORK}/g5_{tag}_health.json"
if os.path.exists(rows_path) and os.path.exists(health_path) and len(rows(rows_path)) == len(allreq):
    # Resume: the /chat replies of a finished run (both servers), only mtstudio's prompt ids redone.
    recs = rows(rows_path)
    report["health"] = json.load(open(health_path))
else:
    procs, recs = [], []
    try:
        pe, he = ember(NEW, gguf, 8121, f"g5_{tag}_ember")
        procs.append(pe)
        pm, hm = start([MTS, "chat", spec, "--port", "8122"], 8122, f"{WORK}/logs/g5_{tag}_mtstudio.log")
        procs.append(pm)
        report["health"] = {"ember": he, "mtstudio": hm}
        json.dump(report["health"], open(health_path, "w"))
        for s, i, req in allreq:
            se, e = post(8121, greedy(req))
            sm, m = post(8122, greedy(req))
            assert se == 200 and sm == 200, (s, i, se, e, sm, m)
            recs.append({"set": s, "i": i, "request": req, "ember_prompt_ids": e["prompt_ids"],
                         "ember": e["reply"], "mtstudio": m["reply"],
                         "ember_tokens": e["tokens"], "mts_tokens": m["tokens"], "ember_stop": e["stop_reason"],
                         "mts_stop": m["stop_reason"], "ember_turns_used": e["history_turns_used"],
                         "mts_turns_used": m["history_turns_used"]})
    finally:
        for p in procs:
            stop(p)
# mtstudio's prompt ids: sample-set (its own rendering and tokeniser) on each conversation as mtstudio's
# /chat kept it.
sp_in, sp_out = f"{WORK}/g5_{tag}_sample_in.jsonl", f"{WORK}/g5_{tag}_sample_out.jsonl"
with open(sp_in, "w", encoding="utf-8") as f:
    for x in recs:
        f.write(json.dumps({"messages": messages(x["request"], x["mts_turns_used"]), "uri": f"{x['set']}/{x['i']}"},
                           ensure_ascii=False) + "\n")
r = subprocess.run([MTS, "sample-set", spec, "--prompts", sp_in, "--out", sp_out, "--n", "1", "--max-new", "1",
                    "--temp", "1"], capture_output=True, text=True)
if r.returncode != 0:
    sys.exit("sample-set failed: " + r.stderr[-2000:])
mts_ids = {x["prompt_uri"]: x["prompt_ids"] for x in rows(sp_out)}
for x in recs:
    x["mts_prompt_ids"] = mts_ids[f"{x['set']}/{x['i']}"]
for s in sets:
    rs = [x for x in recs if x["set"] == s]
    report["sets"][s] = {
        "n": len(rs),
        "prompt_ids_identical": sum(x["ember_prompt_ids"] == x["mts_prompt_ids"] for x in rs),
        "reply_identical": sum(x["ember"] == x["mtstudio"] for x in rs),
        "token_count_identical": sum(x["ember_tokens"] == x["mts_tokens"] for x in rs),
        "stop_reason_identical": sum(x["ember_stop"] == x["mts_stop"] for x in rs),
        "history_turns_used_identical": sum(x["ember_turns_used"] == x["mts_turns_used"] for x in rs),
        "distinct_replies": len({x["ember"] for x in rs}),
        "stops": {k: sum(x["ember_stop"] == k for x in rs) for k in ("eos", "turn", "length")}}
report["template_reported"] = {"ember": report["health"]["ember"].get("chat_template"),
                               "mtstudio": report["health"]["mtstudio"].get("chat_template")}
report["pass"] = report["template_reported"]["ember"] == report["template_reported"]["mtstudio"] and all(
    v["n"] == v[k] for v in report["sets"].values() for k in
    ("prompt_ids_identical", "reply_identical", "token_count_identical", "stop_reason_identical",
     "history_turns_used_identical"))
with open(os.path.join(HERE, f"g5_mtstudio_{tag}_rows.jsonl"), "w", encoding="utf-8") as f:
    for x in recs:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(json.dumps({"sets": report["sets"], "template_reported": report["template_reported"], "pass": report["pass"]},
                 indent=1))
json.dump(report, open(os.path.join(HERE, f"g5_mtstudio_{tag}.json"), "w", encoding="utf-8"), indent=1,
          ensure_ascii=False)
