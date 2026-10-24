"""Greedy /chat replies from one server, and comparisons between engines.

    sp_compare.py collect URL PROMPTS.jsonl N MAX_NEW VARIANT OUT.jsonl
    sp_compare.py compare A.jsonl B.jsonl            (prints a JSON summary)

VARIANT spot: the body tools/carry5_eval.py sends (temperature 1, top_k 1, top_p 1, seed 7, return_ids; no
repeat fields, so each engine's default: repeat_penalty 1.0 = off). VARIANT p1: the same plus explicit
repeat_penalty 1.0 and repeat_last_n 64. Every prompt is a first turn (history empty)."""
import json
import sys
import urllib.request


def post(url, body):
    req = urllib.request.Request(url + "/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def collect(url, prompts, n, max_new, variant, out):
    rows = [json.loads(x) for x in open(prompts, encoding="utf-8") if x.strip()][:n]
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            body = {"user_input": r["prompt"], "history": [], "temperature": 1.0, "top_k": 1, "top_p": 1.0, "seed": 7,
                    "max_new_tokens": max_new, "return_ids": True}
            if variant == "p1":
                body.update(repeat_penalty=1.0, repeat_last_n=64)
            e = post(url, body)
            f.write(json.dumps({"i": i, "prompt": r["prompt"], "reply": e["reply"].strip(), "stop": e.get("stop_reason"),
                                "prompt_ids": e.get("prompt_ids"), "ids": e.get("ids")}) + "\n")


def compare(a, b):
    """Replies (text + stop) always; ids and prompt ids where both engines return them (mtstudio's
    /chat returns no ids: then the first divergent step is counted in characters of the reply)."""
    A = [json.loads(x) for x in open(a) if x.strip()]
    B = [json.loads(x) for x in open(b) if x.strip()]
    same, ids_same, prompt_same, have_ids, first = 0, 0, 0, 0, []
    for x, y in zip(A, B):
        ok = x["reply"] == y["reply"] and x["stop"] == y["stop"]
        same += ok
        both = x.get("ids") is not None and y.get("ids") is not None
        if both:
            have_ids += 1
            ids_same += x["ids"] == y["ids"]
            prompt_same += x["prompt_ids"] == y["prompt_ids"]
        if not ok or (both and x["ids"] != y["ids"]):
            if both:
                k = next((j for j, (p, q) in enumerate(zip(x["ids"], y["ids"])) if p != q), min(len(x["ids"]), len(y["ids"])))
                unit = "token"
            else:
                k = next((j for j, (p, q) in enumerate(zip(x["reply"], y["reply"])) if p != q), min(len(x["reply"]), len(y["reply"])))
                unit = "char"
            first.append({"i": x["i"], "first_divergent": k, "unit": unit, "a": x["reply"][:90], "b": y["reply"][:90]})
    return {"a": a, "b": b, "n": len(A), "replies_identical": same, "with_ids": have_ids, "ids_identical": ids_same,
            "prompt_ids_identical": prompt_same, "divergent": first}


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7])
    else:
        print(json.dumps(compare(sys.argv[2], sys.argv[3]), indent=1))
