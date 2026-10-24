"""compare_gens (paperkiln tools/colab_chat/tiny_chat_run.py, the spot-check's own comparison) of a carry5
gen.jsonl against another, over the spot-check's 40 ids."""
import json
import sys
sys.path.insert(0, "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools/colab_chat")
import tiny_chat_run as t

a, b, items = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [r["id"] for r in t.read_items(items)[::t.SPOT_EVERY][:t.SPOT_N]]
ga = {r["id"]: r for r in (json.loads(x) for x in open(a) if x.strip())}
gb = {r["id"]: r for r in (json.loads(x) for x in open(b) if x.strip())}
r = t.compare_gens(ga, gb, ids)
print(json.dumps({k: r[k] for k in ("items", "items_identical", "replies", "replies_identical")}))
if len(sys.argv) > 4:
    json.dump(r, open(sys.argv[4], "w"), indent=1)
