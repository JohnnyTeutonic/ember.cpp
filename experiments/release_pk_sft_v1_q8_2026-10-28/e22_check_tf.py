import json, numpy as np
R = "/home/jonat/e22rel/run/"
f = [json.loads(l) for l in open(R + "f32_server.jsonl")]
meta = json.load(open(R + "tf_f32.f32.json"))
A = np.fromfile(R + "tf_f32.f32", dtype=np.float32).reshape(-1, 16384)
B = np.fromfile(R + "tf_q8.f32", dtype=np.float32).reshape(-1, 16384)
o = 0; agree = 0; total = 0; dis = []
for r, m in zip(f, meta):
    tgt = r["ids"] + ([r["stop_id"]] if r["stop_id"] >= 0 else [])
    am = A[o:o + m["rows"]].argmax(1)
    for j, t in enumerate(tgt):
        total += 1; agree += int(am[j] == t)
    bm = B[o:o + m["rows"]].argmax(1)
    for j in range(m["rows"]):
        if am[j] != bm[j]:
            dis.append((r["i"], j))
    o += m["rows"]
print("score-path F32 argmax == served F32 greedy token:", agree, "/", total)
print("Q8 top-1 disagreements (prompt, position):", dis)
