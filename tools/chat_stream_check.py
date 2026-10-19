"""Streaming /chat ("stream": true, server-sent events) against the plain
reply, request by request, with the same seed: the joined {"token"} pieces
must equal the final reply (up to trailing whitespace, which the stream may
carry and the reply strips), and the final reply must equal the
non-streaming reply. Also times the first piece and the end at the client.
Usage: chat_stream_check.py PORT [N]
"""
import json
import statistics
import sys
import time
import urllib.request

PROMPTS = "/mnt/c/ml_artifacts/chat_base/sft/tinychat_hf_prompts_v1.jsonl"
port = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rows = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")][:N]


def body(i, stream):
    hist = [{"role": "user", "content": rows[(i + 1) % N]["prompt"]},
            {"role": "assistant", "content": rows[(i + 1) % N]["reference"]}]
    return json.dumps({"user_input": rows[i]["prompt"], "history": hist, "max_new_tokens": 64,
                       "temperature": 0.7, "top_k": 40, "top_p": 0.9, "seed": 77 + i,
                       "stream": stream}).encode()


def plain(i):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=body(i, False),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return d, (time.perf_counter() - t0) * 1000


def streamed(i):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/chat", data=body(i, True),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first, pieces, final, buf = None, [], None, b""
    with urllib.request.urlopen(req, timeout=300) as r:
        assert r.headers.get("Content-Type", "").startswith("text/event-stream")
        while True:
            chunk = r.read1(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                if not ev.startswith(b"data: "):
                    continue
                d = json.loads(ev[6:].decode("utf-8"))
                if d.get("done"):
                    final = d
                else:
                    if t_first is None:
                        t_first = (time.perf_counter() - t0) * 1000
                    pieces.append(d["token"])
    return pieces, final, t_first, (time.perf_counter() - t0) * 1000


ok_join = ok_same = 0
firsts, ends, plain_walls = [], [], []
for i in range(N):
    p, pw = plain(i)
    pieces, final, t_first, t_end = streamed(i)
    joined = "".join(pieces)
    ok_join += joined.rstrip() == final["reply"]
    ok_same += final["reply"] == p["reply"]
    if t_first is not None:
        firsts.append(t_first)
    ends.append(t_end)
    plain_walls.append(pw)
    if joined.rstrip() != final["reply"] or final["reply"] != p["reply"]:
        print(f"[{i}] stream {joined!r}\n     final  {final['reply']!r}\n     plain  {p['reply']!r}")
print(f"n={N}: joined pieces == final reply {ok_join}; final == non-stream reply {ok_same}")
print(f"client: first piece p50 {statistics.median(firsts):.1f} ms (max {max(firsts):.1f}); "
      f"stream end p50 {statistics.median(ends):.1f} ms; non-stream reply p50 {statistics.median(plain_walls):.1f} ms")
