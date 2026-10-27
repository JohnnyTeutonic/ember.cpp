"""Each text endpoint alone against a fresh server (a build without the fix kills the process on some): k=12 and 64."""
import json
import os
import sys

sys.path.insert(0, "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/ember.cpp/tools")
import utf8_server_check as u  # noqa: E402

server, model, port, work = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
out = {}
for k in (12, 64):
    chat = dict(u.SAMPLING, user_input=u.USER, max_new_tokens=k)
    oa = dict(u.SAMPLING, messages=[{"role": "user", "content": u.USER}], max_tokens=k)
    for name, path, body in (("chat", "/chat", chat), ("chat_stream", "/chat", dict(chat, stream=True)),
                             ("v1", "/v1/chat/completions", oa),
                             ("v1_stream", "/v1/chat/completions", dict(oa, stream=True))):
        p = u.start(server, model, port, work)
        s, b = u.request(port, path, body)
        alive = p.poll() is None
        try:
            u.urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
        except Exception:
            alive = False
        p.terminate()
        p.wait()
        tail = b[-160:].decode("utf-8", "replace")
        out[f"{name}@{k}"] = {"status": s, "server_alive_after": alive, "tail": tail}
        print(name, k, s, "alive" if alive else "SERVER DIED", tail[-120:].replace("\n", " "))
        try:
            os.remove(os.path.join(work, "debugging.log"))
        except OSError:
            pass
json.dump(out, open(os.path.join(work, "before_probe.json"), "w"), indent=1)
