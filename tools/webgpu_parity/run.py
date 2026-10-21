"""Serve the WebGPU parity page and read its RESULT line through Playwright (Edge). Usage: run.py PORT"""
import functools
import http.server
import json
import os
import sys
import threading

from playwright.sync_api import sync_playwright

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8098
here = os.path.dirname(os.path.abspath(__file__))
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=here)
handler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True, args=["--enable-unsafe-webgpu"])
    page = b.new_page()
    page.on("console", lambda m: print("console:", m.text) if m.type in ("error", "warning") else None)
    page.goto(f"http://127.0.0.1:{port}/index.html")
    page.wait_for_function("document.getElementById('out').textContent !== 'pending'", timeout=180000)
    text = page.inner_text("#out")
    b.close()
srv.shutdown()
if text.startswith("RESULT "):
    r = json.loads(text[7:])
    print(json.dumps({k: r[k] for k in ("adapter", "load_ms", "decode_ms_per_token")}), json.dumps(r["config"]))
    for x in r["prompts"]:
        print(f"prompt {x['i']}: {x['tokens']} tokens, max |d| {x['max_abs']:.2e} ({x['of_range']:.2e} of range), "
              f"top-1 {'ok' if x['top1'] else 'DIFF'}, {x['ms']:.0f} ms")
else:
    print(text[:3000])
