"""Serve the WebGPU parity page and read its RESULT line through Playwright (Edge). Usage: run.py PORT

A browser run uses whatever GPU the browser picks: on the 21 Oct 2026 runs headless Edge took the
laptop's Intel iGPU, and with --use-angle=swiftshader / --use-webgpu-adapter=swiftshader Chrome 156
on Windows still opened a D3D context on the Intel GPU and offered no SwiftShader adapter (26 Oct).
GPU work is remote only, so this refuses to run unless EMBER_WEBGPU_REMOTE=1 says the machine is a
GPU VM. On the laptop, gate the WebGPU code on the software adapter instead: swiftshader_run.py with
ir_gate.mjs or chat_test.mjs (Dawn's node bindings, SwiftShader the only Vulkan driver)."""
import functools
import http.server
import json
import os
import sys
import threading

if os.environ.get("EMBER_WEBGPU_REMOTE") != "1":
    sys.exit("run.py drives a browser on a hardware GPU: remote GPU VMs only (set EMBER_WEBGPU_REMOTE=1 there). "
             "Locally use swiftshader_run.py -- node ir_gate.mjs / chat_test.mjs.")

from playwright.sync_api import sync_playwright  # noqa: E402

PAGE = os.environ.get("PAGE", "index.html")
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
    page.goto(f"http://127.0.0.1:{port}/" + PAGE + "")
    page.wait_for_function("document.getElementById('out').textContent !== 'pending'", timeout=180000)
    text = page.inner_text("#out")
    b.close()
srv.shutdown()
if text.startswith("RESULT ") and PAGE != "index.html":
    print(text[7:])
elif text.startswith("RESULT "):
    r = json.loads(text[7:])
    print(json.dumps({k: r[k] for k in ("adapter", "load_ms", "decode_ms_per_token")}), json.dumps(r["config"]))
    for x in r["prompts"]:
        print(f"prompt {x['i']}: {x['tokens']} tokens, max |d| {x['max_abs']:.2e} ({x['of_range']:.2e} of range), "
              f"top-1 {'ok' if x['top1'] else 'DIFF'}, {x['ms']:.0f} ms")
else:
    print(text[:3000])
