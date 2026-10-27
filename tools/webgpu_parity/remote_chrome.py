"""Headless Chrome on a remote GPU VM's hardware WebGPU adapter (E6, 27 Oct 2026). Remote VMs only.

    EMBER_WEBGPU_REMOTE=1 python3 remote_chrome.py --root DIR --page 'gate_page.html?model=...' --out OUT.json
        [--rows ROWS.jsonl] [--timeout 3600] [--isolate] [--allow-software] [--chrome PATH] [--arg=--flag ...]

Serves DIR on 127.0.0.1 (with --isolate the COOP/COEP headers that make the page cross-origin isolated,
which the threaded WebAssembly build needs), starts Chrome headless with Vulkan and WebGPU on and a
DevTools port, reads the browser version, its command line and its GPU information over the DevTools
protocol (Browser.getVersion, Browser.getBrowserCommandLine, SystemInfo.getInfo; Playwright's
connect_over_cdp), then opens the page and waits for it to POST /e6/result.

Page endpoints (e6_common.js): POST /e6/row appends one JSON line to --rows (GET /e6/rows reads them back,
so a page resumes where a stopped run left off); POST /e6/file?name=N writes a text file beside --out;
POST /e6/log and the page's console go to OUT.log; POST /e6/result ends the run.

OUT.json: {"result": the page's result, "chrome": {version, command line, gpu}, "page", "wall_s"}.
Exit 3 when the page's adapter is a software one (SwiftShader, llvmpipe, lavapipe) unless
--allow-software (or --wasm-only: a WebAssembly-only page has no adapter); 1 when the result has pass false or the run timed out or crashed; else 0.

GPU work is remote only (Jonathan, 26 Oct 2026): this refuses to run without EMBER_WEBGPU_REMOTE=1,
which only the GPU VM's scripts set. On the laptop use swiftshader_run.py (Dawn on SwiftShader).
"""
import argparse
import functools
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

if os.environ.get("EMBER_WEBGPU_REMOTE") != "1":
    sys.exit("remote_chrome.py drives Chrome on a hardware GPU: remote GPU VMs only (EMBER_WEBGPU_REMOTE=1). "
             "Locally use swiftshader_run.py -- node ir_gate.mjs / chat_test.mjs.")

SOFTWARE = re.compile(r"swiftshader|llvmpipe|lavapipe|softpipe|microsoft basic", re.I)
# Chrome's WebGPU on Linux runs on Vulkan; ANGLE on Vulkan as well, and no surface (headless).
GPU_FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUDeveloperFeatures", "--use-angle=vulkan",
             "--use-vulkan=native", "--disable-vulkan-surface", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--isolate", action="store_true")
    ap.add_argument("--allow-software", action="store_true")
    ap.add_argument("--chrome", default=shutil.which("google-chrome") or "/usr/bin/google-chrome")
    ap.add_argument("--arg", action="append", default=[], help="an extra Chrome flag (--arg=--flag)")
    ap.add_argument("--wasm-only", action="store_true", help="a WebAssembly-only page: no adapter to check")
    ap.add_argument("--no-gpu-flags", action="store_true", help="leave out GPU_FLAGS (probe designs)")
    a = ap.parse_args()
    a.rows = a.rows or os.path.splitext(a.out)[0] + ".rows.jsonl"
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.splitext(a.out)[0] + ".log", "a", encoding="utf-8")
    lock = threading.Lock()
    done = threading.Event()
    box = {}

    def logline(s):
        with lock:
            logf.write(f"{time.strftime('%H:%M:%S')} {s}\n")
            logf.flush()

    class H(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            if a.isolate:
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            super().end_headers()

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/e6/rows"):
                rows = []
                if a.rows and os.path.exists(a.rows):
                    rows = [json.loads(x) for x in open(a.rows, encoding="utf-8") if x.strip()]
                body = json.dumps(rows).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n).decode("utf-8", "replace")
            u = urllib.parse.urlparse(self.path)
            if u.path == "/e6/row":
                json.loads(body)
                with lock:
                    with open(a.rows, "a", encoding="utf-8") as f:
                        f.write(body.strip() + "\n")
            elif u.path == "/e6/file":
                name = os.path.basename(urllib.parse.parse_qs(u.query)["name"][0])
                with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
                    f.write(body)
            elif u.path == "/e6/log":
                logline("page: " + json.loads(body).get("msg", "")[:4000])
            elif u.path == "/e6/result":
                box["result"] = json.loads(body)
                done.set()
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(204)
            self.end_headers()

    port = free_port()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H, directory=a.root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    dport = free_port()
    profile = tempfile.mkdtemp(prefix="e6chrome_")
    argv = [a.chrome, "--headless=new", "--no-sandbox", "--no-first-run", "--no-default-browser-check",
            "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows", f"--remote-debugging-port={dport}",
            f"--user-data-dir={profile}", "--enable-logging=stderr", "--v=0",
            *([] if a.no_gpu_flags else GPU_FLAGS), *a.arg, "about:blank"]
    logline("chrome: " + " ".join(argv))
    chrome_err = open(os.path.splitext(a.out)[0] + ".chrome.log", "w", encoding="utf-8")
    proc = subprocess.Popen(argv, stdout=chrome_err, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    rec = {"page": a.page, "isolate": a.isolate}
    rc = 1
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{dport}/json/version", timeout=2).read()
                break
            except OSError:
                time.sleep(0.5)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{dport}")
            cdp = b.new_browser_cdp_session()
            ver = cdp.send("Browser.getVersion")
            cmdline = argv   # Browser.getBrowserCommandLine needs --enable-automation
            gpu = cdp.send("SystemInfo.getInfo").get("gpu", {})
            rec["chrome"] = {"version": ver, "command_line": cmdline,
                             "gpu": {k: gpu.get(k) for k in ("devices", "driverBugWorkarounds", "featureStatus",
                                                             "auxAttributes")}}
            ctx = b.contexts[0] if b.contexts else b.new_context()
            page = ctx.new_page()
            page.on("console", lambda m: logline(f"console.{m.type}: {m.text[:4000]}"))
            page.on("pageerror", lambda e: logline(f"pageerror: {e}"))
            crashed = []
            page.on("crash", lambda *_: crashed.append(1))
            page.goto(f"http://127.0.0.1:{port}/{a.page}", timeout=120000, wait_until="commit")
            while not done.is_set() and not crashed and time.time() - t0 < a.timeout:
                page.wait_for_timeout(1000)
            if crashed:
                rec["error"] = "page crashed"
            elif not done.is_set():
                rec["error"] = f"timed out after {a.timeout} s"
            else:
                rec["result"] = box["result"]
            try:
                b.close()
            except Exception:  # noqa: BLE001  the browser may be gone already
                pass
    except Exception as e:  # noqa: BLE001  any driver failure goes into the receipt
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            pass
        srv.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
    rec["wall_s"] = time.time() - t0
    res = rec.get("result") or {}
    ad = res.get("adapter") or {}
    desc = " | ".join(str(ad.get(k) or "") for k in ("vendor", "architecture", "device", "description"))
    rec["adapter_software"] = bool(SOFTWARE.search(desc)) or not ad
    if "error" in rec or res.get("error"):
        rc = 1
    elif rec["adapter_software"] and not a.allow_software and not a.wasm_only:
        rec["error"] = "the page's adapter is a software one (or none): " + desc
        rc = 3
    else:
        rc = 0 if res.get("pass", True) else 1
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({"out": a.out, "rc": rc, "adapter": desc, "error": rec.get("error") or res.get("error"),
                      "wall_s": round(rec["wall_s"], 1)}), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
