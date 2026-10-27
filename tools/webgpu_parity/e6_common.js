// Helpers for the pages remote_chrome.py drives on a GPU VM (gpu_probe.html, gate_page.html,
// bench_page.html, page_test.html): the adapter's identity for the receipt, and the driver's endpoints
// (POST /e6/row appends one JSON row to the run's rows file, GET /e6/rows reads them back so a page
// resumes, POST /e6/result ends the run, POST /e6/file?name=N stores a text file beside the receipt).

export const q = new URLSearchParams(location.search);
export const num = (k, d) => (q.has(k) ? Number(q.get(k)) : d);

export async function post(path, obj) {
  const r = await fetch(path, { method: 'POST', body: typeof obj === 'string' ? obj : JSON.stringify(obj) });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
}
export const row = (obj) => post('/e6/row', obj);
export const result = (obj) => post('/e6/result', obj);
export const saveFile = (name, text) => post('/e6/file?name=' + encodeURIComponent(name), text);
export async function doneRows() {
  try { return await (await fetch('/e6/rows')).json(); } catch { return []; }
}
export const log = (msg) => post('/e6/log', { t: performance.now(), msg: String(msg) }).catch(() => {});

// The adapter as the page sees it (GPUAdapterInfo), with the limits a run asks for.
export function adapterInfo(adapter) {
  if (!adapter) return null;
  const i = adapter.info || {};
  const lim = {};
  for (const k of ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize',
                   'maxStorageBuffersPerShaderStage', 'maxComputeInvocationsPerWorkgroup'])
    lim[k] = adapter.limits[k];
  return {
    vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description,
    isFallbackAdapter: i.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null,
    subgroupMinSize: i.subgroupMinSize ?? null, subgroupMaxSize: i.subgroupMaxSize ?? null,
    features: [...adapter.features].sort(), limits: lim,
  };
}

export const SOFTWARE = /swiftshader|llvmpipe|lavapipe|softpipe|microsoft basic/i;
export const describe = (a) => (a ? [a.vendor, a.architecture, a.device, a.description].join(' | ') : 'none');

// The device the IR path asks for (tools/webgpu_parity/ir_gate.mjs, worker.js): the adapter's limits.
export async function irDevice(adapter) {
  const lim = (k) => adapter.limits[k];
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: lim('maxStorageBufferBindingSize'), maxBufferSize: lim('maxBufferSize'),
    maxComputeWorkgroupStorageSize: lim('maxComputeWorkgroupStorageSize'),
    maxStorageBuffersPerShaderStage: lim('maxStorageBuffersPerShaderStage') } });
  device.e6Errors = 0;
  device.onuncapturederror = (e) => { device.e6Errors++; if (device.e6Errors <= 3) log('WebGPU error: ' + (e.error && e.error.message)); };
  return device;
}

export const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
export const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };

// Every top-level failure ends the run with the error as its result.
export function guard(main) {
  main().catch(async (e) => {
    try { await result({ error: String(e && e.stack || e), pass: false }); } catch { /* the driver times out */ }
  });
}
