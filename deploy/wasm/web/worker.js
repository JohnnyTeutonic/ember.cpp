// ember chat, in-browser: ember.cpp compiled to WebAssembly (deploy/wasm/build.sh)
// runs here, off the page's main thread. Messages:
//   in:  {id, req}  where req is the /chat body (user_input, history, sampling)
//   out: {type: 'progress', loaded, total} | {type: 'ready', ok, backend, ctx} | {type: 'reply', id, out}
// A cross-origin-isolated page (COOP/COEP headers; not inside the GitHub
// Pages iframe) can run the multi-threaded build, ember-mt.js, if the Space
// ships one; otherwise, or if it is missing, the single-threaded ember.js.
//
// Context cap: ?ctx=N on the page's URL (the page passes its query string to this worker) caps the
// KV state at N positions before the model loads (ember_set_ctx; ?ctx=0: the file's own context);
// without it, the build's default (EMBER_WASM_CTX, 2048). A file with a smaller context keeps its own.
//
// Backends: with WebGPU available (and no ?backend=wasm), the forward pass runs on the GPU and
// ember's WebAssembly keeps the request rules, the tokeniser, the sampler and the stops
// (ember_prepare / ember_sample / ember_finish); otherwise ember_chat runs everything on the CPU.
// The GPU code follows ember's own choice of path: webgpu/llama_webgpu.js for a file ember serves on
// its tuned llama path, webgpu/ir_webgpu.js for an IR document its interpreter runs (ember_ir_doc),
// and WebAssembly for a document with an op the WGSL path lacks. Both give the same replies for the
// same request (tools/webgpu_parity/).
if (self.crossOriginIsolated === true) {
  try {
    importScripts('ember-mt.js');
  } catch (e) {
    importScripts('ember.js');
  }
} else {
  importScripts('ember.js');
}

let ember = null;
let gpu = null;          // the WebGPU forward pass, when available
let logitsPtr = 0;       // WebAssembly scratch for one logits vector

async function initGpu(buf) {
  if (new URLSearchParams(self.location.search).get('backend') === 'wasm' || !self.navigator.gpu) return null;
  try {
    // The IR document ember's interpreter runs for this file; "" when its tuned llama path serves it.
    const doc = ember.ccall('ember_ir_doc', 'string', [], []);
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    if (!doc) {   // the llama path, as before
      const device = await adapter.requestDevice({ requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize } });
      const { createLlama } = await import('./webgpu/llama_webgpu.js');
      const m = await createLlama(device, buf.buffer);
      logitsPtr = ember._malloc(4 * m.config.vocab);
      return m;
    }
    // Any other document: the IR on WebGPU when every op has a kernel (irSupport), else WebAssembly.
    const { createIr, irSupport } = await import('./webgpu/ir_webgpu.js');
    const sup = irSupport(JSON.parse(doc));
    if (!sup.ok) { console.warn('WebGPU: this document stays in WebAssembly:', sup.reason); return null; }
    const lim = (k) => adapter.limits[k];
    const device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: lim('maxStorageBufferBindingSize'), maxBufferSize: lim('maxBufferSize'),
      maxComputeWorkgroupStorageSize: lim('maxComputeWorkgroupStorageSize'),
      maxStorageBuffersPerShaderStage: lim('maxStorageBuffersPerShaderStage') } });
    const m = await createIr(device, buf.buffer, doc, { ctx: ember._ember_ctx() });
    logitsPtr = ember._malloc(4 * m.config.vocab);
    return m;
  } catch (e) {
    console.warn('WebGPU unavailable for this model, using the CPU:', e);
    return null;
  }
}

async function init() {
  ember = await createEmber();
  const ctx = new URLSearchParams(self.location.search).get('ctx');
  if (ctx !== null && /^[0-9]+$/.test(ctx)) ember._ember_set_ctx(Number(ctx));
  const res = await fetch('chat.gguf');
  if (!res.ok) throw new Error('model download failed: ' + res.status);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    postMessage({ type: 'progress', loaded, total });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const p of parts) { buf.set(p, off); off += p.length; }
  ember.FS.writeFile('/model.gguf', buf);
  const ok = ember.ccall('ember_load', 'number', ['string'], ['/model.gguf']);
  ember.FS.unlink('/model.gguf');  // the session holds its own copy of the weights
  if (ok === 1) gpu = await initGpu(buf);
  postMessage({ type: 'ready', ok, backend: gpu ? 'webgpu' : 'wasm', ctx: ember._ember_ctx() });
}

// One reply with the forward pass on the GPU: ember prepares the prompt and samples each token.
async function chatGpu(req) {
  const prep = JSON.parse(ember.ccall('ember_prepare', 'string', ['string'], [JSON.stringify(req)]));
  if (prep.error) return prep;
  const n = gpu.config.vocab;
  let pos = prep.prompt_ids.length, stop = -1;
  let logits = await gpu.forward(prep.prompt_ids, 0);
  for (let s = 0; s < prep.max_new_tokens; s++) {
    ember.HEAPF32.set(logits, logitsPtr >> 2);
    const t = ember._ember_sample(logitsPtr, n);
    if (prep.stop_ids.includes(t)) { stop = t; break; }   // the chat template's stop set
    if (s === prep.max_new_tokens - 1 || pos >= prep.ctx) break;   // the length limit
    logits = await gpu.forward([t], pos++);
  }
  return JSON.parse(ember.ccall('ember_finish', 'string', ['number'], [stop]));
}

const ready = init().catch((e) => postMessage({ type: 'ready', ok: -1, error: String(e) }));

onmessage = async (e) => {
  await ready;
  const out = gpu ? await chatGpu(e.data.req)
                  : JSON.parse(ember.ccall('ember_chat', 'string', ['string'], [JSON.stringify(e.data.req)]));
  postMessage({ type: 'reply', id: e.data.id, out });
};
