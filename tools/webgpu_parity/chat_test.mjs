// The browser chat's two backends side by side, outside a browser: the worker's own steps (worker.js)
// with ember's WebAssembly build (deploy/wasm/build.sh's ember.js, which also runs in node) and the
// WebGPU forward pass on Dawn's node bindings. Per prompt, the WebGPU reply (ember_prepare, the GPU's
// logits, ember_sample, ember_finish) and the WebAssembly reply (ember_chat) of the same greedy
// request, compared with each other and, with a fixture, with ember CPU's /chat replies. The GPU
// path is the one the worker picks: llama_webgpu.js when ember_ir_doc() is "" (ember's tuned llama
// path), ir_webgpu.js for the IR document ember's interpreter runs.
//
//   node --max-opt=1 chat_test.mjs EMBER_JS MODEL.gguf PROMPTS.json OUT.json [--fixture GREEDY.json] [--ctx N] [--first K]
//
// PROMPTS: [{user_input}] (page_test.sh writes it from the 40 frozen prompts), each sent as
// page_test.html's greedy request (60 new tokens, temperature 1, top_k 1, seed 0); or a .jsonl of
// {"i", "request"} rows sent as they are (multi-turn requests with history, e.g.
// experiments/wasm_state_2026-10-25/chat_state4.py's). The fixture, in the same order: GREEDY.json
// (make_greedy_fixture.sh: {"prompts": [{reply}]}) or a receipt with {"replies": [{reply}]}. WEBGPU_NODE and the SwiftShader-only
// rule as in ir_gate.mjs (run it through swiftshader_run.py). --max-opt=1: dawn.node crashes in
// V8's optimised (Maglev) code on this build; the baseline tiers are enough for a gate.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const pos = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const [emberJs, model, promptsPath, outPath] = pos;
if (!outPath) { console.error('usage: chat_test.mjs EMBER_JS MODEL.gguf PROMPTS.json OUT.json [--fixture F] [--ctx N] [--first K]'); process.exit(2); }

const { create, globals } = await import(pathToFileURL(process.env.WEBGPU_NODE).href);
Object.assign(globalThis, globals);
const gpu = create(['backend=vulkan']);
const adapter = await gpu.requestAdapter({ forceFallbackAdapter: true, powerPreference: 'low-power' });
const info = adapter && (adapter.info || {});
const desc = adapter ? [info.vendor, info.architecture, info.device, info.description].join(' | ') : 'none';
if (!adapter || !/swiftshader/i.test(desc) || /nvidia|intel|amd|radeon/i.test(desc)) { console.error('ABORT: the adapter is not SwiftShader: ' + desc); process.exit(3); }
const lim = (k) => adapter.limits[k];
const device = await adapter.requestDevice({ requiredLimits: {
  maxStorageBufferBindingSize: lim('maxStorageBufferBindingSize'), maxBufferSize: lim('maxBufferSize'),
  maxComputeWorkgroupStorageSize: lim('maxComputeWorkgroupStorageSize'),
  maxStorageBuffersPerShaderStage: lim('maxStorageBuffersPerShaderStage') } });
let gpuErrors = 0;   // any WebGPU validation error fails the test
device.addEventListener?.('uncapturederror', (e) => { gpuErrors++; if (gpuErrors <= 3) console.error('WebGPU error:', e.error && e.error.message); });

const createEmber = createRequire(import.meta.url)(path.resolve(emberJs));
const ember = await createEmber();
if (opt('--ctx')) ember._ember_set_ctx(Number(opt('--ctx')));
const file = fs.readFileSync(model);
ember.FS.writeFile('/model.gguf', file);
const ok = ember.ccall('ember_load', 'number', ['string'], ['/model.gguf']);
ember.FS.unlink('/model.gguf');
if (ok !== 1) { console.error('ember_load:', ok); process.exit(2); }
const doc = ember.ccall('ember_ir_doc', 'string', [], []);
const web = path.resolve(here, '../../deploy/wasm/web/webgpu');
const buf = file.byteOffset === 0 && file.byteLength === file.buffer.byteLength ? file.buffer   // no second copy of a large file
  : file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
let m, backend;
if (!doc) {
  const { createLlama } = await import(pathToFileURL(path.join(web, 'llama_webgpu.js')).href);
  m = await createLlama(device, buf); backend = 'llama_webgpu';
} else {
  const { createIr, irSupport } = await import(pathToFileURL(path.join(web, 'ir_webgpu.js')).href);
  const sup = irSupport(JSON.parse(doc));
  if (!sup.ok) { console.error('the document stays in WebAssembly: ' + sup.reason); process.exit(5); }
  m = await createIr(device, buf, doc, { ctx: ember._ember_ctx() }); backend = 'ir_webgpu';
}
const V = m.config.vocab;
const logitsPtr = ember._malloc(4 * V);

async function chatGpu(req) {   // worker.js chatGpu, step for step
  const prep = JSON.parse(ember.ccall('ember_prepare', 'string', ['string'], [JSON.stringify(req)]));
  if (prep.error) return prep;
  let p = prep.prompt_ids.length, stop = -1;
  let logits = await m.forward(prep.prompt_ids, 0);
  for (let s = 0; s < prep.max_new_tokens; s++) {
    ember.HEAPF32.set(logits, logitsPtr >> 2);
    const t = ember._ember_sample(logitsPtr, V);
    if (prep.stop_ids.includes(t)) { stop = t; break; }
    if (s === prep.max_new_tokens - 1 || p >= prep.ctx) break;
    logits = await m.forward([t], p++);
  }
  return JSON.parse(ember.ccall('ember_finish', 'string', ['number'], [stop]));
}

const ptext = fs.readFileSync(promptsPath, 'utf8');
const requests = promptsPath.endsWith('.jsonl')
  ? ptext.split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l).request)
  : JSON.parse(ptext).map((p) => ({ user_input: p.user_input, max_new_tokens: 60, temperature: 1.0, top_k: 1, top_p: 1.0, seed: 0 }));
const fxDoc = opt('--fixture') ? JSON.parse(fs.readFileSync(opt('--fixture'), 'utf8')) : null;
const fx = fxDoc ? (fxDoc.prompts || fxDoc.replies) : null;
const first = Number(opt('--first', 0));
const n = first > 0 ? Math.min(first, requests.length) : requests.length;
const res = { webgpu: [], wasm: [] }, ms = { webgpu: 0, wasm: 0 };
for (let i = 0; i < n; i++) {
  const req = requests[i];
  let t0 = performance.now();
  res.webgpu.push(await chatGpu(req)); ms.webgpu += performance.now() - t0;
  t0 = performance.now();
  res.wasm.push(JSON.parse(ember.ccall('ember_chat', 'string', ['string'], [JSON.stringify(req)]))); ms.wasm += performance.now() - t0;
  const a = res.webgpu[i], b = res.wasm[i];
  console.error(`prompt ${i}: ${a.reply === b.reply && a.stop_reason === b.stop_reason ? 'same' : 'DIFF'} (${a.tokens} tokens, ${a.stop_reason})`);
}
const same = (a, b) => a.reply === b.reply && a.stop_reason === b.stop_reason && a.tokens === b.tokens;
const out = {
  gguf: path.basename(model), adapter: desc, backend, family_hint: doc ? JSON.parse(doc).family_hint : 'llama (tuned path)', n,
  webgpu_vs_wasm: res.webgpu.filter((r, i) => same(r, res.wasm[i])).length,
  webgpu_vs_ember_cpu: fx ? res.webgpu.filter((r, i) => r.reply === fx[i].reply).length : null,
  wasm_vs_ember_cpu: fx ? res.wasm.filter((r, i) => r.reply === fx[i].reply).length : null,
  tokens: res.webgpu.reduce((a, r) => a + (r.tokens || 0), 0),
  ms_per_reply: { webgpu_swiftshader: ms.webgpu / n, wasm_node: ms.wasm / n },
  errors: res.webgpu.filter((r) => r.error).length + res.wasm.filter((r) => r.error).length + gpuErrors,
  first_diffs: res.webgpu.map((r, i) => [i, r.reply, res.wasm[i].reply]).filter(([, a, b]) => a !== b).slice(0, 3),
};
fs.writeFileSync(outPath, JSON.stringify(out, null, 1) + '\n');
console.log(JSON.stringify(out, null, 1));
process.exit(out.webgpu_vs_wasm === n && (!fx || out.webgpu_vs_ember_cpu === n) && !out.errors ? 0 : 1);
