// The regression check of ir_webgpu.js on documents without recurrence: the working tree's module against the
// module at ember HEAD (git show HEAD:deploy/wasm/web/webgpu/ir_webgpu.js, beside HEAD's gguf.js in OLD_DIR), on
// the same device and the same prompts: the logits at every position bit for bit, and the dispatches per token.
//
//   node same_as_head.mjs OLD_DIR CONFIG.json OUT.json
//
// CONFIG: [{name, gguf, doc (null: the file's own), prompts, len}]. Prompts: seeded random ids, one token per
// forward. Each module's model is created, run and destroyed in turn (one copy of the weights on the GPU at a
// time). Exit 1 unless every model is bit-equal with the same dispatch count. SwiftShader only (swiftshader_run.py).
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const [oldDir, cfgPath, outPath] = process.argv.slice(2);
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
let gpuErrors = 0;
device.addEventListener?.('uncapturederror', (e) => { gpuErrors++; if (gpuErrors <= 3) console.error('WebGPU error:', e.error && e.error.message); });
const NEW = await import(pathToFileURL(path.resolve(here, '../../deploy/wasm/web/webgpu/ir_webgpu.js')).href);
const OLD = await import(pathToFileURL(path.join(oldDir, 'ir_webgpu.js')).href);
const { readGguf } = await import(pathToFileURL(path.resolve(here, '../../deploy/wasm/web/webgpu/gguf.js')).href);

const rows = [];
for (const e of JSON.parse(fs.readFileSync(cfgPath, 'utf8'))) {
  const file = fs.readFileSync(e.gguf);
  const buf = file.byteOffset === 0 && file.byteLength === file.buffer.byteLength ? file.buffer
    : file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const docText = e.doc ? fs.readFileSync(e.doc, 'utf8') : readGguf(buf).meta['paperkiln.ir'];
  const run = async (mod) => {
    const sup = mod.irSupport(JSON.parse(docText));
    if (!sup.ok) return { refused: sup.reason };
    const m = await mod.createIr(device, buf, docText, { ctx: 1024 });
    let seed = 777;
    const rnd = (n) => { seed = (seed * 1103515245 + 12345) >>> 0; return seed % n; };
    const parts = [];
    for (let p = 0; p < e.prompts; p++) {
      const ids = Array.from({ length: e.len }, () => rnd(m.config.vocab));
      for (let t = 0; t < e.len; t++) parts.push(new Uint32Array((await m.forward([ids[t]], t)).buffer));
    }
    const d = m.config.dispatchesPerToken;
    m.destroy();
    return { parts, d };
  };
  const t0 = performance.now();
  const a = await run(OLD), b = await run(NEW);
  const r = { name: e.name, prompts: e.prompts, len: e.len };
  if (a.refused || b.refused) Object.assign(r, { old_refused: a.refused || null, new_refused: b.refused || null, pass: false });
  else {
    let same = 0;
    a.parts.forEach((x, i) => { if (x.length === b.parts[i].length && x.every((v, j) => v === b.parts[i][j])) same++; });
    Object.assign(r, { positions: a.parts.length, bit_equal_positions: same, dispatches_old: a.d, dispatches_new: b.d,
                       pass: same === a.parts.length && a.d === b.d });
  }
  r.ms = performance.now() - t0;
  rows.push(r);
  console.error(JSON.stringify(r));
}
const summary = { adapter: desc, models: rows.length, passed: rows.filter((r) => r.pass).length, gpu_errors: gpuErrors, rows };
summary.pass = !gpuErrors && summary.passed === rows.length;
fs.writeFileSync(outPath, JSON.stringify(summary, null, 1) + '\n');
console.log(JSON.stringify({ models: summary.models, passed: summary.passed, gpu_errors: gpuErrors, pass: summary.pass }));
device.destroy();
process.exit(summary.pass ? 0 : 1);
