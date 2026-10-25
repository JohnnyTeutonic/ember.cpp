// The WebGPU IR path (deploy/wasm/web/webgpu/ir_webgpu.js) against ember's native CPU interpreter, on
// tools/ir_wasm_gate's reference: its seeded random prompts (ids.json), the CPU's prefill logits at every
// position (ref_<i>.bin, float32 [len][vocab]) and its greedy continuation (ref.jsonl). So a family's
// WebGPU leg is gated on the same prompts as its WebAssembly and CUDA legs.
//
//   node ir_gate.mjs MODEL.gguf REF_DIR OUT_DIR [--doc DOC.json] [--first K] [--ctx 1024] [--tol 1e-4]
//
// Per prompt: the prompt's ids one token per forward (the path's prefill: IR_SPEC 3.9), each position's
// logits against the CPU's prefill row (max |dlogit| over the row's range, and IR_SPEC 4.3's measure
// max |d| / max(1, max |ref|)) and its argmax; then `gen` greedy tokens against the CPU's. Rows to
// OUT_DIR/check.jsonl (resumes: a prompt with a row is not run again), the summary on stdout and in
// OUT_DIR/summary.json. Exit 1 unless every greedy continuation is identical, every argmax agrees and
// the worst distance is within --tol (the receipts' bound of the WebAssembly and device gates).
//
// The document: --doc, else the file's paperkiln.ir. GPU: Dawn's node bindings (WEBGPU_NODE: the
// path of the `webgpu` package's index.js) on the SOFTWARE adapter only (SwiftShader: run it through
// tools/webgpu_parity/swiftshader_run.py, which leaves SwiftShader the only Vulkan driver); the gate
// refuses any adapter that does not name SwiftShader.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const [model, refDir, outDir] = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
if (!model || !refDir || !outDir) { console.error('usage: ir_gate.mjs MODEL.gguf REF_DIR OUT_DIR [--doc F] [--first K] [--ctx N] [--tol T]'); process.exit(2); }
const first = Number(opt('--first', 0)), ctx = Number(opt('--ctx', 1024)), tol = Number(opt('--tol', 1e-4));

const { create, globals } = await import(pathToFileURL(process.env.WEBGPU_NODE).href);
Object.assign(globalThis, globals);
const gpu = create(['backend=vulkan']);
const adapter = await gpu.requestAdapter({ forceFallbackAdapter: true, powerPreference: 'low-power' });
const info = adapter && (adapter.info || {});
const desc = adapter ? [info.vendor, info.architecture, info.device, info.description].join(' | ') : 'none';
if (!adapter || !/swiftshader/i.test(desc) || /nvidia|intel|amd|radeon/i.test(desc)) {
  console.error('ABORT: the adapter is not SwiftShader: ' + desc);
  process.exit(3);
}
const lim = (k) => adapter.limits[k];
const device = await adapter.requestDevice({ requiredLimits: {
  maxStorageBufferBindingSize: lim('maxStorageBufferBindingSize'), maxBufferSize: lim('maxBufferSize'),
  maxComputeWorkgroupStorageSize: lim('maxComputeWorkgroupStorageSize'),
  maxStorageBuffersPerShaderStage: lim('maxStorageBuffersPerShaderStage') } });
device.addEventListener?.('uncapturederror', (e) => { console.error('WebGPU error:', e.error && e.error.message); process.exitCode = 4; });

const web = path.resolve(here, '../../deploy/wasm/web/webgpu');
const { createIr } = await import(pathToFileURL(path.join(web, 'ir_webgpu.js')).href);
const { readGguf } = await import(pathToFileURL(path.join(web, 'gguf.js')).href);

const t0 = performance.now();
const file = fs.readFileSync(model);
const buf = file.byteOffset === 0 && file.byteLength === file.buffer.byteLength ? file.buffer   // no second copy of a large file
  : file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
const docText = opt('--doc') ? fs.readFileSync(opt('--doc'), 'utf8') : readGguf(buf).meta['paperkiln.ir'];
if (!docText) { console.error('no IR document: pass --doc'); process.exit(2); }
const m = await createIr(device, buf, docText, { ctx });
const loadMs = performance.now() - t0;
const V = m.config.vocab;

const ids = JSON.parse(fs.readFileSync(path.join(refDir, 'ids.json'), 'utf8'));
if (ids.vocab !== V) { console.error(`vocab ${V} differs from the reference's ${ids.vocab}`); process.exit(2); }
const rows = (p) => fs.existsSync(p) ? fs.readFileSync(p, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)) : [];
const ref = new Map(rows(path.join(refDir, 'ref.jsonl')).map((r) => [r.i, r]));
fs.mkdirSync(outDir, { recursive: true });
const checkPath = path.join(outDir, 'check.jsonl');
const done = new Set(rows(checkPath).map((r) => r.i));
const upto = first > 0 ? Math.min(first, ids.n) : ids.n;
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };

for (let s = 0; s < upto; s++) {
  if (done.has(s)) continue;
  const r = ref.get(s);
  if (!r) throw new Error(`reference prompt ${s} missing`);
  const rb = fs.readFileSync(path.join(refDir, `ref_${s}.bin`));
  const R = new Float32Array(rb.buffer, rb.byteOffset, rb.byteLength / 4);
  const p = ids.ids[s], len = ids.len, gen = ids.gen;
  const t1 = performance.now();
  let rel = 0, spec = 0, top1 = 0, lg = null;
  const flips = [];
  for (let t = 0; t < len; t++) {
    lg = await m.forward([p[t]], t);
    const row = R.subarray(t * V, (t + 1) * V);
    let lo = Infinity, hi = -Infinity, d = 0, mref = 0;
    for (let i = 0; i < V; i++) {
      lo = Math.min(lo, row[i]); hi = Math.max(hi, row[i]);
      d = Math.max(d, Math.abs(lg[i] - row[i])); mref = Math.max(mref, Math.abs(row[i]));
    }
    rel = Math.max(rel, hi > lo ? d / (hi - lo) : d);
    spec = Math.max(spec, d / Math.max(1, mref));
    const a = argmax(row), b = argmax(lg);
    top1 += a === b;
    if (a !== b) flips.push({ pos: t, ref_top: a, webgpu_top: b });
  }
  const g = [];
  let pos = len;
  for (let k = 0; k < gen; k++) {
    const t = argmax(lg);
    g.push(t);
    if (k + 1 < gen) lg = await m.forward([t], pos++);
  }
  let firstDiff = -1;
  for (let k = 0; k < gen; k++) if (g[k] !== r.greedy[k]) { firstDiff = k; break; }
  const out = { i: s, rel, spec_rel: spec, top1, greedy_first_diff: firstDiff, ms: performance.now() - t1 };
  if (flips.length) out.flips = flips;
  fs.appendFileSync(checkPath, JSON.stringify(out) + '\n');
  console.error(`webgpu prompt ${s}: rel ${rel.toExponential(2)}, top1 ${top1}/${len}, greedy ${firstDiff < 0 ? 'same' : 'differs at ' + firstDiff}, ${out.ms.toFixed(0)} ms`);
}

const all = rows(checkPath).filter((r) => r.i < upto);
const worst = Math.max(0, ...all.map((r) => r.rel)), worstSpec = Math.max(0, ...all.map((r) => r.spec_rel));
const top1 = all.reduce((a, r) => a + r.top1, 0), positions = all.length * ids.len;
const same = all.filter((r) => r.greedy_first_diff < 0).length;
const sorted = all.map((r) => r.rel).sort((a, b) => a - b);
const summary = {
  gguf: path.basename(model), adapter: desc, family_hint: m.config.family_hint, prompts: upto, of: ids.n, len: ids.len,
  gen: ids.gen, vocab: V, ctx: m.config.ctx, dispatches_per_token: m.config.dispatchesPerToken,
  worst_rel: worst, median_rel: sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0, worst_spec_rel: worstSpec,
  top1_agree: `${top1}/${positions}`, greedy_identical: `${same}/${upto}`, tol,
  pass: all.length === upto && same === upto && top1 === positions && worst <= tol,
  load_ms: loadMs, ms_per_prompt: all.length ? all.reduce((a, r) => a + r.ms, 0) / all.length : 0,
};
fs.writeFileSync(path.join(outDir, 'summary.json'), JSON.stringify(summary, null, 1) + '\n');
console.log(JSON.stringify(summary, null, 1));
m.destroy();
device.destroy();
process.exit(summary.pass ? 0 : 1);
