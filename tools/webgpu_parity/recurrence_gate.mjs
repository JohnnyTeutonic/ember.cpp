// The WebGPU path's recurrence kernels (deploy/wasm/web/webgpu/ir_webgpu.js: compileRecurrence,
// runRecurrenceNode) on single recurrence nodes: op cases in keramos's format (conformance/ops/recurrence.json,
// IR_SPEC 11.2), each node run as application 0 token by token from its slot's initial state, as ember's
// adapter runs it (tools/ir_adapter.cpp run_node). Per case, the output [T, w] and the slot's tensors after the
// last row against the case's expected values (keramos's numpy reference, tools/ir_numpy.py / ir_recurrence.py)
// and, with --cpu, against ember CPU's (experiments/webgpu_recurrence_2026-10-27/cpu_ops.py): IR_SPEC 4.3's
// measure max |d| / max(1, max |ref|), the bit-identical values and the NaN positions.
//
//   node recurrence_gate.mjs OUT.json CASES.json [CASES.json ...] [--cpu CPU.json]
//
// Exit 1 unless every case is within its tol against both references with the same NaN positions. GPU:
// Dawn's node bindings (WEBGPU_NODE) on SwiftShader only, through swiftshader_run.py (the adapter must name
// SwiftShader).
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const pos = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const [outPath, ...caseFiles] = pos;
if (!outPath || !caseFiles.length) { console.error('usage: recurrence_gate.mjs OUT.json CASES.json [...] [--cpu CPU.json]'); process.exit(2); }

const { create, globals } = await import(pathToFileURL(process.env.WEBGPU_NODE).href);
Object.assign(globalThis, globals);
const gpu = create(['backend=vulkan']);
const adapter = await gpu.requestAdapter({ forceFallbackAdapter: true, powerPreference: 'low-power' });
const info = adapter && (adapter.info || {});
const desc = adapter ? [info.vendor, info.architecture, info.device, info.description].join(' | ') : 'none';
if (!adapter || !/swiftshader/i.test(desc) || /nvidia|intel|amd|radeon/i.test(desc)) { console.error('ABORT: the adapter is not SwiftShader: ' + desc); process.exit(3); }
const device = await adapter.requestDevice();
let gpuErrors = 0;
device.addEventListener?.('uncapturederror', (e) => { gpuErrors++; if (gpuErrors <= 3) console.error('WebGPU error:', e.error && e.error.message); });
const { runRecurrenceNode } = await import(pathToFileURL(path.resolve(here, '../../deploy/wasm/web/webgpu/ir_webgpu.js')).href);

const unhex = (t) => {   // IR_SPEC 11.2: 8 hex digits per value, the binary32 bit pattern
  const h = t.f32, u = new Uint32Array(h.length / 8);
  for (let i = 0; i < u.length; i++) u[i] = parseInt(h.slice(8 * i, 8 * i + 8), 16);
  return new Float32Array(u.buffer);
};
const cmp = (got, ref) => {
  let d = 0, m = 0, same = 0, nan = 0;
  const gb = new Uint32Array(got.buffer, got.byteOffset, got.length), rb = new Uint32Array(ref.buffer, ref.byteOffset, ref.length);
  for (let i = 0; i < ref.length; i++) {
    if (gb[i] === rb[i]) same++;
    if (Number.isNaN(got[i]) !== Number.isNaN(ref[i])) nan++;
    else if (!Number.isNaN(ref[i]) && got[i] !== ref[i]) d = Math.max(d, Math.abs(got[i] - ref[i]));
    if (isFinite(ref[i])) m = Math.max(m, Math.abs(ref[i]));
  }
  return { rel: got.length === ref.length ? d / Math.max(1, m) : Infinity, same, n: ref.length, nan_mismatch: nan };
};
const cpu = opt('--cpu') ? JSON.parse(fs.readFileSync(opt('--cpu'), 'utf8')) : null;

const rows = [];
for (const f of caseFiles) {
  const doc = JSON.parse(fs.readFileSync(f, 'utf8'));
  for (const c of doc.cases) {
    const node = c.node, ins = (Array.isArray(node.in) ? node.in : [node.in]).map((nm) => unhex(c.inputs[nm]));
    const T = c.inputs[(Array.isArray(node.in) ? node.in : [node.in])[0]].shape[0];
    const weights = Object.fromEntries(Object.entries(c.weights || {}).map(([k, v]) => [k, unhex(v)]));
    const t0 = performance.now();
    let r;
    try {
      r = await runRecurrenceNode(device, node, ins, weights, T);
    } catch (e) {
      rows.push({ name: c.name, file: path.basename(f), error: String(e.message || e) });
      console.error(`${c.name}: ERROR ${e.message || e}`);
      continue;
    }
    const res = { name: c.name, file: path.basename(f), class: c.class, exact_bodies: r.exact, tol: c.tol, T, ms: performance.now() - t0 };
    const against = (expOut, expState) => {
      const parts = [cmp(r.y, unhex(expOut[node.out]))];
      for (const [k, v] of Object.entries(expState[node.state] || {})) parts.push(cmp(r.state[k], unhex(v)));
      return { rel: Math.max(...parts.map((p) => p.rel)), same: parts.reduce((a, p) => a + p.same, 0),
               n: parts.reduce((a, p) => a + p.n, 0), nan_mismatch: parts.reduce((a, p) => a + p.nan_mismatch, 0),
               state_tensors: parts.length - 1 };
    };
    res.vs_numpy = against(c.expect, c.expect_state || {});
    if (cpu) {
      const e = cpu[c.name];
      res.vs_ember_cpu = e ? against(e.outputs, e.state) : { missing: true };
    }
    const ok = (x) => x && !x.missing && x.rel <= c.tol && !x.nan_mismatch && x.state_tensors === Object.keys(r.state).length;
    res.pass = ok(res.vs_numpy) && (!cpu || ok(res.vs_ember_cpu));
    rows.push(res);
    const fmt = (x) => x && !x.missing ? `${x.rel.toExponential(2)} (${x.same}/${x.n} bit-equal)` : 'missing';
    console.error(`${c.name} [${c.class}]: vs numpy ${fmt(res.vs_numpy)}${cpu ? ', vs ember CPU ' + fmt(res.vs_ember_cpu) : ''} ${res.pass ? 'pass' : 'FAIL'}`);
  }
}
const summary = { adapter: desc, cases: rows.length, passed: rows.filter((r) => r.pass).length, gpu_errors: gpuErrors, rows };
summary.pass = !gpuErrors && summary.passed === rows.length;
fs.writeFileSync(outPath, JSON.stringify(summary, null, 1) + '\n');
console.log(JSON.stringify({ adapter: desc, cases: summary.cases, passed: summary.passed, gpu_errors: gpuErrors, pass: summary.pass }));
device.destroy();
process.exit(summary.pass ? 0 : 1);
