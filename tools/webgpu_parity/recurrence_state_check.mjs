// The WebGPU IR path's sequence state (deploy/wasm/web/webgpu/ir_webgpu.js) on documents with recurrent state:
// decode = prefill and multi-turn state on the path itself, and keramos's state traces.
//
//   node recurrence_state_check.mjs CONFIG.json OUT.json
//
// CONFIG: [{name, gguf, doc (null: the file's own), seq: {n, len} (optional), trace (optional: a keramos
// conformance state trace, state/<name>.json)}]. Per entry with seq, n seeded random prompts A (len ids) and B:
//   whole      forward(A, 0): the logits after A and the recurrent state (readState);
//   tokenwise  A one token per call: logits and state bit-equal to whole (decode = prefill, IR_SPEC 3.9, 6.10.5);
//   chunks     A in two calls at its middle: bit-equal;
//   reset      forward(B, 0), then forward(A, 0) again: bit-equal to whole (a new sequence restarts from the
//              initial state, the init bodies included: a later turn re-prefilled is the first turn's state);
//   turns      forward(A, 0) then forward(B[:m], len): bit-equal to forward(A + B[:m], 0) (a turn continued on the
//              GPU state equals the conversation prefilled at once);
//   refusal    a forward that skips positions on recurrent state throws (never runs on a wrong state).
// Per entry with a trace: its steps (forward; snapshot / restore by re-prefilling the ids held at the snapshot,
// which IR_SPEC 6.9 allows), each forward's last logits row and, where the trace states it, every recurrent
// slot tensor per application against the trace's expectations (keramos's numpy reference): IR_SPEC 4.3's
// max |d| / max(1, max |ref|) within the trace's tol, the argmax equal. Exit 1 unless every check passes.
// GPU: Dawn's node bindings (WEBGPU_NODE) on SwiftShader only, through swiftshader_run.py.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const [cfgPath, outPath] = process.argv.slice(2);
if (!outPath) { console.error('usage: recurrence_state_check.mjs CONFIG.json OUT.json'); process.exit(2); }
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
const web = path.resolve(here, '../../deploy/wasm/web/webgpu');
const { createIr } = await import(pathToFileURL(path.join(web, 'ir_webgpu.js')).href);
const { readGguf } = await import(pathToFileURL(path.join(web, 'gguf.js')).href);

const unhex = (t) => { const h = t.f32, u = new Uint32Array(h.length / 8); for (let i = 0; i < u.length; i++) u[i] = parseInt(h.slice(8 * i, 8 * i + 8), 16); return new Float32Array(u.buffer); };
const bitEq = (a, b) => a.length === b.length && new Uint32Array(a.buffer, a.byteOffset, a.length).every((x, i) => x === new Uint32Array(b.buffer, b.byteOffset, b.length)[i]);
const stateEq = (s, t) => Object.keys(s).every((k) => s[k].every((app, l) => Object.keys(app).every((n) => bitEq(app[n], t[k][l][n]))));
const rel = (got, ref) => { let d = 0, m = 0; for (let i = 0; i < ref.length; i++) { d = Math.max(d, Math.abs(got[i] - ref[i])); m = Math.max(m, Math.abs(ref[i])); } return d / Math.max(1, m); };
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
let seed = 12345;
const rnd = (n) => { seed = (seed * 1103515245 + 12345) >>> 0; return seed % n; };

const results = [];
for (const e of JSON.parse(fs.readFileSync(cfgPath, 'utf8'))) {
  const file = fs.readFileSync(e.gguf);
  const buf = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const docText = e.doc ? fs.readFileSync(e.doc, 'utf8') : readGguf(buf).meta['paperkiln.ir'];
  const m = await createIr(device, buf, docText, { ctx: 1024 });
  const V = m.config.vocab, r = { name: e.name, recurrent: m.config.recurrent, dispatches_per_token: m.config.dispatchesPerToken };
  if (e.seq) {
    const c = { whole_tokenwise: 0, chunks: 0, reset: 0, turns: 0, refusal: 0, n: e.seq.n };
    for (let p = 0; p < e.seq.n; p++) {
      const L = e.seq.len, A = Array.from({ length: L }, () => rnd(V)), B = Array.from({ length: L }, () => rnd(V));
      const whole = await m.forward(A, 0), sWhole = await m.readState();
      let tw = null;
      for (let i = 0; i < L; i++) tw = await m.forward([A[i]], i);
      const sTw = await m.readState();
      if (bitEq(whole, tw) && stateEq(sWhole, sTw)) c.whole_tokenwise++;
      const h = L >> 1;
      await m.forward(A.slice(0, h), 0);
      const ch = await m.forward(A.slice(h), h);
      if (bitEq(whole, ch) && stateEq(sWhole, await m.readState())) c.chunks++;
      await m.forward(B, 0);
      const again = await m.forward(A, 0);
      if (bitEq(whole, again) && stateEq(sWhole, await m.readState())) c.reset++;
      const mm = Math.min(L, m.config.ctx - L);
      const cont = await m.forward(B.slice(0, mm), L), sCont = await m.readState();
      const all = await m.forward([...A, ...B.slice(0, mm)], 0);
      if (bitEq(cont, all) && stateEq(sCont, await m.readState())) c.turns++;
      let threw = false;
      try { await m.forward([A[0]], L + mm + 3); } catch { threw = true; }
      if (threw) c.refusal++;
    }
    c.pass = ['whole_tokenwise', 'chunks', 'reset', 'turns', 'refusal'].every((k) => c[k] === c.n);
    r.seq = c;
  }
  if (e.trace) {
    const tr = JSON.parse(fs.readFileSync(e.trace, 'utf8'));
    const held = [], snaps = {}, steps = [];
    let k = 0, worst = 0, worstState = 0, top1 = 0, forwards = 0, stateChecks = 0;
    const checkState = async (ex, st) => {
      if (!ex.recurrent) return;
      const g = await m.readState();
      let ds = 0;
      for (const [slot, apps] of Object.entries(ex.recurrent))
        apps.forEach((app, l) => { for (const [n, t] of Object.entries(app)) { ds = Math.max(ds, rel(g[slot][l][n], unhex(t))); stateChecks++; } });
      st.state_rel = ds;
      worstState = Math.max(worstState, ds);
    };
    let lenOk = true;
    for (const s of tr.steps) {   // one expectation per step
      const ex = tr.expect[k++], st = { method: s.method };
      if (s.method === 'snapshot') snaps[s.tag] = [...held];
      else if (s.method === 'restore') {   // IR_SPEC 6.9: restore, or run the held ids again from the start
        held.length = 0; held.push(...snaps[s.tag]);
        await m.forward(held, 0);
        await checkState(ex, st);
      } else if (s.method === 'forward') {
        const lg = await m.forward(s.ids, held.length);
        held.push(...s.ids);
        const R = unhex(ex.logits), V2 = ex.logits.shape[1], last = R.subarray(R.length - V2);
        st.rel = rel(lg, last);
        worst = Math.max(worst, st.rel);
        forwards++;
        if (argmax(lg) === argmax(last)) top1++;
        await checkState(ex, st);
      } else throw new Error(`trace ${e.trace}: method ${s.method}`);
      st.len = held.length;
      if (ex.len !== undefined && ex.len !== held.length) lenOk = false;
      steps.push(st);
    }
    r.trace = { file: path.basename(e.trace), tol: tr.tol, forwards, top1: `${top1}/${forwards}`, worst_rel: worst,
                state_tensors_checked: stateChecks, worst_state_rel: worstState, steps,
                pass: lenOk && worst <= tr.tol && worstState <= tr.tol && top1 === forwards && k === tr.expect.length };
  }
  r.pass = (!r.seq || r.seq.pass) && (!r.trace || r.trace.pass);
  results.push(r);
  console.error(`${e.name}: ${JSON.stringify({ seq: r.seq, trace: r.trace && { ...r.trace, steps: undefined } })}`);
  m.destroy();
}
const summary = { adapter: desc, entries: results.length, passed: results.filter((x) => x.pass).length, gpu_errors: gpuErrors, results };
summary.pass = !gpuErrors && summary.passed === results.length;
fs.writeFileSync(outPath, JSON.stringify(summary, null, 1) + '\n');
console.log(JSON.stringify({ adapter: desc, entries: summary.entries, passed: summary.passed, gpu_errors: gpuErrors, pass: summary.pass }));
device.destroy();
process.exit(summary.pass ? 0 : 1);
