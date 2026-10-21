// The llama forward pass on WebGPU, for ember's browser chat (prototype; LOCAL_LLM_STACK.md
// section 7, Research item 1). ember's WebAssembly keeps tokenising, the chat template, sampling
// and the stop rules; this module turns token ids into next-token logits on the GPU.
//
// Weights stay resident on the GPU (f32; Q8_0/f16 GGUFs are dequantised at load). Each token is
// one command buffer of about 13 dispatches per layer, built from bind groups made once at load;
// only a 16-byte step uniform (position, token id) changes per token, and only the last prompt
// token's logits are read back. The maths follows ember's CPU path (model.cpp, utils.cpp): RMSNorm
// (1/sqrt(mean(x^2) + eps)), RoPE on adjacent pairs with theta^(-i/head_dim), causal softmax
// attention scaled by 1/sqrt(head_dim) with grouped KV heads, SwiGLU. Sums are f32 in a different
// order, so logits match the CPU to rounding, not bit for bit.
import { GGML_Q8_0, f16ToF32, readGguf, tensorF32 } from './gguf.js';

const WG = 64;
const MAX_CTX = 2048;   // attention's workgroup score array

const SHADERS = {
  embed: `
struct S { pos: u32, token: u32, n: u32, _p: u32 };
@group(0) @binding(0) var<uniform> st: S;
@group(0) @binding(1) var<storage, read> E: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x < st.n) { x[g.x] = E[st.token * st.n + g.x]; }
}`,
  rmsnorm: `
struct P { n: u32, eps: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3u) {
  var s = 0.0;
  for (var i = l.x; i < p.n; i += 256u) { s += x[i] * x[i]; }
  part[l.x] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(part[0] / f32(p.n) + p.eps);
  for (var i = l.x; i < p.n; i += 256u) { y[i] = x[i] * inv * w[i]; }
}`,
  matvec: `
struct P { rows: u32, cols: u32, accumulate: u32, _p: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let row = wg.x;
  let base = row * p.cols;
  var s = 0.0;
  for (var c = l.x; c < p.cols; c += ${WG}u) { s += W[base + c] * x[c]; }
  part[l.x] = s;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  if (l.x == 0u) { if (p.accumulate != 0u) { y[row] = y[row] + part[0]; } else { y[row] = part[0]; } }
}`,
  // Q8_0 kept quantised: per 32-weight block an f32 scale and 8 words of 4 int8. Each weight is
  // scale * q in f32, the value ember's dequantise-at-load produces, then the same dot as matvec.
  matvec_q8: `
struct P { rows: u32, cols: u32, accumulate: u32, _p: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<u32>;   // [rows][cols / 4] packed int8
@group(0) @binding(2) var<storage, read> D: array<f32>;   // [rows][cols / 32] scales
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let row = wg.x;
  let words = p.cols / 4u;
  var s = 0.0;
  for (var w = l.x; w < words; w += ${WG}u) {
    let packed = Q[row * words + w];
    let sc = D[row * (p.cols / 32u) + w / 8u];
    let c = 4u * w;
    for (var b = 0u; b < 4u; b++) {
      let q = f32((i32(packed << (24u - 8u * b))) >> 24u);
      s += (sc * q) * x[c + b];
    }
  }
  part[l.x] = s;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  if (l.x == 0u) { if (p.accumulate != 0u) { y[row] = y[row] + part[0]; } else { y[row] = part[0]; } }
}`,
  rope_kv: `
struct S { pos: u32, token: u32, n: u32, _p: u32 };
struct P { n_heads: u32, n_kv: u32, head_dim: u32, _p: u32 };
@group(0) @binding(0) var<uniform> st: S;
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> cs: array<f32>;            // [ctx][head_dim/2][cos, sin]
@group(0) @binding(3) var<storage, read_write> q: array<f32>;
@group(0) @binding(4) var<storage, read> k: array<f32>;
@group(0) @binding(5) var<storage, read> v: array<f32>;
@group(0) @binding(6) var<storage, read_write> kc: array<f32>;      // [ctx][n_kv * head_dim]
@group(0) @binding(7) var<storage, read_write> vc: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let half = p.head_dim / 2u;
  let kvd = p.n_kv * p.head_dim;
  let i = g.x;
  if (i < p.n_heads * half) {                      // rotate q in place
    let h = i / half; let j = i % half;
    let a = h * p.head_dim + 2u * j;
    let c = cs[2u * (st.pos * half + j)]; let s = cs[2u * (st.pos * half + j) + 1u];
    let x0 = q[a]; let x1 = q[a + 1u];
    q[a] = x0 * c - x1 * s; q[a + 1u] = x0 * s + x1 * c;
  }
  if (i < p.n_kv * half) {                         // rotate k into the cache
    let h = i / half; let j = i % half;
    let a = h * p.head_dim + 2u * j;
    let c = cs[2u * (st.pos * half + j)]; let s = cs[2u * (st.pos * half + j) + 1u];
    let x0 = k[a]; let x1 = k[a + 1u];
    kc[st.pos * kvd + a] = x0 * c - x1 * s; kc[st.pos * kvd + a + 1u] = x0 * s + x1 * c;
  }
  if (i < kvd) { vc[st.pos * kvd + i] = v[i]; }
}`,
  attention: `
struct S { pos: u32, token: u32, n: u32, _p: u32 };
struct P { n_heads: u32, n_kv: u32, head_dim: u32, _p: u32 };
@group(0) @binding(0) var<uniform> st: S;
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kc: array<f32>;
@group(0) @binding(4) var<storage, read> vc: array<f32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
var<workgroup> sc: array<f32, ${MAX_CTX}>;
var<workgroup> part: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let h = wg.x;
  let kvh = h / (p.n_heads / p.n_kv);
  let kvd = p.n_kv * p.head_dim;
  let qo = h * p.head_dim; let ko = kvh * p.head_dim;
  let n = st.pos + 1u;
  let scale = 1.0 / sqrt(f32(p.head_dim));
  var m = -3.4e38;
  for (var t = l.x; t < n; t += ${WG}u) {
    var s = 0.0;
    for (var d = 0u; d < p.head_dim; d++) { s += q[qo + d] * kc[t * kvd + ko + d]; }
    s = s * scale; sc[t] = s; m = max(m, s);
  }
  part[l.x] = m;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] = max(part[l.x], part[l.x + k]); } workgroupBarrier(); }
  let mx = part[0];
  workgroupBarrier();
  var z = 0.0;
  for (var t = l.x; t < n; t += ${WG}u) { let e = exp(sc[t] - mx); sc[t] = e; z += e; }
  part[l.x] = z;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  let zz = part[0];
  for (var d = l.x; d < p.head_dim; d += ${WG}u) {
    var acc = 0.0;
    for (var t = 0u; t < n; t++) { acc += sc[t] * vc[t * kvd + ko + d]; }
    o[qo + d] = acc / zz;
  }
}`,
  // Mixture of experts, as ember's moe_ffn_f32_cpu (utils.cpp): top-k router logits (ties to the
  // lower index), gates = a softmax over the k selected logits (renorm), experts in ascending index.
  moe_select: `
struct P { E: u32, k: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> sel: array<u32>;
@group(0) @binding(3) var<storage, read_write> gate: array<f32>;
@compute @workgroup_size(1) fn main() {
  var taken: array<bool, 256>;
  var pick: array<u32, 16>;
  for (var j = 0u; j < p.k; j++) {
    var best = 0xffffffffu;
    for (var e = 0u; e < p.E; e++) {
      if (!taken[e] && (best == 0xffffffffu || r[e] > r[best])) { best = e; }
    }
    taken[best] = true; pick[j] = best;
  }
  for (var a = 1u; a < p.k; a++) {                 // ascending expert index
    let v = pick[a]; var b = a;
    while (b > 0u && pick[b - 1u] > v) { pick[b] = pick[b - 1u]; b--; }
    pick[b] = v;
  }
  var mx = -3.4e38;
  for (var j = 0u; j < p.k; j++) { mx = max(mx, r[pick[j]]); }
  var z = 0.0;
  for (var j = 0u; j < p.k; j++) { z += exp(r[pick[j]] - mx); }
  for (var j = 0u; j < p.k; j++) { sel[j] = pick[j]; gate[j] = exp(r[pick[j]] - mx) / z; }
}`,
  matvec_exp: `
struct P { rows: u32, cols: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;      // [E][rows][cols]
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [k][rows]
var<workgroup> part: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let row = wg.x; let j = wg.y;
  let base = (sel[j] * p.rows + row) * p.cols;
  var s = 0.0;
  for (var c = l.x; c < p.cols; c += ${WG}u) { s += W[base + c] * x[c]; }
  part[l.x] = s;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  if (l.x == 0u) { y[j * p.rows + row] = part[0]; }
}`,
  moe_down: `
struct P { rows: u32, cols: u32, k: u32, _a: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<f32>;      // [E][rows = d][cols = d_ff]
@group(0) @binding(4) var<storage, read> h: array<f32>;      // [k][d_ff]
@group(0) @binding(5) var<storage, read_write> x: array<f32>; // residual stream, += sum_j gate_j * down_j h_j
var<workgroup> part: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let row = wg.x;
  var acc = 0.0;
  for (var j = 0u; j < p.k; j++) {
    let base = (sel[j] * p.rows + row) * p.cols;
    var s = 0.0;
    for (var c = l.x; c < p.cols; c += ${WG}u) { s += W[base + c] * h[j * p.cols + c]; }
    part[l.x] = s;
    workgroupBarrier();
    for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
    acc += gate[j] * part[0];
    workgroupBarrier();
  }
  if (l.x == 0u) { x[row] = x[row] + acc; }
}`,
  swiglu: `
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> g: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) i: vec3u) {
  if (i.x < p.n) { let a = g[i.x]; g[i.x] = a / (1.0 + exp(-a)) * u[i.x]; }
}`,
};

// Test hook: one weight matrix through matvec (dequantised f32) and matvec_q8 (kept quantised)
// with the same input x; returns both outputs and a JS f64 reference.
export async function testMatvecQ8(device, ggufBuffer, name, x) {
  const g = readGguf(ggufBuffer), t = g.tensors[name];
  const cols = t.dims[0], rows = t.dims[1], S = GPUBufferUsage;
  const buf = (arr, extra = 0) => {
    const b = device.createBuffer({ size: Math.max(16, arr.byteLength), usage: S.STORAGE | S.COPY_DST | S.COPY_SRC | extra });
    device.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return b;
  };
  const f = tensorF32(g, name);
  const nb = t.n / 32, raw = new Uint8Array(g.buf, t.start, 34 * nb), dv = new DataView(g.buf, t.start);
  const q = new Uint8Array(32 * nb), sc = new Float32Array(nb);
  for (let b = 0; b < nb; b++) { sc[b] = f16ToF32(dv.getUint16(34 * b, true)); q.set(raw.subarray(34 * b + 2, 34 * b + 34), 32 * b); }
  const P = device.createBuffer({ size: 16, usage: S.UNIFORM | S.COPY_DST });
  device.queue.writeBuffer(P, 0, new Uint32Array([rows, cols, 0, 0]));
  const X = buf(x), Y1 = buf(new Float32Array(rows)), Y2 = buf(new Float32Array(rows));
  const run = (code, bufs, out) => {
    const pl = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
    const bg = device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const e = device.createCommandEncoder(), c = e.beginComputePass();
    c.setPipeline(pl); c.setBindGroup(0, bg); c.dispatchWorkgroups(rows); c.end();
    device.queue.submit([e.finish()]);
  };
  run(SHADERS.matvec, [P, buf(f), X, Y1]);
  run(SHADERS.matvec_q8, [P, buf(q), buf(sc), X, Y2]);
  const read = async (b) => {
    const r = device.createBuffer({ size: 4 * rows, usage: S.MAP_READ | S.COPY_DST });
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, r, 0, 4 * rows); device.queue.submit([e.finish()]);
    await r.mapAsync(GPUMapMode.READ); const v = new Float32Array(r.getMappedRange().slice(0)); r.unmap(); return v;
  };
  const ref = new Float64Array(rows);
  for (let i = 0; i < rows; i++) { let s = 0; for (let c = 0; c < cols; c++) s += f[i * cols + c] * x[c]; ref[i] = s; }
  return { rows, cols, f32: await read(Y1), q8: await read(Y2), ref };
}

export async function createLlama(device, ggufBuffer) {
  const g = readGguf(ggufBuffer), M = g.meta;
  if (M['general.architecture'] !== 'llama') throw new Error('WebGPU path: llama GGUFs only');
  const d = M['llama.embedding_length'], L = M['llama.block_count'], ff = M['llama.feed_forward_length'];
  const nh = M['llama.attention.head_count'], nkv = M['llama.attention.head_count_kv'] || nh;
  const eps = M['llama.attention.layer_norm_rms_epsilon'], theta = M['llama.rope.freq_base'] || 10000;
  const ctx = Math.min(M['llama.context_length'], MAX_CTX), hd = d / nh, kvd = nkv * hd;
  const moe = !!g.tensors['blk.0.ffn_gate_inp.weight'];
  const E = moe ? (M['llama.expert_count'] || g.tensors['blk.0.ffn_gate_inp.weight'].n / d) : 0;
  const topk = moe ? (M['llama.expert_used_count'] || 2) : 0;
  const ffE = moe ? (M['llama.expert_feed_forward_length'] || g.tensors['blk.0.ffn_gate_exps.weight'].n / (E * d)) : 0;
  if (moe && (E > 256 || topk > 16)) throw new Error('WebGPU path: at most 256 experts, top-16');
  // Shapes come from the metadata: GGUFs written before the dims fix record 2-D tensors as
  // [out, in] (bytes are row-major [out][in] either way), so dims order cannot be trusted.
  const vocab = M['llama.vocab_size'] || (M['tokenizer.ggml.tokens'] || []).length ||
                Math.max(...g.tensors['token_embd.weight'].dims);
  const S = GPUBufferUsage;
  const storage = (arr) => {
    const b = device.createBuffer({ size: Math.max(16, arr.byteLength), usage: S.STORAGE | S.COPY_DST });
    device.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return b;
  };
  const scratch = (n) => device.createBuffer({ size: Math.max(16, 4 * n), usage: S.STORAGE | S.COPY_SRC | S.COPY_DST });
  const uniform = (words) => {
    const b = device.createBuffer({ size: 16, usage: S.UNIFORM | S.COPY_DST });
    device.queue.writeBuffer(b, 0, words);
    return b;
  };
  const u4 = (a, b = 0, c = 0, e = 0) => new Uint32Array([a, b, c, e]);
  const weight = (name) => storage(tensorF32(g, name));
  // A matrix for matvec: Q8_0 stays quantised (repacked to scales + packed int8), others go to f32.
  const mat = (name) => {
    const t = g.tensors[name];
    if (t.type !== GGML_Q8_0) return { q8: false, buf: weight(name) };
    const nb = t.n / 32, raw = new Uint8Array(g.buf, t.start, 34 * nb), dv = new DataView(g.buf, t.start);
    const q = new Uint8Array(32 * nb), sc = new Float32Array(nb);
    for (let b = 0; b < nb; b++) {
      sc[b] = f16ToF32(dv.getUint16(34 * b, true));
      q.set(raw.subarray(34 * b + 2, 34 * b + 34), 32 * b);
    }
    return { q8: true, Q: storage(q), D: storage(sc) };
  };
  const mv = (P, m, x, y, n) => m.q8 ? ['matvec_q8', bind('matvec_q8', [P, m.Q, m.D, x, y]), n]
                                    : ['matvec', bind('matvec', [P, m.buf, x, y]), n];
  const outName = g.tensors['output.weight'] ? 'output.weight' : 'token_embd.weight';   // tied embeddings

  // RoPE table exactly as ember builds it: f32 freq = 1 / theta^(i / head_dim), angle pos * freq.
  const cs = new Float32Array(ctx * hd);
  for (let pos = 0; pos < ctx; pos++)
    for (let i = 0; i < hd; i += 2) {
      const freq = Math.fround(1 / Math.fround(Math.pow(theta, i / hd)));
      const val = Math.fround(pos * freq);
      cs[pos * hd + i] = Math.cos(val); cs[pos * hd + i + 1] = Math.sin(val);
    }

  const W = {
    embd: weight('token_embd.weight'), outNorm: weight('output_norm.weight'), out: mat(outName), rope: storage(cs),
    layers: Array.from({ length: L }, (_, l) => ({
      attnNorm: weight(`blk.${l}.attn_norm.weight`), q: mat(`blk.${l}.attn_q.weight`), k: mat(`blk.${l}.attn_k.weight`),
      v: mat(`blk.${l}.attn_v.weight`), o: mat(`blk.${l}.attn_output.weight`), ffnNorm: weight(`blk.${l}.ffn_norm.weight`),
      ...(moe ? { router: mat(`blk.${l}.ffn_gate_inp.weight`), gateE: weight(`blk.${l}.ffn_gate_exps.weight`),
                  upE: weight(`blk.${l}.ffn_up_exps.weight`), downE: weight(`blk.${l}.ffn_down_exps.weight`) }
              : { gate: mat(`blk.${l}.ffn_gate.weight`), up: mat(`blk.${l}.ffn_up.weight`),
                  down: mat(`blk.${l}.ffn_down.weight`) }),
      kc: scratch(ctx * kvd), vc: scratch(ctx * kvd),
    })),
  };
  const A = { x: scratch(d), xn: scratch(d), q: scratch(d), k: scratch(kvd), v: scratch(kvd), att: scratch(d),
              g: scratch(moe ? topk * ffE : ff), u: scratch(moe ? topk * ffE : ff), logits: scratch(vocab),
              r: scratch(Math.max(E, 4)), sel: scratch(16), gate: scratch(16) };
  const readback = device.createBuffer({ size: 4 * vocab, usage: S.MAP_READ | S.COPY_DST });
  const step = device.createBuffer({ size: 16, usage: S.UNIFORM | S.COPY_DST });

  const pipe = {};
  for (const [k, code] of Object.entries(SHADERS))
    pipe[k] = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const bind = (k, bufs) => device.createBindGroup({ layout: pipe[k].getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const f32u = (n, e) => { const a = new ArrayBuffer(16); new Uint32Array(a)[0] = n; new Float32Array(a)[1] = e; return a; };
  const pNorm = uniform(f32u(d, eps)), pHeads = uniform(u4(nh, nkv, hd)), pFF = uniform(u4(ff));
  const pMat = (rows, cols, acc) => uniform(u4(rows, cols, acc ? 1 : 0));
  const pQ = pMat(d, d, false), pKV = pMat(kvd, d, false), pO = pMat(d, d, true), pUp = pMat(ff, d, false),
        pDown = pMat(d, ff, true), pOut = pMat(vocab, d, false), pEmb = step;
  const pRouter = moe ? pMat(E, d, false) : null, pSel = moe ? uniform(u4(E, topk)) : null,
        pExp = moe ? uniform(u4(ffE, d)) : null, pMoeDown = moe ? uniform(u4(d, ffE, topk)) : null,
        pFFk = moe ? uniform(u4(topk * ffE)) : null;
  const ffnSteps = (lw) => moe ? [
    mv(pRouter, lw.router, A.xn, A.r, E),
    ['moe_select', bind('moe_select', [pSel, A.r, A.sel, A.gate]), 1],
    ['matvec_exp', bind('matvec_exp', [pExp, A.sel, lw.gateE, A.xn, A.g]), [ffE, topk]],
    ['matvec_exp', bind('matvec_exp', [pExp, A.sel, lw.upE, A.xn, A.u]), [ffE, topk]],
    ['swiglu', bind('swiglu', [pFFk, A.g, A.u]), Math.ceil(topk * ffE / WG)],
    ['moe_down', bind('moe_down', [pMoeDown, A.sel, A.gate, lw.downE, A.g, A.x]), d],
  ] : [
    mv(pUp, lw.gate, A.xn, A.g, ff),
    mv(pUp, lw.up, A.xn, A.u, ff),
    ['swiglu', bind('swiglu', [pFF, A.g, A.u]), Math.ceil(ff / WG)],
    mv(pDown, lw.down, A.g, A.x, d),
  ];

  // The per-token program: [pipeline, bind group, workgroups] in order.
  const prog = [['embed', bind('embed', [pEmb, W.embd, A.x]), Math.ceil(d / WG)]];
  for (const lw of W.layers) {
    prog.push(['rmsnorm', bind('rmsnorm', [pNorm, A.x, lw.attnNorm, A.xn]), 1],
              mv(pQ, lw.q, A.xn, A.q, d),
              mv(pKV, lw.k, A.xn, A.k, kvd),
              mv(pKV, lw.v, A.xn, A.v, kvd),
              ['rope_kv', bind('rope_kv', [step, pHeads, W.rope, A.q, A.k, A.v, lw.kc, lw.vc]), Math.ceil(Math.max(d / 2, kvd) / WG)],
              ['attention', bind('attention', [step, pHeads, A.q, lw.kc, lw.vc, A.att]), nh],
              mv(pO, lw.o, A.att, A.x, d),
              ['rmsnorm', bind('rmsnorm', [pNorm, A.x, lw.ffnNorm, A.xn]), 1],
              ...ffnSteps(lw));
  }
  prog.push(['rmsnorm', bind('rmsnorm', [pNorm, A.x, W.outNorm, A.xn]), 1],
            mv(pOut, W.out, A.xn, A.logits, vocab));

  function submitToken(token, pos, wantLogits) {
    device.queue.writeBuffer(step, 0, u4(pos, token, d));
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    for (const [k, bg, n] of prog) {
      pass.setPipeline(pipe[k]); pass.setBindGroup(0, bg);
      if (Array.isArray(n)) pass.dispatchWorkgroups(n[0], n[1]); else pass.dispatchWorkgroups(n);
    }
    pass.end();
    if (wantLogits) enc.copyBufferToBuffer(A.logits, 0, readback, 0, 4 * vocab);
    device.queue.submit([enc.finish()]);
  }

  return {
    config: { d, L, ff: moe ? ffE : ff, nh, nkv, hd, ctx, vocab, eps, theta, experts: E, topk,
              dispatchesPerToken: prog.length },
    // Run tokens at positions startPos.. and return the next-token logits after the last one.
    async forward(tokens, startPos) {
      if (startPos + tokens.length > ctx) throw new Error('past the context');
      tokens.forEach((t, i) => submitToken(t, startPos + i, i === tokens.length - 1));
      await readback.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      return out;
    },
  };
}
