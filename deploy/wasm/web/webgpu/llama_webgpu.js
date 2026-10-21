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
import { readGguf, tensorF32 } from './gguf.js';

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
  swiglu: `
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> g: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) i: vec3u) {
  if (i.x < p.n) { let a = g[i.x]; g[i.x] = a / (1.0 + exp(-a)) * u[i.x]; }
}`,
};

export async function createLlama(device, ggufBuffer) {
  const g = readGguf(ggufBuffer), M = g.meta;
  if (M['general.architecture'] !== 'llama') throw new Error('WebGPU path: llama GGUFs only');
  const d = M['llama.embedding_length'], L = M['llama.block_count'], ff = M['llama.feed_forward_length'];
  const nh = M['llama.attention.head_count'], nkv = M['llama.attention.head_count_kv'] || nh;
  const eps = M['llama.attention.layer_norm_rms_epsilon'], theta = M['llama.rope.freq_base'] || 10000;
  const ctx = Math.min(M['llama.context_length'], MAX_CTX), hd = d / nh, kvd = nkv * hd;
  if (g.tensors['blk.0.ffn_gate_inp.weight']) throw new Error('WebGPU path: mixture-of-experts not supported yet');
  const vocab = g.tensors['token_embd.weight'].dims[1];
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
    embd: weight('token_embd.weight'), outNorm: weight('output_norm.weight'), out: weight(outName), rope: storage(cs),
    layers: Array.from({ length: L }, (_, l) => ({
      attnNorm: weight(`blk.${l}.attn_norm.weight`), q: weight(`blk.${l}.attn_q.weight`), k: weight(`blk.${l}.attn_k.weight`),
      v: weight(`blk.${l}.attn_v.weight`), o: weight(`blk.${l}.attn_output.weight`), ffnNorm: weight(`blk.${l}.ffn_norm.weight`),
      gate: weight(`blk.${l}.ffn_gate.weight`), up: weight(`blk.${l}.ffn_up.weight`), down: weight(`blk.${l}.ffn_down.weight`),
      kc: scratch(ctx * kvd), vc: scratch(ctx * kvd),
    })),
  };
  const A = { x: scratch(d), xn: scratch(d), q: scratch(d), k: scratch(kvd), v: scratch(kvd), att: scratch(d),
              g: scratch(ff), u: scratch(ff), logits: scratch(vocab) };
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

  // The per-token program: [pipeline, bind group, workgroups] in order.
  const prog = [['embed', bind('embed', [pEmb, W.embd, A.x]), Math.ceil(d / WG)]];
  for (const lw of W.layers) {
    prog.push(['rmsnorm', bind('rmsnorm', [pNorm, A.x, lw.attnNorm, A.xn]), 1],
              ['matvec', bind('matvec', [pQ, lw.q, A.xn, A.q]), d],
              ['matvec', bind('matvec', [pKV, lw.k, A.xn, A.k]), kvd],
              ['matvec', bind('matvec', [pKV, lw.v, A.xn, A.v]), kvd],
              ['rope_kv', bind('rope_kv', [step, pHeads, W.rope, A.q, A.k, A.v, lw.kc, lw.vc]), Math.ceil(Math.max(d / 2, kvd) / WG)],
              ['attention', bind('attention', [step, pHeads, A.q, lw.kc, lw.vc, A.att]), nh],
              ['matvec', bind('matvec', [pO, lw.o, A.att, A.x]), d],
              ['rmsnorm', bind('rmsnorm', [pNorm, A.x, lw.ffnNorm, A.xn]), 1],
              ['matvec', bind('matvec', [pUp, lw.gate, A.xn, A.g]), ff],
              ['matvec', bind('matvec', [pUp, lw.up, A.xn, A.u]), ff],
              ['swiglu', bind('swiglu', [pFF, A.g, A.u]), Math.ceil(ff / WG)],
              ['matvec', bind('matvec', [pDown, lw.down, A.g, A.x]), d]);
  }
  prog.push(['rmsnorm', bind('rmsnorm', [pNorm, A.x, W.outNorm, A.xn]), 1],
            ['matvec', bind('matvec', [pOut, W.out, A.xn, A.logits]), vocab]);

  function submitToken(token, pos, wantLogits) {
    device.queue.writeBuffer(step, 0, u4(pos, token, d));
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    for (const [k, bg, n] of prog) { pass.setPipeline(pipe[k]); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(n); }
    pass.end();
    if (wantLogits) enc.copyBufferToBuffer(A.logits, 0, readback, 0, 4 * vocab);
    device.queue.submit([enc.finish()]);
  }

  return {
    config: { d, L, ff, nh, nkv, hd, ctx, vocab, eps, theta, dispatchesPerToken: prog.length },
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
