// The paperkiln IR (keramos IR_SPEC) on WebGPU, for ember's browser chat: the documents ember's
// interpreter runs (paperkiln's families and the third-party graphs ember builds from llama.cpp
// metadata) as WGSL, one row per token. The llama path (llama_webgpu.js) stays as it is for the
// files ember serves on its tuned llama path; this module takes every other document whose ops it
// has, and anything else stays in WebAssembly.
//
// How it runs. At load the three node lists (embed, block x repeat, head) are walked once with the
// document's own names: every write gets a fresh GPU buffer (so no dispatch reads and writes one
// buffer), per-layer attributes and `{i}` (weight_index, tensor_index) are resolved, and the result
// is one list of dispatches per token with bind groups made once. Per token only a small step
// buffer (position, token, the turn_state code of the position) changes; the logits of the last
// token of a call are read back. Prefill runs token by token (T = 1 per step), which the IR
// defines to equal prefill (IR_SPEC 3.9, 6.3); recurrent state stays on the GPU between calls.
//
// Numerics follow ember's CPU interpreter op for op (graph_ir.cpp, graph_ir_ops.cpp,
// graph_ir_mamba.cpp; IR_SPEC sections 5-7) in f32. Reductions (matrix products, norms, softmax
// sums) use other summation orders, so results are ROUNDING against the CPU (IR_SPEC 4.2, 4.3):
// gated on greedy tokens, the argmax at every position and logits to rounding
// (tools/webgpu_parity/ir_gate.mjs).
//
// What it refuses (irSupport, then the worker falls back to WebAssembly): recurrence nodes, rope
// heads "first", lists that fork, depth_attend with more than two sources beside its list, expr with
// more than 9 inputs, stored types other than F32, F16, BF16 and Q8_0 for matrices (vectors are read
// at load for every type gguf.js reads), and attention contexts past MAX_KEYS positions.
import { GGML_F16, GGML_F32, GGML_Q8_0, f16ToF32, readGguf, tensorF32 } from './gguf.js';

const GGML_BF16 = 30;
const WG = 64;
const RG = 256;              // single-workgroup reductions
export const MAX_KEYS = 4096; // attention's workgroup score array (16 KB)
const MAXWG = 65535;

const SUPPORTED = new Set(['embedding', 'pos_embedding', 'linear', 'scale', 'split', 'rmsnorm', 'layernorm', 'gelu',
  'silu', 'sigmoid', 'relu', 'add', 'mul', 'sub', 'affine', 'mul_col', 'rms_row', 'rope', 'attention', 'moe_ffn',
  'list', 'append', 'depth_attend', 'expr', 'linear_attention', 'ssm_scan', 'conv1d_causal', 'selective_scan',
  'depth_route']);

// ---- WGSL ----------------------------------------------------------------------------------------

const MATH = `
fn tanh_s(x: f32) -> f32 { let e = exp(-2.0 * abs(x)); let t = (1.0 - e) / (1.0 + e); return select(-t, t, x >= 0.0); }
fn erf_s(x: f32) -> f32 {   // Numerical Recipes' erfc (fractional error < 1.2e-7)
  let z = abs(x); let t = 1.0 / (1.0 + 0.5 * z);
  let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
          t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return select(r - 1.0, 1.0 - r, x >= 0.0);
}
fn log1p_s(x: f32) -> f32 { let u = 1.0 + x; if (u == 1.0) { return x; } return log(u) * x / (u - 1.0); }
fn pow_s(x: f32, y: f32) -> f32 {
  if (y == 0.0) { return 1.0; }
  let r = exp2(y * log2(abs(x)));
  if (x < 0.0) {
    if (fract(y) != 0.0) { return (x - x) / (x - x); }
    return select(r, -r, abs(y) % 2.0 == 1.0);
  }
  return r;
}
`;

// Matrices are stored TRANSPOSED at load ([in][out]: element (row r, column c) at c * rows + r), so
// one invocation per output row reads consecutive addresses across a workgroup (coalesced on a GPU,
// and no barriers: a software adapter runs these about 30x faster than a workgroup per row). By
// stored type (override WT): 0 f32 bits, 1 f16 pairs, 2 Q8_0 (int8 packed four to a word, the f32
// scale of the row's block of 32 columns at (c / 32) * rows + r), 3 bf16 pairs. Each element is the
// IR_SPEC 4.7 value, exactly.
const WVAL = `
override WT: u32 = 0u;
fn wv(c: u32, r: u32, rows: u32) -> f32 {
  let i = c * rows + r;
  if (WT == 0u) { return bitcast<f32>(A[i]); }
  if (WT == 1u) { let h = unpack2x16float(A[i >> 1u]); return select(h.x, h.y, (i & 1u) == 1u); }
  if (WT == 2u) { let q = f32(i32(A[i >> 2u] << (24u - 8u * (i & 3u))) >> 24u); return D[(c >> 5u) * rows + r] * q; }
  return bitcast<f32>(((A[i >> 1u] >> (16u * (i & 1u))) & 0xffffu) << 16u);
}`;

const P = `@group(0) @binding(0) var<storage, read> p: array<u32>;\nfn pf(k: u32) -> f32 { return bitcast<f32>(p[k]); }\n`;

export const IR_SHADERS = {
  // p: d, vocab, has_state; st: [pos, token, ...]
  embed: `${P}
@group(0) @binding(1) var<storage, read> st: array<u32>;
@group(0) @binding(2) var<storage, read> A: array<u32>;
@group(0) @binding(3) var<storage, read> D: array<f32>;
@group(0) @binding(4) var<storage, read> SW: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
${WVAL}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let d = p[0]; let tok = st[1];
  if (g.x >= d) { return; }
  if (p[2] != 0u && tok >= p[1]) { y[g.x] = SW[(tok - p[1]) * d + g.x]; } else { y[g.x] = wv(g.x, tok, p[3]); }
}`,
  // p: d ; y = x + P[pos]
  pos_embed: `${P}
@group(0) @binding(1) var<storage, read> st: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> PE: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x < p[0]) { y[g.x] = x[g.x] + PE[st[0] * p[0] + g.x]; }
}`,
  // p: rows, cols
  matvec: `${P}
@group(0) @binding(1) var<storage, read> A: array<u32>;
@group(0) @binding(2) var<storage, read> D: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
${WVAL}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let rows = p[0]; let cols = p[1];
  let row = g.y * ${MAXWG * WG}u + g.x;
  if (row >= rows) { return; }
  var s = 0.0;
  for (var c = 0u; c < cols; c++) { s += wv(c, row, rows) * x[c]; }
  y[row] = s;
}`,
  // p: n ; y += b
  addbias: `${P}
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) { if (g.x < p[0]) { y[g.x] = y[g.x] + b[g.x]; } }`,
  // p: n, f0, f1 ; OP: 0 silu, 1 sigmoid, 2 relu, 3 gelu tanh, 4 gelu erf, 5 affine
  ew1: `${P}${MATH}
override OP: u32 = 0u;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p[0]) { return; }
  let v = x[g.x];
  var r = 0.0;
  if (OP == 0u) { let sg = 1.0 / (1.0 + exp(-v)); r = v * sg; }
  else if (OP == 1u) { r = 1.0 / (1.0 + exp(-v)); }
  else if (OP == 2u) { r = select(0.0, v, v > 0.0); }
  else if (OP == 3u) { let c = 0.7978845608028654; let v3 = 0.044715 * v * v * v; r = 0.5 * v * (1.0 + tanh_s(c * (v + v3))); }
  else if (OP == 4u) { r = 0.5 * v * (1.0 + erf_s(v / 1.4142135623730951)); }
  else { let pr = v * pf(1u); r = pr + pf(2u); }
  y[g.x] = r;
}`,
  // p: n ; OP: 0 add, 1 mul, 2 sub
  ew2: `${P}
override OP: u32 = 0u;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p[0]) { return; }
  if (OP == 0u) { y[g.x] = a[g.x] + b[g.x]; } else if (OP == 1u) { y[g.x] = a[g.x] * b[g.x]; } else { y[g.x] = a[g.x] - b[g.x]; }
}`,
  // p: n ; y = x * c[0]
  mulcol: `${P}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) { if (g.x < p[0]) { y[g.x] = x[g.x] * c[0]; } }`,
  // p: n, has_bias ; y = (x * g) + b
  scale: `${P}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p[0]) { return; }
  let pr = x[g.x] * w[g.x];
  if (p[1] != 0u) { y[g.x] = pr + b[g.x]; } else { y[g.x] = pr; }
}`,
  // p: n, src offset, dst offset
  copy: `${P}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) { if (g.x < p[0]) { y[p[2] + g.x] = x[p[1] + g.x]; } }`,
  // p: g, eps ; one workgroup per group of g columns, weight [g] shared
  rmsnorm: `${P}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, ${RG}>;
@compute @workgroup_size(${RG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let g = p[0]; let o = wg.x * g;
  var s = 0.0;
  for (var i = l.x; i < g; i += ${RG}u) { s += x[o + i] * x[o + i]; }
  part[l.x] = s;
  workgroupBarrier();
  for (var k = ${RG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(part[0] / f32(g) + pf(1u));
  for (var i = l.x; i < g; i += ${RG}u) { y[o + i] = x[o + i] * inv * w[i]; }
}`,
  // p: n, eps ; (x - mean) / sqrt(var + eps) * w + b
  layernorm: `${P}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, ${RG}>;
@compute @workgroup_size(${RG}) fn main(@builtin(local_invocation_id) l: vec3u) {
  let n = p[0];
  var s = 0.0;
  for (var i = l.x; i < n; i += ${RG}u) { s += x[i]; }
  part[l.x] = s;
  workgroupBarrier();
  for (var k = ${RG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  let mean = part[0] / f32(n);
  workgroupBarrier();
  var v = 0.0;
  for (var i = l.x; i < n; i += ${RG}u) { let c = x[i] - mean; v += c * c; }
  part[l.x] = v;
  workgroupBarrier();
  for (var k = ${RG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(part[0] / f32(n) + pf(1u));
  for (var i = l.x; i < n; i += ${RG}u) { y[i] = (x[i] - mean) * inv * w[i] + b[i]; }
}`,
  // p: w, eps ; [1, w] -> [1, 1]: sqrt(ss / w + eps)
  rmsrow: `${P}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, ${RG}>;
@compute @workgroup_size(${RG}) fn main(@builtin(local_invocation_id) l: vec3u) {
  let n = p[0];
  var s = 0.0;
  for (var i = l.x; i < n; i += ${RG}u) { s += x[i] * x[i]; }
  part[l.x] = s;
  workgroupBarrier();
  for (var k = ${RG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  if (l.x == 0u) { y[0] = sqrt(part[0] / f32(n) + pf(1u)); }
}`,
  // p: heads (of this tensor), head_dim, n_rot, neox ; table [ctx][n_rot/2][cos, sin]
  rope: `${P}
@group(0) @binding(1) var<storage, read> st: array<u32>;
@group(0) @binding(2) var<storage, read> cs: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let hd = p[1]; let nr = p[2]; let half = nr / 2u;
  if (g.x >= p[0] * hd) { return; }
  let h = g.x / hd; let i = g.x % hd; let b = h * hd;
  if (i >= nr) { y[g.x] = x[g.x]; return; }
  let tb = 2u * st[0] * half;
  if (p[3] != 0u) {
    if (i < half) { let c = cs[tb + 2u * i]; let s = cs[tb + 2u * i + 1u]; y[g.x] = x[b + i] * c - x[b + i + half] * s; }
    else { let j = i - half; let c = cs[tb + 2u * j]; let s = cs[tb + 2u * j + 1u]; y[g.x] = x[b + j] * s + x[b + i] * c; }
  } else {
    let j = i / 2u; let c = cs[tb + 2u * j]; let s = cs[tb + 2u * j + 1u];
    if ((i & 1u) == 0u) { y[g.x] = x[g.x] * c - x[g.x + 1u] * s; } else { y[g.x] = x[g.x - 1u] * s + x[g.x] * c; }
  }
}`,
  // p: kvd, app, mor_step (0xffffffff: none) ; st: [pos, token, ...]; mor: [depth, gate]; cnt: rows per app
  kvstore: `${P}
@group(0) @binding(1) var<storage, read> st: array<u32>;
@group(0) @binding(2) var<storage, read> mor: array<u32>;
@group(0) @binding(3) var<storage, read> cnt: array<u32>;
@group(0) @binding(4) var<storage, read> k: array<f32>;
@group(0) @binding(5) var<storage, read> v: array<f32>;
@group(0) @binding(6) var<storage, read_write> kc: array<f32>;
@group(0) @binding(7) var<storage, read_write> vc: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let kvd = p[0];
  if (g.x >= kvd) { return; }
  var row = st[0];
  if (p[2] != 0xffffffffu) { if (mor[0] <= p[2]) { return; } row = cnt[p[1]]; }
  kc[row * kvd + g.x] = k[g.x]; vc[row * kvd + g.x] = v[g.x];
}`,
  // p: heads, kv_heads, head_dim, scale, window, sinks, mode (0 causal, 1 window, 2 turn_state, 3 MoR rows), app
  attn: `${P}
@group(0) @binding(1) var<storage, read> st: array<u32>;
@group(0) @binding(2) var<storage, read> cnt: array<u32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> q: array<f32>;
@group(0) @binding(5) var<storage, read> kc: array<f32>;
@group(0) @binding(6) var<storage, read> vc: array<f32>;
@group(0) @binding(7) var<storage, read_write> o: array<f32>;
var<workgroup> sc: array<f32, MAXK>;
var<workgroup> part: array<f32, ${WG}>;
fn visible(t: u32, pos: u32) -> bool {
  let mode = p[6];
  if (mode == 1u) { return t < p[5] || t + p[4] >= pos + 1u; }
  if (mode == 2u) { return codes[2u * t + 1u] == codes[2u * pos + 1u] || codes[2u * t] == 1u; }
  return true;
}
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let nh = p[0]; let hd = p[2]; let h = wg.x;
  let kvh = h / (nh / p[1]);
  let kvd = p[1] * hd;
  let qo = h * hd; let ko = kvh * hd;
  let pos = st[0];
  var n = pos + 1u;
  if (p[6] == 3u) { n = cnt[p[7]] + 1u; }
  let scale = pf(3u);
  var m = -3.40282e38;
  for (var t = l.x; t < n; t += ${WG}u) {
    if (visible(t, pos)) {
      var s = 0.0;
      for (var d = 0u; d < hd; d++) { s += q[qo + d] * kc[t * kvd + ko + d]; }
      s = s * scale; sc[t] = s; m = max(m, s);
    }
  }
  part[l.x] = m;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] = max(part[l.x], part[l.x + k]); } workgroupBarrier(); }
  let mx = part[0];
  workgroupBarrier();
  var z = 0.0;
  for (var t = l.x; t < n; t += ${WG}u) {
    var e = 0.0;
    if (visible(t, pos)) { e = exp(sc[t] - mx); }
    sc[t] = e; z += e;
  }
  part[l.x] = z;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  let zz = part[0];
  workgroupBarrier();
  for (var t = l.x; t < n; t += ${WG}u) { sc[t] = sc[t] / zz; }
  workgroupBarrier();
  for (var d = l.x; d < hd; d += ${WG}u) {
    var acc = 0.0;
    for (var t = 0u; t < n; t++) { let a = sc[t]; if (a != 0.0) { acc += a * vc[t * kvd + ko + d]; } }
    o[qo + d] = acc;
  }
}`,
  // p: E, k, renorm ; top-k router logits (ties to the lower index), ascending expert order, gates
  moe_select: `${P}
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> sel: array<u32>;
@group(0) @binding(3) var<storage, read_write> gate: array<f32>;
@compute @workgroup_size(1) fn main() {
  let E = p[0]; let K = p[1];
  var taken: array<bool, 256>;
  var pick: array<u32, 16>;
  for (var j = 0u; j < K; j++) {
    var best = 0xffffffffu;
    for (var e = 0u; e < E; e++) { if (!taken[e] && (best == 0xffffffffu || r[e] > r[best])) { best = e; } }
    taken[best] = true; pick[j] = best;
  }
  for (var a = 1u; a < K; a++) { let v = pick[a]; var b = a; while (b > 0u && pick[b - 1u] > v) { pick[b] = pick[b - 1u]; b--; } pick[b] = v; }
  var mx = -3.4e38;
  var z = 0.0;
  if (p[2] != 0u) {
    for (var j = 0u; j < K; j++) { mx = max(mx, r[pick[j]]); }
    for (var j = 0u; j < K; j++) { z += exp(r[pick[j]] - mx); }
  } else {
    for (var e = 0u; e < E; e++) { mx = max(mx, r[e]); }
    for (var e = 0u; e < E; e++) { z += exp(r[e] - mx); }
  }
  for (var j = 0u; j < K; j++) { sel[j] = pick[j]; gate[j] = exp(r[pick[j]] - mx) / z; }
}`,
  // p: rows, cols ; y[j][row] = W[sel[j]] x, each expert's matrix transposed ([E][cols][rows])
  matvec_exp: `${P}
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let row = g.x; let j = g.y; let rows = p[0]; let cols = p[1];
  if (row >= rows) { return; }
  let base = sel[j] * rows * cols;
  var s = 0.0;
  for (var c = 0u; c < cols; c++) { s += W[base + c * rows + row] * x[c]; }
  y[j * rows + row] = s;
}`,
  // p: n ; g = silu(g) * u
  swiglu: `${P}
@group(0) @binding(1) var<storage, read_write> g: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) i: vec3u) {
  if (i.x < p[0]) { let a = g[i.x]; let sg = 1.0 / (1.0 + exp(-a)); g[i.x] = a * sg * u[i.x]; }
}`,
  // p: rows (d), cols (d_ff), k ; y = sum_j gate_j down_{sel_j} h_j, ascending j
  moe_down: `${P}
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read> h: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
override K: u32 = 1u;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let row = g.x; let rows = p[0]; let cols = p[1];
  if (row >= rows) { return; }
  var acc = 0.0;
  for (var j = 0u; j < K; j++) {
    let base = sel[j] * rows * cols;
    var s = 0.0;
    for (var c = 0u; c < cols; c++) { s += W[base + c * rows + row] * h[j * cols + c]; }
    acc += gate[j] * s;
  }
  y[row] = acc;
}`,
  // p: d, eps ; sources: NB bank rows, then NE extra tensors (overrides) ; weight [d]
  depth_attend: `${P}
override NB: u32 = 1u;
override NE: u32 = 0u;
@group(0) @binding(1) var<storage, read> bank: array<f32>;
@group(0) @binding(2) var<storage, read> e1: array<f32>;
@group(0) @binding(3) var<storage, read> e2: array<f32>;
@group(0) @binding(4) var<storage, read> w: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, ${RG}>;
var<workgroup> la: array<f32, 128>;
fn src(i: u32, j: u32, d: u32) -> f32 {
  if (i < NB) { return bank[i * d + j]; }
  if (i == NB) { return e1[j]; }
  return e2[j];
}
@compute @workgroup_size(${RG}) fn main(@builtin(local_invocation_id) l: vec3u) {
  let d = p[0]; let S = NB + NE;
  if (S == 1u) { for (var j = l.x; j < d; j += ${RG}u) { y[j] = src(0u, j, d); } return; }
  for (var i = 0u; i < S; i++) {
    var s = 0.0;
    for (var j = l.x; j < d; j += ${RG}u) { let v = src(i, j, d); s += v * v; }
    part[l.x] = s;
    workgroupBarrier();
    for (var k = ${RG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
    let r = 1.0 / sqrt(part[0] / f32(d) + pf(1u));
    workgroupBarrier();
    var lg = 0.0;
    for (var j = l.x; j < d; j += ${RG}u) { let nv = src(i, j, d) * r; lg += nv * w[j]; }
    part[l.x] = lg;
    workgroupBarrier();
    for (var k = ${RG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
    if (l.x == 0u) { la[i] = part[0]; }
    workgroupBarrier();
  }
  if (l.x == 0u) {
    var mx = -3.40282e38;
    for (var i = 0u; i < S; i++) { mx = max(mx, la[i]); }
    var z = 0.0;
    for (var i = 0u; i < S; i++) { la[i] = exp(la[i] - mx); z += la[i]; }
    for (var i = 0u; i < S; i++) { la[i] = la[i] / z; }
  }
  workgroupBarrier();
  for (var j = l.x; j < d; j += ${RG}u) {
    var o = src(0u, j, d) * la[0];
    for (var i = 1u; i < S; i++) { let pr = src(i, j, d) * la[i]; o = o + pr; }
    y[j] = o;
  }
}
`,
  // p: w, den_floor ; elu+1 linear attention, num/den updated before the output
  linattn: `${P}
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> num: array<f32>;
@group(0) @binding(5) var<storage, read_write> den: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
fn elu1(x: f32) -> f32 { var r = x; if (r <= 0.0) { r = exp(r) - 1.0; } return r + 1.0; }
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let c = g.x;
  if (c >= p[0]) { return; }
  let pk = elu1(k[c]);
  let nm = num[c] + pk * v[c]; let dn = den[c] + pk;
  num[c] = nm; den[c] = dn;
  y[c] = elu1(q[c]) * nm / max(dn, pf(1u));
}`,
  // p: n ; one workgroup: s_t = A s_{t-1} + b u (ascending j), y = c s_t + d u
  ssmscan: `${P}
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read> C: array<f32>;
@group(0) @binding(5) var<storage, read> Dv: array<f32>;
@group(0) @binding(6) var<storage, read_write> s: array<f32>;
@group(0) @binding(7) var<storage, read_write> y: array<f32>;
var<workgroup> old: array<f32, 1024>;
@compute @workgroup_size(${RG}) fn main(@builtin(local_invocation_id) l: vec3u) {
  let n = p[0];
  for (var i = l.x; i < n; i += ${RG}u) { old[i] = s[i]; }
  workgroupBarrier();
  for (var i = l.x; i < n; i += ${RG}u) {
    var acc = 0.0;
    for (var j = 0u; j < n; j++) { acc += A[i * n + j] * old[j]; }
    acc += B[i] * u[i];
    s[i] = acc;
    y[i] = C[i] * acc + Dv[0] * u[i];
  }
}`,
  // p: C, K, has_bias ; depthwise causal conv over the K - 1 previous inputs, oldest first
  conv1d: `${P}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> s: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let C = p[0]; let K = p[1]; let c = g.x;
  if (c >= C) { return; }
  var acc = 0.0;
  for (var k = 0u; k < K; k++) {
    var v = x[c];
    if (k < K - 1u) { v = s[k * C + c]; }
    acc = acc + w[c * K + k] * v;
  }
  if (p[2] != 0u) { y[c] = acc + bias[c]; } else { y[c] = acc; }
  for (var k = 0u; k + 2u < K; k++) { s[k * C + c] = s[(k + 1u) * C + c]; }
  if (K > 1u) { s[(K - 2u) * C + c] = x[c]; }
}`,
  // p: D, N, has_d, softplus ; Mamba-1's selective scan, one channel per thread
  selscan: `${P}${MATH}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dt: array<f32>;
@group(0) @binding(3) var<storage, read> Bm: array<f32>;
@group(0) @binding(4) var<storage, read> Cm: array<f32>;
@group(0) @binding(5) var<storage, read> A: array<f32>;   // [D * N] then d [D]
@group(0) @binding(6) var<storage, read_write> s: array<f32>;
@group(0) @binding(7) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let D = p[0]; let N = p[1]; let c = g.x;
  if (c >= D) { return; }
  var delta = dt[c];
  if (p[3] != 0u && delta <= 20.0) { delta = log1p_s(exp(delta)); }
  let xd = x[c] * delta;
  var acc = 0.0;
  for (var n = 0u; n < N; n++) {
    let decay = s[c * N + n] * exp(delta * A[c * N + n]);
    let nv = decay + Bm[n] * xd;
    s[c * N + n] = nv;
    acc = acc + nv * Cm[n];
  }
  if (p[2] != 0u) { y[c] = acc + A[D * N + c] * x[c]; } else { y[c] = acc; }
}`,
  // p: R ; MoR routing from the router logits: mor = [depth, gate bits]
  mor_route: `${P}
@group(0) @binding(1) var<storage, read> l: array<f32>;
@group(0) @binding(2) var<storage, read_write> mor: array<u32>;
@compute @workgroup_size(1) fn main() {
  let R = p[0];
  var jm = 0u;
  for (var j = 1u; j < R; j++) { if (l[j] > l[jm]) { jm = j; } }
  var z = 0.0;
  for (var j = 0u; j < R; j++) { z += exp(l[j] - l[jm]); }
  mor[0] = jm + 1u; mor[1] = bitcast<u32>(exp(l[jm] - l[jm]) / z);
}`,
  // p: n, step, app, last_of_step, gate_score ; end of a MoR application: keep x for an inactive
  // token, count the row written, and at the step's last application gate x against the step input
  mor_end: `${P}
@group(0) @binding(1) var<storage, read> mor: array<u32>;
@group(0) @binding(2) var<storage, read> xprev: array<f32>;
@group(0) @binding(3) var<storage, read> xnew: array<f32>;
@group(0) @binding(4) var<storage, read> xin: array<f32>;
@group(0) @binding(5) var<storage, read_write> cnt: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let act = mor[0] > p[1];
  if (g.x == 0u && act) { cnt[p[2]] = cnt[p[2]] + 1u; }
  if (g.x >= p[0]) { return; }
  if (!act) { y[g.x] = xprev[g.x]; return; }
  if (p[3] != 0u && p[4] != 0u) {
    let gt = bitcast<f32>(mor[1]);
    let diff = xnew[g.x] - xin[g.x]; let pr = gt * diff; y[g.x] = xin[g.x] + pr;
  } else { y[g.x] = xnew[g.x]; }
}`,
};

// ---- expr -> WGSL (IR_SPEC 5.19) -------------------------------------------------------------------

const EXPR_FN = { exp: 1, log: 1, sqrt: 1, tanh: 1, erf: 1, abs: 1, sigmoid: 1, relu: 1, max: 2, min: 2, where: 3,
                  rowsum: 1, rowmean: 1, rowmax: 1 };

function f32lit(v) {
  const b = new Uint32Array(new Float32Array([v]).buffer)[0];
  return `bitcast<f32>(0x${b.toString(16)}u)`;
}

// Parses an expr string into WGSL for element j, with row reductions hoisted to workgroup scalars.
export function compileExpr(src, nIn) {
  const s = src.replace(/[ \t]/g, '');
  let i = 0;
  const reds = [];   // {kind, body}
  const names = ['x', ...'abcdefghijklmnopqrstuv'];
  const fail = (m) => { throw new Error(`expr: ${m} at column ${i + 1}`); };
  const peek = () => s[i];
  function expr() { let a = term(); while (peek() === '+' || peek() === '-') { const op = s[i++]; a = `(${a} ${op} ${term()})`; } return a; }
  function term() { let a = unary(); while (peek() === '*' || peek() === '/') { const op = s[i++]; a = `(${a} ${op} ${unary()})`; } return a; }
  function unary() { if (peek() === '-') { i++; return `(-${unary()})`; } return power(); }
  function power() {
    const start = i;
    const a = primary();
    if (peek() !== '^') return a;
    i++;
    const bStart = i;
    const b = unary();
    const lit = s.slice(bStart, i);
    if (/^[0-9]+$/.test(lit) && Number(lit) >= 1 && Number(lit) <= 4) return '(' + Array(Number(lit)).fill(a).join(' * ') + ')';
    void start;
    return `pow_s(${a}, ${b})`;
  }
  function primary() {
    const c = peek();
    if (c === undefined) fail('unexpected end');
    if (c === '(') { i++; const e = expr(); if (s[i++] !== ')') fail("expected ')'"); return e; }
    const num = /^(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)/.exec(s.slice(i));
    if (num) { i += num[0].length; const v = Math.fround(parseFloat(num[0])); if (!isFinite(v)) fail('non-finite number'); return f32lit(v); }
    const id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(s.slice(i));
    if (!id) fail(`unexpected '${c}'`);
    const name = id[0];
    i += name.length;
    if (name === 'pi') return f32lit(Math.fround(Math.PI));
    if (peek() === '(' && EXPR_FN[name]) {
      i++;
      const args = [expr()];
      while (peek() === ',') { i++; args.push(expr()); }
      if (s[i++] !== ')') fail("expected ')'");
      if (args.length !== EXPR_FN[name]) fail(`${name} takes ${EXPR_FN[name]} argument(s)`);
      const [a, b, cc] = args;
      switch (name) {
        case 'exp': case 'log': case 'sqrt': case 'abs': return `${name}(${a})`;
        case 'tanh': return `tanh_s(${a})`;
        case 'erf': return `erf_s(${a})`;
        case 'sigmoid': return `(1.0 / (1.0 + exp(-(${a}))))`;
        case 'relu': return `relu_s(${a})`;
        case 'max': return `max_s(${a}, ${b})`;
        case 'min': return `min_s(${a}, ${b})`;
        case 'where': return `select(${cc}, ${b}, (${a}) > 0.0)`;
        default: { reds.push({ kind: name, body: a }); return name === 'rowmean' ? `(red[${reds.length - 1}] / f32(W))` : `red[${reds.length - 1}]`; }
      }
    }
    const k = names.indexOf(name);
    if (k < 0 || k >= nIn) fail(`unknown name '${name}'`);
    return `in${k}[j]`;
  }
  const out = expr();
  if (i !== s.length) fail(`unexpected '${s[i]}'`);
  return { out, reds };
}

export function exprShader(src, nIn) {
  const { out, reds } = compileExpr(src, nIn);
  const binds = Array.from({ length: nIn }, (_, k) => `@group(0) @binding(${k + 1}) var<storage, read> in${k}: array<f32>;`).join('\n');
  const redCode = reds.map((r, k) => r.kind === 'rowmax'
    ? `{ var j = 0u; var m = ${r.body}; for (j = 1u; j < W; j++) { let v = ${r.body}; if (v > m) { m = v; } } red[${k}] = m; }`
    : `{ var acc = 0.0; for (var j = 0u; j < W; j++) { acc = acc + (${r.body}); } red[${k}] = acc; }`).join('\n    ');
  return `${P}${MATH}
${binds}
@group(0) @binding(${nIn + 1}) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, ${Math.max(1, reds.length)}>;
fn relu_s(e: f32) -> f32 { return select(0.0, e, e > 0.0); }
fn max_s(a: f32, b: f32) -> f32 { return select(b, a, a >= b); }
fn min_s(a: f32, b: f32) -> f32 { return select(b, a, a <= b); }
@compute @workgroup_size(${RG}) fn main(@builtin(local_invocation_id) l: vec3u) {
  let W = p[0];   // the row width (binding 0 is read, so the auto layout keeps it)
  if (l.x == 0u) {
    ${redCode}
  }
  workgroupBarrier();
  for (var j = l.x; j < W; j += ${RG}u) { y[j] = ${out}; }
}`;
}

// ---- support check ------------------------------------------------------------------------------------

function nodesOf(doc) { return [...doc.embed.nodes, ...doc.block.nodes, ...doc.head.nodes]; }

// {ok: true} when this module runs the document, else {ok: false, reason}.
export function irSupport(doc) {
  try {
    if (!doc || !doc.block || !doc.embed || !doc.head) return { ok: false, reason: 'not an IR document' };
    for (const n of nodesOf(doc)) {
      if (!SUPPORTED.has(n.op)) return { ok: false, reason: `op '${n.op}' has no WGSL kernel` };
      if (n.op === 'rope' && n.rope_heads === 'first') return { ok: false, reason: 'rope_heads "first"' };
      // T5's relative position bias (IR_SPEC 5.15, revision 1.3) has no WGSL kernel: the CPU interpreter runs it.
      if (n.op === 'attention' && n.position_bias !== undefined) return { ok: false, reason: 'attention position_bias' };
      if (n.op === 'expr') {
        if (n.in.length > 9) return { ok: false, reason: 'expr with more than 9 inputs' };
        compileExpr(n.expr, n.in.length);
      }
      if (n.op === 'depth_attend' && n.in.length > 3) return { ok: false, reason: 'depth_attend with more than two extra sources' };
      if (n.op === 'moe_ffn' && (n.n_experts > 256 || n.top_k > 16)) return { ok: false, reason: 'moe_ffn past 256 experts or top-16' };
    }
    return { ok: true };
  } catch (e) {
    return { ok: false, reason: String(e.message || e) };
  }
}

// ---- the model --------------------------------------------------------------------------------------

// createIr(device, ggufBuffer, doc, {ctx}): doc is the IR document (an object or its JSON text) ember's
// interpreter runs for this file; ctx caps the positions held (ember_ctx(), default model.ctx).
export async function createIr(device, ggufBuffer, docIn, opts = {}) {
  const doc = typeof docIn === 'string' ? JSON.parse(docIn) : docIn;
  const sup = irSupport(doc);
  if (!sup.ok) throw new Error('WebGPU IR path: ' + sup.reason);
  const g = readGguf(ggufBuffer);
  const model = doc.model, L = doc.block.repeat, vocab = model.vocab;
  const ctx = Math.min(model.ctx, opts.ctx > 0 ? opts.ctx : model.ctx);
  const S = GPUBufferUsage;
  const wmap = doc.block.weight_index ? doc.block.weight_index.per_layer : null;
  const at = (v, l) => (v && typeof v === 'object' && Array.isArray(v.per_layer)) ? v.per_layer[l] : v;
  const tname = (pat, l, n) => {
    if (!pat.includes('{i}')) return pat;
    if (l < 0) throw new Error(`'${pat}' uses {i} outside the block`);
    const idx = (n && n.tensor_index === 'application') || !wmap ? l : wmap[l];
    return pat.replace('{i}', String(idx));
  };
  const has = (n, k) => n[k] !== undefined && n[k] !== null;

  // Buffers.
  const allBufs = [];
  const mk = (bytes, usage) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage }); allBufs.push(b); return b; };
  const storage = (arr) => {
    const b = mk(arr.byteLength, S.STORAGE | S.COPY_DST);
    device.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength & ~3);
    if (arr.byteLength & 3) {   // a tail of fewer than 4 bytes
      const tail = new Uint8Array(4); tail.set(new Uint8Array(arr.buffer, arr.byteOffset + (arr.byteLength & ~3), arr.byteLength & 3));
      device.queue.writeBuffer(b, arr.byteLength & ~3, tail);
    }
    return b;
  };
  const scratch = (n) => mk(4 * Math.max(1, n), S.STORAGE | S.COPY_SRC | S.COPY_DST);
  const words = (vals) => {   // a parameter block: numbers (u32) or {f: x} (f32)
    const a = new ArrayBuffer(4 * Math.max(4, vals.length)), u = new Uint32Array(a), f = new Float32Array(a);
    vals.forEach((v, k) => { if (typeof v === 'object') f[k] = v.f; else u[k] = v >>> 0; });
    return storage(new Uint8Array(a));
  };
  const dummy = scratch(4);

  // Weights, by name (a shared stored layer is uploaded once).
  const tensor = (name) => { const t = g.tensors[name]; if (!t) throw new Error(`GGUF has no tensor '${name}'`); return t; };
  const vecCache = new Map(), matCache = new Map();
  const vecF32 = (name) => {
    const t = tensor(name);
    if (t.type === GGML_BF16) { const o = new Float32Array(t.n), u = new Uint32Array(o.buffer), dv = new DataView(g.buf, t.start); for (let k = 0; k < t.n; k++) u[k] = dv.getUint16(2 * k, true) << 16; return o; }
    return tensorF32(g, name);
  };
  const vec = (name, offset = 0) => {
    const key = name + '|' + offset;
    if (!vecCache.has(key)) {
      let v = vecF32(name);
      if (offset) v = v.map((x) => Math.fround(x + offset));
      vecCache.set(key, { buf: storage(v), n: v.length });
    }
    return vecCache.get(key);
  };
  const mat = (name) => {   // {wt, A, D, rows, cols}
    if (matCache.has(name)) return matCache.get(name);
    const t = tensor(name);
    const cols = t.dims[0], rows = t.n / cols;
    let m;
    if (t.type === GGML_F32 || t.type === GGML_F16 || t.type === GGML_BF16) {
      const wide = t.type === GGML_F32;
      const al = t.start % (wide ? 4 : 2) === 0;   // GGUF data is 32-byte aligned; copy only if not
      const src = wide ? (al ? new Uint32Array(g.buf, t.start, t.n) : new Uint32Array(g.buf.slice(t.start, t.start + 4 * t.n)))
                       : (al ? new Uint16Array(g.buf, t.start, t.n) : new Uint16Array(g.buf.slice(t.start, t.start + 2 * t.n)));
      const T = wide ? new Uint32Array(t.n) : new Uint16Array(t.n + 1);
      for (let r = 0; r < rows; r++) { const o = r * cols; for (let c = 0; c < cols; c++) T[c * rows + r] = src[o + c]; }
      m = { wt: t.type === GGML_F32 ? 0 : t.type === GGML_F16 ? 1 : 3, A: storage(wide ? T : new Uint8Array(T.buffer, 0, 2 * t.n)), D: dummy };
    } else if (t.type === GGML_Q8_0) {
      if (cols % 32) throw new Error(`Q8_0 tensor '${name}': ne0 ${cols} is not a multiple of 32`);
      const nb = cols / 32, dv = new DataView(g.buf, t.start), raw = new Int8Array(g.buf, t.start, 34 * nb * rows);
      const q = new Int8Array(t.n), sc = new Float32Array(nb * rows);
      for (let r = 0; r < rows; r++)
        for (let b = 0; b < nb; b++) {
          const o = 34 * (r * nb + b);
          sc[b * rows + r] = f16ToF32(dv.getUint16(o, true));
          for (let j = 0; j < 32; j++) q[(32 * b + j) * rows + r] = raw[o + 2 + j];
        }
      m = { wt: 2, A: storage(new Uint8Array(q.buffer)), D: storage(sc) };
    } else throw new Error(`WebGPU IR path: tensor '${name}' has GGML type ${t.type}; matrices must be F32, F16, BF16 or Q8_0`);
    Object.assign(m, { rows, cols });
    matCache.set(name, m);
    return m;
  };
  const expCache = new Map();
  const experts = (name, E) => {   // a 3-D stack [E][out][in] as E transposed matrices [E][in][out]
    if (!expCache.has(name)) {
      const t = tensor(name), cols = t.dims[0], rows = t.dims[1], v = vecF32(name), T = new Float32Array(t.n);
      for (let e = 0; e < E; e++) {
        const b = e * rows * cols;
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) T[b + c * rows + r] = v[b + r * cols + c];
      }
      expCache.set(name, storage(T));
    }
    return expCache.get(name);
  };
  const weightName = (n, l) => {
    let w = tname(n.weight, l, n);
    if (!g.tensors[w] && has(n, 'tied_to')) w = tname(n.tied_to, l, n);
    return w;
  };

  // Pipelines, by shader and override constants.
  const pipes = new Map();
  const pipe = (key, code, constants) => {
    const k = key + JSON.stringify(constants || {});
    if (!pipes.has(k))
      pipes.set(k, device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main', constants: constants || {} } }));
    return pipes.get(k);
  };
  const prog = [];
  const emit = (key, bufs, n, constants, code) => {
    const pl = pipe(key, code || IR_SHADERS[key], constants);
    const bg = device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const wx = Array.isArray(n) ? n : [Math.min(n, MAXWG), Math.ceil(n / MAXWG)];   // matvec rows: g.y * MAXWG * WG + g.x
    prog.push([pl, bg, wx]);
  };
  const groups = (n) => Math.ceil(n / WG);

  // Per-token inputs: st = [pos, token], the turn_state codes [ctx][c, s], MoR [depth, gate], rows per app.
  const st = mk(16, S.STORAGE | S.COPY_DST);
  const codes = scratch(2 * ctx);
  const morBuf = scratch(4);
  const cnt = scratch(Math.max(4, L));

  // Slots.
  const slots = Object.fromEntries((doc.block.state || []).map((s) => [s.name, s]));
  const kv = [], rec = [];   // per application: {slotName: {kc, vc}} / {slotName: {tensor: buf}}
  let maxKeys = 0;
  for (let l = 0; l < L; l++) {
    kv.push({}); rec.push({});
    for (const s of Object.values(slots)) {
      if (s.kind === 'kv') {
        const rows = Math.min(ctx, s.max_len || ctx);
        maxKeys = Math.max(maxKeys, rows);
        const kvd = s.kv_heads * s.head_dim;
        kv[l][s.name] = { kc: scratch(rows * kvd), vc: scratch(rows * kvd), rows };
      } else if (s.kind === 'recurrent') {
        rec[l][s.name] = Object.fromEntries(s.tensors.map((t) => [t.name, scratch(t.shape.reduce((a, b) => a * b, 1))]));
      }
    }
  }
  const keysCap = Math.min(MAX_KEYS, Math.floor((device.limits.maxComputeWorkgroupStorageSize - 4 * WG) / 4));
  if (maxKeys > keysCap) throw new Error(`WebGPU IR path: ${maxKeys} kv positions; this device holds ${keysCap} (cap the context)`);
  const keysArr = Math.max(WG, Math.ceil(maxKeys / WG) * WG);
  const exprIns = Math.max(0, ...nodesOf(doc).filter((n) => n.op === 'expr').map((n) => n.in.length));
  if (exprIns + 2 > device.limits.maxStorageBuffersPerShaderStage)
    throw new Error(`WebGPU IR path: an expr with ${exprIns} inputs needs more storage buffers than the device binds`);
  const turnState = doc.block.nodes.some((n) => n.op === 'attention' && n.mask === 'turn_state');
  const stateNode = doc.embed.nodes.find((n) => n.op === 'embedding' && has(n, 'state_weight'));
  const stateK = stateNode ? tensor(stateNode.state_weight).n / model.d : 0;

  // RoPE tables, per (theta, freq_scale, n_rot), as ember builds them.
  const ropeTabs = new Map();
  const ropeTable = (theta, fs, nr) => {
    const key = `${theta}|${fs}|${nr}`;
    if (!ropeTabs.has(key)) {
      const half = nr / 2, cs = new Float32Array(ctx * nr);
      for (let pos = 0; pos < ctx; pos++)
        for (let i = 0; i < nr; i += 2) {
          const freq = Math.fround(1 / Math.fround(Math.pow(Math.fround(theta), Math.fround(i / nr))));
          const val = Math.fround(Math.fround(pos * fs) * freq);
          cs[(pos * half + i / 2) * 2] = Math.fround(Math.cos(val));
          cs[(pos * half + i / 2) * 2 + 1] = Math.fround(Math.sin(val));
        }
      ropeTabs.set(key, storage(cs));
    }
    return ropeTabs.get(key);
  };

  // List capacity: list inputs plus every append that can run.
  const appends = (nodes) => nodes.filter((n) => n.op === 'append').length;
  const listCap = Math.max(1, ...nodesOf(doc).filter((n) => n.op === 'list').map((n) => n.in.length)) +
                  appends(doc.block.nodes) * L + appends(doc.head.nodes) + appends(doc.embed.nodes);

  // MoR (IR_SPEC 7.4): the block's first node, a router at the first application of step 0.
  const route = doc.block.nodes[0] && doc.block.nodes[0].op === 'depth_route' ? doc.block.nodes[0] : null;
  const stepOf = (l) => route ? at(route.step, l) : -1;

  const env = new Map();
  const tens = (n, k) => {
    const v = env.get(n.in[k]);
    if (!v) throw new Error(`IR op '${n.op}': input '${n.in[k]}' is not defined`);
    if (v.list) throw new Error(`IR op '${n.op}': input '${n.in[k]}' is a list`);
    return v;
  };
  const put = (name, w) => { const v = { buf: scratch(w), w }; env.set(name, v); return v; };
  const frozen = new Set();

  function runNode(n, l) {
    const op = n.op, out = Array.isArray(n.out) ? n.out : [n.out];
    if (op === 'embedding') {
      const wn = tname(n.weight, l, n), m = mat(wn);
      const sw = has(n, 'state_weight') ? vec(n.state_weight).buf : dummy;
      const y = put(out[0], m.cols);
      emit('embed', [words([m.cols, vocab, has(n, 'state_weight') ? 1 : 0, m.rows]), st, m.A, m.D, sw, y.buf], groups(m.cols), { WT: m.wt });
    } else if (op === 'pos_embedding') {
      const x = tens(n, 0), pe = vec(weightName(n, l));
      const y = put(out[0], x.w);
      emit('pos_embed', [words([x.w]), st, x.buf, pe.buf, y.buf], groups(x.w));
    } else if (op === 'linear') {
      const x = tens(n, 0), m = mat(weightName(n, l));
      if (m.cols !== x.w) throw new Error(`IR linear '${n.weight}': input width ${x.w}, weight ne0 ${m.cols}`);
      const y = put(out[0], m.rows);
      emit('matvec', [words([m.rows, m.cols]), m.A, m.D, x.buf, y.buf], groups(m.rows), { WT: m.wt });
      if (has(n, 'bias')) emit('addbias', [words([m.rows]), vec(tname(n.bias, l, n)).buf, y.buf], groups(m.rows));
    } else if (op === 'scale') {
      const x = tens(n, 0), w = vec(weightName(n, l));
      const b = has(n, 'bias') ? vec(tname(n.bias, l, n)).buf : dummy;
      const y = put(out[0], x.w);
      emit('scale', [words([x.w, has(n, 'bias') ? 1 : 0]), x.buf, w.buf, b, y.buf], groups(x.w));
    } else if (op === 'split') {
      const x = tens(n, 0);
      let col = 0;
      n.sizes.forEach((sz, k) => { const y = put(out[k], sz); emit('copy', [words([sz, col, 0]), x.buf, y.buf], groups(sz)); col += sz; });
      if (col !== x.w) throw new Error('IR split: sizes do not sum to the width');
    } else if (op === 'rmsnorm') {
      const x = tens(n, 0), gsz = has(n, 'group') ? n.group : x.w;
      const w = vec(weightName(n, l), has(n, 'weight_offset') ? n.weight_offset : 0);
      if (x.w % gsz || w.n !== gsz) throw new Error(`IR rmsnorm: group ${gsz} does not fit width ${x.w}`);
      const y = put(out[0], x.w);
      emit('rmsnorm', [words([gsz, { f: n.eps ?? 1e-5 }]), x.buf, w.buf, y.buf], [x.w / gsz, 1]);
    } else if (op === 'layernorm') {
      const x = tens(n, 0), w = vec(weightName(n, l)), b = vec(tname(n.bias, l, n));
      const y = put(out[0], x.w);
      emit('layernorm', [words([x.w, { f: n.eps ?? 1e-5 }]), x.buf, w.buf, b.buf, y.buf], [1, 1]);
    } else if (['silu', 'sigmoid', 'relu', 'gelu', 'affine'].includes(op)) {
      const x = tens(n, 0);
      let OP = { silu: 0, sigmoid: 1, relu: 2, affine: 5 }[op];
      if (op === 'gelu') { const ap = n.approx ?? 'none'; if (ap !== 'tanh' && ap !== 'none') throw new Error('gelu approx ' + ap); OP = ap === 'tanh' ? 3 : 4; }
      const y = put(out[0], x.w);
      const pw = op === 'affine' ? [x.w, { f: at(n.scale, l) }, { f: at(n.shift, l) }] : [x.w];
      emit('ew1', [words(pw), x.buf, y.buf], groups(x.w), { OP });
    } else if (op === 'add' || op === 'mul' || op === 'sub') {
      const a = tens(n, 0), b = tens(n, 1);
      if (a.w !== b.w) throw new Error(`IR ${op}: operand widths differ`);
      const y = put(out[0], a.w);
      emit('ew2', [words([a.w]), a.buf, b.buf, y.buf], groups(a.w), { OP: { add: 0, mul: 1, sub: 2 }[op] });
    } else if (op === 'mul_col') {
      const x = tens(n, 0), c = tens(n, 1);
      const y = put(out[0], x.w);
      emit('mulcol', [words([x.w]), x.buf, c.buf, y.buf], groups(x.w));
    } else if (op === 'rms_row') {
      const x = tens(n, 0), y = put(out[0], 1);
      emit('rmsrow', [words([x.w, { f: n.eps }]), x.buf, y.buf], [1, 1]);
    } else if (op === 'rope') {
      const q = tens(n, 0), k = tens(n, 1), hd = n.head_dim, nr = n.n_rot ?? hd;
      const tab = ropeTable(at(n.theta, l), at(n.freq_scale, l) ?? 1, nr);
      const neox = n.pairing === 'neox' ? 1 : 0;
      if (n.pairing !== 'neox' && n.pairing !== 'interleaved') throw new Error('rope pairing ' + n.pairing);
      const qo = put(out[0], q.w), ko = put(out[1], k.w);
      emit('rope', [words([q.w / hd, hd, nr, neox]), st, tab, q.buf, qo.buf], groups(q.w));
      emit('rope', [words([k.w / hd, hd, nr, neox]), st, tab, k.buf, ko.buf], groups(k.w));
    } else if (op === 'attention') {
      const q = tens(n, 0), k = tens(n, 1), v = tens(n, 2), slot = slots[n.state], c = kv[l][n.state];
      const hd = n.head_dim, nh = n.heads, nkv = n.kv_heads ?? nh;
      if (nkv !== slot.kv_heads || hd !== slot.head_dim) throw new Error('IR attention shape differs from its kv slot');
      const s = stepOf(l);
      const window = at(has(n, 'window') ? n.window : at(slot.window, l), l) || 0;
      const sinks = at(has(n, 'sinks') ? n.sinks : at(slot.sinks, l), l) || 0;
      const mode = s >= 0 ? 3 : n.mask === 'turn_state' ? 2 : window > 0 ? 1 : 0;
      const scale = has(n, 'scale') ? n.scale : Math.fround(1 / Math.sqrt(hd));
      emit('kvstore', [words([nkv * hd, l, s >= 0 ? s : 0xffffffff]), st, morBuf, cnt, k.buf, v.buf, c.kc, c.vc], groups(nkv * hd));
      const y = put(out[0], nh * hd);
      emit('attn', [words([nh, nkv, hd, { f: scale }, window, sinks, mode, l]), st, cnt, codes, q.buf, c.kc, c.vc, y.buf], [nh, 1],
           undefined, IR_SHADERS.attn.replace('MAXK', String(keysArr)));
    } else if (op === 'moe_ffn') {
      const x = tens(n, 0), E = n.n_experts, K = n.top_k, d = x.w;
      const R = mat(tname(n.router, l, n));
      const gn = tname(n.experts.gate, l, n), gt = tensor(gn), ffE = gt.dims[1];
      const G = experts(gn, E), Up = experts(tname(n.experts.up, l, n), E), Dn = experts(tname(n.experts.down, l, n), E);
      const r = scratch(E), sel = scratch(16), gate = scratch(16), hg = scratch(K * ffE), hu = scratch(K * ffE);
      emit('matvec', [words([E, d]), R.A, R.D, x.buf, r], groups(E), { WT: R.wt });
      emit('moe_select', [words([E, K, (n.gate_norm ?? 'renorm') === 'renorm' ? 1 : 0]), r, sel, gate], [1, 1]);
      emit('matvec_exp', [words([ffE, d]), sel, G, x.buf, hg], [groups(ffE), K]);
      emit('matvec_exp', [words([ffE, d]), sel, Up, x.buf, hu], [groups(ffE), K]);
      emit('swiglu', [words([K * ffE]), hg, hu], groups(K * ffE));
      const y = put(out[0], d);
      emit('moe_down', [words([d, ffE]), sel, gate, Dn, hg, y.buf], [groups(d), 1], { K });
    } else if (op === 'list') {
      const w = tens(n, 0).w, bank = scratch(listCap * w);
      n.in.forEach((nm, k) => { const t = tens(n, k); emit('copy', [words([w, 0, k * w]), t.buf, bank], groups(w)); });
      env.set(out[0], { list: true, bank, count: n.in.length, w, id: {} });
    } else if (op === 'append') {
      const Lv = env.get(n.in[0]);
      if (!Lv || !Lv.list) throw new Error(`IR append: '${n.in[0]}' is not a list`);
      if (frozen.has(Lv.id)) throw new Error('WebGPU IR path: a list appended twice from one value (forked lists)');
      frozen.add(Lv.id);
      const x = tens(n, 1);
      if (Lv.count >= listCap) throw new Error('IR list capacity');
      emit('copy', [words([x.w, 0, Lv.count * x.w]), x.buf, Lv.bank], groups(x.w));
      env.set(out[0], { list: true, bank: Lv.bank, count: Lv.count + 1, w: Lv.w, id: {} });
    } else if (op === 'depth_attend') {
      const Lv = env.get(n.in[0]);
      if (!Lv || !Lv.list) throw new Error(`IR depth_attend: '${n.in[0]}' is not a list`);
      const ex = n.in.slice(1).map((_, k) => tens(n, k + 1));
      const Sn = Lv.count + ex.length;
      if (Sn > 128) throw new Error('depth_attend: more than 128 sources');
      const w = Sn >= 2 ? vec(tname(n.weight, l, n)).buf : dummy;
      const y = put(out[0], Lv.w);
      emit('depth_attend', [words([Lv.w, { f: n.eps ?? 1e-5 }]), Lv.bank, ex[0] ? ex[0].buf : dummy, ex[1] ? ex[1].buf : dummy, w, y.buf],
           [1, 1], { NB: Lv.count, NE: ex.length });
    } else if (op === 'expr') {
      const xs = n.in.map((_, k) => tens(n, k)), w = xs[0].w;
      if (xs.some((t) => t.w !== w)) throw new Error('IR expr: inputs must share one width');
      const y = put(out[0], w);
      emit('expr:' + n.expr + '|' + n.in.length, [words([w]), ...xs.map((t) => t.buf), y.buf], [1, 1], undefined, exprShader(n.expr, n.in.length));
    } else if (op === 'linear_attention') {
      const q = tens(n, 0), k = tens(n, 1), v = tens(n, 2), w = n.heads * n.head_dim, r = rec[l][n.state];
      const y = put(out[0], w);
      emit('linattn', [words([w, { f: n.den_floor ?? 1e-8 }]), q.buf, k.buf, v.buf, r.num, r.den, y.buf], groups(w));
    } else if (op === 'ssm_scan') {
      const u = tens(n, 0), sn = n.n, r = rec[l][n.state];
      if (sn > 1024) throw new Error('ssm_scan: n past 1024');
      const A = vec(tname(n.a, l, n)).buf, B = vec(tname(n.b, l, n)).buf, C = vec(tname(n.c, l, n)).buf, Dd = vec(tname(n.d, l, n)).buf;
      const y = put(out[0], sn);
      emit('ssmscan', [words([sn]), u.buf, A, B, C, Dd, r.s, y.buf], [1, 1]);
    } else if (op === 'conv1d_causal') {
      const x = tens(n, 0), C = n.channels, K = n.kernel, r = rec[l][n.state];
      const w = vec(tname(n.weight, l, n)).buf, b = has(n, 'bias') ? vec(tname(n.bias, l, n)).buf : dummy;
      const y = put(out[0], C);
      emit('conv1d', [words([C, K, has(n, 'bias') ? 1 : 0]), x.buf, w, b, r.x, y.buf], groups(C));
    } else if (op === 'selective_scan') {
      const x = tens(n, 0), dt = tens(n, 1), Bm = tens(n, 2), Cm = tens(n, 3), D = n.channels, N = n.d_state, r = rec[l][n.state];
      const a = vecF32(tname(n.a, l, n)), ad = new Float32Array(D * N + D);
      ad.set(a.subarray(0, D * N));
      if (has(n, 'd')) ad.set(vecF32(tname(n.d, l, n)).subarray(0, D), D * N);
      const y = put(out[0], D);
      emit('selscan', [words([D, N, has(n, 'd') ? 1 : 0, (n.dt_softplus ?? true) ? 1 : 0]), x.buf, dt.buf, Bm.buf, Cm.buf, storage(ad), r.s, y.buf], groups(D));
    } else if (op === 'depth_route') {
      // handled by the block loop
    } else {
      throw new Error(`WebGPU IR path: op '${op}'`);
    }
  }

  for (const n of doc.embed.nodes) runNode(n, -1);
  let xin = null;
  for (let l = 0; l < L; l++) {
    const s = stepOf(l);
    if (s < 0) { for (const n of doc.block.nodes) runNode(n, l); continue; }
    const xname = route.in[0], x0 = env.get(xname);
    const first = l === 0 || stepOf(l - 1) !== s, last = l === L - 1 || stepOf(l + 1) !== s;
    if (s === 0 && first) {
      const R = mat(route.weight), lg = scratch(route.n_rec);
      emit('matvec', [words([route.n_rec, R.cols]), R.A, R.D, x0.buf, lg], groups(route.n_rec), { WT: R.wt });
      emit('mor_route', [words([route.n_rec]), lg, morBuf], [1, 1]);
    }
    if (first) xin = x0;
    for (const n of doc.block.nodes.slice(1)) runNode(n, l);
    const xnew = env.get(xname);
    const y = put(xname, x0.w);
    emit('mor_end', [words([x0.w, s, l, last ? 1 : 0, route.gate === 'score' ? 1 : 0]), morBuf, x0.buf, xnew.buf, xin.buf, cnt, y.buf], groups(x0.w));
  }
  for (const n of doc.head.nodes) runNode(n, -1);
  const lg = env.get('logits');
  if (!lg || lg.list) throw new Error("IR head produced no 'logits'");
  if (lg.w !== vocab) throw new Error(`IR logits are ${lg.w} wide, model.vocab ${vocab}`);
  const readback = device.createBuffer({ size: 4 * vocab, usage: S.MAP_READ | S.COPY_DST });

  // Host state: the sequence's ids (turn_state codes), and what is on the GPU.
  let seqLen = 0;
  const ids = [];
  let tsRun = 0, tsDone = 0;
  const recBufs = rec.flatMap((r) => Object.values(r).flatMap((t) => Object.values(t)));
  const hasRec = recBufs.length > 0;
  function reset() {
    const enc = device.createCommandEncoder();
    for (const b of recBufs) enc.clearBuffer(b);
    enc.clearBuffer(cnt);
    device.queue.submit([enc.finish()]);
    seqLen = 0; ids.length = 0; tsRun = 0; tsDone = 0;
  }
  function submitToken(token, pos, wantLogits) {
    if (token < 0 || token >= vocab + stateK) throw new Error(`token id ${token} out of range`);
    if (token >= vocab && !stateK) throw new Error(`token id ${token} out of range`);
    device.queue.writeBuffer(st, 0, new Uint32Array([pos, token, 0, 0]));
    if (turnState || stateK) {
      const c = token >= vocab ? 1 : 0, sseg = tsDone;
      if (token < vocab) tsRun = 0;
      else if (token === vocab) tsRun = 1;
      else if (token === vocab + tsRun) tsRun++;
      else tsRun = 0;
      if (stateK && tsRun === stateK) { tsDone++; tsRun = 0; }
      device.queue.writeBuffer(codes, 8 * pos, new Uint32Array([c, sseg]));
    }
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    for (const [pl, bg, n] of prog) { pass.setPipeline(pl); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(n[0], n[1]); }
    pass.end();
    if (wantLogits) enc.copyBufferToBuffer(lg.buf, 0, readback, 0, 4 * vocab);
    device.queue.submit([enc.finish()]);
  }

  return {
    config: { family_hint: doc.family_hint, d: model.d, L, vocab, ctx, state_k: stateK, dispatchesPerToken: prog.length,
              recurrent: hasRec, turn_state: turnState, mor: !!route },
    // Run tokens at positions startPos.. and return the next-token logits after the last one.
    // startPos 0 starts a new sequence; otherwise it continues the one on the GPU (startPos must be
    // its length when the document has recurrent or routed state, or state tokens).
    async forward(tokens, startPos) {
      if (startPos + tokens.length > ctx) throw new Error('past the context');
      if (startPos === 0) reset();
      else if (startPos > seqLen || ((hasRec || route || stateK) && startPos !== seqLen))
        throw new Error(`WebGPU IR path: a forward at ${startPos} after ${seqLen} positions`);
      if (startPos < seqLen) {   // attention-only: later rows are overwritten; the codes are recomputed
        ids.length = startPos;
        tsRun = 0; tsDone = 0;
        for (const t of ids) { if (t < vocab) tsRun = 0; else if (t === vocab) tsRun = 1; else if (t === vocab + tsRun) tsRun++; else tsRun = 0; if (stateK && tsRun === stateK) { tsDone++; tsRun = 0; } }
      }
      tokens.forEach((t, i) => { submitToken(t, startPos + i, i === tokens.length - 1); ids.push(t); });
      seqLen = startPos + tokens.length;
      await readback.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      return out;
    },
    destroy() { for (const b of allBufs) b.destroy(); readback.destroy(); },
  };
}
