// Minimal GGUF reader for the WebGPU path: metadata and tensor views over one ArrayBuffer.
// Reads GGUF v3 (little-endian). Tensor dims follow GGUF order (ne[0] fastest), so a llama
// weight [ne0 = in, ne1 = out] is `out` contiguous rows of `in` floats.
export const GGML_F32 = 0, GGML_F16 = 1, GGML_Q8_0 = 8;

export function readGguf(buf) {
  const dv = new DataView(buf), dec = new TextDecoder();
  let p = 0;
  const u32 = () => { const v = dv.getUint32(p, true); p += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(p, true)); p += 8; return v; };
  const str = () => { const n = u64(); const s = dec.decode(new Uint8Array(buf, p, n)); p += n; return s; };
  const scalar = {
    0: () => dv.getUint8(p++), 1: () => dv.getInt8(p++),
    2: () => { const v = dv.getUint16(p, true); p += 2; return v; }, 3: () => { const v = dv.getInt16(p, true); p += 2; return v; },
    4: u32, 5: () => { const v = dv.getInt32(p, true); p += 4; return v; },
    6: () => { const v = dv.getFloat32(p, true); p += 4; return v; }, 7: () => dv.getUint8(p++) !== 0,
    10: u64, 11: () => { const v = Number(dv.getBigInt64(p, true)); p += 8; return v; },
    12: () => { const v = dv.getFloat64(p, true); p += 8; return v; },
  };
  const value = (t) => {
    if (t === 8) return str();
    if (t === 9) { const et = u32(), n = u64(); const a = new Array(n); for (let i = 0; i < n; i++) a[i] = value(et); return a; }
    const f = scalar[t];
    if (!f) throw new Error('GGUF: unknown value type ' + t);
    return f();
  };
  if (u32() !== 0x46554747) throw new Error('not a GGUF file');
  const version = u32();
  if (version < 2) throw new Error('GGUF version ' + version + ' unsupported');
  const nTensors = u64(), nKv = u64();
  const meta = {};
  for (let i = 0; i < nKv; i++) { const k = str(); meta[k] = value(u32()); }
  const infos = [];
  for (let i = 0; i < nTensors; i++) {
    const name = str(), nd = u32(), dims = [];
    for (let d = 0; d < nd; d++) dims.push(u64());
    infos.push({ name, dims, type: u32(), offset: u64() });
  }
  const align = meta['general.alignment'] || 32;
  const dataStart = Math.ceil(p / align) * align;
  const tensors = {};
  for (const t of infos) tensors[t.name] = { ...t, start: dataStart + t.offset, n: t.dims.reduce((a, b) => a * b, 1) };
  return { meta, tensors, buf };
}

// A tensor's values as a Float32Array (f32 viewed in place when aligned; f16 and Q8_0 dequantised).
export function tensorF32(g, name) {
  const t = g.tensors[name];
  if (!t) throw new Error('GGUF: no tensor ' + name);
  if (t.type === GGML_F32) {
    if (t.start % 4 === 0) return new Float32Array(g.buf, t.start, t.n);
    return new Float32Array(g.buf.slice(t.start, t.start + 4 * t.n));
  }
  const dv = new DataView(g.buf), out = new Float32Array(t.n);
  if (t.type === GGML_F16) {
    for (let i = 0; i < t.n; i++) out[i] = f16(dv.getUint16(t.start + 2 * i, true));
    return out;
  }
  if (t.type === GGML_Q8_0) {   // blocks of 32: an f16 scale, then 32 int8
    for (let b = 0; b < t.n / 32; b++) {
      const o = t.start + 34 * b, d = f16(dv.getUint16(o, true));
      for (let j = 0; j < 32; j++) out[32 * b + j] = d * dv.getInt8(o + 2 + j);
    }
    return out;
  }
  throw new Error(`GGUF: tensor ${name} has type ${t.type}, not supported by the WebGPU path yet`);
}

export function f16ToF32(h) { return f16(h); }

function f16(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}
