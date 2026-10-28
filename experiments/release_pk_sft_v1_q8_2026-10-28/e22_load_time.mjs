// First-load cost of a model in the Space's own ember.js, as worker.js does it (in node): the file's
// bytes into the in-memory filesystem, ember_load, unlink; then one greedy reply. Download time is
// computed from the size at stated bandwidths, not measured.
//   node e22_load_time.mjs PAGE_DIR MODEL.gguf
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const [dir, model] = process.argv.slice(2);
const createEmber = createRequire(import.meta.url)(path.resolve(dir, 'ember.js'));
const t0 = performance.now();
const ember = await createEmber();
const t1 = performance.now();
const buf = fs.readFileSync(model);
const t2 = performance.now();
ember.FS.writeFile('/model.gguf', buf);
const ok = ember.ccall('ember_load', 'number', ['string'], ['/model.gguf']);
ember.FS.unlink('/model.gguf');
const t3 = performance.now();
const req = { user_input: 'Hello, how are you today?', max_new_tokens: 60, temperature: 1.0, top_k: 1, top_p: 1.0, seed: 0 };
const out = JSON.parse(ember.ccall('ember_chat', 'string', ['string'], [JSON.stringify(req)]));
const t4 = performance.now();
const mb = buf.length / 1e6;
const res = {
  model: path.basename(model), bytes: buf.length, ember_load_ok: ok, heap_bytes_after_load: ember.HEAPF32.buffer.byteLength,
  ms_instantiate_wasm: t1 - t0, ms_fs_write_and_ember_load: t3 - t2, ms_first_reply: t4 - t3, first_reply_tokens: out.tokens,
  download_s_at: { '0.4 MB/s': mb / 0.4, '2.5 MB/s (20 Mbit/s)': mb / 2.5, '12.5 MB/s (100 Mbit/s)': mb / 12.5 },
};
console.log(JSON.stringify(res, null, 1));
