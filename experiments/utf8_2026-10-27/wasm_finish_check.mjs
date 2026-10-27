// The WebGPU page's text path without a GPU: ember_prepare, then ember_sample on one-hot logits for each of the
// native server's ids (top_k 1 picks it), then ember_finish(-1) (the length limit): its reply must equal the
// native /chat reply, U+FFFD where the reply was cut, and the JSON must parse.
//   node wasm_finish_check.mjs WASM_DIR MODEL.gguf REF.jsonl   (REF.jsonl: wasm_ref.py SERVER_CHECK.json REF.jsonl)
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const [dir, gguf, ref] = process.argv.slice(2);
const require = createRequire(import.meta.url);
const ember = await (require(path.resolve(dir, 'ember.js')))();
ember.FS.writeFile('/model.gguf', fs.readFileSync(gguf));
if (ember.ccall('ember_load', 'number', ['string'], ['/model.gguf']) !== 1) throw new Error('ember_load');
const rows = fs.readFileSync(ref, 'utf8').trim().split('\n').map(JSON.parse);
let ok = 0;
for (const r of rows) {
  const prep = JSON.parse(ember.ccall('ember_prepare', 'string', ['string'], [JSON.stringify(r.request)]));
  if (prep.error) throw new Error(prep.error);
  const n = r.vocab;
  const ptr = ember._malloc(4 * n);
  let picked = true;
  for (const id of r.ember.ids) {
    const lg = new Float32Array(n);
    lg[id] = 100;
    ember.HEAPF32.set(lg, ptr >> 2);
    if (ember._ember_sample(ptr, n) !== id) picked = false;
  }
  ember._free(ptr);
  const fin = JSON.parse(ember.ccall('ember_finish', 'string', ['number'], [-1]));
  const same = picked && fin.reply === r.ember.reply && fin.tokens === r.ember.tokens;
  ok += same;
  console.log(`k=${r.i} ${same ? 'equal' : 'DIFFERS'}: finish ${JSON.stringify(fin.reply.slice(-12))} ` +
              `native ${JSON.stringify(r.ember.reply.slice(-12))} (${fin.stop_reason})`);
}
console.log(`ember_finish: ${ok}/${rows.length} replies equal to the native server's`);
process.exit(ok === rows.length ? 0 : 1);
