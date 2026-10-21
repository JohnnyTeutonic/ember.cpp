// Greedy parity of the WebAssembly build against recorded replies, in node.
//   node tools/wasm_parity.mjs WASM_DIR MODEL.gguf REF.jsonl [ember.js] [OUT.json]
// REF rows: {i, prompt, <field>, mts_tokens?}; the field is REF_FIELD (default "mtstudio";
// "ember" compares with ember's own replies from tools/release_chat.py). MAX_NEW (60),
// REPEAT_PENALTY / REPEAT_LAST_N as for /chat. Exits 1 unless every reply is identical.
// OUT gets {n, identical, tokens_identical, ms_per_reply, replies}.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const [dir, gguf, ref, js = 'ember.js', outPath] = process.argv.slice(2);
const field = process.env.REF_FIELD || 'mtstudio';
const require = createRequire(import.meta.url);
const createEmber = require(path.resolve(dir, js));
const ember = await createEmber();
ember.FS.writeFile('/model.gguf', fs.readFileSync(gguf));
const ok = ember.ccall('ember_load', 'number', ['string'], ['/model.gguf']);
if (ok !== 1) throw new Error('ember_load returned ' + ok + ' (a larger model may need EMBER_WASM_MAX_MEMORY)');
const rows = fs.readFileSync(ref, 'utf8').trim().split('\n').map(JSON.parse);
let same = 0, sameTok = 0;
const replies = [];
const t0 = Date.now();
for (const r of rows) {
  const req = { user_input: r.prompt, max_new_tokens: Number(process.env.MAX_NEW || 60), temperature: 1.0, top_k: 1, top_p: 1.0, seed: 0 };
  if (process.env.REPEAT_PENALTY) {
    req.repeat_penalty = Number(process.env.REPEAT_PENALTY);
    req.repeat_last_n = Number(process.env.REPEAT_LAST_N || 64);
  }
  const out = JSON.parse(ember.ccall('ember_chat', 'string', ['string'], [JSON.stringify(req)]));
  replies.push(out.reply);
  if (out.reply === r[field]) same++; else console.log(`[${r.i}] wasm ${JSON.stringify(out.reply)}\n     ref  ${JSON.stringify(r[field])}`);
  if (r.mts_tokens !== undefined && out.tokens === r.mts_tokens) sameTok++;
}
const ms = (Date.now() - t0) / rows.length;
console.log(`n=${rows.length}: WASM reply == ${field} ${same}, token count ${sameTok} (${ms.toFixed(0)} ms per reply in node)`);
if (outPath) fs.writeFileSync(outPath, JSON.stringify({ n: rows.length, identical: same, tokens_identical: sameTok, ms_per_reply: ms, replies }));
process.exit(same === rows.length ? 0 : 1);
