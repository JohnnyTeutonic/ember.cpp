// Greedy parity of the WebAssembly build against recorded replies, in node.
//   node tools/wasm_parity.mjs WASM_DIR MODEL.gguf REF.jsonl [ember.js] [OUT.json]
// REF rows: {i, prompt or request, <field>, <tokens>?, <stop>?} (request: a /chat body with system and
// history, replayed with the parity sampling); the field is REF_FIELD (default "mtstudio";
// "ember" compares with ember's own replies from tools/release_chat.py), the token count and stop
// reason REF_TOKENS / REF_STOP (default mts_tokens / mts_stop). MAX_NEW (60), REPEAT_PENALTY /
// REPEAT_LAST_N as for /chat. Exits 1 unless every reply is identical and, where the rows record
// them, every token count and stop reason too.
// OUT gets {n, identical, tokens_identical, stop_identical, ms_per_reply, replies}.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const [dir, gguf, ref, js = 'ember.js', outPath] = process.argv.slice(2);
const field = process.env.REF_FIELD || 'mtstudio';
const tokField = process.env.REF_TOKENS || 'mts_tokens';
const stopField = process.env.REF_STOP || 'mts_stop';
const require = createRequire(import.meta.url);
const createEmber = require(path.resolve(dir, js));
const ember = await createEmber();
ember.FS.writeFile('/model.gguf', fs.readFileSync(gguf));
const ok = ember.ccall('ember_load', 'number', ['string'], ['/model.gguf']);
if (ok !== 1) throw new Error('ember_load returned ' + ok + ' (a larger model may need EMBER_WASM_MAX_MEMORY)');
const rows = fs.readFileSync(ref, 'utf8').trim().split('\n').map(JSON.parse);
let same = 0, sameTok = 0, sameStop = 0, haveTok = 0, haveStop = 0;
const replies = [];
const t0 = Date.now();
for (const r of rows) {
  // A row with a recorded /chat "request" (system message, history) replays it; otherwise its prompt.
  const req = { ...(r.request || { user_input: r.prompt }), max_new_tokens: Number(process.env.MAX_NEW || 60), temperature: 1.0, top_k: 1, top_p: 1.0, seed: 0 };
  if (process.env.REPEAT_PENALTY) {
    req.repeat_penalty = Number(process.env.REPEAT_PENALTY);
    req.repeat_last_n = Number(process.env.REPEAT_LAST_N || 64);
  }
  const out = JSON.parse(ember.ccall('ember_chat', 'string', ['string'], [JSON.stringify(req)]));
  replies.push(out.reply);
  if (out.reply === r[field]) same++; else console.log(`[${r.i}] wasm ${JSON.stringify(out.reply)}\n     ref  ${JSON.stringify(r[field])}`);
  if (r[tokField] !== undefined) { haveTok++; if (out.tokens === r[tokField]) sameTok++; }
  if (r[stopField] !== undefined) { haveStop++; if (out.stop_reason === r[stopField]) sameStop++; }
}
const ms = (Date.now() - t0) / rows.length;
console.log(`n=${rows.length}: WASM reply == ${field} ${same}, token count ${sameTok}/${haveTok}, ` +
            `stop reason ${sameStop}/${haveStop} (${ms.toFixed(0)} ms per reply in node)`);
if (outPath) fs.writeFileSync(outPath, JSON.stringify({ n: rows.length, identical: same, tokens_identical: sameTok,
                                                        stop_identical: sameStop, ms_per_reply: ms, replies }));
process.exit(same === rows.length && sameTok === haveTok && sameStop === haveStop ? 0 : 1);
