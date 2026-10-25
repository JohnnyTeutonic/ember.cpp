// The browser page's chat path (deploy/wasm/build.sh's ember.js: ember_chat, ember_prepare) against
// ember's native server, in node, request by request.
//   node tools/wasm_chat_check.mjs WASM_DIR MODEL.gguf REF.jsonl OUT.json [--ctx N] [--js ember.js]
// REF rows: {i, request (the /chat body the native server was sent), ember: the server's reply object
// (reply, stop_reason, tokens, prompt_tokens, history_turns_used, and with return_ids prompt_ids)}.
// --ctx N calls ember_set_ctx(N) before ember_load (the page's ?ctx=N); without it the build's default
// cap applies. Per row: ember_prepare's prompt ids, prompt token count and turns used (the step-by-step
// path), then ember_chat's reply, token count, stop reason, prompt token count and turns used (the
// page's CPU path), each compared with the server's. OUT gets the counts, the context after the cap,
// the heap size after load, load and reply times, and every reply. Exits 1 unless everything agrees.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const args = process.argv.slice(2);
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const [dir, gguf, ref, outPath] = args;
const require = createRequire(import.meta.url);
const createEmber = require(path.resolve(dir, opt('--js', 'ember.js')));
const ember = await createEmber();
if (args.includes('--ctx')) ember._ember_set_ctx(Number(opt('--ctx')));
const t0 = Date.now();
ember.FS.writeFile('/model.gguf', fs.readFileSync(gguf));
const ok = ember.ccall('ember_load', 'number', ['string'], ['/model.gguf']);
ember.FS.unlink('/model.gguf');  // as the page
const loadMs = Date.now() - t0;
if (ok !== 1) throw new Error('ember_load returned ' + ok);
const ctx = ember._ember_ctx();
const heap = ember.HEAPF32.buffer.byteLength;
console.log(`loaded in ${loadMs} ms: ctx ${ctx}, heap ${(heap / 2 ** 20).toFixed(0)} MiB`);
const rows = fs.readFileSync(ref, 'utf8').trim().split('\n').map(JSON.parse);
const n = { prompt_ids: 0, prepare_counts: 0, reply: 0, tokens: 0, stop_reason: 0, chat_counts: 0 };
let haveIds = 0;
const out = [];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const t1 = Date.now();
for (const r of rows) {
  const e = r.ember;
  const prep = JSON.parse(ember.ccall('ember_prepare', 'string', ['string'], [JSON.stringify(r.request)]));
  if (prep.error) throw new Error(`[${r.i}] ember_prepare: ${prep.error}`);
  if (e.prompt_ids) { haveIds++; n.prompt_ids += same(prep.prompt_ids, e.prompt_ids); }
  n.prepare_counts += prep.prompt_tokens === e.prompt_tokens && prep.history_turns_used === e.history_turns_used;
  const c = JSON.parse(ember.ccall('ember_chat', 'string', ['string'], [JSON.stringify(r.request)]));
  if (c.error) throw new Error(`[${r.i}] ember_chat: ${c.error}`);
  n.reply += c.reply === e.reply;
  n.tokens += c.tokens === e.tokens;
  n.stop_reason += c.stop_reason === e.stop_reason;
  n.chat_counts += c.prompt_tokens === e.prompt_tokens && c.history_turns_used === e.history_turns_used;
  if (c.reply !== e.reply) console.log(`[${r.i}] wasm   ${JSON.stringify(c.reply)}\n     native ${JSON.stringify(e.reply)}`);
  out.push({ i: r.i, reply: c.reply, tokens: c.tokens, stop_reason: c.stop_reason, prompt_tokens: c.prompt_tokens,
             history_turns_used: c.history_turns_used, prompt_ids_equal: e.prompt_ids ? same(prep.prompt_ids, e.prompt_ids) : null });
}
const msPerReply = (Date.now() - t1) / rows.length;
const N = rows.length;
const pass = n.reply === N && n.tokens === N && n.stop_reason === N && n.chat_counts === N && n.prepare_counts === N &&
  n.prompt_ids === haveIds;
const summary = { n: N, ctx, heap_bytes: heap, load_ms: loadMs, ms_per_reply: msPerReply,
                  reply_identical: `${n.reply}/${N}`, tokens_identical: `${n.tokens}/${N}`,
                  stop_identical: `${n.stop_reason}/${N}`, prompt_tokens_and_turns_identical: `${n.chat_counts}/${N}`,
                  prepare_prompt_ids_identical: `${n.prompt_ids}/${haveIds}`,
                  prepare_counts_identical: `${n.prepare_counts}/${N}`, pass, replies: out };
fs.writeFileSync(outPath, JSON.stringify(summary, null, 1) + '\n');
console.log(JSON.stringify({ ...summary, replies: undefined }));
process.exit(pass ? 0 : 1);
