// The browser page's chat path under conversation schema v2 (paperkiln/2), in node, against the native server.
//   node tools/wasm_v2_check.mjs WASM_DIR TOY_DIR REF.jsonl OUT.json
// REF rows come from tools/chat_v2_server_check.py's 7th argument:
//   {"kind": "chat", request, ember}: a /chat request with tools, tool calls and results (some past the
//     256-token context) and the native server's greedy reply with return_ids: ember_prepare's prompt ids,
//     prompt token count and turns used, then ember_chat's reply, tool_calls, tokens, stop reason, prompt
//     token count and turns used, must equal the server's;
//   {"kind": "scripted", request, ids, max_new_tokens, expect}: ember_prepare, then ember_sample on one-hot
//     logits for each scripted id (top_k 1 picks it) until a stop id (then ember_finish(that id)) or the length
//     limit (ember_finish(-1)): the reply, tool_calls and stop reason must be the expected ones (the server's
//     scripted replies).
// And ember_load refuses toy_v2_nocontrol.gguf and toy_v2_badtype.gguf (-1). Exits 1 unless all agree.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const [dir, toyDir, ref, outPath] = process.argv.slice(2);
const require = createRequire(import.meta.url);
const ember = await (require(path.resolve(dir, 'ember.js')))();
const load = (f) => {
  ember.FS.writeFile('/model.gguf', fs.readFileSync(path.join(toyDir, f)));
  const r = ember.ccall('ember_load', 'number', ['string'], ['/model.gguf']);
  ember.FS.unlink('/model.gguf');
  return r;
};
const out = { load: {}, chat: [], scripted: [] };
let fails = 0;
for (const bad of ['toy_v2_nocontrol.gguf', 'toy_v2_badtype.gguf']) {
  const r = load(bad);
  out.load[bad] = r;
  if (r !== -1) fails++;
}
if (load('toy_v2.gguf') !== 1) throw new Error('ember_load toy_v2.gguf');
const call = (fn, req) => JSON.parse(ember.ccall(fn, 'string', ['string'], [JSON.stringify(req)]));
// JSON values compared with object members in sorted order (ember's JSON sorts keys; the reference keeps
// OpenAI's member order).
const canon = (x) => Array.isArray(x) ? x.map(canon)
  : x && typeof x === 'object' ? Object.fromEntries(Object.keys(x).sort().map((k) => [k, canon(x[k])])) : x;
const same = (a, b) => JSON.stringify(canon(a ?? null)) === JSON.stringify(canon(b ?? null));
const rows = fs.readFileSync(ref, 'utf8').trim().split('\n').map(JSON.parse);
const vocab = 16392;
for (const r of rows) {
  if (r.kind === 'chat') {
    const prep = call('ember_prepare', r.request);
    const chat = call('ember_chat', r.request);
    const e = r.ember;
    const ok = !prep.error && !chat.error && same(prep.prompt_ids, e.prompt_ids) && prep.prompt_tokens === e.prompt_tokens &&
      prep.history_turns_used === e.history_turns_used && chat.reply === e.reply && same(chat.tool_calls, e.tool_calls) &&
      chat.tokens === e.tokens && chat.stop_reason === e.stop_reason && chat.prompt_tokens === e.prompt_tokens &&
      chat.history_turns_used === e.history_turns_used;
    if (!ok) fails++;
    out.chat.push({ i: r.i, ok, reply: chat.reply, native: e.reply, prompt_tokens: prep.prompt_tokens,
                    history_turns_used: prep.history_turns_used, error: prep.error || chat.error || null });
    continue;
  }
  const prep = call('ember_prepare', r.request);
  if (prep.error) throw new Error(prep.error);
  const ptr = ember._malloc(4 * vocab);
  let stop = -1, picked = true, n = 0;
  for (const id of r.ids) {
    if (n >= prep.max_new_tokens) break;
    const lg = new Float32Array(vocab);
    lg[id] = 100;
    ember.HEAPF32.set(lg, ptr >> 2);
    const t = ember._ember_sample(ptr, vocab);
    if (t !== id) picked = false;
    n++;
    if (prep.stop_ids.includes(t)) { stop = t; break; }
  }
  ember._free(ptr);
  const fin = JSON.parse(ember.ccall('ember_finish', 'string', ['number'], [stop]));
  const ok = picked && fin.reply === r.expect.reply && same(fin.tool_calls ?? [], r.expect.tool_calls) &&
    fin.stop_reason === r.expect.stop_reason;
  if (!ok) fails++;
  out.scripted.push({ case: r.case, ok, reply: fin.reply, tool_calls: fin.tool_calls ?? [], stop_reason: fin.stop_reason });
}
out.summary = { chat: out.chat.length, chat_equal: out.chat.filter((x) => x.ok).length,
                scripted: out.scripted.length, scripted_equal: out.scripted.filter((x) => x.ok).length,
                load_refused: Object.values(out.load).filter((x) => x === -1).length, failures: fails };
fs.writeFileSync(outPath, JSON.stringify(out, null, 1) + '\n');
console.log(JSON.stringify(out.summary));
process.exit(fails === 0 ? 0 : 1);
