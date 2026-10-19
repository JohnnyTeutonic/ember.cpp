// ember chat, in-browser: ember.cpp compiled to WebAssembly (deploy/wasm/build.sh)
// runs here, off the page's main thread. Messages:
//   in:  {id, req}  where req is the /chat body (user_input, history, sampling)
//   out: {type: 'progress', loaded, total} | {type: 'ready', ok} | {type: 'reply', id, out}
importScripts('ember.js');

let ember = null;

async function init() {
  ember = await createEmber();
  const res = await fetch('chat.gguf');
  if (!res.ok) throw new Error('model download failed: ' + res.status);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    postMessage({ type: 'progress', loaded, total });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const p of parts) { buf.set(p, off); off += p.length; }
  ember.FS.writeFile('/model.gguf', buf);
  const ok = ember.ccall('ember_load', 'number', ['string'], ['/model.gguf']);
  ember.FS.unlink('/model.gguf');  // the session holds its own copy of the weights
  postMessage({ type: 'ready', ok });
}

const ready = init().catch((e) => postMessage({ type: 'ready', ok: -1, error: String(e) }));

onmessage = async (e) => {
  await ready;
  const out = JSON.parse(ember.ccall('ember_chat', 'string', ['string'], [JSON.stringify(e.data.req)]));
  postMessage({ type: 'reply', id: e.data.id, out });
};
