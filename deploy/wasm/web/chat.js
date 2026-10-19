// ember chat page, in-browser edition: the same chat layout as www/ and the
// server's /chat contract, answered by ember.cpp running in a Web Worker.
(function () {
  const history = document.getElementById('history');
  const input = document.getElementById('message');
  const send = document.getElementById('send');
  const info = document.getElementById('model-info');
  const turns = [];  // [{user, assistant}] for the model's context
  let nextId = 1, busy = true;
  const pending = new Map();

  function add(cls, text) {
    const p = document.createElement('p');
    p.className = cls;
    p.textContent = text;
    history.appendChild(p);
    history.scrollTop = history.scrollHeight;
    return p;
  }

  function setBusy(b) {
    busy = b;
    send.disabled = b;
    input.disabled = b;
    if (!b) input.focus();
  }

  const worker = new Worker('worker.js');
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'progress') {
      const mb = (x) => (x / 1048576).toFixed(1);
      info.textContent = m.total ? `Loading the model into your browser: ${mb(m.loaded)} of ${mb(m.total)} MB`
                                 : `Loading the model into your browser: ${mb(m.loaded)} MB`;
    } else if (m.type === 'ready') {
      if (m.ok === 1) {
        info.textContent = 'Ready. ember.cpp is running in your browser; nothing you type leaves this page.';
        setBusy(false);
      } else {
        info.textContent = 'The model could not be loaded here' + (m.error ? ': ' + m.error : '.');
      }
    } else if (m.type === 'reply') {
      const done = pending.get(m.id);
      pending.delete(m.id);
      if (done) done(m.out);
    }
  };

  function ask(req) {
    return new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      worker.postMessage({ id, req });
    });
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text || busy) return;
    input.value = '';
    add('user-message', 'You: ' + text);
    setBusy(true);
    const thinking = add('server-message', 'AI: ...');
    const t0 = performance.now();
    // Sampling as the server's defaults; a fresh seed per message.
    const out = await ask({ user_input: text, history: turns.slice(-16), max_new_tokens: 64,
                            temperature: 0.7, top_k: 40, top_p: 0.9,
                            seed: Math.floor(Math.random() * 4294967295) });
    if (out.error) {
      thinking.textContent = 'AI: (' + out.error + ')';
    } else {
      thinking.textContent = 'AI: ' + out.reply;
      turns.push({ user: text, assistant: out.reply });
      const ms = Math.round(performance.now() - t0);
      add('timing', `${out.tokens} tokens in ${ms} ms`);
    }
    setBusy(false);
  }

  send.addEventListener('click', sendMessage);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendMessage(); });
  setBusy(true);
})();
