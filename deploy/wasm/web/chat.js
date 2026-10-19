// ember chat page, in-browser edition: the same chat layout as www/ and the
// server's /chat contract, answered by ember.cpp running in a Web Worker.
(function () {
  // Colour theme: follows the system unless the toggle has set one (as on
  // johnnyteutonic.github.io).
  const root = document.documentElement;
  try {
    const stored = localStorage.getItem('theme');
    if (stored === 'light' || stored === 'dark') root.setAttribute('data-theme', stored);
  } catch (e) { /* storage blocked */ }
  const toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const cur = root.getAttribute('data-theme') ||
        (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      const next = cur === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) { /* ignore */ }
      toggle.setAttribute('aria-label', next === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    });
  }

  const history = document.getElementById('history');
  const input = document.getElementById('message');
  const send = document.getElementById('send');
  const info = document.getElementById('model-info');
  const progress = document.getElementById('progress');
  const bar = progress ? progress.firstElementChild : null;
  const turns = [];  // [{user, assistant}] for the model's context
  let nextId = 1, busy = false, ready = false;
  const pending = new Map();
  // Typing and Send work while the model downloads: a message sent early
  // waits in the worker's queue and is answered as soon as the model is in.

  function scrollDown() { history.scrollTop = history.scrollHeight; }

  function typingDots() {
    const t = document.createElement('span');
    t.className = 'typing';
    t.setAttribute('role', 'img');
    t.setAttribute('aria-label', 'ember is writing');
    for (let i = 0; i < 3; i++) t.appendChild(document.createElement('span'));
    return t;
  }

  // One message row: user turns on the right, ember's on the left with an
  // avatar and name. text === null shows the typing indicator instead.
  function add(role, text) {
    const row = document.createElement('div');
    row.className = 'message appear ' + (role === 'user' ? 'user-message' : 'server-message');
    const body = document.createElement('div');
    body.className = 'msg-body';
    const who = document.createElement('span');
    if (role === 'user') {
      who.className = 'sr-only';
      who.textContent = 'You';
    } else {
      const avatar = document.createElement('span');
      avatar.className = 'avatar';
      avatar.setAttribute('aria-hidden', 'true');
      avatar.textContent = '◆';
      row.appendChild(avatar);
      who.className = 'who';
      who.textContent = 'ember';
    }
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    if (text === null) bubble.appendChild(typingDots());
    else bubble.textContent = text;
    body.append(who, bubble);
    row.appendChild(body);
    history.appendChild(row);
    scrollDown();
    return { row, body, bubble };
  }

  function addTiming(msg, text) {
    const t = document.createElement('div');
    t.className = 'timing';
    t.textContent = text;
    msg.body.appendChild(t);
    scrollDown();
  }

  function setBusy(b) {
    busy = b;
    send.disabled = b;
    send.textContent = b ? '...' : (ready ? 'Send' : 'Send (loading)');
    send.setAttribute('aria-busy', b ? 'true' : 'false');
    if (!b) input.focus();
  }

  input.placeholder = 'Type now; it answers once the model has loaded...';
  send.textContent = 'Send (loading)';

  const worker = new Worker('worker.js');
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'progress') {
      const mb = (x) => (x / 1e6).toFixed(1);  // decimal MB, as the page text
      info.dataset.state = 'loading';
      info.textContent = m.total ? `Loading the model into your browser: ${mb(m.loaded)} of ${mb(m.total)} MB`
                                 : `Loading the model into your browser: ${mb(m.loaded)} MB`;
      if (bar && m.total) bar.style.width = Math.min(100, (100 * m.loaded) / m.total).toFixed(1) + '%';
    } else if (m.type === 'ready') {
      if (progress) progress.hidden = true;
      if (m.ok === 1) {
        ready = true;
        info.dataset.state = 'ready';
        info.textContent = 'Ready. ember.cpp is running in your browser; nothing you type leaves this page.';
        input.placeholder = 'Say something...';
        if (!busy) setBusy(false);
      } else {
        info.dataset.state = 'error';
        info.textContent = 'The model could not be loaded here' + (m.error ? ': ' + m.error : '.');
        input.placeholder = 'The model did not load; try reloading the page.';
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
    add('user', text);
    setBusy(true);
    const thinking = add('model', null);
    if (!ready) {
      const note = document.createElement('span');
      note.className = 'note';
      note.textContent = 'Still loading the model; your message is next.';
      thinking.bubble.appendChild(note);
    }
    const t0 = performance.now();
    // Sampling as the server's defaults; a fresh seed per message.
    const out = await ask({ user_input: text, history: turns.slice(-16), max_new_tokens: 64,
                            temperature: 0.7, top_k: 40, top_p: 0.9,
                            seed: Math.floor(Math.random() * 4294967295) });
    if (out.error) {
      thinking.bubble.textContent = '(' + out.error + ')';
      thinking.row.classList.add('error-message');
    } else {
      thinking.bubble.textContent = out.reply;
      turns.push({ user: text, assistant: out.reply });
      const ms = Math.round(performance.now() - t0);
      addTiming(thinking, `${out.tokens} tokens in ${ms} ms`);
    }
    scrollDown();
    setBusy(false);
  }

  send.addEventListener('click', sendMessage);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); sendMessage(); }
  });
  input.focus();
})();
