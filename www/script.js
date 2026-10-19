document.addEventListener('DOMContentLoaded', () => {
    // Colour theme: follows the system unless the toggle has set one (as on
    // johnnyteutonic.github.io).
    const root = document.documentElement;
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

    const messageInput = document.getElementById('message');
    const sendButton = document.getElementById('send');
    const historyDiv = document.getElementById('history');
    const serverUrl = '/chat'; // Endpoint defined in server.cpp

    let conversationHistory = []; // Store history as {role: 'user'/'assistant', content: '...'}
    let busy = false;  // one reply at a time: Send is disabled while the model answers

    function scrollDown() {
        historyDiv.scrollTop = historyDiv.scrollHeight;
    }

    function typingDots() {
        const t = document.createElement('span');
        t.className = 'typing';
        t.setAttribute('role', 'img');
        t.setAttribute('aria-label', 'ember is writing');
        for (let i = 0; i < 3; i++) t.appendChild(document.createElement('span'));
        return t;
    }

    // One message row: user turns on the right, the model's on the left with
    // an avatar and name. text === null shows the typing indicator instead.
    function renderMessage(role, text) {
        const row = document.createElement('div');
        row.classList.add('message', 'appear', role === 'user' ? 'user-message' : 'server-message');
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
        historyDiv.appendChild(row);
        scrollDown();
        return { row, body, bubble };
    }

    function addTiming(msg, text) {
        const timing = document.createElement('div');
        timing.classList.add('timing');
        timing.textContent = text;
        msg.body.appendChild(timing);
        scrollDown();
    }

    function addMessageToHistory(role, text, isError = false) {
        if (!isError) {  // error banners must not enter the model's prompt
            conversationHistory.push({ role: role, content: text });
        }
        const msg = renderMessage(role, text);
        if (isError) msg.row.classList.add('error-message');
        return msg;
    }

    function setBusy(b) {
        busy = b;
        sendButton.disabled = b;
        sendButton.setAttribute('aria-busy', b ? 'true' : 'false');
        if (!b) messageInput.focus();
    }

    // Build a multi-turn prompt from recent history. The word-level models
    // have a 128-word context window, so keep the prompt to a sliding window
    // of the most recent turns under a word budget (leaving room to reply).
    function buildPrompt(latestUserText) {
        const WORD_BUDGET = 80;
        const turns = conversationHistory
            .map(t => (t.role === 'user' ? 'user: ' : 'assistant: ') + t.content)
            .concat(['user: ' + latestUserText, 'assistant:']);
        // Take turns from the end until the budget is filled
        let words = 0;
        const kept = [];
        for (let i = turns.length - 1; i >= 0; i--) {
            const w = turns[i].split(/\s+/).length;
            if (words + w > WORD_BUDGET && kept.length > 1) break;
            kept.unshift(turns[i]);
            words += w;
        }
        return kept.join(' ');
    }

    // BPE chat models (paperkiln): the server applies the chat template and
    // fits the history to the context, so send the turns as they are.
    let isBpe = false;
    const info = document.getElementById('model-info');
    fetch('/health').then(r => r.ok ? r.json() : null).then(h => {
        if (!h) {
            if (info) { info.dataset.state = 'error'; info.textContent = 'The server did not report its model.'; }
            return;
        }
        isBpe = h.tokenizer === 'bpe';
        const name = (h.model || '').split('/').pop();
        document.title = 'ember chat: ' + name;
        if (info) {
            info.dataset.state = 'ready';
            info.textContent = `${name} · ${h.layers} layers · d${h.hidden_size} · ctx ${h.context} · ${h.tokenizer}`;
        }
    }).catch(() => {
        if (info) { info.dataset.state = 'error'; info.textContent = 'Could not reach the server.'; }
    });

    async function sendMessage() {
        const messageText = messageInput.value.trim();
        if (!messageText || busy) return; // Don't send empty messages, or while a reply is coming

        const sampling = {
            temperature: 0.7,  // Tiny models at 0.1 collapse to their most generic phrases
            top_k: 40,        // Limit to top 40 tokens
            top_p: 0.9        // Sample from 90% of probability mass
        };
        const body = isBpe
            ? { user_input: messageText, history: conversationHistory.slice(), max_new_tokens: 64, stream: true, ...sampling }
            : { user_input: buildPrompt(messageText), max_new_tokens: 45, ...sampling };
        addMessageToHistory('user', messageText);
        messageInput.value = ''; // Clear input field
        setBusy(true);
        try {
            if (isBpe) await streamReply(body);
            else await plainReply(body);
        } finally {
            setBusy(false);
        }
    }

    // Word-level models: one JSON reply.
    async function plainReply(body) {
        const msg = renderMessage('assistant', null);
        try {
            const response = await fetch(serverUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data.reply) {
                conversationHistory.push({ role: 'assistant', content: data.reply });
                msg.bubble.textContent = data.reply;
            } else {
                // The model ended its turn at once: show it, keep it out of the prompt.
                msg.bubble.textContent = '(no reply: the model ended its turn at once; try rephrasing)';
                msg.row.classList.add('error-message');
            }

        } catch (error) {
            console.error('Error sending message:', error);
            msg.bubble.textContent = `Error: ${error.message}`;
            msg.row.classList.add('error-message');
        }
        scrollDown();
    }

    // BPE chat models: the reply streams in as server-sent events
    // ({"token": text} pieces, then the full reply object with "done": true).
    // Under the reply: time to the first word and to the end, measured here
    // in the browser, and the server's own time to first token.
    async function streamReply(body) {
        const msg = renderMessage('assistant', null);
        const t0 = performance.now();
        let tFirst = null;
        let shown = '';
        try {
            const response = await fetch(serverUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';
            let final = null;
            for (;;) {
                const { value, done } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                let cut;
                while ((cut = buf.indexOf('\n\n')) >= 0) {
                    const line = buf.slice(0, cut);
                    buf = buf.slice(cut + 2);
                    if (!line.startsWith('data: ')) continue;
                    const ev = JSON.parse(line.slice(6));
                    if (ev.done) {
                        final = ev;
                    } else if (ev.token) {
                        if (tFirst === null) tFirst = performance.now();
                        shown += ev.token;
                        msg.bubble.textContent = shown;
                        scrollDown();
                    }
                }
            }
            const tEnd = performance.now();
            if (!final) throw new Error('the stream ended without a reply');
            if (final.reply) {
                msg.bubble.textContent = final.reply;
                conversationHistory.push({ role: 'assistant', content: final.reply });
                // Keep the turn pairs aligned: the user turn was pushed by addMessageToHistory.
            } else {
                msg.bubble.textContent = '(no reply: the model ended its turn at once; try rephrasing)';
                msg.row.classList.add('error-message');
                conversationHistory.pop();  // drop the unanswered user turn from the prompt
            }
            const first = tFirst === null ? '-' : `${Math.round(tFirst - t0)} ms`;
            addTiming(msg, `first word ${first} · done ${Math.round(tEnd - t0)} ms · ` +
                `${final.tokens} tokens · server TTFT ${final.ttft.ttft_ms.toFixed(1)} ms`);
        } catch (error) {
            console.error('Error streaming reply:', error);
            msg.bubble.textContent = `Error: ${error.message}`;
            msg.row.classList.add('error-message');
            conversationHistory.pop();
        }
    }

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.isComposing) {
            event.preventDefault();
            sendMessage();
        }
    });

    // The greeting is the page's own (index.html).

    messageInput.focus();
});

// Apply a remembered theme before first paint (this file loads at the end of
// <body>, ahead of DOMContentLoaded).
try {
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme === 'light' || storedTheme === 'dark') document.documentElement.setAttribute('data-theme', storedTheme);
} catch (e) { /* storage blocked */ }
