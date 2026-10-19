document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message');
    const sendButton = document.getElementById('send');
    const historyDiv = document.getElementById('history');
    const serverUrl = '/chat'; // Endpoint defined in server.cpp

    let conversationHistory = []; // Store history as {role: 'user'/'assistant', content: '...'} 

    function addMessageToHistory(role, text, isError = false) {
        if (!isError) {  // error banners must not enter the model's prompt
            conversationHistory.push({ role: role, content: text });
        }
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', role === 'user' ? 'user-message' : 'server-message');
        if (isError) {
            messageDiv.classList.add('error-message');
        }
        // Add prefix based on role
        const prefix = role === 'user' ? 'You: ' : 'AI: ';
        messageDiv.textContent = prefix + text; // Display with prefix
        historyDiv.appendChild(messageDiv);
        historyDiv.scrollTop = historyDiv.scrollHeight; // Scroll to the bottom
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
    fetch('/health').then(r => r.ok ? r.json() : null).then(h => {
        if (!h) return;
        isBpe = h.tokenizer === 'bpe';
        const name = (h.model || '').split('/').pop();
        document.title = 'ember chat: ' + name;
        const info = document.getElementById('model-info');
        if (info) info.textContent = `${name} · ${h.layers} layers · d${h.hidden_size} · ctx ${h.context} · ${h.tokenizer}`;
    }).catch(() => {});

    async function sendMessage() {
        const messageText = messageInput.value.trim();
        if (!messageText) return; // Don't send empty messages

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

        if (isBpe) {
            await streamReply(body);
            return;
        }

        try {
            const response = await fetch('/chat', {
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
                addMessageToHistory('assistant', data.reply);
            } else {
                // The model ended its turn at once: show it, keep it out of the prompt.
                addMessageToHistory('assistant', '(no reply: the model ended its turn at once; try rephrasing)', true);
            }

        } catch (error) {
            console.error('Error sending message:', error);
            addMessageToHistory('assistant', `Error: ${error.message}`, true);
        }
    }

    // BPE chat models: the reply streams in as server-sent events
    // ({"token": text} pieces, then the full reply object with "done": true).
    // Under the reply: time to the first word and to the end, measured here
    // in the browser, and the server's own time to first token.
    async function streamReply(body) {
        const div = document.createElement('div');
        div.classList.add('message', 'server-message');
        div.textContent = 'AI: ';
        historyDiv.appendChild(div);
        const t0 = performance.now();
        let tFirst = null;
        let shown = '';
        try {
            const response = await fetch('/chat', {
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
                        div.textContent = 'AI: ' + shown;
                        historyDiv.scrollTop = historyDiv.scrollHeight;
                    }
                }
            }
            const tEnd = performance.now();
            if (!final) throw new Error('the stream ended without a reply');
            if (final.reply) {
                div.textContent = 'AI: ' + final.reply;
                conversationHistory.push({ role: 'assistant', content: final.reply });
                // Keep the turn pairs aligned: the user turn was pushed by addMessageToHistory.
            } else {
                div.textContent = 'AI: (no reply: the model ended its turn at once; try rephrasing)';
                div.classList.add('error-message');
                conversationHistory.pop();  // drop the unanswered user turn from the prompt
            }
            const timing = document.createElement('div');
            timing.classList.add('timing');
            const first = tFirst === null ? '-' : `${Math.round(tFirst - t0)} ms`;
            timing.textContent = `first word ${first} · done ${Math.round(tEnd - t0)} ms · ` +
                `${final.tokens} tokens · server TTFT ${final.ttft.ttft_ms.toFixed(1)} ms`;
            historyDiv.appendChild(timing);
            historyDiv.scrollTop = historyDiv.scrollHeight;
        } catch (error) {
            console.error('Error streaming reply:', error);
            div.textContent = `AI: Error: ${error.message}`;
            div.classList.add('error-message');
            conversationHistory.pop();
        }
    }

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    const initialAIMessage = historyDiv.querySelector('.server-message');
    if(initialAIMessage) initialAIMessage.textContent = "AI: Hello! How can I help you today?";
    
    messageInput.focus();
}); 