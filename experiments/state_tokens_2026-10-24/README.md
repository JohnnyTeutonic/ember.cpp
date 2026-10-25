# Per-turn state tokens (24 Oct 2026)

The contract is paperkiln's `experiments/tiny_chat_long_2026-10/IR_SPEC_DRAFT.md` at paperkiln 0a58e5b6,
sections 1.1-1.4 and 5. ember reviewed and signed off on that text. Everything here runs on the CPU.
There are two sets of tests:
- a synthetic model: paperkiln's chat_sftfinal_v3 with an untrained `state_embd.weight` added
  (`make_state_gguf.py`);
- section 5's gate 2 on paperkiln's first trained state-token model, state4 (the last section).

## What ember does

- **Extended embedding** (1.1). `"state_weight"` on the embed subgraph's embedding node. Ids in
  [vocab, vocab + k) read row id - vocab of that tensor, and k is its rows ([k, d], ne0 = the token
  embedding's d). When the file names `paperkiln.state_tokens.k`, it must equal k. A file with k > 0
  and no state_weight is refused at load. Without a state_weight, an id past the table is refused,
  as before. The head is unchanged: its logits cover ids 0 .. vocab - 1 only, and no state id is
  ever sampled.
- **The mask** (1.3). Attention `"mask": "turn_state"` (`"causal"` is the default; any other value
  is refused). s(p) and c(p) come from the ids alone. `forward()` also refuses a state id out of
  order (a block opens with S_0 and runs S_0 .. S_{k-1}) and a text id inside an open block. A block
  may stay open across calls, as token-by-token decode needs. Each query gets a list of its visible
  kv rows, ascending: the held rows first, then this call's rows up to itself. The new CPU kernel
  `attention_rows_batch_cpu` (cpu_attention.cpp) is the sliding-window kernel's body run on that
  list. Its scores, maximum and sum cover the visible keys only and run sequentially in list order,
  so masked keys contribute exactly 0. Positions stay absolute (RoPE unchanged). The mask is refused
  with a window or sinks (on the node or the kv state), with depth_route / kv_rows "active", and with
  values other than causal or turn_state.
- **The template contract** (1.2). The session reads `paperkiln.state_tokens.k` from the GGUF and
  needs paperkiln's chat format and the interpreter. `ember::insert_state_blocks` (`state_tokens.h`)
  applies to every chat-mode prompt that `generate()` runs (/chat, the OpenAI endpoint and the WASM
  `ember_chat` all go through it), and to the token count used by the context rule:
  - after every `<|endoftext|>` that closes an exchange, the k state ids. As in paperkiln's
    serialiser (`tools/state_tokens.py`), an exchange opens at `<|user|>` and has a reply once an
    `<|assistant|>` follows it. An `<|assistant|>` with no open `<|user|>` starts nothing;
  - before a `<|user|>` that follows an unclosed reply (max_new_tokens, a stop string, an abort),
    `<|endoftext|>` and then the k state ids.

  Raw ids >= vocab are accepted only as a complete, ordered block directly after such an
  `<|endoftext|>`, and are kept as given rather than doubled. Anything else is refused. When a reply
  ends with a generated `<|endoftext|>`, the generation loop prefills `<|endoftext|>` S_0 .. S_{k-1}
  (k + 1 positions, no sampling) before returning. The cache then holds the finished exchange exactly
  as the next prompt writes it, and the next request reuses it. A reply cut short gets the
  `<|endoftext|>` and the block from the next prompt, which the template builds from the history
  text.
- **Eviction** (1.4). kv slot `"evict": "turn_state"` requires every attention node to use mask
  turn_state, and is refused with recurrent state. `EMBER_EVICT=0` keeps every row. `EMBER_EVICT=1`
  evicts for any all-turn_state IR. Unset, the IR decides. The cache is compact: row r holds position
  `kv_pos_[r]`, which is the identity when nothing is evicted. When a position of segment s is
  processed (at the start of a call, or at the end of one that crossed a block), the text rows of
  every segment below s are dropped, their state rows are kept, and the kept rows are moved down in
  order. A forward that starts inside a segment m whose text rows are gone throws. `resume_point()`
  then returns that segment's start, and the forward from there drops every row with s >= m and
  re-prefills. The session's prefix reuse calls `resume_point()` under eviction, so an edited earlier
  message takes this path. Rows of the current segment are kept, so regenerating the latest reply
  needs no re-prefill. That holds even after the post-reply block, because segment n's text goes
  only when a segment n + 1 position arrives.
- **CUDA.** `device_supported()` returns false for any IR with a state_weight, a turn_state mask or
  an evict slot, so CUDA builds serve these files on the CPU. Building the interpreter for the device
  path with such an IR is refused, as for any unsupported op. There is no device mask kernel yet.
- **Routing.** `has_state_tokens()` keeps these files off the llama path whatever their family_hint
  (model.cpp and api.cpp), as `has_windowed_attention()` does.
- **Speculative decoding.** A prompt-lookup draft is cut at its first state id (the head has no
  logits for one).

## Tests (CPU, ~/ember_wt/build_cpu)

Model: `chat_state_k4.gguf`, written by `make_state_gguf.py` into `~/ember_wt/state_tokens/` (ext4,
scratch). It is chat_sftfinal_v3 (llama, d 384, 6 layers, vocab 8192, ctx 256) with every tensor and
key copied, plus `state_embd.weight` [4, 384] (seeded normal rows at the token embedding's standard
deviation), `paperkiln.state_tokens.k = 4`, and an IR with state_weight, mask turn_state and evict
turn_state. family_hint stays "llama", so the routing above is exercised. The IR variants are
`ts` (the embedded one), `ts_noevict`, `ts_causal` (state_weight with the causal mask) and `plain`.
The gate is `state_tokens_gate.py` and its receipt is `state_tokens_gate_cpu.json` (ember f4fc5e0's CPU build;
the IR tools from `$KERAMOS_TOOLS`, else paperkiln's `spec/tools`, else its `tools/`).

| check | result |
|---|---|
| (a) no state ids: turn_state (evict on and off) vs the original file's causal IR, 3 sequences (42-80 tokens, specials included) | prefill and decode bit-equal |
| (b) prefill vs token-by-token decode, 3 multi-turn sequences with state blocks (84, 31, 55 tokens; an empty reply; one ends inside an open block), evict on and off | bit-equal; on = off bit-equal |
| (b) the mask vs an independent numpy reference of section 1.3 (keramos's ir_numpy, from paperkiln's `spec/tools`, node by node; the gate's own attention and extended embedding) | within 8.3e-7 of the range, top-1 at every position |
| (b) the mask is read: turn_state vs the causal mask, same ids | segment 0 bit-equal; later positions move by up to 7.4-13.6 (the causal reference itself is within 8.3e-7 of numpy) |
| (c) eviction on vs off vs a fresh full prefill, 12 calls (`state_tokens_test run`): turn-by-turn prompts, replies token by token with the block, a regenerated reply, an edited turn 1 (rewind into an evicted turn), a five-turn prefill in one call, then a rewind into evicted turn 0 | logits bit-equal across all three at every call (max diff 0.0). The evicting engine refuses a forward straight into an evicted turn and resumes at its start (27 and 0). Rows held: 26 vs 132 after five turns |
| (d) template rule (`state_tokens_test template`): 17 id sequences | all pass: block after a closing eot; eot + block before a `<|user|>` after a cut reply (also an empty one); a given block kept; eot outside an exchange untouched; an `<|assistant|>` with no open `<|user|>` starts nothing; incomplete, out-of-order, cut-off, misplaced and past-k state ids refused. Against paperkiln's serialiser on 3000 random sequences, see state4 below |
| (d) /chat end to end, greedy, 5 requests, 3 servers (evict on, `EMBER_EVICT=0`, `EMBER_PREFIX_REUSE=0`) | reply ids and prompt ids equal across the three servers on every request. Reply 1 stopped by max_new_tokens 6, and the next prompt has `<|endoftext|>` + S_0..S_3 after it. Every state id sits in a complete block after `<|endoftext|>` and before `<|user|>`. Request 3 reused 49 tokens (through reply 2's generated eot and block); the regenerated reply reused 60. The edited turn-1 message reused 20 with eviction (turn 1's start) and 25 without (the edit) |
| refusals (load and run time) | 13/13: window, sinks, evict without the mask, evict "lru", mask "bidirectional", k in the metadata with no state_weight, a state_weight whose rows differ from k, a state_weight of the wrong width, a block out of order, text inside a block, a block not starting at S_0, an id past k, a state id with no state_weight |

Regressions, on the new build:
- `recursion_gate.py` and `mor_gate.py` rewrite their receipts byte for byte. Recursion: 2.0e-7,
  1.8e-7 and 2.7e-7, greedy 40/40. MoR: 2.3e-7 and 1.9e-7, greedy and depths 40/40.
- `mor_refusals.py`: 9/9 refused.
- `recursion_smoke.py`: 3.0e-7 / 2.5e-7 against ir_numpy, top-1 at every position, decode equal to
  prefill, 4/4 bad maps refused.
- `scale_smoke.py`: 1.9e-7 and 2.4e-7 against ir_numpy, top-1 13/13, decode equal to prefill. That
  README quotes 1.9e-7. ember's logits there are unchanged bit for bit (below), so any move is on the
  reference side: ir_numpy.py has uncommitted paperkiln edits.
- **ember against itself.** `bit_regression.py` runs ir_logits built at 7d2133d (the commit before
  this change, `~/ember_old`) and the new build on 14 model/IR cases: gpt2_tiny (plain,
  the scale smoke IR, recursion 0 1 0 1), chat_sftfinal_v3, swa_tiny, kimi_tiny, mamba_tiny,
  moe_tiny_v3, attnres_tiny, the three recursion gate models and the two MoR models. Prefill and
  decode are bit-equal on all 14 (628 positions; `bit_regression_cpu.json`).
- CUDA tree (`build_cuda_check.sh`, HAS_CUDA on, compile only): graph_ir, cpu_attention, api and
  model compile with no errors.

## Commands

From Windows, each through `wsl.exe --cd ~ -e bash -lc "..."`; the gate goes through the CPU budget:

    bash ember.cpp/experiments/state_tokens_2026-10-24/build_cpu.sh        # sync + build ir_logits, state_tokens_test, tinyllama_server
    python3 ember.cpp/experiments/state_tokens_2026-10-24/make_state_gguf.py  # ~/ember_wt/state_tokens/chat_state_k4.gguf + IR variants
    cd /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete && \
      python3 scripts/cpu_budget.py run --session ember --job "ember state-token gate (CPU)" --cores 4 \
        --resume "rerun state_tokens_gate.py" -- python3 ember.cpp/experiments/state_tokens_2026-10-24/state_tokens_gate.py

## Choices to agree with paperkiln

- **Where the block rule lives.** The interpreter checks the block structure (ordered, complete
  before text) from the ids alone. The rule that a block sits directly after an `<|endoftext|>`
  closing an exchange needs the special ids, so the template layer (`insert_state_blocks`) enforces
  it. ember has no raw-ids endpoint today, so the rule covers every prompt the session runs. The
  interpreter tool `ir_logits` checks structure only.
- **When rows are evicted.** Section 1.4 allows deleting as soon as block n is written. ember waits
  until a later segment's position arrives. Logits are identical either way. The wait keeps a
  regenerated reply free of a re-prefill after the post-reply block.
- **The repetition penalty** (section 3 point 4) counts state ids as positions of the window and
  gives them no penalty (they have no logits). mtstudio's chat needs the same rule.
- **CUDA, WASM.** No device mask kernel, so the CPU serves these files, and the CUDA legs of gate 2
  are still open. The WASM build was not rebuilt or gated: it shares `generate()` and the
  interpreter, but nothing ran it.

## Gate 2 on paperkiln's state4 (CPU)

The model is `/mnt/c/ml_artifacts/state_gate/state4/`: llama_state, d 64, 2 layers, vocab 400, k 4,
mask turn_state, kv evict turn_state. It has 40 multi-turn prompts, and prompt 7's second reply is
cut after one word. The gate is `state4_gate.py` and its receipt is `state4_gate_cpu.json`. Every
file's sha256 equals manifest.json's (12/12).

| check | result |
|---|---|
| interpreter (IR embedded in state4.gguf), eviction on: log-probs of the 40 continuations against `mtstudio score` | worst 7.9e-7 relative (paperkiln's numpy vs mtstudio: 2.4e-6); greedy 40/40; decode bit-equal to prefill |
| eviction off (`EMBER_EVICT=0`) | the same numbers; ir_logits output text identical to eviction on |
| kv rows held after the context | equal to mtstudio's for every prompt, with eviction (mean 25.4) and without (mean 76.2); on = off = a fresh prefill bit for bit |
| template rule against paperkiln's serialiser (`state_tokens.insert` / `error`), 3000 random sequences, 1340 with state ids | 3000/3000 agree: same refusals, same output |
| /chat end to end, greedy, history parsed from prompts_text.jsonl; servers with eviction on, off and `EMBER_PREFIX_REUSE=0` | prompt ids equal score_in's 40/40 on each, prompt 7's cut reply closed with `<|endoftext|>` + block included. Reply + stop id equal the continuation 40/40 on each. Every prompt equals the serialiser's insert of its own text ids. Replies identical across the three servers. With eviction, each new conversation's shared prefix ends inside evicted turn 0, so it resumes at 0 |
| state4_q8_0.gguf (reported, not gated) | loads; log-probs within 7.6e-4, greedy 40/40, decode equal to prefill |

    python3 scripts/cpu_budget.py run --session ember --job "ember state4 gate (CPU)" --cores 4 \
      --resume "rerun state4_gate.py" -- python3 ember.cpp/experiments/state_tokens_2026-10-24/state4_gate.py
