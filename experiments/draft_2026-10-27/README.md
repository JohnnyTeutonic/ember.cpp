# Draft-free drafting for chat: a corpus n-gram table behind prompt lookup (27 Oct 2026)

With the verification pass at about one decode step (`../verify_2026-10-27`), speculation with prompt-lookup drafts
was 1.11-1.14x plain decode on chat: at 92% of the decode steps on the 40 frozen chat prompts (pick_e1_L16, k = 4)
prompt lookup has no draft at all, so they run as plain steps. `EMBER_DRAFT=table` (opt-in, with `EMBER_SPEC=k`) fills
them from a small n-gram table built offline from paperkiln's chat text. On an L4, at k = 8, it is **1.91x plain
decode on chat40 for pick_e1_L16 (prompt lookup 1.13x) and 2.23x for pk_base_v1 (1.14x)**: 1.69x and 1.95x
prompt lookup's speed. On multi-turn conversations whose history repeats (Carry-5's served turns) it is 2.14x and
2.31x (prompt lookup 1.41x and 1.19x); on copy-heavy prompts 2.42x and 6.54x (2.13x and 5.80x). Ids equal plain
decode's on every request in every configuration (CUDA and CPU), and under `EMBER_DETERMINISTIC=1` every sampled
row's logits are plain decode's bit for bit. The drafter costs 2-8 µs per decode position. With `EMBER_DRAFT` unset
nothing changes. One limit: under `EMBER_DETERMINISTIC=1` the exact verification pass (`rows`) costs more than
the decode step it replaces, and the table's frequent drafts then lose on chat (0.74-0.94x); keep prompt lookup there.

## Why this drafter (the literature check)

The research item (LOCAL_LLM_STACK.md section 11 item 7) named Oilbird. Reading it and its neighbours against what
limits ember on chat (drafts missing at most positions, at batch 1, with a chain verification pass):

- **Oilbird** (arXiv 2608.03839) re-keys a draft pool by the verifier's hidden state at layer 0.85 L (cosine top 8,
  threshold 0.8, copy length set by the similarity). Its pool is **earlier requests' generated tokens** (a request
  cannot retrieve from its own trace), and the semantic key fires **only when the lexical match is non-empty**: it
  extends existing chains, never starts one. Its gains are on repetitive tool calling (API-Bank +6.2% from the key
  alone on the full benchmark, +24-29% accepted length when transplanted into three drafters there); on MT-Bench its
  accepted length (2.28) is Token Recycling's (2.30). In ember it would need the hidden states captured inside both
  captured graphs (decode and verify), a cross-request store (which this benchmark's warm pass would contaminate) and
  a tree merge, and it would not touch the positions where prompt lookup has nothing. Not chosen.
- **PLD+** (arXiv 2412.01447) and AdaPLD (2606.05742) use attention or hidden states to rank candidate spans of the
  input: better choice among matches, no drafts where there is no match.
- **Token Recycling** (arXiv 2408.08696, ACL 2025): an adjacency matrix of every row's top-k candidates, a draft
  tree, tree attention; about 2x on SpecBench for 7B models. Needs tree verification and top-k per row (ember's device
  record holds the top-1 and the runner-up's value).
- **Lookahead decoding** (Jacobi iterations): an n-gram pool from parallel Jacobi rows; needs a custom attention mask
  and W x N extra rows in every pass, inside the verify graph.
- **REST** (arXiv 2311.08252) drafts from a datastore built from a corpus by suffix match; **CREST** compacts the
  datastore to its most common n-grams at the same accepted length; **SuffixDecoding** (arXiv 2411.04975) keeps suffix
  trees over prompts and earlier outputs. The EACL 2026 study of 1-2B models (Mainardi, Sandikci and Vanschoren)
  finds drafting overhead, not draft quality, limits small models, and retrieval with negligible overhead gives the
  consistent gains.

A CREST-style compact table is the best gain per unit of complexity here: host code only (no device code, no graph
change), drafts at the positions prompt lookup misses, microseconds per lookup, exact by construction (verification
decides). The tiny models' replies are formulaic, close to their training dialogue, which a table of the most
frequent continuations covers.

## The drafter (`draft_table.h`, `api.cpp`)

- **Table** (`dr_table.py build`): for each order o = 1..4 and each context of the last o tokens seen at least 8
  times in the corpus, the most frequent next token and its share p (stored as round(255 p)); kept when p >= 0.10
  and the next-lower order would not give the same token anyway. Orders 1-3 are keyed by the packed ids, order 4 by a
  48-bit hash of them (a collision only costs a wrong draft). Sorted keys, binary search. The file names its
  vocabulary size; a table for another vocabulary is refused at load (prompt lookup only, one warning).
- **Draft**: prompt lookup first, unchanged; where it finds nothing, from the history plus the draft so far the
  longest order whose context is in the table gives the next token, until k tokens or until the product of the
  entries' shares falls below theta (`EMBER_DRAFT_THETA`, default 0.02). The verification pass and the accept loop
  are untouched, so the replies are plain decode's whatever the table holds.
- **Switches**: `EMBER_DRAFT=table` with `EMBER_DRAFT_TABLE=<file>`; unset or `pld` is the previous behaviour. The
  drafter's own time is counted (`spec_draft_ms`, `spec_draft_calls` in `/chat`'s ttft block, only when speculation
  ran).
- **Tables measured** (`C:/ml_artifacts/ember_draft/tables/`, sha256 in `receipts/tables.json`), each tokenised
  through the models' shared 16k Llama 3-scheme vocabulary (`dr_sim.py vocab`: the two GGUFs' tokens, merges, model
  and pre-tokeniser equal):
  - `h_c8_p10.bin`, 10.3 MB (14,151 / 267,966 / 231,134 / 422,618 entries for orders 1-4), from the 13 human
    dialogue sources of the tiny-chat SFT data (40.1M tokens), for pick_e1_L16;
  - `s_c8_p10.bin`, 8.4 MB (14,689 / 132,267 / 154,980 / 459,193), from lever2's SODA (first 150 MB) and OASST
    text (36.2M tokens), pk_base_v1's dialogue sources, for pk_base_v1.

## How the settings were chosen (`dr_sim.py`, `receipts/sim/dev_sweep.jsonl`)

Greedy speculation is exact, so a run's passes, drafts and acceptances follow from plain decode's prompt ids and
ids alone. `dr_sim.py` replays api.cpp's loop over CPU records of plain decode (`dr_collect.sh`, HEAD 08ead73).
Checked against ember: prompt lookup's counts on verify_2026-10-27's CUDA runs, 60/60 requests at k = 2, 4, 8; and
here **every one of the 22 GPU configurations below, 260/260 requests each**, drafter table included
(`receipts/cuda/predict_vs_simulator.txt`).

Table, theta and k were chosen on development sets only: chatdev (the other 160 tinychat_hf_prompts_v1 prompts) and
carrydev (40 other Carry-5 items, none in carry40). Nine candidate tables (min count 2-8, min share 0.05-0.25, orders
3 or 4, human or SODA text) at theta 0-0.1. Estimated speed-up at k = 8 (L4 pass costs from verify_2026-10-27):

| model | dev set | prompt lookup | chosen table, theta 0.02 |
|---|---|---|---|
| pick_e1_L16 | chatdev | 1.27 | 2.01 |
| | carrydev | 1.32 | 1.86 |
| pk_base_v1 | chatdev | 1.09 | 2.24 |
| | carrydev | 1.20 | 2.19 |

Each model does best with the table from its own training dialogue (crossed, chatdev: pick_e1_L16 with the SODA
table 1.82, pk_base_v1 with the human table 1.81). A lower min count or min share grows the table (24-69 MB) for at
most 0.6% more; a min share of 0.25 loses 9-13%; orders 1-3 only lose 1-5% (pick_e1_L16).

## Results (one Colab L4, session `ember-dr-1027`, 31 min; `receipts/cuda/`)

Build: ember main 08ead73 plus this change, Release for sm_89. Requests (`dr_prompts.py`, 260 per run):
**chat40** (verify_2026-10-27's 40 frozen chat prompts, 60 new tokens), **copy20** (its 20 text-repeating requests,
80), **carry40** (the 40 Carry-5 spot-check items as carry5_eval serves the self turns: five turns, each carrying the
earlier user turns and the model's own replies, 24 new tokens; 200 requests). Each request sent twice, the second
recorded (`dr_e2e.py`). Decode tokens per second, after the first token. Plain decode measured at the start and the
end of each model's runs: within 0.7%.

k = 8, theta 0.02 (`e2e_summary.txt`):

| model | set | plain tok/s | prompt lookup (x plain) | table (x plain) | table / prompt lookup | tokens per pass, PL / table | positions run (steps + passes), PL / table |
|---|---|---|---|---|---|---|---|
| pick_e1_L16 | chat40 | 715 | 809 (1.131) | **1367 (1.911)** | **1.69** | 2.58 / 1.65 | 523 / 352 of 591 |
| | copy20 | 645 | 1373 (2.127) | 1559 (2.415) | 1.14 | 4.09 / 2.55 | 313 / 278 of 702 |
| | carry40 | 647 | 911 (1.407) | **1384 (2.138)** | **1.52** | 2.92 / 2.02 | 1390 / 935 of 2034 |
| pk_base_v1 | chat40 | 374 | 427 (1.142) | **833 (2.225)** | **1.95** | 3.66 / 1.89 | 793 / 464 of 902 |
| | copy20 | 341 | 1977 (5.799) | 2230 (6.542) | 1.13 | 7.28 / 6.00 | 168 / 154 of 984 |
| | carry40 | 342 | 406 (1.186) | **789 (2.307)** | **1.95** | 2.62 / 2.04 | 1974 / 1071 of 2341 |

Every position now has a draft; most are short or wrong (acceptance 0.21-0.25 on chat, 1.65-1.89 tokens per pass),
but a pass costs about a decode step and returns picks instead of a vocabulary row, so a draft is nearly free and
each accepted token is a step saved. The measured gains exceed the simulator's estimates (pick chat40 1.61, pk 2.05),
which leave out the decode step's host sampler and logits copy.

Other settings (x plain decode; chat40 / copy20 / carry40):

| | pick_e1_L16 | pk_base_v1 |
|---|---|---|
| prompt lookup k = 4 | 1.131 / 1.968 / 1.388 | 1.129 / 4.239 / 1.179 |
| prompt lookup k = 15 | 1.128 / 2.159 / 1.393 | 1.140 / 7.080 / 1.184 |
| table k = 4 | 1.916 / 2.239 / 2.080 | 2.186 / 4.680 / 2.275 |
| table k = 15 | 1.902 / 2.473 / 2.089 | 2.221 / 8.199 / 2.295 |
| table k = 8, theta 0.1 | 1.839 / 2.422 / 2.071 | 2.195 / 6.469 / 2.247 |

**The drafter's own time** (`spec_draft_ms` / `spec_draft_calls`, which include building the history vector):
table runs 3.7-5.5 µs per decode position on pick_e1_L16 and 2.3-7.8 µs on pk_base_v1, against 0.5-1.6 µs for
prompt lookup alone; a decode step is 1.4 ms and 2.7 ms. Under 0.4% of a step.

**Deterministic mode** (`EMBER_DETERMINISTIC=1`, which selects the exact `rows` verification pass, n GEMV launches
per layer; x the mode's plain decode, chat40 / copy20 / carry40):

| | pick_e1_L16 | pk_base_v1 |
|---|---|---|
| prompt lookup k = 4 | 1.012 / 1.386 / 1.079 | 1.024 / 1.807 / 0.992 |
| prompt lookup k = 8 | 0.950 / 1.246 / 0.949 | 0.990 / 1.816 / 0.908 |
| table k = 4 | 0.939 / 1.310 / 1.065 | 0.805 / 1.741 / 0.870 |
| table k = 8 | 0.871 / 1.181 / 0.936 | 0.736 / 1.739 / 0.762 |

There a pass costs well over a decode step, and a drafter that drafts at every position with acceptance near 0.2
loses on chat. In the mode, prompt lookup at k = 4 remains the setting to use.

## Exactness

- **CUDA**: ids, stop ids and prompt ids equal plain decode's, **260/260 requests in every configuration** (both
  models; prompt lookup and table at k = 4, 8, 15; theta 0.1; and under `EMBER_DETERMINISTIC=1` at k = 4, 8;
  `e2e/*_vs_*.json`). Every recorded reply equals its warm pass's.
- **Bits under `EMBER_DETERMINISTIC=1`** (`bits/`, every sampled row's logits dumped and compared as float32 bit
  patterns against the mode's plain decode): table at k = 4 and 8, **3829/3829 rows on pick_e1_L16 and 4725/4725 on
  pk_base_v1, max |d| 0**.
- **CPU** (WSL, 4 threads, `receipts/cpu/`): table at k = 4 and 8 on both models, ids 260/260 against plain decode;
  draft, accept and pass counts equal the CUDA runs' and the simulator's. CPU speed is a shared laptop's and the CPU
  pass is the existing batched forward (pick 1.07x at k = 4, 0.95x at k = 8; pk 0.94x, 0.92x): no CPU claim.
- Plain decode: CUDA (this build) = CPU (HEAD 08ead73), 260/260 on both models; = verify_2026-10-27's L4 records
  (345d7f1), 60/60.

## With the switch off: unchanged

The drafter code runs only inside the speculative branch (`EMBER_SPEC` set) and, there, only with
`EMBER_DRAFT=table`; the new `/chat` fields appear only when speculation ran.

- **CPU receipts** (`../cuda_state_2026-10-25/cpu_regress.sh` with this build against the committed receipts,
  `cpu/cpu_legs.log`): recursion, MoR, MoR refusals, attnres, state4, regress_parity and G5 third party
  byte-identical. state_tokens_gate equals HEAD 08ead73's own CPU build's receipt byte for byte
  (`state_tokens_gate_cpu_head.json`, `_new.json`); both differ from the committed file in the nine numpy-reference
  distances only, as before. regress_bits (old side: the HEAD build's ir_logits): every result identical; the receipt
  differs from the committed one only in its `new` field, the binary's path.
- **IR adapter self-test**: 189/189 checks pass, 3072 requests.
- **Plain decode**: HEAD's CPU build against this one, ids 260/260 on both models.

## Files

- `../../draft_table.h` (the table and the drafter), `api.cpp` (the switch), `api.h` / `server.cpp` (the drafter's
  time), `docs/USAGE.md` (the switch table).
- `dr_table.py` (tokenise, build, read), `dr_tables.sh` (the two corpora and the two tables), `dr_sim.py` (the
  replay; `check` against a CUDA run, `sweep`, `vocab`), `dr_dev.sh` (the development sweep), `dr_prompts.py`,
  `dr_collect.sh` (CPU records), `dr_e2e.py` (runs, compare, bits), `dr_predict.py` (runs against the simulator),
  `dr_summary.py`, `dr_receipts.py` (these receipts from the full outputs).
- `build.sh`, `cpu_legs.sh`; `stage.sh`, `drive.sh`, `vm.sh`: the Colab run (microtorch/AGENTS.md path C; stopped,
  and gone from the listing). Full VM output: `C:/ml_artifacts/ember_draft/ember-dr-1027/`.
- `receipts/cuda/`: VM log, hardware, binary and model sha256, end-to-end records (gzipped; plain runs keep their
  ids, the others a sha256 per request) and meta, compares, bits, the summary, the simulator check;
  `receipts/cpu/`: CPU records the simulator replays (`collect/`, gzipped), CPU end-to-end records, the regression;
  `receipts/sim/`: the development sweep, the tokeniser logs; `receipts/tables.json`: every candidate table's and
  corpus's size and sha256.

Run (WSL): `bash build.sh cpu build_new`; `bash dr_collect.sh <HEAD CPU build> ~/dr/work`; `bash dr_tables.sh
~/dr/work`; `bash dr_dev.sh ~/dr/work <out>`; `bash cpu_legs.sh <build> <HEAD build> ~/dr/work <REF> ~/dr/work/tables
h_c8_p10.bin s_c8_p10.bin`; `bash stage.sh ~/dr/stage ~/dr/work/tables h_c8_p10.bin s_c8_p10.bin 0.02 && bash
drive.sh <session> 120 ~/dr/stage`; `python3 dr_summary.py <VM out>`; `python3 dr_predict.py ~/dr/work/collect <VM out>
~/dr/work/tables`.
