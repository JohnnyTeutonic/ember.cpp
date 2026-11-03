# E10 PARTS4: actual-route instrumentation and frozen collector

Prepared for Codex's admission; **unadmitted**, with no GPU or provider call made. It answers the dispatch-evidence and collector requirements of the prospective [PARTS4 protocol](../parts4_full_model_protocol_20261102.json) (public at `41b6cd5`). The protocol's question, arms, row legs, timing triples, thresholds and stopping rules are unchanged. Nothing here enables PARTS4, compact picking or deterministic mode by default.

## Instrumentation (source re-pin)

`decode_route_trace.h` and trace-guarded insertions in `api.cpp`, `model.cpp`, `model.h` and `verify_kernels.{h,cu}` record the route of every sampled position when `EMBER_ROUTE_TRACE=<file>` is set.

`test_source_reduction.py` removes the guarded statements, the flags and the trace helpers, and obtains files byte-identical to the protocol's pinned `api.cpp` and `model.cpp` and to the other three files at `41b6cd5`. It accepts a guard only when the bare flag is a conjunct and the other conjuncts have no side effects; negated, disjunctive or side-effecting guards are refused, not deleted. This is a source-difference proof. With tracing off, the build still runs:

- the cached flag reads;
- the `trace_request` initialisation;
- the untaken branches' conditions.

The model also carries route-state members. All arms run this one binary, so the gate makes no comparison with an uninstrumented build.

The selector kernels (`row_pick_partitioned.cuh`), the policy header and `api.cpp`'s dispatch fragment are unchanged. The 12 existing selector source contracts (`tools/row_pick_profile/test_integration.py`) pass on the patched tree.

Tracing changes no arithmetic. Before each replay of a graph that holds a selector, it fills the record and the four partition records with an all-`0xFF` sentinel the selector cannot write (its NaN flag is 0 or 1), then reads them back with the row the same replay produced. If merge ran before the partitions, it would read the sentinel and its record would fail the oracle, so the per-replay checks also cover kernel order.

| Protocol minimum inventory | Evidence |
|---|---|
| Prefill versus eligible sampled decode positions | Each request has its prefill token plus one event per decode position, consecutive from the prompt length. Together these must equal the client's ids, for every request in the gate and in the matrix |
| Requested arm, effective partitions, capture/replay, per-launch fallback | The request event reports the server's routing environment. The token event reports `graph` (`replay`, `off`, `disabled`, `capture_failed` or `unavailable`) and the replayed graph's `graph_has_pick` and `graph_parts`. Capture ids must rise by one at each capture and replay indices by one at each position. A captured graph must hold exactly the arm's selector nodes, with their captured arguments: vocabulary, `parts=4`, `count=4`, grids, and the decode step's own buffers. Across arms, graphs differ only by those nodes (host + 1, host + 2), whatever cuBLAS nodes the runtime cannot name. Route `per_launch_fallback` marks the per-launch path |
| API exclusions: IR, CPU offload, non-greedy, repetition penalties | A bit mask mirrors `want_device_pick` (`exclusions`, `eligibility_consistent`). Top-k and the repeat penalty run in the route matrix, each request checked against the configuration sent. IR and CPU offload are named by the mask and unit-tested, not executed: every traced registered request must report `ir: false, cpu_offload_layers: 0` |
| Compact pick versus host row after a near tie, nonfinite record or debug rows | The checker recomputes each decision from the serialized record bits in float32 and requires the reported decision and route to follow from them; a silent host route cannot pass. A non-forced pick leg must consume at least one compact record. `det_route_parts4_rows` records the intended debug-row fallback. `det_route_parts4_forced` tests the plumbing only: it hands the API an undecidable copy of every record, so the host samples each row, and the true record's decision must equal the host's token wherever it decides. Real near-tie, NaN and infinity records are the retained handcrafted kernel cases; natural near ties are counted |
| Selector execution on graph replay, not only capture | At each replay: the sentinel is overwritten, the record equals the whole-row oracle, and for PARTS4 each partition record equals its partition's oracle and the four merge to the record. These checks are made in C++ on every route-leg replay. For the dumping PARTS4 leg they are also recomputed independently in NumPy from the dumped rows |
| Route and timing tied to one source, input and configuration | One binary and one plan. Every leg's environment and command must be exactly canonical, so no appended override or duplicate gets through. Route legs differ from timing legs only by the trace variables. Ids must match across all 17 legs, and the traced PARTS4 dump must reproduce the untraced PARTS4 row bits. The matrix must carry the gate's binary and model hashes. Trace-on throughput is overhead disclosure, never a ratio input |

## Collector

- **`parts4_gate.py plan|analyze`.** Seventeen legs in this order:
  - five untimed route legs: one per arm, PARTS4 forced, PARTS4 with dumps;
  - the protocol's six row legs;
  - its timing triples `[host_a, parts0_a, parts4_a]` and `[parts4_b, parts0_b, host_b]`.

  The client (`dr_e2e.py`) and every leg predicate (`decode_pick_gate.load_run`, `ids`, `speed`) are the registered ones, imported unchanged and hash-bound.

  Analysis keeps two tracks. A candidate failure demonstrated against valid controls is `CORRECTNESS_FAIL`, and later missing evidence never downgrades it. Such failures are:
  - PARTS4 row bits, ids, row inventory or finiteness;
  - a written PARTS4 record that disagrees with its row;
  - a PARTS4 matrix response that differs from host.

  Otherwise, any operational, control, plan or route-evidence failure gives `INCONCLUSIVE`, which is ineligible for adoption. Next comes `PERFORMANCE_MISS`: each pair is judged alone, against 1.15 versus host and 1.0 versus PARTS0. Otherwise `PASS`, which grants deterministic-greedy, model-specific opt-in eligibility only. An archived output replays wherever it is unpacked; `--relocate` names the inputs' new homes and the binary's recorded hash.
- **`parts4_route_matrix.py`.** It uses `decode_pick_fallback.py`'s prompts under five configurations: registered greedy, repeat penalty, top-k 4, and temperatures 0 and 0.049. It runs three arms with the graph on and off, all traced. Every request must match the configuration sent, with a complete trace and capture continuity, and every position's route must be the one its configuration requires. Responses must agree across arms in each graph mode.
- **`parts4_vm.py`.** The supervised payload, following `ordered_vm.py`'s procedure:
  - a byte-checked manifest, a cleaned environment, one L4, and a 9,600 s deadline;
  - every command runs in its own process group, killed whole on timeout or signal;
  - a zip checkpoint every two minutes, except while timing legs run;
  - no retry for a pass.

  On device it builds, then runs the CPU contracts, the retained handcrafted records and `decode_pick_fallback.py` under PARTS0 and PARTS4, the matrix, and the 17 legs, each verified at once by `load_run`. It ends with a sealed `evidence_sha256.json`. All legs write to one local directory.
- **`stage_upload.py`.** Stages the upload from a commit. Using the staged payload's own helpers, it repeats every pre-device check offline, including the 12 selector contracts. Model and prompts are hard-linked or symlinked to their canonical files, never copied; transport must send the resolved regular files, as the V3 transport does.
- **`make_packet.py`.** Writes `packet.json`: the source, collector and input pins, the schedule and the test count.

## Commands

```
cd experiments/spec_r1_2026-10-28/parts4_route_20261102
python3 -m pytest -q test_source_reduction.py test_parts4_gate.py
g++ -std=c++17 -O1 -I ../../.. test_route_trace.cpp -o /tmp/test_route_trace && /tmp/test_route_trace
python3 make_packet.py --check
python3 stage_upload.py --commit <pin> --model <model.gguf> --prompts <eval100.jsonl> --out <upload>
python3 parts4_vm.py --upload <upload> --manifest-sha256 <sha> --out <out>        # admitted VM only
python3 parts4_gate.py analyze <out>/science/gate --matrix <out>/science/matrix \
    --relocate model=<model.gguf> --relocate prompts=<eval100.jsonl> --relocate binary=sha256:<hex>
```

## Review evidence

Codex (GPT; `scripts/codex_review.py`, methods-audit preset with a packet-specific brief) reviewed the first version (`2d89f2d`) on 2 Nov 2026 and advised against admission. Its four blocking findings are resolved here:

- correctness failures could be downgraded to inconclusive;
- the matrix could pass with positions missing;
- the matrix was not bound to the gate's binary and model;
- appended environment overrides were accepted.

Its should-fix findings are also resolved:

- guards are restricted, and the reduction wording is narrowed;
- capture continuity, kernel arguments and cross-arm node counts are now checked;
- decisions are recomputed from the record, and a compact minimum is enforced;
- the forced leg is labelled, with per-position flags;
- a candidate row-count change is now a correctness failure;
- commands run in process groups;
- the preflight repeats the selector contracts;
- no checkpoints run during timing;
- the evidence inventory is sealed.

Its notes on test independence are answered by the dumped-row NumPy oracle and the real-row fixtures, and the C++ property test now generates `+inf`.

Round 2 (`80560d9`) advised against admission again, with three blocking findings:

- a missing sibling leg or a malformed later event could still mask a demonstrated candidate failure;
- demonstrated-wrong PARTS4 records in the matrix were only evidence failures;
- the executed client was checked by basename only.

All three are resolved:

- controls are established first, then each candidate leg is compared separately;
- demonstrated failures are committed as they are found;
- the matrix has a candidate track;
- the recorded client's bytes must hash to the frozen client.

Its should-fix findings are also resolved:

- coverage is per request, with true immediate stops classified;
- the matrix records the request body sent, and the trace records the penalty, window and token limit the server applied;
- group survivors are checked independently of the leader;
- the quiet flag is set before the last pre-timing pack and rechecked under the lock;
- guards must be one of five exact audited forms.

Round 3 (`b35319f`) resolved all of those but one partly. It found that the matrix compared the server's float32 temperature (0.049 serializes as 0.04899999871850014) with a Python double, which would have made every real low-temperature request fail. It also found that a malformed trace tail or an unrelated missing matrix receipt could still hide a demonstrated failure, and that an aborted collection skipped analysis.

All are resolved:

- the comparison uses float32, and the fixtures carry the serialized value;
- traces are read incrementally, so every complete request before a bad tail is checked;
- each matrix leg loads on its own;
- an aborted payload writes `gate_summary_partial.json`.

The seed is not traced, because the session does not store it; the matrix's recorded request bodies carry it.

Round 4 (`f2a77e6`) confirmed those fixes. One case remained: a complete warm request whose recorded partner was truncated went unchecked. Every complete request is now checked on its own, with request `j` belonging to record `j // 2`. A first round-4 run had reviewed a stale bundle (`b35319f`); it is kept, marked as such, and was not used. Round 5 (`dd6dade`) found it resolved and no new blocking or should-fix problem, and its verdict was ADMIT.

The raw reviews are kept outside the stack repositories, in the workspace's `AI_ML/e10_parts4_claude/`.

## Not established here

The instrumentation's device behaviour is untested until admission. CPU evidence:

- the reduction proof;
- the header's C++ test, 9,113 checks;
- the checker's tests (`packet_test_count` in `packet.json`);
- the 12 selector contracts;
- a CPU end-to-end trace through the registered client, aligning 8 records (one a five-turn conversation) with 16 traced requests.

The protocol's `not_established` list applies unchanged: sampler RNG-state equivalence, other models, nondeterministic-mode PARTS4, and broad quality or speed claims.
