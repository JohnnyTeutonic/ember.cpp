# E10 PARTS4: actual-route instrumentation and frozen collector

Prepared for Codex's admission; **unadmitted**, with no GPU or provider call made. It answers the dispatch-evidence and collector requirements of the prospective [PARTS4 protocol](../parts4_full_model_protocol_20261102.json) (public at `41b6cd5`). The protocol's question, arms, row legs, timing triples, thresholds and stopping rules are unchanged. Nothing here enables PARTS4, compact picking or deterministic mode by default.

## Instrumentation (source re-pin)

`decode_route_trace.h` and trace-only additions to `api.cpp`, `model.cpp`, `model.h` and `verify_kernels.{h,cu}` record the route of every sampled position when `EMBER_ROUTE_TRACE=<file>` is set. Unset, every added statement sits behind that cached flag. `test_source_reduction.py` removes the trace statements, flags and helpers, and obtains files byte-identical to the protocol's pinned `api.cpp` and `model.cpp` and to `41b6cd5`'s other three. The selector kernels (`row_pick_partitioned.cuh`), the policy header and `api.cpp`'s dispatch fragment are unchanged; the existing 12 selector source contracts (`tools/row_pick_profile/test_integration.py`) pass on the patched tree.

Tracing changes no arithmetic. Before each replay of a graph that holds a selector, it fills the record and the four partition records with an all-`0xFF` sentinel the selector cannot write (its NaN flag is 0 or 1). After the replay it reads them back with the row the same replay produced.

| Protocol minimum inventory | Evidence |
|---|---|
| Prefill versus eligible sampled decode positions | Request event: `prefill`, `prefill_sampled`, `start_pos`. One token event per decode position, consecutive from the prompt length; the checker composes them into the client's ids |
| Requested arm, effective partitions, capture/replay, per-launch fallback | Request event: the server's routing environment. Token event: `graph` (`replay`, `off`, `disabled`, `capture_failed`, `unavailable`), `captured`, `capture_id`, `replay_index`, the replayed graph's `graph_has_pick` and `graph_parts`, and route `per_launch_fallback` |
| API exclusions: IR, CPU offload, non-greedy, repetition penalties | A bit mask mirroring `want_device_pick` (`exclusions`, `eligibility_consistent`). Top-k and the repeat penalty run in the route matrix. IR and CPU offload are named by the mask and unit-tested, not executed: the registered requests are all-GPU and non-IR, and every traced request reports `ir: false, cpu_offload_layers: 0` |
| Compact pick versus host row after a near tie, nonfinite record or debug rows | Route `compact`, `host_after_near_tie`, `host_after_nonfinite` or `host_after_debug_rows`, and `picked_on_device`. The `det_route_parts4_rows` leg makes the intended debug-row host fallback explicit. The `det_route_parts4_forced` leg hands the API an undecidable copy of every record, so it takes the near-tie/nonfinite branch at every position; the trace keeps the true record, whose decision must equal the host sampler's token wherever it decides |
| Selector execution on graph replay, not only capture | At capture: a graph inventory naming the selector kernel nodes (`select_rows_kernel`; or `partition_rows_kernel` at grid (1,4,1) plus `merge_parts`). At every replay: sentinel overwritten, the record equal to the whole-row oracle, and for PARTS4 all four partition records equal to their partitions' oracles and merging to the record |
| Route and timing tied to one source, input and configuration | One binary and one plan. Route legs differ from timing legs only by the trace variables. Ids must match across all 17 legs, and the traced PARTS4 dump must reproduce the untraced PARTS4 row bits. Trace-on throughput is reported as overhead disclosure, never as a ratio input |

## Collector

- `parts4_gate.py plan|analyze`. Seventeen legs in this order:
  - five untimed route legs: one per arm, PARTS4 forced, PARTS4 with dumps;
  - the protocol's six row legs;
  - its timing triples `[host_a, parts0_a, parts4_a]` and `[parts4_b, parts0_b, host_b]`.

  The client (`dr_e2e.py`) and every leg predicate (`decode_pick_gate.load_run`, `ids`, `speed`) are the registered ones, imported unchanged and hash-bound. Verdicts:
  - `PASS`: deterministic-greedy, model-specific opt-in eligibility only;
  - `PERFORMANCE_MISS`: each pair is judged alone, against 1.15 versus host and 1.0 versus PARTS0;
  - `CORRECTNESS_FAIL`: PARTS4 rows, ids or finiteness differ while the controls are valid;
  - `INCONCLUSIVE`: invalid inputs, plan or controls, or incomplete route evidence, which is also ineligible for adoption.
- `parts4_route_matrix.py`: `decode_pick_fallback.py`'s prompts under five configurations (registered greedy, repeat penalty, top-k 4, temperatures 0 and 0.049), for three arms with the graph on and off, all traced. Ids must agree across arms in each graph mode, and every position's route must be the one its configuration requires.
- `parts4_vm.py`: the supervised payload, following `ordered_vm.py`'s procedure:
  - manifest checked byte for byte, cleaned environment, one L4, a 9,600 s deadline, a zip checkpoint every two minutes, no retry for a pass;
  - on device it builds, then runs the CPU contracts, the retained handcrafted records and `decode_pick_fallback.py` under PARTS0 and PARTS4, the matrix, and the 17 legs, each verified at once by `load_run`;
  - all legs write to one local output directory, so storage is uniform across the timing arms.
- `stage_upload.py`: stages the upload from a commit and repeats every pre-device check offline. The model and prompts are hard-linked, not copied.
- `make_packet.py`: writes `packet.json` (source, collector and input pins; schedule; test count).

## Commands

```
cd experiments/spec_r1_2026-10-28/parts4_route_20261102
python3 -m pytest -q test_source_reduction.py test_parts4_gate.py
g++ -std=c++17 -O1 -I ../../.. test_route_trace.cpp -o /tmp/test_route_trace && /tmp/test_route_trace
python3 make_packet.py --check
python3 stage_upload.py --commit <pin> --model <model.gguf> --prompts <eval100.jsonl> --out <upload>
python3 parts4_vm.py --upload <upload> --manifest-sha256 <sha> --out <out>        # admitted VM only
python3 parts4_gate.py analyze <out>/science/gate --matrix <out>/science/matrix   # offline replay
```

## Not established here

Device behaviour of the instrumentation is untested until admission; CPU evidence is the reduction proof, the header's C++ test (9,113 checks), 48 checker tests and a CPU end-to-end trace through the registered client. A trace-disabled instrumented binary is not timed against an uninstrumented one: the reduction proof shows that only untaken trace-flag branches were added. The protocol's `not_established` list applies unchanged: sampler RNG-state equivalence, other models, nondeterministic-mode PARTS4 and broad quality or speed claims.
