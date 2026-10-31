# Ordered default reductions: verified registered result

The candidate routes tuned CUDA RMSNorm, standalone vector softmax and both legacy attention fallbacks through ordered reductions by default. This does not enable `EMBER_DETERMINISTIC=1`: explicit deterministic mode still additionally controls cuBLAS workspace/atomics and verification policy. Atomic predecessors remain only in the pinned benchmark baseline; no unmeasured production atomic opt-in is installed. The single-block vector reductions have release-active launch-geometry guards. Compact selection remains opt-in.

Independent offline replay verified the complete candidate 14-leg gate on one L4, including 3,829/3,829 identical default off/off-repeat rows, exact off/on rows and IDs, and both unchanged 1.15 throughput pairs in each mode. All 24 total legs are home. The baseline's ten controls remain a partial comparison, not a full baseline adoption result. See the [independent summary](receipts/actual_l4_v3/actual_v3_independent_summary.json), [storage/timing audit](receipts/actual_l4_v3/actual_v3_storage_replay_v2.json) and [numerical base audit](receipts/actual_l4_v3/actual_v3_storage_replay_v2.json.numerical_base.json).

| Candidate mode | First compact/off throughput pair | Second pair | Original row/ID and 1.15 gate |
| --- | ---: | ---: | --- |
| Default, explicit deterministic mode off | 1.15658139 | 1.15895616 | PASS |
| Explicit deterministic mode | 1.24335201 | 1.19765622 | PASS |

**This gate pass does not show an absolute old-to-new throughput improvement.** The ratios above compare compact selection on/off within the new candidate. The candidate's off path was slower than the matched old off controls, while its on path was similar or slightly slower. Consequently the larger relative on/off gains partly reflect a slower off path. Descriptive new/old throughput ratios, recomputed from all sixteen timing legs, are:

| Mode / selector | Old pair A tok/s | New pair A tok/s | New/old A | Old pair B tok/s | New pair B tok/s | New/old B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default / off | 819.29 | 794.53 | 0.96977 | 824.33 | 783.57 | 0.95055 |
| default / on | 922.94 | 918.93 | 0.99566 | 925.71 | 908.12 | 0.98099 |
| det / off | 803.04 | 734.21 | 0.91429 | 813.55 | 764.87 | 0.94016 |
| det / on | 913.24 | 912.88 | 0.99960 | 916.58 | 916.05 | 0.99943 |

A value below one means lower new-build throughput. These matched observations retain the original interleaving and wall-minus-TTFT estimator; they do not isolate a unique causal kernel, storage or scheduling mechanism and are not a new acceptance gate. The ordered default is supported here as a repeatability/correctness change. No automatic selector adoption or general speedup is claimed.

## Frozen sources and requested comparison

The benchmark baseline is `1b2789e656dda996efb79490cf689d0eff45eefe`. Production routing was implemented in `326c0986cd2b150ee1dd2eccea94be39adf25906`, followed only by wrapper-formatting cleanup in `bca00166f75d53a01094e5ebfa599a004062a21e`. The frozen scientific packet is `d83d00e96254be214f0b32227f1d3fea8181d148`. Later transport regression documentation does not replace that archive. See [protocol.json](protocol.json), [source preflight](source_preflight.json) and [packet compile-only preflight](packet_preflight.json).

The registered model is `pick_e1_L16` (d320, 16 layers, vocabulary 16,384, context 512), SHA-256 `96f6afdcbbcf1a4cf53ad6c0e451e75301dc4187f726f3c2218600fbe72a8cc9`. Frozen `eval100.jsonl`, SHA-256 `a25357100c3eb1e59bc94bb26c0629e72e6ae18172b2d4fbbf148ae5168d1740`, contains 100 items expanding to 260 requests: 40 chat items, 20 copy items and 40 five-turn conversations.

The candidate runs the complete unchanged 14-leg gate: default and explicit deterministic modes, each with `rows_off`, `rows_off_repeat`, `rows_on`, followed by the dump-free `off_a`, `on_a`, `on_b`, `off_b` quartet. The baseline has ten declared controls: default off/off-repeat dumps and both four-leg timing quartets. It is **not** a full baseline adoption result. Source interleaving preserves each quartet's original ABBA order and pairs baseline/candidate measurements on one L4.

`EMBER_DECODE_PICK_PARTS=0` is explicit throughout the registered comparison. The experimental four-part selector is outside this packet. Inherited Ember routing flags, `EMBER_IR` pathname override and cuBLAS workspace settings are cleared before applying each frozen leg's settings. Default health must not report deterministic mode enabled. The frozen checker retains exact complete row bits, IDs, replies, stops, prompt IDs, token counts, stable warm passes and no-speculation requirements. Both individual throughput pairs must reach 1.15; pooling cannot rescue a failed pair. Ordering changes need not reproduce historical unordered bits across binaries: the candidate's within-source controls remain strict.

## Separate component and release-guard evidence

The component probe calls actual production RMSNorm and standalone softmax wrappers, plus both explicitly selected legacy attention wrappers. Binary64 CPU references and finite-output checks use a separate 1e-5 absolute semantic bound; this does not relax the registered exact-row gate. Raw inputs, weights or Q/K/V, first output rows, references, repeated-bit checks and timing batches are retained. Component old/new timings use source ABBA order. Their scope is resident component execution, not full-model acceleration.

Six isolated processes launch the actual guarded RMSNorm or standalone softmax kernel with an invalid grid axis under `NDEBUG`. Acceptance requires valid allocation, the named kernel/stage and the expected device trap, rather than an arbitrary nonzero process status. Compile-only SASS and CPU geometry tests are supporting evidence; they are not substitutes for actual device execution. Independent replay verified 1,584 component case evaluations, maximum absolute reference error 1.9106984e-7, 660 candidate cross-process first-row comparisons and all six release-guard probes. The twenty-repeat checks are recorded executable assertions: all twenty outputs were not retained, so their complete sequence was not independently rechecked.

## Storage-only operational amendment

The first baseline dump was slow on network storage. A separately reviewed amendment left that active leg untouched and placed 23 future per-request `.run` directories and seven full-row files on owned local `/tmp` scratch through absent-only links. All 16 dump-free timing legs use the same local-storage policy. The baseline's first off dump and its local off-repeat therefore have mixed physical storage; their partial-control comparison must disclose this limitation. Candidate legs use uniform local placement. No numerical code, requests, environments, thresholds or completed scientific draws were changed or repeated.

The packer follows the full `.f32` file links, retaining their complete bytes under canonical archive member names. It does not descend linked per-request `.run` directories, so their ephemeral last-request `dump.txt`/debugging/www subtree is omitted. Full row arrays and the retained request/metadata/server logs are distinct from that omitted ephemeral subtree. The reviewed archive behavior and its limits are recorded in `/home/jonat/review_receipts/ordered_default_20261101/storage_amendment_review.json`; the operational amendment must accompany the final raw evidence.

## Historical negatives and setup failures remain intact

The earlier registered default off/off repeat matched only 101 of 3,829 complete rows bit for bit, although every recorded top-1 ID matched. Its `BASELINE_UNSTABLE` verdict remains unchanged. The recorded deterministic throughput pairs were 1.15924656 and 1.14146241: the second missed 1.15, so that historical adoption gate did not pass. See the [source-bound historical audit](../spec_r1_2026-10-28/receipts/default_repeatability_20261101.json) and [historical final receipt](../spec_r1_2026-10-28/receipts/decode_pick_20261030_final.json). Unordered atomic reductions provide an order-sensitive mechanism; those receipts do not uniquely attribute every model-row difference to one kernel. A new result cannot retroactively change them.

Two later setup attempts ended before any build or scientific test. V1's nested diagnostic uploads were not reassembled by the runner's top-level enumeration. V2 used flat names, but rsync skipped a symlinked baseline archive with return code zero; the error receipt identified `/content/chat/upload/baseline.tgz`. V3 resolves every upload source to a real regular file before frozen transport argv. V1 and V2 remain negative operational receipts, not numerical failures or successful gates. See [transport repair history](TRANSPORT_REPAIRS.md), [V1 repair record](transport_v2_repair.json), [actual runner regression](test_transport_v2.py) and [real rsync regression receipt](transport_v3_rsync_receipt.json).

## Raw evidence and replay

The canonical full archive is `C:/ml_artifacts/ember_ordered_default_20261101_v3/ordered_default_gate/out.zip`. It is home: 2,072,950,310 bytes, SHA-256 `6b64284e674fec5cb9a371fe792a6686f7809a57a9b4836fc3e77235525b2e67`. Independent replay verified all 170 member hashes/CRCs. The [compact public receipt](receipts/actual_l4_v3/README.md) binds this local archive; complete float arrays remain in that one canonical home. Independent replay runs the unchanged `experiments/spec_r1_2026-10-28/decode_pick_gate.py` analyzer through a ZIP filesystem adapter over the complete archived raw data, without duplicating the raw tree. Model/prompt bytes are hash-bound; executed binary identities are checked against captured build/plan metadata. Executable binary bytes are not retained in the ZIP. Baseline ten-control summaries remain explicitly partial.

The compact publication helper `package_receipt.py` runs only after the canonical ZIP is HOME and its expected SHA-256 is supplied. It streams every member for hash/CRC binding and extracts an explicit small-file allowlist, with component arrays removed from derived kernel summaries. This is packaging, not a numerical gate. Full raw arrays and the registered model/evaluation inputs remain necessary for full replay; a compact public receipt alone cannot reproduce every comparison. Account/admission snapshots, models, corpora and large float arrays are not extracted. A separately authored scientific audit may be supplied unchanged with `--audit`.

The source and transport tests run with pytest. The frozen payload CLI is `python3 ordered_vm.py --upload DIR --manifest-sha256 SHA --out OUT`. Upload/source/binary identities, raw complete row arrays, component records, metadata, negative outcomes and operational amendments must remain available for independent replay. Server/probe executable identities are hash metadata; executable bytes are not in the raw ZIP. Independent replay recomputes recorded raw comparisons rather than executing the model again. Frozen HTTP requests retain seed and IDs, not complete sampler RNG state; this packet does not establish full RNG-state equivalence, all-model determinism or general engine superiority.

## Verifier provenance and cleanup

The first offline storage verifier reported `STORAGE_AUDIT_ERROR_OR_INCOMPLETE` because it expected the amendment body alone, whereas transport had archived its exact `ENV_PREAMBLE` followed by that body. That [original failed verifier receipt](receipts/actual_l4_v3/actual_v3_storage_replay.json) and both original verifier sources are preserved. A separate corrected wrapper proves the complete byte-exact prefix/body decomposition from the pinned transport; it does not strip or ignore bytes. Thirty fixtures passed across the original and corrected wrapper suites. The immutable numerical helper and scientific acceptance rules were unchanged; no GPU leg was rerun and no raw receipt was replaced. This was an offline verifier binding failure, not a numerical failure.

The [independent cleanup receipt](receipts/actual_l4_v3/final_cleanup.json) verifies owned pod/volume absence, completed relay, disabled watchdog entry and exited runner/guard, with unrelated protected volumes retained. No balance or launch/account snapshots are included in the public package. This task introduced no E9 changes.
