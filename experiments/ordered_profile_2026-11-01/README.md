# Ordered-default timing diagnostic

The ordered candidate is descriptively near baseline on this small fixed subset, with no established speed improvement or isolated causal explanation. Independent raw replay verifies all 80 L4 legs and all 24 primary pairs; every old/new pair has identical generated token paths.

| Mode | Selector | Paired new/old throughput median | Range | Matched pairs |
|---|---|---:|---:|---:|
| Default | Off | 0.974925 | 0.964920-1.031229 | 6/6 |
| Default | On | 0.995178 | 0.983276-1.007208 | 6/6 |
| Deterministic | Off | 0.996561 | 0.969191-1.108397 | 6/6 |
| Deterministic | On | 1.000582 | 0.980124-1.010797 | 6/6 |

Ratios summarize the six registered adjacent ABBA comparisons per arm, not pooled marginal throughput. All pairs and variability remain in the [independent audit](receipts/actual_l4_20261101/independent_actual_audit.json). The default selector-off median is about 2.5% lower here; deterministic selector-off varies across parity. This eight-item/sixteen-request diagnostic neither replaces nor overturns the earlier full 260-request result. The historical default repeat failure (101/3829 equal rows) and deterministic 1.159/1.141 speed pairs remain preserved, as does the later ordered-default correctness/adoption-gate receipt that did not establish absolute old/new improvement. See [the full registered result and qualifications](../ordered_reductions_2026-11-01/README.md).

## Frozen design and evidence

Eight fixed eval100 items (indices 0,1,2,3,40,41,60,61: four chat, two copy, two five-turn) expand to sixteen requests. Primary graph-on, timing-unset source ABBA runs repeat three times per default/deterministic mode and selector off/on: 48 legs. Sixteen separate graph-on/timer-on legs and sixteen graph-off timer ABBA controls complete the 80-leg schedule. The graph-off controls comprise eight timed and eight untimed legs. All use PARTS=0, the frozen client, stable warm passes, no speculative decoding and the original wall-minus-first-token estimator. No performance threshold or automatic default change is attached to this diagnostic.

The separate graph-off timer-on/off median throughput ratios are approximately 0.813/0.813 for default old/new and 0.726/0.661 for deterministic old/new. Instrumentation changes performance materially in those controls. Its per-phase numbers are synchronized walltime, include warm and measured requests, and are not pure GPU event times. Graph capture need not expose every kernel subphase. These 32 diagnostic legs are excluded from primary estimates, and all their matched-work flags are retained.

Common one-hertz telemetry supplies 189 samples from one L4 UUID, covering every leg. SM clock is predominantly 2040 MHz (startup minimum 210 MHz), temperature spans 36-53 C, and sampled utilization spans 0-94%. Neither these measurements nor the phase timings identify a unique kernel, clock, thermal or control-path cause. No statistical significance, all-model speed, quality, or new adoption claim follows. Native startup INFO logs are suppressed in these serverlogs; actual-device evidence is bound to the CUDA source/build/binary, L4 hardware and telemetry, not invented health fields.

The [compact receipt](receipts/actual_l4_20261101/compact_manifest.json) includes source inventories, binary/input hashes, telemetry, the frozen replay result, all primary ratios and raw-entry hash inventory. The complete raw ZIP remains in its single canonical local home:

```
/mnt/c/ml_artifacts/ember_ordered_profile_1101/ordered_profile/out.zip
SHA256 b8b3cb4fa8b4a18df7c60143a19a18663e4654b09fd99c4b2eb40cea7379fdcc
```

All 340 ZIP entries pass CRC. The compact public receipt omits full per-leg raw rows/logs and duplicate summaries, so it alone cannot reproduce every comparison. Complete raw evidence, the frozen stage/source archives and original canonical model/eval inputs are required for the full audit; no model, corpus, binary or raw ZIP is duplicated here.

After safe extraction of the canonical ZIP, replay with:

```
python experiments/ordered_profile_2026-11-01/replay_profile.py /path/to/extracted/profile
```

For the complete source/archive/metadata/telemetry audit, run [the independent audit helper](receipts/actual_l4_20261101/audit_ordered_profile_actual.py) under the workspace CPU-budget wrapper with `--canonical-job`, `--stage`, `--source`, `--python` and a fresh `--out`. It refuses existing output and performs no provider, configuration, driver or GPU action.

The baseline is `1b2789e656dda996efb79490cf689d0eff45eefe`; the candidate packet is `0e364d1a5ac1c393f0c991af0b1fc3dc4cac9946`, whose production bytes match public ordered-default `db3edc5`. Full inventories, source archive hashes and manifest `368d19cb0778b6acf97fdeec69f7b0271d913ee89d655282d8cef6866e3a8d20` are retained. Production arithmetic/routing, selector default, historical gates and Claude's E9 work are unchanged.
