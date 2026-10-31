# Synthetic integrated selector correctness

Actual L4 V2 and independent landed-archive replay pass: 96 kernel cases, 30 graph legs / 240 complete model rows, 960 sampling ID/RNG comparisons, and 48 real HTTP fallback requests. Tested capture-key transitions and nonnull stable scratch pass. All 2,329 ZIP members pass CRC; exact source, protocol, fixture and binaries are bound in independent_audit.json and summary.json.

Scope is the deterministic synthetic two-layer F32 model. It is not the registered pick_e1_L16 model, a timing measurement, a resolution of historical default-mode instability, or an adoption pass. The previous failed speed pair and defaults remain unchanged. V1's pathname-override setup failure is preserved in integrated_gate_20261031/execution_v1/; V2 changes only that environment setup.

The compact public receipt contains tiny raw kernel records, all full-row hashes/capture fields, selected IDs/RNG hashes, raw fallback responses and source-contract/device logs. It does not contain the full float rows or full RNG strings. Those remain in the canonical local archive C:/ml_artifacts/e10_integrated_v2_20261031/e10_integrated_20261031/out.zip, SHA256 d7fa39d85bc8392a4197e90cba1a575fbaec8484ae789b579bb0d32ec034140c. The public compact packet alone cannot replay every raw comparison. audit_archive.py verifies the canonical archive; replay_kernel_receipt.py checks the public raw kernel subset.

Owned provider pod and volume were reclaimed after relay. V1 and V2 reported about USD0.12 and USD0.16 respectively, within the original combined USD1 ceiling. No production arithmetic, default, fixture, scientific threshold or speed gate was changed.
