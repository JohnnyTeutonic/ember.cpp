# S-full local-chat runtime preparation, 3 November 2026

The native Ember server builds successfully in the existing Release configuration with HAS_CUDA=OFF. This preparation uses no local GPU or provider compute. The existing state-token template suite passes. Four explicit k=4 smoke cases additionally check completed exchanges, interrupted replies followed by a new user turn, already-inserted blocks without duplication, and rejection of an orphan state token. Inputs and outcomes are preserved alongside the source/binary binding.

The registered final SFT GGUF is not yet home. No S-full model load, numerical parity, reply quality or successful chat is claimed. Once recovery brings the export home, verify its identity, tokenizer/template and four-state-token metadata; load it with the CPU-only server; check health and multi-turn replies; then expose the existing localhost browser UI. The prepared positional command is in readiness.json and explicitly selects zero GPU layers. Revalidate/rebuild source references after the authorised history purge.

A preliminary insert-mode invocation omitted its required JSON stdin and was refused. The corrected explicit fixtures above passed; this was a harness invocation error, not a model result. Chat-schema-v2 fixture tests were built but not run because no appropriate v2 fixture was supplied; they are not counted as passing.
