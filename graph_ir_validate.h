#pragma once
// Load-time validation of an IR document (paperkiln spec/IR_SPEC.md section 10.1, the rules its
// tools/ir_validate.py checks), so ember refuses at load what every consumer must refuse, naming the
// rule. Messages read "IR_SPEC <section>: <what>", as the validator's do.
//
// Document (ir_document_errors):
//   rule 1      the JSON Schema (graph_ir_schema.h, vendored from paperkiln's spec/ir.schema.json):
//               types, required keys and the closed key sets of the document, every op's node and every
//               slot (an attribute an op does not define is refused by name; keys starting with "_" are
//               prose, allowed anywhere); with it rules 4, 5 and 8
//   rule 2      ir_version 0 or 1
//   rules 3, 6-10, 14   model fields and block.repeat = model.n_layers; per-layer scope and length; arity;
//               expr parses; split, rmsnorm, rope, attention, moe_ffn and conv1d_causal values; causal true
//   rules 11-13 slots: kinds, declared once, each read, the stateful node's slot of its kind, attention
//               against its kv slot (window, sinks, kv_heads, head_dim, per application), recurrent slots
//   rules 15-17a  turn_state, evict, state_weight; {i} once and only in the block; tokens and pos; one
//               attention node per kv slot; nothing reads depth_route's out
//   rules 18-23 weight_index, tensor_index placement, depth_route, kv_rows active, kv_from, depth_route
//               with turn_state or list values
// Container (ir_container_errors, the GGUF the document loads from):
//   rule 24     paperkiln.ir_version: a uint32 equal to the file's embedded document's ir_version
//   rule 26     the depth_route router's ne {d, n_rec}
//   rule 27     weight_index: architecture paperkiln, <arch>.block_count = the stored layers
//   rule 28     state tokens: paperkiln.state_tokens.k with an embedding state_weight and the reverse,
//               state_weight ne {d, k}, architecture paperkiln, tokenizer.specials naming paperkiln/1's
//               markers, no chat template but paperkiln/1
// Rule 25 (every tensor named is in the file) is the Interpreter's binding; a missing S = 1 depth_attend
// pseudo-query is accepted (section 5.18). Rule 29 (sidecars) does not apply: ember reads no sidecar.
//
// The Interpreter calls validate_ir after its own load checks (so a document those refuse keeps their
// message), then refuses what ember does not implement among valid documents (section 10.3, Appendix C:
// more than one kv slot per block).
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

struct GGUFData;

namespace ember_ir {

// Rule 1: the schema's failures, one message each.
std::vector<std::string> ir_schema_errors(const nlohmann::json& doc);

// Every violation of the document rules above, in the validator's order, duplicates removed.
std::vector<std::string> ir_document_errors(const nlohmann::json& doc);

// The container rules above for doc loaded from gguf (doc replaces the file's own under EMBER_IR; rule 24
// is about the file's own).
std::vector<std::string> ir_container_errors(const nlohmann::json& doc, const GGUFData& gguf);

// Throws std::runtime_error listing every violation of both ("; " between them) when there is one.
void validate_ir(const nlohmann::json& doc, const GGUFData& gguf);

}  // namespace ember_ir
