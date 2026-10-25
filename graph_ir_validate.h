#pragma once
// Load-time validation of an IR document (paperkiln spec/IR_SPEC.md section 10.1, the rules its
// tools/ir_validate.py checks), so ember refuses at load what every consumer must refuse, naming the
// rule. Messages read "IR_SPEC <section>: <what>", or "IR_SPEC <section> (rule <n>): <what>" where they cite
// the rule (the schema's failures, rules 35 to 37, the recurrence rules), as the validator's do.
//
// Document (ir_document_errors; revision 1.2, keramos v1.2.0):
//   rule 35     no object of the text holds a member name twice, at any depth, "_" members included
//               (parse_ir_document, the parse every load uses)
//   rule 1      the JSON Schema (graph_ir_schema.h, vendored from keramos ir.schema.json): types, required keys
//               and the closed key sets of the document, every op's node and every slot (an attribute an op
//               does not define is refused by name; keys starting with "_" are prose, allowed anywhere). A
//               schema failure cites the rule 10.1 files it under: rule 1 only for what no other rule names
//               (the version rule 2, the model's fields 3, an op or attribute 4, a closed set 5, a required
//               attribute 8, causal 14, a slot's kind 11 and fields 12 and 13, kv_from 22, weight_index 18,
//               a recurrence's declarations 31)
//   rule 2      ir_version 0 or 1
//   rules 3, 6-10, 14   model fields and block.repeat = model.n_layers; per-layer scope and length; arity;
//               expr parses; split, rmsnorm, rope, attention, moe_ffn and conv1d_causal values; causal true
//   rules 11-13 slots: kinds, declared once, each read, the stateful node's slot of its kind, attention
//               against its kv slot (window, sinks, kv_heads, head_dim, per application), recurrent slots
//   rules 15-17a  turn_state, evict, state_weight; {i} once and only in the block; tokens and pos; one
//               attention node per kv slot; nothing reads depth_route's out
//   rules 18-23 weight_index, tensor_index placement, depth_route, kv_rows active, kv_from, depth_route
//               with turn_state or list values
//   rules 31-33 recurrence (revision 1.1, section 6.10; graph_ir_recurrence.h): the node's grammar, names and
//               reads, its slot's tensors (init "body" exactly where the node's init assigns), one reader
//   rule 36     every value a node reads is written before it (embed, the block as application 0, head;
//               tokens and pos excepted); the embed section writes x and the head logits; under depth_route the
//               block's nodes after it read only x, pos and values written before them in the application, and
//               the head x and the embed section's values (sections 3.5, 7.4(b))
// Container (ir_container_errors, the GGUF the document loads from):
//   rule 24     paperkiln.ir_version: a uint32 equal to the file's embedded document's ir_version
//   rule 30     paperkiln.ir_sha256 (when present): a string equal to the digest of the file's embedded
//               document (ir_digest.h), which must have one; checked against the embedded document even when
//               another is run in its place
//   rule 26     the depth_route router's ne {d, n_rec}
//   rule 27     weight_index: architecture paperkiln, <arch>.block_count = the stored layers
//   rule 28     state tokens: paperkiln.state_tokens.k with an embedding state_weight and the reverse,
//               state_weight ne {d, k}, architecture paperkiln, tokenizer.specials naming paperkiln/1's
//               markers, no chat template but paperkiln/1
// Rule 25 (every tensor named is in the file) and rule 34 (a recurrence weight's element count; an input's
// width when the rows arrive) are the Interpreter's binding and run, as is rule 37 (the widths a node reads fit
// it: scale, split, rmsnorm group, add/mul/sub, mul_col, rope, depth_attend, expr), checked when the rows arrive
// (section 10.1.1) on the CPU and the device, the error naming the rule; a missing S = 1 depth_attend
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

// A document's JSON text parsed, refusing (std::runtime_error, "IR_SPEC 3.2 (rule 35)") an object that holds a
// member name twice, at any depth, "_" members included: nlohmann's own parse keeps the last of the two. Text that
// is not JSON throws nlohmann's parse_error, as json::parse does. Every document ember loads (EMBER_IR, the GGUF's
// paperkiln.ir, the adapter's "ir") is read through it.
nlohmann::json parse_ir_document(const std::string& text);

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
