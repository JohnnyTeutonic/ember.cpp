#pragma once
// Load-time validation of an IR document (paperkiln spec/IR_SPEC.md section 10.1, the rules its
// tools/ir_validate.py checks), so ember refuses at load what every consumer must refuse, naming the
// rule. Messages read "IR_SPEC <section>: <what>", as the validator's do.
//
//   rule 1      the JSON Schema (graph_ir_schema.h, vendored from paperkiln's spec/ir.schema.json):
//               types, required keys and the closed key sets of the document, every op's node and every
//               slot (an attribute an op does not define is refused by name; keys starting with "_" are
//               prose, allowed anywhere)
//
// The Interpreter calls validate_ir after its own load checks (so a document those refuse keeps their
// message), then refuses what ember does not implement among valid documents (section 10.3: the
// node key tensor_index, section 7.2, until it lands).
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

struct GGUFData;

namespace ember_ir {

// Rule 1: the schema's failures, one message each.
std::vector<std::string> ir_schema_errors(const nlohmann::json& doc);

// Every violation of the document rules above, in the validator's order, duplicates removed.
std::vector<std::string> ir_document_errors(const nlohmann::json& doc);

// Throws std::runtime_error listing every violation ("; " between them) when there is one.
void validate_ir(const nlohmann::json& doc, const GGUFData& gguf);

}  // namespace ember_ir
