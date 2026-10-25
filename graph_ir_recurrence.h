#pragma once
// The IR's `recurrence` op (paperkiln spec/IR_SPEC.md section 6.10, the recurrence profile, revision 1.1) on
// the CPU, written for ember from the specification text (no code taken from paperkiln): a recurrent op
// defined in the document by an index map, declarations of its inputs and weights, and bodies in a closed
// grammar (expr's, section 5.19, plus indexed reads with integer offsets, bare index values and an ordered
// sum) for the initial state, intermediate lets, the per-token update and the readout.
//
// Per token (section 6.10.3): every input reads row t; the lets are evaluated in order; every state tensor's
// update is evaluated from the state before the token, and the new state replaces the old only when all of
// it is formed; the readout is evaluated with the state after the update. A body is evaluated at every point
// of its assignment's index space, row-major. A read outside a tensor's shape is +0; sum(j, e) is
// ((+0 + e|0) + e|1) + ..., in ascending j; every primitive is f32 and rounded after it (this file is
// compiled with -ffp-contract=off, CMakeLists.txt). The functions are expr's: where(c, a, b) is a for c > 0
// (a NaN condition selects b), max(a, b) a for a >= b, min(a, b) a for a <= b, relu(e) e for e > 0 else +0,
// sigmoid(e) 1 / (1 + exp(-e)), ^ is powf; numbers are strtof's nearest float, pi the nearest float to pi.
//
// Class (section 6.10.4): EXACT when no body uses exp, log, tanh, erf, sigmoid or ^; ROUNDING otherwise.
//
// compile() checks rules 31 and 32 of section 10.1 and names them: "IR_SPEC 6.10 (rule N): ...", naming the
// body (for example update[1]) and its column for a body that does not parse. The slot (rule 33), the weights'
// element counts (rule 34, container) and the inputs' widths (rule 34, when the rows arrive) are checked by
// the callers through slot_errors(), weights() and input_widths().
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ember_ir {
namespace rec {

struct Program;

class Recurrence {
 public:
  // Throws std::runtime_error "IR_SPEC 6.10 (rule 31|32): ..." for a node the grammar or the read rules refuse.
  static Recurrence compile(const nlohmann::json& node);

  struct StateTensor {
    std::string name;
    std::vector<int64_t> shape;
    size_t numel = 0;
    bool init_body = false;  // the node's init assigns it
  };
  struct WeightDecl {
    std::string decl;    // the declared name (local to the node)
    std::string tensor;  // the tensor name ({i} unresolved)
    size_t numel = 0;
  };
  const std::vector<StateTensor>& state() const;      // the update's tensors, in update order
  const std::vector<WeightDecl>& weights() const;     // in the weights map's order
  const std::vector<size_t>& input_widths() const;    // per value of `in`
  size_t out_width() const;
  bool exact() const;  // no exp, log, tanh, erf, sigmoid or ^ in any body

  // Rule 33 against the node's slot declaration (block.state entry): empty when the slot's tensors are the
  // update's (names, shapes, init "body" exactly for those init assigns), else the message.
  std::string slot_errors(const nlohmann::json& slot) const;

  // The initial state: state[s] (state()[s].numel floats) set to its init body over weights (weights()[w]
  // .numel floats each), or +0.
  void init(const std::vector<const float*>& weights, const std::vector<float*>& state) const;

  // One token: in[v] is input v's row (input_widths()[v] floats), state[s] is read and replaced, out gets
  // out_width() floats. Not thread-safe (scratch is per object).
  void step(const std::vector<const float*>& in, const std::vector<const float*>& weights,
            const std::vector<float*>& state, float* out) const;

  Recurrence();
  ~Recurrence();
  Recurrence(Recurrence&&) noexcept;
  Recurrence& operator=(Recurrence&&) noexcept;

 private:
  std::unique_ptr<Program> p_;
};

}  // namespace rec
}  // namespace ember_ir
