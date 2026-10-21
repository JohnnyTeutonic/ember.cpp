# The IR's `expr` op in ember (22 Oct 2026)

Runtime-defined elementwise ops (paperkiln ir.hpp, op `expr`): a paper's new activation or gate as a
string in a closed grammar, run without recompiling. ember's CPU interpreter: `graph_ir_expr.{h,cpp}`
(the parser vendored from paperkiln src/expr.cpp at 1159d63e with its messages unchanged; ember's own
forward on row-major buffers, compiled with -ffp-contract=off).

**Op level** (`op_gate.log`): paperkiln's own tests/test_ir_expr.py with `EXPR_EVAL_BIN` pointing at
ember's `tools/ir_expr_eval.cpp` (paperkiln's expr_eval JSON contract): 46 passed. The 80 random
cases through every primitive and the activations papers use are within 1.19e-7 (scaled) of the
numpy reference (tools/ir_expr.py; the gate is 1e-5), with non-finite entries in the same places,
and every rejected string is rejected at the same column.

**Whole model** (`gpt2_tiny_gate.json`, `tools/expr_model_gate.py`): gpt2_tiny (paperkiln export;
sha256 gguf 37a30e0a..., IR 184174e5..., mtstudio greedy reference 695b1412...) with its
`gelu` (tanh) node replaced by `expr` "0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))", served
through EMBER_IR on the CPU interpreter: greedy replies vs mtstudio 40/40 (tokens 40/40, stop
reasons 40/40); first-token logits vs the built-in gelu within 8.1e-7 of the logit range, top-1
40/40.

Not yet: a CUDA kernel (an IR with `expr` names it and asks for EMBER_IR_DEVICE=cpu), WASM
(built from the same source, not yet gated).
