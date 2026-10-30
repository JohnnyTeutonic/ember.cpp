// Reproduce Windows SDK's CONST macro before parsing the expression header.
// The old Op::CONST enumerator failed the Windows wheel build under this macro.
#define CONST const
#include "graph_ir_expr.h"

int main() {
    using namespace ember_ir::expr;
    static_assert(static_cast<int>(Op::CONSTANT) == 0, "preserve device instruction encoding");
    const auto stages = lower(Expr::parse("1.25 + x", 1));
    for (const auto& stage : stages)
        for (const auto& instruction : stage.code)
            if (instruction.op == Op::CONSTANT && instruction.c == 1.25f) return 0;
    return 1;
}
