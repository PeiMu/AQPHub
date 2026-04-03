#pragma once

#include "jit/aqp_jit_abi.h"
#include "simplest_ir.h"

// Forward-declare LLVM types to avoid pulling LLVM headers into every TU
// that includes this header.
// Note: IRBuilder is a template in LLVM and cannot be forward-declared here;
// it is only used inside ir_to_llvm.cpp.
namespace llvm {
class LLVMContext;
class Module;
class Function;
class Value;
class Type;
namespace orc { class LLJIT; }
}

namespace ir_sql_converter {
class AQPStmt;
class AQPExpr;
class SimplestAttr;
class SimplestConstVar;
class SimplestVarConstComparison;
class SimplestVarComparison;
class SimplestLogicalExpr;
class SimplestIsNullExpr;
class SimplestInExpr;
}

namespace aqp_jit {

/**
 * Column schema entry: maps a column index (in AQPChunkView) to its
 * attribute metadata from the IR.  Built by walking ancestor target_list
 * nodes in the IR tree before compiling a filter.
 */
struct ColSchema {
    unsigned int table_idx;
    unsigned int col_idx;
    int32_t      dtype;   // AQP_DTYPE_* constant
};

/**
 * IrToLlvmCompiler
 *
 * Compiles filter expressions from AQP IR subtrees to native machine code
 * using LLVM ORC JIT.  One instance can compile multiple expressions; each
 * compiled function is independent.
 *
 * Usage:
 *   IrToLlvmCompiler compiler(/*use_o3=*\/false);
 *   AQPExprFn fn = compiler.CompileFilter(*filter_node, schema);
 *   // fn is now a native function pointer valid for the lifetime of compiler
 */
class IrToLlvmCompiler {
public:
    explicit IrToLlvmCompiler(bool use_o3 = false);
    ~IrToLlvmCompiler();

    // Non-copyable, movable
    IrToLlvmCompiler(const IrToLlvmCompiler &) = delete;
    IrToLlvmCompiler &operator=(const IrToLlvmCompiler &) = delete;

    /**
     * Compile the filter expression(s) inside a SimplestFilter IR node.
     * schema: maps AQPChunkView column indices to their attribute metadata.
     * Returns a compiled AQPExprFn, or nullptr on failure.
     */
    AQPExprFn CompileFilter(
        const ir_sql_converter::AQPStmt &filter_node,
        const std::vector<ColSchema> &schema);

    /**
     * Compile a single expression directly (used for testing).
     * Returns nullptr on failure.
     */
    AQPExprFn CompileExpr(
        const ir_sql_converter::AQPExpr &expr,
        const std::vector<ColSchema> &schema);

private:
    bool use_o3_;

    // LLVM state — managed via unique_ptr to avoid including LLVM headers here
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // LLVM IR emission helpers (used inside ir_to_llvm.cpp only)
};

} // namespace aqp_jit
