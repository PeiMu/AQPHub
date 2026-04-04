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
    explicit IrToLlvmCompiler(bool use_o3 = false, bool use_simd = false);
    ~IrToLlvmCompiler();

    // Non-copyable, movable
    IrToLlvmCompiler(const IrToLlvmCompiler &) = delete;
    IrToLlvmCompiler &operator=(const IrToLlvmCompiler &) = delete;

    // SIMD configuration (detected at init)
    unsigned GetVecWidth() const;
    bool HasSIMD() const;

    /**
     * Level 2+: Compile ALL filter expressions inside a SimplestFilter IR node,
     * fused into a single function with AND semantics.
     * Returns a compiled AQPExprFn, or nullptr on failure.
     */
    AQPExprFn CompileFilter(
        const ir_sql_converter::AQPStmt &filter_node,
        const std::vector<ColSchema> &schema);

    /**
     * Level 1: Compile a single expression into its own function.
     * Used for per-expression granularity and testing.
     * Returns nullptr on failure.
     */
    AQPExprFn CompileExpr(
        const ir_sql_converter::AQPExpr &expr,
        const std::vector<ColSchema> &schema);

    /**
     * Level 2: Compile a projection operator.
     * in_schema: column layout of the input chunk.
     * proj_node: the SimplestProjection IR node whose target_list defines
     *            which input columns map to which output columns.
     * Returns compiled AQPOperatorFn (int32_t fn(AQPChunkView*in, AQPChunkView*out)),
     * or nullptr on failure.
     */
    AQPOperatorFn CompileProjection(
        const ir_sql_converter::AQPStmt &proj_node,
        const std::vector<ColSchema> &in_schema);

    /**
     * Level 2: Compile hash join build side.
     * Generates a function that loops over the input chunk, extracts key
     * columns (from SimplestHash::hash_keys), and inserts each row into
     * the portable hash table via aqp_ht_insert.
     *
     * Signature: void fn(AQPChunkView *in, void *hash_table)
     * The hash table must be created by aqp_ht_create before calling.
     * Payload = full row (all input columns concatenated as raw bytes).
     *
     * Returns function pointer, or nullptr on failure.
     */
    void *CompileHashBuild(
        const ir_sql_converter::AQPStmt &hash_node,
        const std::vector<ColSchema> &in_schema);

    /**
     * Level 2: Compile hash join probe side.
     * Generates a function that loops over the probe input chunk, extracts
     * probe keys (from SimplestJoin::join_conditions), probes the hash table
     * via aqp_ht_probe, and writes matching row indices to a selection vector.
     *
     * Signature: uint64_t fn(AQPChunkView *probe_chunk, void *hash_table, AQPSelView *sel)
     * Returns count of matching probe rows.
     */
    void *CompileHashProbe(
        const ir_sql_converter::AQPStmt &join_node,
        const std::vector<ColSchema> &probe_schema);

    /**
     * Level 2: Compile an aggregate operator (ungrouped).
     * Generates a function that loops over the input chunk and updates
     * accumulator state (SUM, COUNT, MIN, MAX, AVG).
     *
     * Signature: void fn(AQPChunkView *in, void *agg_state)
     * agg_state layout: one 8-byte slot per aggregate function in agg_fns order.
     *   SUM/COUNT/CountStar: int64_t accumulator
     *   MIN/MAX: int64_t (for integers) or double (for floats), initialized by caller
     *   AVG: { int64_t sum, int64_t count } = 16 bytes
     *
     * Returns function pointer cast to AQPOperatorFn, or nullptr on failure.
     * (We reuse AQPOperatorFn as a generic fn pointer; caller casts appropriately.)
     */
    void *CompileAggUpdate(
        const ir_sql_converter::AQPStmt &agg_node,
        const std::vector<ColSchema> &in_schema);

    /**
     * Level 3: Compile a fused pipeline (Filter → Projection).
     * Generates a single row loop that evaluates filter predicates and,
     * for matching rows, copies projected columns to the output chunk.
     * Eliminates intermediate DataChunk materialization.
     *
     * Signature: int64_t fn(AQPChunkView *in, AQPChunkView *out)
     * Returns count of output rows.
     *
     * filter_node: IR FilterNode (qual_vec), may be null (no filter)
     * proj_node:   IR ProjectionNode (target_list), may be null (no projection)
     * in_schema:   column layout of the input chunk (from source/scan)
     */
    AQPPipelineFn CompilePipeline(
        const ir_sql_converter::AQPStmt *filter_node,
        const ir_sql_converter::AQPStmt *proj_node,
        const std::vector<ColSchema> &in_schema);

    /**
     * Level 3: Compile Filter + Aggregate fusion.
     * Fused loop: for each row, evaluate filter; if match, update accumulator.
     * No intermediate DataChunk between filter and aggregate.
     *
     * Signature: void fn(AQPChunkView *in, void *agg_state)
     * Returns function pointer, or nullptr on failure.
     */
    void *CompileFilterAggFusion(
        const ir_sql_converter::AQPStmt *filter_node,
        const ir_sql_converter::AQPStmt *agg_node,
        const std::vector<ColSchema> &in_schema);

    /**
     * Level 4: Compile an entire sub-plan into a coordinator function.
     * The coordinator orchestrates multiple pipelines:
     *   1. Identifies pipeline segments from the IR tree
     *   2. Runs build-side pipelines first (populating hash tables)
     *   3. Runs probe-side pipelines using the hash tables
     *
     * sub_ir: the complete sub-plan IR tree (from ir_query_splitter)
     * Returns AQPSubPlanFn, or nullptr if the sub-plan is too complex.
     */
    void *CompileSubPlan(const ir_sql_converter::AQPStmt &sub_ir);

private:
    bool use_o3_;
    bool use_simd_;

    // LLVM state — managed via unique_ptr to avoid including LLVM headers here
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // LLVM IR emission helpers (used inside ir_to_llvm.cpp only)
};

} // namespace aqp_jit
