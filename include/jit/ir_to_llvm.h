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
namespace orc {
class LLJIT;
}
} // namespace llvm

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
} // namespace ir_sql_converter

namespace aqp_jit {

/* LLVM optimization level — maps to llvm::OptimizationLevel */
enum class OptLevel { O0, O1, O2, O3 };

/* SIMD ISA selection — determines vector width and available instructions */
enum class SimdISA { OFF, SSE2, AVX, AVX2, AVX512, AUTO };

/**
 * Column schema entry: maps a column index (in AQPChunkView) to its
 * attribute metadata from the IR.  Built by walking ancestor target_list
 * nodes in the IR tree before compiling a filter.
 */
struct ColSchema {
  unsigned int table_idx;
  unsigned int col_idx;
  int32_t dtype; // AQP_DTYPE_* constant
};

// Aggregate JIT disabled: JOB has no aggregate-heavy queries (only MIN on
// VARCHAR).  Saves compilation time.  Re-enable with -DDISABLE_AGG_JIT=0.
#ifndef DISABLE_AGG_JIT
#define DISABLE_AGG_JIT 1
#endif

#if !DISABLE_AGG_JIT
/**
 * Aggregate operation descriptor — built from the physical plan.
 * agg_type: 1=MIN, 2=MAX, 3=SUM, 4=AVG, 5=COUNT, 6=CountStar
 */
struct AggOp {
  int col_idx;           // input chunk column index (-1 for COUNT*)
  int32_t agg_type;
  unsigned state_offset; // byte offset in agg_state
  int32_t dtype;         // AQP_DTYPE_* of the column
};
#endif // !DISABLE_AGG_JIT

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
  explicit IrToLlvmCompiler(OptLevel opt = OptLevel::O1,
                            SimdISA simd = SimdISA::OFF);
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
  AQPExprFn CompileFilter(const ir_sql_converter::AQPStmt &filter_node,
                          const std::vector<ColSchema> &schema);

  /**
   * Level 1: Compile a single expression into its own function.
   * Used for per-expression granularity and testing.
   * Returns nullptr on failure.
   */
  AQPExprFn CompileExpr(const ir_sql_converter::AQPExpr &expr,
                        const std::vector<ColSchema> &schema);

  /**
   * Level 2: Compile a projection operator.
   * in_schema: column layout of the input chunk.
   * proj_node: the SimplestProjection IR node whose target_list defines
   *            which input columns map to which output columns.
   * Returns compiled AQPOperatorFn (int32_t fn(AQPChunkView*in,
   * AQPChunkView*out)), or nullptr on failure.
   */
  AQPOperatorFn CompileProjection(const ir_sql_converter::AQPStmt &proj_node,
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
  void *CompileHashBuild(const ir_sql_converter::AQPStmt &hash_node,
                         const std::vector<ColSchema> &in_schema,
                         const std::vector<int> &needed_payload_cols = {});

  /**
   * Level 2: Compile hash join probe side.
   * Generates a function that loops over the probe input chunk, extracts
   * probe keys (from SimplestJoin::join_conditions), probes the hash table
   * via aqp_ht_probe, and writes matching row indices to a selection vector.
   *
   * Signature: uint64_t fn(AQPChunkView *probe_chunk, void *hash_table,
   * AQPSelView *sel) Returns count of matching probe rows.
   */
  void *CompileHashProbe(const ir_sql_converter::AQPStmt &join_node,
                         const std::vector<ColSchema> &probe_schema);

#if !DISABLE_AGG_JIT
  /**
   * Level 2: Compile an aggregate operator (ungrouped).
   * Generates a function that loops over the input chunk and updates
   * accumulator state (SUM, COUNT, MIN, MAX, AVG).
   *
   * Signature: void fn(AQPChunkView *in, void *agg_state)
   * agg_state layout: one 8-byte slot per aggregate function in agg_fns order.
   *   SUM/COUNT/CountStar: int64_t accumulator
   *   MIN/MAX: int64_t (for integers) or double (for floats), initialized by
   * caller AVG: { int64_t sum, int64_t count } = 16 bytes
   *
   * Returns function pointer cast to AQPOperatorFn, or nullptr on failure.
   * (We reuse AQPOperatorFn as a generic fn pointer; caller casts
   * appropriately.)
   */
  void *CompileAggUpdate(const ir_sql_converter::AQPStmt &agg_node,
                         const std::vector<ColSchema> &in_schema);

  /**
   * Direct aggregate compilation — builds from AggOp descriptors without IR.
   * Same signature and state layout as CompileAggUpdate, but resolves columns
   * entirely from the physical plan.  Preferred over CompileAggUpdate.
   */
  void *CompileAggUpdateDirect(const std::vector<AggOp> &agg_ops,
                               unsigned total_state_size);
#endif // !DISABLE_AGG_JIT

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
  AQPPipelineFn CompilePipeline(const ir_sql_converter::AQPStmt *filter_node,
                                const ir_sql_converter::AQPStmt *proj_node,
                                const std::vector<ColSchema> &in_schema);

#if !DISABLE_AGG_JIT
  /**
   * Level 3: Compile Filter + Aggregate fusion.
   * Fused loop: for each row, evaluate filter; if match, update accumulator.
   * No intermediate DataChunk between filter and aggregate.
   *
   * Signature: void fn(AQPChunkView *in, void *agg_state)
   * Returns function pointer, or nullptr on failure.
   */
  void *CompileFilterAggFusion(const ir_sql_converter::AQPStmt *filter_node,
                               const ir_sql_converter::AQPStmt *agg_node,
                               const std::vector<ColSchema> &in_schema);
#endif // !DISABLE_AGG_JIT

  /**
   * Level 3: Compile Filter + HashBuild fusion.
   * Fused loop: for each row, evaluate filter; if match, extract key,
   * hash inline (FNV-1a), insert into hash table, copy payload.
   * No intermediate DataChunk between filter and hash build.
   *
   * Signature: int64_t fn(AQPChunkView *in, AQPChunkView *, void *ht)
   * Uses AQPPipelineFn signature. The hash table pointer is passed via
   * pipeline_state (3rd arg). Returns 0 (no output rows — data is sunk
   * into the hash table).
   */
  void *
  CompileFilterHashBuildFusion(const ir_sql_converter::AQPStmt *filter_node,
                               const ir_sql_converter::AQPStmt &hash_node,
                               const std::vector<ColSchema> &in_schema,
                               const std::vector<int> &needed_payload_cols = {});

  /**
   * Level 3: Compile Filter + HashProbe + Projection fusion (probe pipeline).
   * Fused loop: for each probe row, evaluate filter; if match, extract key,
   * hash inline (FNV-1a), probe hash table; if found, copy projected columns
   * from BOTH the probe input chunk AND the build-side payload to the output.
   * Eliminates two intermediate DataChunks (filter→probe, probe→projection).
   *
   * Signature: int64_t fn(AQPChunkView *in, AQPChunkView *out, void *ht)
   * Uses AQPPipelineFn signature. The hash table pointer is passed via
   * pipeline_state (3rd arg). Returns count of output rows.
   *
   * filter_node: may be null (no filter before probe).
   * join_node: SimplestJoin with join_conditions defining probe keys.
   * proj_node: may be null (output all probe + payload columns).
   * probe_schema: column layout of the probe input chunk.
   * payload_schema: column layout of the build-side payload (in the order
   *   columns were written by CompileHashBuild / fusion). Needed to map
   *   projection attrs to payload byte offsets.
   */
  void *CompileFilterProbeProjectFusion(
      const ir_sql_converter::AQPStmt *filter_node,
      const ir_sql_converter::AQPStmt &join_node,
      const ir_sql_converter::AQPStmt *proj_node,
      const std::vector<ColSchema> &probe_schema,
      const std::vector<ColSchema> &payload_schema);

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

  /**
   * Level 4: Compile a whole-SQL IR tree (non-split path).
   * Semantically identical to CompileSubPlan; separated for clarity.
   */
  void *CompileSQL(const ir_sql_converter::AQPStmt &sql_ir);

  void SetPrefetch(bool enable, int distance = 8) {
    prefetch_ = enable;
    prefetch_distance_ = distance;
  }

  void SetBatchProbe(bool enable) { batch_probe_ = enable; }

  void SetCache(bool enable, const std::string &dir = "");

private:
  OptLevel opt_level_;
  SimdISA simd_isa_;
  bool use_simd_; // derived: true if simd_isa_ != OFF
  bool prefetch_ = false;
  int prefetch_distance_ = 8;
  bool batch_probe_ = false;
  bool cache_enabled_ = false;
  std::string cache_dir_;

  // LLVM state — managed via unique_ptr to avoid including LLVM headers here
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // LLVM IR emission helpers (used inside ir_to_llvm.cpp only)
};

} // namespace aqp_jit
