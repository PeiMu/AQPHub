#pragma once

#include "jit/aqp_jit_abi.h"
#include "simplest_ir.h"

namespace middleware { namespace storage {
struct PipelineKernelPlan;
}} // namespace middleware::storage

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

// Opaque handle for an isolated LLVM ORC ResourceTracker.
// Used to manage lifetime of background-compiled JIT modules independently
// from the main compilation session's ResetModules() cycle.
struct JITTrackerHandle {
  void *ptr = nullptr; // opaque ResourceTrackerSP*
  ~JITTrackerHandle();
  JITTrackerHandle() = default;
  JITTrackerHandle(JITTrackerHandle &&o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
  JITTrackerHandle &operator=(JITTrackerHandle &&o) noexcept {
    if (this != &o) { Reset(); ptr = o.ptr; o.ptr = nullptr; }
    return *this;
  }
  JITTrackerHandle(const JITTrackerHandle &) = delete;
  JITTrackerHandle &operator=(const JITTrackerHandle &) = delete;
  void Reset(); // removes tracked modules and frees the handle
};

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
   * Compile a range filter: key >= min_val AND key <= max_val.
   * Generates a selection-vector function (AQPExprFn) that checks one
   * integer column against compile-time constant bounds.
   * chunk_col_idx: position of the key column in the input chunk.
   * dtype: AQP_DTYPE_INT32 or AQP_DTYPE_INT64.
   */
  AQPExprFn CompileRangeFilter(unsigned chunk_col_idx, int32_t dtype,
                               int64_t min_val, int64_t max_val);

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

  // Note: CompileHashBuild and CompileHashProbe (standalone hash-join
  // operator JIT against the obsolete AQPHashTable) were removed. The active
  // probe path is CompileFilterProbeProjectFusion below, which targets
  // DuckDB's JoinHashTable directly. Build is left to DuckDB native; see the
  // rationale in duckdb_adapter.cpp near the hash-join RegisterJIT block.

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

  // Note: CompileFilterHashBuildFusion was removed alongside
  // CompileHashBuild. The build path uses DuckDB native; only the build
  // side's *filter* is JIT'd, via CompilePipeline / dispatch in
  // physical_filter.cpp.

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
   * payload_schema: column layout of the build-side payload, in the order
   *   DuckDB's JoinHashTable lays them out in its row store. Needed to map
   *   projection attrs to payload byte offsets.
   */
  // payload_row_indices[i] is the row-format column index of payload_schema[i]
  // within DuckDB's build-side row layout. Required for direct-HT probe:
  // the JIT'd code reads payload data at view->data_offsets[num_keys + payload_row_indices[i]].
  //
  // lhs_output_idxs: indices into probe_schema for LHS output cols, in chunk order.
  // rhs_output_layout_idxs: indices into HT layout = [keys, payload] for RHS
  // output cols, in chunk order. Output chunk shape is [lhs cols, rhs cols].
  // lhs_output_dtypes / rhs_output_dtypes: AQP_DTYPE_* of each output column,
  // derived from DuckDB's actual chunk schema (NOT from AQP IR's possibly-
  // reordered schemas). Used for output elem_size — critical when probe/payload
  // schemas may be out-of-sync with the physical HT layout.
  // If lhs_output_idxs and rhs_output_layout_idxs are both empty, falls back
  // to all probe + all payload cols (legacy).
  // lhs_key_chunk_idxs / lhs_key_dtypes (optional): DuckDB-authoritative LHS
  // join key positions (from PhysicalComparisonJoin::conditions[i].left as
  // BoundReferenceExpression::index) and their dtypes. When non-empty, the
  // JIT uses these directly for key extraction instead of looking up via
  // AQP IR's (table_idx, col_idx) match against probe_schema — that lookup is
  // unsafe when AQP IR's probe_schema ordering diverges from DuckDB's
  // physical chunk ordering even when dtypes happen to coincide.
  void *CompileFilterProbeProjectFusion(
      const ir_sql_converter::AQPStmt *filter_node,
      const ir_sql_converter::AQPStmt &join_node,
      const ir_sql_converter::AQPStmt *proj_node,
      const std::vector<ColSchema> &probe_schema,
      const std::vector<ColSchema> &payload_schema,
      const std::vector<int> &payload_row_indices,
      const std::vector<int> &lhs_output_idxs = {},
      const std::vector<int> &rhs_output_layout_idxs = {},
      const std::vector<int32_t> &lhs_output_dtypes = {},
      const std::vector<int32_t> &rhs_output_dtypes = {},
      const std::vector<int> &lhs_key_chunk_idxs = {},
      const std::vector<int32_t> &lhs_key_dtypes = {});


  // --- Isolated compilation for background threads ---
  // Create an isolated tracker. Modules compiled with this tracker are
  // independent from ResetModules() and survive until the handle is Reset().
  JITTrackerHandle CreateIsolatedTracker();

  // Compile with an isolated tracker (for background thread use).
  // The returned function pointer is valid until tracker.Reset() is called.
  AQPExprFn CompileExpr(const ir_sql_converter::AQPExpr &expr,
                        const std::vector<ColSchema> &schema,
                        JITTrackerHandle &tracker);
  AQPExprFn CompileFilter(const ir_sql_converter::AQPStmt &filter_node,
                          const std::vector<ColSchema> &schema,
                          JITTrackerHandle &tracker);
  AQPPipelineFn CompilePipeline(const ir_sql_converter::AQPStmt *filter_node,
                                const ir_sql_converter::AQPStmt *proj_node,
                                const std::vector<ColSchema> &in_schema,
                                JITTrackerHandle &tracker);
  AQPPipelineKernelFn CompilePipelineKernel(
      const middleware::storage::PipelineKernelPlan &plan,
      JITTrackerHandle &tracker);

  void SetPrefetch(bool enable, int distance = 8) {
    prefetch_ = enable;
    prefetch_distance_ = distance;
  }

  // Phase 6: separate look-ahead distances for the ROF fused-probe path
  // (CompileFilterProbeProjectFusion). entry_dist prefetches the ht_entry_t
  // cache line; row_dist prefetches the build-side row reached via the entry.
  // 0 disables that level. Defaults (24 / 12) are starting values from plan
  // §10.4. On JOB the impact is small — most probes land in L2/L3 already
  // because the HT for JOB tables fits in cache — but the knobs are kept
  // exposed so workloads with larger HTs (TPC-H SF100+, custom warehouse
  // data) can tune. Use --jit-prefetch-entry-dist=N / --jit-prefetch-row-
  // dist=N from the CLI.
  void SetProbePrefetchDistances(int entry_dist, int row_dist) {
    prefetch_entry_distance_ = entry_dist;
    prefetch_row_distance_ = row_dist;
  }

  void SetBatchProbe(bool enable) { batch_probe_ = enable; }

  void SetInlineHash(bool enable) { inline_hash_ = enable; }

  void SetCache(bool enable, const std::string &dir = "");

  // Releases all JIT-compiled modules (machine code, IR, symbol-table
  // entries, ExecutionSession state) added since the last reset. The
  // runtime helper symbols (aqp_like_match, memcpy, etc.) survive across
  // resets because they are registered without a tracker.
  //
  // **Caller invariant**: no JIT function pointer obtained from this
  // compiler may be in use when ResetModules() is called. Pair it with
  // clearing whatever map holds those pointers (e.g. clear the per-query
  // AQPJITContext) BEFORE this call. Currently single-threaded; for
  // future multi-thread dispatch this needs an additional lock that
  // serialises Reset against dispatch.
  void ResetModules();

private:
  OptLevel opt_level_;
  SimdISA simd_isa_;
  bool use_simd_; // derived: true if simd_isa_ != OFF
  bool prefetch_ = false;
  int prefetch_distance_ = 8;
  // Phase 6: stage-2 look-ahead distances used by
  // CompileFilterProbeProjectFusion (consumer-side prefetch). entry_distance
  // targets the ht_entry_t cache line; row_distance targets the build-side
  // row reached via that entry. Defaults (24 / 12) from plan §10.4; on JOB
  // the effect is marginal because the HT typically fits in L3 — tune via
  // SetProbePrefetchDistances or the CLI flags for larger workloads.
  int prefetch_entry_distance_ = 24;
  int prefetch_row_distance_ = 12;
  bool batch_probe_ = false;
  bool inline_hash_ = true;
  bool cache_enabled_ = false;
  std::string cache_dir_;

  // LLVM state — managed via unique_ptr to avoid including LLVM headers here
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // LLVM IR emission helpers (used inside ir_to_llvm.cpp only)
};

} // namespace aqp_jit
