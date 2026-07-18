/*
 * query_jit_steps.h — Query-JIT (--jit-level=query) IR analysis + step plan.
 *
 * AnalyzeQueryJit (Phase 0): loose accept/reject over the sub-query IR with
 * [AQP-QJIT] stderr traces (coverage measurement).
 *
 * BuildExecutionSteps (Phase 3): STRICT builder that decomposes the IR tree
 * into execution steps (lingo-db SplitIntoExecutionSteps model degenerated
 * to a tree walk): the probe spine of each (sub)tree is one step; every
 * hash-join build child spawns its own step sinking into a hash table.
 * Steps are emitted in post-order = execution order (builds before probers).
 * The strictness rationale: ir_to_llvm's expression emitters fall back to
 * "pass all rows" for unrecognized constructs, which is safe under
 * pipeline-jit (DuckDB re-filters) but FATAL when query-jit owns the result.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "qjit/query_jit_runtime.h" // QjitAggFn

namespace ir_sql_converter {
class AQPStmt;
class AQPExpr;
}

namespace qjit {

struct QjitAnalysisResult {
  bool accepted = false;
  std::string reject_reason; // empty when accepted
  int num_joins = 0;
  bool has_aggregate = false;
  // Mark-join IN-list patterns rewritten to streaming in-set filters (§5.2).
  int num_mark_in = 0;
};

// Walk the sub-query IR and report v1 compilability. `label` identifies the
// sub-query in the stderr trace (e.g. temp-table name or "result").
QjitAnalysisResult AnalyzeQueryJit(const ir_sql_converter::AQPStmt &root,
                                   const std::string &label);

// ---------------------------------------------------------------------------
// Execution-step plan
// ---------------------------------------------------------------------------

// One referenced source column. (table_index, column_index) is the identity
// the expression codegen matches against (FindColIdx); column_name resolves
// the column in the FlatTable (pipeline-kernel convention: by name).
struct QjitColumnRef {
  unsigned table_index = 0;
  unsigned column_index = 0;
  std::string column_name;
  // Expected dtype from the IR attr var_type: AQP_DTYPE_INT32 or
  // AQP_DTYPE_VARCHAR (aqp_jit_abi.h). The executor cross-checks this
  // against the actual FlatTable column type and falls back on mismatch.
  int expected_dtype = 0;
};

// Where a value lives at a given point of a step's row loop: either a step
// source column (src_col >= 0) or a payload/key slot of a hash table this
// step has already probed (ht_id/layout_col >= 0). Exactly one is set.
struct QjitValueLoc {
  int src_col = -1;    // index into QjitStep::cols
  int ht_id = -1;      // index into QjitQueryPlan::hts
  int layout_col = -1; // index into QjitHtDesc::cols
  int dtype = 0;       // AQP_DTYPE_INT32 or AQP_DTYPE_VARCHAR
};

// One hash-table row column. cols[0..num_keys) are the join keys (stored
// sign-extended as i64); the rest are payloads (VARCHAR = 16-byte
// QjitString shallow copy, INT32 = 4 bytes). A payload requirement whose
// attr matches a key reuses the key slot (load i64 + trunc) — matching
// scans keys first, so no duplicate slot is created.
struct QjitHtCol {
  unsigned table_index = 0;
  unsigned column_index = 0;
  std::string column_name; // debug/trace only
  int dtype = 0;           // key: always AQP_DTYPE_INT32 in v1
  uint32_t offset = 0;     // byte offset within the row (assigned last)
};

// Hash-table row layout:
//   [validity byte prefix, padded to 8 | keys i64 | VARCHAR | INT32]
// Validity bit for layout col c: byte c/8, bit 1<<(c%8), 1=valid
// (TupleDataLayout convention). Key bits are always 1 (NULL keys are
// skipped at build).
struct QjitHtDesc {
  uint32_t num_keys = 0;
  std::vector<QjitHtCol> cols; // keys then payloads
  uint32_t prefix_bytes = 0;   // assigned in the offset pass
  uint32_t tuple_size = 0;     // assigned in the offset pass
  // All scan table indices in the build subtree (transitive, including
  // nested probe sides) — the resolution pass routes attr references
  // through the HT whose build set contains the attr's table.
  std::vector<unsigned> build_tables;
};

// In-loop operation, executed in vector order per source row.
struct QjitStepOp {
  enum Kind { Filter, Probe } kind = Filter;
  // Filter: conjunct expression (raw pointer into the sub-query IR — the
  // IR must outlive the plan). Restricted to step-source columns: EmitExpr
  // can only address ctx source columns, and silently passes rows for
  // unknown ones (reject "filter:payload-ref").
  const ir_sql_converter::AQPExpr *filter = nullptr;
  // Probe: chain-walk lookup into hts[ht_id]; keys[i] is the probe-side
  // value of join condition i (source col or payload of an earlier probe
  // in this step). NULL key => row produces no matches.
  int ht_id = -1;
  std::vector<QjitValueLoc> keys;
};

struct QjitAggCellPlan {
  QjitAggFn fn = QjitAggFn::Count;
  bool has_arg = false; // false only for CountStar
  QjitValueLoc arg;
};

struct QjitStep {
  // ScanNode base-table name, or ChunkNode temp-table name (source_is_temp).
  std::string source_table;
  unsigned source_table_index = 0;
  // Temp-table source (qjit_temps_ / on-demand CDC conversion): columns
  // resolve POSITIONALLY by QjitColumnRef::column_index (chunk attr names
  // may be "colN" placeholders, so name lookup is unreliable for temps).
  bool source_is_temp = false;
  // Deduplicated referenced source columns; defines the QjitTableView the
  // morsel body sees (ctx->sources[step_index].cols[i]).
  std::vector<QjitColumnRef> cols;
  std::vector<QjitStepOp> ops;

  // §5.5 A+ join-filter pushdown. A guard hoists a cheap row test derived
  // from a Probe op (all of whose keys are step-source columns) ahead of
  // the expensive (string-predicate) filters: build-key-0 range check,
  // plus an HT-membership existence pre-probe when `membership` is set
  // (i.e. expensive filters actually run between the guard and the probe).
  // Guards only drop rows their probe would drop anyway, and inner-join
  // probes commute with filters, so hoisting preserves results.
  struct Guard {
    int op_index = -1;       // the guarded Probe op (post-reorder index)
    bool membership = false; // emit existence pre-probe too
  };
  // ops[0..guard_pos) are the cheap filters (reordered ahead of expensive
  // ones); guards are emitted at position guard_pos, before the rest.
  int guard_pos = 0;
  std::vector<Guard> guards; // ordered by op_index
  // Step-source column (index into cols) for morsel-level block skipping
  // against per-block min/max stats: key 0 of guards[0]. -1 = none.
  int block_skip_col = -1;

  enum SinkKind { Result, HtBuild, Agg } sink = Result;
  int sink_ht = -1; // HtBuild: target hts[] index
  // Result: one loc per output column, in root target_list order.
  // HtBuild: one loc per hts[sink_ht].cols entry (keys then payloads).
  std::vector<QjitValueLoc> outputs;
  // Agg sink: one plan per agg_fns entry (cell i = agg_fns[i]).
  std::vector<QjitAggCellPlan> agg_cells;
};

struct QjitSortCol {
  int col_idx;     // index into result columns
  bool ascending;
  int32_t dtype;   // AQP_DTYPE_* for type-aware comparison
};

struct QjitQueryPlan {
  // Out-of-line special members: owned_exprs holds forward-declared AQPExpr.
  QjitQueryPlan();
  QjitQueryPlan(QjitQueryPlan &&) noexcept;
  QjitQueryPlan &operator=(QjitQueryPlan &&) noexcept;
  ~QjitQueryPlan();

  std::vector<QjitStep> steps; // execution order; steps.back() = root step
  std::vector<QjitHtDesc> hts;
  bool has_agg = false;
  // has_agg: result column i = agg cell agg_output_cells[i] (root
  // projections over the aggregate may reorder/duplicate cells).
  std::vector<int> agg_output_cells;
  // Synthesized filter expressions (mark-join IN-list rewrite) referenced
  // by QjitStepOp::filter; owned here so they outlive codegen.
  std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> owned_exprs;

  std::vector<QjitSortCol> order_by; // peeled ORDER BY spec (empty = none)
  int64_t limit = -1; // peeled LIMIT value (-1 = no limit)
};

// Strict builder. Returns true and fills `out` when every construct is
// inside the verified codegen whitelist; on false `reason` carries the
// fallback trace token. Requires AnnotateBuildSides to have run (rejects
// "join:build-side-unannotated" otherwise).
bool BuildExecutionSteps(const ir_sql_converter::AQPStmt &root,
                         QjitQueryPlan &out, std::string &reason);

} // namespace qjit
