/**
 * query_jit_executor.h — execution glue between the adapter, the compiled
 * entry function (QjitQueryFn) and the qjit runtime:
 *   - ResolveSource: FlatTable -> QjitColView/QjitTableView adapter, with
 *     per-column name + dtype cross-checks (mismatch => fallback reason).
 *     VARCHAR columns get a fabricated QjitString[] view (pointers into the
 *     FlatTable string_pool, no byte copies), memoized per (table, column)
 *     for the lifetime of this executor — FlatTables are loaded once per
 *     process by the storage plan and never move.
 *   - Run: builds the QjitQueryContext (one source view per step, hash
 *     tables, per-worker agg states), invokes the compiled entry over the
 *     worker pool, runs the aggregate merge epilogue (C++: once per query,
 *     not worth inlining in LLVM), finalizes the result table.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "qjit/query_jit_abi.h"
#include "qjit/query_jit_runtime.h"
#include "qjit/query_jit_scheduler.h"
#include "qjit/query_jit_steps.h"

namespace middleware {
namespace storage {
struct FlatTable;
}
} // namespace middleware

namespace qjit {

/* Column views for one resolved step source. `cols` order matches
 * QjitStep::cols (= the compiled module's per-step schema order).
 * The QjitTableView array passed to the compiled code is built by Run()
 * over cols.data(), so the struct can be moved/copied freely before Run. */
struct QjitResolvedSource {
  std::vector<QjitColView> cols;
  uint64_t nrows = 0;
  /* §5.5 A+ block skipping: per-2048-row {min,max} pairs for the step's
   * block_skip_col, or nullptr (temps / no guard / non-INT32). Delivered
   * to the compiled code via QjitQueryContext::user. */
  const int32_t *block_stats = nullptr;
};

class QjitExecutor {
public:
  /* threads <= 0 => hardware_concurrency; morsel == 0 => 20000. */
  QjitExecutor(int threads, uint64_t morsel);

  uint32_t NumWorkers() const { return pool_.NumWorkers(); }

  /* Resolve every step column against `flat` (by name) and cross-check
   * dtypes. On failure returns false and sets `reason` (fallback trace
   * token). stats_col >= 0 requests block min/max stats for that step
   * column (QjitStep::block_skip_col); computed once per (table, column)
   * and memoized, INT32 columns only. */
  bool ResolveSource(const middleware::storage::FlatTable &flat,
                     const std::vector<QjitColumnRef> &cols,
                     QjitResolvedSource &out, std::string &reason,
                     int stats_col = -1);

  /* Resolve step columns against a finalized temp QjitTable. POSITIONAL:
   * ref.column_index indexes the temp schema (chunk attr names may be
   * "colN" placeholders, so names are not authoritative for temps). Views
   * point into `table` — it must outlive the Run (qjit_temps_ lifetime). */
  bool ResolveTempSource(const QjitTable &table,
                         const std::vector<QjitColumnRef> &cols,
                         QjitResolvedSource &out, std::string &reason);

  /* Invoke the compiled entry:
   *   ctx->sources[k]      = srcs[k]            (one per step)
   *   ctx->hash_tables[i]  = fresh QjitHashTable(ht_tuple_sizes[i])
   *   ctx->worker_states[w]= per-worker QjitAggState (when agg_descs set)
   *   ctx->result          = result
   * Blocks until all morsels are done (pool quiescent on return — safe to
   * ResetModules afterwards). When agg_descs is non-empty, merges the
   * per-worker states and emits the single result row (result column i =
   * cell agg_output_cells[i]; COUNT cells emit 0 on empty input, others
   * NULL — DuckDB semantics). Returns result.NumRows() after Finalize, or
   * the negative error code from the entry function. */
  int64_t Run(QjitQueryFn fn, const std::vector<QjitResolvedSource> &srcs,
              const std::vector<uint32_t> &ht_tuple_sizes,
              const std::vector<QjitAggCellDesc> &agg_descs,
              const std::vector<int> &agg_output_cells, QjitTable &result,
              const std::vector<uint32_t> &ht_key0_offsets = {},
              const std::vector<uint8_t> &params_buf = {});

private:
  QjitWorkerPool pool_;
  uint64_t morsel_size_;
  /* Memoized fabricated VARCHAR views, keyed "<table>.<col_index>". */
  std::unordered_map<std::string, std::unique_ptr<std::vector<QjitString>>>
      varchar_views_;
  /* Memoized §5.5 block stats ({min,max} per 2048-row block), same key. */
  std::unordered_map<std::string, std::unique_ptr<std::vector<int32_t>>>
      block_stats_;
  const int32_t *GetBlockStats(const middleware::storage::FlatTable &flat,
                               int flat_col_idx);
};

} // namespace qjit
