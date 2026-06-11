#include "qjit/query_jit_executor.h"

#include <algorithm>
#include <cstring>
#include <thread>

#include "storage/flat_table.h"

namespace qjit {

using middleware::storage::FlatColumn;
using middleware::storage::FlatColumnType;
using middleware::storage::FlatTable;

QjitExecutor::QjitExecutor(int threads, uint64_t morsel)
    : pool_(threads > 0
                ? static_cast<uint32_t>(threads)
                : std::max(1u, std::thread::hardware_concurrency())),
      morsel_size_(morsel == 0 ? 20000 : morsel) {}

const int32_t *QjitExecutor::GetBlockStats(const FlatTable &flat,
                                           int flat_col_idx) {
  const FlatColumn &col = flat.columns[flat_col_idx];
  if (col.type != FlatColumnType::INT32)
    return nullptr;
  std::string key = flat.table_name + "." + std::to_string(flat_col_idx);
  auto it = block_stats_.find(key);
  if (it == block_stats_.end()) {
    const uint64_t nblocks =
        (flat.row_count + QJIT_BLOCK_ROWS - 1) / QJIT_BLOCK_ROWS;
    auto stats = std::make_unique<std::vector<int32_t>>(nblocks * 2);
    for (uint64_t b = 0; b < nblocks; ++b) {
      const uint64_t lo = b * QJIT_BLOCK_ROWS;
      const uint64_t hi = std::min<uint64_t>(lo + QJIT_BLOCK_ROWS,
                                             flat.row_count);
      int32_t bmin = INT32_MAX, bmax = INT32_MIN; // all-NULL => min > max
      for (uint64_t row = lo; row < hi; ++row) {
        if (col.IsNull(row))
          continue;
        const int32_t v = col.GetInt32(row);
        bmin = std::min(bmin, v);
        bmax = std::max(bmax, v);
      }
      (*stats)[2 * b] = bmin;
      (*stats)[2 * b + 1] = bmax;
    }
    it = block_stats_.emplace(std::move(key), std::move(stats)).first;
  }
  return it->second->data();
}

bool QjitExecutor::ResolveSource(const FlatTable &flat,
                                 const std::vector<QjitColumnRef> &cols,
                                 QjitResolvedSource &out, std::string &reason,
                                 int stats_col) {
  out.cols.clear();
  out.nrows = flat.row_count;
  out.block_stats = nullptr;
  out.cols.reserve(cols.size());

  for (size_t ci = 0; ci < cols.size(); ++ci) {
    const QjitColumnRef &ref = cols[ci];
    int idx = flat.FindColumn(ref.column_name);
    if (idx < 0) {
      reason = "source:col-missing:" + ref.column_name;
      return false;
    }
    if ((int)ci == stats_col)
      out.block_stats = GetBlockStats(flat, idx);
    const FlatColumn &col = flat.columns[idx];
    const bool want_varchar = ref.expected_dtype == AQP_DTYPE_VARCHAR;
    if (want_varchar != (col.type == FlatColumnType::VARCHAR) ||
        (!want_varchar && ref.expected_dtype != AQP_DTYPE_INT32)) {
      reason = "source:dtype-mismatch:" + ref.column_name;
      return false;
    }

    QjitColView view{};
    view.validity = col.null_bitmap.get(); // nullptr = all valid
    if (col.type == FlatColumnType::INT32) {
      view.data = col.data.get();
      view.dtype = AQP_DTYPE_INT32;
    } else {
      std::string key = flat.table_name + "." + std::to_string(idx);
      auto it = varchar_views_.find(key);
      if (it == varchar_views_.end()) {
        auto strs = std::make_unique<std::vector<QjitString>>(flat.row_count);
        for (uint64_t row = 0; row < flat.row_count; ++row) {
          if (col.IsNull(row)) {
            std::memset(&(*strs)[row], 0, sizeof(QjitString));
            continue; // never read: strict guards check validity first
          }
          uint32_t len = 0;
          const char *ptr = col.GetVarchar(row, len);
          (*strs)[row] = MakeString(ptr, len);
        }
        it = varchar_views_.emplace(std::move(key), std::move(strs)).first;
      }
      view.data = it->second->data();
      view.dtype = AQP_DTYPE_VARCHAR;
    }
    out.cols.push_back(view);
  }
  return true;
}

bool QjitExecutor::ResolveTempSource(const QjitTable &table,
                                     const std::vector<QjitColumnRef> &cols,
                                     QjitResolvedSource &out,
                                     std::string &reason) {
  out.cols.clear();
  out.nrows = table.NumRows();
  out.cols.reserve(cols.size());

  for (const QjitColumnRef &ref : cols) {
    if (ref.column_index >= table.NumCols()) {
      reason = "source:tmp-col-range:" + ref.column_name;
      return false;
    }
    const QjitTable::ColumnDesc &cd = table.Col(ref.column_index);
    if (cd.dtype != ref.expected_dtype) {
      reason = "source:tmp-dtype:" + ref.column_name;
      return false;
    }
    // Same layout the compiled code expects: INT32 flat array or
    // QjitString[16B] array + 1-bit validity (nullptr = all valid) — the
    // per-column equivalent of QjitTable::FillView.
    QjitColView view{};
    view.data = const_cast<void *>(table.Data(ref.column_index));
    view.validity = const_cast<uint64_t *>(table.Validity(ref.column_index));
    view.dtype = cd.dtype;
    out.cols.push_back(view);
  }
  return true;
}

int64_t QjitExecutor::Run(QjitQueryFn fn,
                          const std::vector<QjitResolvedSource> &srcs,
                          const std::vector<uint32_t> &ht_tuple_sizes,
                          const std::vector<QjitAggCellDesc> &agg_descs,
                          const std::vector<int> &agg_output_cells,
                          QjitTable &result,
                          const std::vector<uint32_t> &ht_key0_offsets) {
  const uint32_t nworkers = pool_.NumWorkers();

  std::vector<QjitTableView> views(srcs.size());
  std::vector<const int32_t *> step_stats(srcs.size(), nullptr);
  for (size_t k = 0; k < srcs.size(); k++) {
    views[k].cols = const_cast<QjitColView *>(srcs[k].cols.data());
    views[k].nrows = srcs[k].nrows;
    views[k].ncols = srcs[k].cols.size();
    step_stats[k] = srcs[k].block_stats;
  }

  std::vector<std::unique_ptr<QjitHashTable>> hts;
  std::vector<void *> ht_ptrs;
  hts.reserve(ht_tuple_sizes.size());
  for (size_t i = 0; i < ht_tuple_sizes.size(); i++) {
    const int64_t k0off =
        i < ht_key0_offsets.size() ? (int64_t)ht_key0_offsets[i] : -1;
    hts.push_back(
        std::make_unique<QjitHashTable>(ht_tuple_sizes[i], nworkers, k0off));
    ht_ptrs.push_back(hts.back().get());
  }

  // Per-worker aggregate states. Arenas are sized up front: QjitAggState
  // keeps a raw arena pointer, so the vector must never reallocate.
  std::vector<QjitStringArena> agg_arenas;
  std::vector<std::unique_ptr<QjitAggState>> agg_states;
  std::vector<void *> worker_states;
  if (!agg_descs.empty()) {
    agg_arenas.resize(nworkers);
    agg_states.reserve(nworkers);
    worker_states.reserve(nworkers);
    for (uint32_t w = 0; w < nworkers; w++) {
      agg_states.push_back(
          std::make_unique<QjitAggState>(agg_descs, &agg_arenas[w]));
      worker_states.push_back(agg_states.back().get());
    }
  }

  QjitQueryContext ctx{};
  ctx.pool = &pool_;
  ctx.sources = views.data();
  ctx.num_sources = views.size();
  ctx.hash_tables = ht_ptrs.empty() ? nullptr : ht_ptrs.data();
  ctx.num_hash_tables = ht_ptrs.size();
  ctx.worker_states = worker_states.empty() ? nullptr : worker_states.data();
  ctx.num_workers = nworkers;
  ctx.morsel_size = morsel_size_;
  ctx.result = &result;
  ctx.user = srcs.empty() ? nullptr : (void *)step_stats.data();

  int64_t rc = fn(&ctx);
  if (rc < 0)
    return rc;

  if (!agg_descs.empty()) {
    // Merge epilogue (single-threaded) + single result row.
    QjitStringArena merge_arena;
    QjitAggState merged(agg_descs, &merge_arena);
    for (uint32_t w = 0; w < nworkers; w++)
      merged.Merge(*agg_states[w]);
    for (size_t i = 0; i < agg_output_cells.size(); i++) {
      const size_t cell = (size_t)agg_output_cells[i];
      const QjitAggCell &c = merged.Cell(cell);
      const QjitAggCellDesc &d = merged.Desc(cell);
      if (d.fn == QjitAggFn::Count || d.fn == QjitAggFn::CountStar) {
        result.AppendI64(0, i, (int64_t)c.count); // empty input => 0
        continue;
      }
      if (!c.seen) {
        result.AppendNull(0, i); // empty input => NULL (DuckDB semantics)
        continue;
      }
      switch (result.Col(i).dtype) {
      case AQP_DTYPE_INT32:
        result.AppendI32(0, i, (int32_t)c.i64);
        break;
      case AQP_DTYPE_INT64:
        result.AppendI64(0, i, c.i64);
        break;
      case AQP_DTYPE_DOUBLE:
        result.AppendF64(0, i, c.f64);
        break;
      case AQP_DTYPE_VARCHAR:
        result.AppendStr(0, i, c.str);
        break;
      default:
        return -2; // adapter type-checks outputs; unreachable
      }
    }
    result.FinishRow(0);
  }

  result.Finalize();
  return static_cast<int64_t>(result.NumRows());
}

} // namespace qjit
