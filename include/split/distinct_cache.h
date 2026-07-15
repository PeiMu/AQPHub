/*
 * Lazy per-(table, column) distinct-count snapshot with a persisted cache
 * file. Stands in for DuckDB's base-table HLL distinct counts in the
 * IR-native join optimizer (new_split_strategy_analysis.md §9.2): looked up
 * once per join column via plain SQL (COUNT(DISTINCT col)) through the
 * generic adapter, then reused across queries and runs.
 */

#pragma once

#include <map>
#include <string>
#include <unordered_map>

namespace middleware {

class EngineAdapter;

class DistinctCache {
public:
  // cache_path == "" disables persistence (in-memory only).
  explicit DistinctCache(std::string cache_path);
  ~DistinctCache();

  // Distinct count of table.column. On miss runs
  // SELECT COUNT(DISTINCT "column") FROM "table" through the adapter and
  // persists the result. Returns <= 0 when the lookup fails (caller falls
  // back to relation cardinality, the no-HLL path).
  double Get(EngineAdapter &adapter, const std::string &table,
             const std::string &column);

  // |corr(rowid, column)| of a base table: how well the physical row order
  // tracks the column, i.e. how effective join-filter block skipping is when
  // probing this table on that column. On miss runs SELECT corr(rowid, ...)
  // through the adapter and persists the result (stored under the same file
  // format with a "corr:" column-field prefix). Returns < 0 on failure;
  // callers should clamp to [0, 1].
  double GetCorrelation(EngineAdapter &adapter, const std::string &table,
                        const std::string &column);

  // Unfiltered row count of a base table (SELECT COUNT(*)), persisted under
  // the column-field tag "rows:". Returns <= 0 on failure.
  double GetRowCount(EngineAdapter &adapter, const std::string &table);

  // Return all cached row counts (from "rows:" entries loaded from file)
  // without touching the adapter. Used to pre-populate base_count_cache_
  // on bg-thread TopDownSplitter instances.
  std::map<std::string, double> GetAllCachedRowCounts() const;

  // In-memory-only lookups (no adapter calls). Return <= 0 on miss.
  double GetCached(const std::string &table, const std::string &column) const;
  double GetCachedRowCount(const std::string &table) const;

  static std::string DefaultPath();

private:
  void Load();
  void Save();

  std::string path_;
  std::unordered_map<std::string, double> map_; // "table\tcolumn" -> distinct
  bool dirty_ = false;
};

} // namespace middleware
