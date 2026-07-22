#include "split/distinct_cache.h"

#include <cstdlib>
#include <fstream>

#include "adapters/db_adapter.h"

namespace middleware {

namespace {
bool SafeIdentifier(const std::string &name) {
  return !name.empty() && name.find('"') == std::string::npos;
}
} // namespace

std::string DistinctCache::DefaultPath() {
  if (const char *env = std::getenv("AQP_DISTINCT_CACHE"))
    return env;
  return "/tmp/aqp_distinct.cache";
}

DistinctCache::DistinctCache(std::string cache_path)
    : path_(std::move(cache_path)) {
  Load();
}

DistinctCache::~DistinctCache() { Save(); }

void DistinctCache::Load() {
  if (path_.empty())
    return;
  std::ifstream in(path_);
  if (!in)
    return;
  std::string table, column;
  double distinct;
  while (std::getline(in, table, '\t') && std::getline(in, column, '\t') &&
         (in >> distinct)) {
    in.ignore(); // trailing newline
    map_[table + "\t" + column] = distinct;
  }
}

void DistinctCache::Save() {
  if (path_.empty() || !dirty_)
    return;
  std::ofstream out(path_, std::ios::trunc);
  if (!out)
    return;
  for (const auto &kv : map_) {
    if (kv.second > 0.0) // failure sentinels stay in-memory only
      out << kv.first << '\t' << kv.second << '\n';
  }
  dirty_ = false;
}

double DistinctCache::Get(EngineAdapter &adapter, const std::string &table,
                          const std::string &column) {
  const std::string key = table + "\t" + column;
  auto it = map_.find(key);
  if (it != map_.end())
    return it->second;

  double distinct = -1.0;
  if (SafeIdentifier(table) && SafeIdentifier(column)) {
    try {
      std::string sql;
      if (adapter.GetEngineName() == "PostgreSQL") {
        sql = "SELECT CASE WHEN s.n_distinct >= 0 THEN s.n_distinct "
              "ELSE -s.n_distinct * c.reltuples END "
              "FROM pg_stats s JOIN pg_class c ON c.relname = s.tablename "
              "WHERE s.tablename='" + table + "' AND s.attname='" +
              column + "';";
      } else {
        sql = "SELECT COUNT(DISTINCT \"" + column + "\") FROM \"" + table +
              "\";";
      }
      auto result = adapter.ExecuteSQL(sql);
      if (result.num_rows >= 1 && !result.rows.empty() &&
          !result.rows[0].empty())
        distinct = std::strtod(result.rows[0][0].c_str(), nullptr);
    } catch (...) {
      distinct = -1.0;
    }
  }

  // Cache failures too (as <= 0) so a broken lookup is not retried per query;
  // only successful counts are worth persisting.
  map_[key] = distinct;
  if (distinct > 0.0)
    dirty_ = true;
  return distinct;
}

double DistinctCache::GetCorrelation(EngineAdapter &adapter,
                                     const std::string &table,
                                     const std::string &column) {
  const std::string key = table + "\tcorr:" + column;
  auto it = map_.find(key);
  if (it != map_.end())
    return it->second;

  double corr = -1.0;
  if (SafeIdentifier(table) && SafeIdentifier(column)) {
    try {
      std::string sql;
      if (adapter.GetEngineName() == "PostgreSQL") {
        sql = "SELECT abs(correlation) FROM pg_stats WHERE tablename='" +
              table + "' AND attname='" + column + "';";
      } else {
        sql = "SELECT abs(corr(rowid, \"" + column + "\")) FROM \"" + table +
              "\";";
      }
      auto result = adapter.ExecuteSQL(sql);
      if (result.num_rows >= 1 && !result.rows.empty() &&
          !result.rows[0].empty())
        corr = std::strtod(result.rows[0][0].c_str(), nullptr);
    } catch (...) {
      corr = -1.0;
    }
  }

  map_[key] = corr;
  if (corr > 0.0)
    dirty_ = true;
  return corr;
}

double DistinctCache::GetRowCount(EngineAdapter &adapter,
                                  const std::string &table) {
  const std::string key = table + "\trows:";
  auto it = map_.find(key);
  if (it != map_.end())
    return it->second;

  double rows = -1.0;
  if (SafeIdentifier(table)) {
    try {
      std::string sql;
      if (adapter.GetEngineName() == "PostgreSQL") {
        sql = "SELECT reltuples FROM pg_class WHERE relname='" + table + "';";
      } else {
        sql = "SELECT COUNT(*) FROM \"" + table + "\";";
      }
      auto result = adapter.ExecuteSQL(sql);
      if (result.num_rows >= 1 && !result.rows.empty() &&
          !result.rows[0].empty())
        rows = std::strtod(result.rows[0][0].c_str(), nullptr);
    } catch (...) {
      rows = -1.0;
    }
  }

  map_[key] = rows;
  if (rows > 0.0)
    dirty_ = true;
  return rows;
}

double DistinctCache::GetCached(const std::string &table,
                                const std::string &column) const {
  auto it = map_.find(table + "\t" + column);
  return it != map_.end() ? it->second : -1.0;
}

double DistinctCache::GetCachedRowCount(const std::string &table) const {
  auto it = map_.find(table + "\trows:");
  return it != map_.end() ? it->second : -1.0;
}

std::map<std::string, double> DistinctCache::GetAllCachedRowCounts() const {
  std::map<std::string, double> out;
  const std::string suffix = "\trows:";
  for (const auto &kv : map_) {
    if (kv.second > 0.0 && kv.first.size() > suffix.size() &&
        kv.first.compare(kv.first.size() - suffix.size(), suffix.size(),
                         suffix) == 0) {
      out[kv.first.substr(0, kv.first.size() - suffix.size())] = kv.second;
    }
  }
  return out;
}

} // namespace middleware
