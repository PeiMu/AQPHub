#pragma once
#include "adapters/postgres_adapter.h"

namespace middleware {
class UmbraAdapter : public PostgreSQLAdapter {
public:
  explicit UmbraAdapter(const std::string &connection_string);
  ~UmbraAdapter() override = default;

  std::string GetEngineName() const override { return "Umbra"; }

  // Skip ANALYZE on temp tables (Umbra auto-collects stats)
  void ExecuteSQLandCreateTempTable(const std::string &sql,
                                    const std::string &temp_table_name,
                                    bool update_temp_card,
                                    bool enable_timing) override;

  void SetTempTableCardinality(const std::string &temp_table_name,
                               uint64_t estimated_rows) override;

  // evaluate multiple EXPLAIN queries in one round-trip
  // Default implementation calls GetEstimatedCost sequentially (fine for
  // in-process engines like DuckDB; overridden for network-based engines)
  std::vector<std::pair<double, double>>
  BatchGetEstimatedCosts(const std::vector<std::string> &sqls) override;

  // Umbra auto-populates pg_class.reltuples — cheaper than COUNT(*)
  uint64_t GetTempTableCardinality(const std::string &temp_table_name) override;

  // Umbra EXPLAIN JSON: {"plan":{"cardinality":N,...}} (no "Total Cost")
  std::pair<double, double> GetEstimatedCost(const std::string &sql) override;
};
} // namespace middleware
