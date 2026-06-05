#pragma once

#include "adapters/db_adapter.h"
#include "pg_query.h"
#include <nlohmann/json.hpp>

#include <lingodb/catalog/Catalog.h>
#include <lingodb/runtime/Session.h>

namespace arrow { class Table; }

namespace middleware {

class LingoDBAdapter : public EngineAdapter {
public:
  explicit LingoDBAdapter(const std::string &db_path);
  ~LingoDBAdapter() override;

  void ParseSQL(const std::string &sql) override;
  std::unique_ptr<ir_sql_converter::AQPStmt> ConvertPlanToIR() override;

  QueryResult ExecuteSQL(const std::string &sql) override;
  void ExecuteSQLandCreateTempTable(const std::string &sql,
                                    const std::string &temp_table_name,
                                    bool update_temp_card) override;

  void CreateTempTable(const std::string &table_name,
                       const QueryResult &result) override;
  void DropTempTable(const std::string &table_name) override;
  bool TempTableExists(const std::string &table_name) override;

  uint64_t
  GetTempTableCardinality(const std::string &temp_table_name) override;
  void SetTempTableCardinality(const std::string &temp_table_name,
                               uint64_t cardinality) override;

  std::pair<double, double> GetEstimatedCost(const std::string &sql) override;

  std::string GetEngineName() const override { return "LingoDB"; }

  void CleanUp() override;

  void LoadTablesFromCSV(const std::string &schema_path,
                         const std::string &csv_dir);

private:
  std::shared_ptr<lingodb::runtime::Session> session_;
  nlohmann::json parse_tree_;
  bool scheduler_started_ = false;

  QueryResult ExecuteSingleSQL(const std::string &sql);

  void CreateTempTableFromArrow(const std::string &table_name,
                                std::shared_ptr<arrow::Table> table);
};

} // namespace middleware
