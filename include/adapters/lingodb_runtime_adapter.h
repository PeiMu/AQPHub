#pragma once

#include "adapters/lingodb_adapter.h"

namespace middleware {

class LingoDBRuntimeAdapter : public LingoDBAdapter {
public:
  explicit LingoDBRuntimeAdapter(const std::string &db_path);
  ~LingoDBRuntimeAdapter() override = default;

  std::string GetEngineName() const override { return "LingoDB-Runtime"; }

  void ExecuteIRandCreateTempTable(
      ir_sql_converter::AQPStmt &ir,
      const std::string &temp_table_name,
      bool update_temp_card) override;

  QueryResult ExecuteIRQuery(ir_sql_converter::AQPStmt &ir) override;
};

} // namespace middleware
