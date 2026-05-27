#pragma once

#include "storage/csr_index.h"
#include "storage/flat_table.h"
#include "util/param_config.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace ir_sql_converter {
class AQPStmt;
}

namespace middleware {
namespace storage {

class StoragePlan;

struct KernelJoinStep {
  const CSRIndex *csr = nullptr;
  int scan_key_col_idx = -1;
  const FlatTable *joined_table = nullptr;
  bool is_semi = true;

  // For dim table patterns: bitset of valid PK values (pre-filtered)
  std::vector<bool> pk_bitset;
  bool use_bitset = false;
};

struct KernelOutputCol {
  enum Source { FROM_SCAN, FROM_JOIN };
  Source source = FROM_SCAN;
  int step_idx = -1;
  int col_idx = -1;
  FlatColumnType type = FlatColumnType::INT32;
  std::string name;
};

struct SubQueryPlan {
  const FlatTable *scan_table = nullptr;
  std::string scan_table_name;
  std::vector<KernelJoinStep> join_steps;
  std::vector<KernelOutputCol> output_cols;
  bool valid = false;
};

SubQueryPlan AnalyzeSubIR(
    const ir_sql_converter::AQPStmt *sub_ir,
    CsrSupportLevel level,
    const StoragePlan *storage_plan,
    const std::unordered_map<std::string, const FlatTable *> &kernel_temps,
    const std::unordered_map<std::string, CSRIndex> &runtime_csrs);

std::unique_ptr<FlatTable> ExecuteSubQueryPlan(const SubQueryPlan &plan,
                                               const std::string &table_name);

} // namespace storage
} // namespace middleware
