#pragma once

#include "adapters/db_adapter.h"
#include "storage/csr_index.h"
#include "storage/flat_table.h"
#include "storage/sorted_index.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ir_sql_converter {
class AQPStmt;
}

namespace middleware {
namespace storage {

class StoragePlan;
class DimensionCache;

// A compiled filter predicate that can be evaluated against a FlatTable row
using RowPredicate = std::function<bool(const FlatTable &table, uint64_t row)>;

struct KernelJoinStep {
  const CSRIndex *csr = nullptr;
  int scan_key_col_idx = -1;
  const FlatTable *joined_table = nullptr;
  bool is_semi = true;

  // For dim table patterns: bitset of valid PK values (pre-filtered)
  std::vector<bool> pk_bitset;
  // pk_to_row[pk_val] = row index in joined_table (for inner join with bitset)
  std::vector<uint32_t> pk_to_row;
  bool use_bitset = false;

  // Per-row filters on joined (lookup) table rows during CSR traversal
  std::vector<RowPredicate> join_filters;
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
  std::vector<RowPredicate> scan_filters; // filters on the scan table
  bool valid = false;
};

SubQueryPlan AnalyzeSubIR(
    const ir_sql_converter::AQPStmt *sub_ir,
    const StoragePlan *storage_plan,
    const std::unordered_map<std::string, const FlatTable *> &kernel_temps,
    const std::unordered_map<std::string, CSRIndex> &runtime_csrs,
    const DimensionCache *dim_cache = nullptr);

std::unique_ptr<FlatTable> ExecuteSubQueryPlan(const SubQueryPlan &plan,
                                               const std::string &table_name);

struct MinColumnInfo {
  int output_idx = -1;
  unsigned int ir_table_idx = 0;
  int flat_col_idx = -1;
  FlatColumnType type = FlatColumnType::INT32;
  const SortedIndex *sorted = nullptr;
  const FlatTable *table = nullptr;
  bool on_scan_table = false;
  std::string name;
};

struct FinalAggregatePlan {
  std::vector<MinColumnInfo> min_cols;
  const FlatTable *scan_table = nullptr;
  std::string scan_table_name;
  std::vector<KernelJoinStep> join_steps;
  std::vector<RowPredicate> scan_filters;
  std::vector<std::string> output_names;
  bool valid = false;
};

FinalAggregatePlan AnalyzeFinalIR(
    const ir_sql_converter::AQPStmt *ir,
    const StoragePlan *storage_plan,
    const std::unordered_map<std::string, const FlatTable *> &kernel_temps,
    const std::unordered_map<std::string, CSRIndex> &runtime_csrs,
    const DimensionCache *dim_cache = nullptr);

QueryResult ExecuteFinalAggregate(const FinalAggregatePlan &plan);

} // namespace storage
} // namespace middleware
