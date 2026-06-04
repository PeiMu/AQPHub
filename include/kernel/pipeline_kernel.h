#pragma once

#include "kernel/sub_query_plan.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace middleware {
namespace storage {

class StoragePlan;
class DimensionCache;

enum class ProbeMethod : uint8_t {
  SKIP,    // HT size 0: no matches possible
  POINT,   // HT size 1: direct key compare, no hash
  LINEAR,  // HT size 2-15: linear scan of keys array
  HASH,    // HT size 16+: chained hash probe
  DIRECT,  // small key range: array[key - min] lookup
};

struct PipelineJoinStep {
  const FlatTable *build_table = nullptr;
  int build_key_col = -1;
  int scan_key_col = -1;
  // -1 = probe key from scan table, >=0 = key from step[idx]'s build table
  int probe_step_idx = -1;
  int probe_key_col = -1;
  bool is_semi = true;
  bool build_key_unique = false;
  std::vector<RowPredicate> build_filters;
  std::vector<RowPredicate> join_filters;
  // Hash table built at execution time (opaque, defined in .cpp)
  struct HashJoinTable;
  std::unique_ptr<HashJoinTable> ht;
  ProbeMethod probe_method = ProbeMethod::HASH;

  uint32_t HtSize() const;

  PipelineJoinStep();
  ~PipelineJoinStep();
  PipelineJoinStep(PipelineJoinStep &&) noexcept;
  PipelineJoinStep &operator=(PipelineJoinStep &&) noexcept;
};

struct PipelineKernelPlan {
  const FlatTable *scan_table = nullptr;
  std::string scan_table_name;
  std::vector<PipelineJoinStep> join_steps;
  std::vector<KernelOutputCol> output_cols;
  std::vector<RowPredicate> scan_filters;
  bool valid = false;
};

PipelineKernelPlan AnalyzePipelineKernel(
    const ir_sql_converter::AQPStmt *sub_ir,
    const StoragePlan *storage_plan,
    const std::unordered_map<std::string, const FlatTable *> &kernel_temps,
    const DimensionCache *dim_cache = nullptr);

std::unique_ptr<FlatTable> ExecutePipelineKernel(
    PipelineKernelPlan &plan,
    const std::string &table_name);

} // namespace storage
} // namespace middleware
