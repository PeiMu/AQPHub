#pragma once

#include "jit/aqp_jit_abi.h"
#include "kernel/sub_query_plan.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace middleware {
namespace storage {

class StoragePlan;
class DimensionCache;

// Lightweight filter descriptor for JIT compilation.
// Captured from AQP IR expressions during AnalyzePipelineKernel,
// parallel to the std::function RowPredicates.
struct PipelineFilterDesc {
  enum Kind {
    INT32_EQ, INT32_NE, INT32_LT, INT32_GT, INT32_LE, INT32_GE,
    INT32_IN_SET,
    VARCHAR_EQ, VARCHAR_LIKE_PREFIX, VARCHAR_LIKE_SUFFIX,
    VARCHAR_LIKE_CONTAINS, VARCHAR_LIKE_EXACT,
    VARCHAR_LIKE_SEGMENTS, VARCHAR_LIKE_FULL, VARCHAR_IN_SET,
    IS_NULL, IS_NOT_NULL,
    LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT,
    UNSUPPORTED
  };
  Kind kind = UNSUPPORTED;
  int col_idx = -1;
  FlatColumnType col_type = FlatColumnType::INT32;
  bool nullable = false;
  int32_t int_const = 0;
  std::string str_const;
  std::vector<int32_t> int_set;
  std::vector<std::string> str_set;
  bool negated = false;
  std::vector<PipelineFilterDesc> children;
};

struct PipelineJoinStep {
  const FlatTable *build_table = nullptr;
  int build_key_col = -1;
  int scan_key_col = -1;
  // -1 = probe key from scan table, >=0 = key from step[idx]'s build table
  int probe_step_idx = -1;
  int probe_key_col = -1;
  bool is_semi = true;
  std::vector<RowPredicate> build_filters;
  std::vector<RowPredicate> join_filters;
  std::vector<PipelineFilterDesc> build_filter_descs;
  std::vector<PipelineFilterDesc> join_filter_descs;
  // Hash table built at execution time (opaque, defined in .cpp)
  struct HashJoinTable;
  std::unique_ptr<HashJoinTable> ht;

  // Accessors for JIT — returns raw pointers to HT internals
  const uint32_t *HtBuckets() const;
  const uint32_t *HtNext() const;
  const int32_t  *HtKeys() const;
  const uint32_t *HtRowIds() const;
  uint32_t HtMask() const;
  uint32_t HtSize() const;
  const uint64_t *HtBloomData() const;
  uint64_t HtBloomMask() const;

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
  std::vector<PipelineFilterDesc> scan_filter_descs;
  AQPPipelineKernelFn compiled_fn = nullptr;
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
