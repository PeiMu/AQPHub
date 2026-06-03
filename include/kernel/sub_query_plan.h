#pragma once

#include "adapters/db_adapter.h"
#include "storage/csr_index.h"
#include "storage/flat_table.h"
#include "storage/sorted_index.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declarations insufficient for unscoped enums; include the IR header
#include "simplest_ir.h"

namespace middleware {
namespace storage {

class StoragePlan;
class DimensionCache;
class InvertedIndex;

// A compiled filter predicate that can be evaluated against a FlatTable row
using RowPredicate = std::function<bool(const FlatTable &table, uint64_t row)>;

// ============================================================================
// Shared kernel analysis types (used by both query kernel and pipeline kernel)
// ============================================================================

struct LeafTable {
  std::string name;
  unsigned int ir_table_index = 0;
  bool is_base = false;
  const FlatTable *flat = nullptr;
  std::vector<const std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> *> all_filters;
  bool HasFilters() const;
};

struct JoinEdge {
  unsigned int left_table_idx = 0;
  unsigned int left_col_idx = 0;
  std::string left_col_name;
  unsigned int right_table_idx = 0;
  unsigned int right_col_idx = 0;
  std::string right_col_name;
};

// Result of dimension/inverted-index resolution
struct DimResolutionResult {
  std::unordered_map<unsigned int,
                     std::vector<std::pair<int, std::vector<int32_t>>>>
      dim_derived_filters;
  std::unordered_map<uint64_t, std::pair<unsigned int, std::string>>
      inv_col_remap;
};

// ============================================================================
// Shared helper functions
// ============================================================================

void CollectLeaves(
    const ir_sql_converter::AQPStmt *node,
    std::vector<LeafTable> &leaves,
    const StoragePlan *storage_plan,
    const std::unordered_map<std::string, const FlatTable *> &kernel_temps,
    bool &has_aggregate,
    bool &has_unattached_filter);

void CollectJoinEdges(
    const ir_sql_converter::AQPStmt *node,
    std::vector<JoinEdge> &edges,
    ir_sql_converter::SimplestJoinType &join_type);

const LeafTable *FindLeaf(const std::vector<LeafTable> &leaves,
                          unsigned int table_idx);

RowPredicate CompileOnePredicate(const ir_sql_converter::AQPExpr *expr,
                                 const FlatTable *table);

bool CompileAllLeafFilters(
    const std::vector<const std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> *> &all_filters,
    const FlatTable *table,
    std::vector<RowPredicate> &out);

bool BuildFilteredPKBitset(const FlatTable *dim_table,
                           const std::vector<RowPredicate> &filters,
                           std::vector<bool> &pk_bitset,
                           std::vector<uint32_t> &pk_to_row);

bool LeafHasLikeFilter(const LeafTable *leaf);

DimResolutionResult ResolveDimensions(
    std::vector<LeafTable> &leaves,
    std::vector<JoinEdge> &edges,
    const ir_sql_converter::AQPStmt *sub_ir,
    const StoragePlan *storage_plan,
    const DimensionCache *dim_cache);

void AddDimDerivedFilters(
    const std::unordered_map<unsigned int,
                             std::vector<std::pair<int, std::vector<int32_t>>>> &dim_derived_filters,
    unsigned int table_idx,
    std::vector<RowPredicate> &out);

// ============================================================================
// FlatTableBuilder — per-thread row accumulator for kernel output
// ============================================================================

struct FlatTableBuilder {
  struct ColBuffer {
    FlatColumnType type = FlatColumnType::INT32;
    std::vector<int32_t> int_data;
    std::vector<std::string> str_data;
  };

  std::vector<std::string> column_names;
  std::vector<ColBuffer> col_buffers;
  uint64_t row_count = 0;

  void Init(const std::vector<struct KernelOutputCol> &output_cols);
  void Reserve(uint64_t est_rows);
  void AppendInt(size_t col, int32_t val);
  void AppendStr(size_t col, const char *ptr, uint32_t len);
  void FinishRow();
  std::unique_ptr<FlatTable> Finalize(const std::string &table_name);
};

FlatTableBuilder MergeBuilders(std::vector<FlatTableBuilder> &builders);

// ============================================================================
// Query kernel data structures
// ============================================================================

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

// ---------------------------------------------------------------------------
// LIKE pattern classification — shared by sub_query_plan.cpp and
// pipeline_kernel.cpp (CompileOneFilterDesc).
// ---------------------------------------------------------------------------
enum LikePatternKind {
  LIKE_COMPLEX = 0,
  LIKE_EQUALITY,
  LIKE_PREFIX,
  LIKE_SUFFIX,
  LIKE_CONTAINS,
  LIKE_MULTI_SEGMENT
};

struct LikeSegments {
  std::vector<std::string> segs;
  bool has_leading_pct = false;
  bool has_trailing_pct = false;
};

inline LikePatternKind ClassifyLikePattern(const std::string &pattern,
                                           std::string &literal_out) {
  if (pattern.empty()) { literal_out.clear(); return LIKE_EQUALITY; }
  if (pattern.find('_') != std::string::npos) return LIKE_COMPLEX;
  size_t leading = 0;
  while (leading < pattern.size() && pattern[leading] == '%') ++leading;
  size_t trailing = 0;
  while (trailing < pattern.size() &&
         pattern[pattern.size() - 1 - trailing] == '%') ++trailing;
  size_t mid_start = leading, mid_end = pattern.size() - trailing;
  for (size_t i = mid_start; i < mid_end; ++i)
    if (pattern[i] == '%') return LIKE_COMPLEX;
  literal_out = pattern.substr(mid_start, mid_end - mid_start);
  if (leading == 0 && trailing == 0) return LIKE_EQUALITY;
  if (leading == 0 && trailing > 0) return LIKE_PREFIX;
  if (leading > 0 && trailing == 0) return LIKE_SUFFIX;
  return LIKE_CONTAINS;
}

inline LikePatternKind ClassifyLikePatternEx(const std::string &pattern,
                                             std::string &literal_out,
                                             LikeSegments &seg_out) {
  LikePatternKind k = ClassifyLikePattern(pattern, literal_out);
  if (k != LIKE_COMPLEX) return k;
  if (pattern.find('_') != std::string::npos) return LIKE_COMPLEX;
  seg_out.has_leading_pct = (!pattern.empty() && pattern[0] == '%');
  seg_out.has_trailing_pct = (!pattern.empty() && pattern.back() == '%');
  seg_out.segs.clear();
  std::string cur;
  for (char c : pattern) {
    if (c == '%') {
      if (!cur.empty()) { seg_out.segs.push_back(cur); cur.clear(); }
    } else { cur += c; }
  }
  if (!cur.empty()) seg_out.segs.push_back(cur);
  if (seg_out.segs.size() >= 2) return LIKE_MULTI_SEGMENT;
  return LIKE_COMPLEX;
}

} // namespace storage
} // namespace middleware
