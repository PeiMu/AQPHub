#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "kernel/pipeline_kernel.h"
#include "storage/dimension_cache.h"
#include "storage/inverted_index.h"
#include "storage/storage_plan.h"
#include "simplest_ir.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <queue>
#include <unordered_set>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

using namespace ir_sql_converter;

extern "C" {
void aqp_pk_builder_append_int(void *builder_ptr, uint32_t col_idx, int32_t val) {
    auto *b = static_cast<middleware::storage::FlatTableBuilder *>(builder_ptr);
    b->AppendInt(col_idx, val);
}
void aqp_pk_builder_append_str(void *builder_ptr, uint32_t col_idx,
                                const char *ptr, uint32_t len) {
    auto *b = static_cast<middleware::storage::FlatTableBuilder *>(builder_ptr);
    b->AppendStr(col_idx, ptr, len);
}
void aqp_pk_builder_finish_row(void *builder_ptr) {
    auto *b = static_cast<middleware::storage::FlatTableBuilder *>(builder_ptr);
    b->FinishRow();
}
} // extern "C"

namespace middleware {
namespace storage {

static constexpr uint64_t OMP_PARALLEL_THRESHOLD = 10000;

// ============================================================================
// HashJoinTable — chained hash table with Fibonacci hashing
// ============================================================================

struct PipelineJoinStep::HashJoinTable {
  static constexpr uint32_t EMPTY = UINT32_MAX;

  std::vector<uint32_t> buckets_;
  std::vector<uint32_t> next_;
  std::vector<int32_t> keys_;
  std::vector<uint32_t> row_ids_;
  uint32_t mask_ = 0;
  uint32_t size_ = 0;

  // Bloom filter (populated when size_ > 64)
  std::vector<uint64_t> bloom_;
  uint64_t bf_mask_ = 0;

  void Build(const FlatTable &table, int key_col,
             const std::vector<RowPredicate> &filters) {
    uint64_t nrows = table.row_count;

    // First pass: count qualifying rows (with filters)
    if (filters.empty()) {
      size_ = static_cast<uint32_t>(nrows);
      keys_.resize(nrows);
      row_ids_.resize(nrows);
      const auto &col = table.columns[key_col];
      for (uint64_t i = 0; i < nrows; i++) {
        keys_[i] = col.GetInt32(i);
        row_ids_[i] = static_cast<uint32_t>(i);
      }
    } else {
      keys_.reserve(nrows);
      row_ids_.reserve(nrows);
      const auto &col = table.columns[key_col];
      for (uint64_t i = 0; i < nrows; i++) {
        bool pass = true;
        for (const auto &f : filters) {
          if (!f(table, i)) { pass = false; break; }
        }
        if (pass) {
          keys_.push_back(col.GetInt32(i));
          row_ids_.push_back(static_cast<uint32_t>(i));
        }
      }
      size_ = static_cast<uint32_t>(keys_.size());
    }

    if (size_ == 0) {
      mask_ = 0;
      return;
    }

    // Power-of-2 capacity at ~50% load factor
    uint32_t capacity = 1;
    while (capacity < size_ * 2)
      capacity <<= 1;
    mask_ = capacity - 1;

    buckets_.assign(capacity, EMPTY);
    next_.resize(size_);

    // Insert all entries
    for (uint32_t i = 0; i < size_; i++) {
      uint32_t h = (static_cast<uint32_t>(keys_[i]) * 2654435761u) & mask_;
      next_[i] = buckets_[h];
      buckets_[h] = i;
    }

    // Build bloom filter for larger HTs
    if (size_ > 64) {
      uint64_t bf_slots = 64;
      while (bf_slots < size_ * 8) bf_slots <<= 1;
      uint64_t bf_words = bf_slots / 64;
      bloom_.assign(bf_words, 0);
      bf_mask_ = bf_words - 1;
      for (uint32_t i = 0; i < size_; i++) {
        uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(keys_[i]) * 2654435761u);
        uint64_t sector = (h >> 6) & bf_mask_;
        uint64_t bit = h & 63;
        bloom_[sector] |= (1ULL << bit);
      }
    }
  }

  template <typename Fn> void ForEach(int32_t key, Fn &&fn) const {
    if (size_ == 0) return;
    uint32_t h = (static_cast<uint32_t>(key) * 2654435761u) & mask_;
    uint32_t idx = buckets_[h];
    while (idx != EMPTY) {
      if (keys_[idx] == key) {
        fn(row_ids_[idx]);
      }
      idx = next_[idx];
    }
  }

  bool Contains(int32_t key) const {
    if (size_ == 0) return false;
    uint32_t h = (static_cast<uint32_t>(key) * 2654435761u) & mask_;
    uint32_t idx = buckets_[h];
    while (idx != EMPTY) {
      if (keys_[idx] == key) return true;
      idx = next_[idx];
    }
    return false;
  }

};

constexpr uint32_t PipelineJoinStep::HashJoinTable::EMPTY;

// PipelineJoinStep special members (needed because HashJoinTable is incomplete in header)
PipelineJoinStep::PipelineJoinStep() = default;
PipelineJoinStep::~PipelineJoinStep() = default;
PipelineJoinStep::PipelineJoinStep(PipelineJoinStep &&) noexcept = default;
PipelineJoinStep &PipelineJoinStep::operator=(PipelineJoinStep &&) noexcept = default;

// HT accessor implementations for JIT access
const uint32_t *PipelineJoinStep::HtBuckets() const { return ht ? ht->buckets_.data() : nullptr; }
const uint32_t *PipelineJoinStep::HtNext() const { return ht ? ht->next_.data() : nullptr; }
const int32_t  *PipelineJoinStep::HtKeys() const { return ht ? ht->keys_.data() : nullptr; }
const uint32_t *PipelineJoinStep::HtRowIds() const { return ht ? ht->row_ids_.data() : nullptr; }
uint32_t PipelineJoinStep::HtMask() const { return ht ? ht->mask_ : 0; }
uint32_t PipelineJoinStep::HtSize() const { return ht ? ht->size_ : 0; }
const uint64_t *PipelineJoinStep::HtBloomData() const { return ht && !ht->bloom_.empty() ? ht->bloom_.data() : nullptr; }
uint64_t PipelineJoinStep::HtBloomMask() const { return ht ? ht->bf_mask_ : 0; }

// ============================================================================
// CompileFilterDesc — extract filter descriptor from AQP IR expression
// ============================================================================

static PipelineFilterDesc CompileOneFilterDesc(const AQPExpr *expr, const FlatTable *table) {
  PipelineFilterDesc desc;
  desc.kind = PipelineFilterDesc::UNSUPPORTED;
  if (!expr || !table) return desc;

  auto expr_type = expr->GetSimplestExprType();

  // IS NULL / IS NOT NULL
  if (expr_type == NullType || expr_type == NonNullType) {
    auto *isnull = static_cast<const SimplestIsNullExpr *>(expr);
    int col_idx = table->FindColumn(isnull->attr->GetColumnName());
    if (col_idx < 0) return desc;
    desc.kind = (expr_type == NullType) ? PipelineFilterDesc::IS_NULL : PipelineFilterDesc::IS_NOT_NULL;
    desc.col_idx = col_idx;
    desc.col_type = table->columns[col_idx].type;
    desc.nullable = table->columns[col_idx].nullable;
    return desc;
  }

  // VarConstComparison
  if (expr->GetNodeType() == VarConstComparisonNode) {
    auto *cmp = static_cast<const SimplestVarConstComparison *>(expr);
    int col_idx = table->FindColumn(cmp->attr->GetColumnName());
    if (col_idx < 0) return desc;
    auto col_type = table->columns[col_idx].type;
    auto cmp_type = cmp->GetSimplestExprType();
    auto var_type = cmp->const_var->GetType();

    desc.col_idx = col_idx;
    desc.col_type = col_type;
    desc.nullable = table->columns[col_idx].nullable;

    if (col_type == FlatColumnType::INT32 && var_type == IntVar) {
      desc.int_const = cmp->const_var->GetIntValue();
      switch (cmp_type) {
      case Equal:        desc.kind = PipelineFilterDesc::INT32_EQ; break;
      case NotEqual:     desc.kind = PipelineFilterDesc::INT32_NE; break;
      case LessThan:     desc.kind = PipelineFilterDesc::INT32_LT; break;
      case GreaterThan:  desc.kind = PipelineFilterDesc::INT32_GT; break;
      case LessEqual:    desc.kind = PipelineFilterDesc::INT32_LE; break;
      case GreaterEqual: desc.kind = PipelineFilterDesc::INT32_GE; break;
      default: break;
      }
      return desc;
    }

    if (col_type == FlatColumnType::VARCHAR && var_type == StringVar) {
      desc.str_const = cmp->const_var->GetStringValue();
      switch (cmp_type) {
      case Equal:
        desc.kind = PipelineFilterDesc::VARCHAR_EQ;
        return desc;
      case NotEqual:
        desc.kind = PipelineFilterDesc::VARCHAR_EQ;
        desc.negated = true;
        return desc;
      case TextLike:
      case Text_Not_Like: {
        desc.negated = (cmp_type == Text_Not_Like);
        std::string literal;
        LikeSegments seg_info;
        LikePatternKind lk = ClassifyLikePatternEx(desc.str_const, literal, seg_info);
        desc.str_const = literal;
        switch (lk) {
        case LIKE_EQUALITY:   desc.kind = PipelineFilterDesc::VARCHAR_LIKE_EXACT; break;
        case LIKE_PREFIX:     desc.kind = PipelineFilterDesc::VARCHAR_LIKE_PREFIX; break;
        case LIKE_SUFFIX:     desc.kind = PipelineFilterDesc::VARCHAR_LIKE_SUFFIX; break;
        case LIKE_CONTAINS:   desc.kind = PipelineFilterDesc::VARCHAR_LIKE_CONTAINS; break;
        case LIKE_MULTI_SEGMENT: desc.kind = PipelineFilterDesc::VARCHAR_LIKE_SEGMENTS;
          // Store original pattern for segments
          desc.str_const = cmp->const_var->GetStringValue();
          break;
        default:
          desc.kind = PipelineFilterDesc::VARCHAR_LIKE_FULL;
          desc.str_const = cmp->const_var->GetStringValue();
          break;
        }
        return desc;
      }
      default: break;
      }
    }
    return desc;
  }

  // IN expression
  if (expr->GetNodeType() == InExprNode) {
    auto *in_expr = static_cast<const SimplestInExpr *>(expr);
    int col_idx = table->FindColumn(in_expr->attr->GetColumnName());
    if (col_idx < 0) return desc;
    desc.col_idx = col_idx;
    desc.col_type = table->columns[col_idx].type;
    desc.nullable = table->columns[col_idx].nullable;
    desc.negated = in_expr->negated;

    if (desc.col_type == FlatColumnType::INT32) {
      desc.kind = PipelineFilterDesc::INT32_IN_SET;
      for (const auto &v : in_expr->values)
        if (v->GetType() == IntVar) desc.int_set.push_back(v->GetIntValue());
    } else if (desc.col_type == FlatColumnType::VARCHAR) {
      desc.kind = PipelineFilterDesc::VARCHAR_IN_SET;
      for (const auto &v : in_expr->values)
        if (v->GetType() == StringVar) desc.str_set.push_back(v->GetStringValue());
    }
    return desc;
  }

  // Logical AND / OR / NOT
  if (expr->GetNodeType() == LogicalExprNode) {
    auto *logic = static_cast<const SimplestLogicalExpr *>(expr);
    auto op = logic->GetLogicalOp();
    if (op == LogicalAnd) {
      desc.kind = PipelineFilterDesc::LOGICAL_AND;
      desc.children.push_back(CompileOneFilterDesc(logic->left_expr.get(), table));
      desc.children.push_back(CompileOneFilterDesc(logic->right_expr.get(), table));
    } else if (op == LogicalOr) {
      desc.kind = PipelineFilterDesc::LOGICAL_OR;
      desc.children.push_back(CompileOneFilterDesc(logic->left_expr.get(), table));
      desc.children.push_back(CompileOneFilterDesc(logic->right_expr.get(), table));
    } else if (op == LogicalNot) {
      desc.kind = PipelineFilterDesc::LOGICAL_NOT;
      desc.children.push_back(CompileOneFilterDesc(logic->right_expr.get(), table));
    }
    return desc;
  }

  return desc;
}

static void CompileAllFilterDescs(
    const std::vector<const std::vector<std::unique_ptr<AQPExpr>> *> &all_filters,
    const FlatTable *table,
    std::vector<PipelineFilterDesc> &out) {
  for (const auto *filter_group : all_filters) {
    for (const auto &expr : *filter_group) {
      out.push_back(CompileOneFilterDesc(expr.get(), table));
    }
  }
}

static void AddDimDerivedFilterDescs(
    const std::unordered_map<unsigned int,
                             std::vector<std::pair<int, std::vector<int32_t>>>> &dim_derived_filters,
    unsigned int table_idx,
    const FlatTable *table,
    std::vector<PipelineFilterDesc> &out) {
  auto it = dim_derived_filters.find(table_idx);
  if (it == dim_derived_filters.end())
    return;
  for (const auto &filt : it->second) {
    PipelineFilterDesc desc;
    desc.col_idx = filt.first;
    desc.col_type = (desc.col_idx >= 0 && desc.col_idx < (int)table->columns.size())
                    ? table->columns[desc.col_idx].type : FlatColumnType::INT32;
    const auto &pk_vals = filt.second;
    if (pk_vals.size() == 1) {
      desc.kind = PipelineFilterDesc::INT32_EQ;
      desc.int_const = pk_vals[0];
    } else {
      desc.kind = PipelineFilterDesc::INT32_IN_SET;
      desc.int_set = pk_vals;
    }
    out.push_back(std::move(desc));
  }
}

// ============================================================================
// AnalyzePipelineKernel
// ============================================================================

PipelineKernelPlan AnalyzePipelineKernel(
    const ir_sql_converter::AQPStmt *sub_ir,
    const StoragePlan *storage_plan,
    const std::unordered_map<std::string, const FlatTable *> &kernel_temps,
    const DimensionCache *dim_cache) {

  PipelineKernelPlan plan;
  plan.valid = false;

  if (!sub_ir || !storage_plan)
    return plan;

  // Collect leaf tables and join edges
  std::vector<LeafTable> leaves;
  bool has_aggregate = false;
  bool has_unattached_filter = false;
  CollectLeaves(sub_ir, leaves, storage_plan, kernel_temps,
                has_aggregate, has_unattached_filter);

  if (has_aggregate || has_unattached_filter)
    return plan;

  SimplestJoinType join_type = InvalidJoinType;
  std::vector<JoinEdge> edges;
  CollectJoinEdges(sub_ir, edges, join_type);

  if (!edges.empty() && join_type != Inner)
    return plan;

  // Dimension/inverted-index resolution
  auto dim_result = ResolveDimensions(leaves, edges, sub_ir, storage_plan, dim_cache);
  auto &dim_derived_filters = dim_result.dim_derived_filters;
  auto &inv_col_remap = dim_result.inv_col_remap;

  // All leaf tables must have FlatTable data
  for (const auto &leaf : leaves) {
    if (!leaf.flat)
      return plan;
  }

  // ====== Single-table filtered scan ======
  if (leaves.size() == 1 && edges.empty()) {
    const LeafTable *scan_leaf = &leaves[0];

    std::vector<RowPredicate> scan_predicates;
    std::vector<PipelineFilterDesc> scan_filter_descs;
    if (scan_leaf->HasFilters()) {
      if (!CompileAllLeafFilters(scan_leaf->all_filters, scan_leaf->flat,
                                 scan_predicates))
        return plan;
      CompileAllFilterDescs(scan_leaf->all_filters, scan_leaf->flat, scan_filter_descs);
    }

    // Guard: base table LIKE scans → DuckDB vectorized is faster unless
    // inverted-index PK filter drastically reduces rows.
    if (scan_leaf->is_base && LeafHasLikeFilter(scan_leaf)) {
      bool has_pk_filter = false;
      auto dim_it2 = dim_derived_filters.find(scan_leaf->ir_table_index);
      if (dim_it2 != dim_derived_filters.end()) {
        int pk_col = scan_leaf->flat->FindColumn("id");
        for (const auto &filt : dim_it2->second) {
          if (filt.first == pk_col) { has_pk_filter = true; break; }
        }
      }
      if (!has_pk_filter)
        return plan;
    }

    AddDimDerivedFilters(dim_derived_filters, scan_leaf->ir_table_index,
                         scan_predicates);
    AddDimDerivedFilterDescs(dim_derived_filters, scan_leaf->ir_table_index,
                             scan_leaf->flat, scan_filter_descs);

    const auto &target_list = sub_ir->target_list;
    if (target_list.empty())
      return plan;

    std::vector<KernelOutputCol> output_cols;
    for (const auto &attr : target_list) {
      unsigned int tbl_idx = attr->GetTableIndex();
      std::string col_name = attr->GetColumnName();

      if (tbl_idx != scan_leaf->ir_table_index && !inv_col_remap.empty()) {
        uint64_t remap_key = (static_cast<uint64_t>(tbl_idx) << 32) |
                             std::hash<std::string>{}(col_name);
        auto remap_it = inv_col_remap.find(remap_key);
        if (remap_it != inv_col_remap.end()) {
          tbl_idx = remap_it->second.first;
          col_name = remap_it->second.second;
        }
      }

      if (tbl_idx != scan_leaf->ir_table_index)
        return plan;
      KernelOutputCol out;
      out.source = KernelOutputCol::FROM_SCAN;
      out.col_idx = scan_leaf->flat->FindColumn(col_name);
      if (out.col_idx < 0)
        return plan;
      out.type = scan_leaf->flat->columns[out.col_idx].type;
      out.name = col_name;
      output_cols.push_back(out);
    }

    plan.scan_table = scan_leaf->flat;
    plan.scan_table_name = scan_leaf->name;
    plan.output_cols = std::move(output_cols);
    plan.scan_filters = std::move(scan_predicates);
    plan.scan_filter_descs = std::move(scan_filter_descs);
    plan.valid = true;
    return plan;
  }

  // ====== Multi-table join with hash tables ======
  if (leaves.size() < 2 || edges.empty())
    return plan;

  // Build adjacency from join edges: edge_adj[leaf_idx] = {(other_leaf_idx, edge_idx)}
  // Map ir_table_index → leaf vector index
  std::unordered_map<unsigned int, size_t> ir_to_leaf;
  for (size_t i = 0; i < leaves.size(); i++)
    ir_to_leaf[leaves[i].ir_table_index] = i;

  struct AdjEntry { size_t other_leaf; size_t edge_idx; };
  std::vector<std::vector<AdjEntry>> adj(leaves.size());

  for (size_t ei = 0; ei < edges.size(); ei++) {
    auto lit = ir_to_leaf.find(edges[ei].left_table_idx);
    auto rit = ir_to_leaf.find(edges[ei].right_table_idx);
    if (lit == ir_to_leaf.end() || rit == ir_to_leaf.end())
      return plan;
    adj[lit->second].push_back({rit->second, ei});
    adj[rit->second].push_back({lit->second, ei});
  }

  // Pick scan table = largest by row count.
  // Build hash tables on smaller tables (temps from prior iterations).
  // This minimizes HT build cost: small HTs are cheap to build and fit in cache.
  size_t scan_leaf_idx = 0;
  uint64_t max_rows = 0;
  for (size_t i = 0; i < leaves.size(); i++) {
    if (leaves[i].flat->row_count > max_rows) {
      max_rows = leaves[i].flat->row_count;
      scan_leaf_idx = i;
    }
  }

  // BFS from scan table through join edges to order join steps
  std::vector<bool> visited(leaves.size(), false);
  visited[scan_leaf_idx] = true;

  struct BFSEntry { size_t leaf_idx; int parent_step; };
  std::queue<BFSEntry> bfs_queue;
  bfs_queue.push({scan_leaf_idx, -1});

  struct PendingStep {
    size_t build_leaf_idx;
    size_t edge_idx;
    int probe_step;  // -1 = scan table, >=0 = step index
    size_t probe_leaf_idx;
  };
  std::vector<PendingStep> pending_steps;

  // leaf_to_step[leaf_idx] = step index in pending_steps, or -1 for scan table
  std::vector<int> leaf_to_step(leaves.size(), -2); // -2 = not visited
  leaf_to_step[scan_leaf_idx] = -1; // scan table = probe source -1

  while (!bfs_queue.empty()) {
    BFSEntry cur = bfs_queue.front();
    bfs_queue.pop();

    for (const auto &ae : adj[cur.leaf_idx]) {
      if (visited[ae.other_leaf]) continue;
      visited[ae.other_leaf] = true;

      PendingStep ps;
      ps.build_leaf_idx = ae.other_leaf;
      ps.edge_idx = ae.edge_idx;
      ps.probe_step = cur.parent_step;
      ps.probe_leaf_idx = cur.leaf_idx;
      int this_step = static_cast<int>(pending_steps.size());
      leaf_to_step[ae.other_leaf] = this_step;
      pending_steps.push_back(ps);

      bfs_queue.push({ae.other_leaf, this_step});
    }
  }

  // Check all leaves are reachable
  for (size_t i = 0; i < leaves.size(); i++) {
    if (!visited[i]) return plan;
  }

  // Build PipelineJoinSteps
  const LeafTable *scan_leaf = &leaves[scan_leaf_idx];
  std::vector<PipelineJoinStep> join_steps;
  join_steps.resize(pending_steps.size());

  for (size_t si = 0; si < pending_steps.size(); si++) {
    const auto &ps = pending_steps[si];
    const auto &edge = edges[ps.edge_idx];
    const LeafTable *build_leaf = &leaves[ps.build_leaf_idx];
    const LeafTable *probe_leaf = &leaves[ps.probe_leaf_idx];

    // Determine which side of the edge is build vs probe
    std::string build_col, probe_col;
    if (edge.left_table_idx == build_leaf->ir_table_index) {
      build_col = edge.left_col_name;
      probe_col = edge.right_col_name;
    } else {
      build_col = edge.right_col_name;
      probe_col = edge.left_col_name;
    }

    int build_key = build_leaf->flat->FindColumn(build_col);
    if (build_key < 0) return plan;
    if (build_leaf->flat->columns[build_key].type != FlatColumnType::INT32)
      return plan;

    // Find probe key column
    int probe_key = -1;
    if (ps.probe_step == -1) {
      // Probe from scan table
      probe_key = scan_leaf->flat->FindColumn(probe_col);
      if (probe_key < 0) return plan;
      if (scan_leaf->flat->columns[probe_key].type != FlatColumnType::INT32)
        return plan;
    } else {
      // Probe from a previous step's build table
      const LeafTable *prev_build = &leaves[pending_steps[ps.probe_step].build_leaf_idx];
      probe_key = prev_build->flat->FindColumn(probe_col);
      if (probe_key < 0) return plan;
      if (prev_build->flat->columns[probe_key].type != FlatColumnType::INT32)
        return plan;
    }

    auto &step = join_steps[si];
    step.build_table = build_leaf->flat;
    step.build_key_col = build_key;
    step.scan_key_col = (ps.probe_step == -1) ? probe_key : -1;
    step.probe_step_idx = ps.probe_step;
    step.probe_key_col = (ps.probe_step >= 0) ? probe_key : -1;

    // Compile build-side filters
    if (build_leaf->HasFilters()) {
      if (!CompileAllLeafFilters(build_leaf->all_filters, build_leaf->flat,
                                 step.build_filters))
        return plan;
      CompileAllFilterDescs(build_leaf->all_filters, build_leaf->flat,
                            step.build_filter_descs);
    }

    // Add dim-derived filters on build table as build_filters
    AddDimDerivedFilters(dim_derived_filters, build_leaf->ir_table_index,
                         step.build_filters);
  }

  // Compile scan filters
  std::vector<RowPredicate> scan_predicates;
  std::vector<PipelineFilterDesc> scan_filter_descs_multi;
  if (scan_leaf->HasFilters()) {
    if (!CompileAllLeafFilters(scan_leaf->all_filters, scan_leaf->flat,
                               scan_predicates))
      return plan;
    CompileAllFilterDescs(scan_leaf->all_filters, scan_leaf->flat,
                          scan_filter_descs_multi);
  }
  AddDimDerivedFilters(dim_derived_filters, scan_leaf->ir_table_index,
                       scan_predicates);
  AddDimDerivedFilterDescs(dim_derived_filters, scan_leaf->ir_table_index,
                           scan_leaf->flat, scan_filter_descs_multi);

  // Determine output columns
  const auto &target_list = sub_ir->target_list;
  if (target_list.empty())
    return plan;

  // Determine semi vs inner: if any output col comes from a build table, that step is inner
  std::vector<bool> step_needs_inner(pending_steps.size(), false);

  std::vector<KernelOutputCol> output_cols;
  for (const auto &attr : target_list) {
    unsigned int tbl_idx = attr->GetTableIndex();
    std::string col_name = attr->GetColumnName();

    // Apply inverted index column remapping
    if (!inv_col_remap.empty()) {
      uint64_t remap_key = (static_cast<uint64_t>(tbl_idx) << 32) |
                           std::hash<std::string>{}(col_name);
      auto remap_it = inv_col_remap.find(remap_key);
      if (remap_it != inv_col_remap.end()) {
        tbl_idx = remap_it->second.first;
        col_name = remap_it->second.second;
      }
    }

    KernelOutputCol out;
    out.name = col_name;

    if (tbl_idx == scan_leaf->ir_table_index) {
      out.source = KernelOutputCol::FROM_SCAN;
      out.col_idx = scan_leaf->flat->FindColumn(col_name);
      if (out.col_idx < 0) return plan;
      out.type = scan_leaf->flat->columns[out.col_idx].type;
    } else {
      // Find which join step has this table as build table
      auto leaf_it = ir_to_leaf.find(tbl_idx);
      if (leaf_it == ir_to_leaf.end()) return plan;
      int step_idx = leaf_to_step[leaf_it->second];
      if (step_idx < 0) return plan; // shouldn't happen if leaf is not scan

      out.source = KernelOutputCol::FROM_JOIN;
      out.step_idx = step_idx;
      const FlatTable *build_t = leaves[pending_steps[step_idx].build_leaf_idx].flat;
      out.col_idx = build_t->FindColumn(col_name);
      if (out.col_idx < 0) return plan;
      out.type = build_t->columns[out.col_idx].type;
      step_needs_inner[step_idx] = true;
    }
    output_cols.push_back(out);
  }

  // Mark semi vs inner
  for (size_t si = 0; si < join_steps.size(); si++) {
    join_steps[si].is_semi = !step_needs_inner[si];
  }

  plan.scan_table = scan_leaf->flat;
  plan.scan_table_name = scan_leaf->name;
  plan.join_steps = std::move(join_steps);
  plan.output_cols = std::move(output_cols);
  plan.scan_filters = std::move(scan_predicates);
  plan.scan_filter_descs = std::move(scan_filter_descs_multi);
  plan.valid = true;
  return plan;
}

// ============================================================================
// ExecutePipelineKernel
// ============================================================================

// Helper: get probe key for a join step given the current scan row and
// matched build rows from prior steps.
static inline int32_t GetProbeKey(
    const PipelineKernelPlan &plan,
    const PipelineJoinStep &step,
    uint64_t scan_row,
    const uint32_t *matched_rows) {
  if (step.scan_key_col >= 0)
    return plan.scan_table->columns[step.scan_key_col].GetInt32(scan_row);
  const auto &prev = plan.join_steps[step.probe_step_idx];
  return prev.build_table->columns[step.probe_key_col].GetInt32(
      matched_rows[step.probe_step_idx]);
}

// Helper: check join filters on a build row (returns true if passes).
static inline bool PassJoinFilters(
    const PipelineJoinStep &step,
    uint32_t build_row) {
  for (const auto &jf : step.join_filters) {
    if (!jf(*step.build_table, build_row)) return false;
  }
  return true;
}

// Helper: emit one output row into builder.
static inline void EmitOutputRow(
    const PipelineKernelPlan &plan,
    uint64_t scan_row,
    const uint32_t *matched_rows,
    FlatTableBuilder &builder) {
  for (size_t c = 0; c < plan.output_cols.size(); c++) {
    const auto &out = plan.output_cols[c];
    const FlatTable *src_table;
    uint64_t src_row;
    if (out.source == KernelOutputCol::FROM_SCAN) {
      src_table = plan.scan_table;
      src_row = scan_row;
    } else {
      src_table = plan.join_steps[out.step_idx].build_table;
      src_row = matched_rows[out.step_idx];
    }
    const auto &col = src_table->columns[out.col_idx];
    if (col.type == FlatColumnType::INT32)
      builder.AppendInt(c, col.GetInt32(src_row));
    else {
      uint32_t len;
      const char *ptr = col.GetVarchar(src_row, len);
      builder.AppendStr(c, ptr, len);
    }
  }
  builder.FinishRow();
}

// Iterative depth-first probe through join steps, no std::function.
// Uses an explicit stack to avoid per-row heap allocations.
static void ProbeStepsIterative(
    const PipelineKernelPlan &plan,
    uint64_t scan_row,
    uint32_t *matched_rows,
    FlatTableBuilder &builder) {

  const size_t nsteps = plan.join_steps.size();

  // Stack frame: (step_idx, hash chain cursor).
  // We use a fixed-size stack since nsteps is typically small (< 16).
  struct Frame {
    uint32_t chain_cursor;  // current position in hash chain (EMPTY = done)
    bool is_semi_done;      // for semi: already passed, just continue
  };
  Frame stack[16];
  if (nsteps > 16) return;  // safety guard

  // Initialize: push step 0
  size_t depth = 0;

  auto StartStep = [&](size_t step_idx) -> bool {
    const auto &step = plan.join_steps[step_idx];
    int32_t key = GetProbeKey(plan, step, scan_row, matched_rows);

    if (step.is_semi) {
      if (step.join_filters.empty()) {
        if (!step.ht->Contains(key)) return false;
      } else {
        bool any_match = false;
        step.ht->ForEach(key, [&](uint32_t build_row) {
          if (!any_match && PassJoinFilters(step, build_row))
            any_match = true;
        });
        if (!any_match) return false;
      }
      stack[step_idx].is_semi_done = true;
      stack[step_idx].chain_cursor = PipelineJoinStep::HashJoinTable::EMPTY;
      return true;
    }

    // Inner join: start iterating the hash chain
    if (step.ht->size_ == 0) return false;
    uint32_t h = (static_cast<uint32_t>(key) * 2654435761u) & step.ht->mask_;
    stack[step_idx].chain_cursor = step.ht->buckets_[h];
    stack[step_idx].is_semi_done = false;

    // Advance to first matching entry
    while (stack[step_idx].chain_cursor != PipelineJoinStep::HashJoinTable::EMPTY) {
      uint32_t idx = stack[step_idx].chain_cursor;
      if (step.ht->keys_[idx] == key) {
        uint32_t build_row = step.ht->row_ids_[idx];
        if (step.join_filters.empty() || PassJoinFilters(step, build_row)) {
          matched_rows[step_idx] = build_row;
          return true;
        }
      }
      stack[step_idx].chain_cursor = step.ht->next_[idx];
    }
    return false;  // no match found
  };

  // Advance to next matching entry in an inner join chain (for backtracking).
  auto AdvanceStep = [&](size_t step_idx) -> bool {
    const auto &step = plan.join_steps[step_idx];
    if (step.is_semi) return false;  // semi has no iteration

    int32_t key = GetProbeKey(plan, step, scan_row, matched_rows);
    // Move past current entry
    stack[step_idx].chain_cursor = step.ht->next_[stack[step_idx].chain_cursor];

    while (stack[step_idx].chain_cursor != PipelineJoinStep::HashJoinTable::EMPTY) {
      uint32_t idx = stack[step_idx].chain_cursor;
      if (step.ht->keys_[idx] == key) {
        uint32_t build_row = step.ht->row_ids_[idx];
        if (step.join_filters.empty() || PassJoinFilters(step, build_row)) {
          matched_rows[step_idx] = build_row;
          return true;
        }
      }
      stack[step_idx].chain_cursor = step.ht->next_[idx];
    }
    return false;
  };

  // Iterative DFS
  if (!StartStep(0)) return;
  depth = 0;

  for (;;) {
    // Try to go deeper
    if (depth + 1 < nsteps) {
      if (StartStep(depth + 1)) {
        depth++;
        continue;
      }
      // Can't go deeper — fall through to backtrack at current depth
    } else {
      // All steps matched — emit row
      EmitOutputRow(plan, scan_row, matched_rows, builder);
    }

    // Backtrack: advance current inner join step, or pop
    for (;;) {
      if (!plan.join_steps[depth].is_semi && AdvanceStep(depth)) {
        break;  // found next match at this depth, re-descend
      }
      if (depth == 0) return;  // fully exhausted
      depth--;
    }
  }
}

std::unique_ptr<FlatTable> ExecutePipelineKernel(
    PipelineKernelPlan &plan,
    const std::string &table_name) {
  assert(plan.valid);

  // Phase 1: Build hash tables for each join step
  for (auto &step : plan.join_steps) {
    step.ht = std::make_unique<PipelineJoinStep::HashJoinTable>();
    step.ht->Build(*step.build_table, step.build_key_col, step.build_filters);
  }

  uint64_t scan_rows = plan.scan_table->row_count;
  const bool has_scan_filters = !plan.scan_filters.empty();
  const size_t nsteps = plan.join_steps.size();

  // Check if any step needs inner join
  bool any_inner = false;
  for (const auto &step : plan.join_steps) {
    if (!step.is_semi) { any_inner = true; break; }
  }

  // === JIT Execution Path ===
  if (plan.compiled_fn) {
    // Build the AQPPipelineKernelView
    const size_t ncols = plan.scan_table->columns.size();
    std::vector<const void *> scan_col_data(ncols);
    std::vector<const void *> scan_str_pools(ncols);
    std::vector<const uint64_t *> scan_null_bitmaps(ncols);
    for (size_t c = 0; c < ncols; c++) {
      scan_col_data[c] = plan.scan_table->columns[c].data.get();
      scan_str_pools[c] = plan.scan_table->columns[c].string_pool.get();
      scan_null_bitmaps[c] = plan.scan_table->columns[c].null_bitmap.get();
    }

    std::vector<const uint32_t *> ht_buckets_v(nsteps), ht_next_v(nsteps), ht_row_ids_v(nsteps);
    std::vector<const int32_t *> ht_keys_v(nsteps);
    std::vector<uint32_t> ht_masks_v(nsteps), ht_sizes_v(nsteps);
    std::vector<const uint64_t *> bf_data_v(nsteps);
    std::vector<uint64_t> bf_masks_v(nsteps);

    // Build table column data for inner join output
    std::vector<std::vector<const void *>> build_col_data_vecs(nsteps);
    std::vector<std::vector<const void *>> build_str_pool_vecs(nsteps);
    std::vector<const void **> build_col_data_ptrs(nsteps);
    std::vector<const void **> build_str_pool_ptrs(nsteps);

    for (size_t si = 0; si < nsteps; si++) {
      const auto &step = plan.join_steps[si];
      ht_buckets_v[si] = step.HtBuckets();
      ht_next_v[si] = step.HtNext();
      ht_keys_v[si] = step.HtKeys();
      ht_row_ids_v[si] = step.HtRowIds();
      ht_masks_v[si] = step.HtMask();
      ht_sizes_v[si] = step.HtSize();
      bf_data_v[si] = step.HtBloomData();
      bf_masks_v[si] = step.HtBloomMask();

      if (step.build_table) {
        size_t bc = step.build_table->columns.size();
        build_col_data_vecs[si].resize(bc);
        build_str_pool_vecs[si].resize(bc);
        for (size_t c = 0; c < bc; c++) {
          build_col_data_vecs[si][c] = step.build_table->columns[c].data.get();
          build_str_pool_vecs[si][c] = step.build_table->columns[c].string_pool.get();
        }
        build_col_data_ptrs[si] = build_col_data_vecs[si].data();
        build_str_pool_ptrs[si] = build_str_pool_vecs[si].data();
      }
    }

    AQPPipelineKernelView view = {};
    view.scan_nrows = scan_rows;
    view.scan_col_data = scan_col_data.data();
    view.scan_str_pools = scan_str_pools.data();
    view.scan_null_bitmaps = scan_null_bitmaps.data();
    view.num_join_steps = static_cast<uint32_t>(nsteps);
    view.ht_buckets = ht_buckets_v.data();
    view.ht_next = ht_next_v.data();
    view.ht_keys = ht_keys_v.data();
    view.ht_row_ids = ht_row_ids_v.data();
    view.ht_masks = ht_masks_v.data();
    view.ht_sizes = ht_sizes_v.data();
    view.build_col_data = build_col_data_ptrs.data();
    view.build_str_pools = build_str_pool_ptrs.data();
    view.bf_data = bf_data_v.data();
    view.bf_masks = bf_masks_v.data();

#ifdef HAVE_OPENMP
    if (scan_rows >= OMP_PARALLEL_THRESHOLD) {
      int nthreads = omp_get_max_threads();
      if (nthreads > 12) nthreads = 12;
      if (scan_rows < 100000) nthreads = std::min(nthreads, 4);

      std::vector<FlatTableBuilder> thread_builders(nthreads);
      for (auto &tb : thread_builders)
        tb.Init(plan.output_cols);

      #pragma omp parallel num_threads(nthreads)
      {
        int tid = omp_get_thread_num();
        FlatTableBuilder &my_builder = thread_builders[tid];

        #pragma omp for schedule(dynamic, 8192)
        for (int64_t chunk_start = 0; chunk_start < static_cast<int64_t>(scan_rows);
             chunk_start += 8192) {
          uint64_t chunk_end = std::min<uint64_t>(chunk_start + 8192, scan_rows);
          plan.compiled_fn(static_cast<uint64_t>(chunk_start), chunk_end,
                           &view, &my_builder);
        }
      }

      auto merged = MergeBuilders(thread_builders);
      return merged.Finalize(table_name);
    }
#endif

    FlatTableBuilder builder;
    builder.Init(plan.output_cols);
    builder.Reserve(std::min<uint64_t>(scan_rows / 4, 1024 * 1024));
    plan.compiled_fn(0, scan_rows, &view, &builder);
    return builder.Finalize(table_name);
  }

  // === Interpreted Fallback Path ===

  // Phase 2: Probe
  auto ScanRow = [&](uint64_t row, FlatTableBuilder &builder) {
    if (has_scan_filters) {
      for (const auto &f : plan.scan_filters) {
        if (!f(*plan.scan_table, row)) return;
      }
    }

    if (!any_inner) {
      // All semi-joins: check existence in all hash tables
      for (size_t si = 0; si < nsteps; si++) {
        const auto &step = plan.join_steps[si];
        int32_t key;
        if (step.scan_key_col >= 0) {
          key = plan.scan_table->columns[step.scan_key_col].GetInt32(row);
        } else {
          return;
        }
        if (!step.ht->Contains(key)) return;
        if (!step.join_filters.empty()) {
          bool any_match = false;
          step.ht->ForEach(key, [&](uint32_t build_row) {
            if (!any_match && PassJoinFilters(step, build_row))
              any_match = true;
          });
          if (!any_match) return;
        }
      }

      // Emit row (all FROM_SCAN for semi-only)
      for (size_t c = 0; c < plan.output_cols.size(); c++) {
        const auto &out = plan.output_cols[c];
        const auto &col = plan.scan_table->columns[out.col_idx];
        if (col.type == FlatColumnType::INT32)
          builder.AppendInt(c, col.GetInt32(row));
        else {
          uint32_t len;
          const char *ptr = col.GetVarchar(row, len);
          builder.AppendStr(c, ptr, len);
        }
      }
      builder.FinishRow();
      return;
    }

    // Inner join path: iterative probe with stack-allocated matched_rows
    uint32_t matched_rows[16] = {};
    ProbeStepsIterative(plan, row, matched_rows, builder);
  };

#ifdef HAVE_OPENMP
  if (scan_rows >= OMP_PARALLEL_THRESHOLD) {
    int nthreads = omp_get_max_threads();
    if (nthreads > 12) nthreads = 12;
    if (scan_rows < 100000) nthreads = std::min(nthreads, 4);

    std::vector<FlatTableBuilder> thread_builders(nthreads);
    for (auto &tb : thread_builders)
      tb.Init(plan.output_cols);

    #pragma omp parallel num_threads(nthreads)
    {
      int tid = omp_get_thread_num();
      FlatTableBuilder &my_builder = thread_builders[tid];

      #pragma omp for schedule(dynamic, 8192)
      for (int64_t r = 0; r < static_cast<int64_t>(scan_rows); r++) {
        ScanRow(static_cast<uint64_t>(r), my_builder);
      }
    }

    auto merged = MergeBuilders(thread_builders);
    return merged.Finalize(table_name);
  }
#endif

  FlatTableBuilder builder;
  builder.Init(plan.output_cols);
  builder.Reserve(std::min<uint64_t>(scan_rows / 4, 1024 * 1024));
  for (uint64_t r = 0; r < scan_rows; r++) {
    ScanRow(r, builder);
  }
  return builder.Finalize(table_name);
}

} // namespace storage
} // namespace middleware
