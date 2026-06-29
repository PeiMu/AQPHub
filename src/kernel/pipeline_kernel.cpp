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
#ifdef HAS_AVX2
#include <immintrin.h>
#endif
#include "util/thread_pool.h"

using namespace ir_sql_converter;

namespace middleware {
namespace storage {

static constexpr uint64_t OMP_PARALLEL_THRESHOLD = 10000;

// ============================================================================
// HashJoinTable — chained hash table with Fibonacci hashing
// ============================================================================

struct PipelineJoinStep::HashJoinTable {
  static constexpr uint32_t EMPTY = UINT32_MAX;
  static constexpr int32_t DIRECT_MAP_MAX_RANGE = 10000;

  std::vector<uint32_t> buckets_;
  std::vector<uint32_t> next_;
  std::vector<int32_t> keys_;
  std::vector<uint32_t> row_ids_;
  uint32_t mask_ = 0;
  uint32_t size_ = 0;
  uint32_t prefetch_distance_ = 0;

  // Direct-map for small key ranges (Phase 2/B optimization)
  std::vector<uint32_t> direct_map_;
  int32_t direct_min_ = 0;
  int32_t direct_range_ = 0;

  void Build(const FlatTable &table, int key_col,
             const std::vector<RowPredicate> &filters) {
    uint64_t nrows = table.row_count;

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

    for (uint32_t i = 0; i < size_; i++) {
      uint32_t h = (static_cast<uint32_t>(keys_[i]) * 2654435761u) & mask_;
      next_[i] = buckets_[h];
      buckets_[h] = i;
    }
  }

  // Build direct-map array for O(1) lookup when key range is small.
  // Returns true if direct-map was built, false if range too large.
  bool TryBuildDirectMap() {
    if (size_ == 0) return false;

    int32_t min_key = keys_[0], max_key = keys_[0];
    for (uint32_t i = 1; i < size_; i++) {
      if (keys_[i] < min_key) min_key = keys_[i];
      if (keys_[i] > max_key) max_key = keys_[i];
    }

    int64_t range = static_cast<int64_t>(max_key) - min_key + 1;
    if (range > DIRECT_MAP_MAX_RANGE || range <= 0) return false;

    direct_min_ = min_key;
    direct_range_ = static_cast<int32_t>(range);
    direct_map_.assign(range, EMPTY);
    for (uint32_t i = 0; i < size_; i++) {
      direct_map_[keys_[i] - min_key] = row_ids_[i];
    }
    return true;
  }

  ProbeMethod SelectProbeMethod() const {
    if (size_ == 0) return ProbeMethod::SKIP;
    if (direct_range_ > 0) return ProbeMethod::DIRECT;
    if (size_ == 1) return ProbeMethod::POINT;
    if (size_ <= 15) return ProbeMethod::LINEAR;
    return ProbeMethod::HASH;
  }

  void ComputePrefetchDistance() {
    if (size_ == 0) { prefetch_distance_ = 0; return; }
    uint64_t mem = static_cast<uint64_t>(size_) * 12 +
                   static_cast<uint64_t>(mask_ + 1) * 4;
    if (mem < 32 * 1024)            prefetch_distance_ = 0;   // fits L1
    else if (mem < 256 * 1024)      prefetch_distance_ = 4;   // L2
    else if (mem < 8ULL * 1024 * 1024) prefetch_distance_ = 8;   // L3
    else                            prefetch_distance_ = 16;  // DRAM
  }

  void PrefetchBucket(int32_t key) const {
    uint32_t h = (static_cast<uint32_t>(key) * 2654435761u) & mask_;
    __builtin_prefetch(&buckets_[h], 0, 1);
  }

  // --- Contains variants by probe method ---
  bool ContainsDirect(int32_t key) const {
    int32_t off = key - direct_min_;
    return static_cast<uint32_t>(off) < static_cast<uint32_t>(direct_range_)
        && direct_map_[off] != EMPTY;
  }

  bool ContainsPoint(int32_t key) const {
    return keys_[0] == key;
  }

  bool ContainsLinear(int32_t key) const {
    for (uint32_t i = 0; i < size_; i++) {
      if (keys_[i] == key) return true;
    }
    return false;
  }

  bool ContainsHash(int32_t key) const {
    uint32_t h = (static_cast<uint32_t>(key) * 2654435761u) & mask_;
    uint32_t idx = buckets_[h];
    while (idx != EMPTY) {
      if (keys_[idx] == key) return true;
      idx = next_[idx];
    }
    return false;
  }

  bool Contains(int32_t key, ProbeMethod method) const {
    switch (method) {
    case ProbeMethod::SKIP: return false;
    case ProbeMethod::DIRECT: return ContainsDirect(key);
    case ProbeMethod::POINT: return ContainsPoint(key);
    case ProbeMethod::LINEAR: return ContainsLinear(key);
    case ProbeMethod::HASH: return ContainsHash(key);
    }
    __builtin_unreachable();
  }

  // --- Lookup (returns row_id or EMPTY) variants by probe method ---
  uint32_t LookupDirect(int32_t key) const {
    int32_t off = key - direct_min_;
    if (static_cast<uint32_t>(off) < static_cast<uint32_t>(direct_range_))
      return direct_map_[off];
    return EMPTY;
  }

  uint32_t LookupPoint(int32_t key) const {
    return keys_[0] == key ? row_ids_[0] : EMPTY;
  }

  uint32_t LookupLinear(int32_t key) const {
    for (uint32_t i = 0; i < size_; i++) {
      if (keys_[i] == key) return row_ids_[i];
    }
    return EMPTY;
  }

  uint32_t LookupHash(int32_t key) const {
    uint32_t h = (static_cast<uint32_t>(key) * 2654435761u) & mask_;
    uint32_t idx = buckets_[h];
    while (idx != EMPTY) {
      if (keys_[idx] == key) return row_ids_[idx];
      idx = next_[idx];
    }
    return EMPTY;
  }

  uint32_t Lookup(int32_t key, ProbeMethod method) const {
    switch (method) {
    case ProbeMethod::SKIP: return EMPTY;
    case ProbeMethod::DIRECT: return LookupDirect(key);
    case ProbeMethod::POINT: return LookupPoint(key);
    case ProbeMethod::LINEAR: return LookupLinear(key);
    case ProbeMethod::HASH: return LookupHash(key);
    }
    __builtin_unreachable();
  }

  // --- ForEach variants ---
  template <typename Fn> void ForEachDirect(int32_t key, Fn &&fn) const {
    int32_t off = key - direct_min_;
    if (static_cast<uint32_t>(off) < static_cast<uint32_t>(direct_range_)) {
      uint32_t rid = direct_map_[off];
      if (rid != EMPTY) fn(rid);
    }
  }

  template <typename Fn> void ForEachPoint(int32_t key, Fn &&fn) const {
    if (keys_[0] == key) fn(row_ids_[0]);
  }

  template <typename Fn> void ForEachLinear(int32_t key, Fn &&fn) const {
    for (uint32_t i = 0; i < size_; i++) {
      if (keys_[i] == key) fn(row_ids_[i]);
    }
  }

  template <typename Fn> void ForEachHash(int32_t key, Fn &&fn) const {
    uint32_t h = (static_cast<uint32_t>(key) * 2654435761u) & mask_;
    uint32_t idx = buckets_[h];
    while (idx != EMPTY) {
      if (keys_[idx] == key) fn(row_ids_[idx]);
      idx = next_[idx];
    }
  }

  // ForEach with early-exit for unique keys
  template <typename Fn> void ForEachHashUnique(int32_t key, Fn &&fn) const {
    uint32_t h = (static_cast<uint32_t>(key) * 2654435761u) & mask_;
    uint32_t idx = buckets_[h];
    while (idx != EMPTY) {
      if (keys_[idx] == key) { fn(row_ids_[idx]); return; }
      idx = next_[idx];
    }
  }

  template <typename Fn> void ForEach(int32_t key, ProbeMethod method,
                                      bool unique, Fn &&fn) const {
    switch (method) {
    case ProbeMethod::SKIP: return;
    case ProbeMethod::DIRECT: return ForEachDirect(key, std::forward<Fn>(fn));
    case ProbeMethod::POINT: return ForEachPoint(key, std::forward<Fn>(fn));
    case ProbeMethod::LINEAR:
      if (unique) {
        for (uint32_t i = 0; i < size_; i++) {
          if (keys_[i] == key) { fn(row_ids_[i]); return; }
        }
      } else {
        return ForEachLinear(key, std::forward<Fn>(fn));
      }
      return;
    case ProbeMethod::HASH:
      if (unique) return ForEachHashUnique(key, std::forward<Fn>(fn));
      return ForEachHash(key, std::forward<Fn>(fn));
    }
  }

  // --- Hash chain iteration for inner-join DFS (used by ProbeStepsIterative) ---
  // StartChain: returns first bucket index for the given key hash
  uint32_t StartChain(int32_t key) const {
    if (size_ == 0) return EMPTY;
    uint32_t h = (static_cast<uint32_t>(key) * 2654435761u) & mask_;
    return buckets_[h];
  }
};

constexpr uint32_t PipelineJoinStep::HashJoinTable::EMPTY;

// PipelineJoinStep special members (needed because HashJoinTable is incomplete in header)
PipelineJoinStep::PipelineJoinStep() = default;
PipelineJoinStep::~PipelineJoinStep() = default;
PipelineJoinStep::PipelineJoinStep(PipelineJoinStep &&) noexcept = default;
PipelineJoinStep &PipelineJoinStep::operator=(PipelineJoinStep &&) noexcept = default;

uint32_t PipelineJoinStep::HtSize() const { return ht ? ht->size_ : 0; }

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
    if (scan_leaf->HasFilters()) {
      if (!CompileAllLeafFilters(scan_leaf->all_filters, scan_leaf->flat,
                                 scan_predicates))
        return plan;
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

    if (build_leaf->is_base && storage_plan) {
      for (const auto &kv : storage_plan->GetCSRMap()) {
        if (kv.second.pk_table == build_leaf->name &&
            kv.second.pk_column == build_col) {
          step.build_key_unique = true;
          break;
        }
      }
    }

    // Compile build-side filters
    if (build_leaf->HasFilters()) {
      if (!CompileAllLeafFilters(build_leaf->all_filters, build_leaf->flat,
                                 step.build_filters))
        return plan;
    }

    // Add dim-derived filters on build table as build_filters
    AddDimDerivedFilters(dim_derived_filters, build_leaf->ir_table_index,
                         step.build_filters);
  }

  // Compile scan filters
  std::vector<RowPredicate> scan_predicates;
  if (scan_leaf->HasFilters()) {
    if (!CompileAllLeafFilters(scan_leaf->all_filters, scan_leaf->flat,
                               scan_predicates))
      return plan;
  }
  AddDimDerivedFilters(dim_derived_filters, scan_leaf->ir_table_index,
                       scan_predicates);

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

// Iterative depth-first probe through join steps.
// Uses adaptive probe methods (POINT/LINEAR/DIRECT/HASH) per step
// and unique-key early-exit to skip chain walk after first match.
static void ProbeStepsIterative(
    const PipelineKernelPlan &plan,
    uint64_t scan_row,
    uint32_t *matched_rows,
    FlatTableBuilder &builder) {

  const size_t nsteps = plan.join_steps.size();

  struct Frame {
    uint32_t chain_cursor;
    bool is_semi_done;
  };
  Frame stack[16];
  if (nsteps > 16) return;

  size_t depth = 0;

  auto StartStep = [&](size_t step_idx) -> bool {
    const auto &step = plan.join_steps[step_idx];
    const ProbeMethod method = step.probe_method;
    int32_t key = GetProbeKey(plan, step, scan_row, matched_rows);

    if (step.is_semi) {
      if (step.join_filters.empty()) {
        if (!step.ht->Contains(key, method)) return false;
      } else {
        bool any_match = false;
        step.ht->ForEach(key, method, false, [&](uint32_t build_row) {
          if (!any_match && PassJoinFilters(step, build_row))
            any_match = true;
        });
        if (!any_match) return false;
      }
      stack[step_idx].is_semi_done = true;
      stack[step_idx].chain_cursor = PipelineJoinStep::HashJoinTable::EMPTY;
      return true;
    }

    // Inner join: for unique keys or non-HASH methods, use Lookup (single match)
    if (step.build_key_unique || method != ProbeMethod::HASH) {
      uint32_t rid = step.ht->Lookup(key, method);
      if (rid == PipelineJoinStep::HashJoinTable::EMPTY) return false;
      if (!step.join_filters.empty() && !PassJoinFilters(step, rid)) return false;
      matched_rows[step_idx] = rid;
      stack[step_idx].chain_cursor = PipelineJoinStep::HashJoinTable::EMPTY;
      stack[step_idx].is_semi_done = false;
      return true;
    }

    // Non-unique HASH: iterate chain
    if (step.ht->size_ == 0) return false;
    uint32_t h = (static_cast<uint32_t>(key) * 2654435761u) & step.ht->mask_;
    stack[step_idx].chain_cursor = step.ht->buckets_[h];
    stack[step_idx].is_semi_done = false;

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

  auto AdvanceStep = [&](size_t step_idx) -> bool {
    const auto &step = plan.join_steps[step_idx];
    if (step.is_semi) return false;
    // Unique keys or non-HASH: at most 1 match, no advancement
    if (step.build_key_unique || step.probe_method != ProbeMethod::HASH) return false;

    int32_t key = GetProbeKey(plan, step, scan_row, matched_rows);
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

// ============================================================================
// Vectorized batch processing (Phase 4)
// ============================================================================

static constexpr int BATCH_SIZE = 1024;

#ifdef HAS_AVX2
static void BatchHash8(const int32_t *keys, uint32_t *hashes,
                       int count, uint32_t mask) {
  const __m256i vfib = _mm256_set1_epi32(static_cast<int32_t>(2654435761u));
  const __m256i vmask = _mm256_set1_epi32(static_cast<int32_t>(mask));
  int i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256i vk = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(keys + i));
    __m256i vh = _mm256_mullo_epi32(vk, vfib);
    vh = _mm256_and_si256(vh, vmask);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(hashes + i), vh);
  }
  for (; i < count; i++) {
    hashes[i] = (static_cast<uint32_t>(keys[i]) * 2654435761u) & mask;
  }
}
#endif

static void ExecuteOneBatch(
    const PipelineKernelPlan &plan,
    uint64_t batch_start, uint64_t batch_end,
    FlatTableBuilder &builder) {

  const size_t nsteps = plan.join_steps.size();
  const bool has_scan_filters = !plan.scan_filters.empty();

  alignas(32) uint32_t qualifying[BATCH_SIZE];
  alignas(32) int32_t keys[BATCH_SIZE];
#ifdef HAS_AVX2
  alignas(32) uint32_t hashes[BATCH_SIZE];
#endif
  uint32_t matched_rows[16][BATCH_SIZE];

  int inner_step_indices[16];
  int n_inner_before[16];
  int n_inner_total = 0;
  for (size_t si = 0; si < nsteps; si++) {
    n_inner_before[si] = n_inner_total;
    if (!plan.join_steps[si].is_semi)
      inner_step_indices[n_inner_total++] = static_cast<int>(si);
  }

  // Phase A: Scan filter → qualifying[]
  int n_qual = 0;
  if (has_scan_filters) {
    for (uint64_t r = batch_start; r < batch_end; r++) {
      bool pass = true;
      for (const auto &f : plan.scan_filters) {
        if (!f(*plan.scan_table, r)) { pass = false; break; }
      }
      if (pass) qualifying[n_qual++] = static_cast<uint32_t>(r);
    }
  } else {
    int count = static_cast<int>(batch_end - batch_start);
    for (int i = 0; i < count; i++)
      qualifying[i] = static_cast<uint32_t>(batch_start) + i;
    n_qual = count;
  }
  if (n_qual == 0) return;

  // Phase B: Probe each join step, compact qualifying[]
  for (size_t si = 0; si < nsteps && n_qual > 0; si++) {
    const auto &step = plan.join_steps[si];
    const auto *ht = step.ht.get();
    const ProbeMethod method = step.probe_method;
    const int key_col = step.scan_key_col;

    // B.1: Gather keys
    const auto *col_data = reinterpret_cast<const int32_t *>(
        plan.scan_table->columns[key_col].data.get());
    for (int i = 0; i < n_qual; i++)
      keys[i] = col_data[qualifying[i]];

    // B.2: SIMD hash + batch prefetch (HASH method with large HT)
    if (method == ProbeMethod::HASH && ht->prefetch_distance_ > 0) {
#ifdef HAS_AVX2
      BatchHash8(keys, hashes, n_qual, ht->mask_);
      for (int i = 0; i < n_qual; i++)
        __builtin_prefetch(&ht->buckets_[hashes[i]], 0, 1);
#else
      for (int i = 0; i < n_qual; i++)
        ht->PrefetchBucket(keys[i]);
#endif
    }

    // B.3: Probe + compact
    int new_n_qual = 0;
    const int n_ib = n_inner_before[si];

    if (step.is_semi) {
      for (int i = 0; i < n_qual; i++) {
        if (ht->Contains(keys[i], method)) {
          if (new_n_qual != i) {
            qualifying[new_n_qual] = qualifying[i];
            for (int j = 0; j < n_ib; j++)
              matched_rows[inner_step_indices[j]][new_n_qual] =
                  matched_rows[inner_step_indices[j]][i];
          }
          new_n_qual++;
        }
      }
    } else {
      for (int i = 0; i < n_qual; i++) {
        uint32_t rid = ht->Lookup(keys[i], method);
        if (rid != PipelineJoinStep::HashJoinTable::EMPTY) {
          if (new_n_qual != i) {
            qualifying[new_n_qual] = qualifying[i];
            for (int j = 0; j < n_ib; j++)
              matched_rows[inner_step_indices[j]][new_n_qual] =
                  matched_rows[inner_step_indices[j]][i];
          }
          matched_rows[si][new_n_qual] = rid;
          new_n_qual++;
        }
      }
    }
    n_qual = new_n_qual;
  }
  if (n_qual == 0) return;

  // Phase C: Emit output
  for (int i = 0; i < n_qual; i++) {
    uint32_t scan_row = qualifying[i];
    for (size_t c = 0; c < plan.output_cols.size(); c++) {
      const auto &out = plan.output_cols[c];
      if (out.source == KernelOutputCol::FROM_SCAN) {
        const auto &col = plan.scan_table->columns[out.col_idx];
        if (col.type == FlatColumnType::INT32)
          builder.AppendInt(c, col.GetInt32(scan_row));
        else {
          uint32_t len;
          const char *ptr = col.GetVarchar(scan_row, len);
          builder.AppendStr(c, ptr, len);
        }
      } else {
        uint32_t build_row = matched_rows[out.step_idx][i];
        const auto &col =
            plan.join_steps[out.step_idx].build_table->columns[out.col_idx];
        if (col.type == FlatColumnType::INT32)
          builder.AppendInt(c, col.GetInt32(build_row));
        else {
          uint32_t len;
          const char *ptr = col.GetVarchar(build_row, len);
          builder.AppendStr(c, ptr, len);
        }
      }
    }
    builder.FinishRow();
  }
}

// ============================================================================
// ExecutePipelineKernel
// ============================================================================

std::unique_ptr<FlatTable> ExecutePipelineKernel(
    PipelineKernelPlan &plan,
    const std::string &table_name) {
  assert(plan.valid);

  // Phase 1: Build hash tables and select probe method for each join step
  for (size_t si = 0; si < plan.join_steps.size(); si++) {
    auto &step = plan.join_steps[si];
    step.ht = std::make_unique<PipelineJoinStep::HashJoinTable>();
    step.ht->Build(*step.build_table, step.build_key_col, step.build_filters);
    step.ht->TryBuildDirectMap();
    step.probe_method = step.ht->SelectProbeMethod();
    if (!step.is_semi && !step.build_key_unique)
      step.probe_method = ProbeMethod::HASH;
    step.ht->ComputePrefetchDistance();
  }

  uint64_t scan_rows = plan.scan_table->row_count;
  const bool has_scan_filters = !plan.scan_filters.empty();
  const size_t nsteps = plan.join_steps.size();

  // Classify: can we use the batch path?
  bool can_batch = (scan_rows >= static_cast<uint64_t>(BATCH_SIZE));
  for (size_t si = 0; can_batch && si < nsteps; si++) {
    const auto &step = plan.join_steps[si];
    if (step.scan_key_col < 0) can_batch = false;
    if (!step.is_semi && !step.build_key_unique) can_batch = false;
    if (!step.join_filters.empty()) can_batch = false;
  }

  if (can_batch) {
    // === Batch path ===
#ifdef HAVE_OPENMP
    if (scan_rows >= OMP_PARALLEL_THRESHOLD) {
      int bg = g_bg_active_threads.load(std::memory_order_relaxed);
      int nthreads = omp_get_max_threads();
      if (nthreads > 12 - bg) nthreads = std::max(1, 12 - bg);
      if (scan_rows < 100000) nthreads = std::min(nthreads, 4);

      std::vector<FlatTableBuilder> thread_builders(nthreads);
      for (auto &tb : thread_builders)
        tb.Init(plan.output_cols);

      int64_t n_batches = (static_cast<int64_t>(scan_rows) + BATCH_SIZE - 1)
                          / BATCH_SIZE;

      #pragma omp parallel num_threads(nthreads)
      {
        int tid = omp_get_thread_num();
        FlatTableBuilder &my_builder = thread_builders[tid];

        #pragma omp for schedule(dynamic, 8)
        for (int64_t bi = 0; bi < n_batches; bi++) {
          uint64_t bs = static_cast<uint64_t>(bi) * BATCH_SIZE;
          uint64_t be = std::min(bs + static_cast<uint64_t>(BATCH_SIZE),
                                 scan_rows);
          ExecuteOneBatch(plan, bs, be, my_builder);
        }
      }

      auto merged = MergeBuilders(thread_builders);
      return merged.Finalize(table_name);
    }
#endif

    FlatTableBuilder builder;
    builder.Init(plan.output_cols);
    builder.Reserve(std::min<uint64_t>(scan_rows / 4, 1024 * 1024));
    for (uint64_t bs = 0; bs < scan_rows; bs += BATCH_SIZE) {
      uint64_t be = std::min(bs + static_cast<uint64_t>(BATCH_SIZE), scan_rows);
      ExecuteOneBatch(plan, bs, be, builder);
    }
    return builder.Finalize(table_name);
  }

  // === Scalar fallback path (chain joins, non-unique HASH inner, small scans) ===
  bool any_inner = false;
  for (const auto &step : plan.join_steps) {
    if (!step.is_semi) { any_inner = true; break; }
  }

  struct PrefetchInfo {
    const PipelineJoinStep::HashJoinTable *ht;
    int scan_key_col;
    uint32_t distance;
  };
  PrefetchInfo prefetch_buf[16];
  int n_prefetch = 0;
  for (const auto &step : plan.join_steps) {
    if (step.probe_method == ProbeMethod::HASH &&
        step.scan_key_col >= 0 &&
        step.ht->prefetch_distance_ > 0 &&
        n_prefetch < 16) {
      prefetch_buf[n_prefetch++] = {step.ht.get(), step.scan_key_col,
                                     step.ht->prefetch_distance_};
    }
  }

  auto ScanRow = [&](uint64_t row, FlatTableBuilder &builder) {
    if (has_scan_filters) {
      for (const auto &f : plan.scan_filters) {
        if (!f(*plan.scan_table, row)) return;
      }
    }

    if (!any_inner) {
      for (size_t si = 0; si < nsteps; si++) {
        const auto &step = plan.join_steps[si];
        int32_t key;
        if (step.scan_key_col >= 0) {
          key = plan.scan_table->columns[step.scan_key_col].GetInt32(row);
        } else {
          return;
        }
        if (!step.ht->Contains(key, step.probe_method)) return;
        if (!step.join_filters.empty()) {
          bool any_match = false;
          step.ht->ForEach(key, step.probe_method, step.build_key_unique,
                           [&](uint32_t build_row) {
            if (!any_match && PassJoinFilters(step, build_row))
              any_match = true;
          });
          if (!any_match) return;
        }
      }

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

    uint32_t matched_rows[16] = {};
    ProbeStepsIterative(plan, row, matched_rows, builder);
  };

#ifdef HAVE_OPENMP
  if (scan_rows >= OMP_PARALLEL_THRESHOLD) {
    int bg = g_bg_active_threads.load(std::memory_order_relaxed);
    int nthreads = omp_get_max_threads();
    if (nthreads > 12 - bg) nthreads = std::max(1, 12 - bg);
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
        for (int pi = 0; pi < n_prefetch; pi++) {
          uint64_t future = static_cast<uint64_t>(r) + prefetch_buf[pi].distance;
          if (future < scan_rows) {
            int32_t key = plan.scan_table->columns[prefetch_buf[pi].scan_key_col]
                              .GetInt32(future);
            prefetch_buf[pi].ht->PrefetchBucket(key);
          }
        }
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
    for (int pi = 0; pi < n_prefetch; pi++) {
      uint64_t future = r + prefetch_buf[pi].distance;
      if (future < scan_rows) {
        int32_t key = plan.scan_table->columns[prefetch_buf[pi].scan_key_col]
                          .GetInt32(future);
        prefetch_buf[pi].ht->PrefetchBucket(key);
      }
    }
    ScanRow(r, builder);
  }
  return builder.Finalize(table_name);
}

} // namespace storage
} // namespace middleware
