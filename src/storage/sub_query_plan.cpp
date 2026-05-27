#include "storage/sub_query_plan.h"
#include "storage/storage_plan.h"
#include "simplest_ir.h"
#include <algorithm>
#include <cassert>
#include <iostream>

using namespace ir_sql_converter;

namespace middleware {
namespace storage {

namespace {

struct LeafTable {
  std::string name;
  unsigned int ir_table_index;
  bool is_base;
  const FlatTable *flat = nullptr;
};

struct JoinEdge {
  unsigned int left_table_idx;
  unsigned int left_col_idx;
  std::string left_col_name;
  unsigned int right_table_idx;
  unsigned int right_col_idx;
  std::string right_col_name;
};

void CollectLeaves(const AQPStmt *node,
                   std::vector<LeafTable> &leaves,
                   const StoragePlan *storage_plan,
                   const std::unordered_map<std::string, const FlatTable *> &kernel_temps,
                   bool &has_filter, bool &has_aggregate) {
  if (!node)
    return;

  auto ntype = node->GetNodeType();

  if (ntype == FilterNode) {
    if (!node->qual_vec.empty())
      has_filter = true;
  }

  if (ntype == AggregateNode) {
    has_aggregate = true;
  }

  if (ntype == ScanNode) {
    auto *scan = static_cast<const SimplestScan *>(node);
    if (!node->qual_vec.empty())
      has_filter = true;
    LeafTable leaf;
    leaf.name = scan->GetTableName();
    leaf.ir_table_index = scan->GetTableIndex();
    leaf.is_base = true;
    leaf.flat = storage_plan ? storage_plan->GetTable(leaf.name) : nullptr;
    leaves.push_back(leaf);
    return;
  }

  if (ntype == ChunkNode) {
    auto *chunk = static_cast<const SimplestChunk *>(node);
    LeafTable leaf;
    leaf.name = chunk->GetChunkName();
    leaf.ir_table_index = chunk->GetTableIndex();
    leaf.is_base = false;
    auto it = kernel_temps.find(leaf.name);
    if (it != kernel_temps.end())
      leaf.flat = it->second;
    leaves.push_back(leaf);
    return;
  }

  for (const auto &child : node->children)
    CollectLeaves(child.get(), leaves, storage_plan, kernel_temps,
                  has_filter, has_aggregate);
}

void CollectJoinEdges(const AQPStmt *node,
                      std::vector<JoinEdge> &edges,
                      SimplestJoinType &join_type) {
  if (!node)
    return;

  if (node->GetNodeType() == JoinNode) {
    auto *join = static_cast<const SimplestJoin *>(node);
    join_type = join->GetSimplestJoinType();
    for (const auto &cond : join->join_conditions) {
      if (cond->GetSimplestExprType() != Equal)
        continue;
      JoinEdge edge;
      edge.left_table_idx = cond->left_attr->GetTableIndex();
      edge.left_col_idx = cond->left_attr->GetColumnIndex();
      edge.left_col_name = cond->left_attr->GetColumnName();
      edge.right_table_idx = cond->right_attr->GetTableIndex();
      edge.right_col_idx = cond->right_attr->GetColumnIndex();
      edge.right_col_name = cond->right_attr->GetColumnName();
      edges.push_back(edge);
    }
  }

  for (const auto &child : node->children)
    CollectJoinEdges(child.get(), edges, join_type);
}

const LeafTable *FindLeaf(const std::vector<LeafTable> &leaves,
                          unsigned int table_idx) {
  for (const auto &l : leaves)
    if (l.ir_table_index == table_idx)
      return &l;
  return nullptr;
}

} // anonymous namespace

SubQueryPlan AnalyzeSubIR(
    const ir_sql_converter::AQPStmt *sub_ir,
    CsrSupportLevel level,
    const StoragePlan *storage_plan,
    const std::unordered_map<std::string, const FlatTable *> &kernel_temps,
    const std::unordered_map<std::string, CSRIndex> &runtime_csrs) {

  SubQueryPlan plan;
  plan.valid = false;

  if (!sub_ir || !storage_plan)
    return plan;

  // Collect leaf tables and join edges
  std::vector<LeafTable> leaves;
  bool has_filter = false;
  bool has_aggregate = false;
  CollectLeaves(sub_ir, leaves, storage_plan, kernel_temps,
                has_filter, has_aggregate);

  // Semi/Inner level: reject filters and aggregates (filters need FULL level)
  if (level <= CsrSupportLevel::INNER && has_filter)
    return plan;
  if (level <= CsrSupportLevel::INNER && has_aggregate)
    return plan;

  // Need exactly 2 tables for semi level
  if (leaves.size() != 2)
    return plan;

  // All leaf tables must have FlatTable data
  for (const auto &leaf : leaves) {
    if (!leaf.flat)
      return plan;
  }

  SimplestJoinType join_type = InvalidJoinType;
  std::vector<JoinEdge> edges;
  CollectJoinEdges(sub_ir, edges, join_type);

  // Only support inner joins for now
  if (join_type != Inner)
    return plan;

  if (edges.empty())
    return plan;

  // Determine scan table vs lookup table
  // Heuristic: scan the larger table, CSR-lookup the smaller
  const LeafTable *scan_leaf = &leaves[0];
  const LeafTable *lookup_leaf = &leaves[1];
  if (scan_leaf->flat->row_count < lookup_leaf->flat->row_count)
    std::swap(scan_leaf, lookup_leaf);

  // Find a CSR or runtime CSR for the join edge
  const JoinEdge &edge = edges[0];

  // Determine which side of the edge is scan vs lookup
  unsigned int scan_col_idx = 0;
  std::string scan_col_name;
  unsigned int lookup_col_idx = 0;
  std::string lookup_col_name;

  if (edge.left_table_idx == scan_leaf->ir_table_index) {
    scan_col_idx = edge.left_col_idx;
    scan_col_name = edge.left_col_name;
    lookup_col_idx = edge.right_col_idx;
    lookup_col_name = edge.right_col_name;
  } else if (edge.right_table_idx == scan_leaf->ir_table_index) {
    scan_col_idx = edge.right_col_idx;
    scan_col_name = edge.right_col_name;
    lookup_col_idx = edge.left_col_idx;
    lookup_col_name = edge.left_col_name;
  } else {
    return plan;
  }

  // Find scan column in the FlatTable
  int scan_flat_col = scan_leaf->flat->FindColumn(scan_col_name);
  if (scan_flat_col < 0)
    return plan;

  // Only INT32 join keys supported
  if (scan_leaf->flat->columns[scan_flat_col].type != FlatColumnType::INT32)
    return plan;

  // Look for CSR index
  // Try runtime CSR: "lookup_table_name.lookup_col_name"
  const CSRIndex *csr = nullptr;
  std::string runtime_key = lookup_leaf->name + "." + lookup_col_name;
  auto rt_it = runtime_csrs.find(runtime_key);
  if (rt_it != runtime_csrs.end()) {
    csr = &rt_it->second;
  }

  // Try base CSR: storage_plan->GetCSR(scan_table, scan_col)
  // CSR is built as PK→FK rows, so we need CSR where fk_table=scan, fk_col=scan_col
  if (!csr && storage_plan) {
    csr = storage_plan->GetCSR(scan_leaf->name, scan_col_name);
  }

  // Try reverse: CSR where fk_table=lookup, fk_col=lookup_col
  if (!csr && storage_plan) {
    csr = storage_plan->GetCSR(lookup_leaf->name, lookup_col_name);
  }

  // Also try runtime CSR on scan side
  if (!csr) {
    std::string scan_key = scan_leaf->name + "." + scan_col_name;
    auto sk_it = runtime_csrs.find(scan_key);
    if (sk_it != runtime_csrs.end()) {
      // Swap: scan the lookup table, CSR-lookup the scan table
      std::swap(scan_leaf, lookup_leaf);
      scan_flat_col = lookup_leaf->flat->FindColumn(lookup_col_name);
      if (scan_flat_col < 0)
        return plan;
      if (lookup_leaf->flat->columns[scan_flat_col].type != FlatColumnType::INT32)
        return plan;
      // Re-find the scan flat col on the NEW scan table
      scan_flat_col = scan_leaf->flat->FindColumn(lookup_col_name);
      if (scan_flat_col < 0)
        return plan;
      csr = &sk_it->second;
    }
  }

  if (!csr)
    return plan;

  // Check output columns — for semi level, all must come from scan table
  const auto &target_list = sub_ir->target_list;
  if (target_list.empty())
    return plan;

  std::vector<KernelOutputCol> output_cols;
  for (const auto &attr : target_list) {
    unsigned int tbl_idx = attr->GetTableIndex();
    std::string col_name = attr->GetColumnName();

    KernelOutputCol out;
    if (tbl_idx == scan_leaf->ir_table_index) {
      out.source = KernelOutputCol::FROM_SCAN;
      out.col_idx = scan_leaf->flat->FindColumn(col_name);
      if (out.col_idx < 0)
        return plan;
      out.type = scan_leaf->flat->columns[out.col_idx].type;
    } else if (level >= CsrSupportLevel::INNER &&
               tbl_idx == lookup_leaf->ir_table_index) {
      out.source = KernelOutputCol::FROM_JOIN;
      out.step_idx = 0;
      out.col_idx = lookup_leaf->flat->FindColumn(col_name);
      if (out.col_idx < 0)
        return plan;
      out.type = lookup_leaf->flat->columns[out.col_idx].type;
    } else {
      // Semi level requires all output from scan table
      return plan;
    }
    out.name = col_name;
    output_cols.push_back(out);
  }

  // Build the plan
  KernelJoinStep step;
  step.csr = csr;
  step.scan_key_col_idx = scan_flat_col;
  step.joined_table = lookup_leaf->flat;
  step.is_semi = (level == CsrSupportLevel::SEMI);

  plan.scan_table = scan_leaf->flat;
  plan.scan_table_name = scan_leaf->name;
  plan.join_steps.push_back(step);
  plan.output_cols = std::move(output_cols);
  plan.valid = true;

  return plan;
}

// FlatTableBuilder: accumulate rows, then finalize into a FlatTable
namespace {

struct FlatTableBuilder {
  struct ColBuffer {
    FlatColumnType type;
    std::vector<int32_t> int_data;
    std::vector<std::string> str_data;
  };

  std::vector<std::string> column_names;
  std::vector<ColBuffer> col_buffers;
  uint64_t row_count = 0;

  void Init(const std::vector<KernelOutputCol> &output_cols) {
    col_buffers.resize(output_cols.size());
    column_names.resize(output_cols.size());
    for (size_t i = 0; i < output_cols.size(); i++) {
      col_buffers[i].type = output_cols[i].type;
      column_names[i] = output_cols[i].name;
    }
  }

  void Reserve(uint64_t est_rows) {
    for (auto &buf : col_buffers) {
      if (buf.type == FlatColumnType::INT32)
        buf.int_data.reserve(est_rows);
      else
        buf.str_data.reserve(est_rows);
    }
  }

  void AppendInt(size_t col, int32_t val) {
    col_buffers[col].int_data.push_back(val);
  }

  void AppendStr(size_t col, const char *ptr, uint32_t len) {
    col_buffers[col].str_data.emplace_back(ptr, len);
  }

  void FinishRow() { row_count++; }

  std::unique_ptr<FlatTable> Finalize(const std::string &table_name) {
    auto result = std::make_unique<FlatTable>();
    result->table_name = table_name;
    result->row_count = row_count;
    result->column_names = column_names;
    result->columns.resize(col_buffers.size());

    for (size_t c = 0; c < col_buffers.size(); c++) {
      auto &buf = col_buffers[c];
      auto &col = result->columns[c];
      col.type = buf.type;
      col.row_count = row_count;
      col.nullable = false;

      if (buf.type == FlatColumnType::INT32) {
        col.data = std::unique_ptr<char[]>(
            new char[row_count * sizeof(int32_t)]);
        std::memcpy(col.data.get(), buf.int_data.data(),
                    row_count * sizeof(int32_t));
      } else {
        // VARCHAR: build offset array + string pool
        uint64_t total_len = 0;
        for (const auto &s : buf.str_data)
          total_len += s.size();

        col.data = std::unique_ptr<char[]>(
            new char[(row_count + 1) * sizeof(uint32_t)]);
        col.string_pool = std::unique_ptr<char[]>(new char[total_len]);
        col.string_pool_size = total_len;

        auto *offsets = reinterpret_cast<uint32_t *>(col.data.get());
        uint32_t offset = 0;
        for (uint64_t r = 0; r < row_count; r++) {
          offsets[r] = offset;
          std::memcpy(col.string_pool.get() + offset,
                      buf.str_data[r].data(), buf.str_data[r].size());
          offset += static_cast<uint32_t>(buf.str_data[r].size());
        }
        offsets[row_count] = offset;
      }
    }

    return result;
  }
};

} // anonymous namespace

std::unique_ptr<FlatTable> ExecuteSubQueryPlan(const SubQueryPlan &plan,
                                               const std::string &table_name) {
  assert(plan.valid);

  FlatTableBuilder builder;
  builder.Init(plan.output_cols);

  uint64_t scan_rows = plan.scan_table->row_count;
  builder.Reserve(scan_rows / 4); // estimate 25% selectivity

  // Check if any join step is an inner join (needs row expansion)
  bool any_inner = false;
  for (const auto &step : plan.join_steps) {
    if (!step.is_semi) {
      any_inner = true;
      break;
    }
  }

  // Helper: emit one output row given scan row + joined row indices
  auto EmitRow = [&](uint64_t scan_row,
                     const std::vector<uint64_t> &joined_rows) {
    for (size_t c = 0; c < plan.output_cols.size(); c++) {
      const auto &out = plan.output_cols[c];
      const FlatTable *src_table;
      uint64_t src_row;
      if (out.source == KernelOutputCol::FROM_SCAN) {
        src_table = plan.scan_table;
        src_row = scan_row;
      } else {
        src_table = plan.join_steps[out.step_idx].joined_table;
        src_row = joined_rows[out.step_idx];
      }
      const auto &col = src_table->columns[out.col_idx];
      if (col.type == FlatColumnType::INT32) {
        builder.AppendInt(c, col.GetInt32(src_row));
      } else {
        uint32_t len;
        const char *ptr = col.GetVarchar(src_row, len);
        builder.AppendStr(c, ptr, len);
      }
    }
    builder.FinishRow();
  };

  for (uint64_t row = 0; row < scan_rows; row++) {
    if (!any_inner) {
      // Semi-join path: existence check only, no row expansion
      bool pass = true;
      for (const auto &step : plan.join_steps) {
        int32_t key =
            plan.scan_table->columns[step.scan_key_col_idx].GetInt32(row);
        if (step.use_bitset) {
          if (key < 0 || static_cast<size_t>(key) >= step.pk_bitset.size() ||
              !step.pk_bitset[key]) {
            pass = false;
            break;
          }
        } else if (step.csr) {
          auto result = step.csr->Lookup(key);
          if (result.first == result.second) {
            pass = false;
            break;
          }
        } else {
          pass = false;
          break;
        }
      }
      if (!pass)
        continue;
      std::vector<uint64_t> dummy(plan.join_steps.size(), 0);
      EmitRow(row, dummy);
    } else {
      // Inner-join path: iterate CSR matches for row expansion.
      // Currently supports single join step.
      assert(plan.join_steps.size() == 1);
      const auto &step = plan.join_steps[0];
      int32_t key =
          plan.scan_table->columns[step.scan_key_col_idx].GetInt32(row);

      if (step.use_bitset) {
        if (key < 0 || static_cast<size_t>(key) >= step.pk_bitset.size() ||
            !step.pk_bitset[key])
          continue;
        std::vector<uint64_t> joined_rows = {0};
        EmitRow(row, joined_rows);
      } else if (step.csr) {
        auto [begin, end] = step.csr->Lookup(key);
        if (begin == end)
          continue;
        for (auto it = begin; it != end; ++it) {
          std::vector<uint64_t> joined_rows = {*it};
          EmitRow(row, joined_rows);
        }
      }
    }
  }

  return builder.Finalize(table_name);
}

} // namespace storage
} // namespace middleware
