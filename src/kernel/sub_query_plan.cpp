#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "kernel/sub_query_plan.h"
#include "storage/dimension_cache.h"
#include "storage/inverted_index.h"
#include "storage/storage_plan.h"
#include "simplest_ir.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string.h>
#include <unordered_set>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

using namespace ir_sql_converter;

namespace middleware {
namespace storage {

namespace {

static constexpr uint64_t OMP_PARALLEL_THRESHOLD = 10000;

// DP-based SQL LIKE matching (handles % and _ wildcards)
static bool LikeMatch(const char *str, uint32_t str_len, const char *pat,
                      size_t pat_len) {
  std::vector<bool> dp(pat_len + 1, false);
  dp[0] = true;
  for (size_t j = 1; j <= pat_len; j++) {
    if (pat[j - 1] == '%')
      dp[j] = dp[j - 1];
  }
  for (uint32_t i = 1; i <= str_len; i++) {
    bool prev = dp[0];
    dp[0] = false;
    for (size_t j = 1; j <= pat_len; j++) {
      bool tmp = dp[j];
      if (pat[j - 1] == '%') {
        dp[j] = dp[j - 1] || dp[j];
      } else if (pat[j - 1] == '_' || pat[j - 1] == str[i - 1]) {
        dp[j] = prev;
      } else {
        dp[j] = false;
      }
      prev = tmp;
    }
  }
  return dp[pat_len];
}

enum LikePatternKind {
  LIKE_COMPLEX = 0,
  LIKE_EQUALITY,
  LIKE_PREFIX,
  LIKE_SUFFIX,
  LIKE_CONTAINS,
  LIKE_MULTI_SEGMENT
};

static LikePatternKind ClassifyLikePattern(const std::string &pattern,
                                           std::string &literal_out) {
  if (pattern.empty()) {
    literal_out.clear();
    return LIKE_EQUALITY;
  }
  if (pattern.find('_') != std::string::npos)
    return LIKE_COMPLEX;

  size_t leading = 0;
  while (leading < pattern.size() && pattern[leading] == '%')
    ++leading;
  size_t trailing = 0;
  while (trailing < pattern.size() &&
         pattern[pattern.size() - 1 - trailing] == '%')
    ++trailing;

  size_t mid_start = leading, mid_end = pattern.size() - trailing;
  for (size_t i = mid_start; i < mid_end; ++i)
    if (pattern[i] == '%')
      return LIKE_COMPLEX;

  literal_out = pattern.substr(mid_start, mid_end - mid_start);

  if (leading == 0 && trailing == 0)
    return LIKE_EQUALITY;
  if (leading == 0 && trailing > 0)
    return LIKE_PREFIX;
  if (leading > 0 && trailing == 0)
    return LIKE_SUFFIX;
  return LIKE_CONTAINS;
}

struct LikeSegments {
  std::vector<std::string> segs;
  bool has_leading_pct = false;
  bool has_trailing_pct = false;
};

static LikePatternKind ClassifyLikePatternEx(const std::string &pattern,
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
    } else {
      cur += c;
    }
  }
  if (!cur.empty()) seg_out.segs.push_back(cur);

  if (seg_out.segs.size() >= 2) return LIKE_MULTI_SEGMENT;
  return LIKE_COMPLEX;
}

static bool LikeMatchSegments(const char *str, uint32_t str_len,
                              const LikeSegments &segs) {
  const char *pos = str;
  int32_t remaining = static_cast<int32_t>(str_len);
  for (size_t i = 0; i < segs.segs.size(); i++) {
    const auto &seg = segs.segs[i];
    int32_t seg_len = static_cast<int32_t>(seg.size());
    if (i == 0 && !segs.has_leading_pct) {
      if (remaining < seg_len ||
          memcmp(pos, seg.data(), seg.size()) != 0)
        return false;
      pos += seg_len; remaining -= seg_len;
    } else if (i == segs.segs.size() - 1 && !segs.has_trailing_pct) {
      if (remaining < seg_len ||
          memcmp(pos + remaining - seg_len, seg.data(), seg.size()) != 0)
        return false;
      remaining -= seg_len;
    } else {
      void *found = memmem(pos, (size_t)remaining, seg.data(), seg.size());
      if (!found) return false;
      int32_t offset = static_cast<int32_t>((const char *)found - pos) + seg_len;
      pos += offset; remaining -= offset;
    }
  }
  return true;
}

static bool ExprContainsLike(const AQPExpr *expr) {
  if (!expr) return false;
  if (expr->GetNodeType() == VarConstComparisonNode) {
    auto et = expr->GetSimplestExprType();
    return et == TextLike || et == Text_Not_Like;
  }
  if (expr->GetNodeType() == LogicalExprNode) {
    auto *l = static_cast<const SimplestLogicalExpr *>(expr);
    return ExprContainsLike(l->left_expr.get()) ||
           ExprContainsLike(l->right_expr.get());
  }
  return false;
}

struct LeafTable {
  std::string name;
  unsigned int ir_table_index;
  bool is_base;
  const FlatTable *flat = nullptr;
  // All filter qual_vecs for this leaf (from ScanNode and/or wrapping FilterNode)
  std::vector<const std::vector<std::unique_ptr<AQPExpr>> *> all_filters;

  bool HasFilters() const {
    for (const auto *f : all_filters)
      if (f && !f->empty())
        return true;
    return false;
  }
};

static bool LeafHasLikeFilter(const LeafTable *leaf) {
  for (const auto *qual_vec : leaf->all_filters) {
    if (!qual_vec) continue;
    for (const auto &expr : *qual_vec) {
      if (ExprContainsLike(expr.get())) return true;
    }
  }
  return false;
}

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
                   bool &has_aggregate,
                   bool &has_unattached_filter) {
  if (!node)
    return;

  auto ntype = node->GetNodeType();

  if (ntype == FilterNode) {
    if (!node->qual_vec.empty()) {
      // Find which leaf this filter wraps and attach qual_vec
      if (!node->children.empty()) {
        auto *child = node->children[0].get();
        if (child && child->GetNodeType() == ScanNode) {
          auto *scan = static_cast<const SimplestScan *>(child);
          CollectLeaves(child, leaves, storage_plan, kernel_temps,
                        has_aggregate, has_unattached_filter);
          for (auto &leaf : leaves) {
            if (leaf.ir_table_index == scan->GetTableIndex()) {
              leaf.all_filters.push_back(&node->qual_vec);
              break;
            }
          }
          return;
        }
      }
      // FilterNode wraps something other than ScanNode — can't handle
      has_unattached_filter = true;
    }
  }

  if (ntype == AggregateNode) {
    has_aggregate = true;
  }

  if (ntype == ScanNode) {
    auto *scan = static_cast<const SimplestScan *>(node);
    LeafTable leaf;
    leaf.name = scan->GetTableName();
    leaf.ir_table_index = scan->GetTableIndex();
    leaf.is_base = true;
    leaf.flat = storage_plan ? storage_plan->GetTable(leaf.name) : nullptr;
    if (!node->qual_vec.empty())
      leaf.all_filters.push_back(&node->qual_vec);
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
                  has_aggregate, has_unattached_filter);
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

// ============================================================================
// Filter compilation: convert IR predicates to RowPredicate closures
// ============================================================================

// Try to compile a single AQPExpr into a RowPredicate for the given FlatTable.
// Returns empty function if the predicate type is unsupported.
RowPredicate CompileOnePredicate(const AQPExpr *expr, const FlatTable *table) {
  if (!expr || !table)
    return {};

  auto expr_type = expr->GetSimplestExprType();

  // IS NULL / IS NOT NULL
  if (expr_type == NullType || expr_type == NonNullType) {
    auto *isnull = static_cast<const SimplestIsNullExpr *>(expr);
    int col_idx = table->FindColumn(isnull->attr->GetColumnName());
    if (col_idx < 0)
      return {};
    bool want_null = (expr_type == NullType);
    return [col_idx, want_null](const FlatTable &t, uint64_t row) -> bool {
      bool is_null = t.columns[col_idx].IsNull(row);
      return want_null ? is_null : !is_null;
    };
  }

  // VarConstComparison: column OP constant
  if (expr->GetNodeType() == VarConstComparisonNode) {
    auto *cmp = static_cast<const SimplestVarConstComparison *>(expr);
    int col_idx = table->FindColumn(cmp->attr->GetColumnName());
    if (col_idx < 0)
      return {};
    auto col_type = table->columns[col_idx].type;
    auto cmp_type = cmp->GetSimplestExprType();
    auto var_type = cmp->const_var->GetType();

    // INT32 column vs int constant
    if (col_type == FlatColumnType::INT32 && var_type == IntVar) {
      int32_t const_val = cmp->const_var->GetIntValue();
      switch (cmp_type) {
      case Equal:
        return [col_idx, const_val](const FlatTable &t, uint64_t row) {
          return !t.columns[col_idx].IsNull(row) &&
                 t.columns[col_idx].GetInt32(row) == const_val;
        };
      case NotEqual:
        return [col_idx, const_val](const FlatTable &t, uint64_t row) {
          return !t.columns[col_idx].IsNull(row) &&
                 t.columns[col_idx].GetInt32(row) != const_val;
        };
      case LessThan:
        return [col_idx, const_val](const FlatTable &t, uint64_t row) {
          return !t.columns[col_idx].IsNull(row) &&
                 t.columns[col_idx].GetInt32(row) < const_val;
        };
      case GreaterThan:
        return [col_idx, const_val](const FlatTable &t, uint64_t row) {
          return !t.columns[col_idx].IsNull(row) &&
                 t.columns[col_idx].GetInt32(row) > const_val;
        };
      case LessEqual:
        return [col_idx, const_val](const FlatTable &t, uint64_t row) {
          return !t.columns[col_idx].IsNull(row) &&
                 t.columns[col_idx].GetInt32(row) <= const_val;
        };
      case GreaterEqual:
        return [col_idx, const_val](const FlatTable &t, uint64_t row) {
          return !t.columns[col_idx].IsNull(row) &&
                 t.columns[col_idx].GetInt32(row) >= const_val;
        };
      default:
        return {};
      }
    }

    // VARCHAR column vs string constant
    if (col_type == FlatColumnType::VARCHAR && var_type == StringVar) {
      std::string const_str = cmp->const_var->GetStringValue();
      switch (cmp_type) {
      case Equal:
        return [col_idx, const_str](const FlatTable &t, uint64_t row) {
          if (t.columns[col_idx].IsNull(row))
            return false;
          uint32_t len;
          const char *ptr = t.columns[col_idx].GetVarchar(row, len);
          return len == const_str.size() &&
                 std::memcmp(ptr, const_str.data(), len) == 0;
        };
      case NotEqual:
        return [col_idx, const_str](const FlatTable &t, uint64_t row) {
          if (t.columns[col_idx].IsNull(row))
            return false;
          uint32_t len;
          const char *ptr = t.columns[col_idx].GetVarchar(row, len);
          return len != const_str.size() ||
                 std::memcmp(ptr, const_str.data(), len) != 0;
        };
      case TextLike:
      case Text_Not_Like: {
        bool negate = (cmp_type == Text_Not_Like);
        std::string literal;
        LikeSegments seg_info;
        LikePatternKind kind = ClassifyLikePatternEx(const_str, literal, seg_info);

        if (kind == LIKE_EQUALITY) {
          std::string lit = literal;
          return [col_idx, lit, negate](const FlatTable &t, uint64_t row) {
            if (t.columns[col_idx].IsNull(row)) return false;
            uint32_t len;
            const char *ptr = t.columns[col_idx].GetVarchar(row, len);
            bool match = len == lit.size() &&
                         std::memcmp(ptr, lit.data(), len) == 0;
            return negate ? !match : match;
          };
        }
        if (kind == LIKE_CONTAINS) {
          std::string needle = literal;
          return [col_idx, needle, negate](const FlatTable &t, uint64_t row) {
            if (t.columns[col_idx].IsNull(row)) return false;
            uint32_t len;
            const char *ptr = t.columns[col_idx].GetVarchar(row, len);
            bool match = len >= needle.size() &&
                         memmem(ptr, len, needle.data(), needle.size()) != nullptr;
            return negate ? !match : match;
          };
        }
        if (kind == LIKE_PREFIX) {
          std::string prefix = literal;
          return [col_idx, prefix, negate](const FlatTable &t, uint64_t row) {
            if (t.columns[col_idx].IsNull(row)) return false;
            uint32_t len;
            const char *ptr = t.columns[col_idx].GetVarchar(row, len);
            bool match = len >= prefix.size() &&
                         std::memcmp(ptr, prefix.data(), prefix.size()) == 0;
            return negate ? !match : match;
          };
        }
        if (kind == LIKE_SUFFIX) {
          std::string suffix = literal;
          return [col_idx, suffix, negate](const FlatTable &t, uint64_t row) {
            if (t.columns[col_idx].IsNull(row)) return false;
            uint32_t len;
            const char *ptr = t.columns[col_idx].GetVarchar(row, len);
            bool match = len >= suffix.size() &&
                         std::memcmp(ptr + len - suffix.size(), suffix.data(),
                                     suffix.size()) == 0;
            return negate ? !match : match;
          };
        }
        if (kind == LIKE_MULTI_SEGMENT) {
          auto segs = std::make_shared<LikeSegments>(seg_info);
          return [col_idx, segs, negate](const FlatTable &t, uint64_t row) {
            if (t.columns[col_idx].IsNull(row)) return false;
            uint32_t len;
            const char *ptr = t.columns[col_idx].GetVarchar(row, len);
            bool match = LikeMatchSegments(ptr, len, *segs);
            return negate ? !match : match;
          };
        }
        // LIKE_COMPLEX: DP-based fallback
        std::string pat = const_str;
        return [col_idx, pat, negate](const FlatTable &t, uint64_t row) {
          if (t.columns[col_idx].IsNull(row)) return false;
          uint32_t len;
          const char *ptr = t.columns[col_idx].GetVarchar(row, len);
          bool match = LikeMatch(ptr, len, pat.data(), pat.size());
          return negate ? !match : match;
        };
      }
      default:
        return {};
      }
    }

    return {};
  }

  // IN expression: column IN (v1, v2, ...)
  if (expr->GetNodeType() == InExprNode) {
    auto *in_expr = static_cast<const SimplestInExpr *>(expr);
    int col_idx = table->FindColumn(in_expr->attr->GetColumnName());
    if (col_idx < 0)
      return {};
    auto col_type = table->columns[col_idx].type;
    bool negated = in_expr->negated;

    if (col_type == FlatColumnType::INT32) {
      std::unordered_set<int32_t> val_set;
      for (const auto &v : in_expr->values) {
        if (v->GetType() == IntVar)
          val_set.insert(v->GetIntValue());
      }
      return [col_idx, val_set, negated](const FlatTable &t, uint64_t row) {
        if (t.columns[col_idx].IsNull(row))
          return false;
        bool found = val_set.count(t.columns[col_idx].GetInt32(row)) > 0;
        return negated ? !found : found;
      };
    }

    if (col_type == FlatColumnType::VARCHAR) {
      std::unordered_set<std::string> val_set;
      for (const auto &v : in_expr->values) {
        if (v->GetType() == StringVar)
          val_set.insert(v->GetStringValue());
      }
      return [col_idx, val_set, negated](const FlatTable &t, uint64_t row) {
        if (t.columns[col_idx].IsNull(row))
          return false;
        std::string s = t.columns[col_idx].GetString(row);
        bool found = val_set.count(s) > 0;
        return negated ? !found : found;
      };
    }

    return {};
  }

  // Logical AND / OR
  if (expr->GetNodeType() == LogicalExprNode) {
    auto *logic = static_cast<const SimplestLogicalExpr *>(expr);
    auto op = logic->GetLogicalOp();

    if (op == LogicalAnd) {
      auto left_pred = CompileOnePredicate(logic->left_expr.get(), table);
      auto right_pred = CompileOnePredicate(logic->right_expr.get(), table);
      if (!left_pred || !right_pred)
        return {};
      return [left_pred, right_pred](const FlatTable &t, uint64_t row) {
        return left_pred(t, row) && right_pred(t, row);
      };
    }

    if (op == LogicalOr) {
      auto left_pred = CompileOnePredicate(logic->left_expr.get(), table);
      auto right_pred = CompileOnePredicate(logic->right_expr.get(), table);
      if (!left_pred || !right_pred)
        return {};
      return [left_pred, right_pred](const FlatTable &t, uint64_t row) {
        return left_pred(t, row) || right_pred(t, row);
      };
    }

    if (op == LogicalNot) {
      auto inner_pred = CompileOnePredicate(logic->right_expr.get(), table);
      if (!inner_pred)
        return {};
      return [inner_pred](const FlatTable &t, uint64_t row) {
        return !inner_pred(t, row);
      };
    }
  }

  return {};
}

// Compile all predicates from a leaf's filter sources for a given table.
// Returns true if ALL predicates compiled successfully.
// Returns false if any predicate is unsupported (caller should fall back to DuckDB).
bool CompileAllLeafFilters(
    const std::vector<const std::vector<std::unique_ptr<AQPExpr>> *> &all_filters,
    const FlatTable *table,
    std::vector<RowPredicate> &out) {
  for (const auto *qual_vec : all_filters) {
    if (!qual_vec)
      continue;
    for (const auto &expr : *qual_vec) {
      auto pred = CompileOnePredicate(expr.get(), table);
      if (!pred)
        return false;
      out.push_back(std::move(pred));
    }
  }
  return true;
}

// Build a pk_bitset and pk_to_row map for a dimension table with filters.
// Scans all rows, evaluates filters, collects PK values (from "id" column) of matching rows.
// pk_to_row maps PK value → row index (for inner join to retrieve columns from dim table).
// Returns false if PK column not found.
bool BuildFilteredPKBitset(const FlatTable *dim_table,
                           const std::vector<RowPredicate> &filters,
                           std::vector<bool> &pk_bitset,
                           std::vector<uint32_t> &pk_to_row) {
  int pk_col = dim_table->FindColumn("id");
  if (pk_col < 0)
    return false;
  if (dim_table->columns[pk_col].type != FlatColumnType::INT32)
    return false;

  int32_t max_pk = dim_table->max_pk;
  if (max_pk < 0) {
    const auto *pk_data =
        reinterpret_cast<const int32_t *>(dim_table->columns[pk_col].data.get());
    max_pk = 0;
    for (uint64_t r = 0; r < dim_table->row_count; r++) {
      if (pk_data[r] > max_pk)
        max_pk = pk_data[r];
    }
  }

  size_t domain = static_cast<size_t>(max_pk) + 1;
  pk_bitset.assign(domain, false);
  pk_to_row.assign(domain, 0);

  for (uint64_t r = 0; r < dim_table->row_count; r++) {
    bool pass = true;
    for (const auto &f : filters) {
      if (!f(*dim_table, r)) {
        pass = false;
        break;
      }
    }
    if (pass) {
      int32_t pk = dim_table->columns[pk_col].GetInt32(r);
      if (pk >= 0 && static_cast<size_t>(pk) < domain) {
        pk_bitset[pk] = true;
        pk_to_row[pk] = static_cast<uint32_t>(r);
      }
    }
  }

  return true;
}

// ============================================================================
// FlatTableBuilder
// ============================================================================

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

#ifdef HAVE_OPENMP
static FlatTableBuilder MergeBuilders(std::vector<FlatTableBuilder> &builders) {
  FlatTableBuilder merged;
  if (builders.empty())
    return merged;

  uint64_t total_rows = 0;
  for (const auto &b : builders)
    total_rows += b.row_count;

  merged.column_names = builders[0].column_names;
  merged.col_buffers.resize(builders[0].col_buffers.size());
  merged.row_count = total_rows;

  for (size_t c = 0; c < merged.col_buffers.size(); c++) {
    merged.col_buffers[c].type = builders[0].col_buffers[c].type;
    if (merged.col_buffers[c].type == FlatColumnType::INT32) {
      merged.col_buffers[c].int_data.reserve(total_rows);
      for (auto &b : builders) {
        auto &src = b.col_buffers[c].int_data;
        merged.col_buffers[c].int_data.insert(
            merged.col_buffers[c].int_data.end(),
            std::make_move_iterator(src.begin()),
            std::make_move_iterator(src.end()));
        src.clear();
        src.shrink_to_fit();
      }
    } else {
      merged.col_buffers[c].str_data.reserve(total_rows);
      for (auto &b : builders) {
        auto &src = b.col_buffers[c].str_data;
        merged.col_buffers[c].str_data.insert(
            merged.col_buffers[c].str_data.end(),
            std::make_move_iterator(src.begin()),
            std::make_move_iterator(src.end()));
        src.clear();
        src.shrink_to_fit();
      }
    }
  }
  return merged;
}
#endif

} // anonymous namespace

SubQueryPlan AnalyzeSubIR(
    const ir_sql_converter::AQPStmt *sub_ir,
    const StoragePlan *storage_plan,
    const std::unordered_map<std::string, const FlatTable *> &kernel_temps,
    const std::unordered_map<std::string, CSRIndex> &runtime_csrs,
    const DimensionCache *dim_cache) {

  SubQueryPlan plan;
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

  // Only support inner joins for now
  if (!edges.empty() && join_type != Inner)
    return plan;

  // ====== Dimension resolution ======
  // Resolve dimension table joins to constant filters on the FK column.
  // This can reduce a 2-table join to a 1-table filtered scan, or
  // a 3-table join to a 2-table join.
  struct DimResolution {
    size_t leaf_idx;              // index in leaves[] being eliminated
    size_t edge_idx;              // index in edges[] being eliminated
    unsigned int other_tbl_idx;   // ir_table_index of the non-dim leaf getting the filter
    int fk_col_idx;               // FK column index in the non-dim leaf's FlatTable
    std::vector<int32_t> pk_vals; // resolved PK values
  };
  std::vector<DimResolution> dim_resolutions;

  if (dim_cache) {
    for (size_t li = 0; li < leaves.size(); li++) {
      const auto &leaf = leaves[li];
      if (!leaf.is_base || !dim_cache->IsDimension(leaf.name))
        continue;
      if (!leaf.HasFilters())
        continue;

      auto pk_vals = dim_cache->ResolveFilterToPKs(leaf.name, leaf.all_filters);
      if (pk_vals.empty())
        continue;

      // Find the join edge connecting this dim table to another table
      for (size_t ei = 0; ei < edges.size(); ei++) {
        const auto &e = edges[ei];
        unsigned int dim_tbl_idx = leaf.ir_table_index;
        unsigned int other_tbl_idx;
        std::string other_col_name;

        if (e.left_table_idx == dim_tbl_idx && e.left_col_name == "id") {
          other_tbl_idx = e.right_table_idx;
          other_col_name = e.right_col_name;
        } else if (e.right_table_idx == dim_tbl_idx && e.right_col_name == "id") {
          other_tbl_idx = e.left_table_idx;
          other_col_name = e.left_col_name;
        } else {
          continue;
        }

        // Find the other leaf
        const LeafTable *other_leaf = FindLeaf(leaves, other_tbl_idx);
        if (!other_leaf || !other_leaf->flat)
          continue;

        // Find FK column in the other table
        int fk_col = other_leaf->flat->FindColumn(other_col_name);
        if (fk_col < 0)
          continue;
        if (other_leaf->flat->columns[fk_col].type != FlatColumnType::INT32)
          continue;

        DimResolution res;
        res.leaf_idx = li;
        res.edge_idx = ei;
        res.other_tbl_idx = other_tbl_idx;
        res.fk_col_idx = fk_col;
        res.pk_vals = std::move(pk_vals);
        dim_resolutions.push_back(std::move(res));
        break;
      }
    }
  }

  // ====== Unfiltered dim elimination ======
  // When a dim table has no WHERE filters AND no output columns reference it,
  // the join is a no-op (all FK values pass). Eliminate the dim leaf + edge.
  struct UnfilteredDimElim {
    size_t leaf_idx;
    size_t edge_idx;
  };
  std::vector<UnfilteredDimElim> unfiltered_elims;

  if (dim_cache) {
    // Collect output table indices to check if any column references the dim
    std::unordered_set<unsigned int> output_tbl_indices;
    for (const auto &attr : sub_ir->target_list)
      output_tbl_indices.insert(attr->GetTableIndex());

    for (size_t li = 0; li < leaves.size(); li++) {
      const auto &leaf = leaves[li];
      if (!leaf.is_base || !dim_cache->IsDimension(leaf.name))
        continue;
      if (leaf.HasFilters())
        continue; // filtered dims handled above

      // Already scheduled for filtered dim resolution?
      bool already_resolved = false;
      for (const auto &dr : dim_resolutions) {
        if (dr.leaf_idx == li) { already_resolved = true; break; }
      }
      if (already_resolved)
        continue;

      // Output must not reference this dim table
      if (output_tbl_indices.count(leaf.ir_table_index))
        continue;

      // Find the join edge with "id" on the dim side
      for (size_t ei = 0; ei < edges.size(); ei++) {
        const auto &e = edges[ei];
        bool dim_on_left = (e.left_table_idx == leaf.ir_table_index && e.left_col_name == "id");
        bool dim_on_right = (e.right_table_idx == leaf.ir_table_index && e.right_col_name == "id");
        if (!dim_on_left && !dim_on_right)
          continue;

        // Already scheduled for erase by filtered dim resolution?
        bool edge_taken = false;
        for (const auto &dr : dim_resolutions) {
          if (dr.edge_idx == ei) { edge_taken = true; break; }
        }
        if (edge_taken)
          continue;

        unfiltered_elims.push_back({li, ei});
        break;
      }
    }
  }

  // Apply dimension resolutions: add FK filters, then erase leaves/edges
  // Collect filters first using stored other_tbl_idx (safe regardless of erase order)
  std::unordered_map<unsigned int, std::vector<std::pair<int, std::vector<int32_t>>>>
      dim_derived_filters;

  for (const auto &res : dim_resolutions) {
    dim_derived_filters[res.other_tbl_idx].emplace_back(res.fk_col_idx, res.pk_vals);
  }

  // Erase leaves and edges by index — collect from both filtered dim resolutions
  // and unfiltered dim eliminations, sort descending, erase
  {
    std::vector<size_t> leaf_idxs, edge_idxs;
    for (const auto &res : dim_resolutions) {
      leaf_idxs.push_back(res.leaf_idx);
      edge_idxs.push_back(res.edge_idx);
    }
    for (const auto &elim : unfiltered_elims) {
      leaf_idxs.push_back(elim.leaf_idx);
      edge_idxs.push_back(elim.edge_idx);
    }
    std::sort(leaf_idxs.rbegin(), leaf_idxs.rend());
    std::sort(edge_idxs.rbegin(), edge_idxs.rend());
    // Deduplicate (in case of overlap, though shouldn't happen)
    leaf_idxs.erase(std::unique(leaf_idxs.begin(), leaf_idxs.end()), leaf_idxs.end());
    edge_idxs.erase(std::unique(edge_idxs.begin(), edge_idxs.end()), edge_idxs.end());
    for (size_t idx : leaf_idxs)
      leaves.erase(leaves.begin() + static_cast<long>(idx));
    for (size_t idx : edge_idxs)
      edges.erase(edges.begin() + static_cast<long>(idx));
  }

  // ====== Inverted index resolution ======
  // When dim resolution leaves 2 base tables (e.g., movie_keyword + title),
  // check if an inverted index can eliminate the bridge table entirely.
  // Pattern: dim(filtered) → bridge_table.fk_col → bridge_table.target_col → target_table
  // If inverted index exists for dim→target through bridge, we can:
  //   1. Use dim PK values to look up target PK values directly
  //   2. Build a bitset of target PK values
  //   3. Eliminate the bridge leaf + edge, leaving a single target table scan
  if (!dim_resolutions.empty() && leaves.size() == 2 && edges.size() == 1 && storage_plan) {
    bool all_base = true;
    for (const auto &leaf : leaves) {
      if (!leaf.is_base) { all_base = false; break; }
    }

    if (all_base) {
      // Identify which leaf is the bridge table (the one that received dim FK filters)
      // and which is the target table (joined to bridge via the remaining edge).
      const JoinEdge &remaining_edge = edges[0];
      const LeafTable *bridge_leaf = nullptr;
      const LeafTable *target_leaf = nullptr;
      std::string bridge_col_name, target_col_name;

      for (const auto &leaf : leaves) {
        // The bridge table is the one that got dim-derived filters
        if (dim_derived_filters.count(leaf.ir_table_index)) {
          bridge_leaf = &leaf;
        } else {
          target_leaf = &leaf;
        }
      }

      bool inverted_resolved = false;
      if (bridge_leaf && target_leaf) {
        // Resolve edge column names
        if (remaining_edge.left_table_idx == bridge_leaf->ir_table_index) {
          bridge_col_name = remaining_edge.left_col_name;
          target_col_name = remaining_edge.right_col_name;
        } else if (remaining_edge.right_table_idx == bridge_leaf->ir_table_index) {
          bridge_col_name = remaining_edge.right_col_name;
          target_col_name = remaining_edge.left_col_name;
        }

        if (!bridge_col_name.empty()) {
          // For each dim resolution that targeted the bridge table,
          // check if an inverted index exists
          auto dim_it = dim_derived_filters.find(bridge_leaf->ir_table_index);
          if (dim_it != dim_derived_filters.end()) {
            for (const auto &res : dim_resolutions) {
              if (res.other_tbl_idx != bridge_leaf->ir_table_index)
                continue;
              // Find the dim table name from the original leaves (already erased)
              // We stored it in the DimResolution — but we only have pk_vals.
              // Instead, look up inverted index by bridge_table + bridge_fk_col
              // The dim_derived_filter's fk_col_idx tells us the FK column in bridge_table.
              std::string fk_col_name;
              if (res.fk_col_idx >= 0 && res.fk_col_idx < static_cast<int>(bridge_leaf->flat->column_names.size()))
                fk_col_name = bridge_leaf->flat->column_names[res.fk_col_idx];

              if (fk_col_name.empty())
                continue;

              // Search for inverted index matching this bridge+fk_col→target pattern
              const InvertedIndex *inv = nullptr;
              for (const auto &kv : storage_plan->GetInvertedIndicesMap()) {
                const auto &idx = kv.second;
                if (idx.bridge_table == bridge_leaf->name &&
                    idx.bridge_fk_col == fk_col_name &&
                    idx.target_col == bridge_col_name &&
                    idx.target_table == target_leaf->name) {
                  inv = &idx;
                  break;
                }
              }
              // Also try: bridge_col_name is the FK in bridge that joins to target.
              // The inverted index target_col should match bridge_col_name.
              if (!inv) {
                for (const auto &kv : storage_plan->GetInvertedIndicesMap()) {
                  const auto &idx = kv.second;
                  if (idx.bridge_table == bridge_leaf->name &&
                      idx.bridge_fk_col == fk_col_name &&
                      idx.target_col == bridge_col_name) {
                    inv = &idx;
                    break;
                  }
                }
              }

              if (!inv)
                continue;

              // Use the inverted index: dim PK values → target PK values
              // Build a bitset of qualifying target PK values
              int target_pk_col = target_leaf->flat->FindColumn(target_col_name);
              if (target_pk_col < 0)
                continue;
              if (target_leaf->flat->columns[target_pk_col].type != FlatColumnType::INT32)
                continue;

              // Collect all target values from inverted index for these dim PKs
              std::unordered_set<int32_t> target_vals_set;
              for (int32_t pk : res.pk_vals) {
                auto result = inv->Lookup(pk);
                if (result.first && result.second) {
                  for (auto it = result.first; it != result.second; ++it)
                    target_vals_set.insert(*it);
                }
              }

              if (target_vals_set.empty())
                continue;

              // Selectivity guard: only use inverted index when the resulting
              // target set covers less than 50% of the target table.
              // Otherwise DuckDB's hash join is likely faster.
              uint64_t target_rows = target_leaf->flat->row_count;
              if (target_vals_set.size() > target_rows / 2)
                continue;

              // Replace the dim-derived FK filter on the bridge table with a
              // target PK filter on the target table.
              // Remove the bridge leaf's dim filter for this resolution
              // and add target PK IN-filter on target leaf.
              std::vector<int32_t> target_vals(target_vals_set.begin(), target_vals_set.end());
              dim_derived_filters[target_leaf->ir_table_index].emplace_back(target_pk_col, target_vals);

              // Remove the bridge's dim filter for this particular resolution
              // (it was for fk_col_idx in bridge)
              auto &bridge_filters = dim_derived_filters[bridge_leaf->ir_table_index];
              for (auto fit = bridge_filters.begin(); fit != bridge_filters.end(); ++fit) {
                if (fit->first == res.fk_col_idx) {
                  bridge_filters.erase(fit);
                  break;
                }
              }

              inverted_resolved = true;
              break;
            }
          }
        }
      }

      if (inverted_resolved) {
        // Bridge table is now unnecessary if it has no remaining filters/output cols.
        // Eliminate bridge leaf and remaining edge, leaving target as single table.
        size_t bridge_idx = 0;
        for (size_t i = 0; i < leaves.size(); i++) {
          if (&leaves[i] == bridge_leaf) { bridge_idx = i; break; }
        }
        leaves.erase(leaves.begin() + static_cast<long>(bridge_idx));
        edges.clear();
        dim_derived_filters.erase(bridge_leaf->ir_table_index);
      } else {
        // No inverted index available — fall back to DuckDB for base×base.
        // Base tables can have mismatched column names in the IR (e.g.,
        // movie_link col_idx=2 is "linked_movie_id" but IR labels it "movie_id").
        return plan;
      }
    }
  }

  // ====== 3-table inverted index resolution ======
  // Pattern: source(filtered) + bridge + target where inverted index maps
  // source→target through bridge. Eliminates source+bridge, leaves single target.
  // Column remapping: bridge join-key columns → target equivalent (they're equal via join).
  std::unordered_map<uint64_t, std::pair<unsigned int, std::string>> inv_col_remap;

  if (leaves.size() == 3 && edges.size() == 2 && storage_plan) {
    bool all_base = true;
    for (const auto &leaf : leaves) {
      if (!leaf.is_base) { all_base = false; break; }
    }
    bool all_have_flat = true;
    for (const auto &leaf : leaves) {
      if (!leaf.flat) { all_have_flat = false; break; }
    }

    if (all_base && all_have_flat) {
      bool inv3_resolved = false;

      for (const auto &kv : storage_plan->GetInvertedIndicesMap()) {
        const auto &inv = kv.second;

        // Match leaves to roles
        const LeafTable *source_leaf = nullptr;
        const LeafTable *bridge_leaf3 = nullptr;
        const LeafTable *target_leaf3 = nullptr;
        for (const auto &leaf : leaves) {
          if (leaf.name == inv.dim_table && leaf.HasFilters())
            source_leaf = &leaf;
          else if (leaf.name == inv.bridge_table)
            bridge_leaf3 = &leaf;
          else if (leaf.name == inv.target_table)
            target_leaf3 = &leaf;
        }
        if (!source_leaf || !bridge_leaf3 || !target_leaf3)
          continue;

        // Verify edge topology:
        // Edge A: source.id = bridge.fk_col
        // Edge B: bridge.target_col = target.X
        bool edge_a_ok = false, edge_b_ok = false;
        std::string target_join_col;
        for (const auto &e : edges) {
          // Check source↔bridge edge
          if ((e.left_table_idx == source_leaf->ir_table_index && e.left_col_name == "id" &&
               e.right_table_idx == bridge_leaf3->ir_table_index && e.right_col_name == inv.bridge_fk_col) ||
              (e.right_table_idx == source_leaf->ir_table_index && e.right_col_name == "id" &&
               e.left_table_idx == bridge_leaf3->ir_table_index && e.left_col_name == inv.bridge_fk_col)) {
            edge_a_ok = true;
          }
          // Check bridge↔target edge
          if (e.left_table_idx == bridge_leaf3->ir_table_index && e.left_col_name == inv.target_col &&
              e.right_table_idx == target_leaf3->ir_table_index) {
            edge_b_ok = true;
            target_join_col = e.right_col_name;
          } else if (e.right_table_idx == bridge_leaf3->ir_table_index && e.right_col_name == inv.target_col &&
                     e.left_table_idx == target_leaf3->ir_table_index) {
            edge_b_ok = true;
            target_join_col = e.left_col_name;
          }
        }
        if (!edge_a_ok || !edge_b_ok || target_join_col.empty())
          continue;

        // Check output columns: all must be from target, or from bridge on the
        // join-key column (remappable to target equivalent)
        bool output_ok = true;
        for (const auto &attr : sub_ir->target_list) {
          unsigned int tbl_idx = attr->GetTableIndex();
          if (tbl_idx == target_leaf3->ir_table_index)
            continue;
          if (tbl_idx == bridge_leaf3->ir_table_index) {
            // Only the bridge join-key column can be remapped
            if (attr->GetColumnName() == inv.target_col) {
              continue; // will be remapped to target_join_col
            }
          }
          // Output from source or non-remappable bridge col → can't resolve
          output_ok = false;
          break;
        }
        if (!output_ok)
          continue;

        // Compile source filters
        std::vector<RowPredicate> source_preds;
        if (!CompileAllLeafFilters(source_leaf->all_filters, source_leaf->flat, source_preds))
          continue;

        // Scan source table with compiled filters → collect matching PK values
        int source_pk_col = source_leaf->flat->FindColumn("id");
        if (source_pk_col < 0 || source_leaf->flat->columns[source_pk_col].type != FlatColumnType::INT32)
          continue;

        std::vector<int32_t> source_pks;
        for (uint64_t r = 0; r < source_leaf->flat->row_count; r++) {
          bool pass = true;
          for (const auto &pred : source_preds) {
            if (!pred(*source_leaf->flat, r)) { pass = false; break; }
          }
          if (pass && !source_leaf->flat->columns[source_pk_col].IsNull(r))
            source_pks.push_back(source_leaf->flat->columns[source_pk_col].GetInt32(r));
        }
        if (source_pks.empty())
          continue;

        // Inverted index lookup → collect target PK values
        std::unordered_set<int32_t> target_vals_set;
        for (int32_t pk : source_pks) {
          auto result = inv.Lookup(pk);
          if (result.first && result.second) {
            for (auto it = result.first; it != result.second; ++it)
              target_vals_set.insert(*it);
          }
        }
        if (target_vals_set.empty())
          continue;

        // Selectivity guard
        uint64_t target_rows = target_leaf3->flat->row_count;
        if (target_vals_set.size() > target_rows / 2)
          continue;

        // Find target PK column
        int target_pk_col3 = target_leaf3->flat->FindColumn(target_join_col);
        if (target_pk_col3 < 0 || target_leaf3->flat->columns[target_pk_col3].type != FlatColumnType::INT32)
          continue;

        // Add target IN-filter
        std::vector<int32_t> target_vals(target_vals_set.begin(), target_vals_set.end());
        dim_derived_filters[target_leaf3->ir_table_index].emplace_back(target_pk_col3, target_vals);

        // Store column remapping for bridge join-key → target join-key
        for (const auto &attr : sub_ir->target_list) {
          if (attr->GetTableIndex() == bridge_leaf3->ir_table_index &&
              attr->GetColumnName() == inv.target_col) {
            uint64_t key = (static_cast<uint64_t>(bridge_leaf3->ir_table_index) << 32) |
                           std::hash<std::string>{}(inv.target_col);
            inv_col_remap[key] = {target_leaf3->ir_table_index, target_join_col};
          }
        }

        // Save indices before erasing (pointers become dangling after erase)
        unsigned int source_ir_idx = source_leaf->ir_table_index;
        unsigned int bridge_ir_idx = bridge_leaf3->ir_table_index;

        // Erase source + bridge leaves, clear edges
        std::vector<size_t> erase_idxs;
        for (size_t i = 0; i < leaves.size(); i++) {
          if (&leaves[i] == source_leaf || &leaves[i] == bridge_leaf3)
            erase_idxs.push_back(i);
        }
        std::sort(erase_idxs.rbegin(), erase_idxs.rend());
        for (size_t idx : erase_idxs)
          leaves.erase(leaves.begin() + static_cast<long>(idx));
        edges.clear();

        // Remove source + bridge from dim_derived_filters
        dim_derived_filters.erase(source_ir_idx);
        dim_derived_filters.erase(bridge_ir_idx);

        inv3_resolved = true;
        break;
      }
      // If not resolved, fall through (3 leaves → will fail at leaves.size() checks)
      (void)inv3_resolved;
    }
  }

  // All leaf tables must have FlatTable data
  for (const auto &leaf : leaves) {
    if (!leaf.flat)
      return plan;
  }

  // ====== Single-table filtered scan (after dim resolution) ======
  if (leaves.size() == 1 && edges.empty()) {
    const LeafTable *scan_leaf = &leaves[0];

    std::vector<RowPredicate> scan_predicates;
    if (scan_leaf->HasFilters()) {
      if (!CompileAllLeafFilters(scan_leaf->all_filters, scan_leaf->flat, scan_predicates))
        return plan;
    }

    // Guard: single-table LIKE scan on base tables → DuckDB vectorized is
    // faster unless an inverted-index PK filter drastically reduces rows.
    // Dim-derived FK filters (e.g., role_id=actor) don't help enough because
    // they filter on FK columns with low selectivity.
    // Allow kernel only when a dim_derived_filter targets the PK column ("id"),
    // which guarantees massive row reduction (inverted index pattern).
    if (scan_leaf->is_base && LeafHasLikeFilter(scan_leaf)) {
      bool has_pk_filter = false;
      auto dim_it2 = dim_derived_filters.find(scan_leaf->ir_table_index);
      if (dim_it2 != dim_derived_filters.end()) {
        int pk_col = scan_leaf->flat->FindColumn("id");
        for (const auto &filt : dim_it2->second) {
          if (filt.first == pk_col) {
            has_pk_filter = true;
            break;
          }
        }
      }
      if (!has_pk_filter)
        return plan;
    }

    // Add dimension-derived FK filters
    auto dim_it = dim_derived_filters.find(scan_leaf->ir_table_index);
    if (dim_it != dim_derived_filters.end()) {
      for (const auto &filt : dim_it->second) {
        int fk_col = filt.first;
        const auto &pk_vals = filt.second;
        if (pk_vals.size() == 1) {
          int32_t val = pk_vals[0];
          scan_predicates.push_back(
              [fk_col, val](const FlatTable &t, uint64_t row) {
                return !t.columns[fk_col].IsNull(row) &&
                       t.columns[fk_col].GetInt32(row) == val;
              });
        } else {
          auto val_set = std::make_shared<std::unordered_set<int32_t>>(
              pk_vals.begin(), pk_vals.end());
          scan_predicates.push_back(
              [fk_col, val_set](const FlatTable &t, uint64_t row) {
                return !t.columns[fk_col].IsNull(row) &&
                       val_set->count(t.columns[fk_col].GetInt32(row)) > 0;
              });
        }
      }
    }

    const auto &target_list = sub_ir->target_list;
    if (target_list.empty())
      return plan;

    std::vector<KernelOutputCol> output_cols;
    for (const auto &attr : target_list) {
      unsigned int tbl_idx = attr->GetTableIndex();
      std::string col_name = attr->GetColumnName();

      // Apply inverted index column remapping (bridge col → target col)
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

  // ====== Two-table join (original path, possibly after dim resolution) ======
  if (leaves.size() != 2)
    return plan;

  if (edges.empty())
    return plan;

  // Find a CSR or runtime CSR for the join edge
  const JoinEdge &edge = edges[0];

  // Resolve edge columns for each leaf
  auto ResolveEdgeCols = [&](const LeafTable *s, const LeafTable *l,
                             std::string &s_col, std::string &l_col) -> bool {
    if (edge.left_table_idx == s->ir_table_index) {
      s_col = edge.left_col_name;
      l_col = edge.right_col_name;
    } else if (edge.right_table_idx == s->ir_table_index) {
      s_col = edge.right_col_name;
      l_col = edge.left_col_name;
    } else {
      return false;
    }
    return true;
  };

  // Find CSR for a given scan→lookup direction.
  // CSR must satisfy: Lookup(scan_key) returns lookup_table rows,
  // i.e., csr->fk_table == lookup_table (for base CSRs).
  auto FindCSR = [&](const LeafTable *l, const std::string &l_col)
      -> const CSRIndex * {
    auto it = runtime_csrs.find(l->name + "." + l_col);
    if (it != runtime_csrs.end() && it->second.fk_table == l->name)
      return &it->second;
    if (storage_plan) {
      auto *c = storage_plan->GetCSR(l->name, l_col);
      if (c && c->fk_table == l->name)
        return c;
    }
    return nullptr;
  };

  // Determine scan table vs lookup table.
  // Prefer scanning the SMALLER table and CSR-probing the larger,
  // but fall back to the reverse if no CSR exists for that direction.
  const LeafTable *scan_leaf = &leaves[0];
  const LeafTable *lookup_leaf = &leaves[1];
  if (scan_leaf->flat->row_count > lookup_leaf->flat->row_count)
    std::swap(scan_leaf, lookup_leaf);

  std::string scan_col_name, lookup_col_name;
  if (!ResolveEdgeCols(scan_leaf, lookup_leaf, scan_col_name, lookup_col_name))
    return plan;

  const CSRIndex *csr = FindCSR(lookup_leaf, lookup_col_name);

  // No CSR for small-scans-large — try the reverse direction
  if (!csr) {
    std::swap(scan_leaf, lookup_leaf);
    if (!ResolveEdgeCols(scan_leaf, lookup_leaf, scan_col_name, lookup_col_name))
      return plan;
    csr = FindCSR(lookup_leaf, lookup_col_name);
  }

  if (!csr)
    return plan;

  // Find scan column in the FlatTable
  int scan_flat_col = scan_leaf->flat->FindColumn(scan_col_name);
  if (scan_flat_col < 0)
    return plan;

  // Only INT32 join keys supported
  if (scan_leaf->flat->columns[scan_flat_col].type != FlatColumnType::INT32)
    return plan;

  // ====== Handle filters ======
  // Compile filters for scan table and lookup table separately.
  // Lookup table filters → build pk_bitset (dimension constant resolution).
  // Scan table filters → evaluate during scan loop.
  std::vector<RowPredicate> scan_predicates;
  std::vector<RowPredicate> lookup_predicates;
  bool use_bitset = false;
  std::vector<bool> pk_bitset;
  std::vector<uint32_t> pk_to_row;

  // Compile lookup table filters
  if (lookup_leaf->HasFilters()) {
    if (!CompileAllLeafFilters(lookup_leaf->all_filters, lookup_leaf->flat, lookup_predicates))
      return plan; // unsupported filter → fall back to DuckDB

    // PK bitset only valid when lookup column IS the PK ("id").
    // Otherwise, apply filters as join_filters (evaluated per CSR match).
    if (lookup_col_name == "id") {
      if (!BuildFilteredPKBitset(lookup_leaf->flat, lookup_predicates, pk_bitset, pk_to_row))
        return plan;
      use_bitset = true;
    }
  }

  // Compile scan table filters
  if (scan_leaf->HasFilters()) {
    if (!CompileAllLeafFilters(scan_leaf->all_filters, scan_leaf->flat, scan_predicates))
      return plan; // unsupported filter → fall back to DuckDB
  }

  // Add dimension-derived FK filters to scan or lookup table
  auto dim_scan_it = dim_derived_filters.find(scan_leaf->ir_table_index);
  if (dim_scan_it != dim_derived_filters.end()) {
    for (const auto &filt : dim_scan_it->second) {
      int fk_col = filt.first;
      const auto &pk_vals = filt.second;
      if (pk_vals.size() == 1) {
        int32_t val = pk_vals[0];
        scan_predicates.push_back(
            [fk_col, val](const FlatTable &t, uint64_t row) {
              return !t.columns[fk_col].IsNull(row) &&
                     t.columns[fk_col].GetInt32(row) == val;
            });
      } else {
        auto val_set = std::make_shared<std::unordered_set<int32_t>>(
            pk_vals.begin(), pk_vals.end());
        scan_predicates.push_back(
            [fk_col, val_set](const FlatTable &t, uint64_t row) {
              return !t.columns[fk_col].IsNull(row) &&
                     val_set->count(t.columns[fk_col].GetInt32(row)) > 0;
            });
      }
    }
  }

  // Lookup table filters applied as join_filters when not using bitset
  std::vector<RowPredicate> join_filters;
  if (!use_bitset && !lookup_predicates.empty()) {
    join_filters.insert(join_filters.end(),
                        std::make_move_iterator(lookup_predicates.begin()),
                        std::make_move_iterator(lookup_predicates.end()));
  }

  // Dim-derived filters on the LOOKUP table: add as join_filters
  auto dim_lookup_it = dim_derived_filters.find(lookup_leaf->ir_table_index);
  if (dim_lookup_it != dim_derived_filters.end()) {
    for (const auto &filt : dim_lookup_it->second) {
      int fk_col = filt.first;
      const auto &pk_vals = filt.second;
      if (pk_vals.size() == 1) {
        int32_t val = pk_vals[0];
        join_filters.push_back(
            [fk_col, val](const FlatTable &t, uint64_t row) {
              return !t.columns[fk_col].IsNull(row) &&
                     t.columns[fk_col].GetInt32(row) == val;
            });
      } else {
        auto val_set = std::make_shared<std::unordered_set<int32_t>>(
            pk_vals.begin(), pk_vals.end());
        join_filters.push_back(
            [fk_col, val_set](const FlatTable &t, uint64_t row) {
              return !t.columns[fk_col].IsNull(row) &&
                     val_set->count(t.columns[fk_col].GetInt32(row)) > 0;
            });
      }
    }
  }

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
    } else if (tbl_idx == lookup_leaf->ir_table_index) {
      out.source = KernelOutputCol::FROM_JOIN;
      out.step_idx = 0;
      out.col_idx = lookup_leaf->flat->FindColumn(col_name);
      if (out.col_idx < 0)
        return plan;
      out.type = lookup_leaf->flat->columns[out.col_idx].type;
    } else {
      return plan;
    }
    out.name = col_name;
    output_cols.push_back(out);
  }

  // Build the plan
  KernelJoinStep step;
  step.scan_key_col_idx = scan_flat_col;
  step.joined_table = lookup_leaf->flat;
  step.is_semi = false;

  if (use_bitset) {
    step.use_bitset = true;
    step.pk_bitset = std::move(pk_bitset);
    step.pk_to_row = std::move(pk_to_row);
    step.csr = nullptr;
  } else {
    step.csr = csr;
  }
  step.join_filters = std::move(join_filters);

  plan.scan_table = scan_leaf->flat;
  plan.scan_table_name = scan_leaf->name;
  plan.join_steps.push_back(std::move(step));
  plan.output_cols = std::move(output_cols);
  plan.scan_filters = std::move(scan_predicates);
  plan.valid = true;

  return plan;
}

std::unique_ptr<FlatTable> ExecuteSubQueryPlan(const SubQueryPlan &plan,
                                               const std::string &table_name) {
  assert(plan.valid);

  uint64_t scan_rows = plan.scan_table->row_count;

  bool any_inner = false;
  for (const auto &step : plan.join_steps) {
    if (!step.is_semi) {
      any_inner = true;
      break;
    }
  }

  bool has_scan_filters = !plan.scan_filters.empty();

  std::vector<uint64_t> semi_dummy(plan.join_steps.size(), 0);

  // Scan loop body: process one row, emit to the given builder.
  // All reads (FlatTable, CSR, bitset, filters) are on immutable shared data.
  auto ScanRow = [&](uint64_t row, FlatTableBuilder &builder) {
    if (has_scan_filters) {
      for (const auto &f : plan.scan_filters) {
        if (!f(*plan.scan_table, row))
          return;
      }
    }

    // Helper: emit one output row
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

    if (!any_inner) {
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
          if (!step.join_filters.empty()) {
            uint64_t lr = step.pk_to_row[key];
            for (const auto &jf : step.join_filters) {
              if (!jf(*step.joined_table, lr)) {
                pass = false;
                break;
              }
            }
            if (!pass) break;
          }
        } else if (step.csr) {
          auto result = step.csr->Lookup(key);
          if (result.first == result.second) {
            pass = false;
            break;
          }
          if (!step.join_filters.empty()) {
            bool any_match = false;
            for (auto jit = result.first; jit != result.second; ++jit) {
              bool jpass = true;
              for (const auto &jf : step.join_filters) {
                if (!jf(*step.joined_table, *jit)) {
                  jpass = false;
                  break;
                }
              }
              if (jpass) { any_match = true; break; }
            }
            if (!any_match) { pass = false; break; }
          }
        } else {
          pass = false;
          break;
        }
      }
      if (!pass)
        return;
      EmitRow(row, semi_dummy);
    } else {
      assert(plan.join_steps.size() == 1);
      const auto &step = plan.join_steps[0];
      int32_t key =
          plan.scan_table->columns[step.scan_key_col_idx].GetInt32(row);
      bool has_join_filters = !step.join_filters.empty();

      if (step.use_bitset) {
        if (key < 0 || static_cast<size_t>(key) >= step.pk_bitset.size() ||
            !step.pk_bitset[key])
          return;
        uint64_t lr = step.pk_to_row[key];
        if (has_join_filters) {
          bool jpass = true;
          for (const auto &jf : step.join_filters) {
            if (!jf(*step.joined_table, lr)) { jpass = false; break; }
          }
          if (!jpass) return;
        }
        std::vector<uint64_t> joined_rows = {lr};
        EmitRow(row, joined_rows);
      } else if (step.csr) {
        auto result = step.csr->Lookup(key);
        if (result.first == result.second)
          return;
        for (auto it = result.first; it != result.second; ++it) {
          if (has_join_filters) {
            bool jpass = true;
            for (const auto &jf : step.join_filters) {
              if (!jf(*step.joined_table, *it)) { jpass = false; break; }
            }
            if (!jpass) continue;
          }
          std::vector<uint64_t> joined_rows = {*it};
          EmitRow(row, joined_rows);
        }
      }
    }
  };

#ifdef HAVE_OPENMP
  if (scan_rows >= OMP_PARALLEL_THRESHOLD) {
    int nthreads = std::min(12, omp_get_max_threads());
    std::vector<FlatTableBuilder> thread_builders(nthreads);
    for (auto &tb : thread_builders)
      tb.Init(plan.output_cols);

    #pragma omp parallel num_threads(nthreads)
    {
      int tid = omp_get_thread_num();
      FlatTableBuilder &my_builder = thread_builders[tid];

      #pragma omp for schedule(dynamic, 4096)
      for (int64_t row = 0; row < static_cast<int64_t>(scan_rows); row++) {
        ScanRow(static_cast<uint64_t>(row), my_builder);
      }
    }

    auto merged = MergeBuilders(thread_builders);
    return merged.Finalize(table_name);
  }
#endif

  FlatTableBuilder builder;
  builder.Init(plan.output_cols);
  builder.Reserve(scan_rows / 4);
  for (uint64_t row = 0; row < scan_rows; row++) {
    ScanRow(row, builder);
  }
  return builder.Finalize(table_name);
}

// ============================================================================
// Final aggregate: AnalyzeFinalIR + ExecuteFinalAggregate
// ============================================================================

FinalAggregatePlan AnalyzeFinalIR(
    const ir_sql_converter::AQPStmt *ir,
    const StoragePlan *storage_plan,
    const std::unordered_map<std::string, const FlatTable *> &kernel_temps,
    const std::unordered_map<std::string, CSRIndex> &runtime_csrs,
    const DimensionCache *dim_cache) {

  FinalAggregatePlan plan;
  plan.valid = false;

  if (!ir || !storage_plan)
    return plan;

  // Expect: Projection → Aggregate → child
  if (ir->GetNodeType() != ProjectionNode || ir->children.size() != 1)
    return plan;
  const auto *proj = ir;
  const auto *agg_node = proj->children[0].get();
  if (!agg_node || agg_node->GetNodeType() != AggregateNode)
    return plan;

  const auto *agg = static_cast<const SimplestAggregate *>(agg_node);
  if (!agg->groups.empty())
    return plan;

  // All agg_fns must be MIN
  for (const auto &fn : agg->agg_fns) {
    if (fn.second != SimplestAggFnType::Min)
      return plan;
  }

  // Get the child of Aggregate (the join tree)
  if (agg->children.empty())
    return plan;
  const auto *child = agg->children[0].get();
  if (!child)
    return plan;

  // Extract output column names from Projection's target_list
  for (const auto &attr : proj->target_list) {
    plan.output_names.push_back(attr->GetColumnName());
  }

  // Collect leaves from the child (skip aggregate detection)
  std::vector<LeafTable> leaves;
  bool has_aggregate_dummy = false;
  bool has_unattached_filter = false;
  CollectLeaves(child, leaves, storage_plan, kernel_temps,
                has_aggregate_dummy, has_unattached_filter);

  if (has_unattached_filter)
    return plan;

  SimplestJoinType join_type = InvalidJoinType;
  std::vector<JoinEdge> edges;
  CollectJoinEdges(child, edges, join_type);

  if (!edges.empty() && join_type != Inner)
    return plan;

  // Dimension resolution (same as AnalyzeSubIR)
  struct DimResolution {
    size_t leaf_idx;
    size_t edge_idx;
    unsigned int other_tbl_idx;
    int fk_col_idx;
    std::vector<int32_t> pk_vals;
  };
  std::vector<DimResolution> dim_resolutions;

  if (dim_cache) {
    for (size_t li = 0; li < leaves.size(); li++) {
      const auto &leaf = leaves[li];
      if (!leaf.is_base || !dim_cache->IsDimension(leaf.name))
        continue;
      if (!leaf.HasFilters())
        continue;
      auto pk_vals = dim_cache->ResolveFilterToPKs(leaf.name, leaf.all_filters);
      if (pk_vals.empty())
        continue;
      for (size_t ei = 0; ei < edges.size(); ei++) {
        const auto &e = edges[ei];
        unsigned int dim_tbl_idx = leaf.ir_table_index;
        unsigned int other_tbl_idx;
        std::string other_col_name;
        if (e.left_table_idx == dim_tbl_idx && e.left_col_name == "id") {
          other_tbl_idx = e.right_table_idx;
          other_col_name = e.right_col_name;
        } else if (e.right_table_idx == dim_tbl_idx && e.right_col_name == "id") {
          other_tbl_idx = e.left_table_idx;
          other_col_name = e.left_col_name;
        } else {
          continue;
        }
        const LeafTable *other_leaf = FindLeaf(leaves, other_tbl_idx);
        if (!other_leaf || !other_leaf->flat)
          continue;
        int fk_col = other_leaf->flat->FindColumn(other_col_name);
        if (fk_col < 0 || other_leaf->flat->columns[fk_col].type != FlatColumnType::INT32)
          continue;
        DimResolution res;
        res.leaf_idx = li;
        res.edge_idx = ei;
        res.other_tbl_idx = other_tbl_idx;
        res.fk_col_idx = fk_col;
        res.pk_vals = std::move(pk_vals);
        dim_resolutions.push_back(std::move(res));
        break;
      }
    }
  }

  std::unordered_map<unsigned int, std::vector<std::pair<int, std::vector<int32_t>>>>
      dim_derived_filters;
  for (const auto &res : dim_resolutions)
    dim_derived_filters[res.other_tbl_idx].emplace_back(res.fk_col_idx, res.pk_vals);

  {
    std::vector<size_t> leaf_idxs, edge_idxs;
    for (const auto &res : dim_resolutions) {
      leaf_idxs.push_back(res.leaf_idx);
      edge_idxs.push_back(res.edge_idx);
    }
    std::sort(leaf_idxs.rbegin(), leaf_idxs.rend());
    std::sort(edge_idxs.rbegin(), edge_idxs.rend());
    for (size_t idx : leaf_idxs)
      leaves.erase(leaves.begin() + static_cast<long>(idx));
    for (size_t idx : edge_idxs)
      edges.erase(edges.begin() + static_cast<long>(idx));
  }

  // All leaves must have FlatTable data
  for (const auto &leaf : leaves)
    if (!leaf.flat)
      return plan;

  if (leaves.size() < 2 || edges.empty())
    return plan;

  // Find the star center: the table that appears in the most edges.
  std::unordered_map<unsigned int, int> edge_count;
  for (const auto &e : edges) {
    edge_count[e.left_table_idx]++;
    edge_count[e.right_table_idx]++;
  }

  unsigned int center_tbl_idx = 0;
  int max_count = 0;
  for (const auto &kv : edge_count) {
    if (kv.second > max_count) {
      max_count = kv.second;
      center_tbl_idx = kv.first;
    }
  }

  // Center must participate in all edges
  if (static_cast<size_t>(max_count) != edges.size())
    return plan;

  // Need exactly N-1 edges for N leaves (star topology)
  if (edges.size() != leaves.size() - 1)
    return plan;

  const LeafTable *scan_leaf = nullptr;
  for (const auto &leaf : leaves) {
    if (leaf.ir_table_index == center_tbl_idx) {
      scan_leaf = &leaf;
      break;
    }
  }
  if (!scan_leaf)
    return plan;

  // Bail if any leaf has LIKE filters — sorted MIN interacts incorrectly
  // with LIKE on lookup tables in some join topologies.
  if (LeafHasLikeFilter(scan_leaf))
    return plan;
  for (const auto &leaf : leaves) {
    if (&leaf != scan_leaf && LeafHasLikeFilter(&leaf))
      return plan;
  }

  // Compile scan filters
  std::vector<RowPredicate> scan_predicates;
  if (scan_leaf->HasFilters()) {
    if (!CompileAllLeafFilters(scan_leaf->all_filters, scan_leaf->flat, scan_predicates))
      return plan;
  }

  // Dim-derived filters on scan table
  auto dim_scan_it = dim_derived_filters.find(scan_leaf->ir_table_index);
  if (dim_scan_it != dim_derived_filters.end()) {
    for (const auto &filt : dim_scan_it->second) {
      int fk_col = filt.first;
      const auto &pk_vals = filt.second;
      if (pk_vals.size() == 1) {
        int32_t val = pk_vals[0];
        scan_predicates.push_back([fk_col, val](const FlatTable &t, uint64_t row) {
          return !t.columns[fk_col].IsNull(row) && t.columns[fk_col].GetInt32(row) == val;
        });
      } else {
        auto val_set = std::make_shared<std::unordered_set<int32_t>>(pk_vals.begin(), pk_vals.end());
        scan_predicates.push_back([fk_col, val_set](const FlatTable &t, uint64_t row) {
          return !t.columns[fk_col].IsNull(row) && val_set->count(t.columns[fk_col].GetInt32(row)) > 0;
        });
      }
    }
  }

  plan.scan_table = scan_leaf->flat;
  plan.scan_table_name = scan_leaf->name;
  plan.scan_filters = std::move(scan_predicates);

  // Build one join step per edge (each connecting scan table to a lookup table)
  // Track lookup leaves by ir_table_index for MIN column mapping
  std::unordered_map<unsigned int, std::pair<const LeafTable *, int>> lookup_map;

  for (size_t ei = 0; ei < edges.size(); ei++) {
    const auto &edge = edges[ei];

    // Determine scan-side and lookup-side of the edge
    std::string scan_col_name, lookup_col_name;
    unsigned int lookup_tbl_idx;
    if (edge.left_table_idx == scan_leaf->ir_table_index) {
      scan_col_name = edge.left_col_name;
      lookup_col_name = edge.right_col_name;
      lookup_tbl_idx = edge.right_table_idx;
    } else {
      scan_col_name = edge.right_col_name;
      lookup_col_name = edge.left_col_name;
      lookup_tbl_idx = edge.left_table_idx;
    }

    const LeafTable *lookup_leaf = FindLeaf(leaves, lookup_tbl_idx);
    if (!lookup_leaf || !lookup_leaf->flat)
      return plan;

    int scan_flat_col = scan_leaf->flat->FindColumn(scan_col_name);
    if (scan_flat_col < 0 || scan_leaf->flat->columns[scan_flat_col].type != FlatColumnType::INT32)
      return plan;

    // Find CSR index for this edge.
    // The CSR must return rows in the LOOKUP table (csr->fk_table == lookup).
    // A CSR where fk_table == scan_leaf returns scan-table rows — wrong direction.
    const CSRIndex *csr = nullptr;
    std::string runtime_key = lookup_leaf->name + "." + lookup_col_name;
    auto rt_it = runtime_csrs.find(runtime_key);
    if (rt_it != runtime_csrs.end() && rt_it->second.fk_table == lookup_leaf->name)
      csr = &rt_it->second;

    if (!csr && storage_plan) {
      auto *c = storage_plan->GetCSR(scan_leaf->name, scan_col_name);
      if (c && c->fk_table == lookup_leaf->name)
        csr = c;
    }
    if (!csr && storage_plan) {
      auto *c = storage_plan->GetCSR(lookup_leaf->name, lookup_col_name);
      if (c && c->fk_table == lookup_leaf->name)
        csr = c;
    }

    // No CSR found — try PK bitset for base table lookup (scan_key → lookup PK).
    // Common in star joins where center=temp, arms=base tables joined on PK.
    bool use_bitset = false;
    std::vector<bool> pk_bitset;
    std::vector<uint32_t> pk_to_row;
    if (!csr && lookup_leaf->HasFilters()) {
      std::vector<RowPredicate> lookup_predicates;
      if (!CompileAllLeafFilters(lookup_leaf->all_filters, lookup_leaf->flat, lookup_predicates))
        return plan;
      if (!BuildFilteredPKBitset(lookup_leaf->flat, lookup_predicates, pk_bitset, pk_to_row))
        return plan;
      use_bitset = true;
    } else if (!csr) {
      // No CSR and no filters — build unfiltered PK bitset for existence check
      std::vector<RowPredicate> no_filters;
      if (!BuildFilteredPKBitset(lookup_leaf->flat, no_filters, pk_bitset, pk_to_row))
        return plan;
      use_bitset = true;
    }

    if (!csr && !use_bitset)
      return plan;

    // Compile lookup-leaf filters as join_filters when using CSR
    // (bitset path already incorporates filters into the bitset itself)
    std::vector<RowPredicate> join_filters;
    if (csr && lookup_leaf->HasFilters()) {
      if (!CompileAllLeafFilters(lookup_leaf->all_filters, lookup_leaf->flat, join_filters))
        return plan;
    }

    // Dim-derived filters on this lookup table
    auto dim_lk_it = dim_derived_filters.find(lookup_leaf->ir_table_index);
    if (dim_lk_it != dim_derived_filters.end()) {
      for (const auto &filt : dim_lk_it->second) {
        int fk_col = filt.first;
        const auto &pk_vals = filt.second;
        if (pk_vals.size() == 1) {
          int32_t val = pk_vals[0];
          join_filters.push_back([fk_col, val](const FlatTable &t, uint64_t row) {
            return !t.columns[fk_col].IsNull(row) && t.columns[fk_col].GetInt32(row) == val;
          });
        } else {
          auto val_set = std::make_shared<std::unordered_set<int32_t>>(pk_vals.begin(), pk_vals.end());
          join_filters.push_back([fk_col, val_set](const FlatTable &t, uint64_t row) {
            return !t.columns[fk_col].IsNull(row) && val_set->count(t.columns[fk_col].GetInt32(row)) > 0;
          });
        }
      }
    }

    KernelJoinStep step;
    step.scan_key_col_idx = scan_flat_col;
    step.joined_table = lookup_leaf->flat;
    step.is_semi = true;
    step.csr = csr;
    if (use_bitset) {
      step.use_bitset = true;
      step.pk_bitset = std::move(pk_bitset);
      step.pk_to_row = std::move(pk_to_row);
      step.csr = nullptr;
    }
    step.join_filters = std::move(join_filters);

    int step_idx = static_cast<int>(plan.join_steps.size());
    lookup_map[lookup_leaf->ir_table_index] = {lookup_leaf, step_idx};
    plan.join_steps.push_back(std::move(step));
  }

  // Map MIN columns from Projection's target_list through agg_fns.
  for (size_t i = 0; i < proj->target_list.size(); i++) {
    unsigned int agg_idx = proj->target_list[i]->GetColumnIndex();
    if (agg_idx >= agg->agg_fns.size())
      return plan;

    const auto &fn = agg->agg_fns[agg_idx];
    const auto *attr = fn.first.get();
    unsigned int tbl_idx = attr->GetTableIndex();
    std::string col_name = attr->GetColumnName();

    MinColumnInfo mc;
    mc.output_idx = static_cast<int>(i);
    mc.ir_table_idx = tbl_idx;
    mc.name = col_name;

    if (tbl_idx == scan_leaf->ir_table_index) {
      mc.table = scan_leaf->flat;
      mc.on_scan_table = true;
      mc.flat_col_idx = scan_leaf->flat->FindColumn(col_name);
    } else {
      auto lk_it = lookup_map.find(tbl_idx);
      if (lk_it == lookup_map.end())
        return plan;
      mc.table = lk_it->second.first->flat;
      mc.on_scan_table = false;
      mc.flat_col_idx = mc.table->FindColumn(col_name);
    }

    if (mc.flat_col_idx < 0)
      return plan;

    mc.type = mc.table->columns[mc.flat_col_idx].type;

    // Look up sorted index (only for base tables on scan side)
    if (mc.on_scan_table && scan_leaf->is_base) {
      mc.sorted = storage_plan->GetSortedIndex(mc.table->table_name, col_name);
    } else if (!mc.on_scan_table) {
      auto lk_it = lookup_map.find(tbl_idx);
      if (lk_it != lookup_map.end() && lk_it->second.first->is_base)
        mc.sorted = storage_plan->GetSortedIndex(mc.table->table_name, col_name);
    }

    plan.min_cols.push_back(std::move(mc));
  }

  plan.valid = true;
  return plan;
}

// Compare VARCHAR values: return <0, 0, >0
static int CompareVarchar(const FlatTable &t, int col_idx, uint64_t row,
                          const char *other, uint32_t other_len) {
  uint32_t len;
  const char *ptr = t.columns[col_idx].GetVarchar(row, len);
  uint32_t min_len = len < other_len ? len : other_len;
  int cmp = std::memcmp(ptr, other, min_len);
  if (cmp != 0)
    return cmp;
  return static_cast<int>(len) - static_cast<int>(other_len);
}

QueryResult ExecuteFinalAggregate(const FinalAggregatePlan &plan) {
  assert(plan.valid);

  const auto &steps = plan.join_steps;
  const auto &filters = plan.scan_filters;
  uint64_t scan_rows = plan.scan_table->row_count;

  // Result values (empty = NULL/not found)
  std::vector<std::string> min_values(plan.min_cols.size());
  std::vector<bool> min_found(plan.min_cols.size(), false);

  // Lambda: check if a scan row qualifies (filters + join existence)
  auto row_qualifies = [&](uint64_t row) -> bool {
    for (const auto &f : filters) {
      if (!f(*plan.scan_table, row))
        return false;
    }
    for (const auto &step : steps) {
      int32_t key = plan.scan_table->columns[step.scan_key_col_idx].GetInt32(row);
      if (step.use_bitset) {
        if (key < 0 || static_cast<size_t>(key) >= step.pk_bitset.size() || !step.pk_bitset[key])
          return false;
      } else {
        auto [begin, end] = step.csr->Lookup(key);
        if (begin == end)
          return false;
        if (!step.join_filters.empty()) {
          bool any_match = false;
          for (auto it = begin; it != end; ++it) {
            bool pass = true;
            for (const auto &jf : step.join_filters) {
              if (!jf(*step.joined_table, *it)) {
                pass = false;
                break;
              }
            }
            if (pass) { any_match = true; break; }
          }
          if (!any_match) return false;
        }
      }
    }
    return true;
  };

  // Phase 1: Sorted scans for MIN columns on the scan table with sorted indices
  for (size_t i = 0; i < plan.min_cols.size(); i++) {
    const auto &mc = plan.min_cols[i];
    if (!mc.sorted || !mc.on_scan_table)
      continue;

    const auto &perm = mc.sorted->sorted_perm;
    for (uint32_t rank = 0; rank < perm.size(); rank++) {
      uint64_t row = perm[rank];
      if (!row_qualifies(row))
        continue;

      // First qualifying row in sorted order = MIN
      if (mc.type == FlatColumnType::VARCHAR) {
        min_values[i] = mc.table->columns[mc.flat_col_idx].GetString(row);
      } else {
        min_values[i] = std::to_string(mc.table->columns[mc.flat_col_idx].GetInt32(row));
      }
      min_found[i] = true;
      break;
    }
  }

  // Phase 2: Running-min scan for columns without sorted index or on lookup table
  bool need_running_min = false;
  for (size_t i = 0; i < plan.min_cols.size(); i++) {
    if (!min_found[i] && !(plan.min_cols[i].sorted && plan.min_cols[i].on_scan_table))
      need_running_min = true;
  }

  if (need_running_min) {
    struct RunningMin {
      size_t col_idx;
      bool found = false;
      std::string str_val;
      int32_t int_val = 0;
    };

    auto MakeTrackers = [&]() {
      std::vector<RunningMin> t;
      for (size_t i = 0; i < plan.min_cols.size(); i++) {
        const auto &mc = plan.min_cols[i];
        if (mc.sorted && mc.on_scan_table)
          continue;
        RunningMin rm;
        rm.col_idx = i;
        t.push_back(rm);
      }
      return t;
    };

    // Per-row MIN update logic for a set of trackers
    auto UpdateTrackers = [&](uint64_t row, std::vector<RunningMin> &trk) {
      for (auto &rm : trk) {
        const auto &mc = plan.min_cols[rm.col_idx];
        if (mc.on_scan_table) {
          if (mc.type == FlatColumnType::VARCHAR) {
            if (mc.table->columns[mc.flat_col_idx].IsNull(row))
              continue;
            std::string val = mc.table->columns[mc.flat_col_idx].GetString(row);
            if (!rm.found || val < rm.str_val) {
              rm.str_val = std::move(val);
              rm.found = true;
            }
          } else {
            if (mc.table->columns[mc.flat_col_idx].IsNull(row))
              continue;
            int32_t val = mc.table->columns[mc.flat_col_idx].GetInt32(row);
            if (!rm.found || val < rm.int_val) {
              rm.int_val = val;
              rm.found = true;
            }
          }
        } else {
          int step_idx = -1;
          for (size_t si = 0; si < steps.size(); si++) {
            if (steps[si].joined_table == mc.table) { step_idx = static_cast<int>(si); break; }
          }
          if (step_idx < 0) continue;
          const auto &step = steps[step_idx];
          int32_t key = plan.scan_table->columns[step.scan_key_col_idx].GetInt32(row);
          if (step.use_bitset) {
            if (key >= 0 && static_cast<size_t>(key) < step.pk_to_row.size()) {
              uint64_t joined_row = step.pk_to_row[key];
              if (mc.type == FlatColumnType::VARCHAR) {
                if (!mc.table->columns[mc.flat_col_idx].IsNull(joined_row)) {
                  std::string val = mc.table->columns[mc.flat_col_idx].GetString(joined_row);
                  if (!rm.found || val < rm.str_val) {
                    rm.str_val = std::move(val);
                    rm.found = true;
                  }
                }
              } else {
                if (!mc.table->columns[mc.flat_col_idx].IsNull(joined_row)) {
                  int32_t val = mc.table->columns[mc.flat_col_idx].GetInt32(joined_row);
                  if (!rm.found || val < rm.int_val) {
                    rm.int_val = val;
                    rm.found = true;
                  }
                }
              }
            }
          } else {
            auto result = step.csr->Lookup(key);
            for (auto it = result.first; it != result.second; ++it) {
              uint64_t joined_row = *it;
              bool jf_pass = true;
              for (const auto &jf : step.join_filters) {
                if (!jf(*step.joined_table, joined_row)) {
                  jf_pass = false;
                  break;
                }
              }
              if (!jf_pass)
                continue;
              if (mc.type == FlatColumnType::VARCHAR) {
                if (!mc.table->columns[mc.flat_col_idx].IsNull(joined_row)) {
                  std::string val = mc.table->columns[mc.flat_col_idx].GetString(joined_row);
                  if (!rm.found || val < rm.str_val) {
                    rm.str_val = std::move(val);
                    rm.found = true;
                  }
                }
              } else {
                if (!mc.table->columns[mc.flat_col_idx].IsNull(joined_row)) {
                  int32_t val = mc.table->columns[mc.flat_col_idx].GetInt32(joined_row);
                  if (!rm.found || val < rm.int_val) {
                    rm.int_val = val;
                    rm.found = true;
                  }
                }
              }
            }
          }
        }
      }
    };

    // Merge src tracker into dst tracker
    auto MergeTracker = [&](const RunningMin &src, RunningMin &dst) {
      if (!src.found) return;
      const auto &mc = plan.min_cols[src.col_idx];
      if (mc.type == FlatColumnType::VARCHAR) {
        if (!dst.found || src.str_val < dst.str_val) {
          dst.str_val = src.str_val;
          dst.found = true;
        }
      } else {
        if (!dst.found || src.int_val < dst.int_val) {
          dst.int_val = src.int_val;
          dst.found = true;
        }
      }
    };

    std::vector<RunningMin> trackers = MakeTrackers();

#ifdef HAVE_OPENMP
    if (scan_rows >= OMP_PARALLEL_THRESHOLD) {
      int nthreads = std::min(12, omp_get_max_threads());
      std::vector<std::vector<RunningMin>> tl_trackers(nthreads);
      for (int t = 0; t < nthreads; t++)
        tl_trackers[t] = MakeTrackers();

      #pragma omp parallel num_threads(nthreads)
      {
        int tid = omp_get_thread_num();
        auto &local = tl_trackers[tid];

        #pragma omp for schedule(dynamic, 4096)
        for (int64_t row = 0; row < static_cast<int64_t>(scan_rows); row++) {
          if (!row_qualifies(static_cast<uint64_t>(row)))
            continue;
          UpdateTrackers(static_cast<uint64_t>(row), local);
        }
      }

      for (int t = 0; t < nthreads; t++) {
        for (size_t i = 0; i < trackers.size(); i++)
          MergeTracker(tl_trackers[t][i], trackers[i]);
      }
    } else
#endif
    {
      for (uint64_t row = 0; row < scan_rows; row++) {
        if (!row_qualifies(row))
          continue;
        UpdateTrackers(row, trackers);
      }
    }

    // Copy tracker results
    for (const auto &rm : trackers) {
      if (rm.found) {
        const auto &mc = plan.min_cols[rm.col_idx];
        if (mc.type == FlatColumnType::VARCHAR) {
          min_values[rm.col_idx] = rm.str_val;
        } else {
          min_values[rm.col_idx] = std::to_string(rm.int_val);
        }
        min_found[rm.col_idx] = true;
      }
    }
  }

  // Build QueryResult
  QueryResult result;
  result.num_columns = static_cast<int>(plan.min_cols.size());
  result.column_names = plan.output_names;
  result.rows.resize(1);
  result.rows[0].resize(plan.min_cols.size());
  for (size_t i = 0; i < plan.min_cols.size(); i++) {
    result.rows[0][i] = min_found[i] ? min_values[i] : "NULL";
  }
  result.num_rows = 1;

  return result;
}

} // namespace storage
} // namespace middleware
