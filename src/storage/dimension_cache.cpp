#include "storage/dimension_cache.h"
#include "simplest_ir.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <unordered_set>

using namespace ir_sql_converter;

namespace middleware {
namespace storage {

void DimensionCache::Build(
    const std::unordered_map<std::string, FlatTable> &tables) {
  dim_tables_.clear();
  for (const auto &kv : tables) {
    if (kv.second.row_count <= MAX_DIM_ROWS) {
      dim_tables_[kv.first] = &kv.second;
    }
  }
}

bool DimensionCache::IsDimension(const std::string &table_name) const {
  return dim_tables_.count(table_name) > 0;
}

const FlatTable *
DimensionCache::GetDimTable(const std::string &table_name) const {
  auto it = dim_tables_.find(table_name);
  return it != dim_tables_.end() ? it->second : nullptr;
}

namespace {

bool LikeMatch(const char *str, uint32_t str_len, const char *pat,
               size_t pat_len) {
  // DP-based SQL LIKE: % matches any sequence, _ matches one char
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

int CompareVarchar(const char *a, uint32_t a_len, const char *b,
                   size_t b_len) {
  size_t min_len = a_len < b_len ? a_len : b_len;
  int cmp = std::memcmp(a, b, min_len);
  if (cmp != 0)
    return cmp;
  if (a_len < b_len)
    return -1;
  if (a_len > b_len)
    return 1;
  return 0;
}

bool EvalOnePredicate(const AQPExpr *expr, const FlatTable *table,
                      uint64_t row) {
  if (!expr || !table)
    return true;

  auto expr_type = expr->GetSimplestExprType();

  if (expr_type == NullType || expr_type == NonNullType) {
    auto *isnull = static_cast<const SimplestIsNullExpr *>(expr);
    int col_idx = table->FindColumn(isnull->attr->GetColumnName());
    if (col_idx < 0)
      return true;
    bool is_null = table->columns[col_idx].IsNull(row);
    return (expr_type == NullType) ? is_null : !is_null;
  }

  if (expr->GetNodeType() == VarConstComparisonNode) {
    auto *cmp = static_cast<const SimplestVarConstComparison *>(expr);
    int col_idx = table->FindColumn(cmp->attr->GetColumnName());
    if (col_idx < 0)
      return true;
    auto col_type = table->columns[col_idx].type;
    auto cmp_type = cmp->GetSimplestExprType();
    auto var_type = cmp->const_var->GetType();

    if (col_type == FlatColumnType::INT32 && var_type == IntVar) {
      if (table->columns[col_idx].IsNull(row))
        return false;
      int32_t val = table->columns[col_idx].GetInt32(row);
      int32_t cval = cmp->const_var->GetIntValue();
      switch (cmp_type) {
      case Equal:
        return val == cval;
      case NotEqual:
        return val != cval;
      case LessThan:
        return val < cval;
      case GreaterThan:
        return val > cval;
      case LessEqual:
        return val <= cval;
      case GreaterEqual:
        return val >= cval;
      default:
        return true;
      }
    }

    if (col_type == FlatColumnType::VARCHAR && var_type == StringVar) {
      if (table->columns[col_idx].IsNull(row))
        return false;
      uint32_t len;
      const char *ptr = table->columns[col_idx].GetVarchar(row, len);
      std::string cstr = cmp->const_var->GetStringValue();
      switch (cmp_type) {
      case Equal:
        return len == cstr.size() &&
               std::memcmp(ptr, cstr.data(), len) == 0;
      case NotEqual:
        return len != cstr.size() ||
               std::memcmp(ptr, cstr.data(), len) != 0;
      case LessThan:
        return CompareVarchar(ptr, len, cstr.data(), cstr.size()) < 0;
      case GreaterThan:
        return CompareVarchar(ptr, len, cstr.data(), cstr.size()) > 0;
      case LessEqual:
        return CompareVarchar(ptr, len, cstr.data(), cstr.size()) <= 0;
      case GreaterEqual:
        return CompareVarchar(ptr, len, cstr.data(), cstr.size()) >= 0;
      case TextLike:
        return LikeMatch(ptr, len, cstr.data(), cstr.size());
      case Text_Not_Like:
        return !LikeMatch(ptr, len, cstr.data(), cstr.size());
      default:
        return true;
      }
    }

    return true;
  }

  if (expr->GetNodeType() == InExprNode) {
    auto *in_expr = static_cast<const SimplestInExpr *>(expr);
    int col_idx = table->FindColumn(in_expr->attr->GetColumnName());
    if (col_idx < 0)
      return true;
    if (table->columns[col_idx].IsNull(row))
      return false;
    auto col_type = table->columns[col_idx].type;

    if (col_type == FlatColumnType::INT32) {
      int32_t val = table->columns[col_idx].GetInt32(row);
      bool found = false;
      for (const auto &v : in_expr->values) {
        if (v->GetType() == IntVar && v->GetIntValue() == val) {
          found = true;
          break;
        }
      }
      return in_expr->negated ? !found : found;
    }
    if (col_type == FlatColumnType::VARCHAR) {
      std::string val = table->columns[col_idx].GetString(row);
      bool found = false;
      for (const auto &v : in_expr->values) {
        if (v->GetType() == StringVar && v->GetStringValue() == val) {
          found = true;
          break;
        }
      }
      return in_expr->negated ? !found : found;
    }
    return true;
  }

  if (expr->GetNodeType() == LogicalExprNode) {
    auto *logic = static_cast<const SimplestLogicalExpr *>(expr);
    auto op = logic->GetLogicalOp();
    if (op == LogicalAnd) {
      return EvalOnePredicate(logic->left_expr.get(), table, row) &&
             EvalOnePredicate(logic->right_expr.get(), table, row);
    }
    if (op == LogicalOr) {
      return EvalOnePredicate(logic->left_expr.get(), table, row) ||
             EvalOnePredicate(logic->right_expr.get(), table, row);
    }
    if (op == LogicalNot) {
      return !EvalOnePredicate(logic->left_expr.get(), table, row);
    }
  }

  return true;
}

// Check if all filter expressions can be evaluated by the dim cache.
// Returns false for any unsupported expression type, causing ResolveFilterToPKs
// to bail out (return empty) rather than silently skip the filter.
bool CanEvalAllFilters(
    const std::vector<
        const std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> *>
        &filters,
    const FlatTable *table) {
  for (const auto *filter_vec : filters) {
    if (!filter_vec)
      continue;
    for (const auto &expr : *filter_vec) {
      if (!expr)
        continue;
      auto node_type = expr->GetNodeType();
      auto expr_type = expr->GetSimplestExprType();

      if (expr_type == NullType || expr_type == NonNullType)
        continue; // supported

      if (node_type == VarConstComparisonNode) {
        auto *cmp = static_cast<const SimplestVarConstComparison *>(expr.get());
        int col_idx = table->FindColumn(cmp->attr->GetColumnName());
        if (col_idx < 0)
          return false;
        auto col_type = table->columns[col_idx].type;
        auto cmp_type = cmp->GetSimplestExprType();
        auto var_type = cmp->const_var->GetType();
        if (col_type == FlatColumnType::INT32 && var_type == IntVar) {
          switch (cmp_type) {
          case Equal: case NotEqual: case LessThan: case GreaterThan:
          case LessEqual: case GreaterEqual:
            break;
          default:
            return false;
          }
        } else if (col_type == FlatColumnType::VARCHAR &&
                   var_type == StringVar) {
          switch (cmp_type) {
          case Equal: case NotEqual: case LessThan: case GreaterThan:
          case LessEqual: case GreaterEqual: case TextLike: case Text_Not_Like:
            break;
          default:
            return false;
          }
        } else {
          return false;
        }
        continue;
      }

      if (node_type == InExprNode) {
        auto *in_expr = static_cast<const SimplestInExpr *>(expr.get());
        int col_idx = table->FindColumn(in_expr->attr->GetColumnName());
        if (col_idx < 0)
          return false;
        auto col_type = table->columns[col_idx].type;
        if (col_type != FlatColumnType::INT32 &&
            col_type != FlatColumnType::VARCHAR)
          return false;
        // Value types must match the column type, otherwise EvalOnePredicate
        // would find no match and wrongly evaluate the IN to false for every
        // row (e.g. DATE columns are stored INT32 but carry StringVar values).
        for (const auto &v : in_expr->values) {
          if (col_type == FlatColumnType::INT32 && v->GetType() != IntVar)
            return false;
          if (col_type == FlatColumnType::VARCHAR &&
              v->GetType() != StringVar)
            return false;
        }
        continue;
      }

      if (node_type == LogicalExprNode)
        continue; // children checked recursively by EvalOnePredicate

      return false; // unknown node type
    }
  }
  return true;
}

} // anonymous namespace

std::vector<int32_t> DimensionCache::ResolveFilterToPKs(
    const std::string &table_name,
    const std::vector<
        const std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> *>
        &filters) const {
  auto it = dim_tables_.find(table_name);
  if (it == dim_tables_.end())
    return {};

  const FlatTable *table = it->second;
  int pk_col = table->FindColumn("id");
  if (pk_col < 0 || table->columns[pk_col].type != FlatColumnType::INT32)
    return {};

  if (!CanEvalAllFilters(filters, table))
    return {};

  std::vector<int32_t> result;
  for (uint64_t r = 0; r < table->row_count; r++) {
    bool pass = true;
    for (const auto *filter_vec : filters) {
      if (!filter_vec)
        continue;
      for (const auto &expr : *filter_vec) {
        if (!EvalOnePredicate(expr.get(), table, r)) {
          pass = false;
          break;
        }
      }
      if (!pass)
        break;
    }
    if (pass) {
      int32_t pk = table->columns[pk_col].GetInt32(r);
      result.push_back(pk);
    }
  }

  std::sort(result.begin(), result.end());
  return result;
}

void DimensionCache::PrintSummary() const {
  std::cerr << "[DimensionCache] " << dim_tables_.size()
            << " dimension tables cached:\n";
  for (const auto &kv : dim_tables_) {
    std::cerr << "  " << kv.first << ": " << kv.second->row_count
              << " rows\n";
  }
}

} // namespace storage
} // namespace middleware
