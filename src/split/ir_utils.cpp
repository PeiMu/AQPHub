/*
 * Implementation of IR utility functions
 */

#include "split/ir_utils.h"

#include <stdexcept>
#include <string>

namespace middleware {
namespace ir_utils {

std::unique_ptr<ir_sql_converter::AQPExpr>
CloneExpr(const ir_sql_converter::AQPExpr *expr) {
  if (!expr)
    return nullptr;

  auto node_type = expr->GetNodeType();

  if (node_type == ir_sql_converter::SimplestNodeType::VarConstComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarConstComparison *>(
            expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestVarConstComparison>(
          e->GetSimplestExprType(), CloneAttr(e->attr),
          std::make_unique<ir_sql_converter::SimplestConstVar>(*e->const_var));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarComparison *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestVarComparison>(
          e->GetSimplestExprType(), CloneAttr(e->left_attr),
          CloneAttr(e->right_attr));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::IsNullExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestIsNullExpr>(
          e->GetSimplestExprType(), CloneAttr(e->attr));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarParamComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarParamComparison *>(
            expr);
    if (e && e->param_var) {
      auto cloned_param = std::make_unique<ir_sql_converter::SimplestParam>(
          e->param_var->GetType(), e->param_var->GetParamId());
      return std::make_unique<ir_sql_converter::SimplestVarParamComparison>(
          e->GetSimplestExprType(), CloneAttr(e->attr),
          std::move(cloned_param));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestLogicalExpr>(
          e->GetLogicalOp(), CloneExpr(e->left_expr.get()),
          CloneExpr(e->right_expr.get()));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::SingleAttrExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestSingleAttrExpr *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestSingleAttrExpr>(
          CloneAttr(e->attr));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::InExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    if (e && e->attr) {
      std::vector<std::unique_ptr<ir_sql_converter::SimplestConstVar>> vals;
      for (const auto &v : e->values)
        vals.push_back(
            std::make_unique<ir_sql_converter::SimplestConstVar>(*v));
      return std::make_unique<ir_sql_converter::SimplestInExpr>(
          CloneAttr(e->attr), std::move(vals), e->negated);
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ArithExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestArithExpr *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestArithExpr>(
          e->arith_op, CloneExpr(e->left.get()), CloneExpr(e->right.get()),
          e->result_type);
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::CastExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestCastExpr *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestCastExpr>(
          CloneExpr(e->child.get()), e->target_type);
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestGeneralComparison *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestGeneralComparison>(
          e->GetSimplestExprType(), CloneExpr(e->left_expr.get()),
          CloneExpr(e->right_expr.get()));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ConstVarNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestConstExpr *>(expr);
    if (e && e->value) {
      return std::make_unique<ir_sql_converter::SimplestConstExpr>(
          std::make_unique<ir_sql_converter::SimplestConstVar>(*e->value));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::FunctionExprNodeType) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestFunctionExpr *>(expr);
    if (e) {
      std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> cloned_args;
      for (const auto &arg : e->args)
        cloned_args.push_back(CloneExpr(arg.get()));
      return std::make_unique<ir_sql_converter::SimplestFunctionExpr>(
          e->fn_name, std::move(cloned_args));
    }
  }

  throw std::runtime_error("ir_utils::CloneExpr unsupported: expression node "
                           "type " +
                           std::to_string(static_cast<int>(node_type)));
}

bool ExprInvolvesOnlyTables(const ir_sql_converter::AQPExpr *expr,
                            const std::set<unsigned int> &tables) {
  if (!expr)
    return true;

  auto node_type = expr->GetNodeType();

  if (node_type == ir_sql_converter::SimplestNodeType::VarConstComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarConstComparison *>(
            expr);
    return e && tables.count(e->attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarComparison *>(expr);
    if (!e)
      return false;
    return tables.count(e->left_attr->GetTableIndex()) > 0 &&
           tables.count(e->right_attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::IsNullExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    return e && tables.count(e->attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarParamComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarParamComparison *>(
            expr);
    return e && tables.count(e->attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (!e)
      return false;
    return ExprInvolvesOnlyTables(e->left_expr.get(), tables) &&
           ExprInvolvesOnlyTables(e->right_expr.get(), tables);
  }

  if (node_type == ir_sql_converter::SimplestNodeType::SingleAttrExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestSingleAttrExpr *>(expr);
    return e && tables.count(e->attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::InExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    return e && e->attr && tables.count(e->attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ArithExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestArithExpr *>(expr);
    if (!e)
      return false;
    return ExprInvolvesOnlyTables(e->left.get(), tables) &&
           ExprInvolvesOnlyTables(e->right.get(), tables);
  }

  if (node_type == ir_sql_converter::SimplestNodeType::CastExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestCastExpr *>(expr);
    if (!e)
      return false;
    return ExprInvolvesOnlyTables(e->child.get(), tables);
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestGeneralComparison *>(expr);
    if (!e)
      return false;
    return ExprInvolvesOnlyTables(e->left_expr.get(), tables) &&
           ExprInvolvesOnlyTables(e->right_expr.get(), tables);
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ConstVarNode) {
    return true;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::FunctionExprNodeType) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestFunctionExpr *>(expr);
    if (!e)
      return false;
    for (const auto &arg : e->args) {
      if (!ExprInvolvesOnlyTables(arg.get(), tables))
        return false;
    }
    return true;
  }

  return false;
}

void CollectAttrsFromExpr(
    const ir_sql_converter::AQPExpr *expr,
    const std::set<unsigned int> &target_tables,
    std::set<std::pair<unsigned int, unsigned int>> &seen,
    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> &attrs) {

  if (!expr)
    return;

  auto addAttr = [&](const ir_sql_converter::SimplestAttr *attr) {
    if (!attr)
      return;
    if (target_tables.count(attr->GetTableIndex()) == 0)
      return;
    auto key = std::make_pair(attr->GetTableIndex(), attr->GetColumnIndex());
    if (seen.find(key) == seen.end()) {
      seen.insert(key);
      attrs.push_back(CloneAttr(attr));
    }
  };

  auto node_type = expr->GetNodeType();

  if (node_type == ir_sql_converter::SimplestNodeType::VarConstComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarConstComparison *>(
            expr);
    if (e)
      addAttr(e->attr.get());
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarComparison *>(expr);
    if (e) {
      addAttr(e->left_attr.get());
      addAttr(e->right_attr.get());
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::IsNullExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    if (e)
      addAttr(e->attr.get());
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarParamComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarParamComparison *>(
            expr);
    if (e)
      addAttr(e->attr.get());
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (e) {
      CollectAttrsFromExpr(e->left_expr.get(), target_tables, seen, attrs);
      CollectAttrsFromExpr(e->right_expr.get(), target_tables, seen, attrs);
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::SingleAttrExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestSingleAttrExpr *>(expr);
    if (e)
      addAttr(e->attr.get());
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::InExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    if (e)
      addAttr(e->attr.get());
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ArithExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestArithExpr *>(expr);
    if (e) {
      CollectAttrsFromExpr(e->left.get(), target_tables, seen, attrs);
      CollectAttrsFromExpr(e->right.get(), target_tables, seen, attrs);
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::CastExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestCastExpr *>(expr);
    if (e)
      CollectAttrsFromExpr(e->child.get(), target_tables, seen, attrs);
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestGeneralComparison *>(expr);
    if (e) {
      CollectAttrsFromExpr(e->left_expr.get(), target_tables, seen, attrs);
      CollectAttrsFromExpr(e->right_expr.get(), target_tables, seen, attrs);
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ConstVarNode) {
    // SimplestConstExpr: no attrs to collect
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::FunctionExprNodeType) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestFunctionExpr *>(expr);
    if (e) {
      for (const auto &arg : e->args)
        CollectAttrsFromExpr(arg.get(), target_tables, seen, attrs);
    }
    return;
  }

  throw std::runtime_error(
      "ir_utils::CollectAttrsFromExpr unsupported: expression node type " +
      std::to_string(static_cast<int>(node_type)) +
      "; required columns would be missed");
}

// ===== AND-Splitting for Filter Conditions =====

std::unique_ptr<ir_sql_converter::AQPExpr>
ExtractConjunctsForTables(const ir_sql_converter::AQPExpr *expr,
                          const std::set<unsigned int> &tables) {
  if (!expr)
    return nullptr;

  auto node_type = expr->GetNodeType();

  // Handle LogicalExpr (AND/OR)
  if (node_type == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (!e)
      return nullptr;

    if (e->GetLogicalOp() == ir_sql_converter::SimplestLogicalOp::LogicalAnd) {
      // AND: recursively extract from both sides, combine valid parts
      auto left_extracted =
          ExtractConjunctsForTables(e->left_expr.get(), tables);
      auto right_extracted =
          ExtractConjunctsForTables(e->right_expr.get(), tables);

      if (left_extracted && right_extracted) {
        // Both sides valid → combine with AND
        return std::make_unique<ir_sql_converter::SimplestLogicalExpr>(
            ir_sql_converter::SimplestLogicalOp::LogicalAnd,
            std::move(left_extracted), std::move(right_extracted));
      } else if (left_extracted) {
        // Only left valid → return left
        return left_extracted;
      } else if (right_extracted) {
        // Only right valid → return right
        return right_extracted;
      } else {
        // Neither valid → return nullptr
        return nullptr;
      }
    } else if (e->GetLogicalOp() ==
               ir_sql_converter::SimplestLogicalOp::LogicalOr) {
      // OR: can only include if ALL referenced tables are in the set
      // (can't split OR - must keep entire clause together)
      if (ExprInvolvesOnlyTables(expr, tables)) {
        return CloneExpr(expr);
      }
      return nullptr;
    }
  }

  // For leaf expressions: include if all referenced tables are in the set
  if (ExprInvolvesOnlyTables(expr, tables)) {
    return CloneExpr(expr);
  }

  return nullptr;
}

// ===== Collection Functions =====

std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>>
CollectFilterConditions(ir_sql_converter::AQPStmt *ir,
                        const std::set<unsigned int> &tables) {
  std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> filters;

  if (!ir)
    return filters;

  // Check qual_vec for filter conditions in this node
  for (const auto &qual : ir->qual_vec) {
    // Use AND-splitting: extract conjuncts that involve only cluster tables
    auto extracted = ExtractConjunctsForTables(qual.get(), tables);
    if (extracted) {
      filters.push_back(std::move(extracted));
    }
  }

  // Recurse into children
  for (auto &child : ir->children) {
    auto child_filters = CollectFilterConditions(child.get(), tables);
    for (auto &f : child_filters) {
      filters.push_back(std::move(f));
    }
  }

  return filters;
}

std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
CollectJoinConditions(ir_sql_converter::AQPStmt *ir,
                      const std::set<unsigned int> &tables) {
  std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
      conditions;

  if (!ir)
    return conditions;

  // If this is a Join node, check its join conditions
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *join = dynamic_cast<ir_sql_converter::SimplestJoin *>(ir);
    if (join) {
      for (const auto &cond : join->join_conditions) {
        unsigned int left_table = cond->left_attr->GetTableIndex();
        unsigned int right_table = cond->right_attr->GetTableIndex();
#ifndef NDEBUG
        std::cout << "[ir_utils::CollectJoinConditions] cond "
                  << left_table << "." << cond->left_attr->GetColumnName()
                  << " vs " << right_table << "."
                  << cond->right_attr->GetColumnName()
                  << " in_cluster: " << tables.count(left_table)
                  << "/" << tables.count(right_table) << std::endl;
#endif
        // Include condition if BOTH tables are in our cluster
        if (tables.count(left_table) && tables.count(right_table)) {
          auto cloned = CloneVarComparison(cond.get());
          if (cloned) {
            conditions.push_back(std::move(cloned));
          }
        }
      }
    }
  }

  // Recurse into children
  for (auto &child : ir->children) {
    auto child_conds = CollectJoinConditions(child.get(), tables);
    for (auto &c : child_conds) {
      conditions.push_back(std::move(c));
    }
  }

  return conditions;
}

std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
CollectFilterAttrs(ir_sql_converter::AQPStmt *ir,
                   const std::set<unsigned int> &tables) {
  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> attrs;
  std::set<std::pair<unsigned int, unsigned int>> seen;

  if (!ir)
    return attrs;

  // Collect from qual_vec in this node
  for (const auto &qual : ir->qual_vec) {
    CollectAttrsFromExpr(qual.get(), tables, seen, attrs);
  }

  // Recurse into children
  for (auto &child : ir->children) {
    auto child_attrs = CollectFilterAttrs(child.get(), tables);
    for (auto &attr : child_attrs) {
      auto key = std::make_pair(attr->GetTableIndex(), attr->GetColumnIndex());
      if (seen.find(key) == seen.end()) {
        seen.insert(key);
        attrs.push_back(std::move(attr));
      }
    }
  }

  return attrs;
}

} // namespace ir_utils
} // namespace middleware
