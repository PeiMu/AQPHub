/*
 * Utility functions for IR operations (cloning, checking, etc.)
 * Centralizes common patterns to reduce code duplication
 */

#pragma once

#include "simplest_ir.h"
#include <memory>
#include <set>

namespace middleware {
namespace ir_utils {

// ===== Expression Cloning =====
// Clone expressions (deep copy) - needed when same expression goes to multiple
// IRs

inline std::unique_ptr<ir_sql_converter::SimplestAttr>
CloneAttr(const ir_sql_converter::SimplestAttr *attr) {
  if (!attr)
    return nullptr;
  return std::make_unique<ir_sql_converter::SimplestAttr>(*attr);
}

inline std::unique_ptr<ir_sql_converter::SimplestAttr>
CloneAttr(const std::unique_ptr<ir_sql_converter::SimplestAttr> &attr) {
  return CloneAttr(attr.get());
}

std::unique_ptr<ir_sql_converter::SimplestExpr>
CloneExpr(const ir_sql_converter::SimplestExpr *expr);

inline std::unique_ptr<ir_sql_converter::SimplestExpr>
CloneExpr(const std::unique_ptr<ir_sql_converter::SimplestExpr> &expr) {
  return CloneExpr(expr.get());
}

inline std::unique_ptr<ir_sql_converter::SimplestVarComparison>
CloneVarComparison(const ir_sql_converter::SimplestVarComparison *cond) {
  if (!cond)
    return nullptr;
  return std::make_unique<ir_sql_converter::SimplestVarComparison>(
      cond->GetSimplestExprType(), CloneAttr(cond->left_attr),
      CloneAttr(cond->right_attr));
}

// ===== Expression Table Checking =====
// Check if expression involves only specified tables

bool ExprInvolvesOnlyTables(const ir_sql_converter::SimplestExpr *expr,
                            const std::set<unsigned int> &tables);

inline bool ExprInvolvesOnlyTables(
    const std::unique_ptr<ir_sql_converter::SimplestExpr> &expr,
    const std::set<unsigned int> &tables) {
  return ExprInvolvesOnlyTables(expr.get(), tables);
}

// ===== Attribute Collection from Expressions =====
// Collect all attributes referenced in an expression

void CollectAttrsFromExpr(
    const ir_sql_converter::SimplestExpr *expr,
    const std::set<unsigned int> &target_tables,
    std::set<std::pair<unsigned int, unsigned int>> &seen,
    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> &attrs);

// ===== AND-Splitting for Filter Conditions =====
// Extract AND-connected conjuncts that involve only specified tables
// For: A && B && C where A involves {1}, B involves {3}, C involves {4}
// With tables={1,4}: returns A && C (skips B)
// For OR expressions: only include if ALL tables are in the set (can't split OR)

std::unique_ptr<ir_sql_converter::SimplestExpr>
ExtractConjunctsForTables(const ir_sql_converter::SimplestExpr *expr,
                          const std::set<unsigned int> &tables);

// ===== Collection Functions for Sub-IR Building =====
// These are used by FK-based splitter to collect components for cluster sub-IRs

// Collect filter conditions from IR tree, extracting conjuncts for specified tables
// Uses AND-splitting: for AND expressions, extracts only conjuncts involving cluster tables
std::vector<std::unique_ptr<ir_sql_converter::SimplestExpr>>
CollectFilterConditions(ir_sql_converter::SimplestStmt *ir,
                        const std::set<unsigned int> &tables);

// Collect join conditions between specified tables
std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
CollectJoinConditions(ir_sql_converter::SimplestStmt *ir,
                      const std::set<unsigned int> &tables);

// Collect attributes from filter expressions for specified tables
std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
CollectFilterAttrs(ir_sql_converter::SimplestStmt *ir,
                   const std::set<unsigned int> &tables);

} // namespace ir_utils
} // namespace middleware
