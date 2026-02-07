/*
 * Implementation of IR-level ReorderGet
 * Gets cardinality from IR itself, does NOT execute queries
 */

#include "split/ir_reorder_get.h"
#include "split/ir_utils.h"
#include <algorithm>
#include <iostream>
#include <map>

namespace middleware {

std::unique_ptr<ir_sql_converter::SimplestStmt>
IRReorderGet::Reorder(std::unique_ptr<ir_sql_converter::SimplestStmt> ir) {
#ifndef NDEBUG
  std::cout << "[IRReorderGet] Starting table reordering by cardinality"
            << std::endl;
#endif

  // Step 1: Collect all table scans with their cardinalities
  std::vector<TableInfo> tables;
  CollectTableScans(ir.get(), tables);

  if (tables.size() <= 1) {
#ifndef NDEBUG
    std::cout << "[IRReorderGet] Only " << tables.size()
              << " table(s), no reordering needed" << std::endl;
#endif
    return ir;
  }

#ifndef NDEBUG
  std::cout << "[IRReorderGet] Found " << tables.size()
            << " tables:" << std::endl;
  for (const auto &table : tables) {
    std::cout << "  - Table " << table.table_index << " (" << table.table_name
              << "): cardinality = " << table.cardinality << std::endl;
  }
#endif

  // Step 2: Sort tables by cardinality (SMALLEST FIRST for left-deep join tree)
  std::sort(tables.begin(), tables.end(),
            [](const TableInfo &a, const TableInfo &b) {
              return a.cardinality < b.cardinality; // Ascending order
            });

#ifndef NDEBUG
  std::cout << "[IRReorderGet] Sorted order (smallest first):" << std::endl;
  for (const auto &table : tables) {
    std::cout << "  - Table " << table.table_index << " (" << table.table_name
              << "): cardinality = " << table.cardinality << std::endl;
  }
#endif

  // Step 3: Collect join conditions
  std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
      join_conditions;
  CollectJoinConditions(ir.get(), join_conditions);

#ifndef NDEBUG
  std::cout << "[IRReorderGet] Found " << join_conditions.size()
            << " join condition(s)" << std::endl;
#endif

  if (join_conditions.empty()) {
#ifndef NDEBUG
    std::cout << "[IRReorderGet] No join conditions, returning original IR"
              << std::endl;
#endif
    return ir;
  }

  // Step 4: Rebuild join tree with reordered tables
  auto reordered_join_tree = RebuildJoinTree(tables, join_conditions);

  if (!reordered_join_tree) {
#ifndef NDEBUG
    std::cout << "[IRReorderGet] Tree rebuild failed, returning original IR"
              << std::endl;
#endif
    return ir;
  }

  // Step 5: Preserve top-level operators and attach reordered join tree
  auto rebuilt_tree =
      PreserveTopOperators(std::move(ir), std::move(reordered_join_tree));

#ifndef NDEBUG
  std::cout << "[IRReorderGet] Reordering complete!" << std::endl;
#endif
  return rebuilt_tree;
}

void IRReorderGet::CollectTableScans(ir_sql_converter::SimplestStmt *node,
                                     std::vector<TableInfo> &tables) {
  if (!node)
    return;

  // Check if this is a Scan node
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = dynamic_cast<ir_sql_converter::SimplestScan *>(node);
    if (scan) {
//      uint64_t cardinality = scan->GetEstimatedCardinality();
      tables.emplace_back(scan->GetTableIndex(), 0,
                          scan->GetTableName(), nullptr);
    }
  }
  // Check for Chunk nodes (temp tables from previous iterations)
  else if (node->GetNodeType() ==
           ir_sql_converter::SimplestNodeType::ChunkNode) {
    auto *chunk = dynamic_cast<ir_sql_converter::SimplestChunk *>(node);
#ifndef NDEBUG
    if (chunk) {
      std::cout << "[IRReorderGet] Found Chunk node (table_index="
                << chunk->GetTableIndex() << "), skipping reorder" << std::endl;
    }
#endif
  }

  // Recursively collect from children
  for (auto &child : node->children) {
    CollectTableScans(child.get(), tables);
  }
}

void IRReorderGet::CollectJoinConditions(
    ir_sql_converter::SimplestStmt *node,
    std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
        &join_conds) {
  if (!node)
    return;

  // Check if this is a Join node
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *join = dynamic_cast<ir_sql_converter::SimplestJoin *>(node);
    if (join) {
      // Clone join conditions
      for (const auto &cond : join->join_conditions) {
        auto cloned_cond = ir_utils::CloneVarComparison(cond.get());
        if (cloned_cond) {
          join_conds.push_back(std::move(cloned_cond));
        }
      }
    }
  }

  // Recursively collect from children
  for (auto &child : node->children) {
    CollectJoinConditions(child.get(), join_conds);
  }
}

std::unique_ptr<ir_sql_converter::SimplestStmt> IRReorderGet::RebuildJoinTree(
    std::vector<TableInfo> &sorted_tables,
    std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
        &join_conditions) {

#ifndef NDEBUG
  std::cout << "[IRReorderGet] Building left-deep join tree" << std::endl;
#endif

  // Build a map: (table1_idx, table2_idx) -> join conditions
  std::map<std::pair<unsigned int, unsigned int>,
           std::vector<ir_sql_converter::SimplestVarComparison *>>
      join_map;

  for (const auto &cond : join_conditions) {
    unsigned int left_table = cond->left_attr->GetTableIndex();
    unsigned int right_table = cond->right_attr->GetTableIndex();

    // Normalize key: always put smaller index first
    auto key = std::make_pair(std::min(left_table, right_table),
                              std::max(left_table, right_table));
    join_map[key].push_back(cond.get());
  }

  // Build left-deep join tree: smallest table at the bottom
  // Pattern: ((T1 JOIN T2) JOIN T3) JOIN T4
  std::unique_ptr<ir_sql_converter::SimplestStmt> current_tree = nullptr;
  std::set<unsigned int> joined_tables;

  for (size_t i = 0; i < sorted_tables.size(); i++) {
    auto &table_info = sorted_tables[i];
    unsigned int current_table_idx = table_info.table_index;

#ifndef NDEBUG
    std::cout << "[IRReorderGet] Adding table " << current_table_idx << " ("
              << table_info.table_name << ")" << std::endl;
#endif

    // Create scan node for this table
    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> empty_children;
    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> empty_attrs;
    auto base_stmt = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(empty_children), std::move(empty_attrs),
        ir_sql_converter::SimplestNodeType::StmtNode);
    auto scan_node = std::make_unique<ir_sql_converter::SimplestScan>(
        std::move(base_stmt), current_table_idx, table_info.table_name);
//    scan_node->SetEstimatedCardinality(table_info.cardinality);

    if (i == 0) {
      // First table - becomes the base of the tree
      current_tree = std::move(scan_node);
      joined_tables.insert(current_table_idx);
      continue;
    }

    // Find join conditions between current table and already-joined tables
    std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
        applicable_conditions;

    for (unsigned int already_joined : joined_tables) {
      auto key = std::make_pair(std::min(current_table_idx, already_joined),
                                std::max(current_table_idx, already_joined));

      if (join_map.count(key)) {
        for (auto *cond : join_map[key]) {
          // Clone the condition
          auto cloned = ir_utils::CloneVarComparison(cond);
          if (cloned) {
            applicable_conditions.push_back(std::move(cloned));
          }
        }
      }
    }

    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> join_children;
    join_children.push_back(std::move(current_tree));
    join_children.push_back(std::move(scan_node));
    base_stmt = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(join_children),
        std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>(),
        ir_sql_converter::SimplestNodeType::StmtNode);

    if (applicable_conditions.empty()) {
#ifndef NDEBUG
      std::cout << "[IRReorderGet] Warning: No join condition found, using "
                   "CROSS_PRODUCT"
                << std::endl;
#endif

      // Create cross product
      auto cross_product =
          std::make_unique<ir_sql_converter::SimplestCrossProduct>(
              std::move(base_stmt));
      current_tree = std::move(cross_product);

    } else {
#ifndef NDEBUG
      std::cout << "[IRReorderGet] Found " << applicable_conditions.size()
                << " join condition(s)" << std::endl;
#endif

      // Create join node
      auto join_node = std::make_unique<ir_sql_converter::SimplestJoin>(
          std::move(base_stmt), std::move(applicable_conditions),
          ir_sql_converter::SimplestJoinType::Inner);
      current_tree = std::move(join_node);
    }

    joined_tables.insert(current_table_idx);
  }

#ifndef NDEBUG
  std::cout << "[IRReorderGet] Join tree construction complete" << std::endl;
#endif
  return current_tree;
}

std::unique_ptr<ir_sql_converter::SimplestStmt>
IRReorderGet::PreserveTopOperators(
    std::unique_ptr<ir_sql_converter::SimplestStmt> original_ir,
    std::unique_ptr<ir_sql_converter::SimplestStmt> reordered_join_tree) {

  std::cout << "[IRReorderGet] Preserving top-level operators" << std::endl;

  // Recursively find the JOIN/SCAN subtree and replace it
  return ReplaceJoinSubtree(std::move(original_ir),
                            std::move(reordered_join_tree));
}

std::unique_ptr<ir_sql_converter::SimplestStmt>
IRReorderGet::ReplaceJoinSubtree(
    std::unique_ptr<ir_sql_converter::SimplestStmt> node,
    std::unique_ptr<ir_sql_converter::SimplestStmt> new_subtree) {

  if (!node) {
    return new_subtree;
  }

  auto node_type = node->GetNodeType();

  // If this node is a JOIN/SCAN/CROSS_PRODUCT, replace the whole subtree
  if (node_type == ir_sql_converter::SimplestNodeType::JoinNode ||
      node_type == ir_sql_converter::SimplestNodeType::ScanNode ||
      node_type == ir_sql_converter::SimplestNodeType::CrossProductNode) {
    std::cout << "[IRReorderGet] Replacing join subtree at "
              << GetNodeTypeName(node_type) << " node" << std::endl;
    return new_subtree;
  }

  // Otherwise, recursively replace in children (typically child[0])
  if (!node->children.empty() && node->children[0]) {
    node->children[0] = ReplaceJoinSubtree(std::move(node->children[0]),
                                           std::move(new_subtree));
  }

  return node;
}

std::string
IRReorderGet::GetNodeTypeName(ir_sql_converter::SimplestNodeType type) const {
  switch (type) {
  case ir_sql_converter::SimplestNodeType::JoinNode:
    return "JOIN";
  case ir_sql_converter::SimplestNodeType::ScanNode:
    return "SCAN";
  case ir_sql_converter::SimplestNodeType::CrossProductNode:
    return "CROSS_PRODUCT";
  case ir_sql_converter::SimplestNodeType::ProjectionNode:
    return "PROJECTION";
  case ir_sql_converter::SimplestNodeType::FilterNode:
    return "FILTER";
  case ir_sql_converter::SimplestNodeType::AggregateNode:
    return "AGGREGATE";
  default:
    return "OTHER";
  }
}

std::map<unsigned int, std::vector<ir_sql_converter::SimplestVarComparison *>>
IRReorderGet::BuildJoinMap(
    const std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
        &join_conds) {

  std::map<unsigned int, std::vector<ir_sql_converter::SimplestVarComparison *>>
      join_map;

  for (const auto &cond : join_conds) {
    unsigned int left_table = cond->left_attr->GetTableIndex();
    unsigned int right_table = cond->right_attr->GetTableIndex();

    join_map[left_table].push_back(cond.get());
    join_map[right_table].push_back(cond.get());
  }

  return join_map;
}

} // namespace middleware
