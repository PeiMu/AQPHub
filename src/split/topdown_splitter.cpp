/*
 * Implementation of TopDown split strategy
 */

#include "split/topdown_splitter.h"
#include <iostream>

namespace middleware {

void TopDownSplitter::Preprocess(
    std::unique_ptr<ir_sql_converter::SimplestStmt> &ir) {

  std::cout << "[TopDownSplitter] Preprocessing IR" << std::endl;

  // Reset state
  executed_tables_.clear();
  sub_ir = nullptr;
  query_split_index_ = 0;
  split_iteration_ = 0;

  if (enable_reorder_) {
    std::cout << "[TopDownSplitter] Running IR-level ReorderGet" << std::endl;
    auto reordered = reorder_get_.Reorder(std::move(ir));

    if (reordered) {
      ir = std::move(reordered);
      std::cout << "[TopDownSplitter] ReorderGet completed successfully"
                << std::endl;
    } else {
      std::cout
          << "[TopDownSplitter] ReorderGet returned null, using original IR"
          << std::endl;
    }
  } else {
    std::cout << "[TopDownSplitter] ReorderGet disabled, skipping" << std::endl;
  }

  std::cout << "[TopDownSplitter] Finish Preprocess with IR: "
            << ir->Print(true) << std::endl;
}

void TopDownSplitter::VisitOperator(ir_sql_converter::SimplestStmt *node,
                                    bool is_top_most) {
  if (!node || sub_ir) {
    return;
  }

  // Process children from RIGHT to LEFT (following DuckDB pattern)
  // This ensures we get build side (right child) first in left-deep plans
  for (int idx = node->children.size() - 1; idx >= 0; idx--) {
    auto *child = node->children[idx].get();

    if (!child) {
      continue;
    }

    // Recursively visit this child
    VisitOperator(child, false);

    auto child_type = child->GetNodeType();

    // Check if this child should be a split point
    bool should_split = false;

    switch (child_type) {
    case ir_sql_converter::SimplestNodeType::FilterNode: {
      // Split at FILTER nodes
      // Following DuckDB lines 187-234
      should_split = true;
      // TODO
      std::cout << "[TopDownSplitter] Found FILTER split point at child " << idx
                << std::endl;
      break;
    }

    case ir_sql_converter::SimplestNodeType::JoinNode: {
      // Split at JOIN nodes (build side = right child)
      // Following DuckDB lines 236-264
      auto *join = dynamic_cast<ir_sql_converter::SimplestJoin *>(child);
      if (join) {
        auto join_type = join->GetSimplestJoinType();

        // Skip SEMI/MARK joins (like DuckDB lines 240-243)
        if (join_type == ir_sql_converter::SimplestJoinType::Semi ||
            join_type == ir_sql_converter::SimplestJoinType::Mark) {
          std::cout << "[TopDownSplitter] Skipping SEMI/MARK join" << std::endl;
          should_split = false;
        } else {
          // Split at top-most JOIN or right child (build side)
          // DuckDB line 246: if (top_most || 1 == idx)
          // fixme: check if the parent node is join
          if (is_top_most || idx == 1) {
            should_split = true;
            std::cout << "[TopDownSplitter] Found JOIN split point at child "
                      << idx << " (build side)" << std::endl;
          }
        }
      }
      break;
    }

    case ir_sql_converter::SimplestNodeType::CrossProductNode: {
      // Following DuckDB lines 266-274
      // Split if this is the left child and parent has empty right child
      if (idx == 0 && node->children.size() == 2 && !node->children[1]) {
        should_split = true;
        std::cout << "[TopDownSplitter] Found CROSS_PRODUCT split point"
                  << std::endl;
      }
      break;
    }

//    case ir_sql_converter::SimplestNodeType::AggregateNode: {
//      // Aggregate is also a split point
//      should_split = true;
//      std::cout << "[TopDownSplitter] Found AGGREGATE split point at child "
//                << idx << std::endl;
//      break;
//    }

    default:
      should_split = false;
      break;
    }


    // If this is a split point, add to queue
    if (should_split) {
      query_split_index_++;
      sub_ir = child;
      std::cout << "[TopDownSplitter] Added split point #" << query_split_index_
                << ": " << GetNodeTypeName(child_type) << std::endl;
    }
  }
}

std::unique_ptr<SubqueryExtraction> TopDownSplitter::ExtractNextSubquery(
    ir_sql_converter::SimplestStmt *remaining_ir) {

  split_iteration_++;
  std::cout << "\n[TopDownSplitter] Iteration " << split_iteration_
            << ": Extracting next subquery" << std::endl;

  if (!remaining_ir) {
    std::cout << "[TopDownSplitter] Remaining IR is null" << std::endl;
    return nullptr;
  }

  // Re-visit the UPDATED tree to find the next split point
  sub_ir = nullptr;                  // Clear previous
  VisitOperator(remaining_ir, true); // Re-identify split points

  // Check if there are more subqueries to execute
  if (!sub_ir) {
    std::cout << "[TopDownSplitter] No more subqueries in queue" << std::endl;
    return nullptr;
  }

  std::cout << "[TopDownSplitter] Selected subquery node: "
            << GetNodeTypeName(sub_ir->GetNodeType()) << std::endl;

  // Collect all table indices in this subquery's subtree
  auto table_indices = CollectTableIndices(sub_ir);

  std::cout << "[TopDownSplitter] Tables involved: ";
  for (auto idx : table_indices) {
    std::cout << idx << " ";
    executed_tables_.insert(idx);
  }
  std::cout << "(" << table_indices.size() << " tables)" << std::endl;

  std::string temp_table_name = "temp_" + std::to_string(split_iteration_);

  // Create extraction info
  // The sub-plan INCLUDES the subquery_node itself
  auto extraction =
      std::make_unique<SubqueryExtraction>(table_indices, temp_table_name);

  // Store pointer to the node that represents this subquery
  extraction->pipeline_breaker_ptr = sub_ir;

  std::cout << "[TopDownSplitter] Extraction complete for "
            << GetNodeTypeName(sub_ir->GetNodeType()) << std::endl;

  // Check for same-table issue
  std::unordered_set<std::string> table_names_in_subquery;
  if (CheckSameTableInSubtree(sub_ir, table_names_in_subquery)) {
    // TODO: implement merge-back logic like DuckDB
    throw std::runtime_error("todo: fix same-table issue!");
  }

  return extraction;
}

bool TopDownSplitter::CheckSameTableInSubtree(
    ir_sql_converter::SimplestStmt *node,
    std::unordered_set<std::string> &seen_tables) const {

  if (!node)
    return false;

  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = dynamic_cast<ir_sql_converter::SimplestScan *>(node);
    if (scan) {
      std::string table_name = scan->GetTableName();
      if (seen_tables.count(table_name)) {
        return true; // DUPLICATE FOUND
      }
      seen_tables.insert(table_name);
    }
  }

  for (const auto &child : node->children) {
    if (CheckSameTableInSubtree(child.get(), seen_tables)) {
      return true;
    }
  }

  return false;
}

bool TopDownSplitter::IsComplete(
    const ir_sql_converter::SimplestStmt *remaining_ir) {

  if (!remaining_ir) {
    return true;
  }

  // Check if there are any more subqueries in the queue
  bool complete = !sub_ir;

  std::cout << "[TopDownSplitter] IsComplete: " << (complete ? "YES" : "NO")
            << std::endl;

  return complete;
}

std::set<unsigned int> TopDownSplitter::CollectTableIndices(
    const ir_sql_converter::SimplestStmt *node) const {

  std::set<unsigned int> indices;

  if (!node)
    return indices;

  // If this is a scan node, add its table index
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = dynamic_cast<const ir_sql_converter::SimplestScan *>(node);
    if (scan) {
      indices.insert(scan->GetTableIndex());
    }
  }

  // If this is a chunk node (temp table), add its index
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ChunkNode) {
    auto *chunk = dynamic_cast<const ir_sql_converter::SimplestChunk *>(node);
    if (chunk) {
      indices.insert(chunk->GetTableIndex());
    }
  }

  // Recursively collect from children
  for (const auto &child : node->children) {
    auto child_indices = CollectTableIndices(child.get());
    indices.insert(child_indices.begin(), child_indices.end());
  }

  return indices;
}

int TopDownSplitter::CountBaseTables(
    const ir_sql_converter::SimplestStmt *node) const {

  if (!node)
    return 0;

  int count = 0;

  // Count scan nodes
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    count = 1;
  }

  // Recursively count in children
  for (const auto &child : node->children) {
    count += CountBaseTables(child.get());
  }

  return count;
}

std::string TopDownSplitter::GetNodeTypeName(
    ir_sql_converter::SimplestNodeType type) const {
  switch (type) {
  case ir_sql_converter::SimplestNodeType::JoinNode:
    return "JOIN";
  case ir_sql_converter::SimplestNodeType::AggregateNode:
    return "AGGREGATE";
  case ir_sql_converter::SimplestNodeType::FilterNode:
    return "FILTER";
  case ir_sql_converter::SimplestNodeType::ScanNode:
    return "SCAN";
  case ir_sql_converter::SimplestNodeType::ProjectionNode:
    return "PROJECTION";
  case ir_sql_converter::SimplestNodeType::CrossProductNode:
    return "CROSS_PRODUCT";
  case ir_sql_converter::SimplestNodeType::OrderNode:
    return "ORDER";
  case ir_sql_converter::SimplestNodeType::LimitNode:
    return "LIMIT";
  case ir_sql_converter::SimplestNodeType::ChunkNode:
    return "CHUNK (temp table)";
  default:
    return "UNKNOWN(" + std::to_string((int)type) + ")";
  }
}

} // namespace middleware
