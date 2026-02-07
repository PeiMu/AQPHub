/*
 * Implementation of TopDown split strategy
 */

#include "split/topdown_splitter.h"
#include <functional>
#include <iostream>

namespace middleware {

void TopDownSplitter::Preprocess(
    std::unique_ptr<ir_sql_converter::SimplestStmt> &ir) {

  std::cout << "[TopDownSplitter] Preprocessing IR" << std::endl;

  // Reset state
  executed_tables_.clear();
  found_split_node_ = nullptr;
  query_split_index_ = 0;
  split_iteration_ = 0;
  max_table_index_ = 0;

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

  // Collect table indices and find max
  auto table_indices = CollectTableIndices(ir.get());
  for (unsigned int idx : table_indices) {
    if (idx > max_table_index_) {
      max_table_index_ = idx;
    }
  }
  std::cout << "[TopDownSplitter] Max table index: " << max_table_index_
            << std::endl;

  std::cout << "[TopDownSplitter] Finish Preprocess with IR: "
            << ir->Print(true) << std::endl;
}

void TopDownSplitter::VisitOperator(ir_sql_converter::SimplestStmt *node,
                                    bool is_top_most) {
  if (!node || found_split_node_) {
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
      //      std::cout << "[TopDownSplitter] Found AGGREGATE split point at
      //      child "
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
      found_split_node_ = child;
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
  found_split_node_ = nullptr;       // Clear previous
  VisitOperator(remaining_ir, true); // Re-identify split points

  // Check if there are more subqueries to execute
  if (!found_split_node_) {
    std::cout << "[TopDownSplitter] No more subqueries in queue" << std::endl;
    return nullptr;
  }

  std::cout << "[TopDownSplitter] Selected subquery node: "
            << GetNodeTypeName(found_split_node_->GetNodeType()) << std::endl;

  // Collect all table indices in this subquery's subtree
  auto table_indices = CollectTableIndices(found_split_node_);

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
  extraction->pipeline_breaker_ptr = found_split_node_;

  std::cout << "[TopDownSplitter] Extraction complete for "
            << GetNodeTypeName(found_split_node_->GetNodeType()) << std::endl;

  // Check for same-table issue
  std::unordered_set<std::string> table_names_in_subquery;
  if (CheckSameTableInSubtree(found_split_node_, table_names_in_subquery)) {
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

  // Count remaining base tables - complete when only 1 table left
  int remaining_tables = CountBaseTables(remaining_ir);
  bool complete = (remaining_tables <= 1);

  std::cout << "[TopDownSplitter] IsComplete: " << (complete ? "YES" : "NO")
            << " (remaining tables: " << remaining_tables << ")" << std::endl;

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

std::unique_ptr<ir_sql_converter::SimplestStmt>
TopDownSplitter::UpdateRemainingIR(
    std::unique_ptr<ir_sql_converter::SimplestStmt> remaining_ir,
    const std::set<unsigned int> &executed_table_indices,
    unsigned int temp_table_index, const std::string &temp_table_name,
    uint64_t temp_table_cardinality,
    const std::vector<std::pair<unsigned int, unsigned int>> &column_mappings,
    const std::vector<std::string> &column_names) {

  std::cout << "[TopDownSplitter::UpdateRemainingIR] Replacing executed "
               "subtree with temp table: "
            << temp_table_name << " (index " << temp_table_index
            << ", cardinality " << temp_table_cardinality << ")" << std::endl;

  if (!remaining_ir || !found_split_node_) {
    std::cerr << "[TopDownSplitter::UpdateRemainingIR] Error: null "
                 "remaining_ir or found_split_node_"
              << std::endl;
    return nullptr;
  }

  // Get raw pointer for tree traversal (we still own the IR)
  auto *ir_ptr = remaining_ir.get();

  // Helper lambda to find parent and replace child
  std::function<bool(ir_sql_converter::SimplestStmt *)> ReplaceInTree;
  ReplaceInTree = [&](ir_sql_converter::SimplestStmt *node) -> bool {
    if (!node)
      return false;

    for (size_t i = 0; i < node->children.size(); i++) {
      if (node->children[i].get() == found_split_node_) {
        // Found the split node - create SimplestScan to replace it
        std::cout
            << "[TopDownSplitter::UpdateRemainingIR] Found split node at child "
            << i << ", replacing with SimplestScan for temp table" << std::endl;

        // Build target list using pre-computed column names
        std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
            scan_target_list;
        for (size_t col_idx = 0; col_idx < column_names.size(); col_idx++) {
          ir_sql_converter::SimplestVarType col_type =
              ir_sql_converter::SimplestVarType::IntVar;
          if (col_idx < found_split_node_->target_list.size()) {
            col_type = found_split_node_->target_list[col_idx]->GetType();
          }

          auto attr = std::make_unique<ir_sql_converter::SimplestAttr>(
              col_type, temp_table_index, static_cast<unsigned int>(col_idx),
              column_names[col_idx]);
          scan_target_list.push_back(std::move(attr));
        }

        // Create base SimplestStmt for the scan
        std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>>
            empty_children;
        auto scan_base = std::make_unique<ir_sql_converter::SimplestStmt>(
            std::move(empty_children), std::move(scan_target_list),
            ir_sql_converter::SimplestNodeType::ScanNode);

        // Create SimplestScan node for temp table (treat it like a base table)
        auto scan_node = std::make_unique<ir_sql_converter::SimplestScan>(
            std::move(scan_base), temp_table_index, temp_table_name);
        scan_node->SetEstimatedCardinality(temp_table_cardinality);

        // Replace the child
        node->children[i] = std::move(scan_node);

        std::cout << "[TopDownSplitter::UpdateRemainingIR] Successfully "
                     "replaced subtree"
                  << std::endl;
        return true;
      }

      // Recursively search in children
      if (ReplaceInTree(node->children[i].get())) {
        return true;
      }
    }

    return false;
  };

  // Find and replace the split node in tree
  // Note: The case where remaining_ir == found_split_node_ should not happen
  // because IsComplete returns true when only 1 table remains (like DuckDB line
  // 605)
  if (!ReplaceInTree(ir_ptr)) {
    std::cerr << "[TopDownSplitter::UpdateRemainingIR] Warning: Could not find "
                 "split node in tree"
              << std::endl;
  }

  // Return the modified IR (same IR, modified in-place)
  return remaining_ir;
}

} // namespace middleware
