/*
 * Implementation of TopDown split strategy
 */

#include "split/topdown_splitter.h"
#include <functional>
#include <iostream>

namespace middleware {

// Recursively walk the IR and record (table_name, col_idx) → col_name
// for every attribute that references a ScanNode table.
static void ExtractColumnNames(
    const ir_sql_converter::AQPStmt *node,
    const std::map<unsigned int, std::string> &idx_to_name,
    std::map<std::pair<std::string, unsigned int>, std::string> &out) {
  if (!node)
    return;

  // From target_list
  for (const auto &attr : node->target_list) {
    unsigned int tidx = attr->GetTableIndex();
    auto it = idx_to_name.find(tidx);
    if (it != idx_to_name.end()) {
      auto key = std::make_pair(it->second, attr->GetColumnIndex());
      if (out.find(key) == out.end())
        out[key] = attr->GetColumnName();
    }
  }

  // From join conditions
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *join = dynamic_cast<const ir_sql_converter::SimplestJoin *>(node);
    if (join) {
      for (const auto &cond : join->join_conditions) {
        if (cond->left_attr) {
          auto it = idx_to_name.find(cond->left_attr->GetTableIndex());
          if (it != idx_to_name.end()) {
            auto key =
                std::make_pair(it->second, cond->left_attr->GetColumnIndex());
            if (out.find(key) == out.end())
              out[key] = cond->left_attr->GetColumnName();
          }
        }
        if (cond->right_attr) {
          auto it = idx_to_name.find(cond->right_attr->GetTableIndex());
          if (it != idx_to_name.end()) {
            auto key =
                std::make_pair(it->second, cond->right_attr->GetColumnIndex());
            if (out.find(key) == out.end())
              out[key] = cond->right_attr->GetColumnName();
          }
        }
      }
    }
  }

  for (const auto &child : node->children)
    ExtractColumnNames(child.get(), idx_to_name, out);
}

void TopDownSplitter::BuildColumnNameMap(
    const ir_sql_converter::AQPStmt *ir) {
  // First collect table_index → table_name mapping
  std::map<unsigned int, std::string> idx_to_name;
  std::function<void(const ir_sql_converter::AQPStmt *)> CollectTables;
  CollectTables = [&](const ir_sql_converter::AQPStmt *node) {
    if (!node)
      return;
    if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
      auto *scan = dynamic_cast<const ir_sql_converter::SimplestScan *>(node);
      if (scan)
        idx_to_name[scan->GetTableIndex()] = scan->GetTableName();
    }
    for (const auto &child : node->children)
      CollectTables(child.get());
  };
  CollectTables(ir);

  // Then extract column names from all attrs in the IR
  ExtractColumnNames(ir, idx_to_name, col_name_map_);
}

std::string TopDownSplitter::LookupColumnName(const std::string &table_name,
                                              unsigned int col_idx) const {
  auto it = col_name_map_.find(std::make_pair(table_name, col_idx));
  if (it != col_name_map_.end())
    return it->second;
  return "";
}

// Find the maximum table_index across all attributes in the IR tree.
// DuckDB assigns indices to projection/aggregate nodes too, and those
// can be higher than the scan/chunk indices. We must track them so that
// newly created temp table indices don't collide.
static unsigned int FindMaxAttrIndex(const ir_sql_converter::AQPStmt *node) {
  if (!node)
    return 0;
  unsigned int mx = 0;
  for (const auto &attr : node->target_list) {
    if (attr->GetTableIndex() > mx)
      mx = attr->GetTableIndex();
  }
  for (const auto &child : node->children)
    mx = std::max(mx, FindMaxAttrIndex(child.get()));
  return mx;
}

void TopDownSplitter::Preprocess(
    std::unique_ptr<ir_sql_converter::AQPStmt> &ir) {

#ifndef NDEBUG
  std::cout << "[TopDownSplitter] Preprocessing IR" << std::endl;
#endif

  // Reset state
  executed_tables_.clear();
  found_split_node_ = nullptr;
  query_split_index_ = 0;
  split_iteration_ = 0;
  max_table_index_ = 0;
  col_name_map_.clear();

  // Build column name map from the ORIGINAL IR (before ReOptimizeIR may
  // corrupt names through DuckDB's equivalent-column binding resolution).
  BuildColumnNameMap(ir.get());

  bool did_reoptimize = false;
  if (enable_reorder_) {
#ifndef NDEBUG
    std::cout << "[TopDownSplitter] Running ReOptimizeIR in Preprocess"
              << std::endl;
#endif
    auto re_optimized = adapter_->ReOptimizeIR(std::move(ir));
    if (re_optimized) {
      ir = std::move(re_optimized);
      did_reoptimize = true;
#ifndef NDEBUG
      std::cout << "[TopDownSplitter] ReOptimizeIR completed successfully"
                << std::endl;
#endif
    } else {
      std::cout
          << "[TopDownSplitter] ReOptimizeIR returned null, using original IR"
          << std::endl;
    }
  }

  // Collect table names and find max index (once for all paths)
  table_index_to_name_.clear();
  CollectTableNames(ir.get());
  max_table_index_ = FindMaxAttrIndex(ir.get());

  // Fix attr names corrupted by ReOptimizeIR's equivalent-column resolution
  if (did_reoptimize) {
    FixAllAttrNames(ir.get());
  }
#ifndef NDEBUG
  std::cout << "[TopDownSplitter] Max table index: " << max_table_index_
            << std::endl;

  std::cout << "[TopDownSplitter] Finish Preprocess with IR: "
            << ir->Print(true) << std::endl;
#endif
}

void TopDownSplitter::Visit(ir_sql_converter::AQPStmt *node) {
  if (!node || found_split_node_) {
    return;
  }

  // Process children from RIGHT to LEFT (following DuckDB pattern).
  // This ensures the build side (right child) is considered first in
  // left-deep plans, matching DuckDB's VisitOperator traversal order.
  for (int idx = node->children.size() - 1; idx >= 0; idx--) {
    auto *child = node->children[idx].get();
    if (!child) {
      continue;
    }

    auto child_type = child->GetNodeType();
    bool should_split = false;

    // ── STEP 1: Decide split (BEFORE recursing) ──────────────────────────
    // This mirrors DuckDB's VisitOperator which sets child->split_index
    // before calling VisitOperator(*child).  The key effect is that
    // top_most_ is updated here, so the recursive call already sees the
    // updated value (just like DuckDB's top_most member variable).
    switch (child_type) {

    case ir_sql_converter::SimplestNodeType::FilterNode: {
      // DuckDB lines 182-191 (follow_pipeline_breaker=true path):
      //   if (top_most && 0 == idx) → split, top_most=false
      // DuckDB lines 211-213:
      //   if (follow_pipeline_breaker_ && 0 == idx) → break (NO split)
      // So a FILTER only becomes a subquery when it is the very first
      // split-worthy node we encounter in the traversal (top_most_=true)
      // and it is a left/only child (idx==0).  Inner FILTERs that have
      // been pushed below a JOIN by FilterPushdown are NOT split.
      if (top_most_ && idx == 0) {
        top_most_ = false;
        should_split = true;
#ifndef NDEBUG
        std::cout << "[TopDownSplitter] Found FILTER split point at child "
                  << idx << " (top-most)" << std::endl;
#endif
      }
      // else: non-top-most FILTER at idx==0 → follow_pipeline_breaker break
      break;
    }

    case ir_sql_converter::SimplestNodeType::JoinNode: {
      // DuckDB lines 239-274.
      auto *join = dynamic_cast<ir_sql_converter::SimplestJoin *>(child);
      if (join) {
        auto join_type = join->GetSimplestJoinType();
        if (join_type == ir_sql_converter::SimplestJoinType::Semi ||
            join_type == ir_sql_converter::SimplestJoinType::Mark) {
          // DuckDB lines 243-246: skip SEMI/MARK, set split_index=0
#ifndef NDEBUG
          std::cout << "[TopDownSplitter] Skipping SEMI/MARK join" << std::endl;
#endif
        } else {
          // DuckDB lines 248-253: split if top_most || right child (idx==1)
          if (top_most_ || idx == 1) {
            should_split = true;
#ifndef NDEBUG
            std::cout << "[TopDownSplitter] Found JOIN split point at child "
                      << idx << (top_most_ ? " (top-most)" : " (build side)")
                      << std::endl;
#endif
          }
          top_most_ =
              false; // DuckDB line 274: always clear after any INNER JOIN
        }
      }
      break;
    }

    case ir_sql_converter::SimplestNodeType::CrossProductNode: {
      // DuckDB lines 277-285: CROSS_PRODUCT only acts as a sibling marker
      // (sets split_index = current, no increment) when
      // follow_pipeline_breaker. In the middleware's single-split-per-call
      // design this is a no-op.
      break;
    }

    default:
      break;
    }

    // ── STEP 2: Recurse into child (AFTER split decision) ────────────────
    // By recursing after the decision, top_most_ is already false when
    // children of a JOIN are visited, so pushed-down FILTERs inside the
    // join tree will not be mistakenly picked as split points.
    Visit(child);

    // ── STEP 3: Record the split point and stop ───────────────────────────
    if (should_split) {
      query_split_index_++;
      found_split_node_ = child;
#ifndef NDEBUG
      std::cout << "[TopDownSplitter] Added split point #" << query_split_index_
                << ": " << GetNodeTypeName(child_type) << std::endl;
#endif
      return; // One split per Visit call — stop here.
    }
  }
}

// Walk an expression tree and invoke cb(attr_ptr) on every SimplestAttr.
// Shared visitor used by both FixExprNames and CollectRequiredAttrs.
template <typename Callback>
static void ForEachAttrInExpr(const ir_sql_converter::AQPExpr *expr,
                              Callback &&cb) {
  if (!expr)
    return;
  auto nt = expr->GetNodeType();
  if (nt == ir_sql_converter::SimplestNodeType::VarComparisonNode) {
    auto *vc =
        dynamic_cast<const ir_sql_converter::SimplestVarComparison *>(expr);
    if (vc) {
      cb(vc->left_attr.get());
      cb(vc->right_attr.get());
    }
  } else if (nt == ir_sql_converter::SimplestNodeType::VarConstComparisonNode) {
    auto *vcc =
        dynamic_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
    if (vcc)
      cb(vcc->attr.get());
  } else if (nt == ir_sql_converter::SimplestNodeType::IsNullExprNode) {
    auto *isn =
        dynamic_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    if (isn)
      cb(isn->attr.get());
  } else if (nt == ir_sql_converter::SimplestNodeType::VarParamComparisonNode) {
    auto *vp =
        dynamic_cast<const ir_sql_converter::SimplestVarParamComparison *>(expr);
    if (vp)
      cb(vp->attr.get());
  } else if (nt == ir_sql_converter::SimplestNodeType::SingleAttrExprNode) {
    auto *sa =
        dynamic_cast<const ir_sql_converter::SimplestSingleAttrExpr *>(expr);
    if (sa)
      cb(sa->attr.get());
  } else if (nt == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *le =
        dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (le) {
      ForEachAttrInExpr(le->left_expr.get(), cb);
      ForEachAttrInExpr(le->right_expr.get(), cb);
    }
  } else if (nt == ir_sql_converter::SimplestNodeType::InExprNode) {
    auto *in = dynamic_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    if (in)
      cb(in->attr.get());
  } else if (nt == ir_sql_converter::SimplestNodeType::ArithExprNode) {
    auto *ar =
        dynamic_cast<const ir_sql_converter::SimplestArithExpr *>(expr);
    if (ar) {
      ForEachAttrInExpr(ar->left.get(), cb);
      ForEachAttrInExpr(ar->right.get(), cb);
    }
  } else if (nt == ir_sql_converter::SimplestNodeType::CastExprNode) {
    auto *cast =
        dynamic_cast<const ir_sql_converter::SimplestCastExpr *>(expr);
    if (cast)
      ForEachAttrInExpr(cast->child.get(), cb);
  }
}

// Fix a single attr's column name using the col_name_map_.
void TopDownSplitter::FixAttrName(ir_sql_converter::SimplestAttr *attr) const {
  if (!attr)
    return;
  auto tit = table_index_to_name_.find(attr->GetTableIndex());
  if (tit == table_index_to_name_.end())
    return;
  auto cit =
      col_name_map_.find(std::make_pair(tit->second, attr->GetColumnIndex()));
  if (cit != col_name_map_.end() && cit->second != attr->GetColumnName())
    attr->SetColumnName(cit->second);
}

// Walk an expression tree and fix all attr names.
void TopDownSplitter::FixExprNames(ir_sql_converter::AQPExpr *expr) const {
  ForEachAttrInExpr(expr, [this](ir_sql_converter::SimplestAttr *a) {
    FixAttrName(a);
  });
}

// Recursively fix all attr names in the IR tree.
void TopDownSplitter::FixAllAttrNames(ir_sql_converter::AQPStmt *node) const {
  if (!node)
    return;
  for (auto &attr : node->target_list)
    FixAttrName(attr.get());
  for (auto &qual : node->qual_vec)
    FixExprNames(qual.get());
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *join = dynamic_cast<ir_sql_converter::SimplestJoin *>(node);
    if (join) {
      for (auto &cond : join->join_conditions) {
        FixAttrName(cond->left_attr.get());
        FixAttrName(cond->right_attr.get());
      }
    }
  }
  if (node->GetNodeType() ==
      ir_sql_converter::SimplestNodeType::AggregateNode) {
    auto *agg = dynamic_cast<ir_sql_converter::SimplestAggregate *>(node);
    if (agg) {
      for (auto &grp : agg->groups)
        FixAttrName(grp.get());
      for (auto &fn : agg->agg_fns)
        FixAttrName(fn.first.get());
    }
  }
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::OrderNode) {
    auto *order = dynamic_cast<ir_sql_converter::SimplestOrderBy *>(node);
    if (order) {
      for (auto &ord : order->orders)
        FixAttrName(ord.attr.get());
    }
  }
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::HashNode) {
    auto *hash = dynamic_cast<ir_sql_converter::SimplestHash *>(node);
    if (hash) {
      for (auto &key : hash->hash_keys)
        FixAttrName(key.get());
    }
  }
  for (auto &child : node->children)
    FixAllAttrNames(child.get());
}

void TopDownSplitter::ReorderBeforeSplit(
    std::unique_ptr<ir_sql_converter::AQPStmt> &ir) {
  if (!ir)
    return;

  // Preprocess already ran ReOptimizeIR for iteration 1
  if (split_iteration_ <= 1)
    return;

  // Re-plan through the engine's full optimizer.  This converts
  // CROSS_PRODUCT + filter conditions into proper hash joins, runs
  // join order optimization with actual temp table cardinalities, and
  // prunes unused columns.
  auto re_optimized = adapter_->ReOptimizeIR(std::move(ir));
  if (re_optimized) {
    ir = std::move(re_optimized);
    table_index_to_name_.clear();
    CollectTableNames(ir.get());
    max_table_index_ = FindMaxAttrIndex(ir.get());
    FixAllAttrNames(ir.get());
#ifndef NDEBUG
    std::cout << "[TopDownSplitter::ReorderBeforeSplit] Re-optimized IR "
              << "(max_table_index=" << max_table_index_ << "):\n"
              << ir->Print(false) << std::endl;
#endif
  }
}

std::unique_ptr<SubqueryExtraction>
TopDownSplitter::SplitIR(ir_sql_converter::AQPStmt *remaining_ir) {

  split_iteration_++;
#ifndef NDEBUG
  std::cout << "\n[TopDownSplitter] Iteration " << split_iteration_
            << ": Extracting next subquery" << std::endl;
#endif

  if (!remaining_ir) {
#ifndef NDEBUG
    std::cout << "[TopDownSplitter] Remaining IR is null" << std::endl;
#endif
    return nullptr;
  }

  // Re-visit the UPDATED tree to find the next split point.
  // Reset top_most_ to true before each traversal, mirroring DuckDB's
  // TopDownSplit::Clear() which resets the top_most member between iterations.
  found_split_node_ = nullptr;
  top_most_ = true;
  Visit(remaining_ir);

  // If Visit() found no split point but multiple tables remain, fall back to
  // the deepest right-child subtree.  This handles CROSS_PRODUCT-only trees
  // and SEMI/MARK subtrees that Visit() skips.
  if (!found_split_node_) {
    if (CountBaseTables(remaining_ir) > 1) {
      found_split_node_ = FindDeepestRightChild(remaining_ir);
      if (found_split_node_) {
#ifndef NDEBUG
        std::cout << "[TopDownSplitter] Fallback: using deepest right child as "
                     "split point"
                  << std::endl;
#endif
      }
    }
    if (!found_split_node_) {
#ifndef NDEBUG
      std::cout << "[TopDownSplitter] No more subqueries in queue" << std::endl;
#endif
      return nullptr;
    }
  }

  // In a left-deep tree, Visit() picks the outermost join which covers
  // ALL tables.  When that happens, drill down to the deepest join so the
  // first subquery is the smallest join (2-3 tables), matching DuckDB's
  // Split+ReorderTables+MergeSubquery+ReSplit cycle behavior.
  int total_tables = CountBaseTables(remaining_ir);
  int split_tables = CountBaseTables(found_split_node_);
  if (split_tables >= total_tables && total_tables > 2) {
    auto *deeper = FindDeepestJoin(found_split_node_);
    if (deeper && CountBaseTables(deeper) < split_tables) {
#ifndef NDEBUG
      std::cout << "[TopDownSplitter] Split covers all " << total_tables
                << " tables, drilling down to deepest join ("
                << CountBaseTables(deeper) << " tables)" << std::endl;
#endif
      found_split_node_ = deeper;
    }
  }

#ifndef NDEBUG
  std::cout << "[TopDownSplitter] Selected subquery node: "
            << GetNodeTypeName(found_split_node_->GetNodeType()) << std::endl;
#endif

  // Collect all table indices in this subquery's subtree
  auto table_indices = CollectTableIndices(found_split_node_);

#ifndef NDEBUG
  std::cout << "[TopDownSplitter] Tables involved: ";
#endif
  for (auto idx : table_indices) {
#ifndef NDEBUG
    std::cout << idx << " ";
#endif
    executed_tables_.insert(idx);
  }
#ifndef NDEBUG
  std::cout << "(" << table_indices.size() << " tables)" << std::endl;
#endif

  std::string temp_table_name = "temp_" + std::to_string(split_iteration_);

  // Create extraction info
  auto extraction =
      std::make_unique<SubqueryExtraction>(table_indices, temp_table_name);

  // Wrap found_split_node_ in a Projection with only the columns needed by
  // the remaining IR.  This gives the sub-query a well-defined SELECT list and
  // minimises the columns stored in the temp table.
  auto required_attrs = CollectRequiredAttrs(remaining_ir, table_indices);
#ifndef NDEBUG
  std::cout << "[TopDownSplitter] Wrapping split node in Projection with "
            << required_attrs.size() << " required column(s)" << std::endl;
#endif
  WrapInProjection(remaining_ir, std::move(required_attrs));
  // found_split_node_ now points to the new Projection node

  // Store pointer to the projection (used as both executable IR and replace
  // target in UpdateRemainingIR)
  extraction->pipeline_breaker_ptr = found_split_node_;

#ifndef NDEBUG
  std::cout << "[TopDownSplitter] Extraction complete for "
            << GetNodeTypeName(found_split_node_->GetNodeType()) << std::endl;
#endif

  // Check for same-table issue
  // DuckDB's same-table handling is also commented out in top_down.cpp — just
  // warn and continue rather than crashing.
  std::unordered_set<std::string> table_names_in_subquery;
  if (CheckSameTableInSubtree(found_split_node_, table_names_in_subquery)) {
#ifndef NDEBUG
    std::cerr << "[TopDownSplitter] Warning: same table appears multiple times "
                 "in subquery subtree; same-table merge not yet implemented"
              << std::endl;
#endif
  }

  return extraction;
}

ir_sql_converter::AQPStmt *
TopDownSplitter::FindDeepestRightChild(ir_sql_converter::AQPStmt *node) const {
  if (!node || node->children.size() < 2)
    return nullptr;

  // Recurse into right child first (depth-first)
  auto *deeper = FindDeepestRightChild(node->children.back().get());
  if (deeper)
    return deeper;

  // No deeper right child — this node's right child is the deepest
  return node->children.back().get();
}

ir_sql_converter::AQPStmt *
TopDownSplitter::FindDeepestJoin(ir_sql_converter::AQPStmt *node) const {
  if (!node)
    return nullptr;

  // Walk left-deep chain: follow children that are JoinNodes
  for (auto &child : node->children) {
    auto ct = child->GetNodeType();
    if (ct == ir_sql_converter::SimplestNodeType::JoinNode) {
      auto *deeper = FindDeepestJoin(child.get());
      if (deeper)
        return deeper;
    }
  }

  // No deeper join found — this node is the deepest join
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode)
    return node;

  return nullptr;
}

bool TopDownSplitter::CheckSameTableInSubtree(
    ir_sql_converter::AQPStmt *node,
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
    const ir_sql_converter::AQPStmt *remaining_ir) {

  if (!remaining_ir) {
    return true;
  }

  // Count remaining base tables - complete when only 1 table left
  int remaining_tables = CountBaseTables(remaining_ir);
  bool complete = (remaining_tables <= 1);

#ifndef NDEBUG
  std::cout << "[TopDownSplitter] IsComplete: " << (complete ? "YES" : "NO")
            << " (remaining tables: " << remaining_tables << ")" << std::endl;
#endif

  return complete;
}

std::set<unsigned int> TopDownSplitter::CollectTableIndices(
    const ir_sql_converter::AQPStmt *node) const {

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
    const ir_sql_converter::AQPStmt *node) const {

  if (!node)
    return 0;

  int count = 0;

  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode ||
      node->GetNodeType() == ir_sql_converter::SimplestNodeType::ChunkNode) {
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

std::unique_ptr<ir_sql_converter::AQPStmt> TopDownSplitter::UpdateRemainingIR(
    std::unique_ptr<ir_sql_converter::AQPStmt> remaining_ir,
    const std::set<unsigned int> &executed_table_indices,
    unsigned int temp_table_index, const std::string &temp_table_name,
    uint64_t temp_table_cardinality,
    const std::vector<std::pair<unsigned int, unsigned int>> &column_mappings,
    const std::vector<std::string> &column_names) {

  // Register temp table column names so subsequent ReOptimizeIR roundtrips
  // can resolve correct names for ChunkNode attributes.
  for (size_t i = 0; i < column_names.size(); i++) {
    col_name_map_[std::make_pair(temp_table_name, (unsigned int)i)] =
        column_names[i];
  }

#ifndef NDEBUG
  std::cout << "[TopDownSplitter::UpdateRemainingIR] Replacing executed "
               "subtree with temp table: "
            << temp_table_name << " (index " << temp_table_index
            << ", cardinality " << temp_table_cardinality << ")" << std::endl;
#endif

  if (!remaining_ir || !found_split_node_) {
    std::cerr << "[TopDownSplitter::UpdateRemainingIR] Error: null "
                 "remaining_ir or found_split_node_"
              << std::endl;
    return nullptr;
  }

  // Get raw pointer for tree traversal (we still own the IR)
  auto *ir_ptr = remaining_ir.get();

  // Helper lambda to find parent and replace child
  std::function<bool(ir_sql_converter::AQPStmt *)> ReplaceInTree;
  ReplaceInTree = [&](ir_sql_converter::AQPStmt *node) -> bool {
    if (!node)
      return false;

    for (size_t i = 0; i < node->children.size(); i++) {
      if (node->children[i].get() == found_split_node_) {
        // Found the split node - create SimplestScan to replace it
#ifndef NDEBUG
        std::cout
            << "[TopDownSplitter::UpdateRemainingIR] Found split node at child "
            << i << ", replacing with SimplestScan for temp table" << std::endl;
#endif

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

        // Create base AQPStmt for the scan
        std::vector<std::unique_ptr<ir_sql_converter::AQPStmt>> empty_children;
        auto scan_base = std::make_unique<ir_sql_converter::AQPStmt>(
            std::move(empty_children), std::move(scan_target_list),
            ir_sql_converter::SimplestNodeType::ScanNode);

        // Create SimplestScan node for temp table (treat it like a base table)
        auto scan_node = std::make_unique<ir_sql_converter::SimplestScan>(
            std::move(scan_base), temp_table_index, temp_table_name);
        scan_node->SetEstimatedCardinality(temp_table_cardinality);

        // Replace the child
        node->children[i] = std::move(scan_node);

#ifndef NDEBUG
        std::cout << "[TopDownSplitter::UpdateRemainingIR] Successfully "
                     "replaced subtree"
                  << std::endl;
#endif
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

// Resolve the correct column name for (table_index, col_index) by finding
// the ScanNode or ChunkNode in the IR tree and reading its target_list.
// DuckDB's optimizer may assign equivalent-column names from the wrong side
// of a join, so we trust the scan node's target_list over join-condition attrs.
static std::string ResolveColumnName(const ir_sql_converter::AQPStmt *root,
                                     unsigned int table_index,
                                     unsigned int col_index) {
  if (!root)
    return "";
  if (root->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = dynamic_cast<const ir_sql_converter::SimplestScan *>(root);
    if (scan && scan->GetTableIndex() == table_index) {
      if (col_index < root->target_list.size())
        return root->target_list[col_index]->GetColumnName();
    }
  } else if (root->GetNodeType() ==
             ir_sql_converter::SimplestNodeType::ChunkNode) {
    auto *chunk = dynamic_cast<const ir_sql_converter::SimplestChunk *>(root);
    if (chunk && chunk->GetTableIndex() == table_index) {
      if (col_index < root->target_list.size())
        return root->target_list[col_index]->GetColumnName();
    }
  }
  for (const auto &child : root->children) {
    auto name = ResolveColumnName(child.get(), table_index, col_index);
    if (!name.empty())
      return name;
  }
  return "";
}

std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
TopDownSplitter::CollectRequiredAttrs(
    const ir_sql_converter::AQPStmt *full_ir,
    const std::set<unsigned int> &subquery_tables) const {

  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> required_attrs;
  std::set<std::pair<unsigned int, unsigned int>> seen_attrs;

  auto AddIfSubqueryAttr = [&](const ir_sql_converter::SimplestAttr *attr) {
    if (!attr)
      return;
    if (subquery_tables.count(attr->GetTableIndex())) {
      auto key = std::make_pair(attr->GetTableIndex(), attr->GetColumnIndex());
      if (!seen_attrs.count(key)) {
        seen_attrs.insert(key);
        // Resolve correct column name: prefer the pre-ReOptimizeIR map
        // (built from the original IR which has correct names), then
        // fall back to the scan node in the current IR, then the attr itself.
        std::string resolved_name;
        auto tname_it = table_index_to_name_.find(attr->GetTableIndex());
        if (tname_it != table_index_to_name_.end()) {
          resolved_name =
              LookupColumnName(tname_it->second, attr->GetColumnIndex());
        }
        if (resolved_name.empty())
          resolved_name = ResolveColumnName(full_ir, attr->GetTableIndex(),
                                            attr->GetColumnIndex());
        if (resolved_name.empty())
          resolved_name = attr->GetColumnName();
        required_attrs.push_back(
            std::make_unique<ir_sql_converter::SimplestAttr>(
                attr->GetType(), attr->GetTableIndex(),
                attr->GetColumnIndex(), resolved_name));
      }
    }
  };

  // (a) Top-level target_list attrs that come from subquery tables
  for (const auto &attr : full_ir->target_list) {
    AddIfSubqueryAttr(attr.get());
  }

  // (b) AGGR/ORDER node attrs that reference subquery tables
  std::function<void(const ir_sql_converter::AQPStmt *)> CollectPlanAttrs;
  CollectPlanAttrs = [&](const ir_sql_converter::AQPStmt *node) {
    if (!node)
      return;
    if (node->GetNodeType() ==
        ir_sql_converter::SimplestNodeType::AggregateNode) {
      auto *agg =
          dynamic_cast<const ir_sql_converter::SimplestAggregate *>(node);
      if (agg) {
        for (const auto &fn_pair : agg->agg_fns) {
          AddIfSubqueryAttr(fn_pair.first.get());
        }
        for (const auto &grp : agg->groups) {
          AddIfSubqueryAttr(grp.get());
        }
      }
    }
    if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::OrderNode) {
      auto *order =
          dynamic_cast<const ir_sql_converter::SimplestOrderBy *>(node);
      if (order) {
        for (const auto &ord : order->orders) {
          AddIfSubqueryAttr(ord.attr.get());
        }
      }
    }
    for (const auto &child : node->children) {
      CollectPlanAttrs(child.get());
    }
  };
  CollectPlanAttrs(full_ir);

  // (c) Cross-boundary join conditions: one attr in subquery, other outside
  std::function<void(const ir_sql_converter::AQPStmt *)> CollectCrossBoundary;
  CollectCrossBoundary = [&](const ir_sql_converter::AQPStmt *node) {
    if (!node)
      return;
    if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
      auto *join = dynamic_cast<const ir_sql_converter::SimplestJoin *>(node);
      if (join) {
        for (const auto &cond : join->join_conditions) {
          unsigned int left_table = cond->left_attr->GetTableIndex();
          unsigned int right_table = cond->right_attr->GetTableIndex();
          bool left_in = subquery_tables.count(left_table) > 0;
          bool right_in = subquery_tables.count(right_table) > 0;
          if (left_in && !right_in) {
            AddIfSubqueryAttr(cond->left_attr.get());
          }
          if (right_in && !left_in) {
            AddIfSubqueryAttr(cond->right_attr.get());
          }
        }
      }
    }
    for (const auto &child : node->children) {
      CollectCrossBoundary(child.get());
    }
  };
  CollectCrossBoundary(full_ir);

  // (d) Filter predicates (qual_vec) referencing subquery tables
  std::function<void(const ir_sql_converter::AQPStmt *)> CollectFilterAttrs;
  CollectFilterAttrs = [&](const ir_sql_converter::AQPStmt *node) {
    if (!node)
      return;
    if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::FilterNode) {
      for (const auto &qual : node->qual_vec) {
        ForEachAttrInExpr(qual.get(), [&](ir_sql_converter::SimplestAttr *a) {
          AddIfSubqueryAttr(a);
        });
      }
    }
    for (const auto &child : node->children) {
      CollectFilterAttrs(child.get());
    }
  };
  CollectFilterAttrs(full_ir);

  return required_attrs;
}

ir_sql_converter::AQPStmt *TopDownSplitter::WrapInProjection(
    ir_sql_converter::AQPStmt *remaining_ir,
    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
        required_attrs) {

  if (!remaining_ir || !found_split_node_) {
    std::cerr << "[TopDownSplitter::WrapInProjection] Error: null input"
              << std::endl;
    return nullptr;
  }

  if (required_attrs.empty()) {
    std::cerr
        << "[TopDownSplitter::WrapInProjection] Warning: no required attrs, "
           "skipping projection wrap"
        << std::endl;
    return nullptr;
  }

  // Find the parent of found_split_node_ and replace the child with a
  // Projection that wraps it.
  std::function<bool(ir_sql_converter::AQPStmt *)> FindAndWrap;
  FindAndWrap = [&](ir_sql_converter::AQPStmt *node) -> bool {
    if (!node)
      return false;
    for (size_t i = 0; i < node->children.size(); i++) {
      if (node->children[i].get() == found_split_node_) {
        // Extract the split node from its parent
        auto split_node = std::move(node->children[i]);

        // Build projection target list (clone required_attrs)
        std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> proj_tgt;
        for (const auto &attr : required_attrs) {
          proj_tgt.push_back(
              std::make_unique<ir_sql_converter::SimplestAttr>(*attr));
        }

        // Create the projection wrapping the original split node
        std::vector<std::unique_ptr<ir_sql_converter::AQPStmt>> proj_children;
        proj_children.push_back(std::move(split_node));
        auto proj_base = std::make_unique<ir_sql_converter::AQPStmt>(
            std::move(proj_children), std::move(proj_tgt),
            ir_sql_converter::SimplestNodeType::ProjectionNode);
        auto projection =
            std::make_unique<ir_sql_converter::SimplestProjection>(
                std::move(proj_base), 0);

        // Put the projection back at the same child slot
        node->children[i] = std::move(projection);

        // Update found_split_node_ to the new projection
        found_split_node_ = node->children[i].get();
        return true;
      }
      if (FindAndWrap(node->children[i].get())) {
        return true;
      }
    }
    return false;
  };

  if (!FindAndWrap(remaining_ir)) {
    std::cerr << "[TopDownSplitter::WrapInProjection] Warning: could not find "
                 "split node in tree; skipping projection wrap"
              << std::endl;
    return nullptr;
  }

#ifndef NDEBUG
  std::cout << "[TopDownSplitter::WrapInProjection] Wrapped split node in "
               "Projection with "
            << required_attrs.size() << " column(s)" << std::endl;
#endif
  return found_split_node_;
}

} // namespace middleware
