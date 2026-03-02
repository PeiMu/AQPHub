/*
 * Implementation of FK-based split strategies
 * Implements PostgreSQL's List2Graph algorithm for query splitting
 */

#include "split/fk_based_splitter.h"
#include "split/ir_utils.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>

namespace middleware {

// ===== FKBasedSplitter Base Class =====

void FKBasedSplitter::Preprocess(
    std::unique_ptr<ir_sql_converter::SimplestStmt> &ir) {

#ifndef NDEBUG
  std::cout << "[" << GetStrategyName() << "] Preprocessing IR" << std::endl;
#endif

  // Step 1: Collect tables from IR
  table_index_to_name_.clear();
  CollectTableNames(ir.get());

#ifndef NDEBUG
  std::cout << "[" << GetStrategyName() << "] Found "
            << table_index_to_name_.size() << " table(s)" << std::endl;
#endif

  // Build reverse mapping and track max table index
  max_table_index_ = 0;
  for (const auto &[idx, name] : table_index_to_name_) {
    if (idx > max_table_index_) {
      max_table_index_ = idx;
    }
#ifndef NDEBUG
    std::cout << "  Table " << idx << ": " << name << std::endl;
#endif
  }

  // Step 2: Update statistics for PostgreSQL (needed for accurate cost
  // estimates) DuckDB maintains statistics automatically, so this is only
  // needed for PostgreSQL
  if (enable_analyze_) {
    for (const auto &[idx, name] : table_index_to_name_) {
      try {
        auto analyze_str =
            engine_ == BackendEngine::MARIADB ? "ANALYZE TABLE " : "ANALYZE ";
        adapter_->ExecuteSQL(analyze_str + name);
      } catch (const std::exception &e) {
        // ANALYZE failure is not fatal, just log it
#ifndef NDEBUG
        std::cerr << "[" << GetStrategyName() << "] Warning: ANALYZE " << name
                  << " failed: " << e.what() << std::endl;
#endif
      }
    }
#ifndef NDEBUG
    std::cout << "[" << GetStrategyName()
              << "] Updated statistics for query tables" << std::endl;
#endif
  }

  // Step 3: Extract foreign keys
  std::set<std::string> table_names;
  for (const auto &[idx, name] : table_index_to_name_) {
    table_names.insert(name);
  }

  fk_graph_ = fk_extractor_.ExtractForTables(table_names);

#ifndef NDEBUG
  fk_graph_.Print();
#endif

  // Step 4: Mark entity vs relationship tables
  // Referenced tables (pk_table in FK) = entities
  // Referencing tables (fk_table in FK) = relationships
  is_relationship_ = MarkEntityRelationship(fk_graph_, table_index_to_name_);

#ifndef NDEBUG
  std::cout << "[" << GetStrategyName()
            << "] Entity/Relationship classification:" << std::endl;
  for (const auto &[idx, name] : table_index_to_name_) {
    std::cout << "  " << name << ": "
              << (is_relationship_[idx] ? "RELATIONSHIP" : "ENTITY")
              << std::endl;
  }
#endif

  // Step 5: Build join graph (implements PostgreSQL's List2Graph)
  BuildJoinGraph(ir.get());

  // Reset state
  split_iteration_ = 0;
  executed_tables_.clear();
  explain_cache_.clear();

#ifndef NDEBUG
  std::cout << "[" << GetStrategyName() << "] Preprocessing complete"
            << std::endl;
#endif
}

bool FKBasedSplitter::IsFKJoin(unsigned int table1, unsigned int table2) const {
  auto it1 = table_index_to_name_.find(table1);
  auto it2 = table_index_to_name_.find(table2);

  if (it1 == table_index_to_name_.end() || it2 == table_index_to_name_.end()) {
    return false;
  }

  return fk_graph_.HasDirectFK(it1->second, it2->second) ||
         fk_graph_.HasDirectFK(it2->second, it1->second);
}

std::vector<std::pair<unsigned int, unsigned int>>
FKBasedSplitter::RemoveRedundantJoins(
    const std::vector<std::pair<unsigned int, unsigned int>> &joins) const {
  // Implements PostgreSQL's rRj (removeRedundantJoin) function
  // Removes joins between two relationship tables (FK-FK joins)
  // These are considered redundant because they don't connect entities

  std::vector<std::pair<unsigned int, unsigned int>> filtered_joins;

  for (const auto &[t1, t2] : joins) {
    // Check if both tables are relationship tables
    bool t1_is_relationship =
        (t1 < is_relationship_.size()) && is_relationship_[t1];
    bool t2_is_relationship =
        (t2 < is_relationship_.size()) && is_relationship_[t2];

    if (t1_is_relationship && t2_is_relationship) {
      // Both are relationship tables - this is a redundant FK-FK join
#ifndef NDEBUG
      std::cout << "[" << GetStrategyName()
                << "] Removing redundant FK-FK join: " << t1 << " ("
                << table_index_to_name_.at(t1) << ") <-> " << t2 << " ("
                << table_index_to_name_.at(t2) << ")" << std::endl;
#endif
      continue;
    }

    // Keep this join
    filtered_joins.push_back({t1, t2});
  }

#ifndef NDEBUG
  std::cout << "[" << GetStrategyName() << "] Removed "
            << (joins.size() - filtered_joins.size())
            << " redundant FK-FK join(s)" << std::endl;
#endif

  return filtered_joins;
}

void FKBasedSplitter::BuildJoinGraph(const ir_sql_converter::SimplestStmt *ir) {
#ifndef NDEBUG
  std::cout << "[" << GetStrategyName() << "] Building join graph (List2Graph)"
            << std::endl;
#endif

  // Collect all join conditions from IR
  current_join_pairs_ = CollectJoinConditions(ir);

#ifndef NDEBUG
  std::cout << "[" << GetStrategyName() << "] Found "
            << current_join_pairs_.size() << " join condition(s) from IR"
            << std::endl;
#endif

  // For RelationshipCenter and EntityCenter, remove redundant FK-FK joins
  // (PostgreSQL rRj - only done once at initialization)
  if (strategy_ == SplitStrategy::RELATIONSHIP_CENTER ||
      strategy_ == SplitStrategy::ENTITY_CENTER) {
    current_join_pairs_ = RemoveRedundantJoins(current_join_pairs_);
  }

  // Build the graph from current state
  RebuildJoinGraph();
}

void FKBasedSplitter::RebuildJoinGraph() {
  // Determine graph size from current tables
  unsigned int max_idx = 0;
  for (const auto &[idx, name] : table_index_to_name_) {
    if (idx > max_idx)
      max_idx = idx;
  }
  int graph_size = static_cast<int>(max_idx) + 1;

  // Reset graph (PostgreSQL: memset(graph, false, ...))
  join_graph_ = JoinGraph(graph_size);

  // Separate FK and non-FK joins
  std::vector<std::pair<unsigned int, unsigned int>> fk_joins;
  std::vector<std::pair<unsigned int, unsigned int>> non_fk_joins;

  for (const auto &[t1, t2] : current_join_pairs_) {
    if (IsFKJoin(t1, t2)) {
      fk_joins.push_back({t1, t2});
    } else {
      non_fk_joins.push_back({t1, t2});
    }
  }

#ifndef NDEBUG
  std::cout << "[" << GetStrategyName()
            << "] RebuildJoinGraph: FK joins: " << fk_joins.size()
            << ", Non-FK joins: " << non_fk_joins.size() << std::endl;
#endif

  if (strategy_ == SplitStrategy::MIN_SUBQUERY) {
    // MinSubquery: symmetric (upper triangular) — all pairs
    for (const auto &[t1, t2] : current_join_pairs_) {
      unsigned int i = std::min(t1, t2);
      unsigned int j = std::max(t1, t2);
      join_graph_.SetEdge(i, j, true);
    }
  } else {
    // Step 1: FK joins with directed edges (PostgreSQL List2Graph lines
    // 1827-1858)
    for (const auto &[t1, t2] : fk_joins) {
      auto it1 = table_index_to_name_.find(t1);
      auto it2 = table_index_to_name_.find(t2);
      if (it1 == table_index_to_name_.end() ||
          it2 == table_index_to_name_.end())
        continue;

      unsigned int fk_owner_idx, pk_ref_idx;
      if (fk_graph_.HasDirectFK(it1->second, it2->second)) {
        fk_owner_idx = t1;
        pk_ref_idx = t2;
      } else {
        fk_owner_idx = t2;
        pk_ref_idx = t1;
      }

      if (strategy_ == SplitStrategy::RELATIONSHIP_CENTER) {
        // graph[con_relid][ref_relid] (PostgreSQL line 1853)
        join_graph_.SetEdge(fk_owner_idx, pk_ref_idx, true);
      } else {
        // graph[ref_relid][con_relid] (PostgreSQL line 1857)
        join_graph_.SetEdge(pk_ref_idx, fk_owner_idx, true);
      }
    }

    // Step 2: Non-FK joins with direction based on is_relationship
    // (PostgreSQL List2Graph lines 1861-1917)
    for (const auto &[t1, t2] : non_fk_joins) {
      bool t1_is_rel = (t1 < is_relationship_.size()) && is_relationship_[t1];
      bool t2_is_rel = (t2 < is_relationship_.size()) && is_relationship_[t2];

#ifndef NDEBUG
      std::cout << "  Non-FK join: " << t1 << " <-> " << t2
                << " (is_rel: " << t1_is_rel << "/" << t2_is_rel << ")"
                << std::endl;
#endif

      if (strategy_ == SplitStrategy::RELATIONSHIP_CENTER) {
        if (t1_is_rel && !t2_is_rel) {
          join_graph_.SetEdge(t1, t2, true);
        } else if (!t1_is_rel && t2_is_rel) {
          join_graph_.SetEdge(t2, t1, true);
        } else {
          join_graph_.SetEdge(t1, t2, true);
          join_graph_.SetEdge(t2, t1, true);
        }
      } else { // EntityCenter
        if (!t1_is_rel && t2_is_rel) {
          join_graph_.SetEdge(t1, t2, true);
        } else if (t1_is_rel && !t2_is_rel) {
          join_graph_.SetEdge(t2, t1, true);
        } else {
          join_graph_.SetEdge(t1, t2, true);
          join_graph_.SetEdge(t2, t1, true);
        }
      }
    }
  }

#ifndef NDEBUG
  join_graph_.Print();
#endif
}

std::vector<std::pair<unsigned int, unsigned int>>
FKBasedSplitter::CollectJoinConditions(
    const ir_sql_converter::SimplestStmt *ir) {
  std::vector<std::pair<unsigned int, unsigned int>> joins;

  if (!ir)
    return joins;

  // Check if this is a Join node
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *join = dynamic_cast<const ir_sql_converter::SimplestJoin *>(ir);
    if (join) {
      // Skip Mark joins: they represent IN-clause filters, not real relational
      // joins between base tables, so they must not appear in the join graph.
      if (join->GetSimplestJoinType() !=
          ir_sql_converter::SimplestJoinType::Mark) {
        for (const auto &cond : join->join_conditions) {
          unsigned int left_table = cond->left_attr->GetTableIndex();
          unsigned int right_table = cond->right_attr->GetTableIndex();
          joins.emplace_back(left_table, right_table);
        }
      }
    }
  }

  // Recursively collect from children
  for (const auto &child : ir->children) {
    auto child_joins = CollectJoinConditions(child.get());
    joins.insert(joins.end(), child_joins.begin(), child_joins.end());
  }

  return joins;
}

std::vector<bool> FKBasedSplitter::MarkEntityRelationship(
    const ForeignKeyGraph &fk_graph,
    const std::map<unsigned int, std::string> &tables) {

  // Size vector to accommodate max table index (not count of tables)
  // PostgreSQL rRj (line 263): palloc(length * sizeof(bool))
  // where length = rtable->length (max index)
  unsigned int max_idx = 0;
  for (const auto &[idx, name] : tables) {
    if (idx > max_idx)
      max_idx = idx;
  }
  std::vector<bool> is_relationship(max_idx + 1, true); // Default: relationship

  // Referenced tables are entities (PostgreSQL rRj line 270-271):
  //   is_relationship[ref_relid] = false
  for (const auto &[idx, table_name] : tables) {
    auto referencing = fk_graph.GetReferencingTables(table_name);
    if (!referencing.empty()) {
      // This table is referenced by others -> it's an entity
      is_relationship[idx] = false;
    }
  }

  return is_relationship;
}

std::set<unsigned int> FKBasedSplitter::CollectTableIndices(
    const ir_sql_converter::SimplestStmt *node) const {
  std::set<unsigned int> indices;

  if (!node)
    return indices;

  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = dynamic_cast<const ir_sql_converter::SimplestScan *>(node);
    if (scan) {
      indices.insert(scan->GetTableIndex());
    }
  }

  // ChunkNode is a constant value list used by Mark joins (IN clauses).
  // It is not a real relational table and must not be counted as one.

  for (const auto &child : node->children) {
    auto child_indices = CollectTableIndices(child.get());
    indices.insert(child_indices.begin(), child_indices.end());
  }

  return indices;
}

bool FKBasedSplitter::IsComplete(
    const ir_sql_converter::SimplestStmt *remaining_ir) {
  // Count edges only between non-executed tables
  int remaining_edges = 0;
  for (int i = 0; i < join_graph_.Size(); i++) {
    if (executed_tables_.count(i)) {
      continue; // Skip executed tables
    }
    for (int j = i + 1; j < join_graph_.Size(); j++) {
      if (executed_tables_.count(j)) {
        continue; // Skip executed tables
      }
      if (join_graph_.HasEdge(i, j) || join_graph_.HasEdge(j, i)) {
        remaining_edges++;
      }
    }
  }

  bool complete = false;

  if (strategy_ == SplitStrategy::MIN_SUBQUERY) {
    // MinSubquery: complete when only 1 edge (pair) remains
    complete = (remaining_edges == 1);
#ifndef NDEBUG
    std::cout << "[" << GetStrategyName()
              << "] IsComplete: " << (complete ? "YES" : "NO")
              << " (remaining edges between non-executed tables: "
              << remaining_edges << ")" << std::endl;
#endif
  } else {
    // EntityCenter/RelationshipCenter: complete when only 1 center remains
    // A center = any table with outgoing edges in the directed graph
    // (PostgreSQL does NOT filter centers by is_relationship)
    int remaining_centers = 0;
    for (int i = 0; i < join_graph_.Size(); i++) {
      if (executed_tables_.count(i)) {
        continue;
      }
      if (table_index_to_name_.find(i) == table_index_to_name_.end()) {
        continue;
      }

      // Check if this table has any outgoing edges
      bool has_outgoing = false;
      for (int j = 0; j < join_graph_.Size(); j++) {
        if (i == j || executed_tables_.count(j)) {
          continue;
        }
        if (join_graph_.HasEdge(i, j)) {
          has_outgoing = true;
          break;
        }
      }
      if (has_outgoing) {
        remaining_centers++;
      }
    }
    complete = (remaining_centers == 1);

#ifndef NDEBUG
    std::cout << "[" << GetStrategyName()
              << "] IsComplete check: remaining_centers=" << remaining_centers
              << std::endl;
#endif
  }

  return complete;
}

std::string
FKBasedSplitter::GenerateSQLForCluster(const std::vector<int> &cluster,
                                       ir_sql_converter::SimplestStmt *ir) {
  if (cluster.size() < 2 || !ir) {
    return "";
  }

  // Build a simple SQL query for the cluster tables
  // Format: SELECT * FROM t1, t2, ... WHERE join_conditions
  std::string sql = "SELECT * FROM ";

  // Add table names with aliases to handle duplicate table names
  // Convention matches ir_to_sql.cpp: chunk_name AS table_name_index
  bool first = true;
  for (int idx : cluster) {
    auto it = table_index_to_name_.find(static_cast<unsigned int>(idx));
    if (it == table_index_to_name_.end())
      continue;

    if (!first)
      sql += ", ";
    sql += it->second + " AS " + it->second + "_" + std::to_string(idx);
    first = false;
  }

  // Collect join conditions for tables in the cluster using actual column names
  std::set<unsigned int> cluster_tables;
  for (int idx : cluster) {
    cluster_tables.insert(static_cast<unsigned int>(idx));
  }

  // Get actual join conditions from IR (with real column names)
  auto join_conds = ir_utils::CollectJoinConditions(ir, cluster_tables);
  std::vector<std::string> where_clauses;

  for (const auto &cond : join_conds) {
    unsigned int left_table = cond->left_attr->GetTableIndex();
    unsigned int right_table = cond->right_attr->GetTableIndex();

    auto it1 = table_index_to_name_.find(left_table);
    auto it2 = table_index_to_name_.find(right_table);
    if (it1 != table_index_to_name_.end() &&
        it2 != table_index_to_name_.end()) {
      where_clauses.push_back(it1->second + "_" + std::to_string(left_table) +
                              "." + cond->left_attr->GetColumnName() + " = " +
                              it2->second + "_" + std::to_string(right_table) +
                              "." + cond->right_attr->GetColumnName());
    }
  }

  if (!where_clauses.empty()) {
    sql += " WHERE ";
    first = true;
    for (const auto &clause : where_clauses) {
      if (!first)
        sql += " AND ";
      sql += clause;
      first = false;
    }
  }

  return sql;
}

std::pair<double, double>
FKBasedSplitter::GetClusterCost(const std::vector<int> &cluster,
                                ir_sql_converter::SimplestStmt *ir) {
  std::string sql = GenerateSQLForCluster(cluster, ir);
  if (sql.empty()) {
    return {std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
  }

  // Check cache (safe: non-temp clusters produce same SQL with same stats)
  auto it = explain_cache_.find(sql);
  if (it != explain_cache_.end()) {
    return it->second;
  }

  auto result = adapter_->GetEstimatedCost(sql);
  explain_cache_[sql] = result;
  return result;
}

void FKBasedSplitter::BatchEvaluateClusterCosts(
    const std::vector<std::vector<int>> &clusters,
    ir_sql_converter::SimplestStmt *ir,
    std::vector<std::pair<double, double>> &results) {
  results.resize(clusters.size());

  // Phase 1: Generate SQL for each cluster, check cache
  std::vector<std::string> all_sqls(clusters.size());
  // Indices of clusters that need EXPLAIN (cache miss)
  std::vector<size_t> uncached_indices;
  std::vector<std::string> uncached_sqls;

  for (size_t i = 0; i < clusters.size(); i++) {
    std::string sql = GenerateSQLForCluster(clusters[i], ir);
    all_sqls[i] = sql;

    if (sql.empty()) {
      results[i] = {std::numeric_limits<double>::max(),
                    std::numeric_limits<double>::max()};
      continue;
    }

    auto it = explain_cache_.find(sql);
    if (it != explain_cache_.end()) {
      results[i] = it->second;
    } else {
      uncached_indices.push_back(i);
      uncached_sqls.push_back(sql);
    }
  }

  if (uncached_sqls.empty()) {
    return;
  }

  // Phase 2: Batch EXPLAIN all cache misses in one round-trip
  auto batch_results = adapter_->BatchGetEstimatedCosts(uncached_sqls);

  // Phase 3: Populate cache and fill results
  for (size_t k = 0; k < uncached_indices.size(); k++) {
    size_t orig_idx = uncached_indices[k];
    explain_cache_[all_sqls[orig_idx]] = batch_results[k];
    results[orig_idx] = batch_results[k];
  }
}

bool FKBasedSplitter::NodeContainsAnyTable(
    ir_sql_converter::SimplestStmt *node,
    const std::set<unsigned int> &target_tables) {
  if (!node)
    return false;

  // Check if this is a Scan node with a target table
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = dynamic_cast<ir_sql_converter::SimplestScan *>(node);
    if (scan && target_tables.count(scan->GetTableIndex())) {
      return true;
    }
  }

  // Check if this is a Chunk node (temp table)
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ChunkNode) {
    auto *chunk = dynamic_cast<ir_sql_converter::SimplestChunk *>(node);
    if (chunk && target_tables.count(chunk->GetTableIndex())) {
      return true;
    }
  }

  // Recursively check children
  for (const auto &child : node->children) {
    if (NodeContainsAnyTable(child.get(), target_tables)) {
      return true;
    }
  }

  return false;
}

bool FKBasedSplitter::NodeContainsExactlyTables(
    ir_sql_converter::SimplestStmt *node,
    const std::set<unsigned int> &target_tables) {
  if (!node)
    return false;

  // Collect all tables in this node
  auto node_tables = CollectTableIndices(node);

  // Check if the sets are equal
  return node_tables == target_tables;
}

ir_sql_converter::SimplestStmt *FKBasedSplitter::FindSubIRForCluster(
    ir_sql_converter::SimplestStmt *ir,
    const std::set<unsigned int> &cluster_tables) {
  if (!ir || cluster_tables.empty()) {
    return nullptr;
  }

  // Strategy: Find the smallest subtree that contains exactly the cluster
  // tables This is the "lowest common ancestor" that spans all cluster tables

  // First, check if current node contains exactly the cluster tables
  auto node_tables = CollectTableIndices(ir);
  if (node_tables == cluster_tables) {
#ifndef NDEBUG
    std::cout << "[FindSubIRForCluster] Found exact match at node type "
              << static_cast<int>(ir->GetNodeType()) << std::endl;
#endif
    return ir;
  }

  // If current node has more tables than cluster, search children
  // to find a smaller subtree that contains exactly the cluster
  for (auto &child : ir->children) {
    if (!child)
      continue;

    auto child_tables = CollectTableIndices(child.get());

    // If child contains exactly the cluster tables, return it
    if (child_tables == cluster_tables) {
#ifndef NDEBUG
      std::cout << "[FindSubIRForCluster] Found exact match in child"
                << std::endl;
#endif
      return child.get();
    }

    // If child contains all cluster tables (but maybe more), recurse
    bool contains_all = true;
    for (unsigned int t : cluster_tables) {
      if (child_tables.find(t) == child_tables.end()) {
        contains_all = false;
        break;
      }
    }

    if (contains_all) {
      auto result = FindSubIRForCluster(child.get(), cluster_tables);
      if (result) {
        return result;
      }
    }
  }

  // If we're a Join node and our children together form the cluster,
  // then this Join node is the subplan
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    // Check if this join's tables match the cluster
    if (node_tables == cluster_tables ||
        std::includes(node_tables.begin(), node_tables.end(),
                      cluster_tables.begin(), cluster_tables.end())) {
      // This join contains the cluster - it might be the best we can do
      // if no exact match was found in children
#ifndef NDEBUG
      std::cout << "[FindSubIRForCluster] Using Join node that contains cluster"
                << std::endl;
#endif
      return ir;
    }
  }

  return nullptr;
}

std::vector<ir_sql_converter::SimplestScan *>
FKBasedSplitter::CollectScansForTables(ir_sql_converter::SimplestStmt *ir,
                                       const std::set<unsigned int> &tables) {
  std::vector<ir_sql_converter::SimplestScan *> scans;

  if (!ir)
    return scans;

  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = dynamic_cast<ir_sql_converter::SimplestScan *>(ir);
    if (scan && tables.count(scan->GetTableIndex())) {
      scans.push_back(scan);
    }
  }

  for (auto &child : ir->children) {
    auto child_scans = CollectScansForTables(child.get(), tables);
    scans.insert(scans.end(), child_scans.begin(), child_scans.end());
  }

  return scans;
}

// Helper: Collect attributes from cluster tables that are needed for join
// conditions with tables OUTSIDE the cluster (for the remaining plan) NOTE:
// Only considers joins that still exist in join_graph_ (after
// RemoveRedundantJoins)
std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
FKBasedSplitter::CollectExternalJoinAttrs(
    ir_sql_converter::SimplestStmt *ir,
    const std::set<unsigned int> &cluster_tables) {
  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> attrs;
  std::set<std::pair<unsigned int, unsigned int>> seen; // (table_idx, col_idx)

  if (!ir)
    return attrs;

  // Check join conditions in Join nodes
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *join = dynamic_cast<ir_sql_converter::SimplestJoin *>(ir);
    if (join) {
      for (const auto &cond : join->join_conditions) {
        unsigned int left_table = cond->left_attr->GetTableIndex();
        unsigned int right_table = cond->right_attr->GetTableIndex();

        // NOTE: Do NOT filter by join_graph_ here. Redundant FK-FK joins
        // were removed from join_graph_ for graph partitioning only.
        // The join conditions still exist in the IR and must be satisfied.
        // All external join attrs must be in the cluster output so that
        // cross-boundary conditions can be rewritten to the temp table.

        // If one table is in cluster and the other is NOT, we need to include
        // the cluster table's attr in our output (for the remaining plan's
        // join)
        bool left_in_cluster = cluster_tables.count(left_table) > 0;
        bool right_in_cluster = cluster_tables.count(right_table) > 0;

        if (left_in_cluster && !right_in_cluster) {
          auto key =
              std::make_pair(left_table, cond->left_attr->GetColumnIndex());
          if (seen.find(key) == seen.end()) {
            seen.insert(key);
            attrs.push_back(std::make_unique<ir_sql_converter::SimplestAttr>(
                *cond->left_attr));
          }
        }
        if (right_in_cluster && !left_in_cluster) {
          auto key =
              std::make_pair(right_table, cond->right_attr->GetColumnIndex());
          if (seen.find(key) == seen.end()) {
            seen.insert(key);
            attrs.push_back(std::make_unique<ir_sql_converter::SimplestAttr>(
                *cond->right_attr));
          }
        }
      }
    }
  }

  // Recurse into children
  for (auto &child : ir->children) {
    auto child_attrs = CollectExternalJoinAttrs(child.get(), cluster_tables);
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

void FKBasedSplitter::MoveValidJoins(
    std::unique_ptr<ir_sql_converter::SimplestStmt> &node,
    const std::set<unsigned int> &remaining_tables,
    const std::set<unsigned int> &executed_table_indices,
    std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
        &internal_joins,
    std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
        &cross_boundary_joins) {
  if (!node)
    return;

  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *join = dynamic_cast<ir_sql_converter::SimplestJoin *>(node.get());
    if (join) {
      for (auto &cond : join->join_conditions) {
        if (!cond)
          continue;

        unsigned int left_table = cond->left_attr->GetTableIndex();
        unsigned int right_table = cond->right_attr->GetTableIndex();

        // Skip FK-FK joins not in current_join_pairs_
        if (strategy_ == SplitStrategy::RELATIONSHIP_CENTER ||
            strategy_ == SplitStrategy::ENTITY_CENTER) {
          bool pair_found = false;
          for (const auto &[p1, p2] : current_join_pairs_) {
            if ((p1 == left_table && p2 == right_table) ||
                (p1 == right_table && p2 == left_table)) {
              pair_found = true;
              break;
            }
          }
          if (!pair_found)
            continue;
        }

        bool left_in_remaining = remaining_tables.count(left_table) > 0;
        bool right_in_remaining = remaining_tables.count(right_table) > 0;
        bool left_in_executed = executed_table_indices.count(left_table) > 0;
        bool right_in_executed = executed_table_indices.count(right_table) > 0;

        if (left_in_remaining && right_in_remaining) {
          internal_joins.push_back(std::move(cond));
        } else if ((left_in_remaining && right_in_executed) ||
                   (left_in_executed && right_in_remaining)) {
          cross_boundary_joins.push_back(std::move(cond));
        }
      }
    }
  }

  for (auto &child : node->children) {
    MoveValidJoins(child, remaining_tables, executed_table_indices,
                   internal_joins, cross_boundary_joins);
  }
}

void FKBasedSplitter::PrepareForNextIteration(
    const std::set<unsigned int> &executed_table_indices,
    unsigned int temp_table_index, const std::string &temp_table_name) {

#ifndef NDEBUG
  std::cout << "[" << GetStrategyName()
            << "] PrepareForNextIteration: temp=" << temp_table_name
            << " (index " << temp_table_index << ")" << std::endl;
#endif

  // Step 1: Update is_relationship_ for temp table
  // (PostgreSQL Prepare4Next lines 1530-1533)
  if (temp_table_index >= is_relationship_.size()) {
    is_relationship_.resize(temp_table_index + 1, false);
  }
  if (strategy_ == SplitStrategy::RELATIONSHIP_CENTER) {
    is_relationship_[temp_table_index] = false; // temp = entity
  } else if (strategy_ == SplitStrategy::ENTITY_CENTER) {
    is_relationship_[temp_table_index] = true; // temp = relationship
  }

  // Step 2: Update fk_graph_ (PostgreSQL Prepare4Next lines 1547-1569)
  std::set<std::string> executed_table_names;
  for (unsigned int idx : executed_table_indices) {
    auto it = table_index_to_name_.find(idx);
    if (it != table_index_to_name_.end()) {
      executed_table_names.insert(it->second);
    }
  }
  fk_graph_.UpdateForTempTable(executed_table_names, temp_table_name);

  // Step 3: Update current_join_pairs_ (redirect to temp, remove both-executed)
  // (PostgreSQL: re-collects Joinlist from updated query, lines 1162-1168)
  std::vector<std::pair<unsigned int, unsigned int>> updated_pairs;
  for (const auto &[t1, t2] : current_join_pairs_) {
    bool t1_executed = executed_table_indices.count(t1) > 0;
    bool t2_executed = executed_table_indices.count(t2) > 0;

    if (t1_executed && t2_executed) {
      continue; // Both executed: remove
    }

    unsigned int new_t1 = t1_executed ? temp_table_index : t1;
    unsigned int new_t2 = t2_executed ? temp_table_index : t2;
    updated_pairs.push_back({new_t1, new_t2});
  }
  current_join_pairs_ = std::move(updated_pairs);

  // Step 4: Remove executed tables and add temp table to table_index_to_name_
  // (PostgreSQL Prepare4Next compacts rtable, removing executed entries)
  for (unsigned int idx : executed_table_indices) {
    table_index_to_name_.erase(idx);
  }
  table_index_to_name_[temp_table_index] = temp_table_name;

  // Step 5: Rebuild join graph from scratch (PostgreSQL List2Graph, line 1171)
  RebuildJoinGraph();
}

std::unique_ptr<ir_sql_converter::SimplestStmt>
FKBasedSplitter::BuildSubIRForCluster(
    ir_sql_converter::SimplestStmt *ir,
    const std::set<unsigned int> &cluster_tables) {

#ifndef NDEBUG
  std::cout << "[BuildSubIRForCluster] Building sub-IR for "
            << cluster_tables.size() << " tables" << std::endl;
#endif

  if (!ir || cluster_tables.empty()) {
    return nullptr;
  }

  // Step 1: Collect Scan nodes for tables in the cluster
  auto scans = CollectScansForTables(ir, cluster_tables);
#ifndef NDEBUG
  std::cout << "[BuildSubIRForCluster] Found " << scans.size()
            << " scan node(s)" << std::endl;
#endif

  if (scans.empty()) {
    std::cerr << "[BuildSubIRForCluster] No scans found for cluster tables"
              << std::endl;
    return nullptr;
  }

  // Step 2: Collect join conditions between cluster tables (internal joins)
  auto join_conditions = ir_utils::CollectJoinConditions(ir, cluster_tables);
  // Filter out redundant FK-FK joins: keep only conditions whose table pair
  // exists in current_join_pairs_ (FK-FK pairs already removed by rRj)
  if (strategy_ == SplitStrategy::RELATIONSHIP_CENTER ||
      strategy_ == SplitStrategy::ENTITY_CENTER) {
    size_t before_size = join_conditions.size();
    auto it = std::remove_if(
        join_conditions.begin(), join_conditions.end(),
        [this](const std::unique_ptr<ir_sql_converter::SimplestVarComparison>
                   &cond) {
          unsigned int lt = cond->left_attr->GetTableIndex();
          unsigned int rt = cond->right_attr->GetTableIndex();
          for (const auto &[p1, p2] : current_join_pairs_) {
            if ((p1 == lt && p2 == rt) || (p1 == rt && p2 == lt)) {
              return false; // keep — pair is in current_join_pairs_
            }
          }
          return true; // remove — FK-FK pair not in current_join_pairs_
        });
    join_conditions.erase(it, join_conditions.end());
#ifndef NDEBUG
    if (join_conditions.size() < before_size) {
      std::cout << "[BuildSubIRForCluster] Filtered "
                << (before_size - join_conditions.size())
                << " redundant FK-FK join condition(s)" << std::endl;
    }
#endif
  }

#ifndef NDEBUG
  std::cout << "[BuildSubIRForCluster] Join conditions after filtering:"
            << std::endl;
  for (const auto &cond : join_conditions) {
    unsigned int lt = cond->left_attr->GetTableIndex();
    unsigned int rt = cond->right_attr->GetTableIndex();
    std::cout << "  " << lt << "." << cond->left_attr->GetColumnName() << " = "
              << rt << "." << cond->right_attr->GetColumnName() << std::endl;
  }
#endif

  // Step 3: Collect filter conditions that involve only cluster tables
  // Uses AND-splitting: extracts conjuncts that involve cluster tables
  auto cluster_filters = ir_utils::CollectFilterConditions(ir, cluster_tables);
#ifndef NDEBUG
  std::cout << "[BuildSubIRForCluster] Found " << cluster_filters.size()
            << " filter condition(s) for cluster" << std::endl;
#endif

  // Step 4: Collect required output attributes
  // - Attributes from original target_list that belong to cluster tables
  // - Attributes needed for joins with tables OUTSIDE the cluster
  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> required_attrs;
  std::set<std::pair<unsigned int, unsigned int>> seen_attrs;

  // Helper to add a cluster attr to required_attrs if not already seen
  auto AddIfClusterAttr = [&](const ir_sql_converter::SimplestAttr *attr) {
    if (cluster_tables.count(attr->GetTableIndex())) {
      auto key = std::make_pair(attr->GetTableIndex(), attr->GetColumnIndex());
      if (seen_attrs.find(key) == seen_attrs.end()) {
        seen_attrs.insert(key);
        required_attrs.push_back(
            std::make_unique<ir_sql_converter::SimplestAttr>(*attr));
      }
    }
  };

  // 4a: From original target_list
  for (const auto &attr : ir->target_list) {
    AddIfClusterAttr(attr.get());
  }

  // 4a2: From Aggregate/OrderBy nodes — these reference original table
  // columns (e.g. MIN(aka_title.title)) that are not in the top-level
  // target_list when DuckDB's plan separates aggregation into its own node
  std::function<void(const ir_sql_converter::SimplestStmt *)> CollectPlanAttrs;
  CollectPlanAttrs = [&](const ir_sql_converter::SimplestStmt *node) {
    if (!node)
      return;
    if (node->GetNodeType() ==
        ir_sql_converter::SimplestNodeType::AggregateNode) {
      auto *agg =
          dynamic_cast<const ir_sql_converter::SimplestAggregate *>(node);
      if (agg) {
        for (const auto &fn_pair : agg->agg_fns) {
          AddIfClusterAttr(fn_pair.first.get());
        }
        for (const auto &grp : agg->groups) {
          AddIfClusterAttr(grp.get());
        }
      }
    }
    if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::OrderNode) {
      auto *order =
          dynamic_cast<const ir_sql_converter::SimplestOrderBy *>(node);
      if (order) {
        for (const auto &ord : order->orders) {
          AddIfClusterAttr(ord.attr.get());
        }
      }
    }
    for (const auto &child : node->children) {
      CollectPlanAttrs(child.get());
    }
  };
  CollectPlanAttrs(ir);

  // 4b: From external join conditions (attrs needed for remaining plan)
  auto external_attrs = CollectExternalJoinAttrs(ir, cluster_tables);
  for (auto &attr : external_attrs) {
    auto key = std::make_pair(attr->GetTableIndex(), attr->GetColumnIndex());
    if (seen_attrs.find(key) == seen_attrs.end()) {
      seen_attrs.insert(key);
      required_attrs.push_back(std::move(attr));
    }
  }

  // Note: Filter attrs are NOT added to projection target list
  // Filter conditions are evaluated internally within the cluster,
  // so their attrs don't need to be projected out

#ifndef NDEBUG
  std::cout << "[BuildSubIRForCluster] Required output attributes: "
            << required_attrs.size() << std::endl;
#endif

  // Step 5: Build the sub-IR tree (left-deep)
  // Clone scan nodes
  std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> scan_nodes;
  for (auto *scan : scans) {
    // Clone the scan node - build manually since SimplestStmt has no copy
    // constructor
    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> empty_children;
    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
        cloned_target_list;
    for (const auto &attr : scan->target_list) {
      cloned_target_list.push_back(
          std::make_unique<ir_sql_converter::SimplestAttr>(*attr));
    }
    auto base_stmt = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(empty_children), std::move(cloned_target_list),
        ir_sql_converter::SimplestNodeType::ScanNode);

    auto cloned_scan = std::make_unique<ir_sql_converter::SimplestScan>(
        std::move(base_stmt), scan->GetTableIndex(), scan->GetTableName());
    scan_nodes.push_back(std::move(cloned_scan));
  }

  if (scan_nodes.size() == 1) {
    // Only one table - wrap in Projection and return
    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> proj_children;
    proj_children.push_back(std::move(scan_nodes[0]));

    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> proj_target;
    for (auto &attr : required_attrs) {
      proj_target.push_back(
          std::make_unique<ir_sql_converter::SimplestAttr>(*attr));
    }

    auto proj_base = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(proj_children), std::move(proj_target),
        ir_sql_converter::SimplestNodeType::ProjectionNode);

    // Use table index 0 for the projection (temporary result)
    return std::make_unique<ir_sql_converter::SimplestProjection>(
        std::move(proj_base), 0);
  }

  // Build left-deep tree using CrossProduct for intermediate joins
  std::unique_ptr<ir_sql_converter::SimplestStmt> result =
      std::move(scan_nodes[0]);

  for (size_t i = 1; i < scan_nodes.size(); i++) {
    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> join_children;
    join_children.push_back(std::move(result));
    join_children.push_back(std::move(scan_nodes[i]));

    // Create base SimplestStmt with empty target_list
    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> empty_attrs;
    auto base_stmt = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(join_children), std::move(empty_attrs),
        ir_sql_converter::SimplestNodeType::CrossProductNode);

    if (i < scan_nodes.size() - 1) {
      // Intermediate join: use CrossProduct (no conditions)
      result = std::make_unique<ir_sql_converter::SimplestCrossProduct>(
          std::move(base_stmt));
    } else {
      // Final join: use Join with all conditions
      base_stmt->ChangeNodeType(ir_sql_converter::SimplestNodeType::JoinNode);
      auto join_node = std::make_unique<ir_sql_converter::SimplestJoin>(
          std::move(base_stmt), ir_sql_converter::Inner);

      // Add all join conditions to the final join
      for (auto &cond : join_conditions) {
        join_node->join_conditions.push_back(std::move(cond));
      }
      result = std::move(join_node);
    }
  }

  if (!cluster_filters.empty()) {
    // Build Filter node
    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>>
        filter_children;
    filter_children.emplace_back(std::move(result));
    // todo: add target_list content
    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> target_list;
    // add qual vec
    std::vector<std::unique_ptr<ir_sql_converter::SimplestExpr>> qual_vec;
    for (auto &filter : cluster_filters) {
      if (filter) {
        qual_vec.push_back(std::move(filter));
      }
    }

    auto base_stmt = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(filter_children), std::move(target_list), std::move(qual_vec),
        ir_sql_converter::SimplestNodeType::FilterNode);

    result = std::make_unique<ir_sql_converter::SimplestFilter>(
        std::move(base_stmt));
  }

  // Step 6: Add Projection node on top with required attributes
  std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> proj_children;
  proj_children.push_back(std::move(result));

  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> proj_target;
  for (auto &attr : required_attrs) {
    proj_target.push_back(
        std::make_unique<ir_sql_converter::SimplestAttr>(*attr));
  }

  auto proj_base = std::make_unique<ir_sql_converter::SimplestStmt>(
      std::move(proj_children), std::move(proj_target),
      ir_sql_converter::SimplestNodeType::ProjectionNode);

  // Use table index 0 for the projection (will be replaced by temp table index)
  auto projection = std::make_unique<ir_sql_converter::SimplestProjection>(
      std::move(proj_base), 0);

#ifndef NDEBUG
  std::cout << "[BuildSubIRForCluster] Built sub-IR with Projection ("
            << projection->target_list.size() << " output columns) and "
            << cluster_filters.size() << " filter(s)" << std::endl;
#endif

  return projection;
}

// ===== MinSubquery Strategy =====

std::unique_ptr<SubqueryExtraction> MinSubquerySplitter::ExtractNextSubquery(
    ir_sql_converter::SimplestStmt *remaining_ir) {

  split_iteration_++;
#ifndef NDEBUG
  std::cout << "\n[MinSubquery] Iteration " << split_iteration_ << std::endl;
#endif

  // Find next pair of tables with lowest estimated cost
  double est_rows = 0;
  auto [table1, table2] = FindNextPair(remaining_ir, est_rows);

  if (table1 == -1 || table2 == -1) {
    std::cout << "[MinSubquery] No more pairs to join" << std::endl;
    return nullptr;
  }

#ifndef NDEBUG
  std::cout << "[MinSubquery] Selected pair: " << table1 << ", " << table2
            << " (" << table_index_to_name_[table1] << ", "
            << table_index_to_name_[table2] << ")" << std::endl;
#endif

  // Create extraction with these two tables
  std::set<unsigned int> table_indices = {static_cast<unsigned int>(table1),
                                          static_cast<unsigned int>(table2)};

  auto extraction = std::make_unique<SubqueryExtraction>(
      table_indices, "temp_" + std::to_string(split_iteration_));
  extraction->estimated_rows = est_rows;

  // Build a NEW sub-IR containing only the selected tables
  // This is equivalent to PostgreSQL's createQuery()
  auto built_sub_ir = BuildSubIRForCluster(remaining_ir, table_indices);

  if (built_sub_ir) {
    extraction->sub_ir = std::move(built_sub_ir);
#ifndef NDEBUG
    std::cout << "[MinSubquery] Built sub-IR for pair" << std::endl;
#endif
  }

  // Always find the node in original tree to replace (for UpdateRemainingIR)
  // This is needed even when we build a new sub_ir
  ir_sql_converter::SimplestStmt *found_node =
      FindSubIRForCluster(remaining_ir, table_indices);
  if (found_node) {
    extraction->pipeline_breaker_ptr = found_node;
#ifndef NDEBUG
    std::cout << "[MinSubquery] Found node to replace in remaining IR"
              << std::endl;
#endif
  } else {
    std::cerr << "[MinSubquery] Warning: Could not find node to replace"
              << std::endl;
  }

  // Track executed tables so they're excluded from future clusters
  for (unsigned int idx : table_indices) {
    executed_tables_.insert(idx);
  }

  return extraction;
}

std::pair<int, int>
MinSubquerySplitter::FindNextPair(ir_sql_converter::SimplestStmt *ir,
                                  double &estimated_rows) {
  // Phase 1: Collect all candidate pairs
  std::vector<std::vector<int>> candidate_clusters;
  std::vector<std::pair<int, int>> candidate_pairs;

#ifndef NDEBUG
  std::cout << "[MinSubquery] Evaluating candidate pairs:" << std::endl;
#endif

  for (int i = 0; i < join_graph_.Size(); i++) {
    if (executed_tables_.count(i)) {
      continue;
    }
    for (int j = i + 1; j < join_graph_.Size(); j++) {
      if (executed_tables_.count(j)) {
        continue;
      }
      if (!join_graph_.HasEdge(i, j)) {
        continue;
      }
      candidate_pairs.push_back({i, j});
      candidate_clusters.push_back({i, j});
    }
  }

  if (candidate_clusters.empty()) {
    return {-1, -1};
  }

  // Phase 2: Batch evaluate all cluster costs
  std::vector<std::pair<double, double>> costs;
  BatchEvaluateClusterCosts(candidate_clusters, ir, costs);

  // Phase 3: Select pair with minimum hybrid_row score = max(rows,1) * cost
  // Matches PostgreSQL query_split.c order_decision == hybrid_row.
  // Hard cap: skip candidates with rows > 10M (avoids catastrophic
  // intermediates).
  static constexpr double kRowCap = 10000000.0;
  std::pair<int, int> best_pair = {-1, -1};
  double best_cost = std::numeric_limits<double>::max();
  estimated_rows = 0;

  for (size_t k = 0; k < candidate_pairs.size(); k++) {
    auto [cost, rows] = costs[k];
    auto [i, j] = candidate_pairs[k];

#ifndef NDEBUG
    std::cout << "  Pair (" << i << ", " << j << ") ["
              << table_index_to_name_[i] << ", " << table_index_to_name_[j]
              << "]: cost=" << cost << ", rows=" << rows << std::endl;
#endif

    if (rows > kRowCap)
      continue;

    double score_new = std::max(rows, 1.0) * cost;
    double score_old = std::max(estimated_rows, 1.0) * best_cost;
    if (score_new < score_old) {
      best_cost = cost;
      best_pair = {i, j};
      estimated_rows = rows;
    }
  }

#ifndef NDEBUG
  if (best_pair.first != -1) {
    std::cout << "[MinSubquery] Best pair: (" << best_pair.first << ", "
              << best_pair.second << ") with cost=" << best_cost << std::endl;
  }
#endif

  return best_pair;
}

// ===== RelationshipCenter Strategy =====

std::unique_ptr<SubqueryExtraction>
RelationshipCenterSplitter::ExtractNextSubquery(
    ir_sql_converter::SimplestStmt *remaining_ir) {

  split_iteration_++;
#ifndef NDEBUG
  std::cout << "\n[RelationshipCenter] Iteration " << split_iteration_
            << std::endl;
#endif

  // Find relationship cluster with lowest estimated cost
  double est_rows = 0;
  auto cluster = FindRelationshipCluster(remaining_ir, est_rows);

  if (cluster.empty()) {
    std::cout << "[RelationshipCenter] No more relationship clusters"
              << std::endl;
    return nullptr;
  }

#ifndef NDEBUG
  std::cout << "[RelationshipCenter] Selected cluster: ";
  for (int idx : cluster) {
    std::cout << idx << "(" << table_index_to_name_[idx] << ") ";
  }
  std::cout << std::endl;
#endif

  // Create extraction
  std::set<unsigned int> table_indices;
  for (int idx : cluster) {
    table_indices.insert(static_cast<unsigned int>(idx));
  }

  auto extraction = std::make_unique<SubqueryExtraction>(
      table_indices, "temp_" + std::to_string(split_iteration_));
  extraction->estimated_rows = est_rows;

  // Build a NEW sub-IR containing only the cluster tables
  // This is equivalent to PostgreSQL's createQuery()
  auto built_sub_ir = BuildSubIRForCluster(remaining_ir, table_indices);

  if (built_sub_ir) {
    extraction->sub_ir = std::move(built_sub_ir);
#ifndef NDEBUG
    std::cout << "[RelationshipCenter] Built sub-IR for cluster" << std::endl;
#endif
  }

  // Always find the node in original tree to replace (for UpdateRemainingIR)
  ir_sql_converter::SimplestStmt *found_node =
      FindSubIRForCluster(remaining_ir, table_indices);
  if (found_node) {
    extraction->pipeline_breaker_ptr = found_node;
#ifndef NDEBUG
    std::cout << "[RelationshipCenter] Found node to replace in remaining IR"
              << std::endl;
#endif
  } else {
    std::cerr << "[RelationshipCenter] Warning: Could not find node to replace"
              << std::endl;
  }

  // Track executed tables so they're excluded from future clusters
  for (unsigned int idx : table_indices) {
    executed_tables_.insert(idx);
  }

  return extraction;
}

std::vector<int> RelationshipCenterSplitter::FindRelationshipCluster(
    ir_sql_converter::SimplestStmt *ir, double &estimated_rows) {
  // Phase 1: Collect all candidate clusters (center + outgoing neighbors)
  std::vector<std::vector<int>> candidate_clusters;

#ifndef NDEBUG
  std::cout << "[RelationshipCenter] Evaluating candidate clusters:"
            << std::endl;
#endif

  for (int i = 0; i < join_graph_.Size(); i++) {
    if (executed_tables_.count(i)) {
      continue;
    }
    if (table_index_to_name_.find(i) == table_index_to_name_.end()) {
      continue;
    }

    std::vector<int> cluster;
    cluster.push_back(i);

    for (int j = 0; j < join_graph_.Size(); j++) {
      if (i == j || executed_tables_.count(j)) {
        continue;
      }
      if (join_graph_.HasEdge(i, j)) {
        cluster.push_back(j);
      }
    }

    if (cluster.size() < 2) {
      continue;
    }

    candidate_clusters.push_back(std::move(cluster));
  }

  if (candidate_clusters.empty()) {
    return {};
  }

  // Phase 2: Batch evaluate all cluster costs
  std::vector<std::pair<double, double>> costs;
  BatchEvaluateClusterCosts(candidate_clusters, ir, costs);

  // Phase 3: Select cluster with minimum hybrid_row score = max(rows,1) * cost
  // Matches PostgreSQL query_split.c order_decision == hybrid_row.
  // Hard cap: skip candidates with rows > 10M (avoids catastrophic
  // intermediates).
  static constexpr double kRowCap = 10000000.0;
  std::vector<int> best_cluster;
  double best_cost = std::numeric_limits<double>::max();
  estimated_rows = 0;

  for (size_t idx = 0; idx < candidate_clusters.size(); idx++) {
    auto [cost, rows] = costs[idx];

#ifndef NDEBUG
    std::cout << "  Cluster center=" << candidate_clusters[idx][0] << " ("
              << table_index_to_name_[candidate_clusters[idx][0]]
              << "): tables=[";
    for (size_t k = 0; k < candidate_clusters[idx].size(); k++) {
      if (k > 0)
        std::cout << ", ";
      std::cout << table_index_to_name_[candidate_clusters[idx][k]];
    }
    std::cout << "] cost=" << cost << ", rows=" << rows << std::endl;
#endif

    if (rows > kRowCap)
      continue;

    double score_new = std::max(rows, 1.0) * cost;
    double score_old = std::max(estimated_rows, 1.0) * best_cost;
    if (score_new < score_old) {
      best_cost = cost;
      best_cluster = candidate_clusters[idx];
      estimated_rows = rows;
    }
  }

#ifndef NDEBUG
  if (!best_cluster.empty()) {
    std::cout << "[RelationshipCenter] Best cluster center=" << best_cluster[0]
              << " (" << table_index_to_name_[best_cluster[0]]
              << ") with cost=" << best_cost << std::endl;
  }
#endif

  return best_cluster;
}

// ===== EntityCenter Strategy =====

std::unique_ptr<SubqueryExtraction> EntityCenterSplitter::ExtractNextSubquery(
    ir_sql_converter::SimplestStmt *remaining_ir) {

  split_iteration_++;
#ifndef NDEBUG
  std::cout << "\n[EntityCenter] Iteration " << split_iteration_ << std::endl;
#endif

  // Find entity cluster with lowest estimated cost
  double est_rows = 0;
  auto cluster = FindEntityCluster(remaining_ir, est_rows);

  if (cluster.empty()) {
    std::cout << "[EntityCenter] No more entity clusters" << std::endl;
    return nullptr;
  }

#ifndef NDEBUG
  std::cout << "[EntityCenter] Selected cluster: ";
  for (int idx : cluster) {
    std::cout << idx << "(" << table_index_to_name_[idx] << ") ";
  }
  std::cout << std::endl;
#endif

  // Create extraction
  std::set<unsigned int> table_indices;
  for (int idx : cluster) {
    table_indices.insert(static_cast<unsigned int>(idx));
  }

  auto extraction = std::make_unique<SubqueryExtraction>(
      table_indices, "temp_" + std::to_string(split_iteration_));
  extraction->estimated_rows = est_rows;

  // Build a NEW sub-IR containing only the cluster tables
  // This is equivalent to PostgreSQL's createQuery()
  auto built_sub_ir = BuildSubIRForCluster(remaining_ir, table_indices);

  if (built_sub_ir) {
    extraction->sub_ir = std::move(built_sub_ir);
#ifndef NDEBUG
    std::cout << "[EntityCenter] Built sub-IR for cluster" << std::endl;
#endif
  }

  // Always find the node in original tree to replace (for UpdateRemainingIR)
  ir_sql_converter::SimplestStmt *found_node =
      FindSubIRForCluster(remaining_ir, table_indices);
  if (found_node) {
    extraction->pipeline_breaker_ptr = found_node;
#ifndef NDEBUG
    std::cout << "[EntityCenter] Found node to replace in remaining IR"
              << std::endl;
#endif
  } else {
    std::cerr << "[EntityCenter] Warning: Could not find node to replace"
              << std::endl;
  }

  // Track executed tables so they're excluded from future clusters
  for (unsigned int idx : table_indices) {
    executed_tables_.insert(idx);
  }

  return extraction;
}

std::vector<int>
EntityCenterSplitter::FindEntityCluster(ir_sql_converter::SimplestStmt *ir,
                                        double &estimated_rows) {
  // Phase 1: Collect all candidate clusters (center + outgoing neighbors)
  std::vector<std::vector<int>> candidate_clusters;

#ifndef NDEBUG
  std::cout << "[EntityCenter] Evaluating candidate clusters:" << std::endl;
#endif

  for (int i = 0; i < join_graph_.Size(); i++) {
    if (executed_tables_.count(i)) {
      continue;
    }
    if (table_index_to_name_.find(i) == table_index_to_name_.end()) {
      continue;
    }

    std::vector<int> cluster;
    cluster.push_back(i);

    for (int j = 0; j < join_graph_.Size(); j++) {
      if (i == j || executed_tables_.count(j)) {
        continue;
      }
      if (join_graph_.HasEdge(i, j)) {
        cluster.push_back(j);
      }
    }

    if (cluster.size() < 2) {
      continue;
    }

    candidate_clusters.push_back(std::move(cluster));
  }

  if (candidate_clusters.empty()) {
    return {};
  }

  // Phase 2: Batch evaluate all cluster costs
  std::vector<std::pair<double, double>> costs;
  BatchEvaluateClusterCosts(candidate_clusters, ir, costs);

  // Phase 3: Select cluster with minimum hybrid_row score = max(rows,1) * cost
  // Matches PostgreSQL query_split.c order_decision == hybrid_row.
  // Hard cap: skip candidates with rows > 10M (avoids catastrophic
  // intermediates).
  static constexpr double kRowCap = 10000000.0;
  std::vector<int> best_cluster;
  double best_cost = std::numeric_limits<double>::max();
  estimated_rows = 0;

  for (size_t idx = 0; idx < candidate_clusters.size(); idx++) {
    auto [cost, rows] = costs[idx];

#ifndef NDEBUG
    std::cout << "  Cluster center=" << candidate_clusters[idx][0] << " ("
              << table_index_to_name_[candidate_clusters[idx][0]]
              << "): tables=[";
    for (size_t k = 0; k < candidate_clusters[idx].size(); k++) {
      if (k > 0)
        std::cout << ", ";
      std::cout << table_index_to_name_[candidate_clusters[idx][k]];
    }
    std::cout << "] cost=" << cost << ", rows=" << rows << std::endl;
#endif

    if (rows > kRowCap)
      continue;

    double score_new = std::max(rows, 1.0) * cost;
    double score_old = std::max(estimated_rows, 1.0) * best_cost;
    if (score_new < score_old) {
      best_cost = cost;
      best_cluster = candidate_clusters[idx];
      estimated_rows = rows;
    }
  }

#ifndef NDEBUG
  if (!best_cluster.empty()) {
    std::cout << "[EntityCenter] Best cluster center=" << best_cluster[0]
              << " (" << table_index_to_name_[best_cluster[0]]
              << ") with cost=" << best_cost << std::endl;
  }
#endif

  return best_cluster;
}

// ===== UpdateRemainingIR (PostgreSQL-style rebuild) =====
std::unique_ptr<ir_sql_converter::SimplestStmt>
FKBasedSplitter::UpdateRemainingIR(
    std::unique_ptr<ir_sql_converter::SimplestStmt> remaining_ir,
    const std::set<unsigned int> &executed_table_indices,
    unsigned int temp_table_index, const std::string &temp_table_name,
    uint64_t temp_table_cardinality,
    const std::vector<std::pair<unsigned int, unsigned int>> &column_mappings,
    const std::vector<std::string> &column_names) {

#ifndef NDEBUG
  std::cout << "[FKBasedSplitter::UpdateRemainingIR] Rebuilding remaining IR "
               "with temp table: "
            << temp_table_name << " (index " << temp_table_index
            << ", cardinality " << temp_table_cardinality << ")" << std::endl;
#endif

  if (!remaining_ir) {
    return nullptr;
  }

  // Step 1: Collect all remaining tables (not executed)
  auto all_tables = CollectTableIndices(remaining_ir.get());
  std::set<unsigned int> remaining_tables;
  for (unsigned int t : all_tables) {
    if (executed_table_indices.find(t) == executed_table_indices.end()) {
      remaining_tables.insert(t);
    }
  }

#ifndef NDEBUG
  std::cout << "[FKBasedSplitter::UpdateRemainingIR] Remaining tables: ";
  for (unsigned int t : remaining_tables) {
    std::cout << t << " ";
  }
  std::cout << std::endl;
#endif

  // Step 2: Collect Scan nodes for remaining tables
  auto remaining_scans =
      CollectScansForTables(remaining_ir.get(), remaining_tables);
#ifndef NDEBUG
  std::cout << "[FKBasedSplitter::UpdateRemainingIR] Found "
            << remaining_scans.size() << " scan node(s) for remaining tables"
            << std::endl;
#endif

  // Step 3: Move join conditions from old IR, filtering out FK-FK joins
  // using current_join_pairs_ (FK-FK pairs already removed by rRj).
  // Must run BEFORE PrepareForNextIteration so indices still match old IR.
  std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
      internal_joins;
  std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
      cross_boundary_joins;

  MoveValidJoins(remaining_ir, remaining_tables, executed_table_indices,
                 internal_joins, cross_boundary_joins);
#ifndef NDEBUG
  std::cout << "[FKBasedSplitter::UpdateRemainingIR] Moved "
            << internal_joins.size() << " internal join condition(s)"
            << std::endl;
  std::cout << "[FKBasedSplitter::UpdateRemainingIR] Moved "
            << cross_boundary_joins.size()
            << " cross-boundary join condition(s)" << std::endl;
#endif

  // Step 4: Update join graph for temp table (after MoveValidJoins so
  // current_join_pairs_ still has old indices matching the old IR)
  PrepareForNextIteration(executed_table_indices, temp_table_index,
                          temp_table_name);

  // Step 5: Rewrite cross-boundary join conditions: replace executed-table
  // attrs with temp table attrs (PostgreSQL Prepare4Next rewrites all Var nodes
  // to temp)
  for (auto &cond : cross_boundary_joins) {
    if (!cond)
      continue;

    unsigned int left_table = cond->left_attr->GetTableIndex();
    unsigned int right_table = cond->right_attr->GetTableIndex();
    bool left_executed = executed_table_indices.count(left_table) > 0;
    bool right_executed = executed_table_indices.count(right_table) > 0;

    auto rewriteAttr =
        [&](std::unique_ptr<ir_sql_converter::SimplestAttr> &attr) {
          unsigned int old_table = attr->GetTableIndex();
          unsigned int old_col = attr->GetColumnIndex();
          // Find this column in the temp table's column_mappings
          for (size_t i = 0; i < column_mappings.size(); i++) {
            if (column_mappings[i].first == old_table &&
                column_mappings[i].second == old_col) {
              attr = std::make_unique<ir_sql_converter::SimplestAttr>(
                  attr->GetType(), temp_table_index,
                  static_cast<unsigned int>(i), column_names[i]);
#ifndef NDEBUG
              std::cout << "[FKBasedSplitter::UpdateRemainingIR] Rewrote attr ("
                        << old_table << ", " << old_col << ") -> ("
                        << temp_table_index << ", " << i << ") "
                        << column_names[i] << std::endl;
#endif
              break;
            }
          }
        };

    if (left_executed) {
      rewriteAttr(cond->left_attr);
    }
    if (right_executed) {
      rewriteAttr(cond->right_attr);
    }
  }

  // Step 6: Extract filter conditions for remaining tables using AND-splitting
  // For a filter like: #(1,1)... && #(3,1)... && #(4,4)...
  // If tables 1,4 are executed and table 3 is remaining,
  // we extract: #(3,1)... for the remaining IR
  std::vector<std::unique_ptr<ir_sql_converter::SimplestExpr>>
      remaining_filters;

  std::function<void(std::unique_ptr<ir_sql_converter::SimplestStmt> &)>
      ExtractRemainingFilters;
  ExtractRemainingFilters =
      [&](std::unique_ptr<ir_sql_converter::SimplestStmt> &node) {
        if (!node)
          return;

        // Use AND-splitting to extract conjuncts for remaining tables
        for (auto &qual : node->qual_vec) {
          if (qual) {
            auto extracted = ir_utils::ExtractConjunctsForTables(
                qual.get(), remaining_tables);
            if (extracted) {
              remaining_filters.push_back(std::move(extracted));
            }
          }
        }

        for (auto &child : node->children) {
          ExtractRemainingFilters(child);
        }
      };

  ExtractRemainingFilters(remaining_ir);
#ifndef NDEBUG
  std::cout << "[FKBasedSplitter::UpdateRemainingIR] Moved "
            << remaining_filters.size() << " remaining filter condition(s)"
            << std::endl;
#endif

  // Step 7a. Find Aggregate node in original IR
  std::function<ir_sql_converter::SimplestAggregate *(
      const std::unique_ptr<ir_sql_converter::SimplestStmt> &)>
      FindAggregateNode;
  FindAggregateNode =
      [&](const std::unique_ptr<ir_sql_converter::SimplestStmt> &node)
      -> ir_sql_converter::SimplestAggregate * {
    if (!node)
      return nullptr;
    if (node->GetNodeType() ==
        ir_sql_converter::SimplestNodeType::AggregateNode) {
      return dynamic_cast<ir_sql_converter::SimplestAggregate *>(node.get());
    }
    for (const auto &child : node->children) {
      auto *found = FindAggregateNode(child);
      if (found)
        return found;
    }
    return nullptr;
  };
  auto *orig_agg = FindAggregateNode(remaining_ir);

  // Step 7b. Find OrderBy node in original IR
  std::function<ir_sql_converter::SimplestOrderBy *(
      const std::unique_ptr<ir_sql_converter::SimplestStmt> &)>
      FindOrderByNode;
  FindOrderByNode =
      [&](const std::unique_ptr<ir_sql_converter::SimplestStmt> &node)
      -> ir_sql_converter::SimplestOrderBy * {
    if (!node)
      return nullptr;
    if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::OrderNode) {
      return dynamic_cast<ir_sql_converter::SimplestOrderBy *>(node.get());
    }
    for (const auto &child : node->children) {
      auto *found = FindOrderByNode(child);
      if (found)
        return found;
    }
    return nullptr;
  };
  auto *orig_order = FindOrderByNode(remaining_ir);

  // Step 7c. Find Limit node in original IR
  std::function<ir_sql_converter::SimplestLimit *(
      const std::unique_ptr<ir_sql_converter::SimplestStmt> &)>
      FindLimitNode;
  FindLimitNode =
      [&](const std::unique_ptr<ir_sql_converter::SimplestStmt> &node)
      -> ir_sql_converter::SimplestLimit * {
    if (!node)
      return nullptr;
    if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::LimitNode) {
      return dynamic_cast<ir_sql_converter::SimplestLimit *>(node.get());
    }
    for (const auto &child : node->children) {
      auto *found = FindLimitNode(child);
      if (found)
        return found;
    }
    return nullptr;
  };
  auto *orig_limit = FindLimitNode(remaining_ir);

  // Step 8: Build the new remaining IR tree

  // 8a: Create SimplestScan for the temp table using pre-computed column names
  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
      temp_scan_target_list;
  for (size_t i = 0; i < column_names.size(); i++) {
    ir_sql_converter::SimplestVarType col_type =
        ir_sql_converter::SimplestVarType::IntVar;

    // Get type from original IR's target list
    for (const auto &attr : remaining_ir->target_list) {
      if (attr && attr->GetTableIndex() == column_mappings[i].first &&
          attr->GetColumnIndex() == column_mappings[i].second) {
        col_type = attr->GetType();
        break;
      }
    }

    auto attr = std::make_unique<ir_sql_converter::SimplestAttr>(
        col_type, temp_table_index, static_cast<unsigned int>(i),
        column_names[i]);
    temp_scan_target_list.push_back(std::move(attr));
  }

  std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> empty_children;
  auto temp_scan_base = std::make_unique<ir_sql_converter::SimplestStmt>(
      std::move(empty_children), std::move(temp_scan_target_list),
      ir_sql_converter::SimplestNodeType::ScanNode);

  // Create SimplestScan for temp table (treat it like a base table)
  auto temp_scan_node = std::make_unique<ir_sql_converter::SimplestScan>(
      std::move(temp_scan_base), temp_table_index, temp_table_name);
  temp_scan_node->SetEstimatedCardinality(temp_table_cardinality);

  // 8b: Build scan nodes for remaining tables - move target_list from old IR.
  // If a scan is wrapped in a Filter->MarkJoin->Chunk subtree (an IN-clause
  // filter), preserve the whole subtree so the SQL generator can emit
  // "col IN (...)" instead of dismantling it into a broken inner join.
  //
  // Helper: find and extract the Filter(SingleAttr)->MarkJoin->[scan,Chunk]
  // (or bare MarkJoin->[scan,Chunk]) subtree that wraps `scan_idx`, moving it
  // out of its slot in the tree (leaves a nullptr in that slot).
  std::function<std::unique_ptr<ir_sql_converter::SimplestStmt>(
      std::unique_ptr<ir_sql_converter::SimplestStmt> &, unsigned int)>
      ExtractMarkJoinSubtree;
  ExtractMarkJoinSubtree =
      [&](std::unique_ptr<ir_sql_converter::SimplestStmt> &node,
          unsigned int scan_idx)
      -> std::unique_ptr<ir_sql_converter::SimplestStmt> {
    if (!node)
      return nullptr;

    // Returns true if `n` is a MarkJoin that directly wraps a scan for scan_idx
    auto isMarkJoinForScan = [&](ir_sql_converter::SimplestStmt *n) -> bool {
      if (!n ||
          n->GetNodeType() != ir_sql_converter::SimplestNodeType::JoinNode)
        return false;
      auto *j = dynamic_cast<ir_sql_converter::SimplestJoin *>(n);
      if (!j ||
          j->GetSimplestJoinType() != ir_sql_converter::SimplestJoinType::Mark)
        return false;
      for (auto &c : j->children) {
        if (c &&
            c->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
          auto *s = dynamic_cast<ir_sql_converter::SimplestScan *>(c.get());
          if (s && s->GetTableIndex() == scan_idx)
            return true;
        }
      }
      return false;
    };

    // Case 1: current node IS the MarkJoin wrapping our scan
    if (isMarkJoinForScan(node.get()))
      return std::move(node);

    // Case 2: current node is a Filter whose child is the MarkJoin
    if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::FilterNode &&
        !node->children.empty() && isMarkJoinForScan(node->children[0].get())) {
      return std::move(node);
    }

    // Recurse: look for the target in each child slot
    for (auto &child : node->children) {
      auto result = ExtractMarkJoinSubtree(child, scan_idx);
      if (result)
        return result;
    }
    return nullptr;
  };

  std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> scan_nodes;
  scan_nodes.push_back(std::move(temp_scan_node)); // Add temp table first

  for (auto *scan : remaining_scans) {
    // Check if this scan is wrapped in a Mark-join IN-clause subtree
    auto mark_subtree =
        ExtractMarkJoinSubtree(remaining_ir, scan->GetTableIndex());
    if (mark_subtree) {
      // Use the entire Filter+MarkJoin+Chunk subtree as the "scan unit"
      scan_nodes.push_back(std::move(mark_subtree));
      continue;
    }

    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> scan_children;

    // Move target_list from old scan node (we own remaining_ir)
    auto base_stmt = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(scan_children), std::move(scan->target_list),
        ir_sql_converter::SimplestNodeType::ScanNode);

    auto new_scan = std::make_unique<ir_sql_converter::SimplestScan>(
        std::move(base_stmt), scan->GetTableIndex(), scan->GetTableName());
    new_scan->SetEstimatedCardinality(scan->GetEstimatedCardinality());
    scan_nodes.push_back(std::move(new_scan));
  }

  // 8c: Build left-deep join tree
  std::unique_ptr<ir_sql_converter::SimplestStmt> result =
      std::move(scan_nodes[0]);

  // Combine all join conditions
  std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
      all_joins;
  for (auto &j : internal_joins) {
    all_joins.push_back(std::move(j));
  }
  for (auto &j : cross_boundary_joins) {
    all_joins.push_back(std::move(j));
  }

  for (size_t i = 1; i < scan_nodes.size(); i++) {
    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> join_children;
    join_children.push_back(std::move(result));
    join_children.push_back(std::move(scan_nodes[i]));

    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> empty_attrs;
    auto base_stmt = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(join_children), std::move(empty_attrs),
        ir_sql_converter::SimplestNodeType::JoinNode);

    auto join_node = std::make_unique<ir_sql_converter::SimplestJoin>(
        std::move(base_stmt), ir_sql_converter::Inner);

    // Add relevant join conditions
    // For now, add all remaining join conditions to the last join
    if (i == scan_nodes.size() - 1) {
      for (auto &cond : all_joins) {
        if (cond) {
          join_node->join_conditions.push_back(std::move(cond));
        }
      }
    }

    result = std::move(join_node);
  }

  // 8d: Add filter conditions to qual_vec
  if (!remaining_filters.empty()) {
    // Build Filter node
    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>>
        filter_children;
    filter_children.emplace_back(std::move(result));
    // todo: add target_list content
    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> target_list;
    // add qual vec
    std::vector<std::unique_ptr<ir_sql_converter::SimplestExpr>> qual_vec;
    for (auto &filter : remaining_filters) {
      if (filter) {
        qual_vec.push_back(std::move(filter));
      }
    }

    auto base_stmt = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(filter_children), std::move(target_list), std::move(qual_vec),
        ir_sql_converter::SimplestNodeType::FilterNode);

    result = std::make_unique<ir_sql_converter::SimplestFilter>(
        std::move(base_stmt));
  }

  // 8e: Add Projection node on top - move target_list from old IR
  std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> proj_children;
  proj_children.push_back(std::move(result));

  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> proj_target_list;
  if (!orig_agg && !orig_order && !orig_limit) {
    proj_target_list = std::move(remaining_ir->target_list);
  }
  auto proj_base = std::make_unique<ir_sql_converter::SimplestStmt>(
      std::move(proj_children), std::move(proj_target_list),
      ir_sql_converter::SimplestNodeType::ProjectionNode);

  result = std::make_unique<ir_sql_converter::SimplestProjection>(
      std::move(proj_base), 0);

  // 8f: Add Aggregate node if original IR had aggregation
  if (orig_agg) {
#ifndef NDEBUG
    std::cout << "[FKBasedSplitter::UpdateRemainingIR] Found aggregation in "
                 "original IR, "
              << "adding Aggregate node" << std::endl;
#endif

    // Build target_list first (need clones since groups/agg_fns will be moved)
    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
        agg_target_list;
    for (const auto &grp : orig_agg->groups) {
      agg_target_list.push_back(
          std::make_unique<ir_sql_converter::SimplestAttr>(*grp));
    }
    for (const auto &fn_pair : orig_agg->agg_fns) {
      agg_target_list.push_back(
          std::make_unique<ir_sql_converter::SimplestAttr>(*fn_pair.first));
    }

    // Build Aggregate node - move groups and agg_fns directly from old IR
    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> agg_children;
    agg_children.push_back(std::move(result));

    auto agg_base = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(agg_children), std::move(agg_target_list),
        ir_sql_converter::SimplestNodeType::AggregateNode);

    result = std::make_unique<ir_sql_converter::SimplestAggregate>(
        std::move(agg_base), std::move(orig_agg->agg_fns),
        std::move(orig_agg->groups), orig_agg->GetAggIndex(),
        orig_agg->GetGroupIndex());
  }

  // 8g: Add OrderBy node if original IR had ORDER BY
  if (orig_order) {
#ifndef NDEBUG
    std::cout << "[FKBasedSplitter::UpdateRemainingIR] Found ORDER BY in "
                 "original IR, "
              << "adding OrderBy node" << std::endl;
#endif

    // Build OrderBy node - move target_list and orders directly from old IR
    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> order_children;
    order_children.push_back(std::move(result));

    auto order_base = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(order_children), std::move(orig_order->target_list),
        ir_sql_converter::SimplestNodeType::OrderNode);

    result = std::make_unique<ir_sql_converter::SimplestOrderBy>(
        std::move(order_base), std::move(orig_order->orders));
  }

  // 8h: Add Limit node if original IR had LIMIT
  if (orig_limit) {
#ifndef NDEBUG
    std::cout
        << "[FKBasedSplitter::UpdateRemainingIR] Found LIMIT in original IR, "
        << "adding Limit node" << std::endl;
#endif

    // Build Limit node - move target_list directly from old IR
    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> limit_children;
    limit_children.push_back(std::move(result));

    auto limit_base = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(limit_children), std::move(orig_limit->target_list),
        ir_sql_converter::SimplestNodeType::LimitNode);

    result = std::make_unique<ir_sql_converter::SimplestLimit>(
        std::move(limit_base), orig_limit->limit_val, orig_limit->offset_val);
  }

#ifndef NDEBUG
  std::cout
      << "[FKBasedSplitter::UpdateRemainingIR] Built new remaining IR with "
      << remaining_tables.size() << " remaining tables + temp table"
      << (orig_agg ? " + aggregation" : "") << (orig_order ? " + order by" : "")
      << (orig_limit ? " + limit" : "") << std::endl;
#endif

  return result;
}

} // namespace middleware
