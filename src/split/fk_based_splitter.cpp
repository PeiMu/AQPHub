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
  table_index_to_name_ = CollectTables(ir.get());

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

  // Step 2: Extract foreign keys
  std::set<std::string> table_names;
  for (const auto &[idx, name] : table_index_to_name_) {
    table_names.insert(name);
  }

  fk_graph_ = fk_extractor_.ExtractForTables(table_names);

#ifndef NDEBUG
  fk_graph_.Print();
#endif

  // Step 3: Mark entity vs relationship tables
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

  // Step 4: Build join graph (implements PostgreSQL's List2Graph)
  BuildJoinGraph(ir.get());

#ifndef NDEBUG
  join_graph_.Print();
#endif

  // Reset state
  split_iteration_ = 0;
  executed_tables_.clear();

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

  // Resize graph to accommodate all tables
  int num_tables = table_index_to_name_.size();
  join_graph_.Resize(num_tables);

  // Collect all join conditions from IR
  join_pairs_ = CollectJoinConditions(ir);

#ifndef NDEBUG
  std::cout << "[" << GetStrategyName() << "] Found " << join_pairs_.size()
            << " join condition(s) from IR" << std::endl;
#endif

  // For RelationshipCenter and EntityCenter, remove redundant FK-FK joins
  // This implements PostgreSQL's rRj (removeRedundantJoin) function
  // which is called before List2Graph when algorithm != Minsubquery
  if (strategy_ == SplitStrategy::RELATIONSHIP_CENTER ||
      strategy_ == SplitStrategy::ENTITY_CENTER) {
    join_pairs_ = RemoveRedundantJoins(join_pairs_);
  }

  // Separate FK joins from non-FK joins (PostgreSQL List2Graph pattern)
  std::vector<std::pair<unsigned int, unsigned int>> fk_joins;
  std::vector<std::pair<unsigned int, unsigned int>> non_fk_joins;

  for (const auto &[t1, t2] : join_pairs_) {
    if (IsFKJoin(t1, t2)) {
      fk_joins.push_back({t1, t2});
    } else {
      non_fk_joins.push_back({t1, t2});
    }
  }

#ifndef NDEBUG
  std::cout << "[" << GetStrategyName() << "] FK joins: " << fk_joins.size()
            << ", Non-FK joins: " << non_fk_joins.size() << std::endl;
#endif

  // === Process based on algorithm type ===
  // This implements PostgreSQL's List2Graph() from query_split.c:1811-1911

  if (strategy_ == SplitStrategy::MIN_SUBQUERY) {
    // MinSubquery: symmetric graph with i < j (upper triangular)
    // PostgreSQL lines 1856-1866
    for (const auto &[t1, t2] : join_pairs_) {
      unsigned int i = std::min(t1, t2);
      unsigned int j = std::max(t1, t2);
      join_graph_.SetEdge(i, j, true);
#ifndef NDEBUG
      std::cout << "  MinSubquery edge: " << i << " -> " << j << std::endl;
#endif
    }
  } else if (strategy_ == SplitStrategy::RELATIONSHIP_CENTER ||
             strategy_ == SplitStrategy::ENTITY_CENTER) {

    // Step 1: Process FK joins FIRST with directed edges
    // PostgreSQL lines 1818-1851
    for (const auto &[t1, t2] : fk_joins) {
      auto it1 = table_index_to_name_.find(t1);
      auto it2 = table_index_to_name_.find(t2);

      if (it1 == table_index_to_name_.end() ||
          it2 == table_index_to_name_.end()) {
        continue;
      }

      // Determine which table is the FK owner (con_relid) and which is
      // referenced (ref_relid)
      unsigned int fk_owner_idx, pk_ref_idx;
      if (fk_graph_.HasDirectFK(it1->second, it2->second)) {
        // t1 has FK to t2: t1 is fk_owner (relationship), t2 is pk_ref (entity)
        fk_owner_idx = t1;
        pk_ref_idx = t2;
      } else {
        // t2 has FK to t1: t2 is fk_owner (relationship), t1 is pk_ref (entity)
        fk_owner_idx = t2;
        pk_ref_idx = t1;
      }

      if (strategy_ == SplitStrategy::RELATIONSHIP_CENTER) {
        // RelationshipCenter: graph[fk_owner][pk_ref] = true
        // (relationship -> entity)
        // PostgreSQL line 1844
        join_graph_.SetEdge(fk_owner_idx, pk_ref_idx, true);
#ifndef NDEBUG
        std::cout << "  RelationshipCenter FK edge: " << fk_owner_idx << " -> "
                  << pk_ref_idx << " (" << table_index_to_name_.at(fk_owner_idx)
                  << " -> " << table_index_to_name_.at(pk_ref_idx) << ")"
                  << std::endl;
#endif
      } else { // EntityCenter
        // EntityCenter: graph[pk_ref][fk_owner] = true (entity -> relationship)
        // PostgreSQL line 1848
        join_graph_.SetEdge(pk_ref_idx, fk_owner_idx, true);
#ifndef NDEBUG
        std::cout << "  EntityCenter FK edge: " << pk_ref_idx << " -> "
                  << fk_owner_idx << " (" << table_index_to_name_.at(pk_ref_idx)
                  << " -> " << table_index_to_name_.at(fk_owner_idx) << ")"
                  << std::endl;
#endif
      }
    }

    // Step 2: Process non-FK joins with direction based on entity/relationship
    // PostgreSQL lines 1852-1908
    for (const auto &[t1, t2] : non_fk_joins) {
      bool t1_is_relationship = is_relationship_[t1];
      bool t2_is_relationship = is_relationship_[t2];

      if (strategy_ == SplitStrategy::RELATIONSHIP_CENTER) {
        // PostgreSQL lines 1867-1886
        if (t1_is_relationship && !t2_is_relationship) {
          // t1 is relationship, t2 is entity: graph[t1][t2] = true
          join_graph_.SetEdge(t1, t2, true);
#ifndef NDEBUG
          std::cout << "  RelationshipCenter non-FK: " << t1 << " -> " << t2
                    << std::endl;
#endif
        } else if (!t1_is_relationship && t2_is_relationship) {
          // t1 is entity, t2 is relationship: graph[t2][t1] = true
          join_graph_.SetEdge(t2, t1, true);
#ifndef NDEBUG
          std::cout << "  RelationshipCenter non-FK: " << t2 << " -> " << t1
                    << std::endl;
#endif
        } else {
          // Both entities or both relationships: bidirectional
          join_graph_.SetEdge(t1, t2, true);
          join_graph_.SetEdge(t2, t1, true);
#ifndef NDEBUG
          std::cout << "  RelationshipCenter non-FK bidirectional: " << t1
                    << " <-> " << t2 << std::endl;
#endif
        }
      } else { // EntityCenter
        // PostgreSQL lines 1888-1907
        if (!t1_is_relationship && t2_is_relationship) {
          // t1 is entity, t2 is relationship: graph[t1][t2] = true
          join_graph_.SetEdge(t1, t2, true);
#ifndef NDEBUG
          std::cout << "  EntityCenter non-FK: " << t1 << " -> " << t2
                    << std::endl;
#endif
        } else if (t1_is_relationship && !t2_is_relationship) {
          // t1 is relationship, t2 is entity: graph[t2][t1] = true
          join_graph_.SetEdge(t2, t1, true);
#ifndef NDEBUG
          std::cout << "  EntityCenter non-FK: " << t2 << " -> " << t1
                    << std::endl;
#endif
        } else {
          // Both entities or both relationships: bidirectional
          join_graph_.SetEdge(t1, t2, true);
          join_graph_.SetEdge(t2, t1, true);
#ifndef NDEBUG
          std::cout << "  EntityCenter non-FK bidirectional: " << t1 << " <-> "
                    << t2 << std::endl;
#endif
        }
      }
    }
  }
}

std::map<unsigned int, std::string>
FKBasedSplitter::CollectTables(const ir_sql_converter::SimplestStmt *ir) {
  std::map<unsigned int, std::string> tables;

  if (!ir)
    return tables;

  // Collect Scan nodes
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = dynamic_cast<const ir_sql_converter::SimplestScan *>(ir);
    if (scan) {
      tables[scan->GetTableIndex()] = scan->GetTableName();
    }
  }

  // Recursively collect from children
  for (const auto &child : ir->children) {
    auto child_tables = CollectTables(child.get());
    tables.insert(child_tables.begin(), child_tables.end());
  }

  return tables;
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
      // Extract table indices from join conditions
      for (const auto &cond : join->join_conditions) {
        unsigned int left_table = cond->left_attr->GetTableIndex();
        unsigned int right_table = cond->right_attr->GetTableIndex();
        joins.emplace_back(left_table, right_table);
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

  int num_tables = tables.size();
  std::vector<bool> is_relationship(num_tables, true); // Default: relationship

  // Referenced tables are entities (PostgreSQL pattern from rRj)
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

  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ChunkNode) {
    auto *chunk = dynamic_cast<const ir_sql_converter::SimplestChunk *>(node);
    if (chunk) {
      indices.insert(chunk->GetTableIndex());
    }
  }

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

  bool complete = (remaining_edges == 1);

#ifndef NDEBUG
  std::cout << "[" << GetStrategyName()
            << "] IsComplete: " << (complete ? "YES" : "NO")
            << " (remaining edges between non-executed tables: "
            << remaining_edges << ")" << std::endl;
#endif

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

  // Add table names
  bool first = true;
  for (int idx : cluster) {
    auto it = table_index_to_name_.find(static_cast<unsigned int>(idx));
    if (it == table_index_to_name_.end())
      continue;

    if (!first)
      sql += ", ";
    sql += it->second;
    first = false;
  }

  // Collect join conditions for tables in the cluster
  std::set<unsigned int> cluster_tables;
  for (int idx : cluster) {
    cluster_tables.insert(static_cast<unsigned int>(idx));
  }

  // Get join conditions from IR
  auto all_joins = CollectJoinConditions(ir);
  std::vector<std::string> where_clauses;

  for (const auto &[t1, t2] : all_joins) {
    // Only include joins where both tables are in the cluster
    if (cluster_tables.count(t1) && cluster_tables.count(t2)) {
      // Generate a simple join condition
      // We need the actual column names, but for cost estimation we can use
      // a placeholder that the optimizer will handle
      auto it1 = table_index_to_name_.find(t1);
      auto it2 = table_index_to_name_.find(t2);
      if (it1 != table_index_to_name_.end() &&
          it2 != table_index_to_name_.end()) {
        // Use "1=1" as placeholder - the actual condition doesn't matter much
        // for cost estimation as long as the join type is correct
        // For more accurate estimation, we'd need to extract actual column
        // names
        where_clauses.push_back(it1->second + ".id = " + it2->second + ".id");
      }
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

  return adapter_->GetEstimatedCost(sql);
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

        // Skip joins that no longer exist in join_graph_ (removed as redundant)
        // Check both directions since graph may be asymmetric
        if (!join_graph_.HasEdge(left_table, right_table) &&
            !join_graph_.HasEdge(right_table, left_table)) {
          continue;
        }

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
#ifndef NDEBUG
  std::cout << "[BuildSubIRForCluster] Found " << join_conditions.size()
            << " internal join condition(s)" << std::endl;
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

  // 4a: From original target_list
  for (const auto &attr : ir->target_list) {
    if (cluster_tables.count(attr->GetTableIndex())) {
      auto key = std::make_pair(attr->GetTableIndex(), attr->GetColumnIndex());
      if (seen_attrs.find(key) == seen_attrs.end()) {
        seen_attrs.insert(key);
        required_attrs.push_back(
            std::make_unique<ir_sql_converter::SimplestAttr>(*attr));
      }
    }
  }

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
  auto [table1, table2] = FindNextPair(remaining_ir);

  if (table1 == -1 || table2 == -1) {
    std::cout << "[MinSubquery] No more pairs to join" << std::endl;
    return nullptr;
  }

#ifndef NDEBUG
  std::cout << "[MinSubquery] Selected pair: " << table1 << ", " << table2
            << " (" << table_index_to_name_[table1] << ", "
            << table_index_to_name_[table2] << ")" << std::endl;
#endif

  // Remove this edge from graph
  // PostgreSQL lines 1345: graph[X][Y] = false
  join_graph_.SetEdge(table1, table2, false);

  // PostgreSQL lines 1346-1368: Handle transitive closure
  // If a third table i is connected to both X and Y, remove edge to X
  int length = join_graph_.Size();
  for (int i = 0; i < length; i++) {
    if (i == table1 || i == table2)
      continue;

    bool connected_to_t1 = false;
    bool connected_to_t2 = false;

    // Check connection to table1 (X)
    if (i < table1) {
      connected_to_t1 = join_graph_.HasEdge(i, table1);
    } else {
      connected_to_t1 = join_graph_.HasEdge(table1, i);
    }

    // Check connection to table2 (Y)
    if (i < table2) {
      connected_to_t2 = join_graph_.HasEdge(i, table2);
    } else {
      connected_to_t2 = join_graph_.HasEdge(table2, i);
    }

    // If connected to both, remove edge to table1 (X)
    if (connected_to_t1 && connected_to_t2) {
      if (i < table1) {
        join_graph_.SetEdge(i, table1, false);
      } else {
        join_graph_.SetEdge(table1, i, false);
      }
#ifndef NDEBUG
      std::cout << "[MinSubquery] Removed transitive edge between " << i
                << " and " << table1 << std::endl;
#endif
    }
  }

  // Create extraction with these two tables
  std::set<unsigned int> table_indices = {static_cast<unsigned int>(table1),
                                          static_cast<unsigned int>(table2)};

  auto extraction = std::make_unique<SubqueryExtraction>(
      table_indices, "temp_" + std::to_string(split_iteration_));

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
MinSubquerySplitter::FindNextPair(ir_sql_converter::SimplestStmt *ir) {
  // PostgreSQL-style: evaluate all candidate pairs and select lowest cost
  // Lines 1267-1304 in query_split.c

  std::pair<int, int> best_pair = {-1, -1};
  double best_cost = std::numeric_limits<double>::max();

#ifndef NDEBUG
  std::cout << "[MinSubquery] Evaluating candidate pairs:" << std::endl;
#endif

  for (int i = 0; i < join_graph_.Size(); i++) {
    // Skip tables that have already been executed
    if (executed_tables_.count(i)) {
      continue;
    }

    for (int j = i + 1; j < join_graph_.Size(); j++) {
      // Skip tables that have already been executed
      if (executed_tables_.count(j)) {
        continue;
      }

      if (!join_graph_.HasEdge(i, j)) {
        continue;
      }

      // Create cluster for this pair
      std::vector<int> cluster = {i, j};
      auto [cost, rows] = GetClusterCost(cluster, ir);

#ifndef NDEBUG
      std::cout << "  Pair (" << i << ", " << j << ") ["
                << table_index_to_name_[i] << ", " << table_index_to_name_[j]
                << "]: cost=" << cost << ", rows=" << rows << std::endl;
#endif

      // PostgreSQL's tarfunc comparison (simplified: use cost)
      if (cost < best_cost) {
        best_cost = cost;
        best_pair = {i, j};
      }
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
  auto cluster = FindRelationshipCluster(remaining_ir);

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

  // Remove edges for this relationship
  // AQP-PostgreSQL lines 1330-1337: clear all edges from center X
  int relationship_idx = cluster[0];
  for (int j = 0; j < join_graph_.Size(); j++) {
    if (join_graph_.HasEdge(relationship_idx, j)) {
      join_graph_.SetEdge(relationship_idx, j, false);
      join_graph_.SetEdge(j, relationship_idx, false);
    }
  }

  // Create extraction
  std::set<unsigned int> table_indices;
  for (int idx : cluster) {
    table_indices.insert(static_cast<unsigned int>(idx));
  }

  auto extraction = std::make_unique<SubqueryExtraction>(
      table_indices, "temp_" + std::to_string(split_iteration_));

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
    ir_sql_converter::SimplestStmt *ir) {
  // PostgreSQL-style: evaluate all candidate clusters and select lowest cost
  // Lines 1220-1254 in query_split.c

  std::vector<int> best_cluster;
  double best_cost = std::numeric_limits<double>::max();

#ifndef NDEBUG
  std::cout << "[RelationshipCenter] Evaluating candidate clusters:"
            << std::endl;
#endif

  // First pass: check relationship tables as centers
  for (int i = 0; i < join_graph_.Size(); i++) {
    // Skip tables that have already been executed
    if (executed_tables_.count(i)) {
      continue;
    }

    // For RelationshipCenter, prefer relationship tables as centers
    if (!is_relationship_[i]) {
      continue;
    }

    std::vector<int> cluster;
    cluster.push_back(i); // Add center itself

    // Collect all connected tables (entities pointed to by this relationship)
    for (int j = 0; j < join_graph_.Size(); j++) {
      // Skip tables that have already been executed
      if (executed_tables_.count(j)) {
        continue;
      }
      if (join_graph_.HasEdge(i, j)) {
        cluster.push_back(j);
      }
    }

    if (cluster.size() < 2) {
      continue;
    }

    // Get estimated cost for this cluster
    auto [cost, rows] = GetClusterCost(cluster, ir);

#ifndef NDEBUG
    std::cout << "  Cluster center=" << i << " (" << table_index_to_name_[i]
              << "): tables=[";
    for (size_t k = 0; k < cluster.size(); k++) {
      if (k > 0)
        std::cout << ", ";
      std::cout << table_index_to_name_[cluster[k]];
    }
    std::cout << "] cost=" << cost << ", rows=" << rows << std::endl;
#endif

    // PostgreSQL's tarfunc comparison (simplified: use cost)
    if (cost < best_cost) {
      best_cost = cost;
      best_cluster = cluster;
    }
  }

  // If no relationship center found, try any table with connections
  if (best_cluster.empty()) {
    for (int i = 0; i < join_graph_.Size(); i++) {
      // Skip tables that have already been executed
      if (executed_tables_.count(i)) {
        continue;
      }

      std::vector<int> cluster;
      cluster.push_back(i);

      for (int j = 0; j < join_graph_.Size(); j++) {
        // Skip tables that have already been executed
        if (executed_tables_.count(j)) {
          continue;
        }
        if (join_graph_.HasEdge(i, j)) {
          cluster.push_back(j);
        }
      }

      if (cluster.size() < 2) {
        continue;
      }

      auto [cost, rows] = GetClusterCost(cluster, ir);

#ifndef NDEBUG
      std::cout << "  Fallback cluster center=" << i << " ("
                << table_index_to_name_[i] << "): cost=" << cost
                << ", rows=" << rows << std::endl;
#endif

      if (cost < best_cost) {
        best_cost = cost;
        best_cluster = cluster;
      }
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
  auto cluster = FindEntityCluster(remaining_ir);

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

  // Remove edges for this entity
  // PostgreSQL lines 1330-1337: clear all edges from center X
  int entity_idx = cluster[0];
  for (int j = 0; j < join_graph_.Size(); j++) {
    if (join_graph_.HasEdge(entity_idx, j)) {
      join_graph_.SetEdge(entity_idx, j, false);
      join_graph_.SetEdge(j, entity_idx, false);
    }
  }

  // Create extraction
  std::set<unsigned int> table_indices;
  for (int idx : cluster) {
    table_indices.insert(static_cast<unsigned int>(idx));
  }

  auto extraction = std::make_unique<SubqueryExtraction>(
      table_indices, "temp_" + std::to_string(split_iteration_));

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
EntityCenterSplitter::FindEntityCluster(ir_sql_converter::SimplestStmt *ir) {
  // PostgreSQL-style: evaluate all candidate clusters and select lowest cost
  // Inverse of RelationshipCenter

  std::vector<int> best_cluster;
  double best_cost = std::numeric_limits<double>::max();

#ifndef NDEBUG
  std::cout << "[EntityCenter] Evaluating candidate clusters:" << std::endl;
#endif

  // First pass: check entity tables as centers
  for (int i = 0; i < join_graph_.Size(); i++) {
    // Skip tables that have already been executed
    if (executed_tables_.count(i)) {
      continue;
    }

    // For EntityCenter, prefer entity tables as centers
    if (is_relationship_[i]) {
      continue;
    }

    std::vector<int> cluster;
    cluster.push_back(i); // Add center itself

    // Collect all connected tables (relationships pointed to by this entity)
    for (int j = 0; j < join_graph_.Size(); j++) {
      // Skip tables that have already been executed
      if (executed_tables_.count(j)) {
        continue;
      }
      if (join_graph_.HasEdge(i, j)) {
        cluster.push_back(j);
      }
    }

    if (cluster.size() < 2) {
      continue;
    }

    // Get estimated cost for this cluster
    auto [cost, rows] = GetClusterCost(cluster, ir);

#ifndef NDEBUG
    std::cout << "  Cluster center=" << i << " (" << table_index_to_name_[i]
              << "): tables=[";
    for (size_t k = 0; k < cluster.size(); k++) {
      if (k > 0)
        std::cout << ", ";
      std::cout << table_index_to_name_[cluster[k]];
    }
    std::cout << "] cost=" << cost << ", rows=" << rows << std::endl;
#endif

    // PostgreSQL's tarfunc comparison (simplified: use cost)
    if (cost < best_cost) {
      best_cost = cost;
      best_cluster = cluster;
    }
  }

  // If no entity center found, try any table with connections
  if (best_cluster.empty()) {
    for (int i = 0; i < join_graph_.Size(); i++) {
      // Skip tables that have already been executed
      if (executed_tables_.count(i)) {
        continue;
      }

      std::vector<int> cluster;
      cluster.push_back(i);

      for (int j = 0; j < join_graph_.Size(); j++) {
        // Skip tables that have already been executed
        if (executed_tables_.count(j)) {
          continue;
        }
        if (join_graph_.HasEdge(i, j)) {
          cluster.push_back(j);
        }
      }

      if (cluster.size() < 2) {
        continue;
      }

      auto [cost, rows] = GetClusterCost(cluster, ir);

#ifndef NDEBUG
      std::cout << "  Fallback cluster center=" << i << " ("
                << table_index_to_name_[i] << "): cost=" << cost
                << ", rows=" << rows << std::endl;
#endif

      if (cost < best_cost) {
        best_cost = cost;
        best_cluster = cluster;
      }
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

  // Step 3: Move join conditions from old IR (no cloning needed since we own
  // the IR) Helper to check if a table pair exists in join_pairs_
  auto pairExistsInJoinPairs = [this](unsigned int t1,
                                      unsigned int t2) -> bool {
    for (const auto &[p1, p2] : join_pairs_) {
      if ((p1 == t1 && p2 == t2) || (p1 == t2 && p2 == t1)) {
        return true;
      }
    }
    return false;
  };

  // Move join conditions from IR (we own the IR so we can move)
  std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
      internal_joins;
  std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
      cross_boundary_joins;

  std::function<void(std::unique_ptr<ir_sql_converter::SimplestStmt> &)>
      MoveValidJoins;
  MoveValidJoins = [&](std::unique_ptr<ir_sql_converter::SimplestStmt> &node) {
    if (!node)
      return;

    if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
      auto *join = dynamic_cast<ir_sql_converter::SimplestJoin *>(node.get());
      if (join) {
        // Iterate and move conditions we need
        for (auto &cond : join->join_conditions) {
          if (!cond)
            continue;

          unsigned int left_table = cond->left_attr->GetTableIndex();
          unsigned int right_table = cond->right_attr->GetTableIndex();

          // Skip if this join pair was removed as redundant
          if (!pairExistsInJoinPairs(left_table, right_table)) {
#ifndef NDEBUG
            std::cout << "[FKBasedSplitter::UpdateRemainingIR] Skipping "
                         "redundant join: "
                      << left_table << " <-> " << right_table << std::endl;
#endif
            continue;
          }

          bool left_in_remaining = remaining_tables.count(left_table) > 0;
          bool right_in_remaining = remaining_tables.count(right_table) > 0;
          bool left_in_executed = executed_table_indices.count(left_table) > 0;
          bool right_in_executed =
              executed_table_indices.count(right_table) > 0;

          // Internal join: both tables in remaining_tables
          if (left_in_remaining && right_in_remaining) {
            internal_joins.push_back(std::move(cond));
          }
          // Cross-boundary: one in remaining, one in CURRENT executed tables
          else if ((left_in_remaining && right_in_executed) ||
                   (left_in_executed && right_in_remaining)) {
            cross_boundary_joins.push_back(std::move(cond));
          }
        }
      }
    }

    for (auto &child : node->children) {
      MoveValidJoins(child);
    }
  };

  MoveValidJoins(remaining_ir);
#ifndef NDEBUG
  std::cout << "[FKBasedSplitter::UpdateRemainingIR] Moved "
            << internal_joins.size() << " internal join condition(s)"
            << std::endl;
  std::cout << "[FKBasedSplitter::UpdateRemainingIR] Moved "
            << cross_boundary_joins.size()
            << " cross-boundary join condition(s)" << std::endl;
#endif

  // Step 4: Extract filter conditions for remaining tables using AND-splitting
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

  // Step 5a. Find Aggregate node in original IR
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

  // Step 5b. Find OrderBy node in original IR
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

  // Step 5c. Find Limit node in original IR
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

  // Step 6: Build the new remaining IR tree

  // 6a: Create SimplestScan for the temp table using pre-computed column names
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

  // 6b: Build scan nodes for remaining tables - move target_list from old IR
  std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> scan_nodes;
  scan_nodes.push_back(std::move(temp_scan_node)); // Add temp table first

  for (auto *scan : remaining_scans) {
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

  // 6c: Build left-deep join tree
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

  // 6d: Add filter conditions to qual_vec
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

  // 6e: Add Projection node on top - move target_list from old IR
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

  // 6f: Add Aggregate node if original IR had aggregation
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

  // 6g: Add OrderBy node if original IR had ORDER BY
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

  // 6h: Add Limit node if original IR had LIMIT
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
