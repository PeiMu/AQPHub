/*
 * Implementation of FK-based split strategies
 * Implements PostgreSQL's List2Graph algorithm for query splitting
 */

#include "split/fk_based_splitter.h"
#include <algorithm>
#include <iostream>
#include <limits>

namespace middleware {

// ===== FKBasedSplitter Base Class =====

void FKBasedSplitter::Preprocess(
    std::unique_ptr<ir_sql_converter::SimplestStmt> &ir) {

  std::cout << "[" << GetStrategyName() << "] Preprocessing IR" << std::endl;

  // Step 1: Collect tables from IR
  table_index_to_name_ = CollectTables(ir.get());

  std::cout << "[" << GetStrategyName() << "] Found "
            << table_index_to_name_.size() << " table(s)" << std::endl;

  // Build reverse mapping
  for (const auto &[idx, name] : table_index_to_name_) {
    table_name_to_index_[name] = idx;
    std::cout << "  Table " << idx << ": " << name << std::endl;
  }

  // Step 2: Extract foreign keys
  std::set<std::string> table_names;
  for (const auto &[idx, name] : table_index_to_name_) {
    table_names.insert(name);
  }

  fk_graph_ = fk_extractor_.ExtractForTables(table_names);

  fk_graph_.Print();

  // Step 3: Mark entity vs relationship tables
  // Referenced tables (pk_table in FK) = entities
  // Referencing tables (fk_table in FK) = relationships
  is_relationship_ = MarkEntityRelationship(fk_graph_, table_index_to_name_);

  std::cout << "[" << GetStrategyName()
            << "] Entity/Relationship classification:" << std::endl;
  for (const auto &[idx, name] : table_index_to_name_) {
    std::cout << "  " << name << ": "
              << (is_relationship_[idx] ? "RELATIONSHIP" : "ENTITY")
              << std::endl;
  }

  // Step 4: Build join graph (implements PostgreSQL's List2Graph)
  BuildJoinGraph(ir.get());

  join_graph_.Print();

  split_iteration_ = 0;

  std::cout << "[" << GetStrategyName() << "] Preprocessing complete"
            << std::endl;
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
    bool t1_is_relationship = (t1 < is_relationship_.size()) && is_relationship_[t1];
    bool t2_is_relationship = (t2 < is_relationship_.size()) && is_relationship_[t2];

    if (t1_is_relationship && t2_is_relationship) {
      // Both are relationship tables - this is a redundant FK-FK join
      std::cout << "[" << GetStrategyName() << "] Removing redundant FK-FK join: "
                << t1 << " (" << table_index_to_name_.at(t1) << ") <-> "
                << t2 << " (" << table_index_to_name_.at(t2) << ")" << std::endl;
      continue;
    }

    // Keep this join
    filtered_joins.push_back({t1, t2});
  }

  std::cout << "[" << GetStrategyName() << "] Removed "
            << (joins.size() - filtered_joins.size())
            << " redundant FK-FK join(s)" << std::endl;

  return filtered_joins;
}

void FKBasedSplitter::BuildJoinGraph(const ir_sql_converter::SimplestStmt *ir) {
  std::cout << "[" << GetStrategyName() << "] Building join graph (List2Graph)"
            << std::endl;

  // Resize graph to accommodate all tables
  int num_tables = table_index_to_name_.size();
  join_graph_.Resize(num_tables);

  // Collect all join conditions from IR
  auto join_pairs = CollectJoinConditions(ir);

  std::cout << "[" << GetStrategyName() << "] Found " << join_pairs.size()
            << " join condition(s) from IR" << std::endl;

  // For RelationshipCenter and EntityCenter, remove redundant FK-FK joins
  // This implements PostgreSQL's rRj (removeRedundantJoin) function
  // which is called before List2Graph when algorithm != Minsubquery
  if (strategy_ == SplitStrategy::RELATIONSHIP_CENTER ||
      strategy_ == SplitStrategy::ENTITY_CENTER) {
    join_pairs = RemoveRedundantJoins(join_pairs);
  }

  // Separate FK joins from non-FK joins (PostgreSQL List2Graph pattern)
  std::vector<std::pair<unsigned int, unsigned int>> fk_joins;
  std::vector<std::pair<unsigned int, unsigned int>> non_fk_joins;

  for (const auto &[t1, t2] : join_pairs) {
    if (IsFKJoin(t1, t2)) {
      fk_joins.push_back({t1, t2});
    } else {
      non_fk_joins.push_back({t1, t2});
    }
  }

  std::cout << "[" << GetStrategyName() << "] FK joins: " << fk_joins.size()
            << ", Non-FK joins: " << non_fk_joins.size() << std::endl;

  // === Process based on algorithm type ===
  // This implements PostgreSQL's List2Graph() from query_split.c:1811-1911

  if (strategy_ == SplitStrategy::MIN_SUBQUERY) {
    // MinSubquery: symmetric graph with i < j (upper triangular)
    // PostgreSQL lines 1856-1866
    for (const auto &[t1, t2] : join_pairs) {
      unsigned int i = std::min(t1, t2);
      unsigned int j = std::max(t1, t2);
      join_graph_.SetEdge(i, j, true);
      std::cout << "  MinSubquery edge: " << i << " -> " << j << std::endl;
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
        std::cout << "  RelationshipCenter FK edge: " << fk_owner_idx << " -> "
                  << pk_ref_idx << " ("
                  << table_index_to_name_.at(fk_owner_idx) << " -> "
                  << table_index_to_name_.at(pk_ref_idx) << ")" << std::endl;
      } else { // EntityCenter
        // EntityCenter: graph[pk_ref][fk_owner] = true (entity -> relationship)
        // PostgreSQL line 1848
        join_graph_.SetEdge(pk_ref_idx, fk_owner_idx, true);
        std::cout << "  EntityCenter FK edge: " << pk_ref_idx << " -> "
                  << fk_owner_idx << " (" << table_index_to_name_.at(pk_ref_idx)
                  << " -> " << table_index_to_name_.at(fk_owner_idx) << ")"
                  << std::endl;
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
          std::cout << "  RelationshipCenter non-FK: " << t1 << " -> " << t2
                    << std::endl;
        } else if (!t1_is_relationship && t2_is_relationship) {
          // t1 is entity, t2 is relationship: graph[t2][t1] = true
          join_graph_.SetEdge(t2, t1, true);
          std::cout << "  RelationshipCenter non-FK: " << t2 << " -> " << t1
                    << std::endl;
        } else {
          // Both entities or both relationships: bidirectional
          join_graph_.SetEdge(t1, t2, true);
          join_graph_.SetEdge(t2, t1, true);
          std::cout << "  RelationshipCenter non-FK bidirectional: " << t1
                    << " <-> " << t2 << std::endl;
        }
      } else { // EntityCenter
        // PostgreSQL lines 1888-1907
        if (!t1_is_relationship && t2_is_relationship) {
          // t1 is entity, t2 is relationship: graph[t1][t2] = true
          join_graph_.SetEdge(t1, t2, true);
          std::cout << "  EntityCenter non-FK: " << t1 << " -> " << t2
                    << std::endl;
        } else if (t1_is_relationship && !t2_is_relationship) {
          // t1 is relationship, t2 is entity: graph[t2][t1] = true
          join_graph_.SetEdge(t2, t1, true);
          std::cout << "  EntityCenter non-FK: " << t2 << " -> " << t1
                    << std::endl;
        } else {
          // Both entities or both relationships: bidirectional
          join_graph_.SetEdge(t1, t2, true);
          join_graph_.SetEdge(t2, t1, true);
          std::cout << "  EntityCenter non-FK bidirectional: " << t1 << " <-> "
                    << t2 << std::endl;
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
  // Check if join graph has any remaining edges
  int remaining_edges = join_graph_.CountEdges();
  bool complete = (remaining_edges == 0);

  std::cout << "[" << GetStrategyName()
            << "] IsComplete: " << (complete ? "YES" : "NO")
            << " (remaining edges: " << remaining_edges << ")" << std::endl;

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
      if (it1 != table_index_to_name_.end() && it2 != table_index_to_name_.end()) {
        // Use "1=1" as placeholder - the actual condition doesn't matter much
        // for cost estimation as long as the join type is correct
        // For more accurate estimation, we'd need to extract actual column names
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

  // Strategy: Find the smallest subtree that contains exactly the cluster tables
  // This is the "lowest common ancestor" that spans all cluster tables

  // First, check if current node contains exactly the cluster tables
  auto node_tables = CollectTableIndices(ir);
  if (node_tables == cluster_tables) {
    std::cout << "[FindSubIRForCluster] Found exact match at node type "
              << static_cast<int>(ir->GetNodeType()) << std::endl;
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
      std::cout << "[FindSubIRForCluster] Found exact match in child"
                << std::endl;
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
      std::cout << "[FindSubIRForCluster] Using Join node that contains cluster"
                << std::endl;
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

std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
FKBasedSplitter::CollectJoinConditionsForTables(
    ir_sql_converter::SimplestStmt *ir,
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

        // Include condition if BOTH tables are in our cluster
        if (tables.count(left_table) && tables.count(right_table)) {
          // Clone the condition - constructor order: (comparison_type, left_attr, right_attr)
          auto cloned = std::make_unique<ir_sql_converter::SimplestVarComparison>(
              cond->GetSimplestExprType(),
              std::make_unique<ir_sql_converter::SimplestAttr>(*cond->left_attr),
              std::make_unique<ir_sql_converter::SimplestAttr>(*cond->right_attr));
          conditions.push_back(std::move(cloned));
        }
      }
    }
  }

  // Recurse into children
  for (auto &child : ir->children) {
    auto child_conds = CollectJoinConditionsForTables(child.get(), tables);
    for (auto &c : child_conds) {
      conditions.push_back(std::move(c));
    }
  }

  return conditions;
}

std::vector<std::unique_ptr<ir_sql_converter::SimplestExpr>>
FKBasedSplitter::CollectFilterConditionsForTables(
    ir_sql_converter::SimplestStmt *ir,
    const std::set<unsigned int> &tables) {
  std::vector<std::unique_ptr<ir_sql_converter::SimplestExpr>> filters;

  if (!ir)
    return filters;

  // Check qual_vec for filter conditions in this node
  for (const auto &qual : ir->qual_vec) {
    // Check if filter involves only tables in our cluster
    // For SimplestVarConstComparison, check the attr's table
    if (qual->GetNodeType() == ir_sql_converter::SimplestNodeType::VarConstComparisonNode) {
      auto *var_const = dynamic_cast<ir_sql_converter::SimplestVarConstComparison *>(qual.get());
      if (var_const && tables.count(var_const->attr->GetTableIndex())) {
        // Clone the filter
        auto cloned = std::make_unique<ir_sql_converter::SimplestVarConstComparison>(
            var_const->GetSimplestExprType(),
            std::make_unique<ir_sql_converter::SimplestAttr>(*var_const->attr),
            std::make_unique<ir_sql_converter::SimplestConstVar>(*var_const->const_var));
        filters.push_back(std::move(cloned));
      }
    }
  }

  // Recurse into children
  for (auto &child : ir->children) {
    auto child_filters = CollectFilterConditionsForTables(child.get(), tables);
    for (auto &f : child_filters) {
      filters.push_back(std::move(f));
    }
  }

  return filters;
}

// Helper: Collect attributes from cluster tables that are needed for join conditions
// with tables OUTSIDE the cluster (for the remaining plan)
// NOTE: Only considers joins that still exist in join_graph_ (after RemoveRedundantJoins)
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
        // the cluster table's attr in our output (for the remaining plan's join)
        bool left_in_cluster = cluster_tables.count(left_table) > 0;
        bool right_in_cluster = cluster_tables.count(right_table) > 0;

        if (left_in_cluster && !right_in_cluster) {
          auto key = std::make_pair(left_table, cond->left_attr->GetColumnIndex());
          if (seen.find(key) == seen.end()) {
            seen.insert(key);
            attrs.push_back(std::make_unique<ir_sql_converter::SimplestAttr>(*cond->left_attr));
          }
        }
        if (right_in_cluster && !left_in_cluster) {
          auto key = std::make_pair(right_table, cond->right_attr->GetColumnIndex());
          if (seen.find(key) == seen.end()) {
            seen.insert(key);
            attrs.push_back(std::make_unique<ir_sql_converter::SimplestAttr>(*cond->right_attr));
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

  std::cout << "[BuildSubIRForCluster] Building sub-IR for "
            << cluster_tables.size() << " tables" << std::endl;

  if (!ir || cluster_tables.empty()) {
    return nullptr;
  }

  // Step 1: Collect Scan nodes for tables in the cluster
  auto scans = CollectScansForTables(ir, cluster_tables);
  std::cout << "[BuildSubIRForCluster] Found " << scans.size()
            << " scan node(s)" << std::endl;

  if (scans.empty()) {
    std::cerr << "[BuildSubIRForCluster] No scans found for cluster tables"
              << std::endl;
    return nullptr;
  }

  // Step 2: Collect join conditions between cluster tables (internal joins)
  auto join_conditions = CollectJoinConditionsForTables(ir, cluster_tables);
  std::cout << "[BuildSubIRForCluster] Found " << join_conditions.size()
            << " internal join condition(s)" << std::endl;

  // Step 3: Collect required output attributes
  // - Attributes from original target_list that belong to cluster tables
  // - Attributes needed for joins with tables OUTSIDE the cluster
  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> required_attrs;
  std::set<std::pair<unsigned int, unsigned int>> seen_attrs;

  // 3a: From original target_list
  for (const auto &attr : ir->target_list) {
    if (cluster_tables.count(attr->GetTableIndex())) {
      auto key = std::make_pair(attr->GetTableIndex(), attr->GetColumnIndex());
      if (seen_attrs.find(key) == seen_attrs.end()) {
        seen_attrs.insert(key);
        required_attrs.push_back(std::make_unique<ir_sql_converter::SimplestAttr>(*attr));
      }
    }
  }

  // 3b: From external join conditions (attrs needed for remaining plan)
  auto external_attrs = CollectExternalJoinAttrs(ir, cluster_tables);
  for (auto &attr : external_attrs) {
    auto key = std::make_pair(attr->GetTableIndex(), attr->GetColumnIndex());
    if (seen_attrs.find(key) == seen_attrs.end()) {
      seen_attrs.insert(key);
      required_attrs.push_back(std::move(attr));
    }
  }

  std::cout << "[BuildSubIRForCluster] Required output attributes: "
            << required_attrs.size() << std::endl;

  // Step 4: Build the sub-IR tree (left-deep)
  // Clone scan nodes
  std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> scan_nodes;
  for (auto *scan : scans) {
    // Clone the scan node - build manually since SimplestStmt has no copy constructor
    std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> empty_children;
    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> cloned_target_list;
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
      proj_target.push_back(std::make_unique<ir_sql_converter::SimplestAttr>(*attr));
    }

    auto proj_base = std::make_unique<ir_sql_converter::SimplestStmt>(
        std::move(proj_children), std::move(proj_target),
        ir_sql_converter::SimplestNodeType::ProjectionNode);

    // Use table index 0 for the projection (temporary result)
    return std::make_unique<ir_sql_converter::SimplestProjection>(
        std::move(proj_base), 0);
  }

  // Build left-deep tree using CrossProduct for intermediate joins
  std::unique_ptr<ir_sql_converter::SimplestStmt> result = std::move(scan_nodes[0]);

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

  // Step 5: Add Projection node on top with required attributes
  std::vector<std::unique_ptr<ir_sql_converter::SimplestStmt>> proj_children;
  proj_children.push_back(std::move(result));

  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> proj_target;
  for (auto &attr : required_attrs) {
    proj_target.push_back(std::make_unique<ir_sql_converter::SimplestAttr>(*attr));
  }

  auto proj_base = std::make_unique<ir_sql_converter::SimplestStmt>(
      std::move(proj_children), std::move(proj_target),
      ir_sql_converter::SimplestNodeType::ProjectionNode);

  // Use table index 0 for the projection (will be replaced by temp table index)
  auto projection = std::make_unique<ir_sql_converter::SimplestProjection>(
      std::move(proj_base), 0);

  std::cout << "[BuildSubIRForCluster] Built sub-IR with Projection ("
            << projection->target_list.size() << " output columns)" << std::endl;

  return projection;
}

// ===== MinSubquery Strategy =====

std::unique_ptr<SubqueryExtraction> MinSubquerySplitter::ExtractNextSubquery(
    ir_sql_converter::SimplestStmt *remaining_ir) {

  split_iteration_++;
  std::cout << "\n[MinSubquery] Iteration " << split_iteration_ << std::endl;

  // Find next pair of tables with lowest estimated cost
  auto [table1, table2] = FindNextPair(remaining_ir);

  if (table1 == -1 || table2 == -1) {
    std::cout << "[MinSubquery] No more pairs to join" << std::endl;
    return nullptr;
  }

  std::cout << "[MinSubquery] Selected pair: " << table1 << ", " << table2
            << " (" << table_index_to_name_[table1] << ", "
            << table_index_to_name_[table2] << ")" << std::endl;

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
      std::cout << "[MinSubquery] Removed transitive edge between " << i
                << " and " << table1 << std::endl;
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
    std::cout << "[MinSubquery] Built sub-IR for pair" << std::endl;
  } else {
    // Fallback: try to find existing subtree
    ir_sql_converter::SimplestStmt *found_sub_ir =
        FindSubIRForCluster(remaining_ir, table_indices);
    if (found_sub_ir) {
      extraction->pipeline_breaker_ptr = found_sub_ir;
      std::cout << "[MinSubquery] Using found sub-IR for pair" << std::endl;
    } else {
      std::cerr << "[MinSubquery] Warning: Could not build or find sub-IR"
                << std::endl;
      extraction->pipeline_breaker_ptr = remaining_ir;
    }
  }

  return extraction;
}

std::pair<int, int>
MinSubquerySplitter::FindNextPair(ir_sql_converter::SimplestStmt *ir) {
  // PostgreSQL-style: evaluate all candidate pairs and select lowest cost
  // Lines 1267-1304 in query_split.c

  std::pair<int, int> best_pair = {-1, -1};
  double best_cost = std::numeric_limits<double>::max();

  std::cout << "[MinSubquery] Evaluating candidate pairs:" << std::endl;

  for (int i = 0; i < join_graph_.Size(); i++) {
    for (int j = i + 1; j < join_graph_.Size(); j++) {
      if (!join_graph_.HasEdge(i, j)) {
        continue;
      }

      // Create cluster for this pair
      std::vector<int> cluster = {i, j};
      auto [cost, rows] = GetClusterCost(cluster, ir);

      std::cout << "  Pair (" << i << ", " << j << ") ["
                << table_index_to_name_[i] << ", " << table_index_to_name_[j]
                << "]: cost=" << cost << ", rows=" << rows << std::endl;

      // PostgreSQL's tarfunc comparison (simplified: use cost)
      if (cost < best_cost) {
        best_cost = cost;
        best_pair = {i, j};
      }
    }
  }

  if (best_pair.first != -1) {
    std::cout << "[MinSubquery] Best pair: (" << best_pair.first << ", "
              << best_pair.second << ") with cost=" << best_cost << std::endl;
  }

  return best_pair;
}

// ===== RelationshipCenter Strategy =====

std::unique_ptr<SubqueryExtraction>
RelationshipCenterSplitter::ExtractNextSubquery(
    ir_sql_converter::SimplestStmt *remaining_ir) {

  split_iteration_++;
  std::cout << "\n[RelationshipCenter] Iteration " << split_iteration_
            << std::endl;

  // Find relationship cluster with lowest estimated cost
  auto cluster = FindRelationshipCluster(remaining_ir);

  if (cluster.empty()) {
    std::cout << "[RelationshipCenter] No more relationship clusters"
              << std::endl;
    return nullptr;
  }

  std::cout << "[RelationshipCenter] Selected cluster: ";
  for (int idx : cluster) {
    std::cout << idx << "(" << table_index_to_name_[idx] << ") ";
  }
  std::cout << std::endl;

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
    std::cout << "[RelationshipCenter] Built sub-IR for cluster" << std::endl;
  } else {
    // Fallback: try to find existing subtree
    ir_sql_converter::SimplestStmt *found_sub_ir =
        FindSubIRForCluster(remaining_ir, table_indices);
    if (found_sub_ir) {
      extraction->pipeline_breaker_ptr = found_sub_ir;
      std::cout << "[RelationshipCenter] Using found sub-IR" << std::endl;
    } else {
      std::cerr << "[RelationshipCenter] Warning: Could not build or find sub-IR"
                << std::endl;
      extraction->pipeline_breaker_ptr = remaining_ir;
    }
  }

  return extraction;
}

std::vector<int> RelationshipCenterSplitter::FindRelationshipCluster(
    ir_sql_converter::SimplestStmt *ir) {
  // PostgreSQL-style: evaluate all candidate clusters and select lowest cost
  // Lines 1220-1254 in query_split.c

  std::vector<int> best_cluster;
  double best_cost = std::numeric_limits<double>::max();

  std::cout << "[RelationshipCenter] Evaluating candidate clusters:"
            << std::endl;

  // First pass: check relationship tables as centers
  for (int i = 0; i < join_graph_.Size(); i++) {
    // For RelationshipCenter, prefer relationship tables as centers
    if (!is_relationship_[i]) {
      continue;
    }

    std::vector<int> cluster;
    cluster.push_back(i); // Add center itself

    // Collect all connected tables (entities pointed to by this relationship)
    for (int j = 0; j < join_graph_.Size(); j++) {
      if (join_graph_.HasEdge(i, j)) {
        cluster.push_back(j);
      }
    }

    if (cluster.size() < 2) {
      continue;
    }

    // Get estimated cost for this cluster
    auto [cost, rows] = GetClusterCost(cluster, ir);

    std::cout << "  Cluster center=" << i << " (" << table_index_to_name_[i]
              << "): tables=[";
    for (size_t k = 0; k < cluster.size(); k++) {
      if (k > 0)
        std::cout << ", ";
      std::cout << table_index_to_name_[cluster[k]];
    }
    std::cout << "] cost=" << cost << ", rows=" << rows << std::endl;

    // PostgreSQL's tarfunc comparison (simplified: use cost)
    if (cost < best_cost) {
      best_cost = cost;
      best_cluster = cluster;
    }
  }

  // If no relationship center found, try any table with connections
  if (best_cluster.empty()) {
    for (int i = 0; i < join_graph_.Size(); i++) {
      std::vector<int> cluster;
      cluster.push_back(i);

      for (int j = 0; j < join_graph_.Size(); j++) {
        if (join_graph_.HasEdge(i, j)) {
          cluster.push_back(j);
        }
      }

      if (cluster.size() < 2) {
        continue;
      }

      auto [cost, rows] = GetClusterCost(cluster, ir);

      std::cout << "  Fallback cluster center=" << i << " ("
                << table_index_to_name_[i] << "): cost=" << cost
                << ", rows=" << rows << std::endl;

      if (cost < best_cost) {
        best_cost = cost;
        best_cluster = cluster;
      }
    }
  }

  if (!best_cluster.empty()) {
    std::cout << "[RelationshipCenter] Best cluster center="
              << best_cluster[0] << " ("
              << table_index_to_name_[best_cluster[0]]
              << ") with cost=" << best_cost << std::endl;
  }

  return best_cluster;
}

// ===== EntityCenter Strategy =====

std::unique_ptr<SubqueryExtraction> EntityCenterSplitter::ExtractNextSubquery(
    ir_sql_converter::SimplestStmt *remaining_ir) {

  split_iteration_++;
  std::cout << "\n[EntityCenter] Iteration " << split_iteration_ << std::endl;

  // Find entity cluster with lowest estimated cost
  auto cluster = FindEntityCluster(remaining_ir);

  if (cluster.empty()) {
    std::cout << "[EntityCenter] No more entity clusters" << std::endl;
    return nullptr;
  }

  std::cout << "[EntityCenter] Selected cluster: ";
  for (int idx : cluster) {
    std::cout << idx << "(" << table_index_to_name_[idx] << ") ";
  }
  std::cout << std::endl;

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
    std::cout << "[EntityCenter] Built sub-IR for cluster" << std::endl;
  } else {
    // Fallback: try to find existing subtree
    ir_sql_converter::SimplestStmt *found_sub_ir =
        FindSubIRForCluster(remaining_ir, table_indices);
    if (found_sub_ir) {
      extraction->pipeline_breaker_ptr = found_sub_ir;
      std::cout << "[EntityCenter] Using found sub-IR" << std::endl;
    } else {
      std::cerr << "[EntityCenter] Warning: Could not build or find sub-IR"
                << std::endl;
      extraction->pipeline_breaker_ptr = remaining_ir;
    }
  }

  return extraction;
}

std::vector<int>
EntityCenterSplitter::FindEntityCluster(ir_sql_converter::SimplestStmt *ir) {
  // PostgreSQL-style: evaluate all candidate clusters and select lowest cost
  // Inverse of RelationshipCenter

  std::vector<int> best_cluster;
  double best_cost = std::numeric_limits<double>::max();

  std::cout << "[EntityCenter] Evaluating candidate clusters:" << std::endl;

  // First pass: check entity tables as centers
  for (int i = 0; i < join_graph_.Size(); i++) {
    // For EntityCenter, prefer entity tables as centers
    if (is_relationship_[i]) {
      continue;
    }

    std::vector<int> cluster;
    cluster.push_back(i); // Add center itself

    // Collect all connected tables (relationships pointed to by this entity)
    for (int j = 0; j < join_graph_.Size(); j++) {
      if (join_graph_.HasEdge(i, j)) {
        cluster.push_back(j);
      }
    }

    if (cluster.size() < 2) {
      continue;
    }

    // Get estimated cost for this cluster
    auto [cost, rows] = GetClusterCost(cluster, ir);

    std::cout << "  Cluster center=" << i << " (" << table_index_to_name_[i]
              << "): tables=[";
    for (size_t k = 0; k < cluster.size(); k++) {
      if (k > 0)
        std::cout << ", ";
      std::cout << table_index_to_name_[cluster[k]];
    }
    std::cout << "] cost=" << cost << ", rows=" << rows << std::endl;

    // PostgreSQL's tarfunc comparison (simplified: use cost)
    if (cost < best_cost) {
      best_cost = cost;
      best_cluster = cluster;
    }
  }

  // If no entity center found, try any table with connections
  if (best_cluster.empty()) {
    for (int i = 0; i < join_graph_.Size(); i++) {
      std::vector<int> cluster;
      cluster.push_back(i);

      for (int j = 0; j < join_graph_.Size(); j++) {
        if (join_graph_.HasEdge(i, j)) {
          cluster.push_back(j);
        }
      }

      if (cluster.size() < 2) {
        continue;
      }

      auto [cost, rows] = GetClusterCost(cluster, ir);

      std::cout << "  Fallback cluster center=" << i << " ("
                << table_index_to_name_[i] << "): cost=" << cost
                << ", rows=" << rows << std::endl;

      if (cost < best_cost) {
        best_cost = cost;
        best_cluster = cluster;
      }
    }
  }

  if (!best_cluster.empty()) {
    std::cout << "[EntityCenter] Best cluster center=" << best_cluster[0]
              << " (" << table_index_to_name_[best_cluster[0]]
              << ") with cost=" << best_cost << std::endl;
  }

  return best_cluster;
}

} // namespace middleware
