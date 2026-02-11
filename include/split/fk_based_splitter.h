/*
 * Base class for FK-based split strategies (MinSubquery, RelationshipCenter,
 * EntityCenter) Implements PostgreSQL's graph-based query splitting approach
 */

#pragma once

#include "simplest_ir.h"
#include "split/foreign_key_extractor.h"
#include "split/split_algorithm.h"
#include "util/param_config.h"
#include <map>
#include <set>
#include <vector>

namespace middleware {

// Join graph: adjacency matrix representation
// graph[i][j] = true means tables i and j have a join condition
class JoinGraph {
public:
  // Default constructor (for member variable initialization)
  JoinGraph() : size_(0) {}

  JoinGraph(int num_tables) : size_(num_tables) {
    graph_.resize(size_ * size_, false);
  }

  // Add resize method for later initialization
  void Resize(int num_tables) {
    size_ = num_tables;
    graph_.resize(size_ * size_, false);
  }

  void SetEdge(int i, int j, bool value) {
    if (i >= 0 && i < size_ && j >= 0 && j < size_) {
      graph_[i * size_ + j] = value;
    }
  }

  bool HasEdge(int i, int j) const {
    if (i >= 0 && i < size_ && j >= 0 && j < size_) {
      return graph_[i * size_ + j];
    }
    return false;
  }

  int Size() const { return size_; }

  // Expand graph to accommodate a new table index
  // Preserves existing edges
  void ExpandToSize(int new_size) {
    if (new_size <= size_) {
      return;
    }
    std::vector<bool> new_graph(new_size * new_size, false);
    // Copy existing edges
    for (int i = 0; i < size_; i++) {
      for (int j = 0; j < size_; j++) {
        if (graph_[i * size_ + j]) {
          new_graph[i * new_size + j] = true;
        }
      }
    }
    graph_ = std::move(new_graph);
    size_ = new_size;
  }

  // Count remaining edges
  int CountEdges() const {
    int count = 0;
    for (int i = 0; i < size_; i++) {
      for (int j = i + 1; j < size_; j++) {
        if (HasEdge(i, j) || HasEdge(j, i)) {
          count++;
        }
      }
    }
    return count;
  }

  void Print() const {
    std::cout << "Join Graph (" << size_ << "x" << size_ << "):" << std::endl;
    for (int i = 0; i < size_; i++) {
      for (int j = 0; j < size_; j++) {
        std::cout << (HasEdge(i, j) ? "1" : "0") << " ";
      }
      std::cout << std::endl;
    }
  }

private:
  int size_;
  std::vector<bool> graph_;
};

// Base class for FK-based splitting strategies
class FKBasedSplitter : public SplitAlgorithm {
public:
  FKBasedSplitter(DBAdapter *adapter, BackendEngine engine,
                  SplitStrategy strategy)
      : SplitAlgorithm(adapter), engine_(engine), strategy_(strategy),
        fk_extractor_(adapter, engine) {}

  void Preprocess(std::unique_ptr<ir_sql_converter::SimplestStmt> &ir) override;

  bool IsComplete(const ir_sql_converter::SimplestStmt *remaining_ir) override;

protected:
  // Build join graph from IR and FK information (implements PostgreSQL's
  // List2Graph)
  void BuildJoinGraph(const ir_sql_converter::SimplestStmt *ir);

  // Collect table scans from IR
  std::map<unsigned int, std::string>
  CollectTables(const ir_sql_converter::SimplestStmt *ir);

  // Collect join conditions from IR - returns pairs of (table1_idx, table2_idx)
  std::vector<std::pair<unsigned int, unsigned int>>
  CollectJoinConditions(const ir_sql_converter::SimplestStmt *ir);

  // Mark entity vs relationship tables based on FK
  // Referenced tables = entities (is_relationship[i] = false)
  // Referencing tables = relationships (is_relationship[i] = true)
  std::vector<bool>
  MarkEntityRelationship(const ForeignKeyGraph &fk_graph,
                         const std::map<unsigned int, std::string> &tables);

  // Helper to collect table indices from a node
  std::set<unsigned int>
  CollectTableIndices(const ir_sql_converter::SimplestStmt *node) const;

  // Check if a join between two tables is a FK join
  bool IsFKJoin(unsigned int table1, unsigned int table2) const;

  // Remove redundant FK-FK joins (joins between two relationship tables)
  // This implements PostgreSQL's rRj (removeRedundantJoin) function
  std::vector<std::pair<unsigned int, unsigned int>> RemoveRedundantJoins(
      const std::vector<std::pair<unsigned int, unsigned int>> &joins) const;

  // Generate SQL for a cluster of tables (for cost estimation)
  // Returns empty string if generation fails
  std::string GenerateSQLForCluster(const std::vector<int> &cluster,
                                    ir_sql_converter::SimplestStmt *ir);

  // Get estimated cost for a cluster using adapter's EXPLAIN
  // Returns {cost, rows}
  std::pair<double, double> GetClusterCost(const std::vector<int> &cluster,
                                           ir_sql_converter::SimplestStmt *ir);

  // Find the sub-IR node that represents the join of the given cluster tables
  // Returns the lowest common ancestor node that contains all cluster tables
  ir_sql_converter::SimplestStmt *
  FindSubIRForCluster(ir_sql_converter::SimplestStmt *ir,
                      const std::set<unsigned int> &cluster_tables);

  // Helper: Check if a node contains all the specified tables (and only those)
  bool NodeContainsExactlyTables(ir_sql_converter::SimplestStmt *node,
                                 const std::set<unsigned int> &target_tables);

  // Helper: Check if a node contains any of the specified tables
  bool NodeContainsAnyTable(ir_sql_converter::SimplestStmt *node,
                            const std::set<unsigned int> &target_tables);

  // Build a new sub-IR containing only the specified cluster tables
  // This is equivalent to PostgreSQL's createQuery() function
  std::unique_ptr<ir_sql_converter::SimplestStmt>
  BuildSubIRForCluster(ir_sql_converter::SimplestStmt *ir,
                       const std::set<unsigned int> &cluster_tables);

  // Helper: Collect Scan nodes for specified tables
  std::vector<ir_sql_converter::SimplestScan *>
  CollectScansForTables(ir_sql_converter::SimplestStmt *ir,
                        const std::set<unsigned int> &tables);

  // Helper: Collect attributes from cluster tables needed for join conditions
  // with tables OUTSIDE the cluster (for the remaining plan's joins)
  // NOTE: This stays in FK-based splitter because it uses join_graph_ member
  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
  CollectExternalJoinAttrs(ir_sql_converter::SimplestStmt *ir,
                           const std::set<unsigned int> &cluster_tables);

  // Move valid join conditions from old IR, filtering out FK-FK joins
  // Classifies into internal (both remaining) and cross-boundary (one executed)
  void MoveValidJoins(
      std::unique_ptr<ir_sql_converter::SimplestStmt> &node,
      const std::set<unsigned int> &remaining_tables,
      const std::set<unsigned int> &executed_table_indices,
      std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
          &internal_joins,
      std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
          &cross_boundary_joins);

  // Prepare data structures for the next iteration (PostgreSQL Prepare4Next +
  // List2Graph)
  void
  PrepareForNextIteration(const std::set<unsigned int> &executed_table_indices,
                          unsigned int temp_table_index,
                          const std::string &temp_table_name);

  // Rebuild join_graph_ from scratch using current state
  // (PostgreSQL List2Graph - called after each iteration)
  void RebuildJoinGraph();

  // Update remaining IR by rebuilding (PostgreSQL style)
  // Because cluster tables may be scattered throughout the tree
  // Takes ownership of old IR to move expressions instead of cloning
  std::unique_ptr<ir_sql_converter::SimplestStmt> UpdateRemainingIR(
      std::unique_ptr<ir_sql_converter::SimplestStmt> remaining_ir,
      const std::set<unsigned int> &executed_table_indices,
      unsigned int temp_table_index, const std::string &temp_table_name,
      uint64_t temp_table_cardinality,
      const std::vector<std::pair<unsigned int, unsigned int>> &column_mappings,
      const std::vector<std::string> &column_names) override;

  BackendEngine engine_;
  SplitStrategy strategy_;
  ForeignKeyExtractor fk_extractor_;

  // Current state
  JoinGraph join_graph_;
  std::map<unsigned int, std::string> table_index_to_name_;
  ForeignKeyGraph fk_graph_;
  std::vector<bool> is_relationship_;

  // Current join pairs (updated each iteration, equivalent to PostgreSQL's
  // Joinlist)
  std::vector<std::pair<unsigned int, unsigned int>> current_join_pairs_;
};

// ===== MinSubquery Strategy =====
// Finds pairs of tables to join (smallest possible subqueries)
class MinSubquerySplitter : public FKBasedSplitter {
public:
  MinSubquerySplitter(DBAdapter *adapter, BackendEngine engine)
      : FKBasedSplitter(adapter, engine, SplitStrategy::MIN_SUBQUERY) {}

  std::unique_ptr<SubqueryExtraction>
  ExtractNextSubquery(ir_sql_converter::SimplestStmt *remaining_ir) override;

  std::string GetStrategyName() const override { return "MinSubquery"; }

private:
  // Find next pair of tables that have a join edge (cost-based selection)
  // estimated_rows: output parameter for the optimizer's estimated row count
  std::pair<int, int> FindNextPair(ir_sql_converter::SimplestStmt *ir,
                                   double &estimated_rows);
};

// ===== RelationshipCenter Strategy =====
// Joins relationship tables with all connected entity tables
class RelationshipCenterSplitter : public FKBasedSplitter {
public:
  RelationshipCenterSplitter(DBAdapter *adapter, BackendEngine engine)
      : FKBasedSplitter(adapter, engine, SplitStrategy::RELATIONSHIP_CENTER) {}

  std::unique_ptr<SubqueryExtraction>
  ExtractNextSubquery(ir_sql_converter::SimplestStmt *remaining_ir) override;

  std::string GetStrategyName() const override { return "RelationshipCenter"; }

private:
  // Find next relationship table and its connected entities (cost-based
  // selection)
  // estimated_rows: output parameter for the optimizer's estimated row count
  std::vector<int> FindRelationshipCluster(ir_sql_converter::SimplestStmt *ir,
                                           double &estimated_rows);
};

// ===== EntityCenter Strategy =====
// Joins entity tables with all connected relationship tables
class EntityCenterSplitter : public FKBasedSplitter {
public:
  EntityCenterSplitter(DBAdapter *adapter, BackendEngine engine)
      : FKBasedSplitter(adapter, engine, SplitStrategy::ENTITY_CENTER) {}

  std::unique_ptr<SubqueryExtraction>
  ExtractNextSubquery(ir_sql_converter::SimplestStmt *remaining_ir) override;

  std::string GetStrategyName() const override { return "EntityCenter"; }

private:
  // Find next entity table and its connected relationships (cost-based
  // selection)
  // estimated_rows: output parameter for the optimizer's estimated row count
  std::vector<int> FindEntityCluster(ir_sql_converter::SimplestStmt *ir,
                                     double &estimated_rows);
};

} // namespace middleware
