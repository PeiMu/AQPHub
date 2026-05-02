/*
 * IR-level implementation of ReorderGet optimization
 * Reorders table scans by cardinality from smaller (lower) to larger (higher)
 *  for better join performance
 */

#pragma once

#include "adapters/db_adapter.h"
#include "simplest_ir.h"
#include <map>
#include <memory>
#include <vector>

namespace middleware {

class IRReorderGet {
public:
  explicit IRReorderGet(EngineAdapter *adapter) : adapter_(adapter) {}

  // Reorder tables in IR by cardinality (smallest first)
  std::unique_ptr<ir_sql_converter::AQPStmt>
  Reorder(std::unique_ptr<ir_sql_converter::AQPStmt> ir);

private:
  struct TableInfo {
    unsigned int table_index;
    uint64_t cardinality;
    std::string table_name;
    std::unique_ptr<ir_sql_converter::SimplestScan> scan_node;
    bool is_chunk = false;

    TableInfo(unsigned int idx, uint64_t card, std::string name,
              std::unique_ptr<ir_sql_converter::SimplestScan> scan,
              bool chunk = false)
        : table_index(idx), cardinality(card), table_name(std::move(name)),
          scan_node(std::move(scan)), is_chunk(chunk) {}
  };

  // Collect all table scans from IR tree
  void CollectTableScans(ir_sql_converter::AQPStmt *node,
                         std::vector<TableInfo> &tables);

  // Get cardinality estimate from adapter
  uint64_t GetTableCardinality(const std::string &table_name);

  // Collect all join conditions from JOIN nodes
  void CollectJoinConditions(
      ir_sql_converter::AQPStmt *node,
      std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
          &join_conds);

  // Rebuild join tree with tables ordered by cardinality
  std::unique_ptr<ir_sql_converter::AQPStmt> RebuildJoinTree(
      std::vector<TableInfo> &sorted_tables,
      std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
          &join_conditions);

  // Build a map of which tables participate in which join conditions
  std::map<unsigned int, std::vector<ir_sql_converter::SimplestVarComparison *>>
  BuildJoinMap(const std::vector<std::unique_ptr<
                   ir_sql_converter::SimplestVarComparison>> &join_conds);

  // Preserve top-level operators (PROJECTION, AGGREGATE, etc.)
  // and attach reordered join tree underneath
  std::unique_ptr<ir_sql_converter::AQPStmt> PreserveTopOperators(
      std::unique_ptr<ir_sql_converter::AQPStmt> original_ir,
      std::unique_ptr<ir_sql_converter::AQPStmt> reordered_join_tree);

  // Recursively find and replace the join subtree
  std::unique_ptr<ir_sql_converter::AQPStmt> ReplaceJoinSubtree(
      std::unique_ptr<ir_sql_converter::AQPStmt> node,
      std::unique_ptr<ir_sql_converter::AQPStmt> new_subtree);

  // Helper to get node type name for logging
  std::string GetNodeTypeName(ir_sql_converter::SimplestNodeType type) const;

  EngineAdapter *adapter_;
};

} // namespace middleware
