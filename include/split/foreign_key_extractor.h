/*
 * Extracts foreign key relationships from database metadata
 * Used by FK-based split strategies
 */

#pragma once

#include "adapters/db_adapter.h"
#include "util/param_config.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace middleware {

// Represents a foreign key relationship
struct ForeignKey {
  std::string fk_table;  // Foreign key table name
  std::string fk_column; // Foreign key column name
  std::string pk_table;  // Primary key (referenced) table name
  std::string pk_column; // Primary key column name

  ForeignKey(std::string fk_t, std::string fk_c, std::string pk_t,
             std::string pk_c)
      : fk_table(std::move(fk_t)), fk_column(std::move(fk_c)),
        pk_table(std::move(pk_t)), pk_column(std::move(pk_c)) {}
};

// Foreign key graph for join analysis
class ForeignKeyGraph {
public:
  void AddForeignKey(const ForeignKey &fk);

  // Get all tables that reference the given table (via FK)
  std::vector<std::string> GetReferencingTables(const std::string &table) const;

  // Update FK graph after creating a temp table (PostgreSQL Prepare4Next)
  // - Remove FKs where both tables are executed
  // - Redirect executed table references to temp table
  void UpdateForTempTable(const std::set<std::string> &executed_table_names,
                          const std::string &temp_table_name);

  // Check if there's a direct FK relationship between two tables
  bool HasDirectFK(const std::string &from_table,
                   const std::string &to_table) const;

  // Get all FK relationships involving a table
  std::vector<ForeignKey> GetForeignKeys(const std::string &table) const;

  // Print the FK graph for debugging
  void Print() const;

private:
  // Map: FK table -> list of FKs from that table
  std::map<std::string, std::vector<ForeignKey>> fk_from_table_;

  // Map: PK table -> list of FKs to that table
  std::map<std::string, std::vector<ForeignKey>> fk_to_table_;
};

class ForeignKeyExtractor {
public:
  explicit ForeignKeyExtractor(DBAdapter *adapter, BackendEngine engine,
                               const std::string &fkeys_path = "")
      : adapter_(adapter), engine_(engine), fkeys_path_(fkeys_path) {}

  // Extract foreign keys for all tables in the given set
  ForeignKeyGraph ExtractForTables(const std::set<std::string> &table_names);

  // Extract all foreign keys from the database
  ForeignKeyGraph ExtractAll();

private:
  // Engine-specific FK extraction
  std::vector<ForeignKey>
  ExtractFromDuckDB(const std::set<std::string> &table_names);
  std::vector<ForeignKey>
  ExtractFromPostgreSQL(const std::set<std::string> &table_names);

  // MariaDB FK extraction via information_schema
  std::vector<ForeignKey>
  ExtractFromMariaDB(const std::set<std::string> &table_names);

  // Parse FK constraints from an SQL file (ALTER TABLE ... ADD FOREIGN KEY)
  std::vector<ForeignKey>
  ExtractFromFile(const std::set<std::string> &table_names);

  DBAdapter *adapter_;
  BackendEngine engine_;
  std::string fkeys_path_;
};

} // namespace middleware
