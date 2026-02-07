/*
 * Implementation of foreign key extraction
 */

#include "split/foreign_key_extractor.h"
#include <iostream>
#include <sstream>

namespace middleware {

// ===== ForeignKeyGraph Implementation =====

void ForeignKeyGraph::AddForeignKey(const ForeignKey &fk) {
  fk_from_table_[fk.fk_table].push_back(fk);
  fk_to_table_[fk.pk_table].push_back(fk);
}

std::vector<std::string>
ForeignKeyGraph::GetReferencingTables(const std::string &table) const {
  std::vector<std::string> result;
  auto it = fk_to_table_.find(table);
  if (it != fk_to_table_.end()) {
    for (const auto &fk : it->second) {
      result.push_back(fk.fk_table);
    }
  }
  return result;
}

std::vector<std::string>
ForeignKeyGraph::GetReferencedTables(const std::string &table) const {
  std::vector<std::string> result;
  auto it = fk_from_table_.find(table);
  if (it != fk_from_table_.end()) {
    for (const auto &fk : it->second) {
      result.push_back(fk.pk_table);
    }
  }
  return result;
}

bool ForeignKeyGraph::HasDirectFK(const std::string &from_table,
                                  const std::string &to_table) const {
  auto it = fk_from_table_.find(from_table);
  if (it != fk_from_table_.end()) {
    for (const auto &fk : it->second) {
      if (fk.pk_table == to_table) {
        return true;
      }
    }
  }
  return false;
}

std::vector<ForeignKey>
ForeignKeyGraph::GetForeignKeys(const std::string &table) const {
  std::vector<ForeignKey> result;

  // Add FKs from this table
  auto from_it = fk_from_table_.find(table);
  if (from_it != fk_from_table_.end()) {
    result.insert(result.end(), from_it->second.begin(), from_it->second.end());
  }

  return result;
}

void ForeignKeyGraph::Print() const {
  std::cout << "=== Foreign Key Graph ===" << std::endl;
  for (const auto &[table, fks] : fk_from_table_) {
    std::cout << "Table: " << table << std::endl;
    for (const auto &fk : fks) {
      std::cout << "  " << fk.fk_table << "." << fk.fk_column << " -> "
                << fk.pk_table << "." << fk.pk_column << std::endl;
    }
  }
  std::cout << "=========================" << std::endl;
}

// ===== ForeignKeyExtractor Implementation =====

ForeignKeyGraph ForeignKeyExtractor::ExtractForTables(
    const std::set<std::string> &table_names) {

#ifndef NDEBUG
  std::cout << "[ForeignKeyExtractor] Extracting foreign keys for "
            << table_names.size() << " table(s)" << std::endl;
#endif

  std::vector<ForeignKey> fks;

  if (engine_ == BackendEngine::DUCKDB) {
    fks = ExtractFromDuckDB(table_names);
  } else if (engine_ == BackendEngine::POSTGRESQL) {
    fks = ExtractFromPostgreSQL(table_names);
  }

  ForeignKeyGraph graph;
  for (const auto &fk : fks) {
    graph.AddForeignKey(fk);
  }

#ifndef NDEBUG
  std::cout << "[ForeignKeyExtractor] Found " << fks.size()
            << " foreign key relationship(s)" << std::endl;
#endif

  return graph;
}

ForeignKeyGraph ForeignKeyExtractor::ExtractAll() {
#ifndef NDEBUG
  std::cout << "[ForeignKeyExtractor] Extracting all foreign keys from database"
            << std::endl;
#endif

  // Pass empty set to extract all FKs
  return ExtractForTables(std::set<std::string>{});
}

std::vector<ForeignKey> ForeignKeyExtractor::ExtractFromDuckDB(
    const std::set<std::string> &table_names) {

  std::vector<ForeignKey> fks;

  // DuckDB stores FK constraints in duckdb_constraints() table function
  std::string query = R"(
   SELECT
     table_name as fk_table,
     constraint_column_names[1] as fk_column,
     constraint_text as constraint_def
   FROM duckdb_constraints()
   WHERE constraint_type = 'FOREIGN KEY'
 )";

  // Add table filter if specific tables requested
  if (!table_names.empty()) {
    std::ostringstream table_list;
    bool first = true;
    for (const auto &table : table_names) {
      if (!first)
        table_list << ", ";
      table_list << "'" << table << "'";
      first = false;
    }
    query += " AND table_name IN (" + table_list.str() + ")";
  }

  try {
    auto result = adapter_->ExecuteSQL(query);

    for (const auto &row : result.rows) {
      if (row.size() >= 3) {
        std::string fk_table = row[0];
        std::string fk_column = row[1];
        std::string constraint_text = row[2];

        // Parse constraint text to extract referenced table and column
        // Format: "FOREIGN KEY (column) REFERENCES table(column)"
        // This is simplified - real implementation needs robust parsing

        size_t ref_pos = constraint_text.find("REFERENCES");
        if (ref_pos != std::string::npos) {
          std::string ref_part = constraint_text.substr(ref_pos + 10);
          size_t paren_pos = ref_part.find('(');
          if (paren_pos != std::string::npos) {
            std::string pk_table = ref_part.substr(0, paren_pos);
            // Trim whitespace
            pk_table.erase(0, pk_table.find_first_not_of(" \t"));
            pk_table.erase(pk_table.find_last_not_of(" \t") + 1);

            size_t close_paren = ref_part.find(')', paren_pos);
            std::string pk_column =
                ref_part.substr(paren_pos + 1, close_paren - paren_pos - 1);

            fks.emplace_back(fk_table, fk_column, pk_table, pk_column);
          }
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[ForeignKeyExtractor] Error extracting FKs from DuckDB: "
              << e.what() << std::endl;
  }

  return fks;
}

std::vector<ForeignKey> ForeignKeyExtractor::ExtractFromPostgreSQL(
    const std::set<std::string> &table_names) {

  std::vector<ForeignKey> fks;

  // PostgreSQL stores FK constraints in information_schema
  std::string query = R"(
   SELECT
     tc.table_name as fk_table,
     kcu.column_name as fk_column,
     ccu.table_name AS pk_table,
     ccu.column_name AS pk_column
   FROM information_schema.table_constraints AS tc
   JOIN information_schema.key_column_usage AS kcu
     ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema = kcu.table_schema
   JOIN information_schema.constraint_column_usage AS ccu
     ON ccu.constraint_name = tc.constraint_name
     AND ccu.table_schema = tc.table_schema
   WHERE tc.constraint_type = 'FOREIGN KEY'
 )";

  // Add table filter if specific tables requested
  if (!table_names.empty()) {
    std::ostringstream table_list;
    bool first = true;
    for (const auto &table : table_names) {
      if (!first)
        table_list << ", ";
      table_list << "'" << table << "'";
      first = false;
    }
    query += " AND tc.table_name IN (" + table_list.str() + ")";
  }

  try {
    auto result = adapter_->ExecuteSQL(query);

    for (const auto &row : result.rows) {
      if (row.size() >= 4) {
        fks.emplace_back(row[0], row[1], row[2], row[3]);
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[ForeignKeyExtractor] Error extracting FKs from PostgreSQL: "
              << e.what() << std::endl;
  }

  return fks;
}

} // namespace middleware
