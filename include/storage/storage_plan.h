#pragma once

#include "storage/csr_index.h"
#include "storage/dimension_cache.h"
#include "storage/flat_table.h"
#include "storage/sorted_index.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {
class Connection;
}

namespace middleware {

namespace storage {

class StoragePlan {
public:
  // Load all base tables from DuckDB into flat column arrays.
  // Called once at startup after DuckDB has loaded data.
  void LoadFromDuckDB(duckdb::Connection &conn);

  // Build CSR indexes on all FK columns declared in fkeys.sql.
  void BuildCSRIndexes(const std::string &fkeys_path);

  // Serialize entire storage plan (flat tables + CSR indexes) to a binary file.
  void SaveToFile(const std::string &path) const;

  // Deserialize storage plan from a binary file created by SaveToFile.
  // Returns false if the file doesn't exist or is corrupt.
  bool LoadFromFile(const std::string &path);

  const FlatTable *GetTable(const std::string &table_name) const;

  // Lookup CSR index by "fk_table.fk_column"
  const CSRIndex *GetCSR(const std::string &fk_table,
                         const std::string &fk_column) const;

  uint64_t GetMemoryUsage() const;
  void PrintSummary() const;
  bool IsLoaded() const { return loaded_; }

  const std::unordered_map<std::string, FlatTable> &GetTables() const {
    return tables_;
  }

  const DimensionCache *GetDimensionCache() const {
    return loaded_ ? &dim_cache_ : nullptr;
  }

  void BuildSortedIndices();
  const SortedIndex *GetSortedIndex(const std::string &table_name,
                                    const std::string &col_name) const;

private:
  bool loaded_ = false;
  std::unordered_map<std::string, FlatTable> tables_;
  // Key: "fk_table.fk_column"
  std::unordered_map<std::string, CSRIndex> csr_indexes_;
  DimensionCache dim_cache_;
  // Key: "table_name.column_name"
  std::unordered_map<std::string, SortedIndex> sorted_indices_;
};

} // namespace storage
} // namespace middleware
