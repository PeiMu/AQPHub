#pragma once

#include "storage/csr_index.h"
#include "storage/dimension_cache.h"
#include "storage/flat_table.h"
#include "storage/inverted_index.h"
#include "storage/sorted_index.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace duckdb {
class Connection;
}

struct pg_conn;
typedef struct pg_conn PGconn;

namespace middleware {

namespace storage {

class StoragePlan {
public:
  // Load all base tables from DuckDB into flat column arrays.
  // Called once at startup after DuckDB has loaded data.
  void LoadFromDuckDB(duckdb::Connection &conn);

  // Load all base tables from PostgreSQL into flat column arrays.
  // Called once at startup; results cached via SaveToFile.
  void LoadFromPostgreSQL(PGconn *conn);

  // Build CSR indexes on all FK columns declared in fkeys.sql.
  void BuildCSRIndexes(const std::string &fkeys_path);

  // Serialize entire storage plan (flat tables + CSR indexes) to a binary file.
  void SaveToFile(const std::string &path) const;

  // Deserialize storage plan from a binary file created by SaveToFile.
  // Returns false if the file doesn't exist or is corrupt.
  // skip_indexes: seek past the CSR/sorted/inverted index sections instead of
  // materializing them (query-jit consumes FlatTables only; the index maps
  // stay empty). The file format is unchanged.
  bool LoadFromFile(const std::string &path, bool skip_indexes = false);

  const FlatTable *GetTable(const std::string &table_name) const;

  // Lookup CSR index by "fk_table.fk_column"
  const CSRIndex *GetCSR(const std::string &fk_table,
                         const std::string &fk_column) const;

  // Check if a base column name appears in any FK/PK relationship
  bool IsJoinKeyColumn(const std::string &col_name) const;

  uint64_t GetMemoryUsage() const;
  void PrintSummary() const;
  bool IsLoaded() const { return loaded_; }

  const std::unordered_map<std::string, FlatTable> &GetTables() const {
    return tables_;
  }

  const std::unordered_map<std::string, CSRIndex> &GetCSRMap() const {
    return csr_indexes_;
  }

  const DimensionCache *GetDimensionCache() const {
    return loaded_ ? &dim_cache_ : nullptr;
  }

  void BuildSortedIndices();
  const SortedIndex *GetSortedIndex(const std::string &table_name,
                                    const std::string &col_name) const;

  const std::unordered_map<std::string, SortedIndex> &
  GetSortedIndicesMap() const {
    return sorted_indices_;
  }

  void BuildInvertedIndices();

  // Lookup inverted index: given a dim table and the target table,
  // return the inverted index that maps dim_pk → target_pk values.
  // E.g., GetInvertedIndex("keyword", "title") returns keyword_id→movie_id.
  const InvertedIndex *GetInvertedIndex(const std::string &dim_table,
                                        const std::string &target_table) const;

  const std::unordered_map<std::string, InvertedIndex> &
  GetInvertedIndicesMap() const {
    return inverted_indices_;
  }

private:
  bool loaded_ = false;
  std::unordered_map<std::string, FlatTable> tables_;
  // Key: "fk_table.fk_column"
  std::unordered_map<std::string, CSRIndex> csr_indexes_;
  DimensionCache dim_cache_;
  // Key: "table_name.column_name"
  std::unordered_map<std::string, SortedIndex> sorted_indices_;
  // Key: "dim_table->target_table"
  std::unordered_map<std::string, InvertedIndex> inverted_indices_;
  // Base column names that appear in FK/PK relationships (e.g., "movie_id", "id")
  mutable std::unordered_set<std::string> join_key_cols_;
  mutable bool join_key_cols_built_ = false;
};

} // namespace storage
} // namespace middleware
