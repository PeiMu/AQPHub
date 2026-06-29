#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace middleware {
namespace storage {

struct FlatTable;

// Inverted index: maps a dimension PK value to target PK values through a
// bridge (FK) table. E.g., keyword_id → [movie_id] via movie_keyword.
//
// Built from an existing CSR on bridge_table.fk_column (which indexes by
// dim PK) and reading bridge_table.target_column for each matched row.
//
// Layout is CSR-like:
//   row_ptr[dim_pk_val]     = start offset in target_vals
//   row_ptr[dim_pk_val + 1] = end offset
//   target_vals[start..end] = sorted, deduplicated target PK values
struct InvertedIndex {
  std::string dim_table;      // e.g., "keyword"
  std::string bridge_table;   // e.g., "movie_keyword"
  std::string bridge_fk_col;  // e.g., "keyword_id" (FK → dim PK)
  std::string target_col;     // e.g., "movie_id" (result values)
  std::string target_table;   // e.g., "title" (table whose PK = target_col values)

  std::unique_ptr<uint64_t[]> row_ptr;
  uint64_t row_ptr_size = 0;

  std::unique_ptr<int32_t[]> target_vals;
  uint64_t target_vals_size = 0;

  inline std::pair<const int32_t *, const int32_t *>
  Lookup(int32_t dim_pk_val) const {
    if (dim_pk_val < 0 || static_cast<uint64_t>(dim_pk_val) >= row_ptr_size - 1)
      return {nullptr, nullptr};
    uint64_t begin = row_ptr[dim_pk_val];
    uint64_t end = row_ptr[dim_pk_val + 1];
    return {target_vals.get() + begin, target_vals.get() + end};
  }

  uint64_t GetMemoryUsage() const {
    return row_ptr_size * sizeof(uint64_t) +
           target_vals_size * sizeof(int32_t);
  }
};

// Build an inverted index from an existing CSR and the bridge table's flat data.
// csr_row_ptr/csr_col_idx: the CSR on bridge_table.bridge_fk_col (indexes by dim PK).
// bridge_flat: the bridge FlatTable (to read target_column values).
// target_col_idx: column index of the target column in bridge_flat.
// dim_domain: row_ptr_size of the CSR (= dim_max_pk + 2).
InvertedIndex BuildInvertedIndex(
    const uint64_t *csr_row_ptr, const uint32_t *csr_col_idx,
    uint64_t dim_domain,
    const FlatTable &bridge_flat, int target_col_idx,
    const std::string &dim_table, const std::string &bridge_table,
    const std::string &bridge_fk_col, const std::string &target_col,
    const std::string &target_table);

} // namespace storage
} // namespace middleware
