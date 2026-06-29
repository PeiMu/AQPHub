#pragma once

#include "storage/flat_table.h"
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace middleware {
namespace storage {

struct CSRIndex {
  std::string fk_table;
  std::string fk_column;
  std::string pk_table;
  std::string pk_column;

  // row_ptr[v] = first position in col_idx for PK value v
  // row_ptr[v+1] = one-past-last position
  // Size: pk_domain_max + 2
  std::unique_ptr<uint64_t[]> row_ptr;
  uint64_t row_ptr_size = 0;

  // col_idx[i] = row index in fk_table where FK column == some PK value
  std::unique_ptr<uint32_t[]> col_idx;
  uint64_t col_idx_size = 0;

  inline std::pair<const uint32_t *, const uint32_t *>
  Lookup(int32_t pk_val) const {
    if (pk_val < 0 || static_cast<uint64_t>(pk_val) >= row_ptr_size - 1)
      return {nullptr, nullptr};
    uint64_t begin = row_ptr[pk_val];
    uint64_t end = row_ptr[pk_val + 1];
    return {col_idx.get() + begin, col_idx.get() + end};
  }

  uint64_t GetMemoryUsage() const {
    return row_ptr_size * sizeof(uint64_t) +
           col_idx_size * sizeof(uint32_t);
  }
};

// Build CSR index from a FlatTable's FK column.
// fk_col_idx: column index in fk_table containing FK values
// pk_domain_max: maximum PK value (determines row_ptr size)
CSRIndex BuildCSR(const FlatTable &fk_table, int fk_col_idx,
                  int32_t pk_domain_max, const std::string &fk_table_name,
                  const std::string &fk_col_name,
                  const std::string &pk_table_name,
                  const std::string &pk_col_name);

} // namespace storage
} // namespace middleware
