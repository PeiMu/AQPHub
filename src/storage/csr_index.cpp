#include "storage/csr_index.h"
#include <cstring>
#include <vector>

namespace middleware {
namespace storage {

CSRIndex BuildCSR(const FlatTable &fk_table, int fk_col_idx,
                  int32_t pk_domain_max, const std::string &fk_table_name,
                  const std::string &fk_col_name,
                  const std::string &pk_table_name,
                  const std::string &pk_col_name) {
  CSRIndex csr;
  csr.fk_table = fk_table_name;
  csr.fk_column = fk_col_name;
  csr.pk_table = pk_table_name;
  csr.pk_column = pk_col_name;

  uint64_t domain_size = static_cast<uint64_t>(pk_domain_max) + 2;
  csr.row_ptr_size = domain_size;
  csr.row_ptr = std::make_unique<uint64_t[]>(domain_size);
  std::memset(csr.row_ptr.get(), 0, domain_size * sizeof(uint64_t));

  const auto &fk_col = fk_table.columns[fk_col_idx];
  const auto *fk_data = reinterpret_cast<const int32_t *>(fk_col.data.get());

  // Pass 1: count occurrences of each FK value (skip NULLs)
  uint64_t non_null_count = 0;
  for (uint64_t i = 0; i < fk_table.row_count; i++) {
    if (fk_col.IsNull(i))
      continue;
    int32_t val = fk_data[i];
    if (val >= 0 && val <= pk_domain_max) {
      csr.row_ptr[val + 1]++;
      non_null_count++;
    }
  }

  // Pass 2: prefix sum
  for (uint64_t i = 1; i < domain_size; i++) {
    csr.row_ptr[i] += csr.row_ptr[i - 1];
  }

  // Allocate col_idx
  csr.col_idx_size = non_null_count;
  csr.col_idx = std::make_unique<uint32_t[]>(non_null_count);

  // Pass 3: scatter row IDs using working cursors
  std::vector<uint64_t> cursors(csr.row_ptr.get(),
                                csr.row_ptr.get() + domain_size);
  for (uint64_t i = 0; i < fk_table.row_count; i++) {
    if (fk_col.IsNull(i))
      continue;
    int32_t val = fk_data[i];
    if (val >= 0 && val <= pk_domain_max) {
      csr.col_idx[cursors[val]++] = static_cast<uint32_t>(i);
    }
  }

  return csr;
}

} // namespace storage
} // namespace middleware
