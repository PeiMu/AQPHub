#include "storage/inverted_index.h"
#include "storage/flat_table.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace middleware {
namespace storage {

InvertedIndex BuildInvertedIndex(
    const uint64_t *csr_row_ptr, const uint32_t *csr_col_idx,
    uint64_t dim_domain,
    const FlatTable &bridge_flat, int target_col_idx,
    const std::string &dim_table, const std::string &bridge_table,
    const std::string &bridge_fk_col, const std::string &target_col,
    const std::string &target_table) {

  InvertedIndex idx;
  idx.dim_table = dim_table;
  idx.bridge_table = bridge_table;
  idx.bridge_fk_col = bridge_fk_col;
  idx.target_col = target_col;
  idx.target_table = target_table;
  idx.row_ptr_size = dim_domain;
  idx.row_ptr = std::make_unique<uint64_t[]>(dim_domain);

  const auto &col = bridge_flat.columns[target_col_idx];
  const auto *target_data = reinterpret_cast<const int32_t *>(col.data.get());

  // Pass 1: For each dim PK value, collect target values from CSR matches,
  // sort and deduplicate, count unique entries.
  // We build a temporary vector of vectors, then flatten into CSR.
  uint64_t num_dim_vals = dim_domain > 1 ? dim_domain - 1 : 0;

  // Count total target values first (upper bound before dedup)
  uint64_t total_raw = 0;
  for (uint64_t d = 0; d < num_dim_vals; d++) {
    total_raw += csr_row_ptr[d + 1] - csr_row_ptr[d];
  }

  // Build per-dim target lists, sort, dedup, then flatten
  std::vector<int32_t> all_targets;
  all_targets.reserve(total_raw);

  std::vector<int32_t> tmp;
  uint64_t offset = 0;

  for (uint64_t d = 0; d < num_dim_vals; d++) {
    idx.row_ptr[d] = offset;
    uint64_t begin = csr_row_ptr[d];
    uint64_t end = csr_row_ptr[d + 1];
    if (begin == end)
      continue;

    tmp.clear();
    for (uint64_t i = begin; i < end; i++) {
      uint32_t bridge_row = csr_col_idx[i];
      int32_t val = target_data[bridge_row];
      tmp.push_back(val);
    }

    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());

    all_targets.insert(all_targets.end(), tmp.begin(), tmp.end());
    offset += tmp.size();
  }

  // Sentinel
  if (num_dim_vals < dim_domain)
    idx.row_ptr[num_dim_vals] = offset;

  idx.target_vals_size = offset;
  idx.target_vals = std::make_unique<int32_t[]>(offset);
  std::memcpy(idx.target_vals.get(), all_targets.data(),
              offset * sizeof(int32_t));

  return idx;
}

} // namespace storage
} // namespace middleware
