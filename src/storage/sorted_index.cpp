#include "storage/sorted_index.h"
#include "storage/flat_table.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace middleware {
namespace storage {

SortedIndex BuildSortedIndex(const FlatTable &table,
                             const std::string &col_name) {
  SortedIndex idx;
  idx.table_name = table.table_name;
  idx.column_name = col_name;

  int col_pos = table.FindColumn(col_name);
  if (col_pos < 0)
    return idx;

  const auto &col = table.columns[col_pos];
  if (col.type != FlatColumnType::INT32 &&
      col.type != FlatColumnType::VARCHAR) {
    // The sort comparators below only understand INT32/VARCHAR layouts.
    throw std::runtime_error("BuildSortedIndex unsupported: column " +
                             table.table_name + "." + col_name +
                             " is neither INT32 nor VARCHAR");
  }

  idx.sorted_perm.reserve(col.row_count);
  for (uint64_t r = 0; r < col.row_count; r++) {
    if (!col.nullable || !col.IsNull(r))
      idx.sorted_perm.push_back(static_cast<uint32_t>(r));
  }

  if (col.type == FlatColumnType::INT32) {
    const auto *data = reinterpret_cast<const int32_t *>(col.data.get());
    std::sort(idx.sorted_perm.begin(), idx.sorted_perm.end(),
              [data](uint32_t a, uint32_t b) { return data[a] < data[b]; });
  } else {
    const auto *offsets =
        reinterpret_cast<const uint32_t *>(col.data.get());
    const char *pool = col.string_pool.get();
    std::sort(idx.sorted_perm.begin(), idx.sorted_perm.end(),
              [offsets, pool](uint32_t a, uint32_t b) {
                uint32_t a_off = offsets[a], a_end = offsets[a + 1];
                uint32_t b_off = offsets[b], b_end = offsets[b + 1];
                uint32_t a_len = a_end - a_off;
                uint32_t b_len = b_end - b_off;
                uint32_t min_len = a_len < b_len ? a_len : b_len;
                int cmp = std::memcmp(pool + a_off, pool + b_off, min_len);
                if (cmp != 0)
                  return cmp < 0;
                return a_len < b_len;
              });
  }

  return idx;
}

} // namespace storage
} // namespace middleware
