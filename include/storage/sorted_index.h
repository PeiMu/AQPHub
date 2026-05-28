#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace middleware {
namespace storage {

struct FlatTable;

struct SortedIndex {
  std::string table_name;
  std::string column_name;
  // sorted_perm[0] = row index with smallest value, [1] = next smallest, etc.
  // NULL rows are excluded.
  std::vector<uint32_t> sorted_perm;
};

SortedIndex BuildSortedIndex(const FlatTable &table,
                             const std::string &col_name);

} // namespace storage
} // namespace middleware
