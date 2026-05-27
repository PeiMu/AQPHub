#include "storage/flat_table.h"
#include <algorithm>
#include <cctype>

namespace middleware {
namespace storage {

int FlatTable::FindColumn(const std::string &name) const {
  for (size_t i = 0; i < column_names.size(); i++) {
    if (column_names[i].size() != name.size())
      continue;
    bool match = true;
    for (size_t j = 0; j < name.size(); j++) {
      if (std::tolower(static_cast<unsigned char>(column_names[i][j])) !=
          std::tolower(static_cast<unsigned char>(name[j]))) {
        match = false;
        break;
      }
    }
    if (match)
      return static_cast<int>(i);
  }
  return -1;
}

} // namespace storage
} // namespace middleware
