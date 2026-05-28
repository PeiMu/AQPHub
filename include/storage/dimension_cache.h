#pragma once

#include "storage/flat_table.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace ir_sql_converter {
class AQPExpr;
}

namespace middleware {
namespace storage {

class DimensionCache {
public:
  static constexpr uint64_t MAX_DIM_ROWS = 200;

  void Build(const std::unordered_map<std::string, FlatTable> &tables);

  bool IsDimension(const std::string &table_name) const;

  const FlatTable *GetDimTable(const std::string &table_name) const;

  // Resolve filter predicates on a dimension table to matching PK ("id") values.
  // filters: pointers to qual_vec vectors from IR nodes (ScanNode, FilterNode).
  // Returns matching PK values, or empty vector if resolution fails.
  std::vector<int32_t> ResolveFilterToPKs(
      const std::string &table_name,
      const std::vector<
          const std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> *>
          &filters) const;

  void PrintSummary() const;

private:
  std::unordered_map<std::string, const FlatTable *> dim_tables_;
};

} // namespace storage
} // namespace middleware
