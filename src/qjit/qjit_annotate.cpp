#include "qjit/qjit_annotate.h"
#include "jit/aqp_jit_abi.h"
#include "storage/storage_plan.h"

namespace middleware {

void CollectIRJoinNodes(ir_sql_converter::AQPStmt *ir,
                        std::vector<ir_sql_converter::SimplestJoin *> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode)
    out.push_back(static_cast<ir_sql_converter::SimplestJoin *>(ir));
  for (auto &child : ir->children)
    CollectIRJoinNodes(child.get(), out);
}

uint64_t EstimateSubtreeCard(
    const ir_sql_converter::AQPStmt *node,
    const storage::StoragePlan *sp,
    const std::unordered_map<std::string, int64_t> &temp_card) {
  if (!node) return 0;
  auto nt = node->GetNodeType();
  if (nt == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(node);
    if (sp) {
      const auto *ft = sp->GetTable(scan->GetTableName());
      if (ft) return ft->row_count;
    }
    return 1000000;
  }
  if (nt == ir_sql_converter::SimplestNodeType::ChunkNode) {
    auto *chunk = static_cast<const ir_sql_converter::SimplestChunk *>(node);
    auto it = temp_card.find(chunk->GetChunkName());
    if (it != temp_card.end()) return it->second;
    return 1000;
  }
  uint64_t card = 0;
  for (const auto &child : node->children) {
    uint64_t c = EstimateSubtreeCard(child.get(), sp, temp_card);
    card = (card == 0) ? c : std::min(card, c);
  }
  return card;
}

void AnnotateUnannotatedJoinsByCard(
    ir_sql_converter::AQPStmt &ir,
    const storage::StoragePlan *sp,
    const std::unordered_map<std::string, int64_t> &temp_card) {
  std::vector<ir_sql_converter::SimplestJoin *> joins;
  CollectIRJoinNodes(&ir, joins);
  for (auto *join : joins) {
    if (join->GetBuildChild() != -1 || join->children.size() != 2)
      continue;
    uint64_t c0 = EstimateSubtreeCard(join->children[0].get(), sp, temp_card);
    uint64_t c1 = EstimateSubtreeCard(join->children[1].get(), sp, temp_card);
    join->SetBuildChild(c1 <= c0 ? 1 : 0);
#ifndef NDEBUG
    std::cerr << "[AQP-QJIT] fallback build-side annotation: c0=" << c0
              << " c1=" << c1 << " -> build=" << join->GetBuildChild() << "\n";
#endif
  }
}

void CollectTableNames(
    const ir_sql_converter::AQPStmt &node,
    std::unordered_map<unsigned int, std::string> &table_names) {
  if (node.GetNodeType() == ir_sql_converter::ScanNode) {
    auto &scan = node.Cast<ir_sql_converter::SimplestScan>();
    table_names[scan.GetTableIndex()] = scan.GetTableName();
  } else if (node.GetNodeType() == ir_sql_converter::ChunkNode) {
    auto &chunk = node.Cast<ir_sql_converter::SimplestChunk>();
    table_names[chunk.GetTableIndex()] = chunk.GetChunkName();
  }
  for (const auto &child : node.children) {
    if (child)
      CollectTableNames(*child, table_names);
  }
}

std::string TruncateIdentifier(const std::string &name) {
  constexpr size_t kMaxLen = 63;
  if (name.size() <= kMaxLen)
    return name;
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : name) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  char buf[18];
  snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
  return std::string("c_") + buf;
}

std::string IrColumnAlias(
    const ir_sql_converter::SimplestAttr &attr,
    const std::unordered_map<unsigned int, std::string> &table_names) {
  unsigned int tidx = attr.GetTableIndex();
  std::string alias;
  auto it = table_names.find(tidx);
  if (it != table_names.end()) {
    alias = it->second + "_" + std::to_string(tidx) + "_" +
            attr.GetColumnName();
  } else {
    alias = "col_" + std::to_string(tidx) + "_" + attr.GetColumnName();
  }
  return TruncateIdentifier(alias);
}

void IrTargetListToDtypes(const ir_sql_converter::AQPStmt &ir,
                          std::vector<int32_t> &dtypes,
                          std::vector<std::string> &col_names) {
  for (const auto &attr : ir.target_list) {
    if (!attr)
      continue;
    int32_t dt;
    switch (attr->GetType()) {
    case ir_sql_converter::IntVar:
      dt = attr->GetBitWidth() == 64 ? AQP_DTYPE_INT64 : AQP_DTYPE_INT32;
      break;
    case ir_sql_converter::Date:
      dt = AQP_DTYPE_DATE;
      break;
    case ir_sql_converter::StringVar:
      dt = AQP_DTYPE_VARCHAR;
      break;
    case ir_sql_converter::FloatVar:
      dt = attr->GetBitWidth() == 64 ? AQP_DTYPE_DOUBLE : AQP_DTYPE_FLOAT;
      break;
    case ir_sql_converter::BoolVar:
      dt = AQP_DTYPE_BOOL;
      break;
    default:
      dt = AQP_DTYPE_VARCHAR;
      break;
    }
    dtypes.push_back(dt);
    col_names.push_back(attr->GetColumnName());
  }
}

} // namespace middleware
