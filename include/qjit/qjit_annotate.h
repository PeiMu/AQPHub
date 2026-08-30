#pragma once

#include "simplest_ir.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace middleware {
namespace storage { class StoragePlan; }

void CollectIRJoinNodes(ir_sql_converter::AQPStmt *ir,
                        std::vector<ir_sql_converter::SimplestJoin *> &out);

uint64_t EstimateSubtreeCard(
    const ir_sql_converter::AQPStmt *node,
    const storage::StoragePlan *sp,
    const std::unordered_map<std::string, int64_t> &temp_card);

void AnnotateUnannotatedJoinsByCard(
    ir_sql_converter::AQPStmt &ir,
    const storage::StoragePlan *sp,
    const std::unordered_map<std::string, int64_t> &temp_card);

void CollectTableNames(
    const ir_sql_converter::AQPStmt &node,
    std::unordered_map<unsigned int, std::string> &table_names);

std::string TruncateIdentifier(const std::string &name);

std::string IrColumnAlias(
    const ir_sql_converter::SimplestAttr &attr,
    const std::unordered_map<unsigned int, std::string> &table_names);

void IrTargetListToDtypes(const ir_sql_converter::AQPStmt &ir,
                          std::vector<int32_t> &dtypes,
                          std::vector<std::string> &col_names);

} // namespace middleware
