#include "adapters/lingodb_runtime_adapter.h"
#include "util/util.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>

#include <arrow/api.h>
#include <arrow/table.h>

#include <lingodb/catalog/Catalog.h>
#include <lingodb/catalog/Defs.h>
#include <lingodb/catalog/MLIRTypes.h>
#include <lingodb/catalog/TableCatalogEntry.h>
#include <lingodb/catalog/Types.h>
#include <lingodb/compiler/Dialect/DB/IR/DBDialect.h>
#include <lingodb/compiler/Dialect/DB/IR/DBOps.h>
#include <lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h>
#include <lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h>
#include <lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h>
#include <lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h>
#include <lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h>
#include <lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h>
#include <lingodb/compiler/Dialect/RelAlg/Passes.h>
#include <lingodb/compiler/helper.h>
#include <lingodb/execution/Execution.h>
#include <lingodb/execution/Frontend.h>
#include <lingodb/execution/ResultProcessing.h>
#include <lingodb/execution/Timing.h>
#include <lingodb/runtime/DatasourceRestrictionProperty.h>
#include <lingodb/runtime/RelationHelper.h>
#include <lingodb/runtime/Session.h>
#include <lingodb/scheduler/Scheduler.h>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

namespace relalg = lingodb::compiler::dialect::relalg;
namespace db = lingodb::compiler::dialect::db;
namespace tuples = lingodb::compiler::dialect::tuples;
namespace subop = lingodb::compiler::dialect::subop;

namespace middleware {
namespace {

class IRToRelAlgConverter {
public:
  IRToRelAlgConverter(mlir::MLIRContext &ctx,
                      lingodb::catalog::Catalog &catalog,
                      lingodb::runtime::Session *session = nullptr)
      : ctx_(ctx), catalog_(catalog), session_(session),
        colMgr_(ctx.getLoadedDialect<tuples::TupleStreamDialect>()
                    ->getColumnManager()),
        memberMgr_(ctx.getLoadedDialect<subop::SubOperatorDialect>()
                       ->getMemberManager()) {}

  mlir::ModuleOp convert(ir_sql_converter::AQPStmt &ir) {
    auto loc = mlir::UnknownLoc::get(&ctx_);
    mlir::OpBuilder builder(&ctx_);

    auto moduleOp = builder.create<mlir::ModuleOp>(loc);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto *queryInnerBlock = new mlir::Block;
    subop::LocalTableType localTableType;
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(queryInnerBlock);

      mlir::Value tree = convertNode(builder, ir);

      llvm::SmallVector<mlir::Attribute> colRefAttrs;
      llvm::SmallVector<mlir::Attribute> colNameAttrs;
      llvm::SmallVector<subop::Member> members;

      for (auto &attr : ir.target_list) {
        auto scope = resolveScope(attr->GetTableIndex());
        auto colName = attr->GetColumnName();
        auto colRef = resolveColRef(scope, colName);
        auto mlirType = colRef.getColumn().type;

        // If this column was remapped by a projection, use the actual
        // stream column ref instead (the projection alias doesn't exist
        // in the tuple stream)
        std::string key = scope + "::" + colName;
        auto mapIt = projToStreamMap_.find(key);
        if (mapIt != projToStreamMap_.end()) {
          colRef = mapIt->second;
          mlirType = colRef.getColumn().type;
        }

        if (!mlirType) {
          throw std::runtime_error(
              "[IRToRelAlg] Column type is null for scope=" + scope +
              " colName=" + colName +
              " tableIdx=" + std::to_string(attr->GetTableIndex()));
        }

        auto tableName = findTableName(attr->GetTableIndex(), ir);
        std::string alias;
        if (!tableName.empty())
          alias = tableName + "_" + std::to_string(attr->GetTableIndex()) +
                  "_" + colName;
        else
          alias = "t" + std::to_string(attr->GetTableIndex()) + "_" + colName;

        colRefAttrs.push_back(colRef);
        colNameAttrs.push_back(builder.getStringAttr(alias));
        members.push_back(memberMgr_.createMember(alias, mlirType));
      }

      localTableType = subop::LocalTableType::get(
          &ctx_, subop::StateMembersAttr::get(&ctx_, members),
          builder.getArrayAttr(colNameAttrs));

      auto matOp = builder.create<relalg::MaterializeOp>(
          loc, localTableType, tree, builder.getArrayAttr(colRefAttrs),
          builder.getArrayAttr(colNameAttrs));
      builder.create<relalg::QueryReturnOp>(loc, matOp.getResult());
    }

    auto *funcBlock = new mlir::Block;
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(funcBlock);

      auto queryOp = builder.create<relalg::QueryOp>(
          loc, mlir::TypeRange{localTableType}, mlir::ValueRange{});
      queryOp.getQueryOps().getBlocks().clear();
      queryOp.getQueryOps().push_back(queryInnerBlock);

      builder.create<subop::SetResultOp>(loc, 0,
                                         queryOp.getResults()[0]);
      builder.create<mlir::func::ReturnOp>(loc);
    }

    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, "main", builder.getFunctionType({}, {}));
    funcOp.getBody().push_back(funcBlock);

    return moduleOp;
  }

private:
  mlir::MLIRContext &ctx_;
  lingodb::catalog::Catalog &catalog_;
  lingodb::runtime::Session *session_;
  tuples::ColumnManager &colMgr_;
  subop::MemberManager &memberMgr_;

  // table_index -> scope name
  std::unordered_map<unsigned int, std::string> scopeMap_;
  // table_index -> catalog table name
  std::unordered_map<unsigned int, std::string> tableNameMap_;
  // Output columns from the last converted aggregate (for projection mapping)
  std::vector<tuples::ColumnRefAttr> lastAggOutputRefs_;
  // Mapping: projection output column ref -> actual stream column ref
  std::unordered_map<std::string, tuples::ColumnRefAttr> projToStreamMap_;
  // table indices that are mark join outputs (always single column "col0")
  std::set<unsigned int> markIndices_;

  std::string resolveScope(unsigned int tableIndex) const {
    auto it = scopeMap_.find(tableIndex);
    if (it != scopeMap_.end())
      return it->second;
    return "unknown_" + std::to_string(tableIndex);
  }

  std::string resolveColumnName(unsigned int tableIndex,
                                unsigned int colIndex,
                                const std::string &irName) {
    if (markIndices_.count(tableIndex))
      return "col0";
    if (!irName.empty())
      return irName;
    auto it = tableNameMap_.find(tableIndex);
    if (it == tableNameMap_.end())
      return "col" + std::to_string(colIndex);
    auto entry =
        catalog_.getTypedEntry<lingodb::catalog::TableCatalogEntry>(it->second);
    if (!entry.has_value())
      return "col" + std::to_string(colIndex);
    auto cols = entry.value()->getColumns();
    if (colIndex < cols.size())
      return std::string(cols[colIndex].getColumnName());
    return "col" + std::to_string(colIndex);
  }

  // Resolve column ref with fallback: if the column isn't defined in the
  // primary scope (DuckDB sometimes mis-attributes columns after
  // optimization), search all known scopes for a matching column name.
  tuples::ColumnRefAttr resolveColRef(const std::string &scope,
                                      const std::string &colName) {
    auto ref = colMgr_.createRef(scope, colName);
    if (ref.getColumn().type)
      return ref;
    // Column not in scope — check projToStreamMap_
    std::string key = scope + "::" + colName;
    auto mapIt = projToStreamMap_.find(key);
    if (mapIt != projToStreamMap_.end() && mapIt->second.getColumn().type)
      return mapIt->second;
    // Search all other scopes
    for (auto &[idx, otherScope] : scopeMap_) {
      if (otherScope == scope)
        continue;
      auto otherRef = colMgr_.createRef(otherScope, colName);
      if (otherRef.getColumn().type)
        return otherRef;
    }
    return ref; // return original (null type) if not found anywhere
  }

  mlir::Type getColumnMLIRType(const std::string &tableName,
                               const std::string &colName) {
    auto entry =
        catalog_
            .getTypedEntry<lingodb::catalog::TableCatalogEntry>(tableName);
    if (entry.has_value()) {
      for (auto &col : entry.value()->getColumns()) {
        if (col.getColumnName() == colName) {
          auto baseType =
              col.getLogicalType().getMLIRTypeCreator()->createType(&ctx_);
          if (col.getIsNullable())
            return db::NullableType::get(&ctx_, baseType);
          return baseType;
        }
      }
    }
    return db::NullableType::get(&ctx_,
                                 db::StringType::get(&ctx_));
  }

  mlir::Type irTypeToMLIR(const ir_sql_converter::SimplestAttr &attr) {
    switch (attr.GetType()) {
    case ir_sql_converter::SimplestVarType::IntVar: {
      unsigned bw = attr.GetBitWidth();
      if (bw == 0 || bw == 64)
        return db::NullableType::get(&ctx_,
                                     mlir::IntegerType::get(&ctx_, 64));
      return db::NullableType::get(
          &ctx_, mlir::IntegerType::get(&ctx_, bw));
    }
    case ir_sql_converter::SimplestVarType::FloatVar:
      return db::NullableType::get(&ctx_, mlir::Float64Type::get(&ctx_));
    case ir_sql_converter::SimplestVarType::StringVar:
      return db::NullableType::get(&ctx_, db::StringType::get(&ctx_));
    case ir_sql_converter::SimplestVarType::Date:
      return db::NullableType::get(&ctx_,
                                   mlir::IntegerType::get(&ctx_, 32));
    default:
      return db::NullableType::get(&ctx_, db::StringType::get(&ctx_));
    }
  }

  // Resolve MLIR type for a column attribute: catalog first, fallback to IR
  mlir::Type resolveColumnType(const ir_sql_converter::SimplestAttr &attr,
                               const std::string &tableName) {
    auto entry =
        catalog_
            .getTypedEntry<lingodb::catalog::TableCatalogEntry>(tableName);
    if (entry.has_value()) {
      for (auto &col : entry.value()->getColumns()) {
        if (col.getColumnName() == attr.GetColumnName()) {
          auto baseType =
              col.getLogicalType().getMLIRTypeCreator()->createType(&ctx_);
          if (col.getIsNullable())
            return db::NullableType::get(&ctx_, baseType);
          return baseType;
        }
      }
    }
    return irTypeToMLIR(attr);
  }

  // Find the table name for a given table_index
  std::string findTableName(unsigned int tableIndex,
                            const ir_sql_converter::AQPStmt &root) {
    // Walk the tree to find a Scan or Chunk with the matching index
    if (auto *scan =
            dynamic_cast<const ir_sql_converter::SimplestScan *>(&root)) {
      if (scan->GetTableIndex() == tableIndex)
        return scan->GetTableName();
    }
    if (auto *chunk =
            dynamic_cast<const ir_sql_converter::SimplestChunk *>(&root)) {
      if (chunk->GetTableIndex() == tableIndex)
        return chunk->GetChunkName();
    }
    for (auto &child : root.children) {
      auto name = findTableName(tableIndex, *child);
      if (!name.empty())
        return name;
    }
    return "";
  }

  // ============== Node Conversion ==============

  mlir::Value convertNode(mlir::OpBuilder &builder,
                          ir_sql_converter::AQPStmt &node) {
    auto loc = builder.getUnknownLoc();
    auto nt = node.GetNodeType();
    using NT = ir_sql_converter::SimplestNodeType;
    switch (nt) {
    case NT::ScanNode:
      return convertScan(builder,
                         static_cast<ir_sql_converter::SimplestScan &>(node));
    case NT::ChunkNode:
      return convertChunk(
          builder, static_cast<ir_sql_converter::SimplestChunk &>(node));
    case NT::FilterNode: {
      auto &filter =
          static_cast<ir_sql_converter::SimplestFilter &>(node);
      // Detect pattern: Filter[mark_col] -> MarkJoin => convert to SemiJoin
      if (isMarkFilterPattern(filter)) {
        return convertMarkFilterAsSemiJoin(builder, filter);
      }
      mlir::Value child = convertNode(builder, *filter.children[0]);
      return convertFilter(builder, filter, child);
    }
    case NT::JoinNode:
      return convertJoin(
          builder, static_cast<ir_sql_converter::SimplestJoin &>(node));
    case NT::CrossProductNode:
      return convertCrossProduct(
          builder,
          static_cast<ir_sql_converter::SimplestCrossProduct &>(node));
    case NT::ProjectionNode: {
      auto &proj =
          static_cast<ir_sql_converter::SimplestProjection &>(node);
      mlir::Value child = convertNode(builder, *proj.children[0]);
      return convertProjection(builder, proj, child);
    }
    case NT::AggregateNode: {
      auto &agg =
          static_cast<ir_sql_converter::SimplestAggregate &>(node);
      mlir::Value child = convertNode(builder, *agg.children[0]);
      return convertAggregate(builder, agg, child);
    }
    case NT::OrderNode: {
      auto &order =
          static_cast<ir_sql_converter::SimplestOrderBy &>(node);
      mlir::Value child = convertNode(builder, *order.children[0]);
      return convertOrderBy(builder, order, child);
    }
    case NT::LimitNode: {
      auto &limit =
          static_cast<ir_sql_converter::SimplestLimit &>(node);
      mlir::Value child = convertNode(builder, *limit.children[0]);
      return convertLimit(builder, limit, child);
    }
    case NT::SortNode: {
      // SimplestSort is also used for ORDER BY in some paths
      mlir::Value child = convertNode(builder, *node.children[0]);
      return child; // pass-through: sort info handled by OrderNode
    }
    case NT::StmtNode:
      // Generic statement wrapper — drill to child
      if (!node.children.empty())
        return convertNode(builder, *node.children[0]);
      throw std::runtime_error(
          "[IRToRelAlg] StmtNode with no children");
    default:
      throw std::runtime_error(
          "[IRToRelAlg] Unsupported IR node type: " +
          std::to_string(static_cast<int>(nt)));
    }
  }

  std::vector<mlir::NamedAttribute> buildAllCatalogColumns(
      mlir::OpBuilder &builder, const std::string &tableName,
      const std::string &scope) {
    std::vector<mlir::NamedAttribute> columns;
    auto entry =
        catalog_
            .getTypedEntry<lingodb::catalog::TableCatalogEntry>(tableName);
    if (!entry.has_value())
      throw std::runtime_error("[IRToRelAlg] Table not in catalog: " +
                               tableName);

    auto catalogCols = entry.value()->getColumns();
    for (auto &cc : catalogCols) {
      auto colName = cc.getColumnName();
      auto attrDef = colMgr_.createDef(scope, colName);
      auto baseType =
          cc.getLogicalType().getMLIRTypeCreator()->createType(&ctx_);
      mlir::Type colType =
          cc.getIsNullable()
              ? (mlir::Type)db::NullableType::get(&ctx_, baseType)
              : baseType;
      attrDef.getColumn().type = colType;
      columns.push_back(builder.getNamedAttr(colName, attrDef));
    }
    return columns;
  }

  void createInlineTable(const std::string &tableName,
                         const std::vector<std::string> &contents) {
    // Create a single-column string table in the catalog
    lingodb::catalog::CreateTableDef def;
    def.name = tableName;
    def.columns.emplace_back("col0", lingodb::catalog::Type::stringType(),
                             true);

    auto entry =
        lingodb::catalog::LingoDBTableCatalogEntry::createFromCreateTable(def);
    catalog_.insertEntry(entry, true);

    // Build Arrow table with the contents
    auto strBuilder = std::make_shared<arrow::StringBuilder>();
    for (auto &v : contents) {
      (void)strBuilder->Append(v);
    }
    std::shared_ptr<arrow::Array> arr;
    (void)strBuilder->Finish(&arr);
    auto schema = arrow::schema({arrow::field("col0", arrow::utf8(), true)});
    auto table = arrow::Table::Make(schema, {arr});

    lingodb::runtime::RelationHelper::appendToTable(
        *session_, tableName, table);
  }

  void collectTableIndices(ir_sql_converter::AQPStmt &node,
                           std::set<unsigned int> &indices) {
    using NT = ir_sql_converter::SimplestNodeType;
    auto nt = node.GetNodeType();
    if (nt == NT::ScanNode) {
      indices.insert(
          static_cast<ir_sql_converter::SimplestScan &>(node).GetTableIndex());
    } else if (nt == NT::ChunkNode) {
      indices.insert(
          static_cast<ir_sql_converter::SimplestChunk &>(node).GetTableIndex());
    }
    for (auto &child : node.children) {
      collectTableIndices(*child, indices);
    }
  }

  mlir::Value ensureI1(mlir::OpBuilder &builder, mlir::Value val) {
    if (mlir::isa<db::NullableType>(val.getType()))
      return builder.create<db::DeriveTruth>(builder.getUnknownLoc(), val);
    return val;
  }

  mlir::Value wrapWithSelection(
      mlir::OpBuilder &builder,
      std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> &quals,
      mlir::Value input) {
    auto loc = builder.getUnknownLoc();
    auto selOp = builder.create<relalg::SelectionOp>(
        loc, tuples::TupleStreamType::get(&ctx_), input);

    auto *pred = new mlir::Block;
    pred->addArgument(tuples::TupleType::get(&ctx_), loc);
    mlir::OpBuilder predBuilder(&ctx_);
    predBuilder.setInsertionPointToStart(pred);

    mlir::Value tupleArg = pred->getArgument(0);
    mlir::Value predResult = buildPredicateFromQuals(predBuilder, quals, tupleArg);
    predBuilder.create<tuples::ReturnOp>(loc, ensureI1(predBuilder, predResult));

    selOp.getPredicate().push_back(pred);
    return selOp.getResult();
  }

  mlir::Value convertScan(mlir::OpBuilder &builder,
                          ir_sql_converter::SimplestScan &scan) {
    auto loc = builder.getUnknownLoc();
    auto tableName = scan.GetTableName();
    auto tableIndex = scan.GetTableIndex();

    auto scope = colMgr_.getUniqueScope(tableName);
    scopeMap_[tableIndex] = scope;
    tableNameMap_[tableIndex] = tableName;

    auto columns = buildAllCatalogColumns(builder, tableName, scope);

    mlir::Value result = builder.create<relalg::BaseTableOp>(
        loc, tuples::TupleStreamType::get(&ctx_), tableName,
        builder.getDictionaryAttr(columns),
        lingodb::runtime::DatasourceRestrictionProperty{});

    if (!scan.qual_vec.empty()) {
      result = wrapWithSelection(builder, scan.qual_vec, result);
    }
    return result;
  }

  mlir::Value convertChunk(mlir::OpBuilder &builder,
                           ir_sql_converter::SimplestChunk &chunk) {
    auto loc = builder.getUnknownLoc();
    auto chunkName = chunk.GetChunkName();
    auto tableIndex = chunk.GetTableIndex();

    if (chunkName.empty() && !chunk.GetContents().empty()) {
      chunkName = "__inline_" + std::to_string(tableIndex);
      createInlineTable(chunkName, chunk.GetContents());
    }

    auto scope = colMgr_.getUniqueScope(chunkName);
    scopeMap_[tableIndex] = scope;
    tableNameMap_[tableIndex] = chunkName;

    auto columns = buildAllCatalogColumns(builder, chunkName, scope);

    mlir::Value result = builder.create<relalg::BaseTableOp>(
        loc, tuples::TupleStreamType::get(&ctx_), chunkName,
        builder.getDictionaryAttr(columns),
        lingodb::runtime::DatasourceRestrictionProperty{});

    if (!chunk.qual_vec.empty()) {
      result = wrapWithSelection(builder, chunk.qual_vec, result);
    }
    return result;
  }

  bool isMarkFilterPattern(ir_sql_converter::SimplestFilter &filter) {
    if (filter.qual_vec.size() != 1 || filter.children.empty())
      return false;
    auto &qual = filter.qual_vec[0];
    if (qual->GetNodeType() != ir_sql_converter::SimplestNodeType::SingleAttrExprNode)
      return false;
    auto &sae = static_cast<ir_sql_converter::SimplestSingleAttrExpr &>(*qual);
    auto tableIdx = sae.attr->GetTableIndex();
    auto &child = *filter.children[0];
    if (child.GetNodeType() != ir_sql_converter::SimplestNodeType::JoinNode)
      return false;
    auto &join = static_cast<ir_sql_converter::SimplestJoin &>(child);
    return join.GetSimplestJoinType() == ir_sql_converter::SimplestJoinType::Mark
        && join.GetMarkIndex() == tableIdx;
  }

  mlir::Value convertMarkFilterAsSemiJoin(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestFilter &filter) {
    auto loc = builder.getUnknownLoc();
    auto &join = static_cast<ir_sql_converter::SimplestJoin &>(*filter.children[0]);

    mlir::Value left = convertNode(builder, *join.children[0]);
    mlir::Value right = convertNode(builder, *join.children[1]);

    std::set<unsigned int> leftTableIndices, rightTableIndices;
    collectTableIndices(*join.children[0], leftTableIndices);
    collectTableIndices(*join.children[1], rightTableIndices);

    auto semiJoinOp = builder.create<relalg::SemiJoinOp>(
        loc, tuples::TupleStreamType::get(&ctx_), left, right);

    auto *pred = new mlir::Block;
    pred->addArgument(tuples::TupleType::get(&ctx_), loc);
    mlir::OpBuilder predBuilder(&ctx_);
    predBuilder.setInsertionPointToStart(pred);
    mlir::Value tupleArg = pred->getArgument(0);

    std::vector<mlir::Value> conditions;
    llvm::SmallVector<mlir::Attribute> leftHashKeys, rightHashKeys, nullsEqualAttrs;
    bool canUseHash = !join.join_conditions.empty();

    for (auto &jc : join.join_conditions) {
      conditions.push_back(convertVarComparison(predBuilder, *jc, tupleArg));
      if (canUseHash && jc->GetSimplestExprType() == ir_sql_converter::SimplestExprType::Equal) {
        auto &la = *jc->left_attr;
        auto &ra = *jc->right_attr;
        auto lScope = resolveScope(la.GetTableIndex());
        auto rScope = resolveScope(ra.GetTableIndex());
        auto lColName = resolveColumnName(la.GetTableIndex(), la.GetColumnIndex(), la.GetColumnName());
        auto rColName = resolveColumnName(ra.GetTableIndex(), ra.GetColumnIndex(), ra.GetColumnName());
        auto lRef = resolveColRef(lScope, lColName);
        auto rRef = resolveColRef(rScope, rColName);

        bool lIsLeft = leftTableIndices.count(la.GetTableIndex()) > 0;
        bool rIsRight = rightTableIndices.count(ra.GetTableIndex()) > 0;
        bool lIsRight = rightTableIndices.count(la.GetTableIndex()) > 0;
        bool rIsLeft = leftTableIndices.count(ra.GetTableIndex()) > 0;

        if (lIsLeft && rIsRight) {
          leftHashKeys.push_back(lRef);
          rightHashKeys.push_back(rRef);
        } else if (lIsRight && rIsLeft) {
          leftHashKeys.push_back(rRef);
          rightHashKeys.push_back(lRef);
        } else {
          canUseHash = false;
        }
        nullsEqualAttrs.push_back(
            mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx_, 8), 0));
      }
    }

    mlir::Value predResult;
    if (conditions.size() == 1) {
      predResult = conditions[0];
    } else if (conditions.size() > 1) {
      predResult = predBuilder.create<db::AndOp>(loc, conditions);
    } else {
      predResult = predBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    }
    predBuilder.create<tuples::ReturnOp>(loc, ensureI1(predBuilder, predResult));
    semiJoinOp.getPredicate().push_back(pred);

    if (canUseHash && !leftHashKeys.empty()) {
      semiJoinOp->setAttr("useHashJoin", mlir::UnitAttr::get(&ctx_));
      semiJoinOp->setAttr("leftHash", builder.getArrayAttr(leftHashKeys));
      semiJoinOp->setAttr("rightHash", builder.getArrayAttr(rightHashKeys));
      semiJoinOp->setAttr("nullsEqual", builder.getArrayAttr(nullsEqualAttrs));
    }

    return semiJoinOp.getResult();
  }

  mlir::Value convertFilter(mlir::OpBuilder &builder,
                            ir_sql_converter::SimplestFilter &filter,
                            mlir::Value input) {
    auto loc = builder.getUnknownLoc();
    auto selOp = builder.create<relalg::SelectionOp>(
        loc, tuples::TupleStreamType::get(&ctx_), input);

    auto *pred = new mlir::Block;
    pred->addArgument(tuples::TupleType::get(&ctx_), loc);
    mlir::OpBuilder predBuilder(&ctx_);
    predBuilder.setInsertionPointToStart(pred);

    mlir::Value tupleArg = pred->getArgument(0);
    mlir::Value predResult = buildPredicateFromQuals(predBuilder, filter.qual_vec, tupleArg);
    predBuilder.create<tuples::ReturnOp>(loc, ensureI1(predBuilder, predResult));

    selOp.getPredicate().push_back(pred);
    return selOp.getResult();
  }

  mlir::Value convertJoin(mlir::OpBuilder &builder,
                          ir_sql_converter::SimplestJoin &join) {
    auto loc = builder.getUnknownLoc();
    auto joinType = join.GetSimplestJoinType();

    mlir::Value left = convertNode(builder, *join.children[0]);
    mlir::Value right = convertNode(builder, *join.children[1]);

    // Collect table indices reachable from left and right subtrees
    std::set<unsigned int> leftTableIndices, rightTableIndices;
    collectTableIndices(*join.children[0], leftTableIndices);
    collectTableIndices(*join.children[1], rightTableIndices);

    // Handle Mark Join (used for IN clauses) — kept as fallback but
    // normally converted to SemiJoin via isMarkFilterPattern
    if (joinType == ir_sql_converter::SimplestJoinType::Mark) {
      auto markIdx = join.GetMarkIndex();
      auto markScope = colMgr_.getUniqueScope("mark");
      scopeMap_[markIdx] = markScope;
      markIndices_.insert(markIdx);
      auto markDef = colMgr_.createDef(markScope, "col0");
      markDef.getColumn().type = builder.getI1Type();

      auto markJoinOp = builder.create<relalg::MarkJoinOp>(
          loc, tuples::TupleStreamType::get(&ctx_), markDef, left, right);

      auto *pred = new mlir::Block;
      pred->addArgument(tuples::TupleType::get(&ctx_), loc);
      mlir::OpBuilder predBuilder(&ctx_);
      predBuilder.setInsertionPointToStart(pred);
      mlir::Value tupleArg = pred->getArgument(0);

      std::vector<mlir::Value> conditions;
      for (auto &jc : join.join_conditions) {
        conditions.push_back(convertVarComparison(predBuilder, *jc, tupleArg));
      }
      mlir::Value predResult;
      if (conditions.size() == 1) {
        predResult = conditions[0];
      } else if (conditions.size() > 1) {
        predResult = predBuilder.create<db::AndOp>(loc, conditions);
      } else {
        predResult = predBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }
      predBuilder.create<tuples::ReturnOp>(loc, ensureI1(predBuilder, predResult));
      markJoinOp.getPredicate().push_back(pred);
      return markJoinOp.getResult();
    }

    auto joinOp = builder.create<relalg::InnerJoinOp>(
        loc, tuples::TupleStreamType::get(&ctx_), left, right);

    auto *pred = new mlir::Block;
    pred->addArgument(tuples::TupleType::get(&ctx_), loc);
    mlir::OpBuilder predBuilder(&ctx_);
    predBuilder.setInsertionPointToStart(pred);

    mlir::Value tupleArg = pred->getArgument(0);

    // Extract hash keys from equality join conditions
    llvm::SmallVector<mlir::Attribute> leftHashKeys, rightHashKeys, nullsEqualAttrs;
    bool canUseHash = !join.join_conditions.empty();

    // Build join predicate from join_conditions + qual_vec
    std::vector<mlir::Value> conditions;
    for (auto &jc : join.join_conditions) {
      conditions.push_back(convertVarComparison(predBuilder, *jc, tupleArg));

      if (canUseHash && jc->GetSimplestExprType() == ir_sql_converter::SimplestExprType::Equal) {
        auto &la = *jc->left_attr;
        auto &ra = *jc->right_attr;
        auto lScope = resolveScope(la.GetTableIndex());
        auto rScope = resolveScope(ra.GetTableIndex());
        auto lColName = resolveColumnName(la.GetTableIndex(), la.GetColumnIndex(), la.GetColumnName());
        auto rColName = resolveColumnName(ra.GetTableIndex(), ra.GetColumnIndex(), ra.GetColumnName());
        auto lRef = resolveColRef(lScope, lColName);
        auto rRef = resolveColRef(rScope, rColName);

        bool lIsLeft = leftTableIndices.count(la.GetTableIndex()) > 0;
        bool rIsRight = rightTableIndices.count(ra.GetTableIndex()) > 0;
        bool lIsRight = rightTableIndices.count(la.GetTableIndex()) > 0;
        bool rIsLeft = leftTableIndices.count(ra.GetTableIndex()) > 0;

        if (lIsLeft && rIsRight) {
          leftHashKeys.push_back(lRef);
          rightHashKeys.push_back(rRef);
        } else if (lIsRight && rIsLeft) {
          leftHashKeys.push_back(rRef);
          rightHashKeys.push_back(lRef);
        } else {
          canUseHash = false;
        }
        nullsEqualAttrs.push_back(
            mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx_, 8), 0));
      }
    }
    for (auto &q : join.qual_vec) {
      conditions.push_back(convertExpr(predBuilder, *q, tupleArg));
    }

    mlir::Value predResult;
    if (conditions.size() == 1) {
      predResult = conditions[0];
    } else if (conditions.size() > 1) {
      predResult = predBuilder.create<db::AndOp>(loc, conditions);
    } else {
      predResult =
          predBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    }

    predBuilder.create<tuples::ReturnOp>(loc, ensureI1(predBuilder, predResult));
    joinOp.getPredicate().push_back(pred);

    // Set hash join attributes
    if (canUseHash && !leftHashKeys.empty()) {
      joinOp->setAttr("useHashJoin", mlir::UnitAttr::get(&ctx_));
      joinOp->setAttr("leftHash", builder.getArrayAttr(leftHashKeys));
      joinOp->setAttr("rightHash", builder.getArrayAttr(rightHashKeys));
      joinOp->setAttr("nullsEqual", builder.getArrayAttr(nullsEqualAttrs));
    }

    return joinOp.getResult();
  }

  mlir::Value convertCrossProduct(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestCrossProduct &cp) {
    auto loc = builder.getUnknownLoc();
    mlir::Value left = convertNode(builder, *cp.children[0]);
    mlir::Value right = convertNode(builder, *cp.children[1]);
    return builder.create<relalg::CrossProductOp>(
        loc, tuples::TupleStreamType::get(&ctx_), left, right);
  }

  mlir::Value convertProjection(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestProjection &proj,
      mlir::Value input) {
    auto loc = builder.getUnknownLoc();

    // First try: resolve all target_list columns directly
    bool allResolved = true;
    llvm::SmallVector<mlir::Attribute> colRefs;
    for (auto &attr : proj.target_list) {
      auto scope = resolveScope(attr->GetTableIndex());
      auto colRef = resolveColRef(scope, attr->GetColumnName());
      if (!colRef.getColumn().type) {
        allResolved = false;
        break;
      }
      colRefs.push_back(colRef);
    }

    if (allResolved) {
      return builder.create<relalg::ProjectionOp>(
          loc, tuples::TupleStreamType::get(&ctx_), relalg::SetSemantic::all,
          input, builder.getArrayAttr(colRefs));
    }

    // Fallback: projection renames columns from child (e.g., aggregate
    // output with empty target_list). Map by column index into child refs.
    // DuckDB may deduplicate aggregates, so multiple projection entries
    // can reference the same child column (via GetColumnIndex()).
    colRefs.clear();
    auto childRefs = lastAggOutputRefs_;
    auto projScope = colMgr_.getUniqueScope("proj");

    for (auto &attr : proj.target_list) {
      scopeMap_[attr->GetTableIndex()] = projScope;
    }

    for (size_t i = 0; i < proj.target_list.size(); i++) {
      auto colIdx = proj.target_list[i]->GetColumnIndex();
      if (colIdx < childRefs.size() && childRefs[colIdx].getColumn().type) {
        auto childRef = childRefs[colIdx];
        auto projColName = proj.target_list[i]->GetColumnName();
        auto projDef = colMgr_.createDef(projScope, projColName);
        projDef.getColumn().type = childRef.getColumn().type;
        colRefs.push_back(childRef);
        std::string key = projScope + "::" + projColName;
        projToStreamMap_[key] = childRef;
      } else {
        auto scope = resolveScope(proj.target_list[i]->GetTableIndex());
        colRefs.push_back(
            resolveColRef(scope,
                          proj.target_list[i]->GetColumnName()));
      }
    }

    return builder.create<relalg::ProjectionOp>(
        loc, tuples::TupleStreamType::get(&ctx_), relalg::SetSemantic::all,
        input, builder.getArrayAttr(colRefs));
  }

  mlir::Value convertAggregate(mlir::OpBuilder &builder,
                               ir_sql_converter::SimplestAggregate &agg,
                               mlir::Value input) {
    auto loc = builder.getUnknownLoc();
    auto tupleStreamTy = tuples::TupleStreamType::get(&ctx_);

    // Group-by column refs
    llvm::SmallVector<mlir::Attribute> groupByRefs;
    for (auto &grp : agg.groups) {
      auto scope = resolveScope(grp->GetTableIndex());
      groupByRefs.push_back(resolveColRef(scope, grp->GetColumnName()));
    }

    // Create a unique scope for aggregate output columns
    auto aggScope = colMgr_.getUniqueScope("agg");

    // Build aggregation function block
    auto *block = new mlir::Block;
    block->addArgument(tupleStreamTy, loc);
    block->addArgument(tuples::TupleType::get(&ctx_), loc);

    mlir::OpBuilder aggrBuilder(&ctx_);
    aggrBuilder.setInsertionPointToStart(block);

    mlir::Value relation = block->getArgument(0);

    std::vector<mlir::Value> createdValues;
    llvm::SmallVector<mlir::Attribute> createdCols;

    auto &aggFns = agg.agg_fns;
    size_t aggIdx = 0;
    for (auto &[aggAttr, aggFnType] : aggFns) {
      std::string outColName;
      if (aggIdx < agg.target_list.size())
        outColName = agg.target_list[aggIdx]->GetColumnName();
      else
        outColName = "agg_" + std::to_string(aggIdx);

      auto outDef = colMgr_.createDef(aggScope, outColName);
      mlir::Value expr;

      if (aggFnType == ir_sql_converter::SimplestAggFnType::CountStar) {
        auto resultType = mlir::IntegerType::get(&ctx_, 64);
        expr = aggrBuilder.create<relalg::CountRowsOp>(loc, resultType,
                                                        relation);
        outDef.getColumn().type = resultType;
      } else {
        auto scope = resolveScope(aggAttr->GetTableIndex());
        auto colRef =
            resolveColRef(scope, aggAttr->GetColumnName());

        relalg::AggrFunc fn;
        switch (aggFnType) {
        case ir_sql_converter::SimplestAggFnType::Min:
          fn = relalg::AggrFunc::min;
          break;
        case ir_sql_converter::SimplestAggFnType::Max:
          fn = relalg::AggrFunc::max;
          break;
        case ir_sql_converter::SimplestAggFnType::Sum:
          fn = relalg::AggrFunc::sum;
          break;
        case ir_sql_converter::SimplestAggFnType::Average:
          fn = relalg::AggrFunc::avg;
          break;
        case ir_sql_converter::SimplestAggFnType::Count:
          fn = relalg::AggrFunc::count;
          break;
        default:
          fn = relalg::AggrFunc::min;
          break;
        }

        // Result type: for MIN/MAX, same as input; for COUNT, i64;
        // for SUM, promote to i64 for integer inputs
        mlir::Type inputType = colRef.getColumn().type;
        mlir::Type baseInputType = inputType;
        if (auto nullTy = mlir::dyn_cast<db::NullableType>(inputType))
          baseInputType = nullTy.getType();

        mlir::Type resultType;
        if (fn == relalg::AggrFunc::count) {
          resultType = mlir::IntegerType::get(&ctx_, 64);
        } else if (fn == relalg::AggrFunc::sum) {
          if (mlir::isa<mlir::IntegerType>(baseInputType))
            resultType = mlir::IntegerType::get(&ctx_, 64);
          else
            resultType = baseInputType;
        } else if (fn == relalg::AggrFunc::avg) {
          resultType = mlir::Float64Type::get(&ctx_);
        } else {
          // min/max: same as input base type
          resultType = baseInputType;
        }

        // Aggregate results are nullable
        auto nullableResultType = db::NullableType::get(&ctx_, resultType);
        expr = aggrBuilder.create<relalg::AggrFuncOp>(
            loc, nullableResultType, fn, relation, colRef);
        outDef.getColumn().type = nullableResultType;
      }

      createdCols.push_back(outDef);
      createdValues.push_back(expr);
      aggIdx++;
    }

    aggrBuilder.create<tuples::ReturnOp>(loc, createdValues);

    auto aggOp = builder.create<relalg::AggregationOp>(
        loc, tupleStreamTy, input, builder.getArrayAttr(groupByRefs),
        builder.getArrayAttr(createdCols));
    aggOp.getAggrFunc().push_back(block);

    lastAggOutputRefs_.clear();
    for (auto &col : createdCols) {
      auto def = mlir::cast<tuples::ColumnDefAttr>(col);
      lastAggOutputRefs_.push_back(
          colMgr_.createRef(&def.getColumn()));
    }

    for (size_t i = 0; i < agg.target_list.size(); i++) {
      scopeMap_[agg.target_list[i]->GetTableIndex()] = aggScope;
    }

    return aggOp.getResult();
  }

  mlir::Value convertOrderBy(mlir::OpBuilder &builder,
                             ir_sql_converter::SimplestOrderBy &orderBy,
                             mlir::Value input) {
    auto loc = builder.getUnknownLoc();

    llvm::SmallVector<mlir::Attribute> sortSpecs;
    for (auto &order : orderBy.orders) {
      auto scope = resolveScope(order.attr->GetTableIndex());
      auto colRef = resolveColRef(scope, order.attr->GetColumnName());

      auto spec = (order.order_type ==
                       ir_sql_converter::SimplestOrderType::Ascending ||
                   order.order_type ==
                       ir_sql_converter::SimplestOrderType::ORDER_DEFAULT)
                      ? relalg::SortSpec::asc
                      : relalg::SortSpec::desc;

      sortSpecs.push_back(
          relalg::SortSpecificationAttr::get(&ctx_, colRef, spec));
    }

    return builder.create<relalg::SortOp>(
        loc, tuples::TupleStreamType::get(&ctx_), input,
        builder.getArrayAttr(sortSpecs));
  }

  mlir::Value convertLimit(mlir::OpBuilder &builder,
                           ir_sql_converter::SimplestLimit &limit,
                           mlir::Value input) {
    auto loc = builder.getUnknownLoc();
    int32_t maxRows = static_cast<int32_t>(limit.limit_val.val);
    return builder.create<relalg::LimitOp>(
        loc, tuples::TupleStreamType::get(&ctx_), maxRows, input);
  }

  // ============== Expression Conversion ==============

  mlir::Value buildPredicateFromQuals(
      mlir::OpBuilder &builder,
      std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> &quals,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    if (quals.empty())
      return builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);

    std::vector<mlir::Value> conditions;
    for (auto &q : quals) {
      conditions.push_back(convertExpr(builder, *q, tuple));
    }

    if (conditions.size() == 1)
      return conditions[0];
    return builder.create<db::AndOp>(loc, conditions);
  }

  mlir::Value convertExpr(mlir::OpBuilder &builder,
                          ir_sql_converter::AQPExpr &expr,
                          mlir::Value tuple) {
    using NT = ir_sql_converter::SimplestNodeType;
    auto nt = expr.GetNodeType();

    switch (nt) {
    case NT::VarComparisonNode:
      return convertVarComparison(
          builder,
          static_cast<ir_sql_converter::SimplestVarComparison &>(expr),
          tuple);
    case NT::VarConstComparisonNode:
      return convertVarConstComparison(
          builder,
          static_cast<ir_sql_converter::SimplestVarConstComparison &>(
              expr),
          tuple);
    case NT::LogicalExprNode:
      return convertLogicalExpr(
          builder,
          static_cast<ir_sql_converter::SimplestLogicalExpr &>(expr),
          tuple);
    case NT::IsNullExprNode:
      return convertIsNullExpr(
          builder,
          static_cast<ir_sql_converter::SimplestIsNullExpr &>(expr),
          tuple);
    case NT::InExprNode:
      return convertInExpr(
          builder,
          static_cast<ir_sql_converter::SimplestInExpr &>(expr), tuple);
    case NT::ArithExprNode:
      return convertArithExpr(
          builder,
          static_cast<ir_sql_converter::SimplestArithExpr &>(expr),
          tuple);
    case NT::CastExprNode:
      return convertCastExpr(
          builder,
          static_cast<ir_sql_converter::SimplestCastExpr &>(expr),
          tuple);
    case NT::SingleAttrExprNode: {
      auto &sae =
          static_cast<ir_sql_converter::SimplestSingleAttrExpr &>(expr);
      auto &attr = *sae.attr;
      auto scope = resolveScope(attr.GetTableIndex());
      auto colName = resolveColumnName(attr.GetTableIndex(),
                                       attr.GetColumnIndex(),
                                       attr.GetColumnName());
      auto colRef = resolveColRef(scope, colName);
      return builder.create<tuples::GetColumnOp>(
          builder.getUnknownLoc(), colRef.getColumn().type, colRef, tuple);
    }
    default:
      throw std::runtime_error(
          "[IRToRelAlg] Unsupported expression type: " +
          std::to_string(static_cast<int>(nt)));
    }
  }

  mlir::Value convertVarComparison(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestVarComparison &cmp,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto &leftAttr = *cmp.left_attr;
    auto &rightAttr = *cmp.right_attr;

    auto leftScope = resolveScope(leftAttr.GetTableIndex());
    auto rightScope = resolveScope(rightAttr.GetTableIndex());

    auto leftColName = resolveColumnName(leftAttr.GetTableIndex(),
                                         leftAttr.GetColumnIndex(),
                                         leftAttr.GetColumnName());
    auto rightColName = resolveColumnName(rightAttr.GetTableIndex(),
                                          rightAttr.GetColumnIndex(),
                                          rightAttr.GetColumnName());
    auto leftRef = resolveColRef(leftScope, leftColName);
    auto rightRef = resolveColRef(rightScope, rightColName);

    if (!leftRef.getColumn().type) {
      throw std::runtime_error(
          "[IRToRelAlg] VarComparison: left column type is null for scope=" +
          leftScope + " col=" + leftColName +
          " tableIdx=" + std::to_string(leftAttr.GetTableIndex()) +
          " colIdx=" + std::to_string(leftAttr.GetColumnIndex()));
    }
    if (!rightRef.getColumn().type) {
      throw std::runtime_error(
          "[IRToRelAlg] VarComparison: right column type is null for scope=" +
          rightScope + " col=" + rightColName +
          " tableIdx=" + std::to_string(rightAttr.GetTableIndex()) +
          " colIdx=" + std::to_string(rightAttr.GetColumnIndex()));
    }

    auto leftVal = builder.create<tuples::GetColumnOp>(
        loc, leftRef.getColumn().type, leftRef, tuple);
    auto rightVal = builder.create<tuples::GetColumnOp>(
        loc, rightRef.getColumn().type, rightRef, tuple);

    auto pred = mapCmpPredicate(cmp.GetSimplestExprType());
    return builder.create<db::CmpOp>(loc, pred, leftVal, rightVal);
  }

  mlir::Value convertVarConstComparison(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestVarConstComparison &cmp,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto &attr = *cmp.attr;
    auto &constVar = *cmp.const_var;

    auto scope = resolveScope(attr.GetTableIndex());
    auto colName = resolveColumnName(attr.GetTableIndex(),
                                     attr.GetColumnIndex(),
                                     attr.GetColumnName());
    auto colRef = resolveColRef(scope, colName);
    if (!colRef.getColumn().type) {
      throw std::runtime_error(
          "[IRToRelAlg] VarConstComparison: column type is null for scope=" +
          scope + " col=" + colName +
          " tableIdx=" + std::to_string(attr.GetTableIndex()) +
          " colIdx=" + std::to_string(attr.GetColumnIndex()));
    }
    auto colVal = builder.create<tuples::GetColumnOp>(
        loc, colRef.getColumn().type, colRef, tuple);

    auto colType = colRef.getColumn().type;
    auto baseColType = colType;
    if (auto nullTy = mlir::dyn_cast<db::NullableType>(colType))
      baseColType = nullTy.getType();

    auto constVal = createConstant(builder, constVar, baseColType);

    auto exprType = cmp.GetSimplestExprType();
    if (exprType == ir_sql_converter::SimplestExprType::TextLike ||
        exprType == ir_sql_converter::SimplestExprType::Text_Not_Like) {
      bool isNullable = mlir::isa<db::NullableType>(colType) ||
                        mlir::isa<db::NullableType>(constVal.getType());
      mlir::Type resType =
          isNullable
              ? (mlir::Type)db::NullableType::get(&ctx_,
                                                  builder.getI1Type())
              : (mlir::Type)builder.getI1Type();
      auto like = builder.create<db::RuntimeCall>(
          loc, resType, "Like", mlir::ValueRange({colVal, constVal}));
      mlir::Value result = like.getRes();
      if (exprType == ir_sql_converter::SimplestExprType::Text_Not_Like)
        result = builder.create<db::NotOp>(loc, result);
      return result;
    }

    auto pred = mapCmpPredicate(exprType);
    return builder.create<db::CmpOp>(loc, pred, colVal, constVal);
  }

  mlir::Value convertLogicalExpr(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestLogicalExpr &expr,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();

    auto op = expr.GetLogicalOp();
    if (op == ir_sql_converter::SimplestLogicalOp::LogicalNot) {
      auto rightVal = convertExpr(builder, *expr.right_expr, tuple);
      return builder.create<db::NotOp>(loc, rightVal);
    }

    auto leftVal = convertExpr(builder, *expr.left_expr, tuple);
    auto rightVal = convertExpr(builder, *expr.right_expr, tuple);

    std::vector<mlir::Value> operands = {leftVal, rightVal};

    if (op == ir_sql_converter::SimplestLogicalOp::LogicalAnd)
      return builder.create<db::AndOp>(loc, operands);
    else
      return builder.create<db::OrOp>(loc, operands);
  }

  mlir::Value convertIsNullExpr(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestIsNullExpr &expr,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto &attr = *expr.attr;

    auto scope = resolveScope(attr.GetTableIndex());
    auto colRef = resolveColRef(scope, attr.GetColumnName());
    auto colVal = builder.create<tuples::GetColumnOp>(
        loc, colRef.getColumn().type, colRef, tuple);

    auto isNull = builder.create<db::IsNullOp>(loc, colVal);

    if (expr.GetSimplestExprType() ==
        ir_sql_converter::SimplestExprType::NonNullType)
      return builder.create<db::NotOp>(loc, isNull);
    return isNull;
  }

  mlir::Value convertInExpr(mlir::OpBuilder &builder,
                            ir_sql_converter::SimplestInExpr &expr,
                            mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto &attr = *expr.attr;

    auto scope = resolveScope(attr.GetTableIndex());
    auto colRef = resolveColRef(scope, attr.GetColumnName());
    auto colVal = builder.create<tuples::GetColumnOp>(
        loc, colRef.getColumn().type, colRef, tuple);

    auto colType = colRef.getColumn().type;
    auto baseColType = colType;
    if (auto nullTy = mlir::dyn_cast<db::NullableType>(colType))
      baseColType = nullTy.getType();

    std::vector<mlir::Value> values;
    values.push_back(colVal);
    for (auto &v : expr.values) {
      values.push_back(createConstant(builder, *v, baseColType));
    }

    mlir::Value result = builder.create<db::OneOfOp>(loc, values);
    if (expr.negated)
      result = builder.create<db::NotOp>(loc, result);
    return result;
  }

  mlir::Value convertArithExpr(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestArithExpr &expr,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto leftVal = convertExpr(builder, *expr.left, tuple);
    auto rightVal = convertExpr(builder, *expr.right, tuple);

    switch (expr.arith_op) {
    case ir_sql_converter::SimplestArithOp::ArithAdd:
      return builder.create<db::AddOp>(loc, leftVal, rightVal);
    case ir_sql_converter::SimplestArithOp::ArithSub:
      return builder.create<db::SubOp>(loc, leftVal, rightVal);
    case ir_sql_converter::SimplestArithOp::ArithMul:
      return builder.create<db::MulOp>(loc, leftVal, rightVal);
    case ir_sql_converter::SimplestArithOp::ArithDiv:
      return builder.create<db::DivOp>(loc, leftVal, rightVal);
    case ir_sql_converter::SimplestArithOp::ArithMod:
      return builder.create<db::ModOp>(loc, leftVal, rightVal);
    default:
      throw std::runtime_error("[IRToRelAlg] Unsupported arith op");
    }
  }

  mlir::Value convertCastExpr(
      mlir::OpBuilder &builder,
      ir_sql_converter::SimplestCastExpr &expr,
      mlir::Value tuple) {
    auto loc = builder.getUnknownLoc();
    auto child = convertExpr(builder, *expr.child, tuple);
    return child;
  }

  // ============== Helpers ==============

  mlir::Value createConstant(mlir::OpBuilder &builder,
                             ir_sql_converter::SimplestConstVar &constVar,
                             mlir::Type targetType) {
    auto loc = builder.getUnknownLoc();

    switch (constVar.GetType()) {
    case ir_sql_converter::SimplestVarType::IntVar: {
      auto val = constVar.GetIntValue();
      return builder.create<db::ConstantOp>(
          loc, targetType, builder.getI32IntegerAttr(val));
    }
    case ir_sql_converter::SimplestVarType::FloatVar: {
      auto val = constVar.GetFloatValue();
      return builder.create<db::ConstantOp>(
          loc, targetType, builder.getStringAttr(std::to_string(val)));
    }
    case ir_sql_converter::SimplestVarType::StringVar: {
      auto val = constVar.GetStringValue();
      return builder.create<db::ConstantOp>(
          loc, targetType, builder.getStringAttr(val));
    }
    case ir_sql_converter::SimplestVarType::StringVarArr: {
      auto vals = constVar.GetStringVecValue();
      if (!vals.empty())
        return builder.create<db::ConstantOp>(
            loc, targetType, builder.getStringAttr(vals[0]));
      return builder.create<db::NullOp>(
          loc, db::NullableType::get(&ctx_, targetType));
    }
    default:
      return builder.create<db::NullOp>(
          loc, db::NullableType::get(&ctx_, targetType));
    }
  }

  db::DBCmpPredicate
  mapCmpPredicate(ir_sql_converter::SimplestExprType exprType) {
    using ET = ir_sql_converter::SimplestExprType;
    switch (exprType) {
    case ET::Equal:
      return db::DBCmpPredicate::eq;
    case ET::NotEqual:
      return db::DBCmpPredicate::neq;
    case ET::LessThan:
      return db::DBCmpPredicate::lt;
    case ET::GreaterThan:
      return db::DBCmpPredicate::gt;
    case ET::LessEqual:
      return db::DBCmpPredicate::lte;
    case ET::GreaterEqual:
      return db::DBCmpPredicate::gte;
    default:
      return db::DBCmpPredicate::eq;
    }
  }
};

class NoOpQueryOptimizer : public lingodb::execution::QueryOptimizer {
public:
  void optimize(mlir::ModuleOp &) override {}
};

class IRFrontend : public lingodb::execution::Frontend {
  ir_sql_converter::AQPStmt *ir_;
  lingodb::runtime::Session *session_;
  mlir::MLIRContext *context_ = nullptr;
  mlir::OwningOpRef<mlir::ModuleOp> module_;

public:
  IRFrontend(ir_sql_converter::AQPStmt *ir, lingodb::runtime::Session *session)
      : ir_(ir), session_(session) {}

  void setContext(mlir::MLIRContext *context) override {
    context_ = context;
  }

  void loadFromString(std::string) override {
    IRToRelAlgConverter converter(*context_, *catalog, session_);
    auto moduleOp = converter.convert(*ir_);
#ifndef NDEBUG
    std::cerr << "[LingoDB-Runtime] Generated MLIR:" << std::endl;
    moduleOp.print(llvm::errs());
    std::cerr << std::endl;
#endif
    module_ = moduleOp;
  }

  void loadFromFile(std::string) override {}

  mlir::ModuleOp *getModule() override {
    assert(module_);
    return module_.operator->();
  }
};

} // anonymous namespace

// ============== LingoDBRuntimeAdapter ==============

LingoDBRuntimeAdapter::LingoDBRuntimeAdapter(const std::string &db_path)
    : LingoDBAdapter(db_path) {}

static bool hasCrossProduct(const ir_sql_converter::AQPStmt *node) {
  if (!node) return false;
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::CrossProductNode)
    return true;
  for (auto &child : node->children)
    if (hasCrossProduct(child.get())) return true;
  return false;
}

void LingoDBRuntimeAdapter::ExecuteIRandCreateTempTable(
    ir_sql_converter::AQPStmt &ir, const std::string &temp_table_name,
    bool update_temp_card) {
  if (hasCrossProduct(&ir)) {
    std::string sql = GenerateSQL(ir, subquery_index++);
    ExecuteSQLandCreateTempTable(sql, temp_table_name, update_temp_card);
    return;
  }

  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing_)
    timer = chrono_tic();

  // 1+2. Build and execute MLIR with minimal optimizer (skip plan-changing passes)
  auto config =
      lingodb::execution::createQueryExecutionConfig(exec_mode_, false);
  config->queryOptimizer = std::make_unique<NoOpQueryOptimizer>();
  config->frontend = std::make_unique<IRFrontend>(&ir, session_.get());

  std::shared_ptr<arrow::Table> result_table;
  config->resultProcessor =
      lingodb::execution::createTableRetriever(result_table);

  lingodb::execution::TimingCollector *timing_collector = nullptr;
  if (enable_timing_) {
    auto collector =
        std::make_unique<lingodb::execution::TimingCollector>();
    timing_collector = collector.get();
    config->timingProcessor = std::move(collector);
  }

  auto executer = lingodb::execution::QueryExecuter::createDefaultExecuter(
      std::move(config), *session_);
  executer->fromData("");
  lingodb::scheduler::awaitEntryTask(
      std::make_unique<lingodb::execution::QueryExecutionTask>(
          std::move(executer)));

  if (enable_timing_) {
    auto execute_time =
        chrono_toc(&timer, "Execute MLIR time\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (execute_time / 1000.0) << ", ";
    log_file.close();
  }

  if (enable_timing_ && timing_collector) {
    WriteLingoDBTimingRow(timing_collector->getTiming());
  }

  if (!result_table) {
    throw std::runtime_error(
        "[LingoDB-Runtime] ExecuteIRandCreateTempTable: no results");
  }

  // 3. Store result as temp table
  CreateTempTableFromArrow(temp_table_name, result_table);

  if (enable_timing_) {
    auto materialize_time =
        chrono_toc(&timer, "Materialize temp table time\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (materialize_time / 1000.0) << ", ";
    log_file.close();
  }

#ifndef NDEBUG
  std::cout << "[LingoDB-Runtime] Created temp table: " << temp_table_name
            << " (rows=" << temp_table_card_[temp_table_name] << ")"
            << std::endl;
#endif
}

QueryResult LingoDBRuntimeAdapter::ExecuteIRQuery(
    ir_sql_converter::AQPStmt &ir) {
  if (hasCrossProduct(&ir)) {
    std::string sql = GenerateSQL(ir, subquery_index++);
    auto result = ExecuteSQL(sql);
    if (enable_timing_) {
      WriteLingoDBTimingRow({});
    }
    return result;
  }

  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing_)
    timer = chrono_tic();

  auto config =
      lingodb::execution::createQueryExecutionConfig(exec_mode_, false);
  config->queryOptimizer = std::make_unique<NoOpQueryOptimizer>();
  config->frontend = std::make_unique<IRFrontend>(&ir, session_.get());

  std::shared_ptr<arrow::Table> result_table;
  config->resultProcessor =
      lingodb::execution::createTableRetriever(result_table);

  lingodb::execution::TimingCollector *timing_collector = nullptr;
  if (enable_timing_) {
    auto collector =
        std::make_unique<lingodb::execution::TimingCollector>();
    timing_collector = collector.get();
    config->timingProcessor = std::move(collector);
  }

  auto executer = lingodb::execution::QueryExecuter::createDefaultExecuter(
      std::move(config), *session_);
  executer->fromData("");
  lingodb::scheduler::awaitEntryTask(
      std::make_unique<lingodb::execution::QueryExecutionTask>(
          std::move(executer)));

  if (enable_timing_) {
    auto execute_time =
        chrono_toc(&timer, "Execute MLIR time\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (execute_time / 1000.0) << ", ";
    log_file.close();
  }

  if (enable_timing_ && timing_collector) {
    WriteLingoDBTimingRow(timing_collector->getTiming());
  }

  if (result_table) {
    return ArrowTableToQueryResult(result_table);
  }
  return QueryResult();
}

} // namespace middleware
