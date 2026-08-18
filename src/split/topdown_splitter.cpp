/*
 * SDS — stats-driven IR splitter (see topdown_splitter.h).
 */

#include "split/topdown_splitter.h"
#include "split/ir_query_splitter.h"
#include "split/ir_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace middleware {

// C++14: out-of-class definitions for ODR-used constexpr statics.
constexpr double TopDownSplitter::kSplitCardThreshold;
constexpr double TopDownSplitter::kSemiSelectivity;

namespace {

// Max table_index across all attrs, scans, chunks and join conditions:
// DuckDB assigns indices to projection/aggregate/chunk bindings too, and
// those can exceed the scan indices (e.g. a Mark join's Chunk node); temp
// indices must not collide, or index rewriting corrupts the preserved Mark
// condition.
unsigned int FindMaxAttrIndex(const ir_sql_converter::AQPStmt *node) {
  if (!node)
    return 0;
  unsigned int mx = 0;
  for (const auto &attr : node->target_list)
    mx = std::max(mx, attr->GetTableIndex());
  switch (node->GetNodeType()) {
  case ir_sql_converter::SimplestNodeType::ScanNode: {
    auto *s = dynamic_cast<const ir_sql_converter::SimplestScan *>(node);
    if (s)
      mx = std::max(mx, s->GetTableIndex());
    break;
  }
  case ir_sql_converter::SimplestNodeType::ChunkNode: {
    auto *c = dynamic_cast<const ir_sql_converter::SimplestChunk *>(node);
    if (c)
      mx = std::max(mx, c->GetTableIndex());
    break;
  }
  case ir_sql_converter::SimplestNodeType::JoinNode: {
    auto *j = dynamic_cast<const ir_sql_converter::SimplestJoin *>(node);
    if (j) {
      if (j->GetSimplestJoinType() == ir_sql_converter::SimplestJoinType::Mark)
        mx = std::max(mx, j->GetMarkIndex());
      for (const auto &cond : j->join_conditions) {
        if (!cond)
          continue;
        if (cond->left_attr)
          mx = std::max(mx, cond->left_attr->GetTableIndex());
        if (cond->right_attr)
          mx = std::max(mx, cond->right_attr->GetTableIndex());
      }
    }
    break;
  }
  case ir_sql_converter::SimplestNodeType::ProjectionNode: {
    auto *p = dynamic_cast<const ir_sql_converter::SimplestProjection *>(node);
    if (p)
      mx = std::max(mx, p->GetIndex());
    break;
  }
  case ir_sql_converter::SimplestNodeType::AggregateNode: {
    auto *a = dynamic_cast<const ir_sql_converter::SimplestAggregate *>(node);
    if (a) {
      mx = std::max(mx, a->GetAggIndex());
      mx = std::max(mx, a->GetGroupIndex());
    }
    break;
  }
  case ir_sql_converter::SimplestNodeType::OrderNode: {
    auto *o = dynamic_cast<const ir_sql_converter::SimplestOrderBy *>(node);
    if (o) {
      for (const auto &ord : o->orders)
        if (ord.attr)
          mx = std::max(mx, ord.attr->GetTableIndex());
    }
    break;
  }
  default:
    break;
  }
  for (const auto &child : node->children)
    mx = std::max(mx, FindMaxAttrIndex(child.get()));
  return mx;
}

// Walk AND-conjuncts of a filter expression, invoking cb on every
// VarComparison (column-to-column predicate). OR subtrees are skipped
// (a predicate under OR is not a guaranteed join edge).
template <typename Callback>
void ForEachJoinPredicate(const ir_sql_converter::AQPExpr *expr,
                          Callback &&cb) {
  if (!expr)
    return;
  auto nt = expr->GetNodeType();
  if (nt == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *le = dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (le && le->GetLogicalOp() == ir_sql_converter::LogicalAnd) {
      ForEachJoinPredicate(le->left_expr.get(), cb);
      ForEachJoinPredicate(le->right_expr.get(), cb);
    }
  } else if (nt == ir_sql_converter::SimplestNodeType::VarComparisonNode) {
    auto *vc =
        dynamic_cast<const ir_sql_converter::SimplestVarComparison *>(expr);
    if (vc)
      cb(vc);
  }
}

// Decompose AND-conjuncts and invoke cb on each leaf (non-AND) expression.
// Unlike ForEachJoinPredicate which only visits VarComparison leaves, this
// visits every conjunct regardless of type.
template <typename Callback>
void ForEachConjunct(const ir_sql_converter::AQPExpr *expr, Callback &&cb) {
  if (!expr)
    return;
  if (expr->GetNodeType() == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *le = dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (le && le->GetLogicalOp() == ir_sql_converter::LogicalAnd) {
      ForEachConjunct(le->left_expr.get(), cb);
      ForEachConjunct(le->right_expr.get(), cb);
      return;
    }
  }
  cb(expr);
}

void CollectMarkLockedTables(const ir_sql_converter::AQPStmt *node,
                             std::set<unsigned int> &locked,
                             bool under_mark = false) {
  if (!node)
    return;
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *j = dynamic_cast<const ir_sql_converter::SimplestJoin *>(node);
    if (j && j->GetSimplestJoinType() == ir_sql_converter::SimplestJoinType::Mark)
      under_mark = true;
  }
  if (under_mark &&
      node->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *s = dynamic_cast<const ir_sql_converter::SimplestScan *>(node);
    if (s)
      locked.insert(s->GetTableIndex());
  }
  for (const auto &child : node->children)
    CollectMarkLockedTables(child.get(), locked, under_mark);
}

// Classify tables under Mark joins (IN-list filters lowered to
// MarkJoin+Chunk). The canonical Filter->Mark->[Scan,Chunk] unit is clonable
// into sub-IRs (BuildSubIRForCluster carries it with the scan), so those
// tables stay splittable and their IN selectivity is recorded as
// (probe column, IN-list size). Any other Mark shape locks its scans.
void CollectMarkInfo(
    const ir_sql_converter::AQPStmt *node,
    std::map<unsigned int, std::pair<std::string, double>> &mark_in,
    std::set<unsigned int> &locked) {
  if (!node)
    return;
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::FilterNode &&
      !node->children.empty()) {
    auto *j = dynamic_cast<const ir_sql_converter::SimplestJoin *>(
        node->children[0].get());
    if (j &&
        j->GetSimplestJoinType() == ir_sql_converter::SimplestJoinType::Mark) {
      const ir_sql_converter::SimplestScan *scan = nullptr;
      const ir_sql_converter::SimplestChunk *chunk = nullptr;
      for (const auto &c : j->children) {
        if (!c)
          continue;
        if (auto *s =
                dynamic_cast<const ir_sql_converter::SimplestScan *>(c.get()))
          scan = s;
        else if (auto *ch =
                     dynamic_cast<const ir_sql_converter::SimplestChunk *>(
                         c.get()))
          chunk = ch;
      }
      std::string probe_col;
      if (scan) {
        for (const auto &cond : j->join_conditions) {
          if (!cond)
            continue;
          if (cond->left_attr &&
              cond->left_attr->GetTableIndex() == scan->GetTableIndex())
            probe_col = cond->left_attr->GetColumnName();
          else if (cond->right_attr &&
                   cond->right_attr->GetTableIndex() == scan->GetTableIndex())
            probe_col = cond->right_attr->GetColumnName();
        }
      }
      if (scan && chunk && !probe_col.empty() &&
          !chunk->GetContents().empty()) {
        mark_in[scan->GetTableIndex()] = {
            probe_col, static_cast<double>(chunk->GetContents().size())};
        return; // handled; nothing below this unit to classify
      }
    }
  }
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *j = dynamic_cast<const ir_sql_converter::SimplestJoin *>(node);
    if (j) {
      auto jt = j->GetSimplestJoinType();
      if (jt == ir_sql_converter::SimplestJoinType::Mark) {
        // Mark join not wrapped by the canonical Filter unit: not clonable.
        CollectMarkLockedTables(node, locked, /*under_mark=*/true);
        return;
      }
      if (jt == ir_sql_converter::SimplestJoinType::Semi ||
          jt == ir_sql_converter::SimplestJoinType::Anti ||
          jt == ir_sql_converter::SimplestJoinType::Left ||
          jt == ir_sql_converter::SimplestJoinType::Right ||
          jt == ir_sql_converter::SimplestJoinType::Full) {
#ifndef NDEBUG
        static const char *jt_names[] = {"SEMI", "ANTI", "LEFT", "RIGHT", "FULL"};
        int jt_idx = (jt == ir_sql_converter::SimplestJoinType::Semi)    ? 0
                   : (jt == ir_sql_converter::SimplestJoinType::Anti)    ? 1
                   : (jt == ir_sql_converter::SimplestJoinType::Left)    ? 2
                   : (jt == ir_sql_converter::SimplestJoinType::Right)   ? 3
                   :                                                       4;
        std::cerr << "[SDS] WARNING: " << jt_names[jt_idx]
                  << " join detected in IR; locking subtree tables "
                     "(cardinality model does not handle this join natively)"
                  << std::endl;
#endif
        CollectMarkLockedTables(node, locked, /*under_mark=*/true);
        return;
      }
    }
  }
  for (const auto &child : node->children)
    CollectMarkInfo(child.get(), mark_in, locked);
}

} // namespace

TopDownSplitter::TopDownSplitter(EngineAdapter *adapter,
                                 BackendEngine engine,
                                 bool apply_engine_settings)
    : FKBasedSplitter(adapter, engine, SplitStrategy::TOP_DOWN,
                      /*enable_analyze=*/false, /*fkeys_path=*/"") {
  // B2 (2026-07-11): SDS sub-SQLs are simple star joins; ~52% of DuckDB's
  // Prepare() cost is the optimizer pipeline, spread thinly across passes
  // that are provable no-ops here (no CTEs, windows, LIMIT, GROUP BY, or
  // correlated subqueries in JOB sub-SQLs). Disabling them session-wide
  // trims Prepare ~11% per statement. compressed_materialization stays ON:
  // disabling it regressed the 15*/16* families ~5-7 ms each (string-heavy
  // join intermediates), and v15 Step 4's proj-remap already handles its
  // injected projections in qjit. Setting only — no engine patch. Ignore
  // failure on engines without the setting.
  // Skipped for bg-thread construction (apply_engine_settings=false): the
  // shared adapter connection must not be touched from the prep thread.
  // Once per process: the setting is engine-global and the ctor runs per
  // query — repeating it would add ~0.2 ms/query of untracked overhead.
  // Kill switch for A/B measurement: AQP_TD_NO_OPTSET disables the SET.
  if (apply_engine_settings && !std::getenv("AQP_TD_NO_OPTSET")) {
    static bool applied = false;
    if (!applied) {
      applied = true;
      try {
        adapter->ApplyEngineSetting(
            "SET disabled_optimizers='filter_pullup,empty_result_pullup,"
            "cte_filter_pusher,regex_range,deliminator,unnest_rewriter,"
            "common_subexpressions,common_aggregate,limit_pushdown,top_n,"
            "top_n_window_elimination,"
            "duplicate_groups,sampling_pushdown,materialized_cte,sum_rewriter,"
            "late_materialization,cte_inlining,common_subplan,join_elimination,"
            "window_self_join'");
      } catch (...) {
      }
    }
  }
}

void TopDownSplitter::Preprocess(
    std::unique_ptr<ir_sql_converter::AQPStmt> &ir) {
  // Engine-independent preprocessing: no FK extraction, no ANALYZE, no
  // ReOptimizeIR. The incoming IR already carries the engine's estimated
  // cardinalities on every node.
  split_iteration_ = 0;
  executed_tables_.clear();
  table_card_.clear();
  temp_indices_.clear();
  mark_in_.clear();
  mark_locked_.clear();
  is_relationship_.clear();
  explain_cache_.clear();
  planned_for_ = nullptr;
  planned_splittable_ = false;
  planned_group_.clear();
  planned_card_ = 0.0;
  col_distinct_hints_.clear();

  table_index_to_name_.clear();
  CollectTableNames(ir.get());
  max_table_index_ = FindMaxAttrIndex(ir.get());

  // Keep the base class's join-pair bookkeeping consistent (used by
  // PrepareForNextIteration inside the inherited UpdateRemainingIR). No
  // FK-FK removal happens under TOP_DOWN.
  BuildJoinGraph(ir.get());

  CollectMarkInfo(ir.get(), mark_in_, mark_locked_);
  CaptureLeafCardinalities(ir.get());
  FetchMissingLeafCardinalities(ir.get());

  // Mark-lowered IN filters are invisible to the qual-based selectivity
  // model (the engine's scan estimate under a Mark join is the unfiltered
  // base count). DuckDB's join-order model never distinct-divides mark/semi
  // joins (DEFAULT_SEMI_ANTI_SELECTIVITY = 5, flat) — the uniform-distinct
  // k/d model collapses on skewed columns (31c ci.note: est 254 vs actual
  // ~1.4M), so take the LOOSER of the two estimates.
  for (const auto &[t, info] : mark_in_) {
    auto it = table_card_.find(t);
    if (it == table_card_.end())
      continue;
    double flat = std::max(it->second / kSemiSelectivity, 1.0);
    double est = flat;
    if (!bg_mode_) {
      double distinct = distinct_cache_.Get(*adapter_, GetTableName(t),
                                            info.first);
      if (distinct > 0.0) {
        double cand = std::ceil(it->second * info.second / distinct);
        est = std::max(est, std::min(it->second, std::max(cand, 1.0)));
      }
    }
    it->second = est;
  }

#ifndef NDEBUG
  std::cout << "[SDS] Preprocess: " << table_card_.size()
            << " leaf relation(s), max_table_index=" << max_table_index_
            << std::endl;
  for (const auto &[idx, card] : table_card_)
    std::cout << "  rel " << idx << " (" << GetTableName(idx)
              << ") card=" << card << std::endl;
#endif
}

// Record the effective (post-filter) cardinality per base table: the minimum
// estimated_cardinality over all nodes whose subtree covers exactly that one
// table (scan + any filter/projection stacked on it).
void TopDownSplitter::CaptureLeafCardinalities(
    const ir_sql_converter::AQPStmt *node) {
  std::function<std::set<unsigned int>(const ir_sql_converter::AQPStmt *)>
      walk = [&](const ir_sql_converter::AQPStmt *n) -> std::set<unsigned int> {
    std::set<unsigned int> tables;
    if (!n)
      return tables;
    if (n->GetNodeType() == ir_sql_converter::SimplestNodeType::SetOpNode)
      return tables;
    if (n->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
      auto *scan = dynamic_cast<const ir_sql_converter::SimplestScan *>(n);
      if (scan)
        tables.insert(scan->GetTableIndex());
    }
    for (const auto &child : n->children) {
      auto sub = walk(child.get());
      tables.insert(sub.begin(), sub.end());
    }
    if (tables.size() == 1) {
      double card = static_cast<double>(n->GetEstimatedCardinality());
      if (card > 0.0) {
        unsigned int t = *tables.begin();
        auto it = table_card_.find(t);
        if (it == table_card_.end() || card < it->second)
          table_card_[t] = card;
      }
    }
    return tables;
  };
  walk(node);
}

void TopDownSplitter::FetchMissingLeafCardinalities(
    ir_sql_converter::AQPStmt *ir) {
  if (bg_mode_) return;
  std::vector<JoinRel> rels;
  std::vector<unsigned int> rel_tables;
  CollectRelations(ir, rels, rel_tables);

  // The engine's single-table estimate is the UNFILTERED base count (filter
  // selectivity is applied by our own model below), so it is a per-table-name
  // constant: cache it for the process lifetime. Without this, every
  // Preprocess re-EXPLAINs every base table (~8 x 113 x repeat round trips,
  // several seconds of MW per iteration).
  std::vector<unsigned int> missing;
  std::vector<std::string> sqls;
  for (auto t : rel_tables) {
    if (table_card_.count(t))
      continue;
    if (base_count_cache_.count(GetTableName(t)))
      continue;
    auto sub_ir = BuildSubIRForCluster(ir, {t});
    if (!sub_ir)
      continue;
    std::string sql = ir_sql_converter::ConvertIRToSQL(*sub_ir, 0);
    if (sql.empty())
      continue;
    missing.push_back(t);
    sqls.push_back(std::move(sql));
  }
  if (!sqls.empty()) {
    auto costs = adapter_->BatchGetEstimatedCosts(sqls);
    for (size_t i = 0; i < costs.size() && i < missing.size(); i++) {
      double base = costs[i].second;
      if (base > 0.0 && base < std::numeric_limits<double>::max())
        base_count_cache_[GetTableName(missing[i])] = base;
    }
  }

  for (auto t : rel_tables) {
    if (table_card_.count(t))
      continue;
    auto it = base_count_cache_.find(GetTableName(t));
    if (it == base_count_cache_.end())
      continue;
    auto conjuncts = ir_utils::CollectFilterConditions(ir, {t});
    table_card_[t] =
        EstimateFilteredCardinality(it->second, GetTableName(t), conjuncts);
  }
}

double TopDownSplitter::EstimateFilteredCardinality(
    double base_cardinality, const std::string &table_name,
    const std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> &conjuncts) {
  double card = base_cardinality;
  bool has_other_filter = false;

  auto apply_equalities = [&](const std::string &column, double k) {
    double distinct = distinct_cache_.Get(*adapter_, table_name, column);
    if (distinct > 0.0) {
      double cand = std::min(base_cardinality,
                             std::ceil(base_cardinality * k / distinct));
      card = std::min(card, std::max(cand, 1.0));
    } else {
      has_other_filter = true;
    }
  };

  // CollectFilterConditions returns one expr per qual, so a multi-conjunct
  // AND arrives as a single LogicalExprNode: flatten before classifying.
  std::function<void(const ir_sql_converter::AQPExpr *)> classify =
      [&](const ir_sql_converter::AQPExpr *c) {
        if (!c)
          return;
        switch (c->GetNodeType()) {
        case ir_sql_converter::SimplestNodeType::LogicalExprNode: {
          auto *le =
              dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(c);
          if (le && le->GetLogicalOp() == ir_sql_converter::LogicalAnd) {
            classify(le->left_expr.get());
            classify(le->right_expr.get());
          } else {
            has_other_filter = true;
          }
          break;
        }
        case ir_sql_converter::SimplestNodeType::VarConstComparisonNode: {
          auto *vcc = dynamic_cast<
              const ir_sql_converter::SimplestVarConstComparison *>(c);
          if (vcc && vcc->attr &&
              vcc->GetSimplestExprType() == ir_sql_converter::Equal)
            apply_equalities(vcc->attr->GetColumnName(), 1.0);
          else
            has_other_filter = true;
          break;
        }
        case ir_sql_converter::SimplestNodeType::InExprNode: {
          auto *in =
              dynamic_cast<const ir_sql_converter::SimplestInExpr *>(c);
          if (in && in->attr && !in->negated && !in->values.empty())
            apply_equalities(in->attr->GetColumnName(),
                             static_cast<double>(in->values.size()));
          else
            has_other_filter = true;
          break;
        }
        default:
          has_other_filter = true;
          break;
        }
      };
  for (const auto &c : conjuncts)
    classify(c.get());

  bool has_equality = card != base_cardinality;
  if (!has_equality && has_other_filter)
    card = std::max(base_cardinality * 0.2, 1.0);
  return card;
}

void TopDownSplitter::CollectRelations(
    const ir_sql_converter::AQPStmt *node, std::vector<JoinRel> &rels,
    std::vector<unsigned int> &rel_tables) const {
  if (!node)
    return;
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::SetOpNode)
    return;
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = dynamic_cast<const ir_sql_converter::SimplestScan *>(node);
    if (scan) {
      JoinRel rel;
      rel.table_index = scan->GetTableIndex();
      rel.table_name = scan->GetTableName();
      rel.is_temp = temp_indices_.count(rel.table_index) > 0;
      auto it = table_card_.find(rel.table_index);
      if (it != table_card_.end()) {
        rel.cardinality = it->second;
      } else {
        double card = static_cast<double>(scan->GetEstimatedCardinality());
        // Unknown cardinality: assume huge so the pair is never picked early.
        rel.cardinality = card > 0.0 ? card : 1e9;
      }
      rel_tables.push_back(rel.table_index);
      rels.push_back(std::move(rel));
    }
  }
  for (const auto &child : node->children)
    CollectRelations(child.get(), rels, rel_tables);
}

// Collect the table indices referenced by an expression. Returns false
// (conservative "don't touch") on any node kind it does not understand.
static bool CollectExprTables(const ir_sql_converter::AQPExpr *e,
                              std::set<unsigned int> &tables) {
  using namespace ir_sql_converter;
  if (!e)
    return false;
  switch (e->GetNodeType()) {
  case SimplestNodeType::VarConstComparisonNode: {
    auto *c = dynamic_cast<const SimplestVarConstComparison *>(e);
    if (!c || !c->attr)
      return false;
    tables.insert(c->attr->GetTableIndex());
    return true;
  }
  case SimplestNodeType::VarComparisonNode: {
    auto *c = dynamic_cast<const SimplestVarComparison *>(e);
    if (!c || !c->left_attr || !c->right_attr)
      return false;
    tables.insert(c->left_attr->GetTableIndex());
    tables.insert(c->right_attr->GetTableIndex());
    return true;
  }
  case SimplestNodeType::IsNullExprNode: {
    auto *c = dynamic_cast<const SimplestIsNullExpr *>(e);
    if (!c || !c->attr)
      return false;
    tables.insert(c->attr->GetTableIndex());
    return true;
  }
  case SimplestNodeType::InExprNode: {
    auto *c = dynamic_cast<const SimplestInExpr *>(e);
    if (!c || !c->attr)
      return false;
    tables.insert(c->attr->GetTableIndex());
    return true;
  }
  case SimplestNodeType::LogicalExprNode: {
    auto *c = dynamic_cast<const SimplestLogicalExpr *>(e);
    if (!c)
      return false;
    if (c->GetLogicalOp() != SimplestLogicalOp::LogicalNot &&
        !CollectExprTables(c->left_expr.get(), tables))
      return false;
    return CollectExprTables(c->right_expr.get(), tables);
  }
  case SimplestNodeType::SingleAttrExprNode: {
    auto *c = dynamic_cast<const SimplestSingleAttrExpr *>(e);
    if (!c || !c->attr)
      return false;
    tables.insert(c->attr->GetTableIndex());
    return true;
  }
  case SimplestNodeType::ArithExprNode: {
    auto *c = dynamic_cast<const SimplestArithExpr *>(e);
    if (!c)
      return false;
    bool ok = true;
    if (c->left)
      ok = CollectExprTables(c->left.get(), tables);
    if (ok && c->right)
      ok = CollectExprTables(c->right.get(), tables);
    return ok;
  }
  case SimplestNodeType::CastExprNode: {
    auto *c = dynamic_cast<const SimplestCastExpr *>(e);
    if (!c)
      return false;
    return CollectExprTables(c->child.get(), tables);
  }
  case SimplestNodeType::ExprNode: {
    auto *c = dynamic_cast<const SimplestGeneralComparison *>(e);
    if (!c)
      return false;
    return CollectExprTables(c->left_expr.get(), tables) &&
           CollectExprTables(c->right_expr.get(), tables);
  }
  case SimplestNodeType::ConstVarNode:
    return true;
  default:
    return false;
  }
}

// Check whether a single (non-AND) expression conjunct references 2+ tables
// and is not a simple VarComparison (which is handled as a join edge).
// If so, those tables must stay in the same cluster.
static void CheckColocateConjunct(
    const ir_sql_converter::AQPExpr *expr,
    std::vector<std::set<unsigned int>> &groups) {
  if (!expr)
    return;
  auto nt = expr->GetNodeType();
  // Decompose AND into individual conjuncts — PG IR chains independent
  // single-table filters into one LogicalAnd tree in qual_vec[0].
  if (nt == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *le = dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (le && le->GetLogicalOp() == ir_sql_converter::LogicalAnd) {
      CheckColocateConjunct(le->left_expr.get(), groups);
      CheckColocateConjunct(le->right_expr.get(), groups);
      return;
    }
  }
  // VarComparison is a join edge, not a colocate constraint.
  if (nt == ir_sql_converter::SimplestNodeType::VarComparisonNode)
    return;
  std::set<unsigned int> expr_tables;
  bool ok = CollectExprTables(expr, expr_tables);
#ifndef NDEBUG
  std::cout << "[SDS]   conjunct node_type=" << (int)nt
            << " ok=" << ok << " tables={";
  for (auto t : expr_tables)
    std::cout << t << " ";
  std::cout << "}" << std::endl;
#endif
  if (ok && expr_tables.size() >= 2)
    groups.push_back(expr_tables);
}

// Collect sets of table indices that MUST stay in the same cluster because
// they share unsplittable predicates (general comparisons, OR clauses
// spanning multiple tables). Each entry is a set of >= 2 table indices.
// LogicalAnd is decomposed so that independent single-table filters chained
// by AND (common in PG IR) do not create false colocate constraints.
static void CollectMustColocateSets(
    const ir_sql_converter::AQPStmt *node,
    std::vector<std::set<unsigned int>> &groups) {
  if (!node)
    return;
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::SetOpNode)
    return;
#ifndef NDEBUG
  std::cout << "[SDS] CollectMustColocateSets: node type="
            << (int)node->GetNodeType() << " qual_vec.size="
            << node->qual_vec.size() << " children=" << node->children.size()
            << std::endl;
#endif
  for (const auto &qual : node->qual_vec)
    CheckColocateConjunct(qual.get(), groups);
  for (const auto &child : node->children)
    CollectMustColocateSets(child.get(), groups);
}

void TopDownSplitter::CollectEdges(
    const ir_sql_converter::AQPStmt *node,
    const std::map<unsigned int, int> &table_to_pos,
    std::vector<JoinEdge> &edges) const {
  if (!node)
    return;

  auto add_edge = [&](const ir_sql_converter::SimplestVarComparison *cond) {
    if (!cond || !cond->left_attr || !cond->right_attr)
      return;
    auto lit = table_to_pos.find(cond->left_attr->GetTableIndex());
    auto rit = table_to_pos.find(cond->right_attr->GetTableIndex());
    if (lit == table_to_pos.end() || rit == table_to_pos.end() ||
        lit->second == rit->second)
      return;
    JoinEdge e;
    e.left_rel = lit->second;
    e.right_rel = rit->second;
    e.left_col = cond->left_attr->GetColumnIndex();
    e.right_col = cond->right_attr->GetColumnIndex();
    e.left_col_name = cond->left_attr->GetColumnName();
    e.right_col_name = cond->right_attr->GetColumnName();
    e.is_equality = (cond->GetSimplestExprType() == ir_sql_converter::Equal);
    edges.push_back(std::move(e));
  };

  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *join = dynamic_cast<const ir_sql_converter::SimplestJoin *>(node);
    // Mark/Semi/Anti joins are excluded: Mark's chunk side is not a
    // relation; Semi/Anti subtrees are locked (CollectMarkInfo).
    if (join) {
      auto jt = join->GetSimplestJoinType();
      if (jt != ir_sql_converter::SimplestJoinType::Mark &&
          jt != ir_sql_converter::SimplestJoinType::Semi &&
          jt != ir_sql_converter::SimplestJoinType::Anti &&
          jt != ir_sql_converter::SimplestJoinType::Left &&
          jt != ir_sql_converter::SimplestJoinType::Right &&
          jt != ir_sql_converter::SimplestJoinType::Full) {
        for (const auto &cond : join->join_conditions) {
          if (cond)
            add_edge(cond.get());
        }
      }
    }
  }
  // Add non-equality connectivity edges for cross-table general expressions
  // (e.g., BETWEEN with date arithmetic). Decompose LogicalAnd first so that
  // independent single-table filters chained by AND (common in PG IR) do not
  // create spurious edges.
  auto add_general_edges =
      [&](const ir_sql_converter::AQPExpr *expr) {
    auto nt = expr->GetNodeType();
    if (nt == ir_sql_converter::SimplestNodeType::VarComparisonNode)
      return;
    std::set<unsigned int> expr_tables;
    if (CollectExprTables(expr, expr_tables) && expr_tables.size() >= 2) {
      std::vector<unsigned int> tvec(expr_tables.begin(), expr_tables.end());
      for (size_t i = 0; i < tvec.size(); ++i) {
        for (size_t j = i + 1; j < tvec.size(); ++j) {
          auto lit = table_to_pos.find(tvec[i]);
          auto rit = table_to_pos.find(tvec[j]);
          if (lit != table_to_pos.end() && rit != table_to_pos.end() &&
              lit->second != rit->second) {
            JoinEdge e;
            e.left_rel = lit->second;
            e.right_rel = rit->second;
            e.left_col = 0;
            e.right_col = 0;
            e.is_equality = false;
            edges.push_back(std::move(e));
          }
        }
      }
    }
  };
  for (const auto &qual : node->qual_vec) {
    ForEachJoinPredicate(qual.get(), add_edge);
    ForEachConjunct(qual.get(), add_general_edges);
  }

  for (const auto &child : node->children) {
    if (child && child->GetNodeType() != ir_sql_converter::SimplestNodeType::SetOpNode)
      CollectEdges(child.get(), table_to_pos, edges);
  }
}

bool TopDownSplitter::PlanNext(
    const ir_sql_converter::AQPStmt *remaining_ir) {
  planned_for_ = remaining_ir;
  planned_splittable_ = false;
  planned_group_.clear();
  planned_card_ = 0.0;

  std::vector<JoinRel> rels;
  std::vector<unsigned int> rel_tables;
  CollectRelations(remaining_ir, rels, rel_tables);
  if (rels.size() <= 2) {
#ifndef NDEBUG
    std::cout << "[SDS] PlanNext: only " << rels.size()
              << " relation(s) remain -> final" << std::endl;
#endif
    return false;
  }

  for (auto &r : rels) {
    if (r.is_temp)
      continue;
    if (!bg_mode_) {
      double bc = distinct_cache_.GetRowCount(*adapter_, r.table_name);
      if (bc > 0.0)
        r.base_cardinality = bc;
    }
  }

  std::map<unsigned int, int> table_to_pos;
  for (size_t i = 0; i < rel_tables.size(); i++)
    table_to_pos[rel_tables[i]] = static_cast<int>(i);

  std::vector<JoinEdge> edges;
  CollectEdges(remaining_ir, table_to_pos, edges);
  const std::vector<JoinEdge> guard_edges = edges; // for boundary-rule checks

  for (const auto &e : edges) {
    for (int side = 0; side < 2; side++) {
      int pos = side == 0 ? e.left_rel : e.right_rel;
      const std::string &col = side == 0 ? e.left_col_name : e.right_col_name;
      if (rels[pos].is_temp || col.empty())
        continue;
      double d = bg_mode_ ? distinct_cache_.GetCached(rels[pos].table_name, col)
                          : distinct_cache_.Get(*adapter_, rels[pos].table_name, col);
      if (d > 0.0) {
        auto &hint = col_distinct_hints_[col];
        hint = std::max(hint, d);
      }
    }
  }

  IRJoinOptimizer optimizer(
      std::move(rels), std::move(edges),
      [this](const std::string &table, const std::string &column) {
        if (table.rfind("temp", 0) == 0) {
          auto it = col_distinct_hints_.find(column);
          if (it != col_distinct_hints_.end())
            return it->second;
          double best = -1.0;
          size_t best_len = 0;
          for (const auto &[name, d] : col_distinct_hints_) {
            if (name.size() >= column.size() || name.size() <= best_len)
              continue;
            size_t off = column.size() - name.size();
            if (column[off - 1] == '_' &&
                column.compare(off, std::string::npos, name) == 0) {
              best = d;
              best_len = name.size();
            }
          }
          return best;
        }
        return bg_mode_ ? distinct_cache_.GetCached(table, column)
                        : distinct_cache_.Get(*adapter_, table, column);
      },
      [this](const std::string &table, const std::string &column) {
        return bg_mode_ ? -1.0
                        : distinct_cache_.GetCorrelation(*adapter_, table, column);
      });
  auto tree = optimizer.Solve();
  if (!tree) {
#ifndef NDEBUG
    std::cout << "[SDS] PlanNext: Solve() failed over " << rel_tables.size()
              << " relation(s)" << std::endl;
#endif
    return false;
  }

  // Candidate groups: maximal subtrees of the optimal tree that do NOT
  // contain the largest relation (the probe side must never be materialized
  // — scanning it belongs in the final query, once) and whose estimated
  // output is under the threshold. This mirrors node-based's boundary rule
  // (materialize build-side subtrees, keep the probe spine) on OUR DP tree,
  // and materializes a whole k-table subtree in ONE round trip instead of a
  // chain of pair temps.
  auto base_card = [](const JoinRel &r) {
    return r.base_cardinality > 0.0 ? r.base_cardinality : r.cardinality;
  };
  // Largest SCAN (base rows), not largest post-filter estimate: the probe
  // side is a physical property of the table, and filtered estimates made
  // mk (4.5M base) lose "max" status to smaller tables (26c).
  int max_pos = 0;
  for (size_t i = 1; i < optimizer.NumRelations(); i++)
    if (base_card(optimizer.Relation((int)i)) >
        base_card(optimizer.Relation(max_pos)))
      max_pos = (int)i;
  const uint64_t max_bit = (uint64_t)1 << max_pos;

  // Must-colocate constraints: table groups tied by unsplittable predicates
  // (general comparisons, OR clauses spanning multiple tables). A candidate
  // that includes SOME but not ALL tables of a group would lose the predicate.
  std::vector<std::set<unsigned int>> colocate_groups;
  CollectMustColocateSets(remaining_ir, colocate_groups);
  // Convert to position bitmasks for fast subtree checks.
  std::vector<uint64_t> colocate_masks;
  for (const auto &grp : colocate_groups) {
    uint64_t mask = 0;
    for (unsigned int ti : grp) {
      auto it = table_to_pos.find(ti);
      if (it != table_to_pos.end())
        mask |= (uint64_t)1 << it->second;
    }
    if (__builtin_popcountll(mask) >= 2) {
      colocate_masks.push_back(mask);
#ifndef NDEBUG
      std::cout << "[SDS] colocate mask 0x" << std::hex << mask << std::dec
                << " tables:";
      for (unsigned int ti : grp)
        std::cout << " " << ti;
      std::cout << std::endl;
#endif
    }
  }

  auto subtree_ok = [&](const JoinTree *t) -> bool {
    if (!t || t->IsLeaf() || t->cross_product)
      return false;
    if (t->card >= kSplitCardThreshold)
      return false;
    for (uint64_t m = t->set; m; m &= m - 1) {
      int pos = __builtin_ctzll(m);
      if (mark_locked_.count(optimizer.Relation(pos).table_index))
        return false;
    }
    // Reject if splitting a must-colocate group: the candidate includes
    // some but not all tables of a colocate set.
    for (uint64_t cm : colocate_masks) {
      uint64_t overlap = t->set & cm;
      if (overlap != 0 && overlap != cm)
        return false;
    }
    return true;
  };

  // Primary rule (node-based granularity): materialize the cheapest
  // bottom-level join — both children are base/temp relations — of the
  // optimal tree. Pair temps maximize feedback (every re-plan sees the
  // exact cardinality) and give the engine one bloom-filtered probe per
  // boundary; coarse multi-table groups measured 2-4x slower on the
  // cast_info-heavy queries (19a: 140 ms vs node-based's 66 ms for the
  // same tables) because one bad internal estimate poisons a whole group
  // and the selective cross-boundary partner never prunes the big scan.
  // A pair containing the largest relation is only allowed when its output
  // is no larger than some subtree it must later join (build-side test =
  // node-based's right-child rule); if it dominates everything left it IS
  // the final probe and must not be materialized (7c: est 112K but actual
  // 1.6M rows of cast_info x prolific-people correlation).
  // Collect ALL qualifying pairs ranked by estimated cardinality; the
  // cheapest is tried first but may be rejected by engine EXPLAIN validation.
  struct PairCandidate {
    const JoinTree *node;
    bool is_noop;
  };
  std::vector<PairCandidate> pair_cands;
  std::vector<PairCandidate> noop_cands;
  std::vector<PairCandidate> max_rejected_cands;
  // Big x temp pairs rejected only because OUR fanout estimate is untrusted;
  // re-examined below with an engine EXPLAIN (exact temp stats). Second
  // element = the temp's exact cardinality (expansion bound for acceptance).
  std::vector<std::pair<std::set<unsigned int>, double>> rejected_pairs;
  std::function<void(const JoinTree *, double)> visit_pairs =
      [&](const JoinTree *t, double max_sibling_card) {
        if (!t || t->IsLeaf())
          return;
        if (t->left && t->right && t->left->IsLeaf() && t->right->IsLeaf()) {
          bool ok = subtree_ok(t);
          bool is_noop = false;
          if (ok) {
            const JoinRel &lr = optimizer.Relation(t->left->rel);
            const JoinRel &rr = optimizer.Relation(t->right->rel);
            if (!lr.is_temp && !rr.is_temp &&
                base_card(lr) >= IRJoinOptimizer::kProbeLocalityMinRows &&
                base_card(rr) >= IRJoinOptimizer::kProbeLocalityMinRows) {
              ok = false;
#ifndef NDEBUG
              std::cout << "[SDS] PlanNext: big-big pair rejected ("
                        << lr.table_name << " x " << rr.table_name << ")"
                        << std::endl;
#endif
            }
            if (ok && lr.is_temp != rr.is_temp) {
              const bool left_base = !lr.is_temp;
              const JoinRel &base = left_base ? lr : rr;
              const int base_pos = left_base ? t->left->rel : t->right->rel;
              const int temp_pos = left_base ? t->right->rel : t->left->rel;
              bool any_unique = false, all_unique = true, any_edge = false;
              for (const auto &e : guard_edges) {
                if (!e.is_equality)
                  continue;
                const std::string *col = nullptr;
                if (e.left_rel == base_pos && e.right_rel == temp_pos)
                  col = &e.left_col_name;
                else if (e.right_rel == base_pos && e.left_rel == temp_pos)
                  col = &e.right_col_name;
                if (!col)
                  continue;
                any_edge = true;
                double dc = bg_mode_
                    ? distinct_cache_.GetCached(base.table_name, *col)
                    : distinct_cache_.Get(*adapter_, base.table_name, *col);
                if (dc > 0.9 * base_card(base))
                  any_unique = true;
                else
                  all_unique = false;
              }
              if (base_card(base) >=
                      IRJoinOptimizer::kProbeLocalityMinRows &&
                  !any_unique) {
                ok = false;
                const JoinRel &tr = optimizer.Relation(temp_pos);
                rejected_pairs.push_back(
                    {{base.table_index, tr.table_index}, tr.cardinality});
#ifndef NDEBUG
                std::cout << "[SDS] PlanNext: big x temp pair rejected ("
                          << base.table_name
                          << " joins temp on fanout column)" << std::endl;
#endif
              }
              if (ok && any_edge && all_unique &&
                  base.cardinality >= 0.9 * base_card(base)) {
                is_noop = true;
#ifndef NDEBUG
                std::cout << "[SDS] PlanNext: no-op pair demoted ("
                          << base.table_name
                          << " unfiltered, unique-column join)" << std::endl;
#endif
              }
            }
          }
          if (ok && !is_noop) {
            const JoinRel &lr = optimizer.Relation(t->left->rel);
            const JoinRel &rr = optimizer.Relation(t->right->rel);
            if (t->card >= 0.95 * (lr.cardinality + rr.cardinality)) {
              is_noop = true;
#ifndef NDEBUG
              std::cout << "[SDS] PlanNext: no-reduction pair demoted ("
                        << lr.table_name << " x " << rr.table_name
                        << " est=" << t->card << ")" << std::endl;
#endif
            }
          }
          if (ok && (t->set & max_bit) && t->card > max_sibling_card) {
            bool has_temp_child = false;
            for (uint64_t m = t->set; m; m &= m - 1)
              if (optimizer.Relation(__builtin_ctzll(m)).is_temp) {
                has_temp_child = true;
                break;
              }
            if (has_temp_child)
              max_rejected_cands.push_back({t, is_noop});
            ok = false;
#ifndef NDEBUG
            std::cout << "[SDS] PlanNext: max-side pair rejected (card="
                      << t->card << " max_sibling=" << max_sibling_card
                      << (has_temp_child ? " saved-for-explain" : "")
                      << ")" << std::endl;
#endif
          }
          if (ok) {
            if (is_noop)
              noop_cands.push_back({t, true});
            else
              pair_cands.push_back({t, false});
          }
          return;
        }
        if (t->left && t->right) {
          visit_pairs(t->left.get(),
                      std::max(max_sibling_card, t->right->card));
          visit_pairs(t->right.get(),
                      std::max(max_sibling_card, t->left->card));
        }
      };
  visit_pairs(tree.get(), 0.0);

  // Sort candidates by estimated cardinality (cheapest first).
  auto by_card = [](const PairCandidate &a, const PairCandidate &b) {
    return a.node->card < b.node->card;
  };
  std::sort(pair_cands.begin(), pair_cands.end(), by_card);
  std::sort(noop_cands.begin(), noop_cands.end(), by_card);

  // EXPLAIN-based pair validation: the tdom model has 12-188× errors on
  // problematic pairs involving temps (14a it×mi, 26c ci×temp). DuckDB's
  // in-process EXPLAIN costs 0.3-1 ms and is 10-100× more accurate on
  // temp joins (exact temp stats). Base×base pairs share the same
  // estimation errors with EXPLAIN (e.g. 10c cn×mc: both say ~14K,
  // actual 1.15M due to filter skew), so only temp-involving pairs are
  // validated.
  constexpr double kExplainRejectRatio = 3.0;
  auto explain_validate = [&](const PairCandidate &cand)
      -> bool {
    bool has_temp = false;
    std::set<unsigned int> tables;
    for (uint64_t m = cand.node->set; m; m &= m - 1) {
      int pos = __builtin_ctzll(m);
      const JoinRel &r = optimizer.Relation(pos);
      tables.insert(r.table_index);
      if (r.is_temp)
        has_temp = true;
    }
    if (!has_temp)
      return true;
    auto sub_ir = BuildSubIRForCluster(
        const_cast<ir_sql_converter::AQPStmt *>(remaining_ir), tables);
    if (!sub_ir)
      return true; // can't build sub-IR: trust DP
    std::string sql = ir_sql_converter::ConvertIRToSQL(*sub_ir, 0);
    if (sql.empty())
      return true;
    auto cache_it = explain_cache_.find(sql);
    double engine_est;
    if (cache_it != explain_cache_.end()) {
      engine_est = cache_it->second.second;
    } else {
      if (bg_mode_)
        throw std::runtime_error("SDS bg-mode: engine EXPLAIN needed");
      auto costs = adapter_->BatchGetEstimatedCosts({sql});
      if (costs.empty())
        return true;
      explain_cache_[sql] = costs[0];
      engine_est = costs[0].second;
    }
    if (engine_est <= 0.0)
      return true;
    bool reject = engine_est > kSplitCardThreshold ||
                  (engine_est > kExplainRejectRatio * cand.node->card &&
                   engine_est > kSplitCardThreshold * 0.5);
#ifndef NDEBUG
    if (reject) {
      std::cout << "[SDS] PlanNext: EXPLAIN rejected pair {";
      for (auto t : tables)
        std::cout << GetTableName(t) << " ";
      std::cout << "} dp_est=" << cand.node->card
                << " engine_est=" << engine_est << std::endl;
    }
#endif
    return !reject;
  };

  const JoinTree *best = nullptr;
  for (auto &cand : pair_cands) {
    if (explain_validate(cand)) {
      best = cand.node;
      break;
    }
  }
  const JoinTree *best_noop = nullptr;
  if (!best) {
    for (auto &cand : noop_cands) {
      if (explain_validate(cand)) {
        best_noop = cand.node;
        break;
      }
    }
  }
  // Max-side rejected pairs with engine EXPLAIN: when no normal pair or
  // no-op qualifies, try pairs containing the max relation that were
  // rejected by the max-sibling-card rule. The engine sees exact temp
  // stats, so its estimate for max×temp pairs is far more accurate than
  // the tdom model (26c/30c: ci×temp pairs otherwise stop splitting).
  if (!best && !best_noop) {
    std::sort(max_rejected_cands.begin(), max_rejected_cands.end(), by_card);
    for (auto &cand : max_rejected_cands) {
      if (explain_validate(cand)) {
        best = cand.node;
#ifndef NDEBUG
        std::cout << "[SDS] PlanNext: max-side pair accepted via EXPLAIN"
                  << std::endl;
#endif
        break;
      }
    }
  }

  const bool noop_selected = !best && best_noop;
  if (!best)
    best = best_noop;

  // Grow the chosen pair into the largest enclosing subtree that stays under
  // the threshold and contains only small base relations and temps (13d/9c:
  // chains of pair temps each pay a separate scan+materialization round
  // trip, 5-19 ms apiece; one k-table sub-SQL does the same joins in a
  // single pipeline — the engine re-orders inside the group anyway). Big
  // base tables never join a group this way: their pairs stay
  // locality-steered probes, and the global max stays in the final query.
  // v18 hybrid: growth only under AQP_TD_V17 — binary NB-granularity pairs
  // qualify for the adapter's pending-IR fast path (closed IR->plan
  // constructor), which removes the per-sub parse+optimize round trip.
  if (best && V17Mode()) {
    std::function<bool(const JoinTree *)> no_cross =
        [&](const JoinTree *t) -> bool {
      if (!t)
        return true;
      if (t->cross_product)
        return false;
      return no_cross(t->left.get()) && no_cross(t->right.get());
    };
    auto growable = [&](const JoinTree *t) -> bool {
      // Big bases never grow into a group, filtered or not (v4 tried
      // filtered-big growth: 9c gained 12 ms but the 19-family/9a/25a/
      // 30c/31a band paid +20-27 ms each — coarse all-base groups lose
      // the cross-boundary bloom pruning; net -0.4 s. The max stays out
      // for the same reason).
      if (!subtree_ok(t) || (t->set & max_bit) || !no_cross(t))
        return false;
      for (uint64_t m = t->set; m; m &= m - 1) {
        const JoinRel &r = optimizer.Relation(__builtin_ctzll(m));
        if (!r.is_temp &&
            base_card(r) >= IRJoinOptimizer::kProbeLocalityMinRows)
          return false;
      }
      return true;
    };
    std::vector<const JoinTree *> ancestors; // bottom-up from best's parent
    std::function<bool(const JoinTree *)> find_path =
        [&](const JoinTree *t) -> bool {
      if (!t)
        return false;
      if (t == best)
        return true;
      if (!t->IsLeaf() &&
          (find_path(t->left.get()) || find_path(t->right.get()))) {
        ancestors.push_back(t);
        return true;
      }
      return false;
    };
    find_path(tree.get());
    for (const JoinTree *anc : ancestors) {
      if (!growable(anc))
        break;
      best = anc;
    }
  }
  // A LARGE no-op pair that growth could not expand (v18: growth is off, so
  // any large no-op pair) stays a pure no-op: materializing it writes the
  // temp back unreduced (10c ct x temp2 596K -> 596K, 13d t x temp2
  // 248K -> 248K) while the final query has to do the same join anyway —
  // go final instead. SMALL no-ops are still materialized: the write is
  // cheap and shrinking the final query's relation count protects against
  // engine mis-ordering (17-family: skipping the 42K t x temp1 no-op cost
  // +19-35 ms in the 4-relation final).
  if (best && noop_selected && best == best_noop && best->card >= 1e5) {
#ifndef NDEBUG
    std::cout << "[SDS] PlanNext: unexpandable no-op pair -> final"
              << std::endl;
#endif
    return false;
  }

  // Fallback: cheapest maximal max-free subtree under the threshold. Covers
  // the case where every bottom join overflows the threshold but a higher
  // subtree's output is small again (filters/marks applied above the join).
  if (!best) {
    std::function<void(const JoinTree *)> visit = [&](const JoinTree *t) {
      if (!t || t->IsLeaf())
        return;
      if (!(t->set & max_bit) && subtree_ok(t)) {
        if (!best || t->card < best->card)
          best = t; // maximal qualifying subtree: don't recurse inside
        return;
      }
      visit(t->left.get());
      visit(t->right.get());
    };
    visit(tree.get());
  }

  // Engine-validated fallback: when the DP tree has no qualifying pair,
  // try EXPLAIN-validating candidates our tdom model distrusts. Two sources:
  // (a) connected components of the join graph minus the max relation, and
  // (b) fanout-rejected big×temp pairs (rejected_pairs).
  // The engine sees exact temp stats, so its estimate is far more
  // trustworthy than the tdom model on temp-involving pairs.
  if (!best && optimizer.NumRelations() >= 4) {
    constexpr double kComponentEstLimit = 100000.0;
    constexpr double kPairEstFloor = 1000.0;

    const size_t n = optimizer.NumRelations();
    std::vector<std::vector<int>> adj(n);
    for (const auto &e : guard_edges) {
      if (e.left_rel == max_pos || e.right_rel == max_pos)
        continue;
      adj[e.left_rel].push_back(e.right_rel);
      adj[e.right_rel].push_back(e.left_rel);
    }
    std::vector<int> comp(n, -1);
    int ncomp = 0;
    for (size_t s = 0; s < n; s++) {
      if ((int)s == max_pos || comp[s] >= 0)
        continue;
      std::vector<int> stack{(int)s};
      comp[s] = ncomp;
      while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();
        for (int v : adj[u])
          if (comp[v] < 0) {
            comp[v] = ncomp;
            stack.push_back(v);
          }
      }
      ncomp++;
    }
    std::vector<std::set<unsigned int>> cand_tables(ncomp);
    std::vector<int> cand_width(ncomp, 0);
    std::vector<bool> cand_ok(ncomp, true);
    std::vector<double> cand_limit(ncomp, kComponentEstLimit);
    std::vector<uint64_t> cand_pos_mask(ncomp, 0);
    for (size_t i = 0; i < n; i++) {
      if ((int)i == max_pos || comp[i] < 0)
        continue;
      const JoinRel &r = optimizer.Relation((int)i);
      cand_tables[comp[i]].insert(r.table_index);
      cand_pos_mask[comp[i]] |= (uint64_t)1 << i;
      cand_width[comp[i]]++;
      if (mark_locked_.count(r.table_index))
        cand_ok[comp[i]] = false;
    }
    for (int c = 0; c < ncomp; c++) {
      if (!cand_ok[c])
        continue;
      for (uint64_t cm : colocate_masks) {
        uint64_t overlap = cand_pos_mask[c] & cm;
        if (overlap != 0 && overlap != cm) {
          cand_ok[c] = false;
          break;
        }
      }
    }

    for (const auto &p : rejected_pairs) {
      cand_tables.push_back(p.first);
      cand_width.push_back((int)p.first.size());
      uint64_t pmask = 0;
      for (unsigned int ti : p.first) {
        auto pit = table_to_pos.find(ti);
        if (pit != table_to_pos.end())
          pmask |= (uint64_t)1 << pit->second;
      }
      cand_pos_mask.push_back(pmask);
      bool ok = true;
      for (uint64_t cm : colocate_masks) {
        uint64_t overlap = pmask & cm;
        if (overlap != 0 && overlap != cm) {
          ok = false;
          break;
        }
      }
      cand_ok.push_back(ok);
      double lim = std::max(p.second, kPairEstFloor);
      cand_limit.push_back(lim < kSplitCardThreshold ? lim
                                                     : kSplitCardThreshold);
      ncomp++;
    }

    std::vector<std::string> sqls(ncomp);
    for (int i = 0; i < ncomp; i++) {
      if (!cand_ok[i] || cand_width[i] < 2)
        continue;
      auto sub_ir = BuildSubIRForCluster(
          const_cast<ir_sql_converter::AQPStmt *>(remaining_ir),
          cand_tables[i]);
      if (sub_ir)
        sqls[i] = ir_sql_converter::ConvertIRToSQL(*sub_ir, 0);
    }

    std::vector<std::string> uncached;
    for (const auto &s : sqls)
      if (!s.empty() && !explain_cache_.count(s))
        uncached.push_back(s);
    if (!uncached.empty()) {
      if (bg_mode_)
        throw std::runtime_error("SDS bg-mode: engine EXPLAIN needed");
      auto costs = adapter_->BatchGetEstimatedCosts(uncached);
      for (size_t i = 0; i < costs.size() && i < uncached.size(); i++)
        explain_cache_[uncached[i]] = costs[i];
    }

    int chosen = -1;
    int chosen_width = 0;
    double chosen_est = 0.0;
    for (int i = 0; i < ncomp; i++) {
#ifndef NDEBUG
      {
        auto dit = sqls[i].empty() ? explain_cache_.end()
                                   : explain_cache_.find(sqls[i]);
        std::cout << "[SDS] engine-validate cand " << i << " width="
                  << cand_width[i] << (sqls[i].empty() ? " sql=EMPTY" : "")
                  << " est="
                  << (dit != explain_cache_.end() ? dit->second.second : -1.0)
                  << std::endl;
      }
#endif
      if (sqls[i].empty())
        continue;
      auto it = explain_cache_.find(sqls[i]);
      if (it == explain_cache_.end())
        continue;
      double est = it->second.second;
      if (est <= 0.0 || est > cand_limit[i])
        continue;
      int width = cand_width[i];
      if (chosen < 0 || width > chosen_width ||
          (width == chosen_width && est < chosen_est)) {
        chosen = (int)i;
        chosen_width = width;
        chosen_est = est;
      }
    }
    if (chosen >= 0) {
      planned_group_ = cand_tables[chosen];
      planned_card_ = chosen_est;
      planned_splittable_ = true;
#ifndef NDEBUG
      std::cout << "[SDS] PlanNext: engine-validated group {";
      for (auto t : planned_group_)
        std::cout << t << "(" << GetTableName(t) << ") ";
      std::cout << "} engine_est=" << chosen_est << std::endl;
#endif
      return true;
    }
  }

  if (!best) {
#ifndef NDEBUG
    std::cout << "[SDS] PlanNext: no qualifying group among "
              << optimizer.NumRelations() << " relation(s) -> final"
              << std::endl;
#endif
    return false;
  }

  for (uint64_t m = best->set; m; m &= m - 1)
    planned_group_.insert(
        optimizer.Relation(__builtin_ctzll(m)).table_index);
  planned_card_ = best->card;
  planned_splittable_ = true;

#ifndef NDEBUG
  std::cout << "[SDS] PlanNext: group {";
  for (auto t : planned_group_)
    std::cout << t << "(" << GetTableName(t) << ") ";
  std::cout << "} est_card=" << planned_card_ << std::endl;
#endif
  return true;
}

bool TopDownSplitter::IsComplete(
    const ir_sql_converter::AQPStmt *remaining_ir) {
  if (!remaining_ir)
    return true;
  if (planned_for_ != remaining_ir)
    PlanNext(remaining_ir);
  return !planned_splittable_;
}

// v18: move single-table filter conjuncts from the sub-IR's top filter node
// onto the owning plain scan leaf's qual_vec. Rationale: the qjit fast path
// compiles the sub-IR AS IS, and BuildExecutionSteps places above-join
// filters on the probe spine — a build-side conjunct there references
// columns that are not live on the spine. NB's fast-path IRs always come
// from optimized plans (filters pushed onto scans); this restores that
// invariant. SQL rendering is unchanged (ir_to_sql collects scan qual_vec
// into WHERE exactly like filter-node quals). Temp scans are skipped (they
// become ChunkNodes below, and ir_to_sql drops chunk quals); Mark-join
// units are opaque; anything unassignable stays in the filter node.
static void PushFilterConjunctsToScans(
    ir_sql_converter::AQPStmt *root,
    const std::set<unsigned int> &temp_indices) {
  using namespace ir_sql_converter;
  if (!root || root->children.size() != 1 || !root->children[0])
    return;
  auto &filt = root->children[0];
  if (filt->GetNodeType() != SimplestNodeType::FilterNode ||
      filt->children.size() != 1 || !filt->children[0])
    return;

  std::map<unsigned int, AQPStmt *> scan_by_table;
  std::function<void(AQPStmt *)> collect = [&](AQPStmt *n) {
    if (!n)
      return;
    if (n->GetNodeType() == SimplestNodeType::JoinNode) {
      auto *j = dynamic_cast<const SimplestJoin *>(n);
      if (j && j->GetSimplestJoinType() == SimplestJoinType::Mark)
        return; // opaque IN-list unit; leave untouched
    }
    if (n->GetNodeType() == SimplestNodeType::ScanNode) {
      auto *s = dynamic_cast<SimplestScan *>(n);
      if (s && !temp_indices.count(s->GetTableIndex()))
        scan_by_table[s->GetTableIndex()] = n;
      return;
    }
    for (auto &c : n->children)
      collect(c.get());
  };
  collect(filt->children[0].get());

  std::vector<std::unique_ptr<AQPExpr>> kept;
  for (auto &q : filt->qual_vec) {
    std::set<unsigned int> tables;
    if (q && CollectExprTables(q.get(), tables) && tables.size() == 1 &&
        scan_by_table.count(*tables.begin())) {
      scan_by_table[*tables.begin()]->qual_vec.push_back(std::move(q));
    } else {
      kept.push_back(std::move(q));
    }
  }
  filt->qual_vec = std::move(kept);
  if (filt->qual_vec.empty()) {
    // Splice the now-empty filter node out of the tree.
    auto child = std::move(filt->children[0]);
    root->children[0] = std::move(child);
  }
}

// v18: rewrite temp-table SimplestScan leaves as SimplestChunk so the qjit
// fast path treats them as temps (BuildExecutionSteps sets source_is_temp
// only for ChunkNode). SQL rendering is unaffected: ir_to_sql renders a
// named chunk exactly like a table ref. Only qual-free, childless scans are
// wrapped (ir_to_sql drops a ChunkNode's qual_vec; BuildSubIRForCluster
// never puts quals on cloned scans, but guard anyway). Mark-unit chunks are
// untouched (they are already ChunkNodes).
static void WrapTempScansAsChunks(ir_sql_converter::AQPStmt *node,
                                  const std::set<unsigned int> &temp_indices) {
  if (!node)
    return;
  for (auto &child : node->children) {
    if (!child)
      continue;
    if (child->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
      auto *scan = dynamic_cast<ir_sql_converter::SimplestScan *>(child.get());
      if (scan && temp_indices.count(scan->GetTableIndex()) &&
          child->qual_vec.empty() && child->children.empty()) {
        unsigned int idx = scan->GetTableIndex();
        std::string name = scan->GetTableName();
        // Donor ctor moves target_list + estimated_cardinality into the chunk.
        child = std::make_unique<ir_sql_converter::SimplestChunk>(
            std::move(child), idx, name, std::vector<std::string>{});
        continue;
      }
    }
    WrapTempScansAsChunks(child.get(), temp_indices);
  }
}

// v18: BuildSubIRForCluster hardcodes the root projection's table index to
// 0. When the cluster contains base-table index 0, qjit's projection remap
// (ThroughProj) misroutes that table's attrs through the projection,
// producing wrong output dtypes on the fast path (SQL rendering never reads
// the index, so v17 was unaffected). Rebuild the root with a fresh index
// above every table index in the subtree.
static void RenumberRootProjection(
    std::unique_ptr<ir_sql_converter::AQPStmt> &sub_ir) {
  using namespace ir_sql_converter;
  if (!sub_ir || sub_ir->GetNodeType() != SimplestNodeType::ProjectionNode)
    return;
  unsigned int max_idx = 0;
  std::function<void(const AQPStmt *)> walk = [&](const AQPStmt *n) {
    if (!n)
      return;
    if (n->GetNodeType() == SimplestNodeType::ScanNode)
      max_idx = std::max(
          max_idx, n->Cast<SimplestScan>().GetTableIndex());
    else if (n->GetNodeType() == SimplestNodeType::ChunkNode)
      max_idx = std::max(
          max_idx, n->Cast<SimplestChunk>().GetTableIndex());
    for (const auto &c : n->children)
      walk(c.get());
  };
  walk(sub_ir.get());
  if (sub_ir->Cast<SimplestProjection>().GetIndex() > max_idx)
    return;
  sub_ir = std::make_unique<SimplestProjection>(std::move(sub_ir),
                                                max_idx + 1);
}

std::unique_ptr<SubqueryExtraction>
TopDownSplitter::SplitIR(ir_sql_converter::AQPStmt *remaining_ir) {
  split_iteration_++;
  if (!remaining_ir)
    return nullptr;
  if (planned_for_ != remaining_ir)
    PlanNext(remaining_ir);
  if (!planned_splittable_)
    return nullptr;

  auto cluster = planned_group_;
  std::string temp_table_name = "temp_" + std::to_string(split_iteration_);

  auto extraction =
      std::make_unique<SubqueryExtraction>(cluster, temp_table_name);
  extraction->sub_ir = BuildSubIRForCluster(remaining_ir, cluster);
  if (!extraction->sub_ir) {
    throw std::runtime_error(
        "TopDownSplitter unsupported: BuildSubIRForCluster failed for planned "
        "cluster of " + std::to_string(cluster.size()) + " table(s)");
  }
  if (!V17Mode()) {
    PushFilterConjunctsToScans(extraction->sub_ir.get(), temp_indices_);
    WrapTempScansAsChunks(extraction->sub_ir.get(), temp_indices_);
    RenumberRootProjection(extraction->sub_ir);
    // The cloned leaves carry no engine estimates (FilterOptimize-only
    // plan); the fast-path plan constructor picks its build side from leaf
    // estimated_cardinality, so stamp the DP's effective per-table cards
    // (exact for temps).
    std::function<void(ir_sql_converter::AQPStmt *)> stamp =
        [&](ir_sql_converter::AQPStmt *n) {
          if (!n)
            return;
          unsigned int ti = 0;
          bool is_leaf = false;
          if (n->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
            ti = n->Cast<ir_sql_converter::SimplestScan>().GetTableIndex();
            is_leaf = true;
          } else if (n->GetNodeType() ==
                     ir_sql_converter::SimplestNodeType::ChunkNode) {
            ti = n->Cast<ir_sql_converter::SimplestChunk>().GetTableIndex();
            is_leaf = true;
          }
          if (is_leaf) {
            auto it = table_card_.find(ti);
            if (it != table_card_.end() && it->second >= 1.0)
              n->SetEstimatedCardinality(
                  static_cast<uint64_t>(it->second));
            return;
          }
          for (const auto &c : n->children)
            stamp(c.get());
        };
    stamp(extraction->sub_ir.get());
  }
  extraction->estimated_rows = planned_card_;

  for (auto t : cluster)
    executed_tables_.insert(t);

#ifndef NDEBUG
  std::cout << "[SDS] Iteration " << split_iteration_ << ": extracted "
            << temp_table_name << " est_rows=" << planned_card_ << std::endl;
#endif
  return extraction;
}

std::unique_ptr<ir_sql_converter::AQPStmt> TopDownSplitter::UpdateRemainingIR(
    std::unique_ptr<ir_sql_converter::AQPStmt> remaining_ir,
    const std::set<unsigned int> &executed_table_indices,
    unsigned int temp_table_index, const std::string &temp_table_name,
    uint64_t temp_table_cardinality,
    const std::vector<std::pair<unsigned int, unsigned int>> &column_mappings,
    const std::vector<std::string> &column_names) {
  auto updated = FKBasedSplitter::UpdateRemainingIR(
      std::move(remaining_ir), executed_table_indices, temp_table_index,
      temp_table_name, temp_table_cardinality, column_mappings, column_names);

  // Feedback: the temp's EXACT cardinality drives the next DP round.
  temp_indices_.insert(temp_table_index);
  table_card_[temp_table_index] =
      std::max<double>(1.0, static_cast<double>(temp_table_cardinality));
  planned_for_ = nullptr; // invalidate the plan cache
#ifndef NDEBUG
  std::cout << "[SDS] UpdateRemainingIR: temp idx=" << temp_table_index
            << " exact_card=" << temp_table_cardinality
            << " updated=" << (updated != nullptr) << std::endl;
#endif

  return updated;
}

#ifdef HAVE_DUCKDB
void TopDownSplitter::MovePreprocessState(CrossQueryPrepResult &out) {
  out.td_table_card = std::move(table_card_);
  out.td_table_index_to_name = std::move(table_index_to_name_);
  out.td_max_table_index = max_table_index_;
  out.td_mark_in = std::move(mark_in_);
  out.td_mark_locked = std::move(mark_locked_);
  out.td_col_distinct_hints = std::move(col_distinct_hints_);
  out.td_join_graph = std::move(join_graph_);
  out.td_current_join_pairs = std::move(current_join_pairs_);
  out.td_is_relationship = std::move(is_relationship_);
}

void TopDownSplitter::InitFromCrossQueryPrep(CrossQueryPrepResult &prep) {
  // If the bg thread already ran iteration 1's SplitIR, resume as if that
  // iteration happened here.
  if (prep.first_extraction) {
    split_iteration_ = prep.td_split_iteration;
    executed_tables_ = std::move(prep.td_executed_tables);
  } else {
    split_iteration_ = 0;
    executed_tables_.clear();
  }
  temp_indices_.clear();
  planned_for_ = nullptr;
  planned_splittable_ = false;
  planned_group_.clear();
  planned_card_ = 0.0;
  explain_cache_.clear();

  table_card_ = std::move(prep.td_table_card);
  table_index_to_name_ = std::move(prep.td_table_index_to_name);
  max_table_index_ = prep.td_max_table_index;
  mark_in_ = std::move(prep.td_mark_in);
  mark_locked_ = std::move(prep.td_mark_locked);
  col_distinct_hints_ = std::move(prep.td_col_distinct_hints);
  join_graph_ = std::move(prep.td_join_graph);
  current_join_pairs_ = std::move(prep.td_current_join_pairs);
  is_relationship_ = std::move(prep.td_is_relationship);
}
#endif

void TopDownSplitter::PrePopulateBaseCountCache() {
  auto rows = distinct_cache_.GetAllCachedRowCounts();
  for (auto &kv : rows) {
    if (base_count_cache_.find(kv.first) == base_count_cache_.end())
      base_count_cache_[kv.first] = kv.second;
  }
}

void TopDownSplitter::CompleteMissingCardinalities(
    std::unique_ptr<ir_sql_converter::AQPStmt> &ir) {
  FetchMissingLeafCardinalities(ir.get());
  for (auto &[t, info] : mark_in_) {
    auto it = table_card_.find(t);
    if (it == table_card_.end())
      continue;
    double flat = std::max(it->second / kSemiSelectivity, 1.0);
    double distinct = distinct_cache_.Get(*adapter_, GetTableName(t),
                                          info.first);
    double est = flat;
    if (distinct > 0.0) {
      double cand = std::ceil(it->second * info.second / distinct);
      est = std::max(est, std::min(it->second, std::max(cand, 1.0)));
    }
    it->second = est;
  }
}

} // namespace middleware
