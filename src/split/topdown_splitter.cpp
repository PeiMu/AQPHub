/*
 * SDS — stats-driven IR splitter (see topdown_splitter.h).
 */

#include "split/topdown_splitter.h"
#include "split/ir_utils.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>

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
          jt == ir_sql_converter::SimplestJoinType::Anti) {
        std::cerr << "[SDS] WARNING: "
                  << (jt == ir_sql_converter::SimplestJoinType::Semi ? "SEMI"
                                                                     : "ANTI")
                  << " join detected in IR; locking subtree tables "
                     "(cardinality model does not handle SEMI/ANTI natively)"
                  << std::endl;
        CollectMarkLockedTables(node, locked, /*under_mark=*/true);
        return;
      }
    }
  }
  for (const auto &child : node->children)
    CollectMarkInfo(child.get(), mark_in, locked);
}

} // namespace

TopDownSplitter::TopDownSplitter(EngineAdapter *adapter, bool /*unused*/)
    : FKBasedSplitter(adapter, BackendEngine::DUCKDB, SplitStrategy::TOP_DOWN,
                      /*enable_analyze=*/false, /*fkeys_path=*/"") {}

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
    double distinct = distinct_cache_.Get(*adapter_, GetTableName(t),
                                          info.first);
    if (distinct > 0.0) {
      double cand = std::ceil(it->second * info.second / distinct);
      est = std::max(est, std::min(it->second, std::max(cand, 1.0)));
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
          jt != ir_sql_converter::SimplestJoinType::Anti) {
        for (const auto &cond : join->join_conditions) {
          if (cond)
            add_edge(cond.get());
        }
      }
    }
  }
  for (const auto &qual : node->qual_vec)
    ForEachJoinPredicate(qual.get(), add_edge);

  for (const auto &child : node->children)
    CollectEdges(child.get(), table_to_pos, edges);
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
    double bc = distinct_cache_.GetRowCount(*adapter_, r.table_name);
    if (bc > 0.0)
      r.base_cardinality = bc;
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
      double d = distinct_cache_.Get(*adapter_, rels[pos].table_name, col);
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
          // Temp columns carry decorated names
          // (<src_table>_<idx>_<col>, nested through temps); the base
          // column name is the trailing component -> longest-suffix match.
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
        return distinct_cache_.Get(*adapter_, table, column);
      },
      [this](const std::string &table, const std::string &column) {
        return distinct_cache_.GetCorrelation(*adapter_, table, column);
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
                if (distinct_cache_.Get(*adapter_, base.table_name, *col) >
                    0.9 * base_card(base))
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
  if (best) {
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
    // A LARGE no-op pair that growth could not expand stays a pure no-op:
    // materializing it writes the temp back unreduced (10c ct x temp2
    // 596K -> 596K, 13d t x temp2 248K -> 248K) while the final query has
    // to do the same join anyway — go final instead. SMALL no-ops are
    // still materialized: the write is cheap and shrinking the final
    // query's relation count protects against engine mis-ordering
    // (17-family: skipping the 42K t x temp1 no-op cost +19-35 ms in the
    // 4-relation final).
    if (noop_selected && best == best_noop && best->card >= 1e5) {
#ifndef NDEBUG
      std::cout << "[SDS] PlanNext: unexpandable no-op pair -> final"
                << std::endl;
#endif
      return false;
    }
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
    for (size_t i = 0; i < n; i++) {
      if ((int)i == max_pos || comp[i] < 0)
        continue;
      const JoinRel &r = optimizer.Relation((int)i);
      cand_tables[comp[i]].insert(r.table_index);
      cand_width[comp[i]]++;
      if (mark_locked_.count(r.table_index))
        cand_ok[comp[i]] = false;
    }

    for (const auto &p : rejected_pairs) {
      cand_tables.push_back(p.first);
      cand_width.push_back((int)p.first.size());
      cand_ok.push_back(true);
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
    std::cerr << "[SDS] SplitIR: BuildSubIRForCluster failed" << std::endl;
    planned_splittable_ = false;
    return nullptr;
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

} // namespace middleware
