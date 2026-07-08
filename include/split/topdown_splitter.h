/*
 * SDS — stats-driven IR splitter (Phase 2 rewrite of the old TopDown
 * strategy; keeps the TOP_DOWN plumbing name).
 *
 * Split decisions are made entirely in the middleware: an IR-native
 * join-order DP (IRJoinOptimizer, a port of DuckDB's plan_enumerator /
 * tdom cardinality model) picks the cheapest first join from the remaining
 * join graph each iteration; that pair is materialized as a temp table and
 * fed back with its EXACT cardinality. No ReOptimizeIR round trips — the
 * engine only sees plain sub-SQL (which it fully re-optimizes anyway).
 *
 * Mechanics (sub-IR construction, remaining-IR rebuild) are inherited from
 * FKBasedSplitter; all FK-specific behavior in that base is gated on
 * RELATIONSHIP_CENTER/ENTITY_CENTER and stays inert under TOP_DOWN.
 * Design: new_split_strategy_analysis.md §9.
 */

#pragma once

#include "split/distinct_cache.h"
#include "split/fk_based_splitter.h"
#include "split/ir_join_optimizer.h"

#include <unordered_map>

namespace middleware {

class TopDownSplitter : public FKBasedSplitter {
public:
  // enable_reorder is accepted for constructor compatibility; SDS never
  // re-optimizes through the engine.
  explicit TopDownSplitter(EngineAdapter *adapter, bool enable_reorder = true);

  void Preprocess(std::unique_ptr<ir_sql_converter::AQPStmt> &ir) override;

  std::unique_ptr<SubqueryExtraction>
  SplitIR(ir_sql_converter::AQPStmt *remaining_ir) override;

  bool IsComplete(const ir_sql_converter::AQPStmt *remaining_ir) override;

  std::string GetStrategyName() const override { return "TopDown"; }

  std::unique_ptr<ir_sql_converter::AQPStmt> UpdateRemainingIR(
      std::unique_ptr<ir_sql_converter::AQPStmt> remaining_ir,
      const std::set<unsigned int> &executed_table_indices,
      unsigned int temp_table_index, const std::string &temp_table_name,
      uint64_t temp_table_cardinality,
      const std::vector<std::pair<unsigned int, unsigned int>> &column_mappings,
      const std::vector<std::string> &column_names) override;

  // Same standalone-abort threshold as node-based
  // (include/split/node_based_splitter.h): don't materialize a pair whose
  // estimated output exceeds this.
  static constexpr double kSplitCardThreshold = 1000000.0;

  // DuckDB join-order parity: flat mark/semi selectivity
  // (CardinalityEstimator::DEFAULT_SEMI_ANTI_SELECTIVITY).
  static constexpr double kSemiSelectivity = 5.0;

private:
  // Run the IR DP over the remaining join graph and cache the chosen next
  // group: the cheapest maximal subtree of the optimal join tree that does
  // not contain the largest relation (the probe spine stays in the final
  // query) and whose estimated output is under the threshold. Returns true
  // when a beneficial split exists.
  bool PlanNext(const ir_sql_converter::AQPStmt *remaining_ir);

  void CollectRelations(const ir_sql_converter::AQPStmt *node,
                        std::vector<JoinRel> &rels,
                        std::vector<unsigned int> &rel_tables) const;
  void CollectEdges(const ir_sql_converter::AQPStmt *node,
                    const std::map<unsigned int, int> &table_to_pos,
                    std::vector<JoinEdge> &edges) const;
  void CaptureLeafCardinalities(const ir_sql_converter::AQPStmt *node);
  // The incoming IR is FilterOptimize-only (join order never ran), so nodes
  // usually carry NO estimates: fill the gaps per base table with base-table
  // rows from one batched EXPLAIN of single-table sub-SQL (the engine does
  // not fold filter selectivity into single-table estimates), then apply the
  // middleware filter-selectivity model.
  void FetchMissingLeafCardinalities(ir_sql_converter::AQPStmt *ir);

  // Port of DuckDB's RelationStatisticsHelper::ExtractGetStats filter model:
  // equality-with-constant -> ceil(base / distinct), min over filters;
  // IN (k values) -> k such equalities; any other filter present without an
  // equality -> DEFAULT_SELECTIVITY (0.2) once.
  double EstimateFilteredCardinality(
      double base_cardinality, const std::string &table_name,
      const std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> &conjuncts);

  // Effective per-table cardinality (filters applied): captured from the
  // initial optimized IR; temps get exact counts in UpdateRemainingIR.
  std::map<unsigned int, double> table_card_;
  // Unfiltered base rows per table NAME (engine single-table estimates carry
  // no filter selectivity) -> constant across queries/repeats, never cleared.
  std::map<std::string, double> base_count_cache_;
  std::set<unsigned int> temp_indices_;

  // Mark joins are large-IN-list filters lowered to MarkJoin+Chunk by the
  // engine. Tables in the canonical Filter->Mark->[Scan,Chunk] shape are
  // splittable (BuildSubIRForCluster clones the whole unit as the scan) and
  // get their IN selectivity modeled: mark_in_ maps table index ->
  // (probe column, IN-list size). Any other Mark shape is not safely
  // clonable: those tables land in mark_locked_, stay out of split groups,
  // and resolve in the final query (the remaining-IR rebuild preserves the
  // subtree).
  std::map<unsigned int, std::pair<std::string, double>> mark_in_;
  std::set<unsigned int> mark_locked_;

  // PlanNext cache (valid for one remaining-IR instance).
  const ir_sql_converter::AQPStmt *planned_for_ = nullptr;
  bool planned_splittable_ = false;
  std::set<unsigned int> planned_group_;
  double planned_card_ = 0.0;

  // Column-name -> max base-table distinct seen at any join-edge endpoint
  // this query. Temps inherit these as distinct upper bounds (a temp column
  // can never have more distincts than its source base column) — without it
  // temps report distinct = cardinality, inflating tdom denominators into
  // est=1 garbage groups. Accumulated across iterations because the source
  // base tables vanish into temps.
  std::unordered_map<std::string, double> col_distinct_hints_;

  DistinctCache distinct_cache_{DistinctCache::DefaultPath()};
};

} // namespace middleware
