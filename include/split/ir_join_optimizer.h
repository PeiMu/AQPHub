/*
 * IR-native join-order optimizer: a port of DuckDB's join-order DP core
 * (plan_enumerator + cardinality_estimator tdom mechanism) operating on
 * plain relation/edge inputs — no engine, no IR dependency.
 * Port spec: new_split_strategy_analysis.md §9.3.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace middleware {

// One enumeration leaf: a base-table scan (filters already reflected in the
// cardinality estimate) or a materialized temp table (exact cardinality).
struct JoinRel {
  unsigned int table_index = 0; // IR table index
  std::string table_name;       // base table name, or temp table name
  bool is_temp = false;
  double cardinality = 1.0; // post-filter estimate (temps: exact)
  // Unfiltered base-table row count (scan size); 0 = unknown, fall back to
  // cardinality. Drives the probe-scan cost term and the big-big pair test.
  double base_cardinality = 0.0;
};

// Join predicate between two relations. Equality edges feed the tdom
// equivalence-class model; non-equality edges (DuckDB: tdom^(2/3)) still
// establish connectivity and reduce cardinality, just less aggressively.
struct JoinEdge {
  int left_rel = -1; // position into the relations vector
  int right_rel = -1;
  unsigned int left_col = 0; // IR column indices (for the splitter's use)
  unsigned int right_col = 0;
  std::string left_col_name;
  std::string right_col_name;
  bool is_equality = true; // false for <, >, !=, etc.
};

// Result join tree (bushy allowed). Leaves reference relation positions.
// Convention: right child is the smaller (build) side; the engine's physical
// planner re-decides sides per sub-SQL anyway.
struct JoinTree {
  int rel = -1; // >= 0: leaf (relation position)
  std::unique_ptr<JoinTree> left;
  std::unique_ptr<JoinTree> right;
  double card = 0.0;   // estimated output cardinality of this subtree
  double cost = 0.0;   // cumulative C_out cost
  uint64_t set = 0;    // bitmask of relation positions in this subtree
  bool cross_product = false; // joins disconnected components

  bool IsLeaf() const { return rel >= 0; }
};

class IRJoinOptimizer {
public:
  // distinct_of(table_name, column_name) -> distinct count; return <= 0 when
  // unknown (falls back to the relation's cardinality, mirroring DuckDB's
  // no-HLL path where distinct = cardinality).
  using DistinctFn =
      std::function<double(const std::string &, const std::string &)>;

  // locality_of(table_name, column_name) -> |corr(rowid, column)| in [0, 1];
  // return <= 0 when unknown. Feeds the probe-scan cost term: joining a big
  // base table on a column correlated with its physical row order lets the
  // engine's join-filter block skipping prune the probe scan; an uncorrelated
  // column forces a full scan regardless of the build side's size.
  using LocalityFn =
      std::function<double(const std::string &, const std::string &)>;

  IRJoinOptimizer(std::vector<JoinRel> relations, std::vector<JoinEdge> edges,
                  DistinctFn distinct_of, LocalityFn locality_of = nullptr);

  // Exact subset DP (DPccp-equivalent search space) for <= kExactThreshold
  // relations within kPairBudget emitted pairs; greedy min-cost-pair merge
  // otherwise. Disconnected graphs are combined with cross products
  // (smallest cardinalities first) — never fails on a valid input.
  std::unique_ptr<JoinTree> Solve();

  // prod(cards) / tdom spanning-forest denominator for an arbitrary set of
  // relation positions. Exposed for the boundary/depth policy.
  double EstimateCardinality(uint64_t set) const;

  size_t NumRelations() const { return relations_.size(); }
  const JoinRel &Relation(int pos) const { return relations_[pos]; }

  static constexpr int kExactThreshold = 12;     // duckdb plan_enumerator.hpp:36
  static constexpr uint64_t kPairBudget = 10000; // duckdb TryEmitPair soft cap
  // Base tables at least this large get a probe-scan cost term (JOB: title,
  // cast_info, movie_info, movie_keyword, movie_companies, person_info, name,
  // char_name).
  static constexpr double kProbeLocalityMinRows = 2e6;
  // Scanned rows are far cheaper than C_out rows (a fused filter scan is
  // ~1-2 ns/row; a materialized/hashed output row is ~20-50 ns): without
  // this weight the scan term drowns the output-cardinality signal and the
  // DP starts preferring big-x-big joins with good locality over tiny
  // selective joins (6e: mk JOIN t, 2M rows materialized, chosen over
  // k JOIN mk, 34 rows).
  static constexpr double kProbeScanWeight = 0.1;

private:
  struct EqClass {
    std::vector<std::pair<int, unsigned int>> members; // (rel pos, col idx)
    double tdom = 1.0;
  };

  void BuildEquivalenceClasses(DistinctFn distinct_of);
  void BuildNonEquiTerms(DistinctFn distinct_of);
  void BuildEdgeLocalities(LocalityFn locality_of);
  // Scan-cost term for one side of a join: full leaf scan discounted by
  // block-skip locality when the side is a single big base relation probed
  // through a locality-bearing column. 0 for temps/small/composite sides.
  double ProbeCostTerm(int leaf_pos, uint64_t other_set,
                       double other_card) const;
  double PairScanCost(uint64_t a_set, double a_card, uint64_t b_set,
                      double b_card) const;
  bool Connected(uint64_t a, uint64_t b) const; // any edge across a|b
  std::vector<uint64_t> ConnectedComponents(uint64_t set) const;

  // Exact DP over one connected component. Returns nullptr if the pair
  // budget is exhausted (caller falls back to greedy).
  std::unique_ptr<JoinTree> SolveExact(uint64_t component);
  std::unique_ptr<JoinTree> SolveGreedy(uint64_t set);
  std::unique_ptr<JoinTree> CombineWithCrossProducts(
      std::vector<std::unique_ptr<JoinTree>> trees) const;

  std::unique_ptr<JoinTree> MakeLeaf(int pos) const;
  std::unique_ptr<JoinTree> MakePair(std::unique_ptr<JoinTree> l,
                                     std::unique_ptr<JoinTree> r,
                                     bool cross) const;

  // Non-equality join predicate: contributes tdom^(2/3) to the cardinality
  // denominator (DuckDB cardinality_estimator.cpp:256). Stored separately
  // from equality equivalence classes because non-equi conditions do not
  // establish column equivalence.
  struct NonEquiTerm {
    int left_rel;
    int right_rel;
    double reduced_tdom; // max(distinct_left, distinct_right)^(2/3)
  };

  std::vector<JoinRel> relations_;
  std::vector<JoinEdge> edges_;
  std::vector<EqClass> classes_;
  std::vector<NonEquiTerm> non_equi_terms_;
  std::vector<uint64_t> rel_neighbors_; // per relation: bitmask of neighbors
  // Per edge: |corr(rowid, col)| of the left/right endpoint column, computed
  // only for big base relations (0 otherwise).
  std::vector<double> edge_left_loc_;
  std::vector<double> edge_right_loc_;
};

} // namespace middleware
