#include "split/ir_join_optimizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

#ifndef NDEBUG
#include <iostream>
#endif

namespace middleware {

namespace {
inline int LowestBit(uint64_t mask) { return __builtin_ctzll(mask); }
inline int PopCount(uint64_t mask) { return __builtin_popcountll(mask); }
inline double BaseCard(const JoinRel &rel) {
  return rel.base_cardinality > 0.0 ? rel.base_cardinality : rel.cardinality;
}
} // namespace

IRJoinOptimizer::IRJoinOptimizer(std::vector<JoinRel> relations,
                                 std::vector<JoinEdge> edges,
                                 DistinctFn distinct_of, LocalityFn locality_of)
    : relations_(std::move(relations)), edges_(std::move(edges)) {
  for (auto &rel : relations_)
    rel.cardinality = std::max(rel.cardinality, 1.0);
  rel_neighbors_.assign(relations_.size(), 0);
  for (const auto &e : edges_) {
    rel_neighbors_[e.left_rel] |= (uint64_t)1 << e.right_rel;
    rel_neighbors_[e.right_rel] |= (uint64_t)1 << e.left_rel;
  }
  auto distinct_copy = distinct_of;
  BuildEquivalenceClasses(distinct_of);
  BuildNonEquiTerms(std::move(distinct_copy));
  BuildEdgeLocalities(std::move(locality_of));
}

void IRJoinOptimizer::BuildEdgeLocalities(LocalityFn locality_of) {
  edge_left_loc_.assign(edges_.size(), 0.0);
  edge_right_loc_.assign(edges_.size(), 0.0);
  if (!locality_of)
    return;
  auto loc = [&](int rel_pos, const std::string &col) {
    const JoinRel &rel = relations_[rel_pos];
    if (rel.is_temp || BaseCard(rel) < kProbeLocalityMinRows)
      return 0.0;
    double c = locality_of(rel.table_name, col);
    return std::isfinite(c) ? std::min(std::max(c, 0.0), 1.0) : 0.0;
  };
  for (size_t i = 0; i < edges_.size(); i++) {
    edge_left_loc_[i] = loc(edges_[i].left_rel, edges_[i].left_col_name);
    edge_right_loc_[i] = loc(edges_[i].right_rel, edges_[i].right_col_name);
  }
}

// Scan cost of probing a single big base relation against `other`: the
// engine's join filter prunes the probe scan only when the probe column is
// correlated with physical row order (block skipping), and only to the
// degree the build side is selective. With locality 0 or a build side as
// large as the leaf this degenerates to the full scan cost leaf.card — a
// constant across all trees containing the leaf, so rankings shift only
// where locality genuinely differs.
double IRJoinOptimizer::ProbeCostTerm(int leaf_pos, uint64_t other_set,
                                      double other_card) const {
  const JoinRel &rel = relations_[leaf_pos];
  double scan_rows = BaseCard(rel);
  if (rel.is_temp || scan_rows < kProbeLocalityMinRows)
    return 0.0;
  double best_loc = 0.0;
  for (size_t i = 0; i < edges_.size(); i++) {
    const auto &e = edges_[i];
    if (e.left_rel == leaf_pos && (other_set >> e.right_rel & 1))
      best_loc = std::max(best_loc, edge_left_loc_[i]);
    else if (e.right_rel == leaf_pos && (other_set >> e.left_rel & 1))
      best_loc = std::max(best_loc, edge_right_loc_[i]);
  }
  double frac = std::min(1.0, other_card / scan_rows);
  return scan_rows * (1.0 - best_loc * (1.0 - frac));
}

double IRJoinOptimizer::PairScanCost(uint64_t a_set, double a_card,
                                     uint64_t b_set, double b_card) const {
  double term = 0.0;
  if (PopCount(a_set) == 1)
    term += ProbeCostTerm(LowestBit(a_set), b_set, b_card);
  if (PopCount(b_set) == 1)
    term += ProbeCostTerm(LowestBit(b_set), a_set, a_card);
  return kProbeScanWeight * term;
}

// Union-find over (rel, col) endpoints of equality edges; one tdom per class
// = max over member columns' distinct counts (DuckDB: max(hll, no_hll);
// unknown distinct falls back to the relation's cardinality = the no-HLL
// path for temps, relation_statistics_helper.cpp:180-192).
void IRJoinOptimizer::BuildEquivalenceClasses(DistinctFn distinct_of) {
  std::unordered_map<uint64_t, int> endpoint_to_slot;
  auto key = [](int rel, unsigned int col) {
    return ((uint64_t)(unsigned)rel << 32) | col;
  };
  std::vector<int> parent;
  std::function<int(int)> find = [&](int x) {
    while (parent[x] != x)
      x = parent[x] = parent[parent[x]];
    return x;
  };
  auto slot_of = [&](int rel, unsigned int col) {
    auto it = endpoint_to_slot.find(key(rel, col));
    if (it != endpoint_to_slot.end())
      return it->second;
    int slot = (int)parent.size();
    parent.push_back(slot);
    endpoint_to_slot.emplace(key(rel, col), slot);
    return slot;
  };

  std::vector<std::pair<int, unsigned int>> slot_endpoint;
  auto record = [&](int rel, unsigned int col) {
    int slot = slot_of(rel, col);
    if ((size_t)slot == slot_endpoint.size())
      slot_endpoint.emplace_back(rel, col);
    return slot;
  };

  std::vector<std::pair<int, int>> edge_slots;
  edge_slots.reserve(edges_.size());
  for (const auto &e : edges_) {
    if (!e.is_equality)
      continue;
    int ls = record(e.left_rel, e.left_col);
    int rs = record(e.right_rel, e.right_col);
    edge_slots.emplace_back(ls, rs);
  }
  for (auto &es : edge_slots)
    parent[find(es.first)] = find(es.second);

  std::unordered_map<int, int> root_to_class;
  std::vector<std::vector<std::string>> class_col_names;
  for (size_t slot = 0; slot < slot_endpoint.size(); slot++) {
    int root = find((int)slot);
    auto it = root_to_class.find(root);
    int cls;
    if (it == root_to_class.end()) {
      cls = (int)classes_.size();
      root_to_class.emplace(root, cls);
      classes_.emplace_back();
      class_col_names.emplace_back();
    } else {
      cls = it->second;
    }
    classes_[cls].members.push_back(slot_endpoint[slot]);
  }

  // Column names per endpoint for the distinct lookup: take them from the
  // edges (each endpoint appears in at least one edge).
  std::unordered_map<uint64_t, std::string> endpoint_name;
  for (const auto &e : edges_) {
    endpoint_name.emplace(key(e.left_rel, e.left_col), e.left_col_name);
    endpoint_name.emplace(key(e.right_rel, e.right_col), e.right_col_name);
  }

  for (auto &cls : classes_) {
    double tdom = 1.0;
    for (const auto &m : cls.members) {
      const JoinRel &rel = relations_[m.first];
      double d = 0.0;
      if (distinct_of) {
        auto it = endpoint_name.find(key(m.first, m.second));
        if (it != endpoint_name.end())
          d = distinct_of(rel.table_name, it->second);
      }
      if (d <= 0.0)
        d = rel.cardinality; // no-HLL fallback: distinct = cardinality
      else if (rel.is_temp)
        d = std::min(d, rel.cardinality); // hint is an upper bound
#ifndef NDEBUG
      {
        auto it = endpoint_name.find(key(m.first, m.second));
        std::cout << "[SDS] tdom member " << rel.table_name << "."
                  << (it != endpoint_name.end() ? it->second : "<?>")
                  << " d=" << d << " (temp=" << rel.is_temp << ")"
                  << std::endl;
      }
#endif
      tdom = std::max(tdom, d);
    }
    cls.tdom = tdom;
  }
}

void IRJoinOptimizer::BuildNonEquiTerms(DistinctFn distinct_of) {
  for (const auto &e : edges_) {
    if (e.is_equality)
      continue;
    const JoinRel &lr = relations_[e.left_rel];
    const JoinRel &rr = relations_[e.right_rel];
    double dl = 0.0, dr = 0.0;
    if (distinct_of) {
      dl = distinct_of(lr.table_name, e.left_col_name);
      dr = distinct_of(rr.table_name, e.right_col_name);
    }
    if (dl <= 0.0)
      dl = lr.cardinality;
    else if (lr.is_temp)
      dl = std::min(dl, lr.cardinality);
    if (dr <= 0.0)
      dr = rr.cardinality;
    else if (rr.is_temp)
      dr = std::min(dr, rr.cardinality);
    double tdom = std::max(dl, dr);
    non_equi_terms_.push_back(
        {e.left_rel, e.right_rel, std::pow(tdom, 2.0 / 3.0)});
#ifndef NDEBUG
    std::cout << "[SDS] non-equi term " << lr.table_name << " x "
              << rr.table_name << " tdom^(2/3)=" << std::pow(tdom, 2.0 / 3.0)
              << std::endl;
#endif
  }
  std::sort(non_equi_terms_.begin(), non_equi_terms_.end(),
            [](const NonEquiTerm &a, const NonEquiTerm &b) {
              return a.reduced_tdom > b.reduced_tdom;
            });
}

bool IRJoinOptimizer::Connected(uint64_t a, uint64_t b) const {
  for (uint64_t m = a; m; m &= m - 1) {
    if (rel_neighbors_[LowestBit(m)] & b)
      return true;
  }
  return false;
}

std::vector<uint64_t> IRJoinOptimizer::ConnectedComponents(uint64_t set) const {
  std::vector<uint64_t> components;
  uint64_t remaining = set;
  while (remaining) {
    uint64_t comp = (uint64_t)1 << LowestBit(remaining);
    while (true) {
      uint64_t grown = comp;
      for (uint64_t m = comp; m; m &= m - 1)
        grown |= rel_neighbors_[LowestBit(m)] & set;
      if (grown == comp)
        break;
      comp = grown;
    }
    components.push_back(comp);
    remaining &= ~comp;
  }
  return components;
}

// prod(cards) / spanning-forest tdom denominator. Classes applied in
// descending tdom order (duckdb CardinalityEstimator::GetDenominator); a
// class with k member relations inside the set contributes tdom^(k'-1)
// where k' = number of its relations newly connected (textbook tdom form).
double IRJoinOptimizer::EstimateCardinality(uint64_t set) const {
  double numerator = 1.0;
  for (uint64_t m = set; m; m &= m - 1)
    numerator *= relations_[LowestBit(m)].cardinality;

  std::vector<int> order(classes_.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return classes_[a].tdom > classes_[b].tdom;
  });

  // Union-find over relation positions inside `set`.
  int uf[64];
  for (int i = 0; i < 64; i++)
    uf[i] = i;
  std::function<int(int)> find = [&](int x) {
    while (uf[x] != x)
      x = uf[x] = uf[uf[x]];
    return x;
  };

  double denominator = 1.0;
  for (int ci : order) {
    const EqClass &cls = classes_[ci];
    int first = -1;
    for (const auto &m : cls.members) {
      if (!(set & ((uint64_t)1 << m.first)))
        continue;
      if (first < 0) {
        first = m.first;
        continue;
      }
      int ra = find(first), rb = find(m.first);
      if (ra != rb) {
        uf[ra] = rb;
        denominator *= cls.tdom;
      }
    }
  }

  // Non-equality edges: tdom^(2/3) per bridging edge (sorted descending).
  for (const auto &ne : non_equi_terms_) {
    if (!(set & ((uint64_t)1 << ne.left_rel)) ||
        !(set & ((uint64_t)1 << ne.right_rel)))
      continue;
    int ra = find(ne.left_rel), rb = find(ne.right_rel);
    if (ra != rb) {
      uf[ra] = rb;
      denominator *= ne.reduced_tdom;
    }
  }

  double card = numerator / std::max(denominator, 1.0);
  return std::max(card, 1.0);
}

std::unique_ptr<JoinTree> IRJoinOptimizer::MakeLeaf(int pos) const {
  auto leaf = std::make_unique<JoinTree>();
  leaf->rel = pos;
  leaf->set = (uint64_t)1 << pos;
  leaf->card = relations_[pos].cardinality;
  leaf->cost = 0.0;
  return leaf;
}

std::unique_ptr<JoinTree>
IRJoinOptimizer::MakePair(std::unique_ptr<JoinTree> l,
                          std::unique_ptr<JoinTree> r, bool cross) const {
  auto node = std::make_unique<JoinTree>();
  node->set = l->set | r->set;
  node->card = cross ? std::max(l->card * r->card, 1.0)
                     : EstimateCardinality(node->set);
  node->cost = node->card + l->cost + r->cost +
               PairScanCost(l->set, l->card, r->set, r->card);
  node->cross_product = cross;
  // Build side (right) = smaller estimated cardinality.
  if (l->card < r->card)
    std::swap(l, r);
  node->left = std::move(l);
  node->right = std::move(r);
  return node;
}

// Exact subset DP over one connected component (search space identical to
// DPccp: only connected pairs of connected subsets are emitted). Pair budget
// mirrors duckdb TryEmitPair's soft timeout.
std::unique_ptr<JoinTree> IRJoinOptimizer::SolveExact(uint64_t component) {
  struct Entry {
    double cost = std::numeric_limits<double>::infinity();
    double card = 0.0;
    uint64_t left = 0; // best split: left submask (0 for leaf)
    bool valid = false;
  };

  // Compact index space over the component's members.
  std::vector<int> members;
  for (uint64_t m = component; m; m &= m - 1)
    members.push_back(LowestBit(m));
  int n = (int)members.size();
  if (n == 1)
    return MakeLeaf(members[0]);

  auto expand = [&](uint64_t local) {
    uint64_t global = 0;
    for (uint64_t m = local; m; m &= m - 1)
      global |= (uint64_t)1 << members[LowestBit(m)];
    return global;
  };

  std::vector<Entry> dp((size_t)1 << n);
  for (int i = 0; i < n; i++) {
    Entry &e = dp[(size_t)1 << i];
    e.cost = 0.0;
    e.card = relations_[members[i]].cardinality;
    e.valid = true;
  }

  uint64_t pairs_emitted = 0;
  for (uint64_t mask = 1; mask < ((uint64_t)1 << n); mask++) {
    if (PopCount(mask) < 2)
      continue;
    Entry &e = dp[mask];
    // Enumerate proper submasks; canonical form: submask contains the
    // lowest bit of mask (avoids double-visiting each split).
    uint64_t low = (uint64_t)1 << LowestBit(mask);
    for (uint64_t sub = (mask - 1) & mask; sub; sub = (sub - 1) & mask) {
      if (!(sub & low))
        continue;
      uint64_t rest = mask ^ sub;
      if (!dp[sub].valid || !dp[rest].valid)
        continue;
      if (!Connected(expand(sub), expand(rest)))
        continue;
      if (++pairs_emitted > kPairBudget)
        return nullptr; // budget exhausted → caller falls back to greedy
      double card = e.valid ? e.card : EstimateCardinality(expand(mask));
      double cost = card + dp[sub].cost + dp[rest].cost +
                    PairScanCost(expand(sub), dp[sub].card, expand(rest),
                                 dp[rest].card);
      if (cost < e.cost) {
        e.cost = cost;
        e.card = card;
        e.left = sub;
        e.valid = true;
      }
    }
  }

  uint64_t full = ((uint64_t)1 << n) - 1;
  if (!dp[full].valid)
    return nullptr; // shouldn't happen on a connected component

  std::function<std::unique_ptr<JoinTree>(uint64_t)> build =
      [&](uint64_t mask) -> std::unique_ptr<JoinTree> {
    if (PopCount(mask) == 1)
      return MakeLeaf(members[LowestBit(mask)]);
    const Entry &e = dp[mask];
    return MakePair(build(e.left), build(mask ^ e.left), false);
  };
  return build(full);
}

// Greedy min-cost connected pair merge (duckdb SolveJoinOrderApproximately):
// repeatedly join the connected pair with the smallest estimated output;
// if nothing is connected, cross-product the two smallest trees.
std::unique_ptr<JoinTree> IRJoinOptimizer::SolveGreedy(uint64_t set) {
  std::vector<std::unique_ptr<JoinTree>> trees;
  for (uint64_t m = set; m; m &= m - 1)
    trees.push_back(MakeLeaf(LowestBit(m)));

  while (trees.size() > 1) {
    double best_card = std::numeric_limits<double>::infinity();
    int best_i = -1, best_j = -1;
    for (size_t i = 0; i < trees.size(); i++) {
      for (size_t j = i + 1; j < trees.size(); j++) {
        if (!Connected(trees[i]->set, trees[j]->set))
          continue;
        double card = EstimateCardinality(trees[i]->set | trees[j]->set);
        if (card < best_card) {
          best_card = card;
          best_i = (int)i;
          best_j = (int)j;
        }
      }
    }
    bool cross = false;
    if (best_i < 0) {
      // Disconnected: cross-product the two smallest-cardinality trees.
      cross = true;
      double s1 = std::numeric_limits<double>::infinity(), s2 = s1;
      for (size_t i = 0; i < trees.size(); i++) {
        if (trees[i]->card < s1) {
          s2 = s1;
          best_j = best_i;
          s1 = trees[i]->card;
          best_i = (int)i;
        } else if (trees[i]->card < s2) {
          s2 = trees[i]->card;
          best_j = (int)i;
        }
      }
    }
    auto joined = MakePair(std::move(trees[best_i]), std::move(trees[best_j]),
                           cross);
    if (best_i > best_j)
      std::swap(best_i, best_j);
    trees.erase(trees.begin() + best_j);
    trees[best_i] = std::move(joined);
  }
  return std::move(trees[0]);
}

std::unique_ptr<JoinTree> IRJoinOptimizer::CombineWithCrossProducts(
    std::vector<std::unique_ptr<JoinTree>> trees) const {
  // Smallest cardinalities first (duckdb greedy's disconnected rule).
  std::sort(trees.begin(), trees.end(),
            [](const std::unique_ptr<JoinTree> &a,
               const std::unique_ptr<JoinTree> &b) { return a->card < b->card; });
  auto result = std::move(trees[0]);
  for (size_t i = 1; i < trees.size(); i++)
    result = MakePair(std::move(result), std::move(trees[i]), true);
  return result;
}

std::unique_ptr<JoinTree> IRJoinOptimizer::Solve() {
  if (relations_.empty() || relations_.size() > 63)
    return nullptr; // bitmask limit; caller keeps the existing order
  uint64_t all = ((uint64_t)1 << relations_.size()) - 1;

  auto components = ConnectedComponents(all);
  std::vector<std::unique_ptr<JoinTree>> trees;
  for (uint64_t comp : components) {
    std::unique_ptr<JoinTree> tree;
    if (PopCount(comp) <= kExactThreshold)
      tree = SolveExact(comp);
    if (!tree)
      tree = SolveGreedy(comp);
    trees.push_back(std::move(tree));
  }
  if (trees.size() == 1)
    return std::move(trees[0]);
  return CombineWithCrossProducts(std::move(trees));
}

} // namespace middleware
