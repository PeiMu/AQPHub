#include "qjit/query_jit_steps.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include "jit/aqp_jit_abi.h"
#include "simplest_ir.h"

using namespace ir_sql_converter;

namespace qjit {

QjitQueryPlan::QjitQueryPlan() = default;
QjitQueryPlan::QjitQueryPlan(QjitQueryPlan &&) noexcept = default;
QjitQueryPlan &QjitQueryPlan::operator=(QjitQueryPlan &&) noexcept = default;
QjitQueryPlan::~QjitQueryPlan() = default;

namespace {

const char *NodeTypeName(SimplestNodeType t) {
  switch (t) {
  case StmtNode:         return "Stmt";
  case ProjectionNode:   return "Projection";
  case AggregateNode:    return "Aggregate";
  case OrderNode:        return "OrderBy";
  case LimitNode:        return "Limit";
  case JoinNode:         return "Join";
  case CrossProductNode: return "CrossProduct";
  case FilterNode:       return "Filter";
  case ScanNode:         return "Scan";
  case ChunkNode:        return "Chunk";
  case HashNode:         return "Hash";
  case SortNode:         return "Sort";
  case RawSQLNode:       return "RawSQL";
  default:               return "Unknown";
  }
}

const char *JoinTypeName(SimplestJoinType t) {
  switch (t) {
  case Inner:       return "Inner";
  case Left:        return "Left";
  case Full:        return "Full";
  case Right:       return "Right";
  case Mark:        return "Mark";
  case Semi:        return "Semi";
  case Anti:        return "Anti";
  case UniqueOuter: return "UniqueOuter";
  case UniqueInner: return "UniqueInner";
  default:          return "Invalid";
  }
}

const char *AggFnName(SimplestAggFnType t) {
  switch (t) {
  case Min:       return "min";
  case Max:       return "max";
  case Sum:       return "sum";
  case Average:   return "avg";
  case Count:     return "count";
  case CountStar: return "count_star";
  default:        return "invalid";
  }
}

// ---------------------------------------------------------------------------
// Mark-join IN-list rewrite (§5.2). DuckDB compiles large IN-lists into a
// MARK hash join against a CHUNK_GET constant table, consumed by a Bool
// SingleAttrExpr qual above it (table_index = the join's mark_index). The
// pattern is rewritten to a streaming in-set filter on the probe spine
// (existing InExprNode codegen, strict per-leaf NULL guard): NULL probe key
// ⇒ mark NULL ⇒ filter drops the row, exactly like NULL IN (list) ⇒ NULL ⇒
// drop, so no hash join is needed at all.
// ---------------------------------------------------------------------------

struct MarkInPattern {
  const SimplestAttr *probe_attr = nullptr;
  const SimplestChunk *chunk = nullptr;
  const ir_sql_converter::AQPStmt *probe_child = nullptr;
  bool is_string = false;
};

// Validate `join` (already known to be a Mark join with a pending consumer)
// against the rewritable shape. Returns "" on success, else the reject token.
std::string MatchMarkInJoin(const SimplestJoin &join, MarkInPattern &out) {
  if (join.children.size() != 2 || !join.children[0] || !join.children[1])
    return "markin:children!=2";
  int chunk_idx = -1;
  for (int i = 0; i < 2; i++) {
    if (join.children[i]->GetNodeType() == ChunkNode) {
      if (chunk_idx >= 0)
        return "markin:two-chunks";
      chunk_idx = i;
    }
  }
  if (chunk_idx < 0)
    return "markin:build-not-chunk"; // build side is a subtree, not constants
  const auto &chunk =
      static_cast<const SimplestChunk &>(*join.children[chunk_idx]);
  if (chunk.GetContents().empty())
    return "markin:chunk-empty"; // named temp placeholder (or empty list)
  if (chunk.target_list.size() != 1 || !chunk.target_list[0])
    return "markin:chunk-shape"; // contents flattening ambiguous for >1 col
  if (join.join_conditions.size() != 1)
    return "markin:multi-condition";
  const auto &cond = join.join_conditions[0];
  if (!cond || cond->GetSimplestExprType() != SimplestExprType::Equal ||
      !cond->left_attr || !cond->right_attr)
    return "markin:condition";
  const SimplestAttr *l = cond->left_attr.get();
  const SimplestAttr *r = cond->right_attr.get();
  unsigned chunk_tidx = chunk.GetTableIndex();
  const SimplestAttr *probe;
  if (l->GetTableIndex() == chunk_tidx && r->GetTableIndex() != chunk_tidx)
    probe = r;
  else if (r->GetTableIndex() == chunk_tidx &&
           l->GetTableIndex() != chunk_tidx)
    probe = l;
  else
    return "markin:condition-sides";
  // Chunk contents are stringified values; only types whose string form
  // round-trips losslessly are admitted (Date prints as "YYYY-MM-DD").
  SimplestVarType pt = probe->GetType();
  SimplestVarType vt = chunk.target_list[0]->GetType();
  if (pt == StringVar && vt == StringVar)
    out.is_string = true;
  else if (pt == IntVar && vt == IntVar && probe->GetBitWidth() != 64)
    out.is_string = false;
  else
    return "markin:value-type";
  out.probe_attr = probe;
  out.chunk = &chunk;
  out.probe_child = join.children[1 - chunk_idx].get();
  return "";
}

// Synthesize `probe_attr IN (chunk contents)`. Returns nullptr + reason on a
// value-parse failure.
std::unique_ptr<SimplestInExpr> BuildMarkInExpr(const MarkInPattern &p,
                                                std::string &err) {
  std::vector<std::unique_ptr<SimplestConstVar>> vals;
  for (const std::string &s : p.chunk->GetContents()) {
    if (p.is_string) {
      vals.push_back(std::make_unique<SimplestConstVar>(s));
    } else {
      errno = 0;
      char *end = nullptr;
      long v = strtol(s.c_str(), &end, 10);
      if (errno != 0 || end == s.c_str() || *end != '\0' ||
          v < (long)INT32_MIN || v > (long)INT32_MAX) {
        err = "markin:const-parse";
        return nullptr;
      }
      vals.push_back(std::make_unique<SimplestConstVar>((int)v));
    }
  }
  return std::make_unique<SimplestInExpr>(
      std::make_unique<SimplestAttr>(*p.probe_attr), std::move(vals),
      /*negated=*/false);
}

struct Analyzer {
  QjitAnalysisResult result;
  // Mark-join IN-list rewrite: mark_index -> #pending SingleAttrExpr
  // consumers seen above (top-down), consumed when the Mark join is reached.
  std::unordered_map<unsigned, int> pending_marks;

  bool Reject(std::string reason) {
    if (result.reject_reason.empty())
      result.reject_reason = std::move(reason);
    return false;
  }

  // Expression whitelist: what the existing expression codegen
  // (ir_to_llvm.cpp EmitFilterExpr family) handles. Anything else rejects.
  bool CheckExpr(const AQPExpr *e, bool under_not = false) {
    if (!e)
      return Reject("expr:null");
    switch (e->GetNodeType()) {
    case VarConstComparisonNode:
    case IsNullExprNode:
    case InExprNode:
      return true;
    case VarComparisonNode:
      return true;
    case LogicalExprNode: {
      auto *log = static_cast<const SimplestLogicalExpr *>(e);
      bool not_op = log->GetLogicalOp() == SimplestLogicalOp::LogicalNot;
      if (log->left_expr && !CheckExpr(log->left_expr.get(), under_not))
        return false;
      if (log->right_expr &&
          !CheckExpr(log->right_expr.get(), under_not || not_op))
        return false;
      return true;
    }
    case ArithExprNode:
      return Reject("expr:arith");
    case CastExprNode:
      return Reject("expr:cast");
    case VarParamComparisonNode:
      return Reject("expr:param-comparison");
    case SingleAttrExprNode:
      // Top-level (rewritable) mark consumers are handled in CheckQuals;
      // reaching here means the mark column is nested in an expression
      // tree (NOT = anti-mark / NOT IN, or under AND/OR).
      return Reject(under_not ? "markin:negated" : "expr:single-attr");
    default:
      return Reject(std::string("expr:") + NodeTypeName(e->GetNodeType()));
    }
  }

  bool CheckQuals(const AQPStmt &node) {
    for (const auto &q : node.qual_vec) {
      if (q && q->GetNodeType() == SingleAttrExprNode) {
        auto *sa = static_cast<const SimplestSingleAttrExpr *>(q.get());
        if (!sa->attr || sa->attr->GetType() != BoolVar)
          return Reject("markin:non-bool");
        pending_marks[sa->attr->GetTableIndex()]++;
        continue;
      }
      if (!CheckExpr(q.get()))
        return false;
    }
    return true;
  }

  // top_of_spine: true while only ProjectionNodes have been passed since the
  // root. DuckDB plans `SELECT MIN(..)` as Projection->Aggregate, so the AGG
  // sink is accepted anywhere under root-side projections (the projection
  // over the single agg row is trivially streaming).
  bool Walk(const AQPStmt &node, bool top_of_spine) {
    switch (node.GetNodeType()) {
    case ScanNode:
    case ChunkNode:
      // Step sources. Scan quals are compiled into the morsel body.
      return CheckQuals(node);

    case FilterNode:
    case ProjectionNode: {
      if (!CheckQuals(node))
        return false;
      if (node.children.size() != 1 || !node.children[0])
        return Reject(std::string(NodeTypeName(node.GetNodeType())) +
                      ":children!=1");
      return Walk(*node.children[0],
                  top_of_spine && node.GetNodeType() == ProjectionNode);
    }

    case JoinNode: {
      auto &join = static_cast<const SimplestJoin &>(node);
      if (join.GetSimplestJoinType() == Mark) {
        auto it = pending_marks.find(join.GetMarkIndex());
        if (it == pending_marks.end())
          return Reject("join:Mark"); // mark column consumed elsewhere
        MarkInPattern pat;
        std::string r = MatchMarkInJoin(join, pat);
        if (!r.empty())
          return Reject(std::move(r));
        if (--it->second == 0)
          pending_marks.erase(it);
        result.num_mark_in++;
        if (!CheckQuals(node))
          return false;
        return Walk(*pat.probe_child, false); // chunk side is dropped
      }
      result.num_joins++;
      if (join.GetSimplestJoinType() != Inner)
        return Reject(std::string("join:") +
                      JoinTypeName(join.GetSimplestJoinType()));
      if (node.children.size() != 2 || !node.children[0] || !node.children[1])
        return Reject("join:children!=2");
      int build = join.GetBuildChild();
      if (build != 0 && build != 1)
        return Reject("join:build-side-unannotated");
      if (join.join_conditions.empty())
        return Reject("join:no-conditions");
      for (const auto &cond : join.join_conditions) {
        if (!cond || cond->GetSimplestExprType() != SimplestExprType::Equal)
          return Reject("join:non-equi-condition");
        if (!cond->left_attr || !cond->right_attr)
          return Reject("join:condition-missing-attr");
      }
      if (!CheckQuals(node))
        return false;
      return Walk(*node.children[0], false) && Walk(*node.children[1], false);
    }

    case AggregateNode: {
      auto &agg = static_cast<const SimplestAggregate &>(node);
      result.has_aggregate = true;
      if (!top_of_spine)
        return Reject("agg:not-top");
      if (!agg.groups.empty())
        return Reject("agg:grouped");
      if (agg.agg_fns.empty())
        return Reject("agg:no-functions");
      for (const auto &fn : agg.agg_fns) {
        switch (fn.second) {
        case Min:
        case Max:
        case Sum:
        case Count:
        case Average:
          if (!fn.first)
            return Reject("agg:missing-arg");
          break;
        case CountStar:
          break;
        default:
          return Reject(std::string("agg:") + AggFnName(fn.second));
        }
      }
      if (!CheckQuals(node))
        return false;
      if (node.children.size() != 1 || !node.children[0])
        return Reject("agg:children!=1");
      return Walk(*node.children[0], false);
    }

    case OrderNode:
    case LimitNode: {
      if (node.children.size() != 1 || !node.children[0])
        return Reject(std::string(NodeTypeName(node.GetNodeType())) +
                      ":children!=1");
      return Walk(*node.children[0], top_of_spine);
    }

    case CrossProductNode:
    case HashNode:
    case SortNode:
    case RawSQLNode:
    case StmtNode:
    default:
      return Reject(std::string("node:") + NodeTypeName(node.GetNodeType()));
    }
  }
};

} // namespace

static const AQPStmt *PeelOrderLimit(const AQPStmt *node) {
  while (node) {
    if (node->GetNodeType() == OrderNode || node->GetNodeType() == LimitNode) {
      node = node->children.empty() ? nullptr : node->children[0].get();
    } else {
      break;
    }
  }
  return node;
}

QjitAnalysisResult AnalyzeQueryJit(const AQPStmt &root,
                                   const std::string &label) {
  Analyzer a;
  const AQPStmt *inner = PeelOrderLimit(&root);
  a.result.accepted = inner && a.Walk(*inner, /*top_of_spine=*/true);
  // A recorded mark consumer whose Mark join was never reached (e.g. the
  // join sits in a build subtree, its mark column carried as payload).
  if (a.result.accepted && !a.pending_marks.empty()) {
    a.result.accepted = false;
    a.result.reject_reason = "markin:unmatched";
  }
#ifndef NDEBUG
  if (a.result.accepted) {
    fprintf(stderr, "[AQP-QJIT] accept label=%s joins=%d agg=%d markin=%d\n",
            label.c_str(), a.result.num_joins, a.result.has_aggregate ? 1 : 0,
            a.result.num_mark_in);
  } else {
    fprintf(stderr, "[AQP-QJIT] reject:%s label=%s\n",
            a.result.reject_reason.c_str(), label.c_str());
    if (a.result.reject_reason.find("CrossProduct") != std::string::npos)
      fprintf(stderr, "[AQP-QJIT] WARNING: query-jit does not support "
              "CrossProduct nodes, falling back to interpreter\n");
  }
#endif
  return a.result;
}

// ---------------------------------------------------------------------------
// Strict execution-step builder.
//
// Pass A (Decompose): recursive spine walk. Each JoinNode allocates an HT
//   descriptor (keys from the build-side condition attrs), recurses into the
//   build child (its steps are appended first => post-order = execution
//   order) and records a Probe op on the spine. Spine ops are collected
//   top-down and reversed (execution is scan-upward). Filters on source
//   columns commute with inner-join probes (a probe only drops rows or
//   attaches payload columns), so the reversal order is correct.
//
// Pass B (Resolve, steps LAST->FIRST): turns attr references into
//   QjitValueLoc {source col | HT key/payload slot}. Payload requirements
//   are appended to the producing HT's layout by its consumers; iterating
//   consumers before producers makes every HT layout final by the time its
//   build step resolves its sink. Filters stay restricted to step-source
//   columns: ir_to_llvm's EmitExpr silently passes rows whose columns are
//   not in the step schema (fatal when query-jit owns the result), hence
//   reject "filter:payload-ref".
//
// Pass C (Offsets): HT row layout
//   [validity prefix pad8 | keys i64 | VARCHAR 16B | INT32 4B].
//
// Strict whitelist rationale (unchanged from Phase 2): every admitted
// construct must have verified NULL-correct codegen; integer/IN leaves get
// per-leaf NULL guards (strict_null_guard) which are only correct in
// monotone AND/OR trees, so Logical NOT rejects except NOT(LIKE) (rewritten
// to Text_Not_Like inside the VARCHAR NULL guard).
// ---------------------------------------------------------------------------

namespace {

// §5.5 A+ predicate cost classification: a filter is "expensive" when it
// evaluates any string predicate (LIKE / string compare / string IN-set —
// per-row memchr/memcmp work); integer compares, integer IN-sets and
// IS NULL tests are "cheap" (a few cycles per row).
bool ExprIsExpensive(const AQPExpr *e) {
  if (!e)
    return false;
  switch (e->GetNodeType()) {
  case VarConstComparisonNode: {
    auto *c = static_cast<const SimplestVarConstComparison *>(e);
    return c->attr && c->attr->GetType() == StringVar;
  }
  case InExprNode: {
    auto *in_e = static_cast<const SimplestInExpr *>(e);
    return in_e->attr && in_e->attr->GetType() == StringVar;
  }
  case LogicalExprNode: {
    auto *log = static_cast<const SimplestLogicalExpr *>(e);
    return (log->left_expr && ExprIsExpensive(log->left_expr.get())) ||
           (log->right_expr && ExprIsExpensive(log->right_expr.get()));
  }
  default:
    return false; // IsNullExpr and friends: cheap
  }
}

// §5.5 A+ join-filter pushdown planning (post-resolution; reordering only
// moves Filter ops, which reference step-source columns exclusively, so
// resolved probe-key HT references stay valid):
//   1. Within the prefix of ops before the first Probe, move cheap filters
//      ahead of expensive ones (filters commute freely).
//   2. Emit a Guard for every Probe whose keys are all step-source columns:
//      build-key-0 range check, plus an HT-membership existence pre-probe
//      when expensive filters run between the guard point and that probe.
//   3. The first guard's key-0 column doubles as the morsel block-skip
//      column (per-block min/max vs build-key range).
void PlanJoinFilterPushdown(QjitQueryPlan &plan) {
  static const bool disabled =
      std::getenv("AQP_QJIT_NO_JOINFILTER") != nullptr;
  if (disabled)
    return;
  for (QjitStep &step : plan.steps) {
    size_t first_probe = step.ops.size();
    for (size_t i = 0; i < step.ops.size(); i++) {
      if (step.ops[i].kind == QjitStepOp::Probe) {
        first_probe = i;
        break;
      }
    }
    std::vector<QjitStepOp> cheap, costly;
    for (size_t i = 0; i < first_probe; i++) {
      if (ExprIsExpensive(step.ops[i].filter))
        costly.push_back(std::move(step.ops[i]));
      else
        cheap.push_back(std::move(step.ops[i]));
    }
    step.guard_pos = (int)cheap.size();
    size_t w = 0;
    for (auto &op : cheap)
      step.ops[w++] = std::move(op);
    bool expensive_seen = !costly.empty();
    for (auto &op : costly)
      step.ops[w++] = std::move(op);

    for (size_t i = first_probe; i < step.ops.size(); i++) {
      const QjitStepOp &op = step.ops[i];
      if (op.kind == QjitStepOp::Filter) {
        if (ExprIsExpensive(op.filter))
          expensive_seen = true;
        continue;
      }
      bool all_src = !op.keys.empty();
      for (const auto &kl : op.keys)
        all_src = all_src && kl.src_col >= 0;
      if (!all_src)
        continue;
      QjitStep::Guard g;
      g.op_index = (int)i;
      g.membership = expensive_seen;
      step.guards.push_back(g);
    }
    if (!step.guards.empty()) {
      step.block_skip_col =
          step.ops[step.guards[0].op_index].keys[0].src_col;
#ifndef NDEBUG
      int n_mem = 0;
      for (const auto &g : step.guards)
        n_mem += g.membership ? 1 : 0;
      fprintf(stderr,
              "[AQP-QJIT] joinfilter src=%s guards=%zu membership=%d "
              "blockskip_col=%d\n",
              step.source_table.c_str(), step.guards.size(), n_mem,
              step.block_skip_col);
#endif
    }
  }
}

struct PlanBuilder {
  QjitQueryPlan &plan;
  std::string &reason;

  // Per-step raw IR references Pass B resolves (parallel to plan.steps).
  struct RawStep {
    // Parallel to step.ops: probe-side attr per join condition (empty for
    // Filter ops).
    std::vector<std::vector<const SimplestAttr *>> probe_keys;
    // Result sink: root target_list attrs.
    std::vector<const SimplestAttr *> output_attrs;
    // Agg sink: agg_fns (arg attr may be null only for CountStar).
    std::vector<std::pair<const SimplestAttr *, SimplestAggFnType>> agg_fns;
  };
  std::vector<RawStep> raw;

  // (table_index, column_index) -> physical column name, recorded from scan
  // target_lists (authoritative; attrs above may carry splitter aliases).
  std::unordered_map<uint64_t, std::string> phys_names_;

  static uint64_t ColKey(unsigned table_index, unsigned column_index) {
    return ((uint64_t)table_index << 32) | column_index;
  }

  // Projection output binding (proj table_index, position) -> underlying
  // child attr. Mid-plan projections (e.g. DuckDB compressed
  // materialization) give join conditions / outputs above them the
  // projection's own binder index, which CollectTables (scans/chunks only)
  // cannot classify. The converter already erases the compress/decompress
  // wrappers (UnwrapToColumnRef), so target_list[k] IS the underlying attr.
  std::unordered_map<uint64_t, const SimplestAttr *> proj_remap_;

  void CollectProjRemap(const AQPStmt &node) {
    if (node.GetNodeType() == ProjectionNode) {
      const auto &p = static_cast<const SimplestProjection &>(node);
      for (size_t k = 0; k < node.target_list.size(); ++k)
        if (node.target_list[k])
          proj_remap_.emplace(ColKey(p.GetIndex(), (unsigned)k),
                              node.target_list[k].get());
    }
    for (const auto &c : node.children)
      if (c)
        CollectProjRemap(*c);
  }

  const SimplestAttr *ThroughProj(const SimplestAttr *a) const {
    for (int hops = 0; a && hops < 8; ++hops) {
      auto it =
          proj_remap_.find(ColKey(a->GetTableIndex(), a->GetColumnIndex()));
      if (it == proj_remap_.end() || it->second == a)
        break;
      a = it->second;
    }
    return a;
  }

  bool Fail(std::string r) {
    if (reason.empty())
      reason = std::move(r);
    return false;
  }

  // Map IR attr var_type to the expected FlatTable dtype. 0 = unsupported.
  static int32_t ExpectedDtype(const SimplestAttr &attr) {
    switch (attr.GetType()) {
    case IntVar:
    case Date:
      return attr.GetBitWidth() == 64 ? AQP_DTYPE_INT64 : AQP_DTYPE_INT32;
    case FloatVar:
      // DECIMAL stored as INT32/INT64 in FlatTable
      return attr.GetBitWidth() == 64 ? AQP_DTYPE_INT64 : AQP_DTYPE_INT32;
    case StringVar:
      return AQP_DTYPE_VARCHAR;
    default:
      return 0;
    }
  }

  // ------------------------------------------------------------------
  // Pass A — decompose
  // ------------------------------------------------------------------

  static void CollectTables(const AQPStmt &node, std::vector<unsigned> &out) {
    switch (node.GetNodeType()) {
    case ScanNode:
      out.push_back(static_cast<const SimplestScan &>(node).GetTableIndex());
      break;
    case ChunkNode:
      out.push_back(static_cast<const SimplestChunk &>(node).GetTableIndex());
      break;
    default:
      for (const auto &c : node.children)
        if (c)
          CollectTables(*c, out);
      break;
    }
  }

  static bool Contains(const std::vector<unsigned> &v, unsigned t) {
    for (unsigned x : v)
      if (x == t)
        return true;
    return false;
  }

  // Decompose the subtree rooted at `node` into one spine step (appended
  // last) plus, recursively, the steps of every build child. Returns the
  // spine step's index in plan.steps, or -1 (reason set).
  int Decompose(const AQPStmt *node, QjitStep::SinkKind sink, int sink_ht) {
    // Top-down collection; reversed into execution order at the end.
    std::vector<QjitStepOp> td_ops;
    std::vector<std::vector<const SimplestAttr *>> td_probe_keys;

    auto add_filter = [&](const AQPExpr *e) {
      QjitStepOp op;
      op.kind = QjitStepOp::Filter;
      op.filter = e;
      td_ops.push_back(op);
      td_probe_keys.emplace_back();
    };

    // Spine-local mark-IN consumers (mark_index -> count); the consumer
    // filter and its Mark join must lie on the SAME spine — the join emits
    // the synthesized in-set filter at its own position, any leftover
    // entries at the spine's end reject.
    std::unordered_map<unsigned, int> pending_marks;
    auto handle_qual = [&](const std::unique_ptr<AQPExpr> &q) -> bool {
      if (q && q->GetNodeType() == SingleAttrExprNode) {
        auto *sa = static_cast<const SimplestSingleAttrExpr *>(q.get());
        if (!sa->attr || sa->attr->GetType() != BoolVar)
          return Fail("markin:non-bool");
        pending_marks[sa->attr->GetTableIndex()]++;
        return true;
      }
      add_filter(q.get());
      return true;
    };

    const SimplestScan *scan = nullptr;
    const SimplestChunk *chunk = nullptr;
    while (!scan && !chunk) {
      switch (node->GetNodeType()) {
      case ScanNode:
        scan = static_cast<const SimplestScan *>(node);
        // Scan target_list attrs carry the PHYSICAL column names; attrs
        // seen higher in the tree (projection targets, join conditions)
        // may carry splitter alias names ("<table>_<idx>_<col>") that do
        // not exist in the FlatTable. Record the authoritative names.
        for (const auto &a : node->target_list)
          if (a)
            phys_names_.emplace(ColKey(a->GetTableIndex(),
                                       a->GetColumnIndex()),
                                a->GetColumnName());
        for (const auto &q : node->qual_vec)
          if (!handle_qual(q))
            return -1;
        break;

      case ChunkNode: {
        const auto &ck = static_cast<const SimplestChunk &>(*node);
        // Named + empty-contents = same-engine temp-table scan (served from
        // qjit_temps_ / converted CDC). Unnamed chunks with embedded
        // contents are IN-list constant tables — not supported.
        if (ck.GetChunkName().empty() || !ck.GetContents().empty()) {
          Fail("source:chunk-inline");
          return -1;
        }
        chunk = &ck;
        for (const auto &q : node->qual_vec)
          if (!handle_qual(q))
            return -1;
        break;
      }

      case FilterNode:
      case ProjectionNode:
        for (const auto &q : node->qual_vec)
          if (!handle_qual(q))
            return -1;
        if (node->children.size() != 1 || !node->children[0]) {
          Fail("spine:children!=1");
          return -1;
        }
        node = node->children[0].get();
        break;

      case JoinNode: {
        const auto &j = static_cast<const SimplestJoin &>(*node);
        if (j.GetSimplestJoinType() == Mark) {
          auto it = pending_marks.find(j.GetMarkIndex());
          if (it == pending_marks.end()) {
            Fail("join:Mark");
            return -1;
          }
          MarkInPattern pat;
          std::string r = MatchMarkInJoin(j, pat);
          if (!r.empty()) {
            Fail(std::move(r));
            return -1;
          }
          std::string err;
          auto in_expr = BuildMarkInExpr(pat, err);
          if (!in_expr) {
            Fail(std::move(err));
            return -1;
          }
          if (--it->second == 0)
            pending_marks.erase(it);
          for (const auto &q : node->qual_vec)
            if (!handle_qual(q))
              return -1;
          add_filter(in_expr.get());
          plan.owned_exprs.push_back(std::move(in_expr));
          node = pat.probe_child;
          break;
        }
        if (j.GetSimplestJoinType() != Inner) {
          Fail(std::string("join:") + JoinTypeName(j.GetSimplestJoinType()));
          return -1;
        }
        if (node->children.size() != 2 || !node->children[0] ||
            !node->children[1]) {
          Fail("join:children!=2");
          return -1;
        }
        int build = j.GetBuildChild();
        if (build != 0 && build != 1) {
          Fail("join:build-side-unannotated");
          return -1;
        }
        const AQPStmt *build_child = node->children[build].get();
        const AQPStmt *probe_child = node->children[1 - build].get();
        if (j.join_conditions.empty()) {
          Fail("join:no-conditions");
          return -1;
        }

        std::vector<unsigned> build_tables, probe_tables;
        CollectTables(*build_child, build_tables);
        CollectTables(*probe_child, probe_tables);

        QjitHtDesc ht;
        ht.build_tables = build_tables;
        std::vector<const SimplestAttr *> probe_keys;
        for (const auto &cond : j.join_conditions) {
          if (!cond ||
              cond->GetSimplestExprType() != SimplestExprType::Equal) {
            Fail("join:non-equi-condition");
            return -1;
          }
          if (!cond->left_attr || !cond->right_attr) {
            Fail("join:condition-missing-attr");
            return -1;
          }
          const SimplestAttr *l = ThroughProj(cond->left_attr.get());
          const SimplestAttr *r = ThroughProj(cond->right_attr.get());
          const SimplestAttr *bk = nullptr, *pk = nullptr;
          bool l_in_b = Contains(build_tables, l->GetTableIndex());
          bool r_in_b = Contains(build_tables, r->GetTableIndex());
          bool l_in_p = Contains(probe_tables, l->GetTableIndex());
          bool r_in_p = Contains(probe_tables, r->GetTableIndex());
          if (l_in_b && r_in_p && !r_in_b && !l_in_p) {
            bk = l;
            pk = r;
          } else if (r_in_b && l_in_p && !l_in_b && !r_in_p) {
            bk = r;
            pk = l;
          } else {
            Fail("join:condition-sides l=(" +
                 std::to_string(l->GetTableIndex()) + "," +
                 std::to_string(l->GetColumnIndex()) + ")" + l->GetColumnName() +
                 "[b" + std::to_string(l_in_b) + "p" + std::to_string(l_in_p) +
                 "] r=(" + std::to_string(r->GetTableIndex()) + "," +
                 std::to_string(r->GetColumnIndex()) + ")" + r->GetColumnName() +
                 "[b" + std::to_string(r_in_b) + "p" + std::to_string(r_in_p) +
                 "]");
            return -1;
          }
          // Integer keys (stored sign-extended as i64 in HT layout).
          int32_t bk_dt = ExpectedDtype(*bk);
          int32_t pk_dt = ExpectedDtype(*pk);
          if ((bk_dt != AQP_DTYPE_INT32 && bk_dt != AQP_DTYPE_INT64) ||
              (pk_dt != AQP_DTYPE_INT32 && pk_dt != AQP_DTYPE_INT64)) {
            Fail("join:key-dtype");
            return -1;
          }
          QjitHtCol kc;
          kc.table_index = bk->GetTableIndex();
          kc.column_index = bk->GetColumnIndex();
          kc.column_name = bk->GetColumnName();
          kc.dtype = bk_dt;
          ht.cols.push_back(std::move(kc));
          probe_keys.push_back(pk);
        }
        ht.num_keys = (uint32_t)ht.cols.size();
        int ht_id = (int)plan.hts.size();
        plan.hts.push_back(std::move(ht));

        // Build child first: its steps precede this spine step.
        if (Decompose(build_child, QjitStep::HtBuild, ht_id) < 0)
          return -1;

        // Execution order is the reverse of collection order, so emit the
        // join quals BEFORE the probe op here (=> they run after it).
        for (const auto &q : node->qual_vec)
          if (!handle_qual(q))
            return -1;
        QjitStepOp op;
        op.kind = QjitStepOp::Probe;
        op.ht_id = ht_id;
        td_ops.push_back(std::move(op));
        td_probe_keys.push_back(std::move(probe_keys));

        node = probe_child;
        break;
      }

      case AggregateNode:
        Fail("agg:not-top");
        return -1;

      default:
        Fail(std::string("node:") + NodeTypeName(node->GetNodeType()));
        return -1;
      }
    }

    if (!pending_marks.empty()) {
      Fail("markin:unmatched");
      return -1;
    }

    QjitStep step;
    if (scan) {
      step.source_table = scan->GetTableName();
      step.source_table_index = scan->GetTableIndex();
    } else {
      step.source_table = chunk->GetChunkName();
      step.source_table_index = chunk->GetTableIndex();
      step.source_is_temp = true;
    }
    step.sink = sink;
    step.sink_ht = sink_ht;
    RawStep rs;
    for (size_t i = td_ops.size(); i-- > 0;) {
      step.ops.push_back(std::move(td_ops[i]));
      rs.probe_keys.push_back(std::move(td_probe_keys[i]));
    }
    plan.steps.push_back(std::move(step));
    raw.push_back(std::move(rs));
    return (int)plan.steps.size() - 1;
  }

  // ------------------------------------------------------------------
  // Pass B — resolution (steps LAST -> FIRST)
  // ------------------------------------------------------------------

  // Register `attr`-like reference as a step source column (dedup by
  // table+column). Returns index into step.cols or -1.
  int RegisterSourceCol(QjitStep &step, unsigned table_index,
                        unsigned column_index, const std::string &name,
                        int32_t dtype) {
    if (dtype == 0) {
      Fail("expr:attr-type");
      return -1;
    }
    for (size_t i = 0; i < step.cols.size(); i++) {
      if (step.cols[i].table_index == table_index &&
          step.cols[i].column_index == column_index) {
        if (step.cols[i].expected_dtype != dtype) {
          Fail("expr:attr-dtype-conflict");
          return -1;
        }
        return (int)i;
      }
    }
    QjitColumnRef ref;
    ref.table_index = table_index;
    ref.column_index = column_index;
    // Prefer the physical name recorded from the scan target_list; `name`
    // may be a splitter alias (resolution against the FlatTable is by
    // name, so an alias would miss and force a fallback).
    auto pn = phys_names_.find(ColKey(table_index, column_index));
    ref.column_name = pn != phys_names_.end() ? pn->second : name;
    ref.expected_dtype = dtype;
    step.cols.push_back(std::move(ref));
    return (int)step.cols.size() - 1;
  }

  // Resolve a value reference in `step`'s environment using probe ops
  // [0, op_limit): step source column, or key/payload slot of a probed HT
  // (appending a payload requirement to the HT layout if new — legal
  // because consumers resolve before producers).
  bool ResolveRef(QjitStep &step, size_t op_limit, unsigned table_index,
                  unsigned column_index, const std::string &name,
                  int32_t dtype, QjitValueLoc &loc) {
    loc = QjitValueLoc();
    loc.dtype = dtype;
    if (dtype == 0)
      return Fail("expr:attr-type");

    if (table_index == step.source_table_index) {
      int idx = RegisterSourceCol(step, table_index, column_index, name,
                                  dtype);
      if (idx < 0)
        return false;
      loc.src_col = idx;
      return true;
    }

    for (size_t i = 0; i < op_limit && i < step.ops.size(); i++) {
      if (step.ops[i].kind != QjitStepOp::Probe)
        continue;
      int ht_id = step.ops[i].ht_id;
      QjitHtDesc &ht = plan.hts[ht_id];
      if (!Contains(ht.build_tables, table_index))
        continue;
      // Keys are scanned too: a payload requirement matching a key attr
      // reuses the key slot (i64 load + trunc in codegen).
      for (size_t c = 0; c < ht.cols.size(); c++) {
        if (ht.cols[c].table_index == table_index &&
            ht.cols[c].column_index == column_index) {
          if (ht.cols[c].dtype != dtype)
            return Fail("resolve:payload-dtype-conflict");
          loc.ht_id = ht_id;
          loc.layout_col = (int)c;
          return true;
        }
      }
      QjitHtCol pc;
      pc.table_index = table_index;
      pc.column_index = column_index;
      pc.column_name = name;
      pc.dtype = dtype;
      ht.cols.push_back(std::move(pc));
      loc.ht_id = ht_id;
      loc.layout_col = (int)ht.cols.size() - 1;
      return true;
    }
    return Fail("resolve:attr-unreachable");
  }

  bool ResolveAttr(QjitStep &step, size_t op_limit, const SimplestAttr &attr,
                   QjitValueLoc &loc) {
    const SimplestAttr *a = ThroughProj(&attr);
    return ResolveRef(step, op_limit, a->GetTableIndex(),
                      a->GetColumnIndex(), a->GetColumnName(),
                      ExpectedDtype(*a), loc);
  }

  // ---- Strict filter-expression whitelist (source columns only) ----

  int RegisterFilterAttr(QjitStep &step, const SimplestAttr &attr) {
    if (attr.GetTableIndex() != step.source_table_index) {
      Fail("filter:payload-ref");
      return -1;
    }
    return RegisterSourceCol(step, attr.GetTableIndex(),
                             attr.GetColumnIndex(), attr.GetColumnName(),
                             ExpectedDtype(attr));
  }

  bool CheckVarConst(QjitStep &step, const SimplestVarConstComparison &cmp) {
    if (!cmp.attr || !cmp.const_var)
      return Fail("expr:varconst-incomplete");
    int32_t dtype = ExpectedDtype(*cmp.attr);
    SimplestExprType op = cmp.GetSimplestExprType();
    SimplestVarType ct = cmp.const_var->GetType();
    if (dtype == AQP_DTYPE_INT32 || dtype == AQP_DTYPE_INT64) {
      if (ct != IntVar && ct != Date && ct != FloatVar)
        return Fail("expr:int-const-type");
      switch (op) {
      case SimplestExprType::Equal:
      case SimplestExprType::NotEqual:
      case SimplestExprType::LessThan:
      case SimplestExprType::GreaterThan:
      case SimplestExprType::LessEqual:
      case SimplestExprType::GreaterEqual:
        break;
      default:
        return Fail("expr:int-op");
      }
    } else if (dtype == AQP_DTYPE_VARCHAR) {
      if (ct != StringVar)
        return Fail("expr:str-const-type");
      switch (op) {
      case SimplestExprType::Equal:
      case SimplestExprType::NotEqual:
      case SimplestExprType::TextLike:
      case SimplestExprType::Text_Not_Like:
      case SimplestExprType::LessThan:
      case SimplestExprType::GreaterThan:
      case SimplestExprType::LessEqual:
      case SimplestExprType::GreaterEqual:
        break;
      default:
        return Fail("expr:str-op");
      }
    } else {
      return Fail("expr:attr-type");
    }
    return RegisterFilterAttr(step, *cmp.attr) >= 0;
  }

  bool CheckFilterStrict(QjitStep &step, const AQPExpr *e) {
    if (!e)
      return Fail("expr:null");
    switch (e->GetNodeType()) {
    case VarConstComparisonNode:
      return CheckVarConst(
          step, *static_cast<const SimplestVarConstComparison *>(e));
    case IsNullExprNode: {
      auto *isnull = static_cast<const SimplestIsNullExpr *>(e);
      if (!isnull->attr)
        return Fail("expr:isnull-no-attr");
      return RegisterFilterAttr(step, *isnull->attr) >= 0;
    }
    case InExprNode: {
      auto *in_e = static_cast<const SimplestInExpr *>(e);
      if (!in_e->attr)
        return Fail("expr:in-no-attr");
      if (in_e->values.empty())
        return Fail("expr:in-empty");
      int32_t dtype = ExpectedDtype(*in_e->attr);
      for (const auto &v : in_e->values) {
        if (!v)
          return Fail("expr:in-null-value");
        SimplestVarType vt = v->GetType();
        if (dtype == AQP_DTYPE_INT32 && vt != IntVar && vt != Date)
          return Fail("expr:in-value-type");
        if (dtype == AQP_DTYPE_VARCHAR && vt != StringVar)
          return Fail("expr:in-value-type");
      }
      return RegisterFilterAttr(step, *in_e->attr) >= 0;
    }
    case VarComparisonNode: {
      // Same-source col-vs-col comparison (SDS temps fold two joined tables
      // into one temp, turning the join predicate into a var-var filter).
      auto *vv = static_cast<const SimplestVarComparison *>(e);
      if (!vv->left_attr || !vv->right_attr)
        return Fail("expr:varvar-incomplete");
      if (ExpectedDtype(*vv->left_attr) != AQP_DTYPE_INT32 ||
          ExpectedDtype(*vv->right_attr) != AQP_DTYPE_INT32)
        return Fail("expr:varvar-type");
      switch (vv->GetSimplestExprType()) {
      case SimplestExprType::Equal:
      case SimplestExprType::NotEqual:
      case SimplestExprType::LessThan:
      case SimplestExprType::GreaterThan:
      case SimplestExprType::LessEqual:
      case SimplestExprType::GreaterEqual:
        break;
      default:
        return Fail("expr:varvar-op");
      }
      return RegisterFilterAttr(step, *vv->left_attr) >= 0 &&
             RegisterFilterAttr(step, *vv->right_attr) >= 0;
    }
    case LogicalExprNode: {
      auto *log = static_cast<const SimplestLogicalExpr *>(e);
      if (log->GetLogicalOp() == SimplestLogicalOp::LogicalNot) {
        // Only NOT(LIKE on VARCHAR) is NULL-safe (rewritten to
        // Text_Not_Like inside EmitVarConst's NULL guard).
        const AQPExpr *child = log->right_expr.get();
        if (!child || child->GetNodeType() != VarConstComparisonNode)
          return Fail("expr:not");
        auto *cmp = static_cast<const SimplestVarConstComparison *>(child);
        if (cmp->GetSimplestExprType() != SimplestExprType::TextLike)
          return Fail("expr:not");
        return CheckVarConst(step, *cmp);
      }
      if (!log->left_expr || !log->right_expr)
        return Fail("expr:logical-incomplete");
      return CheckFilterStrict(step, log->left_expr.get()) &&
             CheckFilterStrict(step, log->right_expr.get());
    }
    default:
      return Fail(std::string("expr:") + NodeTypeName(e->GetNodeType()) + "#" +
                  std::to_string(static_cast<int>(e->GetNodeType())));
    }
  }

  // ---- Per-step resolution ----

  bool ResolveStep(size_t si) {
    QjitStep &step = plan.steps[si];
    RawStep &rs = raw[si];

    // Sink (full op environment).
    switch (step.sink) {
    case QjitStep::Result:
      for (const SimplestAttr *attr : rs.output_attrs) {
        QjitValueLoc loc;
        if (!ResolveAttr(step, step.ops.size(), *attr, loc))
          return false;
        step.outputs.push_back(loc);
      }
      break;
    case QjitStep::Agg:
      for (const auto &fn : rs.agg_fns) {
        QjitAggCellPlan cell;
        switch (fn.second) {
        case Min:       cell.fn = QjitAggFn::Min; break;
        case Max:       cell.fn = QjitAggFn::Max; break;
        case Sum:       cell.fn = QjitAggFn::Sum; break;
        case Count:     cell.fn = QjitAggFn::Count; break;
        case CountStar: cell.fn = QjitAggFn::CountStar; break;
        case Average:   cell.fn = QjitAggFn::Average; break;
        default:
          return Fail(std::string("agg:") + AggFnName(fn.second));
        }
        if (fn.second != CountStar) {
          if (!fn.first)
            return Fail("agg:missing-arg");
          cell.has_arg = true;
          if (!ResolveAttr(step, step.ops.size(), *fn.first, cell.arg))
            return false;
        }
        step.agg_cells.push_back(cell);
      }
      break;
    case QjitStep::HtBuild: {
      // The target layout is final: every consumer (a later step) has
      // already been resolved. Note: the loop body may append payload
      // requirements to DEEPER hts (this step's own probes), never to
      // hts[step.sink_ht] itself (its build set excludes our env).
      QjitHtDesc &ht = plan.hts[step.sink_ht];
      for (size_t c = 0; c < ht.cols.size(); c++) {
        // Copy the identity: ResolveRef may push to plan.hts vectors? No —
        // it only appends to QjitHtDesc::cols of OTHER hts; but appending
        // to ht.cols here is impossible, so iteration by index is safe.
        QjitHtCol id = ht.cols[c];
        QjitValueLoc loc;
        if (!ResolveRef(step, step.ops.size(), id.table_index,
                        id.column_index, id.column_name, id.dtype, loc))
          return false;
        step.outputs.push_back(loc);
      }
      break;
    }
    }

    // Probe keys: op i may only read hts probed by ops [0, i).
    for (size_t i = 0; i < step.ops.size(); i++) {
      if (step.ops[i].kind != QjitStepOp::Probe)
        continue;
      for (const SimplestAttr *pk : rs.probe_keys[i]) {
        QjitValueLoc loc;
        if (!ResolveAttr(step, i, *pk, loc))
          return false;
        step.ops[i].keys.push_back(loc);
      }
    }

    // Filters: strictly whitelisted, step-source columns only.
    for (auto &op : step.ops) {
      if (op.kind != QjitStepOp::Filter)
        continue;
      if (!CheckFilterStrict(step, op.filter))
        return false;
    }
    return true;
  }

  // ------------------------------------------------------------------
  // Pass C — HT row offsets
  // ------------------------------------------------------------------

  void AssignOffsets() {
    for (QjitHtDesc &ht : plan.hts) {
      uint32_t ncols = (uint32_t)ht.cols.size();
      ht.prefix_bytes = (((ncols + 7) / 8) + 7) & ~7u;
      uint32_t off = ht.prefix_bytes;
      for (uint32_t c = 0; c < ht.num_keys; c++) {
        ht.cols[c].offset = off;
        off += 8; // keys stored sign-extended i64
      }
      for (uint32_t c = ht.num_keys; c < ncols; c++) {
        if (ht.cols[c].dtype == AQP_DTYPE_VARCHAR) {
          ht.cols[c].offset = off;
          off += 16; // QjitString
        }
      }
      for (uint32_t c = ht.num_keys; c < ncols; c++) {
        if (ht.cols[c].dtype != AQP_DTYPE_VARCHAR) {
          ht.cols[c].offset = off;
          off += 4; // INT32
        }
      }
      ht.tuple_size = off;
    }
  }
};

} // namespace

bool BuildExecutionSteps(const AQPStmt &root, QjitQueryPlan &out,
                         std::string &reason,
                         bool enable_range_guard,
                         bool enable_block_skip,
                         bool enable_membership) {
  out = QjitQueryPlan();
  reason.clear();

  // Peel ORDER BY / LIMIT wrappers. They don't affect the scan/join/agg
  // plan — we handle them as post-processing on result.rows.
  const AQPStmt *inner = &root;
  const SimplestOrderBy *peeled_order = nullptr;
  const SimplestLimit *peeled_limit = nullptr;
  while (inner) {
    if (inner->GetNodeType() == OrderNode) {
      peeled_order = static_cast<const SimplestOrderBy *>(inner);
      inner = inner->children.empty() ? nullptr : inner->children[0].get();
    } else if (inner->GetNodeType() == LimitNode) {
      peeled_limit = static_cast<const SimplestLimit *>(inner);
      inner = inner->children.empty() ? nullptr : inner->children[0].get();
    } else {
      break;
    }
  }
  if (!inner)
    return false;

  PlanBuilder pb{out, reason};
  pb.CollectProjRemap(*inner);

  // Peel the root: Projection* [Aggregate]. DuckDB plans `SELECT MIN(..)`
  // as Projection -> Aggregate; the projection only reorders the single
  // agg row. Quals above the aggregate would filter the agg result —
  // unsupported (reject), and >1 projection above the agg would make the
  // target mapping indirect (never seen from DuckDB plans; reject).
  const AQPStmt *node = inner;
  const SimplestAggregate *agg = nullptr;
  const AQPStmt *proj_above_agg = nullptr;
  {
    const AQPStmt *p = inner;
    int projs = 0;
    while (p->GetNodeType() == ProjectionNode ||
           p->GetNodeType() == OrderNode ||
           p->GetNodeType() == LimitNode) {
      if (p->children.size() != 1 || !p->children[0])
        return pb.Fail("spine:children!=1");
      if (p->GetNodeType() == OrderNode) {
        if (!peeled_order)
          peeled_order = static_cast<const SimplestOrderBy *>(p);
        p = p->children[0].get();
        continue;
      }
      if (p->GetNodeType() == LimitNode) {
        if (!peeled_limit)
          peeled_limit = static_cast<const SimplestLimit *>(p);
        p = p->children[0].get();
        continue;
      }
      if (p->children[0]->GetNodeType() == AggregateNode ||
          p->children[0]->GetNodeType() == ProjectionNode ||
          p->children[0]->GetNodeType() == OrderNode ||
          p->children[0]->GetNodeType() == LimitNode) {
        // keep peeling only while an aggregate could be underneath
        if (p->children[0]->GetNodeType() == AggregateNode) {
          if (!p->qual_vec.empty())
            return pb.Fail("filter:above-agg");
          if (projs >= 1)
            return pb.Fail("agg:proj-chain");
          proj_above_agg = p;
          agg = static_cast<const SimplestAggregate *>(p->children[0].get());
          break;
        }
        if (!p->qual_vec.empty())
          break; // quals => treat as plain spine (no agg may follow w/o reject)
        projs++;
        p = p->children[0].get();
        continue;
      }
      break;
    }
    if (!agg && inner->GetNodeType() == AggregateNode)
      agg = static_cast<const SimplestAggregate *>(inner);
  }

  if (agg) {
    if (!agg->groups.empty())
      return pb.Fail("agg:grouped");
    if (agg->agg_fns.empty())
      return pb.Fail("agg:no-functions");
    if (!agg->qual_vec.empty())
      return pb.Fail("filter:above-agg"); // HAVING-style qual on agg node
    if (agg->children.size() != 1 || !agg->children[0])
      return pb.Fail("agg:children!=1");
    node = agg->children[0].get();
    out.has_agg = true;
  }

  int root_step = pb.Decompose(node, agg ? QjitStep::Agg : QjitStep::Result,
                               -1);
  if (root_step < 0)
    return false;

  // Root sink raw data.
  PlanBuilder::RawStep &rs = pb.raw[root_step];
  if (agg) {
    for (const auto &fn : agg->agg_fns)
      rs.agg_fns.emplace_back(fn.first.get(), fn.second);
    // Result-column -> agg-cell mapping. With a projection above, target
    // attrs reference (agg_index, position-in-agg_fns). NOTE: CountStar
    // produces NO agg_fns entry in the converter, shifting positions; an
    // out-of-range index rejects here, and the no-projection case is
    // caught by the adapter's output-count check.
    if (proj_above_agg) {
      if (proj_above_agg->target_list.empty())
        return pb.Fail("output:empty-target-list");
      for (const auto &attr : proj_above_agg->target_list) {
        if (!attr)
          return pb.Fail("output:null-attr");
        if (attr->GetTableIndex() != agg->GetAggIndex())
          return pb.Fail("agg:proj-ref");
        unsigned cell = attr->GetColumnIndex();
        if (cell >= agg->agg_fns.size())
          return pb.Fail("agg:proj-cell-range");
        out.agg_output_cells.push_back((int)cell);
      }
    } else {
      for (size_t i = 0; i < agg->agg_fns.size(); i++)
        out.agg_output_cells.push_back((int)i);
    }
  } else {
    if (inner->target_list.empty())
      return pb.Fail("output:empty-target-list");
    for (const auto &attr : inner->target_list) {
      if (!attr)
        return pb.Fail("output:null-attr");
      rs.output_attrs.push_back(attr.get());
    }
  }

  // Pass B: resolve LAST -> FIRST (consumers before producers).
  for (size_t si = out.steps.size(); si-- > 0;)
    if (!pb.ResolveStep(si))
      return false;

  pb.AssignOffsets();
  if (enable_range_guard)
    PlanJoinFilterPushdown(out);
  if (!enable_block_skip)
    for (auto &step : out.steps)
      step.block_skip_col = -1;
  if (!enable_membership)
    for (auto &step : out.steps)
      for (auto &g : step.guards)
        g.membership = false;

  // Record peeled ORDER BY / LIMIT for post-QJIT sorting.
  if (peeled_limit && peeled_limit->limit_val.type ==
          SimplestLimitType::CONSTANT_VALUE)
    out.limit = (int64_t)peeled_limit->limit_val.val;
  if (peeled_order) {
    const auto &last_step = out.steps.back();
    for (const auto &o : peeled_order->orders) {
      if (!o.attr)
        continue;
      const std::string &col_name = o.attr->GetColumnName();
      int col_idx = -1;
      if (out.has_agg) {
        // Agg output columns: match against the projection target names
        if (proj_above_agg) {
          for (size_t i = 0; i < proj_above_agg->target_list.size(); i++) {
            if (proj_above_agg->target_list[i] &&
                proj_above_agg->target_list[i]->GetColumnName() == col_name) {
              col_idx = (int)i;
              break;
            }
          }
        }
      } else {
        for (size_t i = 0; i < last_step.outputs.size(); i++) {
          if (i < inner->target_list.size() && inner->target_list[i] &&
              inner->target_list[i]->GetColumnName() == col_name) {
            col_idx = (int)i;
            break;
          }
        }
      }
      if (col_idx >= 0) {
        int32_t dt = AQP_DTYPE_VARCHAR;
        if (out.has_agg && (size_t)col_idx < out.agg_output_cells.size()) {
          size_t cell = (size_t)out.agg_output_cells[col_idx];
          if (cell < last_step.agg_cells.size())
            dt = last_step.agg_cells[cell].arg.dtype;
        } else if (!out.has_agg && (size_t)col_idx < last_step.outputs.size()) {
          dt = last_step.outputs[col_idx].dtype;
        }
        out.order_by.push_back(
            {col_idx, o.order_type != SimplestOrderType::Descending, dt});
      }
    }
  }

  return true;
}

} // namespace qjit
