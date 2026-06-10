/*
 * NodeBasedSplitter: full implementation of the DuckDB MiddleOptimize-driven
 * split strategy.  See node_based_splitter.h for the overall loop description.
 */

#ifdef HAVE_DUCKDB

#include "split/node_based_splitter.h"

#include "duckdb/planner/operator/logical_column_data_get.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"

#include <iostream>

namespace middleware {

NodeBasedSplitter::NodeBasedSplitter(EngineAdapter *exec_adapter,
                                     DuckDBAdapter *plan_adapter,
                                     bool enable_debug_print)
    : AQPSplitter(exec_adapter), plan_adapter_(plan_adapter),
      external_execution_(exec_adapter !=
                          static_cast<EngineAdapter *>(plan_adapter)),
      enable_debug_print_(enable_debug_print) {}

void NodeBasedSplitter::Preprocess(
    std::unique_ptr<ir_sql_converter::AQPStmt> & /*ir*/) {
  ctx_ = plan_adapter_->GetClientContext();

  // Take ownership of the already-FilterOptimized plan.
  plan_ = plan_adapter_->TakePlan();

  // Initialise DuckDB split machinery for this query.
  qs_ = std::make_unique<duckdb::QuerySplit>(*ctx_);
  sp_ = std::make_unique<duckdb::SubqueryPreparer>(plan_adapter_->GetBinder(),
                                                   *ctx_);
  reorder_get_ = std::make_unique<duckdb::ReorderGet>(*ctx_);

  subqueries_.clear();
  proj_expr_.clear();
  table_expr_queue_.clear();
  last_sibling_node_ = nullptr;
  merge_sibling_expr_ = false;
  terminal_ = false;
}

void NodeBasedSplitter::RunMiddleOptimize() {
  if (ctx_->transaction.IsAutoCommit())
    ctx_->transaction.BeginTransaction();
  {
    duckdb::Optimizer optimizer(plan_adapter_->GetBinder(), *ctx_);
    plan_ = optimizer.MiddleOptimize(std::move(plan_));
    if (enable_debug_print_) {
      std::cout << "[NodeBased] plan after MiddleOptimize:\n";
      plan_->Print();
    }
  }
  if (ctx_->transaction.IsAutoCommit())
    ctx_->transaction.Commit();
}

std::unique_ptr<ir_sql_converter::AQPStmt>
NodeBasedSplitter::TakePlanAsIR() {
  plan_adapter_->SetPlan(std::move(plan_));
  return plan_adapter_->ConvertPlanToIR();
}

// ─────────────────────────────────────────────────────────────────────────────
// SplitIR
// Runs BLOCK 1 + BLOCK 2 then either signals terminal or extracts a sub-plan.
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<SubqueryExtraction> NodeBasedSplitter::SplitIR(
    ir_sql_converter::AQPStmt * /*remaining_ir*/) {

  // ── BLOCK 1 (ALWAYS_SPLIT=true: runs every iteration except the first) ──
  // Merge pending subqueries (from the previous UpdateRemainingIR) back into
  // the plan before re-optimising.
  if (!subqueries_.empty()) {
    sp_->MergeSubquery(plan_, std::move(subqueries_));
    plan_ = sp_->UpdateProjHead(std::move(plan_), proj_expr_);
    merge_sibling_expr_ = false;
  }

  RunMiddleOptimize();

  plan_ = qs_->Clear(std::move(plan_));
  plan_ = qs_->Split(std::move(plan_), true);
  subqueries_ = qs_->GetSubqueries();
  table_expr_queue_ = qs_->GetTableExprQueue();
  proj_expr_ = qs_->GetProjExpr();

  // ── BLOCK 2 (enable_dbshaker_split_jop=false: always runs) ─────────────
  reorder_get_->ReorderTables(subqueries_);
  sp_->MergeSubquery(plan_, std::move(subqueries_));
  plan_ = sp_->UpdateProjHead(std::move(plan_), proj_expr_);
  merge_sibling_expr_ = false;

  plan_ = qs_->Clear(std::move(plan_));
  plan_ = qs_->Split(std::move(plan_), true);
  subqueries_ = qs_->GetSubqueries();
  table_expr_queue_ = qs_->GetTableExprQueue();
  proj_expr_ = qs_->GetProjExpr();

  if (enable_debug_print_)
    std::cout << "[NodeBased] SplitIR: subquery groups="
              << subqueries_.size() << "\n";

  // ── Early terminal: nothing left to split ───────────────────────────────
  // subqueries empty: plan is already the final executable form.
  // subqueries size==1: merge the single child and hand it off.
  // In both cases return is_final so ExecuteOneIteration sets remaining_ir.
  if (subqueries_.empty()) {
    terminal_ = true;
    auto extraction =
        std::make_unique<SubqueryExtraction>(std::set<unsigned int>{});
    extraction->is_final = true;
    extraction->sub_ir = TakePlanAsIR();
    return extraction;
  }
  if (subqueries_.size() == 1) {
    auto &child_node = subqueries_.front()[0];
    bool merged = false;
    // NO UpdateProjHead here — matches client_context.cpp early-terminal path.
    sp_->MergeToSubquery(plan_, child_node, merged);
    terminal_ = true;
    auto extraction =
        std::make_unique<SubqueryExtraction>(std::set<unsigned int>{});
    extraction->is_final = true;
    extraction->sub_ir = TakePlanAsIR();
    return extraction;
  }

  // ── Normal: extract the first subquery group as a sub-plan ─────────────
  // Save sibling for MergeSibling in UpdateRemainingIR.
  last_sibling_node_ = nullptr;
  if (subqueries_.front().size() > 1)
    last_sibling_node_ = std::move(subqueries_.front()[1]);

  sp_->ClearOldTableIndex();
  sp_->AddOldTableIndex(subqueries_.front()[0]); // read before move
  auto sub_plan =
      sp_->GenerateProjHead(plan_, std::move(subqueries_.front()[0]),
                            table_expr_queue_, proj_expr_, merge_sibling_expr_);
  subqueries_.pop_front();
  table_expr_queue_.pop_front();

  // Resolve types and convert sub-plan to IR.
  // sub_plan is separate from plan_ (returned by GenerateProjHead); plan_ is
  // unchanged here and remains valid for future iterations.
  sub_plan->ResolveOperatorTypes();
  sub_plan_types_ = sub_plan->types;

  plan_adapter_->SetPlan(std::move(sub_plan));
  auto sub_ir = plan_adapter_->ConvertPlanToIR();

  // Populate table_index_to_name_ so ComputeColumnAlias can resolve
  // table names for temp table column aliases (e.g. "movie_info_7_movie_id").
  CollectTableNames(sub_ir.get());

  if (enable_debug_print_) {
    std::cout << "[NodeBased] sub-query IR:\n";
    sub_ir->Print();
  }

  auto extraction =
      std::make_unique<SubqueryExtraction>(std::set<unsigned int>{});
  extraction->sub_ir = std::move(sub_ir);
  return extraction;
}

bool NodeBasedSplitter::IsComplete(
    const ir_sql_converter::AQPStmt * /*remaining_ir*/) {
  return terminal_;
}

// ─────────────────────────────────────────────────────────────────────────────
// UpdateRemainingIR
// Called by ExecuteOneIteration after the sub-SQL has been executed and the
// temp table created.  Inserts a CHUNK_GET node and advances split state.
// Returns the final plan IR when late-terminal, nullptr otherwise.
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<ir_sql_converter::AQPStmt>
NodeBasedSplitter::UpdateRemainingIR(
    std::unique_ptr<ir_sql_converter::AQPStmt> remaining_ir,
    const std::set<unsigned int> & /*executed_table_indices*/,
    unsigned int /*temp_table_index*/, const std::string &temp_table_name,
    uint64_t temp_table_cardinality,
    const std::vector<std::pair<unsigned int, unsigned int>> & /*col_mappings*/,
    const std::vector<std::string> &col_names) {

  if (external_execution_) {
    // Execution happened on a non-DuckDB backend; DuckDB never ran
    // ExecuteSQLandCreateTempTable so GetTempTableIndex() is stale.
    // Allocate a fresh DuckDB index and register the temp table name.
    plan_adapter_->RegisterExternalTempTable(temp_table_name, sub_plan_types_,
                                             col_names);
  }

  // Register the DuckDB-assigned index → temp table name so that
  // ComputeColumnAlias can resolve it in future iterations.
  // (The index from GetMaxTableIndex()+iteration_count_ used by
  // ExecuteOneIteration is unrelated to DuckDB's GenerateTableIndex().)
  AddTableMapping(plan_adapter_->GetTempTableIndex(), temp_table_name);

  // Tell SubqueryPreparer the DuckDB chunk index assigned by
  // ExecuteSQLandCreateTempTable (or RegisterExternalTempTable for external
  // backends) stored in temp_table_index_ by the adapter.
  sp_->SetNewTableIndex(plan_adapter_->GetTempTableIndex());

  // Build an empty ColumnDataCollection with the correct types and inject it
  // as a CHUNK_GET node so DuckDB can track cardinality for MiddleOptimize.
  // For DuckDB-native execution: temp_table_types matches sub_plan_types_.
  // For external execution: sub_plan_types_ is the correct type source.
  auto collection =
      duckdb::make_uniq<duckdb::ColumnDataCollection>(*ctx_, sub_plan_types_);
  if (!external_execution_) {
    collection->Types() = plan_adapter_->temp_table_types;
  }
  sp_->MergeDataChunk(subqueries_, std::move(collection),
                      temp_table_cardinality);

  // Merge sibling (parallel-execution path; ENABLE_PARALLEL_EXECUTION=false).
  if (last_sibling_node_) {
    merge_sibling_expr_ =
        sp_->MergeSibling(subqueries_, std::move(last_sibling_node_));
  } else {
    merge_sibling_expr_ = false;
  }

  sp_->UpdateSubqueriesIndex(subqueries_);
  table_expr_queue_ =
      sp_->UpdateTableExpr(std::move(table_expr_queue_), proj_expr_);

  // ── Late terminal: only one subquery group remains ──────────────────────
  // UpdateProjHead IS called here (unlike the early-terminal path).
  if (subqueries_.size() == 1) {
    auto &child_node = subqueries_.front()[0];
    bool merged = false;
    sp_->MergeToSubquery(plan_, child_node, merged);
    plan_ = sp_->UpdateProjHead(std::move(plan_), proj_expr_);
    terminal_ = true;
    return TakePlanAsIR(); // remaining_ir = final plan IR
  }

  // Not terminal yet; remaining_ir is irrelevant for NodeBased but must be
  // returned so the caller can pass it to the next IsComplete / Extract call.
  return remaining_ir;
}

ir_sql_converter::AQPStmt *NodeBasedSplitter::SelectSubIR(
    ir_sql_converter::AQPStmt *ir,
    const std::set<unsigned int> & /*cluster_tables*/) {
  // NodeBased selection is driven by DuckDB's subqueries_.front();
  // the full remaining IR represents the sub-IR for the current cluster.
  return ir;
}

// ─────────────────────────────────────────────────────────────────────────────
// PeekNextSubquery
// After SplitIR(i) extracts the current group, peek at subqueries_.front() to
// predict what the next iteration will extract.  The prediction is speculative:
// the real next SplitIR will MergeSubquery + MiddleOptimize + re-Split, which
// may produce different groups.
//
// The node is BORROWED, not copied (LogicalOperator::Copy crashes on
// query-split plans, e.g. CHUNK_GET nodes).  GenerateProjHead does not mutate
// the subquery subtree — it only reads split_index and AddChild()s it — so we
// pass a placeholder to GenerateProjHead, swap the real node in afterwards,
// convert to IR (read-only on the plan), and move the node back into
// subqueries_.front()[0] before returning.
// ─────────────────────────────────────────────────────────────────────────────
static bool HasNullChildren(const duckdb::LogicalOperator &op) {
  for (auto &child : op.children) {
    if (!child)
      return true;
    if (HasNullChildren(*child))
      return true;
  }
  return false;
}

namespace {
// Reversible mimic of SubqueryPreparer::MergeToSubquery for the spec chunk:
// fill the null child slot whose parent's merge_index equals the chunk's
// split_index, clearing the join's right_projection_map like the real merge
// does (so the converted IR — and thus the speculative SQL — matches the
// real next subquery). Everything is restored by RevertSpecChunk.
struct SpecChunkInjection {
  duckdb::LogicalOperator *dest = nullptr;
  size_t child_idx = 0;
  int saved_merge_index = 0;
  duckdb::vector<duckdb::idx_t> saved_right_projection_map;
  bool injected = false;
};

void InjectSpecChunk(duckdb::unique_ptr<duckdb::LogicalOperator> &dest_op,
                     duckdb::unique_ptr<duckdb::LogicalOperator> &chunk,
                     int chunk_split_index, SpecChunkInjection &res) {
  if (!dest_op)
    return;
  for (int idx = static_cast<int>(dest_op->children.size()) - 1; idx > -1;
       idx--) {
    if (res.injected)
      return;
    auto &child = dest_op->children[idx];
    if (chunk_split_index == dest_op->merge_index && chunk_split_index != 0) {
      if (!child) {
        res.dest = dest_op.get();
        res.child_idx = static_cast<size_t>(idx);
        res.saved_merge_index = dest_op->merge_index;
        dest_op->merge_index = 0;
        child = std::move(chunk);
        res.injected = true;
        if (dest_op->type ==
            duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
          auto &join_op = dest_op->Cast<duckdb::LogicalComparisonJoin>();
          res.saved_right_projection_map = join_op.right_projection_map;
          join_op.right_projection_map.clear();
        }
        return;
      }
      continue;
    }
    InjectSpecChunk(child, chunk, chunk_split_index, res);
  }
}

void RevertSpecChunk(SpecChunkInjection &res) {
  if (!res.injected)
    return;
  res.dest->children[res.child_idx] = nullptr; // destroys the spec chunk
  res.dest->merge_index = res.saved_merge_index;
  if (res.dest->type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    auto &join_op = res.dest->Cast<duckdb::LogicalComparisonJoin>();
    join_op.right_projection_map = std::move(res.saved_right_projection_map);
  }
  res.injected = false;
}
} // namespace

std::unique_ptr<ir_sql_converter::AQPStmt> NodeBasedSplitter::PeekNextSubquery(
    duckdb::idx_t spec_chunk_index,
    const duckdb::vector<duckdb::LogicalType> &chunk_types,
    duckdb::idx_t est_card) {
  if (subqueries_.empty())
    return nullptr;

  auto &slot = subqueries_.front()[0];
  if (!slot) return nullptr;

  // Inject a speculative CHUNK_GET for the temp table the current iteration
  // is about to produce, mirroring what MergeDataChunk will do for real in
  // UpdateRemainingIR. Uses the same chunk index/types the real merge will
  // use (pre-allocated by the adapter before ExecuteRow).
  SpecChunkInjection injection;
  int chunk_split_index = sp_->GetDataChunkSplitIndex();
  if (chunk_split_index != 0 && HasNullChildren(*slot)) {
    auto chunk_collection =
        duckdb::make_uniq<duckdb::ColumnDataCollection>(*ctx_, chunk_types);
    auto chunk_scan = duckdb::make_uniq<duckdb::LogicalColumnDataGet>(
        spec_chunk_index, chunk_types, std::move(chunk_collection));
    chunk_scan->estimated_cardinality = est_card == 0 ? 1 : est_card;
    chunk_scan->has_estimated_cardinality = true;
    chunk_scan->split_index = 0; // post-merge state (the real merge zeroes it)
    duckdb::unique_ptr<duckdb::LogicalOperator> chunk_op =
        std::move(chunk_scan);
    InjectSpecChunk(slot, chunk_op, chunk_split_index, injection);
  }
  if (HasNullChildren(*slot)) {
    // Either no matching merge slot in this group (the real chunk merges into
    // a later group) or more than one unresolved slot — cannot predict.
    RevertSpecChunk(injection);
    if (enable_debug_print_)
      std::cerr
          << "[NodeBased] PeekNextSubquery: skipped (unresolved null slot)\n";
    return nullptr;
  }

  // Run the index rewrites UpdateRemainingIR will perform after execution,
  // for real (not reverted). Without them the next group's BoundColumnRefs
  // still reference the just-executed tables' old indexes and IR conversion
  // fails. They are idempotent: rewritten refs point at the fresh chunk index,
  // which can never match proj_exprs' old table indexes again, so the real
  // UpdateSubqueriesIndex/UpdateTableExpr repeats in UpdateRemainingIR are
  // no-ops (and still rewrite anything MergeSibling adds later).
  sp_->SetNewTableIndex(spec_chunk_index);
  sp_->UpdateSubqueriesIndex(subqueries_);
  table_expr_queue_ =
      sp_->UpdateTableExpr(std::move(table_expr_queue_), proj_expr_);

  auto saved_proj_exprs = sp_->GetProjExprs();
  int saved_split_index = sp_->GetDataChunkSplitIndex();
  // Preserve the adapter's current plan (the sub-plan of iteration i, which
  // must stay alive while its IR is used for JIT/execution).
  auto prev_plan = plan_adapter_->TakePlan();

  duckdb::unique_ptr<duckdb::LogicalOperator> sub_plan;
  std::unique_ptr<ir_sql_converter::AQPStmt> spec_ir;
  try {
    // Placeholder keeps the real node out of GenerateProjHead's ownership so
    // an exception inside it can never destroy the borrowed node.
    auto placeholder = duckdb::make_uniq<duckdb::LogicalDummyScan>(0);
    placeholder->split_index = slot->split_index;
    sub_plan = sp_->GenerateProjHead(plan_, std::move(placeholder),
                                     table_expr_queue_, proj_expr_,
                                     merge_sibling_expr_);
    sub_plan->children[0] = std::move(slot); // borrow the real node
    sub_plan->ResolveOperatorTypes();
    plan_adapter_->SetPlan(std::move(sub_plan));
    spec_ir = plan_adapter_->ConvertPlanToIR();
    sub_plan = plan_adapter_->TakePlan();
    slot = std::move(sub_plan->children[0]); // return the borrowed node
  } catch (const std::exception &e) {
    if (!sub_plan)
      sub_plan = plan_adapter_->TakePlan();
    if (sub_plan && !sub_plan->children.empty() && sub_plan->children[0] &&
        !slot)
      slot = std::move(sub_plan->children[0]); // recover the borrowed node
    spec_ir.reset();
    if (enable_debug_print_)
      std::cerr << "[NodeBased] PeekNextSubquery failed: " << e.what() << "\n";
  }

  // Remove the speculative chunk and restore merge_index /
  // right_projection_map. The dest node lives on the heap, so the pointer is
  // valid whether the subtree currently sits in subqueries_ or in sub_plan.
  RevertSpecChunk(injection);

  sp_->SetProjExprs(std::move(saved_proj_exprs));
  sp_->SetDataChunkSplitIndex(saved_split_index);
  plan_adapter_->SetPlan(std::move(prev_plan));

  if (enable_debug_print_ && spec_ir) {
    std::cout << "[NodeBased] PeekNextSubquery IR:\n";
    spec_ir->Print();
  }
  return spec_ir;
}

} // namespace middleware

#endif // HAVE_DUCKDB
