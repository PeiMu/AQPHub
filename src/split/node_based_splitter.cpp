/*
 * NodeBasedSplitter: full implementation of the DuckDB MiddleOptimize-driven
 * split strategy.  See node_based_splitter.h for the overall loop description.
 */

#ifdef HAVE_DUCKDB

#include "split/node_based_splitter.h"

#include <iostream>

namespace middleware {

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
NodeBasedSplitter::NodeBasedSplitter(DBAdapter *exec_adapter,
                                     DuckDBAdapter *plan_adapter,
                                     bool enable_debug_print)
    : SplitAlgorithm(exec_adapter), plan_adapter_(plan_adapter),
      enable_debug_print_(enable_debug_print) {}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocess
// ─────────────────────────────────────────────────────────────────────────────
void NodeBasedSplitter::Preprocess(
    std::unique_ptr<ir_sql_converter::SimplestStmt> & /*ir*/) {
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

std::unique_ptr<ir_sql_converter::SimplestStmt>
NodeBasedSplitter::TakePlanAsIR() {
  plan_adapter_->SetPlan(std::move(plan_));
  return plan_adapter_->ConvertPlanToIR();
}

// ─────────────────────────────────────────────────────────────────────────────
// ExtractNextSubquery
// Runs BLOCK 1 + BLOCK 2 then either signals terminal or extracts a sub-plan.
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<SubqueryExtraction> NodeBasedSplitter::ExtractNextSubquery(
    ir_sql_converter::SimplestStmt * /*remaining_ir*/) {

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
    std::cout << "[NodeBased] ExtractNextSubquery: subquery groups="
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
    const ir_sql_converter::SimplestStmt * /*remaining_ir*/) {
  return terminal_;
}

// ─────────────────────────────────────────────────────────────────────────────
// UpdateRemainingIR
// Called by ExecuteOneIteration after the sub-SQL has been executed and the
// temp table created.  Inserts a CHUNK_GET node and advances split state.
// Returns the final plan IR when late-terminal, nullptr otherwise.
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<ir_sql_converter::SimplestStmt>
NodeBasedSplitter::UpdateRemainingIR(
    std::unique_ptr<ir_sql_converter::SimplestStmt> remaining_ir,
    const std::set<unsigned int> & /*executed_table_indices*/,
    unsigned int /*temp_table_index*/, const std::string & /*temp_table_name*/,
    uint64_t temp_table_cardinality,
    const std::vector<std::pair<unsigned int, unsigned int>> & /*col_mappings*/,
    const std::vector<std::string> & /*col_names*/) {

  // Tell SubqueryPreparer the DuckDB chunk index assigned by
  // ExecuteSQLandCreateTempTable (stored in temp_table_index_ by the adapter).
  sp_->SetNewTableIndex(plan_adapter_->GetTempTableIndex());

  // Build an empty ColumnDataCollection with the correct types and inject it
  // as a CHUNK_GET node so DuckDB can track cardinality for MiddleOptimize.
  auto collection =
      duckdb::make_uniq<duckdb::ColumnDataCollection>(*ctx_, sub_plan_types_);
  collection->Types() = plan_adapter_->temp_table_types;
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

} // namespace middleware

#endif // HAVE_DUCKDB
