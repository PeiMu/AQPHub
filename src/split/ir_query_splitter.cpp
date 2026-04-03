/*
 * IR-level query splitter main pipeline
 */

#include "split/ir_query_splitter.h"

#ifdef HAVE_DUCKDB
#include "adapters/duckdb_adapter.h"
#endif

namespace middleware {

IRQuerySplitter::IRQuerySplitter(EngineAdapter *adapter, const ParamConfig &config)
    : adapter_(adapter), config_(config) {

  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Initializing with strategy: "
              << config.GetStrategyName() << std::endl;
  }

  // Create the appropriate splitter based on strategy
  switch (config.strategy) {
  case SplitStrategy::TOP_DOWN:
    splitter_ =
        std::make_unique<TopDownSplitter>(adapter, config.enable_reorder_get);
    break;

  case SplitStrategy::MIN_SUBQUERY:
    splitter_ = std::make_unique<MinSubquerySplitter>(
        adapter, config.engine, config.enable_analyze, config.fkeys_path);
    break;

  case SplitStrategy::RELATIONSHIP_CENTER:
    splitter_ = std::make_unique<RelationshipCenterSplitter>(
        adapter, config.engine, config.enable_analyze, config.fkeys_path);
    break;

  case SplitStrategy::ENTITY_CENTER:
    splitter_ = std::make_unique<EntityCenterSplitter>(
        adapter, config.engine, config.enable_analyze, config.fkeys_path);
    break;

  case SplitStrategy::NODE_BASED: {
#ifdef HAVE_DUCKDB
    if (config_.engine == BackendEngine::DUCKDB) {
      duckdb_adapter_ = dynamic_cast<DuckDBAdapter *>(adapter);
    } else {
      // Create and OWN a helper DuckDB adapter for planning.
      owned_duckdb_adapter_ =
          std::make_unique<DuckDBAdapter>(config_.helper_db);
      duckdb_adapter_ = owned_duckdb_adapter_.get();
    }
    if (!duckdb_adapter_)
      throw std::runtime_error(
          "NODE_BASED strategy requires a DuckDB adapter for planning");
    splitter_ = std::make_unique<NodeBasedSplitter>(adapter, duckdb_adapter_,
                                                    config.enable_debug_print);
#else
    throw std::runtime_error("NODE_BASED strategy requires HAVE_DUCKDB");
#endif
    break;
  }

  case SplitStrategy::NONE:
  default:
    splitter_ = nullptr;
    break;
  }
}

QueryResult IRQuerySplitter::ExecuteWithSplit(const std::string &sql) {
  if (config_.enable_debug_print) {
    std::cout
        << "\n[IRQuerySplitter] ========== Starting Split Execution =========="
        << std::endl;
  }

  // Reset per-query state (but NOT subquery_index -- it must keep
  // incrementing across queries so temp table names stay unique)
  temp_tables_.clear();
  sub_plan_sqls_.clear();

  if (!config_.NeedsSplit() || !splitter_) {
    std::cout << "[IRQuerySplitter] No splitting needed, executing directly"
              << std::endl;
    return adapter_->ExecuteSQL(sql);
  }

  // === Phase 1: Parse SQL ===
  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Phase 1: Parsing SQL" << std::endl;
  }
  std::chrono::high_resolution_clock::time_point timer;
  if (config_.enable_timing)
    timer = chrono_tic();
  // For NODE_BASED: parse SQL with the DuckDB helper adapter so it builds a
  // DuckDB logical plan.  All other strategies parse on the execution adapter.
#ifdef HAVE_DUCKDB
  if (config_.strategy == SplitStrategy::NODE_BASED) {
    duckdb_adapter_->ParseSQL(sql);
  } else
#endif
  {
    adapter_->ParseSQL(sql);
  }
  if (config_.enable_timing) {
    auto parse_sql_time = chrono_toc(&timer, "Parse SQL time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3) << (parse_sql_time / 1000.0)
             << ", ";
    log_file.close();
  }

  // === Phase 2: Pre-Optimize (ONLY for DuckDB, or node-based split) ===
#ifdef HAVE_DUCKDB
  {
    DuckDBAdapter *pre_opt = nullptr;
    if (config_.strategy == SplitStrategy::NODE_BASED) {
      pre_opt = duckdb_adapter_; // always a DuckDBAdapter*
    } else if (config_.engine == BackendEngine::DUCKDB) {
      pre_opt = dynamic_cast<DuckDBAdapter *>(adapter_);
    }
    if (pre_opt) {
      if (config_.enable_debug_print)
        std::cout << "[IRQuerySplitter] Phase 2: Pre-Optimization\n";
      pre_opt->FilterOptimize();
      if (config_.enable_debug_print)
        pre_opt->PrintLogicalPlan();
    } else {
      if (config_.enable_debug_print)
        std::cout << "[IRQuerySplitter] Phase 2: Skipping Pre-Optimization\n";
    }
  }
#else
  if (config_.enable_debug_print)
    std::cout << "[IRQuerySplitter] Phase 2: Skipping Pre-Optimization\n";
#endif

  // === Phase 3: Convert to IR ===
  // NODE_BASED skips this: NodeBasedSplitter holds the DuckDB plan directly
  // via TakePlan() in Preprocess, so ConvertPlanToIR must not be called here.
  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Phase 3: Converting to IR" << std::endl;
  }
  std::unique_ptr<ir_sql_converter::AQPStmt> whole_ir;
#ifdef HAVE_DUCKDB
  if (config_.strategy != SplitStrategy::NODE_BASED) {
#endif
    if (config_.enable_timing)
      timer = chrono_tic();
    whole_ir = adapter_->ConvertPlanToIR();
    if (config_.enable_timing) {
      auto convert_plan_to_ir_time =
          chrono_toc(&timer, "Convert Plan to IR time is\n", false);
      // save time to a file
      std::ofstream log_file;
      log_file.open("time_log.csv", std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << (convert_plan_to_ir_time / 1000.0) << ", ";
      log_file.close();
    }
    if (!whole_ir) {
      throw std::runtime_error("Failed to convert plan to IR");
    }
    if (config_.enable_debug_print) {
      std::cout << "\n=== Whole IR (before split) ===" << std::endl;
      whole_ir->Print();
    }
#ifdef HAVE_DUCKDB
  }
#endif

  // === Phase 4: Iterative Split-Execute Loop ===
  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Phase 4: Iterative Split-Execute Loop"
              << std::endl;
  }
  auto result = ExecuteSplitLoop(std::move(whole_ir));

  if (config_.enable_debug_print) {
    std::cout
        << "[IRQuerySplitter] ========== Split Execution Complete =========="
        << std::endl;
    std::cout << "Total iterations: " << iteration_count_ << std::endl;
  }

  return result;
}

QueryResult IRQuerySplitter::ExecuteSplitLoop(
    std::unique_ptr<ir_sql_converter::AQPStmt> whole_ir) {

  iteration_count_ = 0;
  std::unique_ptr<ir_sql_converter::AQPStmt> remaining_ir =
      std::move(whole_ir);

  // === Strategy Preprocessing ===
  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Strategy Preprocessing" << std::endl;
  }
  std::chrono::high_resolution_clock::time_point timer;
  if (config_.enable_timing)
    timer = chrono_tic();
  splitter_->Preprocess(remaining_ir);
  if (config_.enable_timing) {
    auto preprocess_time = chrono_toc(&timer, "Preprocess time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3) << (preprocess_time / 1000.0)
             << ", ";
    log_file.close();
  }

  // Main loop: while (graph has edges) { extract → execute → merge }
  while (!splitter_->IsComplete(remaining_ir.get())) {
    iteration_count_++;

    if (config_.enable_debug_print) {
      std::cout << "\n========== Iteration " << iteration_count_
                << " ==========" << std::endl;
    }

    if (!ExecuteOneIteration(remaining_ir)) {
      std::cerr << "[IRQuerySplitter] Warning: ExecuteOneIteration returned "
                   "false but IsComplete was false. Breaking loop."
                << std::endl;
      break;
    }

    if (config_.enable_debug_print) {
      std::cout << "Iteration " << iteration_count_ << " completed\n";
    }
  }

  if (config_.enable_debug_print) {
    std::cout << "[IRQuerySplitter] Split loop completed after "
              << iteration_count_ << " iteration(s)" << std::endl;
  }

  // === Final Execution ===
  if (!remaining_ir) {
    throw std::runtime_error("Remaining IR is null after split loop");
  }

  if (config_.enable_debug_print) {
    std::cout << "\n=== Final Remaining IR ===" << std::endl;
    remaining_ir->Print();
  }

  if (config_.enable_timing)
    timer = chrono_tic();
  // Determine final SQL (trivial or non-trivial)
  std::string final_sql;
  std::string trivial_temp = GetTrivialTempTable(remaining_ir.get());
  if (!trivial_temp.empty()) {
    if (config_.enable_debug_print) {
      std::cout << "\n[IRQuerySplitter] Final IR is trivial (temp table: "
                << trivial_temp << "), returning directly" << std::endl;
    }
    final_sql = "SELECT * FROM " + trivial_temp;
  } else {
    // Non-trivial case: generate final SQL
    if (config_.enable_debug_print) {
      std::cout << "\n[IRQuerySplitter] Executing final remaining IR"
                << std::endl;
    }
    final_sql =
        adapter_->GenerateSQL(*remaining_ir, adapter_->subquery_index++);
    if (config_.print_sql || config_.enable_debug_print) {
      std::cout << "\n=== Final Generated Sub-SQL ===" << std::endl;
      std::cout << final_sql << std::endl;
    }
  }
  if (config_.enable_timing) {
    auto generate_final_sub_sql_time =
        chrono_toc(&timer, "Generate final sub-SQL time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (generate_final_sub_sql_time / 1000.0) << ", ";
    log_file.close();
  }

  // Print combined sub-plan SQL if enabled (print only; result comes from
  // final_sql executed via the normal temp-table chain below)
  QueryResult query_result;
  if (config_.enable_sub_plan_combiner && !sub_plan_sqls_.empty()) {
    std::string combined = BuildCombinedSQL(sub_plan_sqls_, final_sql);
    if (config_.enable_timing) {
      auto combine_sql_time =
          chrono_toc(&timer, "Combine SQL time is\n", false);
      // save time to a file
      std::ofstream log_file;
      log_file.open("time_log.csv", std::ios_base::app);
      log_file << std::fixed << std::setprecision(3)
               << (combine_sql_time / 1000.0) << ", ";
      log_file.close();
    }
    if (config_.print_sql || config_.enable_debug_print) {
      std::cout << "\n=== Combined Sub-Plan SQL ===" << std::endl;
      std::cout << combined << std::endl;
    }
    // Drop temp tables created by the split loop so the combined SQL can
    // CREATE them fresh (avoiding "already exists" errors)
    for (const auto &plan : sub_plan_sqls_) {
      adapter_->DropTempTable(plan.first);
    }
    query_result = adapter_->ExecuteSQL(combined);
  } else {
    std::cerr << "[AQP-JIT-TRACE] final SQL path: enable_jit=" << config_.enable_jit
              << " engine=" << (int)config_.engine << "\n";
#ifdef HAVE_DUCKDB
#ifdef HAVE_LLVM
    std::cerr << "[AQP-JIT-TRACE] HAVE_DUCKDB+HAVE_LLVM compiled in\n";
    if (config_.enable_jit && config_.engine == BackendEngine::DUCKDB) {
      auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
      std::cerr << "[AQP-JIT-TRACE] duck=" << (void*)duck
                << " remaining_ir=" << (void*)remaining_ir.get() << "\n";
      if (duck && remaining_ir)
        duck->SetJITPendingIR(remaining_ir.get());
    }
#else
    std::cerr << "[AQP-JIT-TRACE] HAVE_LLVM NOT defined\n";
#endif
#else
    std::cerr << "[AQP-JIT-TRACE] HAVE_DUCKDB NOT defined\n";
#endif
    query_result = adapter_->ExecuteSQL(final_sql);
  }
  if (config_.enable_timing) {
    auto execute_final_sql_time =
        chrono_toc(&timer, "Execute final SQL time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (execute_final_sql_time / 1000.0) << ", ";
    log_file.close();
  }
  return query_result;
}

bool IRQuerySplitter::ExecuteOneIteration(
    std::unique_ptr<ir_sql_converter::AQPStmt> &remaining_ir) {

  // === Step 1: Extract Next Subquery ===
  if (config_.enable_debug_print) {
    std::cout << "[Iteration " << iteration_count_
              << "] Step 1: Extracting next subquery" << std::endl;
  }

  std::chrono::high_resolution_clock::time_point timer;
  if (config_.enable_timing)
    timer = chrono_tic();
  // todo: potential optimization - Push Partial Aggregation into Sub-IR
  auto extraction = splitter_->SplitIR(remaining_ir.get());
  if (config_.enable_timing) {
    auto extract_next_sub_sql_time =
        chrono_toc(&timer, "Extract next sub-SQL time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (extract_next_sub_sql_time / 1000.0) << ", ";
    log_file.close();
  }

  if (!extraction) {
    if (config_.enable_debug_print) {
      std::cout << "[Iteration " << iteration_count_
                << "] No more subqueries to extract" << std::endl;
    }
    return false;
  }

  // Terminal extraction: sub_ir holds the final plan IR; no SQL execution.
  // Set remaining_ir so ExecuteSplitLoop can run the final query normally.
  if (extraction->is_final) {
    remaining_ir = std::move(extraction->sub_ir);
    return true;
  }

  if (config_.enable_debug_print) {
    std::cout << "[Iteration " << iteration_count_
              << "] Extracted subquery with "
              << extraction->executed_table_indices.size() << " table(s)"
              << std::endl;
  }

  // === Step 2: Execute Sub-IR ===
  if (config_.enable_debug_print) {
    std::cout << "[Iteration " << iteration_count_
              << "] Step 2: Executing sub-IR" << std::endl;
  }

  ir_sql_converter::AQPStmt *executable_ir = extraction->GetExecutableIR();

  if (!executable_ir) {
    std::cerr << "[Iteration " << iteration_count_
              << "] Error: No executable IR" << std::endl;
    return false;
  }

  if (config_.enable_debug_print) {
    std::cout << "\n=== Sub-IR to Execute ===" << std::endl;
    executable_ir->Print();
  }

  // Generate SQL and execute
  std::string sub_sql =
      adapter_->GenerateSQL(*executable_ir, adapter_->subquery_index++);
  std::string temp_table_name = GenerateTempTableName();

  if (config_.enable_timing) {
    auto generate_sub_sql_time =
        chrono_toc(&timer, "Generate sub-SQL time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (generate_sub_sql_time / 1000.0) << ", ";
    log_file.close();
  }

  if (config_.enable_sub_plan_combiner) {
    sub_plan_sqls_.emplace_back(temp_table_name, sub_sql);
  }

  if (config_.print_sql || config_.enable_debug_print) {
    std::cout << "\n=== Sub-Query SQL ===" << std::endl;
    std::cout << sub_sql << std::endl;
  }

  if (config_.enable_debug_print) {
    std::cout << "Executing sub-query and creating temp table: "
              << temp_table_name << std::endl;
  }

#ifdef HAVE_DUCKDB
#ifdef HAVE_LLVM
  // JIT: compile filter expressions before execution so DuckDB can dispatch
  // to compiled code instead of the interpreted expression executor.
  if (config_.enable_jit && config_.engine == BackendEngine::DUCKDB) {
    auto *duck = dynamic_cast<DuckDBAdapter *>(adapter_);
    if (duck)
      duck->SetJITPendingIR(executable_ir);
  }
#endif
#endif

  adapter_->ExecuteSQLandCreateTempTable(sub_sql, temp_table_name,
                                         config_.enable_update_temp_card,
                                         config_.enable_timing);

  if (config_.enable_timing)
    timer = chrono_tic();

  // Generate temp table index
  unsigned int temp_table_index =
      splitter_->GetMaxTableIndex() + iteration_count_;
  uint64_t cardinality;
  if (config_.enable_update_temp_card || extraction->estimated_rows <= 0) {
    cardinality = adapter_->GetTempTableCardinality(temp_table_name);
  } else {
    // Use optimizer's estimated rows (from EXPLAIN) to simulate inaccurate
    // cardinality for A/B testing
    // fixme: might be a bug where the node-based/top-down split didn't call
    //  BatchGetEstimatedCosts maybe need to collect the card before
    //  SplitIR by calling BatchGetEstimatedCosts
    cardinality = static_cast<uint64_t>(extraction->estimated_rows);
    // Override the engine's internal stats so subsequent EXPLAIN queries
    // also use the estimated cardinality instead of the real one
    adapter_->SetTempTableCardinality(temp_table_name, cardinality);
  }

  TempTableInfo temp_table =
      TempTableInfo(temp_table_name, temp_table_index, cardinality);

  // Store column mappings with correct names (SQL generator uses:
  // {table}_{col})
  std::vector<std::pair<unsigned int, unsigned int>> col_mappings;
  std::vector<std::string> col_names;
  for (const auto &attr : executable_ir->target_list) {
    std::string col_alias =
        ComputeColumnAlias(attr->GetTableIndex(), attr->GetColumnName());
    temp_table.column_names.push_back(col_alias);
    temp_table.column_mappings.emplace_back(attr->GetTableIndex(),
                                            attr->GetColumnIndex(), col_alias);
    col_mappings.emplace_back(attr->GetTableIndex(), attr->GetColumnIndex());
    col_names.push_back(col_alias);
  }

  // Add temp table to the mapping for future iterations
  splitter_->AddTableMapping(temp_table_index, temp_table_name);

  if (config_.enable_debug_print) {
    std::cout << "[Iteration " << iteration_count_
              << "] Created temp table: " << temp_table.table_name
              << " (index=" << temp_table.table_index
              << ", cardinality=" << temp_table.cardinality << ")" << std::endl;
  }

  // === Step 3: Update Remaining IR ===
  if (config_.enable_debug_print) {
    std::cout << "[Iteration " << iteration_count_
              << "] Step 3: Updating remaining IR" << std::endl;
  }

  // Call strategy-specific UpdateRemainingIR (takes ownership of old IR)
  remaining_ir = splitter_->UpdateRemainingIR(
      std::move(remaining_ir), extraction->executed_table_indices,
      temp_table.table_index, temp_table.table_name, temp_table.cardinality,
      col_mappings, col_names);

  if (config_.enable_debug_print) {
    if (remaining_ir) {
      std::cout << "[Iteration " << iteration_count_
                << "] Successfully updated remaining IR" << std::endl;
    } else if (!splitter_->SkipUpdateIndices()) {
      std::cerr << "[Iteration " << iteration_count_
                << "] Warning: Failed to update remaining IR" << std::endl;
    }
  }

  // === Step 4: Update Indices (shared) ===
  // Skipped for NODE_BASED: DuckDB's UpdateSubqueriesIndex / UpdateTableExpr
  // keep all bindings consistent internally.
  if (!splitter_->SkipUpdateIndices()) {
    if (config_.enable_debug_print) {
      std::cout << "[Iteration " << iteration_count_
                << "] Step 4: Updating indices in remaining IR" << std::endl;
    }
    UpdateRemainingIRIndices(remaining_ir.get(), temp_table,
                             extraction->executed_table_indices);
    if (config_.enable_debug_print) {
      std::cout << "\n=== Updated Remaining IR ===" << std::endl;
      remaining_ir->Print();
    }
  }

  temp_tables_.push_back(temp_table);

  if (config_.enable_timing) {
    auto update_ir_time = chrono_toc(&timer, "Update IR time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open("time_log.csv", std::ios_base::app);
    log_file << std::fixed << std::setprecision(3) << (update_ir_time / 1000.0)
             << ", ";
    log_file.close();
  }

  return true;
}

TempTableInfo IRQuerySplitter::ExecuteSubIR(
    std::unique_ptr<ir_sql_converter::AQPStmt> sub_ir,
    const std::set<unsigned int> &executed_table_indices) {

  std::string temp_table_name = GenerateTempTableName();
  std::string sub_sql =
      adapter_->GenerateSQL(*sub_ir, adapter_->subquery_index++);

  adapter_->ExecuteSQLandCreateTempTable(sub_sql, temp_table_name,
                                         config_.enable_update_temp_card,
                                         config_.enable_timing);

  unsigned int temp_table_index = adapter_->subquery_index - 1;
  // TODO: support estimated_rows for enable_update_temp_card=false path
  uint64_t cardinality = adapter_->GetTempTableCardinality(temp_table_name);

  return TempTableInfo(temp_table_name, temp_table_index, cardinality);
}

std::string IRQuerySplitter::GenerateTempTableName() {
  return "temp" + std::to_string(adapter_->subquery_index);
}

std::string
IRQuerySplitter::GetTrivialTempTable(ir_sql_converter::AQPStmt *ir) const {
  if (!ir) {
    return "";
  }

  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ChunkNode) {
    auto *chunk = dynamic_cast<ir_sql_converter::SimplestChunk *>(ir);
    if (chunk && !chunk->GetContents().empty()) {
      return chunk->GetContents()[0];
    }
  }

  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ProjectionNode) {
    if (ir->children.size() == 1 && ir->children[0]) {
      auto *child = ir->children[0].get();
      if (child->GetNodeType() ==
          ir_sql_converter::SimplestNodeType::ChunkNode) {
        auto *chunk = dynamic_cast<ir_sql_converter::SimplestChunk *>(child);
        if (chunk && !chunk->GetContents().empty()) {
          return chunk->GetContents()[0];
        }
      }
    }
  }

  if (ir->children.size() == 1 && ir->children[0]) {
    return GetTrivialTempTable(ir->children[0].get());
  }

  return "";
}

// ===== Shared Index Update Functions =====

void IRQuerySplitter::UpdateExprIndices(
    ir_sql_converter::AQPExpr *expr, const TempTableInfo &temp_table,
    const std::set<unsigned int> &old_table_indices) {

  if (!expr) {
    return;
  }

  auto node_type = expr->GetNodeType();

  if (node_type == ir_sql_converter::SimplestNodeType::VarConstComparisonNode) {
    auto *var_const =
        dynamic_cast<ir_sql_converter::SimplestVarConstComparison *>(expr);
    if (var_const && var_const->attr) {
      auto updated = UpdateAttrIndices(var_const->attr.get(), temp_table,
                                       old_table_indices);
      if (updated) {
        var_const->attr = std::move(updated);
      }
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarComparisonNode) {
    auto *var_cmp =
        dynamic_cast<ir_sql_converter::SimplestVarComparison *>(expr);
    if (var_cmp) {
      if (var_cmp->left_attr) {
        auto updated = UpdateAttrIndices(var_cmp->left_attr.get(), temp_table,
                                         old_table_indices);
        if (updated) {
          var_cmp->left_attr = std::move(updated);
        }
      }
      if (var_cmp->right_attr) {
        auto updated = UpdateAttrIndices(var_cmp->right_attr.get(), temp_table,
                                         old_table_indices);
        if (updated) {
          var_cmp->right_attr = std::move(updated);
        }
      }
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::IsNullExprNode) {
    auto *is_null = dynamic_cast<ir_sql_converter::SimplestIsNullExpr *>(expr);
    if (is_null && is_null->attr) {
      auto updated =
          UpdateAttrIndices(is_null->attr.get(), temp_table, old_table_indices);
      if (updated) {
        is_null->attr = std::move(updated);
      }
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarParamComparisonNode) {
    auto *var_param =
        dynamic_cast<ir_sql_converter::SimplestVarParamComparison *>(expr);
    if (var_param && var_param->attr) {
      auto updated = UpdateAttrIndices(var_param->attr.get(), temp_table,
                                       old_table_indices);
      if (updated) {
        var_param->attr = std::move(updated);
      }
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *logical = dynamic_cast<ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (logical) {
      UpdateExprIndices(logical->left_expr.get(), temp_table,
                        old_table_indices);
      UpdateExprIndices(logical->right_expr.get(), temp_table,
                        old_table_indices);
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::SingleAttrExprNode) {
    auto *single_attr =
        dynamic_cast<ir_sql_converter::SimplestSingleAttrExpr *>(expr);
    if (single_attr && single_attr->attr) {
      auto updated = UpdateAttrIndices(single_attr->attr.get(), temp_table,
                                       old_table_indices);
      if (updated) {
        single_attr->attr = std::move(updated);
      }
    }
    return;
  }
}

std::unique_ptr<ir_sql_converter::SimplestAttr>
IRQuerySplitter::UpdateAttrIndices(
    const ir_sql_converter::SimplestAttr *attr, const TempTableInfo &temp_table,
    const std::set<unsigned int> &old_table_indices) {

  if (!attr) {
    return nullptr;
  }

  unsigned int old_table_idx = attr->GetTableIndex();
  unsigned int old_col_idx = attr->GetColumnIndex();

  if (old_table_indices.find(old_table_idx) == old_table_indices.end()) {
    return nullptr;
  }

  int new_col_idx = temp_table.FindNewColumnIndex(old_table_idx, old_col_idx);

  if (new_col_idx < 0) {
    std::cerr << "[UpdateAttrIndices] Warning: Column [" << old_table_idx << "."
              << old_col_idx << "] (" << attr->GetColumnName()
              << ") not found in temp table mapping" << std::endl;
    return nullptr;
  }

  // Use the column name from column_mappings which matches SQL generator's
  // convention Format: {chunk_name}_{column_name}
  std::string new_col_name =
      temp_table.column_mappings[new_col_idx].column_name;

  auto new_attr = std::make_unique<ir_sql_converter::SimplestAttr>(
      attr->GetType(), temp_table.table_index,
      static_cast<unsigned int>(new_col_idx), new_col_name);

  if (config_.enable_debug_print) {
    std::cout << "[UpdateAttrIndices] Updated [" << old_table_idx << "."
              << old_col_idx << "] (" << attr->GetColumnName() << ") -> ["
              << temp_table.table_index << "." << new_col_idx << "] ("
              << new_col_name << ")" << std::endl;
  }

  return new_attr;
}

void IRQuerySplitter::UpdateNodeIndices(
    ir_sql_converter::AQPStmt *node, const TempTableInfo &temp_table,
    const std::set<unsigned int> &old_table_indices) {

  if (!node) {
    return;
  }

  // Update target_list
  for (size_t i = 0; i < node->target_list.size(); i++) {
    auto updated = UpdateAttrIndices(node->target_list[i].get(), temp_table,
                                     old_table_indices);
    if (updated) {
      node->target_list[i] = std::move(updated);
    }
  }

  // Update qual_vec
  for (auto &qual : node->qual_vec) {
    if (qual) {
      UpdateExprIndices(qual.get(), temp_table, old_table_indices);
    }
  }

  // Update join conditions
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *join = dynamic_cast<ir_sql_converter::SimplestJoin *>(node);
    if (join) {
      for (auto &cond : join->join_conditions) {
        if (cond->left_attr) {
          auto updated = UpdateAttrIndices(cond->left_attr.get(), temp_table,
                                           old_table_indices);
          if (updated) {
            cond->left_attr = std::move(updated);
          }
        }
        if (cond->right_attr) {
          auto updated = UpdateAttrIndices(cond->right_attr.get(), temp_table,
                                           old_table_indices);
          if (updated) {
            cond->right_attr = std::move(updated);
          }
        }
      }
    }
  }

  // Update hash_keys
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::HashNode) {
    auto *hash = dynamic_cast<ir_sql_converter::SimplestHash *>(node);
    if (hash) {
      for (size_t i = 0; i < hash->hash_keys.size(); i++) {
        auto updated = UpdateAttrIndices(hash->hash_keys[i].get(), temp_table,
                                         old_table_indices);
        if (updated) {
          hash->hash_keys[i] = std::move(updated);
        }
      }
    }
  }

  // Update aggregate groups and functions
  if (node->GetNodeType() ==
      ir_sql_converter::SimplestNodeType::AggregateNode) {
    auto *agg = dynamic_cast<ir_sql_converter::SimplestAggregate *>(node);
    if (agg) {
      for (size_t i = 0; i < agg->groups.size(); i++) {
        auto updated = UpdateAttrIndices(agg->groups[i].get(), temp_table,
                                         old_table_indices);
        if (updated) {
          agg->groups[i] = std::move(updated);
        }
      }
      for (auto &fn_pair : agg->agg_fns) {
        auto updated = UpdateAttrIndices(fn_pair.first.get(), temp_table,
                                         old_table_indices);
        if (updated) {
          fn_pair.first = std::move(updated);
        }
      }
    }
  }

  // Update order by
  if (node->GetNodeType() == ir_sql_converter::SimplestNodeType::OrderNode) {
    auto *order = dynamic_cast<ir_sql_converter::SimplestOrderBy *>(node);
    if (order) {
      for (auto &ord : order->orders) {
        auto updated =
            UpdateAttrIndices(ord.attr.get(), temp_table, old_table_indices);
        if (updated) {
          ord.attr = std::move(updated);
        }
      }
    }
  }

  // Recursively update children
  for (auto &child : node->children) {
    UpdateNodeIndices(child.get(), temp_table, old_table_indices);
  }
}

void IRQuerySplitter::UpdateRemainingIRIndices(
    ir_sql_converter::AQPStmt *remaining_ir,
    const TempTableInfo &temp_table,
    const std::set<unsigned int> &old_table_indices) {

  if (!remaining_ir) {
    return;
  }

  if (config_.enable_debug_print) {
    std::cout
        << "[UpdateRemainingIRIndices] Updating indices for executed tables: ";
    for (unsigned int idx : old_table_indices) {
      std::cout << idx << " ";
    }
    std::cout << "-> temp table index " << temp_table.table_index << std::endl;
  }

  UpdateNodeIndices(remaining_ir, temp_table, old_table_indices);
}

std::string
IRQuerySplitter::ComputeColumnAlias(unsigned int table_idx,
                                    const std::string &col_name) const {
  // SQL generator convention: {chunk_name}_{table_index}_{column_name}
  // Index included to disambiguate when same table appears multiple times
  std::string table_name = splitter_->GetTableName(table_idx);
  if (!table_name.empty()) {
    return table_name + "_" + std::to_string(table_idx) + "_" + col_name;
  }
  // Fallback: use table index if name not found
  return std::to_string(table_idx) + "_" + col_name;
}

// Strip trailing whitespace and semicolons from a SQL string
static std::string StripTrailingSemicolon(const std::string &sql) {
  size_t end = sql.size();
  while (end > 0 &&
         (sql[end - 1] == ';' || sql[end - 1] == ' ' || sql[end - 1] == '\n' ||
          sql[end - 1] == '\r' || sql[end - 1] == '\t')) {
    --end;
  }
  return sql.substr(0, end);
}

std::string IRQuerySplitter::BuildCombinedSQL(
    const std::vector<std::pair<std::string, std::string>> &sub_plans,
    const std::string &final_sql) const {
  std::string result;
  for (const auto &plan : sub_plans) {
    result += "CREATE TEMP TABLE " + plan.first + " AS\n";
    result += StripTrailingSemicolon(plan.second) + ";\n\n";
  }
  result += StripTrailingSemicolon(final_sql) + ";";
  return result;
}

} // namespace middleware
