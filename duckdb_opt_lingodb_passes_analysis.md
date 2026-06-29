Referenced by: `duckdb_opt_lingodb_runtime.log` (PartialQueryOptimizer section)

Pass-by-pass analysis of all 26 invocations in createQueryOptPipeline. Each is categorized as SKIP (DuckDB covers it or it's irrelevant to our IR), NEED (must include), or DEPENDS (conditionally useful).

Full Pipeline Analysis

The pipeline from Passes.cpp:17-49:

Pass 1: SimplifyAggregationsPass

What it does: Wraps standalone AggrFuncOp/CountRowsOp (not inside AggregationOp) into proper AggregationOp, and rewrites AVG → SUM/COUNT, STDDEV → VAR_SAMP, VAR_SAMP → (SUM(x²) - SUM(x)²/N) / (N-1).

Verdict: SKIP. Our IRToRelAlgConverter never produces standalone AggrFuncOp — it always creates properly wrapped AggregationOp nodes. These complex aggregation rewrites only apply to forms produced by LingoDB's SQL frontend. Verified: lingodb_runtime_adapter.cpp creates relalg::AggregationOp directly with AggrFuncOp inside its region.

Pass 2: ExtractNestedOperatorsPass

What it does: Extracts nested relational operators from inside lambda regions (e.g., a SortOp nested inside a SelectionOp's predicate block) to the top level.

Verdict: SKIP. Our converter never nests relational operators inside lambda regions. Each operator is created at the top level sequentially. Verified: convertNode() builds the tree recursively but each op is created at the builder's insertion point, not inside another op's region.

Passes 3-4: CSEPass + CanonicalizerPass

What they do: Standard MLIR passes — common subexpression elimination and canonicalization (constant folding, dead code elimination).

Verdict: INCLUDE. These are lightweight, always-beneficial MLIR infrastructure passes. They clean up redundant ops and simplify the IR. No DuckDB equivalent at the MLIR level.

Uncertainty: I assume CSE and Canonicalize run correctly on our IR without the prior passes. They are generic MLIR passes with no dependencies on specific RelAlg structure. Confidence: high — these passes operate on op-level patterns and are routinely used standalone.

Pass 5: InferNotNullConditionsPass (first invocation)

What it does: For every CmpOp, BetweenOp, OneOfOp inside a SelectionOp predicate, if a column is nullable and is used in a comparison (which would produce NULL=false for NULL values), it adds an explicit IS NOT NULL check to the predicate. This makes the non-null constraint visible to later passes like EliminateNullableTypesPass.

Verdict: INCLUDE. This prepares the IR for EliminateNullableTypesPass. Without it, EliminateNullableTypesPass won't know which columns are provably non-null after filters. DuckDB handles nullability differently (validity vectors) and this information is lost during IR conversion.

Uncertainty: Whether our converter already produces enough IS NOT NULL hints. Verified: our converter does NOT add explicit IS NOT NULL predicates — it generates CmpOp directly without null guards. So this pass is needed.

Passes 6-8: DecomposeLambdasPass(true) + CanonicalizerPass + ImplicitToExplicitJoinsPass

What they do:
- DecomposeLambdas: Decomposes InnerJoinOp → CrossProductOp + SelectionOp, then splits multi-predicate SelectionOps into separate single-predicate SelectionOps.
- ImplicitToExplicitJoins: Converts GetScalarOp → SingleJoinOp, ExistsOp → SemiJoinOp, InOp → MarkJoinOp. Also converts CrossProductOp + SelectionOp → InnerJoinOp (reverses the decomposition, but now with cleaner predicate structure).

Verdict: SKIP. These are SQL-frontend cleanup passes. LingoDB's SQL parser produces GetScalarOp/ExistsOp/InOp which need conversion to explicit join ops. Our converter directly creates InnerJoinOp, SemiJoinOp, MarkJoinOp — these forms are already "explicit joins."
Running DecomposeLambdas would actually be counterproductive since it would decompose our already-correct InnerJoinOp into CrossProductOp + SelectionOp.

Uncertainty: Could DecomposeLambdas help split multi-predicate join conditions? Verified: our converter creates one predicate per join condition in the conditions vector, then combines them with AndOp. DecomposeLambdas would split SelectionOp predicates but not join predicates directly. Since we already have the predicates properly structured in the join's predicate block, this is not needed.

Pass 9: InferNotNullConditionsPass (second invocation)

Verdict: Skip this second invocation. The first invocation (pass 5) already handles the same logic. The second invocation exists to catch new SelectionOps created by ImplicitToExplicitJoins which we skip.

Passes 10-11: DecomposeLambdasPass() + PushdownPass (first invocation)

What they do:
- DecomposeLambdas(false): Same as above but without deriveExtraConditions (no OR-to-AND derivation).
- Pushdown: Pushes SelectionOps down through UnaryOperator, BinaryOperator (into left/right children of joins), and into BaseTableOp restrictions. Also converts OuterJoin + IS NULL filter → AntiSemiJoin and OuterJoin + IS NOT NULL filter → InnerJoin.

Verdict: SKIP. DuckDB's optimizer has already done predicate pushdown. Our IR reflects DuckDB's optimized plan where filters are already pushed to the lowest possible position. Running LingoDB's pushdown on top of DuckDB's optimized plan would be redundant.

Uncertainty: LingoDB's Pushdown also pushes predicates into BaseTableOp.restrictions (enabling scan-time filtering). Our converter creates BaseTableOp without restrictions and puts filters as separate SelectionOp nodes. This scan-time filtering could potentially help. However, the impact is likely small because the filter still runs; it's just attached at a different point. The compiled code evaluates the same predicate either way.

Flagging: If you want to verify this assumption, you could test whether scan-time restriction (via BaseTableOp.filterDescription) is faster than a separate SelectionOp for LingoDB's compiled code. I expect the difference is negligible because LingoDB's lowering fuses filter+scan anyway.

Pass 12: UnnestingPass

What it does: Unnests correlated subqueries (converts dependent joins to independent joins with aggregation).

Verdict: SKIP. Our converter never produces correlated subqueries. DuckDB already unnests subqueries in its optimizer. The AQPStmt IR has no mechanism for correlated subqueries.

Pass 13: ColumnFoldingPass

What it does: Tracks which columns are used downstream and eliminates unused columns from operators. Calls foldColumns() and eliminateDeadColumns() on each ColumnFoldable operator.

Verdict: INCLUDE. DuckDB's column pruning happens at the logical plan level, but when we convert to MLIR, our IRToRelAlgConverter creates BaseTableOp with ALL columns from the catalog defined. Intermediate joins carry all available columns. Only the final MaterializeOp selects the output columns. Without this pass, compiled code materializes unnecessary columns at every operator.

Verified: lingodb_runtime_adapter.cpp:276-290 creates BaseTableOp with the full column dictionary from the catalog. Columns not needed by the query are still defined and propagated.

Pass 14: DecomposeLambdasPass() (third invocation)

Verdict: SKIP. Same reasoning as passes 6/10.

Pass 15: PushdownPass (second invocation)

Verdict: SKIP. Same reasoning as pass 11. A second invocation catches opportunities revealed by ColumnFolding, but since we skip ColumnFolding's interaction with Decompose/Pushdown cycle, this is not needed.

Wait — correction: I said include ColumnFolding above. If we include ColumnFolding, should we also include the second Pushdown? No. ColumnFolding removes unused columns but doesn't create new pushdown opportunities. Pushdown moves SelectionOps down, which is already done by DuckDB. The pipeline ordering (ColumnFolding → Pushdown) exists because the original SQL-frontend pipeline needs multiple rounds. For our pre-optimized IR, ColumnFolding alone is sufficient.

Pass 16: AttachMetaDataPass (conditional on catalog)

What it does: Attaches TableMetaDataAttr (table statistics, sample data, indices) from the catalog to each BaseTableOp.

Verdict: INCLUDE. OptimizeImplementationsPass uses this metadata for:
- Selectivity estimation (filter reordering via sample data) — lines 431-446
- Index nested-loop join decision (index names) — lines 186, 498-541
- Cardinality estimation (build/probe side selection) — lines 602-643

Without metadata, OptimizeImplementationsPass falls back to no-op for these features, but still handles hash join extraction, TopK fusion, and GroupJoin.

Uncertainty: Does AttachMetaData work correctly with temp tables
created by our adapter? Temp tables are created via CreateTempTableFromArrow → catalog->insertEntry(). They should be in the catalog. However, they may not have sample data or indices. If they don't have sample data, filter reordering and index-NLJ won't fire for temp tables, but hash join extraction and GroupJoin still work. This is acceptable — temp tables are small intermediate results.

To verify: Run with a debug print to check whether meta.getMeta()->getSample() returns valid data for temp tables. I expect it returns null/empty for temp tables (no sample), meaning filter reordering only works for base tables.

Pass 17: ReduceGroupByKeysPass

What it does: Uses functional dependencies to reduce GROUP BY keys. If a set of columns functionally determines others, removes the redundant keys from the aggregation.

Verdict: DEPENDS. DuckDB's optimizer may or may not reduce GROUP BY keys. JOB queries have very few GROUP BY clauses (mostly MIN aggregates without GROUP BY). Impact is likely negligible for JOB.

Recommendation: INCLUDE — it's cheap and correct. It requires getFDs() which works on our IR (the Operator interface computes FDs from the operator structure). Worst case it's a no-op.

Pass 18: ExpandTransitiveEqualities

What it does: If A = B and B = C are separate equality predicates, derives A = C and adds it as a new SelectionOp. This creates additional filter opportunities.

Verdict: SKIP. DuckDB's optimizer already propagates transitive equalities. Adding redundant equality predicates on top of DuckDB's already-optimized plan won't help and could slightly hurt (extra filter evaluation).

Uncertainty: Could there be transitive equalities that DuckDB discovered but didn't propagate? Unlikely. DuckDB's TransitiveFilter rule in its optimizer handles this. But if you want to verify, check the DuckDB optimizer output for a multi-join query to confirm transitive predicates are present.

Pass 19: OptimizeJoinOrderPass

What it does: Cost-based join reordering using cardinality estimates.

Verdict: SKIP. DuckDB's optimizer already decided the join order. This is the core pass we want to avoid — we want DuckDB's join ordering, not LingoDB's.

Pass 20: CombinePredicatesPass

What it does: Merges SelectionOp(InnerJoinOp) → single InnerJoinOp with combined predicate.

Verdict: SKIP. Our converter already puts predicates directly in the InnerJoinOp's predicate block. There are no dangling SelectionOps on top of InnerJoinOps. Verified: convertJoin() creates InnerJoinOp with both join_conditions and qual_vec inside the predicate region.

Exception: If InferNotNullConditionsPass adds a SelectionOp with IS 
NOT NULL on top of a SelectionOp that's on top of an InnerJoinOp, CombinePredicates wouldn't help anyway (it only combines SelectionOp directly on InnerJoinOp, not chained selections). So even in that case, skipping is correct.

Pass 21: EliminateNullableTypesPass

What it does: For columns with NOT NULL constraints (from BaseTableOp.restrictions) or columns proven non-null by IS NOT NULL filters, replaces NullableType<T> with bare T. Creates new non-nullable column definitions and inserts NullableGetVal/AsNullableOp only where the boundary occurs.

Verdict: INCLUDE. Every nullable column access in compiled code generates: (1) a null check branch, (2) AsNullableOp unwrap/wrap. For JOB queries with many integer key comparisons, this overhead is significant. DuckDB uses validity vectors (a completely different mechanism) and this information is lost during IR conversion.

Dependency: Works best after InferNotNullConditionsPass (pass 5) which adds explicit IS NOT NULL predicates for columns used in comparisons. Without pass 5, this pass only eliminates nullable types for columns with NOT NULL in the schema — still useful but less effective.

Pass 22: OptimizeImplementationsPass

What it does (verified from full source read):

Sub-feature: Hash join extraction (useHashJoin + leftHash/rightHash) 
Lines: 492-596
Effect: Sets up hash join key extraction          
Needed for our IR?: NO — our converter already sets these attrs
────────────────────────────────────────
Sub-feature: Index nested-loop join selection 
Lines: 497-591
Effect: Chooses INLJ when index exists + favorable cardinality ratio Needed for our IR?: YES if metadata attached — requires meta attr
────────────────────────────────────────
Sub-feature: Filter reordering by selectivity 
Lines: 406-481
Effect: Estimates selectivity via table samples, reorders chained SelectionOps
Needed for our IR?: YES if metadata attached — requires sample data
────────────────────────────────────────
Sub-feature: Limit+Sort → TopK fusion
Lines: 483-491
Effect: Fuses LimitOp(SortOp) → TopKOp
Needed for our IR?: YES — our converter creates separate LimitOp and SortOp
────────────────────────────────────────
Sub-feature: SemiJoin/MarkJoin reverseSides
Lines: 598-643
Effect: Flips build/probe sides for semi/mark/outer joins based on cardinality
Needed for our IR?: YES if rows attr exists — requires cardinality estimates
────────────────────────────────────────
Sub-feature: GroupJoin creation
Lines: 771-849
Effect: Fuses AggregationOp(InnerJoinOp) → GroupJoinOp
Needed for our IR?: YES — major optimization when aggregate sits on a join
────────────────────────────────────────
Sub-feature: Join→Aggregation pushdown (keys match)
Lines: 671-770
Effect: Pushes aggregation below join when build side is duplicate-free on join keys
Needed for our IR?: YES — reduces intermediate result size

Verdict: INCLUDE. Even though our converter already handles hash join attrs, the other 5 sub-features (TopK, filter reordering, GroupJoin, join→aggr pushdown, INLJ) are not covered by our converter or DuckDB.

Key finding: Hash join extraction is redundant (our converter already does it), but the pass handles it gracefully — it calls prepareForHash() which re-extracts keys from the predicate block. Since our keys are already set, this should either be a no-op or overwrite with the same values. This needs verification — see concern below.

CONCERN: OptimizeImplementationsPass at line 504 calls prepareForHash(predicateOperator, cache) even when useHashJoin is already set. This means it will re-extract keys from the predicate block using HashJoinUtils::extractKeys(). If our converter's predicate structure differs from what extractKeys() expects (e.g., our ensureI1() wrapper changes the op structure), the re-extraction might produce different or wrong keys.

To verify: Test with a simple 2-table join query. Check the MLIR before and after OptimizeImplementationsPass. If the leftHash/rightHash attributes remain the same, it's safe. If they change, we need to understand why.

Pass 23: DetachMetaDataPass (conditional)

What it does: Removes meta attributes from BaseTableOps (cleanup after optimization).

Verdict: INCLUDE if AttachMetaData is included. Just cleanup.

Pass 24: CommonSubtreeEliminationPass

What it does: Finds structurally equivalent relational subtrees (e.g., two identical BaseTableOp scans of the same table with same restrictions) and merges them. This avoids duplicate computation.

Verdict: DEPENDS. For JOB queries with node-based split, each sub-query typically scans each table at most once. But some queries with self-joins (e.g., aka_name AS an1, aka_name AS an2) could benefit.

Uncertainty: Our converter creates BaseTableOp with unique scopes per table instance (colMgr_.getUniqueScope()). Two scans of the same table will have different scope names, so the CSE pass may not recognize them as equivalent. Need to check: does the EquivalenceChecker compare by scope name or by table identifier?

Verified from source: Line 106-109 shows that for BaseTableOp, it compares getRestriction() and other attributes except columns. Line 92-93 skips columns attr comparison. But scopes in column definitions are compared via getCanonicalAttr which converts them to a local canonical form (line 205: "LocalColRef::" + locIt->second). So scopes don't matter for equivalence checking. The pass should work correctly on our IR.

Recommendation: INCLUDE — it's safe and could help for self-join
queries.

Pass 25: CanonicalizerPass (final)

Verdict: INCLUDE. Standard MLIR cleanup.

Pass 26: IntroduceTmpPass

What it does: When an operator has multiple consumers (its result is used by 2+ downstream operators), wraps it in a TmpOp that materializes the result once and provides multiple independent scan handles.

Verdict: INCLUDE. Without this, if an operator result feeds into multiple consumers, it gets computed multiple times. This can happen with our IR when a scan feeds both a join and a filter in different subtrees.

---
Summary: Recommended PartialQueryOptimizer Pipeline

Passes to INCLUDE (in order):
 1. CanonicalizerPass                     (MLIR cleanup)
 2. InferNotNullConditionsPass            (enables EliminateNullable)
 3. ColumnFoldingPass                     (remove unused columns)
 4. AttachMetaDataPass(catalog)           (enables OptimizeImpl features)
 5. ReduceGroupByKeysPass                 (cheap, correct)
 6. EliminateNullableTypesPass            (remove nullable overhead)
 7. OptimizeImplementationsPass           (TopK, GroupJoin, filter reorder, INLJ)
 8. DetachMetaDataPass                    (cleanup)
 9. CommonSubtreeEliminationPass          (merge duplicate subtrees)
10. CanonicalizerPass                     (final cleanup)
11. IntroduceTmpPass                      (avoid recomputation)

Passes to SKIP (with reason):
 - SimplifyAggregationsPass              (our IR already has proper AggregationOp)
 - ExtractNestedOperatorsPass            (our IR has no nested relational ops)
 - CSEPass (MLIR CSE)                    (the relalg-specific CSE above is better)
 - DecomposeLambdasPass (×3)             (would decompose our already-correct InnerJoinOps)
 - ImplicitToExplicitJoinsPass           (our IR already has explicit join ops)
 - PushdownPass (×2)                     (DuckDB already did predicate pushdown)
 - UnnestingPass                         (our IR has no correlated subqueries)
 - ExpandTransitiveEqualities            (DuckDB already propagated these)
 - OptimizeJoinOrderPass                 (we want DuckDB's join order, not LingoDB's)
 - CombinePredicatesPass                 (our IR already has combined predicates)

Open Concerns That Need Verification

1. OptimizeImplementationsPass re-extracting hash keys: At line 504, prepareForHash() is called even when useHashJoin is already set. It will re-extract keys from the predicate block. If this produces different results from what our converter set, it could break correctness.
  - How to verify: Run a 2-table equi-join query with debug MLIR dump before and after the pass. Compare leftHash/rightHash attributes.
  - What changes if wrong: If re-extraction produces wrong keys, we'd need to add a guard in OptimizeImplementationsPass to skip ops that already have useHashJoin set. Or we strip our useHashJoin attrs and let the pass set them from scratch.
2. AttachMetaDataPass with temp tables: Temp tables created by our
adapter may not have sample data or indices.
  - How to verify: Add a debug print in AttachMetaData::runOnOperation  to check if getSample() returns non-null for temp table BaseTableOps.
  - What changes if wrong: Filter reordering (selectivity estimation) won't work for temp table filters — filters will keep their original order. Index-NLJ won't trigger for temp tables. Both are acceptable since temp tables are typically small.
3. Interaction between InferNotNullConditions and our IR: This pass walks CmpOp inside SelectionOp and adds IS NOT NULL predicates. Our converter puts CmpOps both in SelectionOp predicates and in
InnerJoinOp predicates. The pass only handles SelectionOp parents
(line 45:mlir::dyn_cast_or_null<relalg::SelectionOp>(cmpOp->getParentOp())), so it won't add null checks for join predicates.
  - What changes if wrong: EliminateNullableTypesPass won't eliminate nullable types for columns only used in join predicates (not in standalone filters). The impact is reduced but not zero.
  - Potential fix: Extend InferNotNullConditions to also handle InnerJoinOp parents, or accept the reduced effectiveness.
4. ColumnFoldingPass correctness with our column structure: Our converter registers all catalog columns on BaseTableOp. ColumnFoldingPass calls eliminateDeadColumns() which requires the ColumnFoldable interface. I need to verify BaseTableOp implements this interface.
  - How to verify: grep -rn 'ColumnFoldable' 
/home/pei/Project/lingo-db/include/lingodb/compiler/Dialect/RelAlg/ — check if BaseTableOp implements it.
  - What changes if wrong: ColumnFolding silently does nothing (returns failed()) and we just carry extra columns. No correctness issue, just reduced performance benefit.

Searched for 3 patterns (ctrl+o to expand)

BaseTableOp does NOT implement ColumnFoldable. This means ColumnFoldingPass won't fold columns on BaseTableOp directly. However, it will work on operators above BaseTableOp (like SelectionOp, InnerJoinOp, MapOp which DO implement ColumnFoldable). The dead column elimination part (eliminateDeadColumns) traverses from top to bottom, so columns unused by consumers will be eliminated from intermediate operators even if BaseTableOp still defines them all.

This resolves concern #4: ColumnFolding works on our IR, but BaseTableOp still defines all columns. The benefit comes from eliminating unused columns in join/filter/map operators above it, which still reduces memory bandwidth.

---
Final Answer

The PartialQueryOptimizer should include these 11 passes. The 3 passes I originally suggested were too few — here's the corrected complete list with ordering that respects pass dependencies:

class PartialQueryOptimizer : public
lingodb::execution::QueryOptimizer {
public:
  PartialQueryOptimizer(lingodb::catalog::Catalog& catalog) : catalog_(catalog) {}
  void optimize(mlir::ModuleOp &module) override {
    mlir::PassManager pm(module.getContext());
    auto& funcPM = pm.nest<mlir::func::FuncOp>();

    funcPM.addPass(lingodb::compiler::createCanonicalizerPass());
    funcPM.addPass(relalg::createInferNotNullConditionsPass());
    funcPM.addPass(relalg::createColumnFoldingPass());
    funcPM.addPass(relalg::createAttachMetaDataPass(catalog_));
    funcPM.addPass(relalg::createReduceGroupByKeysPass());
    funcPM.addPass(relalg::createEliminateNullableTypesPass());
    funcPM.addPass(relalg::createOptimizeImplementationsPass());
    funcPM.addPass(relalg::createDetachMetaDataPass());
    funcPM.addPass(relalg::createCommonSubtreeEliminationPass());
    funcPM.addPass(lingodb::compiler::createCanonicalizerPass());
    funcPM.addPass(relalg::createIntroduceTmpPass());

    if (mlir::failed(pm.run(module)))
      llvm::errs() << "PartialQueryOptimizer: pass pipeline failed\n";
  }
private:
  lingodb::catalog::Catalog& catalog_;
};

The highest-impact passes are (in estimated order of performance
impact):
1. OptimizeImplementationsPass — GroupJoin fusion, TopK, filter reorder
2. EliminateNullableTypesPass — removes null-check overhead on every column access
    funcPM.addPass(relalg::createCommonSubtreeEliminationPass());
    funcPM.addPass(lingodb::compiler::createCanonicalizerPass());
    funcPM.addPass(relalg::createIntroduceTmpPass());

    if (mlir::failed(pm.run(module)))
      llvm::errs() << "PartialQueryOptimizer: pass pipeline failed\n";
  }
private:
  lingodb::catalog::Catalog& catalog_;
};

The highest-impact passes are (in estimated order of performance impact):
1. OptimizeImplementationsPass — GroupJoin fusion, TopK, filter reorder
2. EliminateNullableTypesPass — removes null-check overhead on every column access
3. ColumnFoldingPass — reduces memory bandwidth by dropping unused columns
4. IntroduceTmpPass — avoids recomputation for multi-consumer operators

