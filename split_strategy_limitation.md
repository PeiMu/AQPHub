# Split Strategy: Limitations, Lessons, and Design Opportunities

Compiled from all `*.md` and `*.log` files in the AQP_middleware repository.
Intended as a requirements document for designing a new split strategy.

---

## 1. Current Node-Based Split: How It Works

File: `src/split/node_based_splitter.cpp`

Loop (each iteration):
1. `MergeSubquery()` -- merge pending subqueries back into the plan
2. `MiddleOptimize()` -- re-run DuckDB's join ordering + filter pushdown
3. `Split()` -- TopDownSplit assigns split_index to right children of joins
4. `ReorderGet()` -- sort base tables by cardinality (biggest first)
5. `MergeSubquery()` + `Split()` again -- BLOCK 2
6. Extract `subqueries_.front()[0]` as the current subquery
7. Generate SQL, execute, create temp table
8. `UpdateRemainingIR()` -- replace extracted tables with CHUNK_GET node

---

## 2. Fundamental Limitations of Node-Based Splitting

### L1: Pipeline-breaker-based, not cost-based
TopDownSplit splits at structurally determined points (right children of joins)
rather than at cost-optimal points. It does not consider intermediate result
sizes, filter selectivity, or execution cost. (Source: `split_strategy_limitation.md`)

### L2: Greedy, non-backtracking
Each iteration commits to a split decision. There is no mechanism to reconsider
or undo a bad split after seeing actual cardinality. (Source: `split_strategy_limitation.md`)

### L3: Materialization overhead is inherent
Every split forces full materialization of intermediate results. DuckDB natively
pipelines data without materializing. The cumulative effect of 4-7 materializations
adds ~1.4% of total time. (Source: `split_strategy_limitation.md`)

### L4: Re-optimization overhead (ExtractIR dominates MW)
Each iteration runs MiddleOptimize, adding ~0.5-14ms per iteration. Total MW
overhead is 535ms across 113 queries (ExtractIR 451ms, SQL parse 36ms,
extra_mat 16ms, SQL gen 10ms, UpdateIR 9ms, prepare_mw 10ms).
**ExtractIR alone is 84% of MW overhead** -- the IR tree traversal + copy is
the single biggest MW bottleneck. (Source: `CLAUDE.md`, `jfp_for_aqp.log`)

### L5: JFP (Join Filter Pushdown) asymmetry is the dominant performance gap
JOIN_FILTER_PUSHDOWN accounts for **99.8% of the asymmetry** between none-split
and node-based on DuckDB 1.5.2. Disabling JFP flips the gap: node-based goes
from +1.035s slower to -1.316s faster.

Three sub-gaps:
- **Gap 2 (cross-boundary filter loss, ~88-90%)**: Filters from hash join build
  sides in one subquery cannot reach probe-side table scans in other subqueries.
  This is inherent to materialization-based splitting.
- **Gap 1 (temp as JFP target, ~6.5%)**: LogicalColumnDataGet was not a JFP
  target. Fixed by Opt-10, but impact was modest -- range filters on temp tables
  are too broad and bloom filters rarely trigger.
- **Gap 3 (no zone maps on ColumnDataCollection, ~5%)**: ColumnDataCollection has
  no per-chunk statistics. Inherent and unfixable.

(Source: `duckdb_152_3_passes_analysis.md`, `jfp_for_aqp.log`)

### L6: Zone-map pruning is useless on FK columns
cast_info.movie_id has completely unsorted data -- all 590 row groups span
[~0, ~2.5M]. Even injecting range filters via DynamicTableFilterSet would skip
0 row groups. Same for movie_info.movie_id, movie_keyword.movie_id.
(Source: `jfp_for_aqp.log`)

### L7: IN-list injection infeasible for large temps
All relevant temp tables have >50 unique join key values (smallest: 478 in 30b;
top-gap: 571,710 in 19d/8c/10c/9d/8d). (Source: `jfp_for_aqp.log`)

---

## 3. Verified Failure Patterns (per-query analysis)

### Pattern A: Intermediate Cardinality Explosion (8 of 10 worst queries)
The `cn JOIN mc WHERE country_code='[us]' -> 1,153,798 rows` pattern appears in
8c, 9d, 10c, 19d, 16b, 16c, 16d, 12c. DuckDB natively avoids materializing this
by joining with more selective tables first.

Critical: the explosion does NOT always happen at iteration 1. For 16b/16c/16d/12c,
it happens at iteration 2 as a standalone subquery (no temp table reference).
(Source: `split_strategy_limitation.md`, `node-based-opt.log`)

### Pattern B: Selective Predicate Deferred to Final Query (6d, 12c)
Highly selective predicates (e.g., `name LIKE '%Downey%Robert%'` matching 2 rows)
are not included in early subqueries. TopDownSplit extracts from the bottom of the
plan tree, and the selective table is near the top.
(Source: `split_strategy_limitation.md`)

### Pattern C: Sum of Subquery Execution > Single Execution (30b)
Even without cardinality explosion, splitting adds overhead:
1. DuckDB can't pipeline data across temp table boundaries
2. Each subquery runs through parse -> optimize -> execute independently
3. DuckDB's native dynamic filter pushdown cannot cross temp table boundaries

30b has 6 tiny temp tables (2-28 rows). The bottleneck is joining cast_info (36M)
with temp5 (12 rows) -- native DuckDB pushes dynamic filters from the 12-row
result to skip most of cast_info. Split prevents this.
(Source: `split_strategy_limitation.md`)

---

## 4. Optimization History: What Worked, What Didn't

### 4.1 Successful optimizations (active in current code)

| Opt | Description | Gap Impact | Source |
|-----|------------|:----------:|--------|
| Opt-1 | Post-execution cardinality threshold (1M, standalone safeguard) | Saves 626ms net | `node-based-opt.log` |
| Opt-10 | JFP support for LogicalColumnDataGet (temp as JFP target) | -0.067s gap | `jfp_for_aqp.log` |
| Opt-11 + kBF=2M | Cross-boundary BF + range for final query | -0.218s gap | `jfp_for_aqp.log` |
| Opt-3 removal | Removed dead column stats infrastructure (no effect on 1.5.2) | -138ms MW | `jfp_for_aqp.log` |

### 4.2 Failed/reverted optimizations (lessons learned)

| Opt | Description | Result | Lesson |
|-----|------------|--------|--------|
| Opt-2 | Pre-execution selectivity detection | NOT FEASIBLE | DuckDB's pre-execution estimates are raw table sizes, not join result estimates. No heuristic can distinguish beneficial vs harmful first subqueries. |
| Opt-5 | Skip 2-group splitting | REVERTED | Caused jit-cache correctness failures due to non-deterministic unordered_map iteration in MergeSubquery. |
| Opt-6 | Dynamic filter pushdown for temp tables | NOT FEASIBLE | ColumnDataCollection has no zone maps. Scan-level filtering replicates what DuckDB's Filter operator already does. |
| Opt-7 | STATISTICS_PROPAGATION in MiddleOptimize | +101.7% regression | Propagated statistics override real cardinality feedback from temp tables, defeating node-based's core strength. |
| Opt-8 | Temp table HLL distinct counts | NO EFFECT | For temp-to-real joins, real tables have HLL. For all-temp joins, distinct count ~ cardinality. |
| Opt-12a | TempCollectionStatistics (per-column min/max) | +0.416s regression | DuckDB's optimizer makes worse join orderings with simple min/max stats (no histograms/distributions). |
| Opt-12b | Transitive BF injection | NO EFFECT | No multi-hop equality chains exist in JOB queries. |
| Opt-14 | Selectivity check for middleware BFs | +33ms regression | Disabled ineffective BFs but new native BFs from Opt-15 caused redundant filtering. |
| Opt-15 | `IsFiltering()=true` for LOGICAL_CHUNK_GET | +33ms regression | Changed `build_side_has_filter`, altering join ordering decisions. |
| Opt-16 | Transitive BF via join graph analysis | NOT IMPLEMENTED | ~135ms improvement for significant effort. DuckDB's own JFP already covers most transitive paths. |

(Sources: `node-based-opt.log`, `jfp_for_aqp.log`)

### 4.3 Key lessons from optimization history

1. **Providing partial statistics is worse than no statistics**: Opt-7, Opt-8,
   Opt-12a all showed that giving DuckDB's optimizer incomplete information
   (min/max without histograms, distinct counts without distributions) leads
   to worse plans than the optimizer's conservative defaults.

2. **The Opt-1 threshold dilemma**: Threshold=1M is optimal overall, but 4 queries
   (15c, 15d, 13b, 13c) are hurt because their 1.15M standalone iter1 triggers
   abort, but subsequent iterations successfully reduce via iterative correction.
   No single threshold value can distinguish harmful vs beneficial large intermediates.
   A cost-based decision would solve this.

3. **Cross-boundary filter loss is inherent to materialization-based splitting**:
   No amount of BF injection or statistics propagation can fully recover DuckDB's
   native pipeline-internal JFP. This is a fundamental architectural limitation.

4. **Node-based's core strength is iterative cardinality correction**: After each
   subquery, MergeDataChunk injects actual cardinality into LogicalColumnDataGet.
   The next MiddleOptimize uses this real cardinality for join ordering. This
   produces better plans on queries where DuckDB's estimates are wrong.

---

## 5. Performance Envelope: Where Node-Based Wins vs Loses

### 5.1 Current performance (DuckDB 1.5.2, opt_13_fix)

| Config | Total(s) | MW(s) | Exec(s) | Gap vs None |
|--------|----------|-------|---------|-------------|
| none-split | 9.533s | 0.003s | 9.529s | -- |
| node-based | 10.162s | 0.535s | 9.627s | +0.629s (+6.6%) |

Gap decomposition: MW overhead +532ms, execution +98ms.

### 5.2 Queries where node-based wins (iterative correction value)

| Query | Speedup | Mechanism |
|-------|---------|-----------|
| 7c | 2.97x (0.531 -> 0.178) | DuckDB's native join ordering is suboptimal |
| 11d | 4.63x (0.292 -> 0.063) | Real cardinality corrects bad estimates |
| 22d | 1.79x (0.244 -> 0.136) | Iterative correction |
| 17d | 2.34x (0.180 -> 0.077) | Iterative correction |
| 17c | 2.48x (0.119 -> 0.048) | Iterative correction |

Common pattern: moderate first-subquery cardinalities (29K-454K), followed by
successful iterative correction using real cardinalities.

### 5.3 Queries where node-based loses (top 10 exec gaps)

| Query | ExecGap(ms) | Root Cause |
|-------|-------------|------------|
| 19d | +133 | ci: 6.3M(NB) vs 1.1M(none) -- Pattern A (JFP asymmetry) |
| 8c | +117 | Join order diff -- Pattern A (cn JOIN mc = 1.15M) |
| 6d | +92 | Selective predicate deferred -- Pattern B |
| 10c | +73 | ci: same rows, join order diff |
| 30b | +74 | 6 tiny temps, no pipeline -- Pattern C |
| 22a | +60 | All temp-to-temp final query |
| 30a | +58 | Join order diff |

### 5.4 Iteration distribution (113 JOB queries)

| Iterations | Queries | Notes |
|-----------|---------|-------|
| 1 | 9 | Minimal benefit from splitting |
| 2 | 19 | 14 are threshold-aborted |
| 3 | 23 | |
| 4 | 17 | |
| 5 | 9 | |
| 6 | 15 | |
| 7+ | 21 | |
| Threshold-aborted | 28 | Opt-1 triggers |

### 5.5 DuckDB 1.3.2 reference (node-based wins by 0.426s)

| Config | Total(s) | Gap |
|--------|----------|-----|
| none-split | 11.840s | -- |
| node-based (Opt-1) | 11.414s | -0.426s (-3.6%) |

Node-based wins on 1.3.2 because JFP is simpler (no bloom filters, no
SelectivityOptionalFilter). The asymmetry does not exist.

---

## 6. Structural Constraints (verified, not addressable by split strategy)

1. **Zone-map pruning useless on FK columns**: cast_info.movie_id all 590 segments
   span full domain. No threshold-based skipping possible.

2. **IN-list infeasible for large temps**: All relevant temps have >50 unique
   join key values.

3. **Multi-join cascade requires execution reordering**: Fixing cross-boundary
   filter loss requires pre-scanning future subquery tables. This is equivalent
   to changing split order -- a strategy-level change.

4. **BFTableFilter holds reference not ownership**: Cannot safely inject BFs via
   DynamicTableFilterSet without refactoring DuckDB's BFTableFilter.

5. **ColumnDataCollection has no zone maps**: Inherent to DuckDB's in-memory
   collection format. Range filters degrade to row-by-row evaluation.

---

## 7. Insights from Kernel/Storage Plan Work

From `storage_plan_design.md`, `kernel_path_opt.md`, `query-level-kernel-opt.md`:

### 7.1 Sub-query pattern distribution (640 total sub-queries, 113 JOB queries)

| Rank | Pattern | Count | % | Description |
|------|---------|-------|---|-------------|
| 1 | 1base + 1temp | 170 | 25.8% | Join base table with one temp |
| 2 | 1base + 1dim | 79 | 11.9% | Base filtered via dimension |
| 3 | 2base | 63 | 9.6% | Two base tables (first subquery) |
| 4 | 1base + 1temp + FINAL | 57 | 8.6% | Final aggregation |
| 5 | 1base + 1dim + 1temp | 48 | 7.3% | Base + dimension + temp |
| 6 | 2temp | 47 | 7.1% | Intersect two temps |
| 7 | 2base + 1temp | 38 | 5.8% | Two base + temp |

6 patterns cover 70%. 15 patterns cover 97%.

### 7.2 Execution model insights

- **CSR (O(1) lookup) vs hash join**: CSR eliminates hash table build+probe
  (InsertHashes 5-17%, AdvancePointers 6-8% of CPU). Pre-built at load time.
  Runtime CSR on temp tables adapts to actual selectivity.

- **Flat column arrays**: Eliminate DuckDB's decompression overhead
  (fsst_decompress 3-15%, StringScanPartial 2-5%).

- **Dimension constants**: Tables with <200 rows (kind_type=7, role_type=12,
  info_type=113) can be resolved at parse time, eliminating joins entirely.

- **Pipeline kernel performance**: With hash-based joins, adaptive probing
  (DIRECT/POINT/LINEAR/HASH), prefetch, and SIMD batch processing, the kernel
  achieves 10,479ms execution vs DuckDB's 9,529ms. The remaining gap is MW
  overhead (1,462ms) + compile (234ms).

### 7.3 MW overhead breakdown (pipeline kernel path)

| Component | Time | % |
|-----------|------|---|
| extract_next_sub-IR | 675ms | 46.2% |
| generate_final_sub_sql | 474ms | 32.4% |
| generate_sub-SQL | 209ms | 14.3% |
| DuckDB JIT compile | 150ms | (in fallback iters) |
| Other | 104ms | 7.1% |
| **Total** | **1,462ms** | |

### 7.4 BespokeOLAP/GenDB comparison

Systems 14-173x faster than DuckDB. ~90% of speedup from storage plan (CSR,
flat arrays), ~10% from tighter compiled loops. Key differences:

| Aspect | BespokeOLAP | Our Approach |
|--------|-------------|--------------|
| Architecture | Single AOT function per query | Iterative split-execute loop |
| MW overhead | 0ms | 535-1,462ms |
| Temp materialization | None (loop fusion) | Per-iteration |
| Adaptiveness | None (fixed plan) | Adaptive (real cardinalities) |

---

## 8. Design Opportunities for a New Split Strategy

### O1: Cost-based split decisions (addresses L1, L2, Opt-1 dilemma)

Replace structural splitting (right children of joins) with cost-based decisions:
- Estimate intermediate cardinality BEFORE committing to a split
- Consider filter selectivity, join fan-out, and table sizes
- Allow backtracking: if a split would produce >threshold intermediate, skip it

This directly addresses the Opt-1 threshold dilemma -- rather than aborting after
seeing a 1.15M standalone result, a cost-based strategy would avoid producing it
in the first place.

**Challenge**: Pre-execution cardinality estimates from DuckDB are unreliable
(Opt-2 finding). May need lightweight sampling or partial execution to get
estimates.

### O2: Selective predicate awareness (addresses Pattern B)

Detect highly selective predicates (LIKE with rare patterns, equality on
near-unique columns) and ensure they appear in early subqueries. Currently,
TopDownSplit extracts from the bottom of the plan tree without regard for
predicate selectivity.

**Approaches**:
- Score predicates by estimated selectivity (type-based heuristic: LIKE=high,
  equality on dimension FK=medium, range on year=low)
- Prioritize subqueries containing high-selectivity predicates
- Pull selective tables into earlier iterations

### O3: Cardinality-aware split ordering (addresses Pattern A)

The cn JOIN mc (1.15M rows) pattern hurts because it is extracted as a standalone
subquery without connection to more selective tables. A new strategy could:
- Prefer subqueries that connect temp tables with base tables (not standalone)
- Delay large standalone joins until more temps exist to constrain them
- Use the kBFMaxTempCard=2M natural gap as a planning threshold

### O4: Reduce MW overhead (addresses L4, biggest remaining opportunity)

ExtractIR at 451ms is 84% of MW overhead. Opportunities:
- **Incremental IR update**: Instead of re-traversing the full IR tree each
  iteration, maintain a mutable plan and apply deltas
- **Skip MiddleOptimize when unnecessary**: If the plan hasn't changed materially
  (e.g., only added one temp), skip the re-optimization
- **Batch multiple splits**: Extract 2-3 subqueries per iteration instead of one,
  reducing the number of IR traversals
- **Cache SQL generation**: Reuse generated SQL fragments across iterations

### O5: Minimize cross-boundary filter loss (addresses L5, Gap 2)

This is the single biggest performance gap. Approaches:
- **Merge subqueries when JFP benefit is high**: If the next subquery would
  benefit from dynamic filters flowing from the current one, execute them
  together as a single query
- **Pre-inject bloom filters**: Build BFs from known temp data and inject them
  into subsequent subqueries before execution (Opt-11 does this for the final
  query; extend to ALL iterations)
- **Delay large-table scans**: cast_info (36M) benefits most from JFP. Ensure
  it is scanned in a subquery where all relevant filters are available
- **Split less aggressively**: Fewer splits = fewer JFP boundary losses. A
  strategy that produces 2-3 subqueries instead of 5-7 would lose less

### O6: Adaptive granularity (addresses Pattern C)

For queries where splitting adds overhead without benefit (Pattern C: small
intermediates, many splits), detect this pattern and reduce or skip splitting:
- If all intermediates are tiny (<1000 rows), the iterative correction value
  is low -- just run the full query
- If the query has few joins (2-3), splitting provides minimal re-optimization
  opportunity

### O7: Exploit runtime information beyond cardinality

Node-based currently uses only cardinality from temp tables. Additional runtime
information available after each iteration:
- **Actual join selectivity**: ratio of output/input rows
- **Value distributions**: min/max/distinct counts of join keys (Opt-8 showed
  these don't help DuckDB's optimizer, but could inform the split strategy)
- **Filter selectivity**: how many rows passed each filter
- **Execution time**: which operations were expensive

A new strategy could use these signals to adaptively decide:
- Whether to continue splitting or merge remaining
- Which table to scan next
- Whether to re-optimize or reuse the existing plan

### O8: Leverage auxiliary storage structures

With flat column arrays, CSR indexes, dimension cache, and sorted indices:
- **Eliminate dimension joins at split time**: Resolve dimension predicates to
  constants before splitting, reducing join count
- **Use CSR for O(1) joins instead of hash joins**: A split strategy aware of
  CSR availability could prefer subqueries that use CSR-joinable patterns
  (1base + N temps, fully CSR-joinable)
- **Direct PK index for temp joins**: Instead of materializing into
  ColumnDataCollection, store temp results as flat arrays with direct PK index

### O9: Multi-level / hierarchical splitting

Instead of the current flat sequence of iterations, consider:
- **Top-level split**: Divide the query into 2-3 independent subgraphs
  (connected components in the join graph)
- **Per-subgraph optimization**: Within each subgraph, apply iterative
  correction (node-based's strength) or run unsplit (if few joins)
- **Cross-subgraph BF injection**: Build BFs from earlier subgraphs and inject
  into later ones

### O10: Speculative / look-ahead planning

From `spec_jit_design.md`: speculative compilation can predict the next subquery
and pre-compile JIT code. The same idea applies to split strategy:
- After each iteration, tentatively split the remaining plan to predict the next
  subquery
- Use the prediction to start pre-fetching data, building hash tables, or
  compiling JIT code on a background thread
- If the prediction is correct (SQL matches), use the pre-built artifacts
- If wrong, discard and proceed normally

### O11: Fewer, larger subqueries (reduce split count)

Current node-based produces 4.7 iterations on average (533 total / 113 queries).
Each iteration adds ~4.7ms MW overhead. Reducing to 2-3 iterations would save
~200ms MW total and reduce cross-boundary filter loss.

**Approach**: Extract multi-table subqueries instead of 2-table pairs. Group
related tables that form a selective subquery (e.g., dimension + fact with
selective predicate). This requires understanding predicate selectivity (O2).

### O12: Handle the 2-group pattern correctly

28% of queries (32 queries) reach a 2-group state at some point. Currently,
Opt-5 (skip 2-group) was reverted due to MergeSubquery non-determinism (DuckDB
bug with unordered_map). Fixing MergeSubquery's determinism would recover ~69ms.
A new strategy could:
- Use ordered containers in MergeSubquery
- Or handle 2-group as a special case without merging back

---

## 9. Gap Budget for a New Strategy

Starting from the current gap of +0.629s (+6.6%):

| Component | Current Cost | Addressable? | Notes |
|-----------|:------------:|:------------:|-------|
| MW overhead | +532ms | YES | O4 (incremental IR, batch splits) |
| Exec: JFP asymmetry | ~+400ms | PARTIALLY | O5 (merge subqueries, pre-inject BFs) |
| Exec: Pattern A (cardinality explosion) | ~+200ms | YES | O1, O3 (cost-based, cardinality-aware) |
| Exec: Pattern B (deferred predicates) | ~+100ms | YES | O2 (predicate awareness) |
| Exec: Pattern C (tiny temp overhead) | ~+75ms | YES | O6 (adaptive granularity) |
| Exec: inherent materialization | ~+50ms | NO | Fundamental to splitting |
| **Total addressable** | ~+1,250ms | | |
| **Inherent minimum** | ~+50ms | | |

A new strategy that combines O1-O6 could potentially make node-based **faster**
than none-split by preserving the iterative correction benefit (-700ms on winning
queries) while eliminating the overhead on losing queries.

---

## 10. Related Documents

| Document | Content |
|----------|---------|
| `CLAUDE.md` | Project overview, current performance, code locations, verification workflow |
| `jfp_for_aqp.log` | JFP implementation log: Opt-10 through Opt-16, gap analysis, all completed/reverted details |
| `node-based-opt.log` | Opt-1 through Opt-8 details, threshold tuning, gap attribution |
| `duckdb_152_3_passes_analysis.md` | Experiment isolating JFP as root cause |
| `storage_plan_design.md` | Auxiliary storage structures: flat arrays, CSR, sorted indices, dimension cache |
| `kernel_path_opt.md` | Pipeline kernel optimizations: adaptive join, prefetch, SIMD batch |
| `query-level-kernel-opt.md` | Query kernel (CSR-based) optimizations and remaining steps |
| `open_mp_opt.md` | OpenMP parallelization opportunities for CSR build, startup |
| `spec_jit_design.md` | Speculative JIT compilation design |
| `measure/notes/compilation_latency_analysis.md` | Tiered compilation, parametric caching analysis |
| `measure/notes/runtime_execution_optimizations.md` | ROF, predicate reordering, PCQ analysis |
| `measure/notes/hash_join_simd_optimization_analysis.md` | Hash tag, radix partitioning, batch probe, SIMD analysis |
| `perf_analysis.md` | Profiling methodology guide |
| `workflow.md` | Development workflow and iteration process |
