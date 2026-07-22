## Goal

Enable the TopDown/SDS (Stats-Driven IR Splitter) optimizations -- currently DuckDB-only on
`topdown_fix` -- to work with PostgreSQL on the `jit` branch, without regressing any existing
DuckDB or PostgreSQL code paths.

## Branches

| Branch | Role | Key changes since last sync |
|--------|------|-----------------------------|
| `topdown_fix` | develop | TopDown/SDS rewrite, IRJoinOptimizer DP, DistinctCache, TryBuildBinaryPlanFromIR, early termination, cross-query prep for TOP_DOWN, projection remapping in query-jit |
| `jit` | release (collaboration) | PostgreSQL adapter + query-JIT, LaunchSpeculativeCompilePG, StripCollationDependentRangeQuals, StoragePlan for PG, PG measurement scripts |
| Last sync | `f36b16c` (jit) / `645a90a` (topdown_fix) | |

### Constraints

- No changes to the Postgres database codebase (`/home/pei/Project/Postgresql-18.3`) required
- All changes behind `#ifdef HAVE_POSTGRES` or strategy checks -- no DuckDB path regressions
- `V17Mode` (`AQP_TD_V17`) and `NoPlanCtor` (`AQP_TD_NO_PLANCTOR`) kill-switches remain functional

## Current Status

TODOs 1-4 and 6 are complete. Details and performance results in `pg_jit_improvement.log`.

Current performance (JOB, 110 queries, NB+tmpl best config):
- **2.42x** vs no-split+no-jit, **1.25x** vs NB+no-jit (64 winners, 46 losers)
- TopDown splitting regresses on PG (~0.90x) — extra PG round-trips dominate

## TODOs

### TODO 5: Query-JIT feature gaps and performance for PostgreSQL

Goal: make all JOB and DSB SPJ queries query-jit compilable on both DuckDB and PostgreSQL,
then achieve the best possible query-JIT performance on PostgreSQL.

---

#### 5A. QJIT feature gap analysis

**Current QJIT rejection reasons** (from `query_jit_steps.cpp` `PlanBuilder::Walk`):

| Rejection code | Meaning | Triggered by JOB? | Triggered by DSB SPJ? |
|---------------|---------|--------------------|-----------------------|
| `join:<Left/Right/Full>` | Non-inner join (line 270-272) | No (all inner) | Yes: query040_spj (LEFT), query072_spj (LEFT) |
| `agg:grouped` | GROUP BY aggregate (line 296-297) | No | No (SPJ queries have no GROUP BY) |
| `expr:arith` | Arithmetic expression in filter (line 200) | No | Possible (e.g. BETWEEN on decimals) |
| `expr:cast` | CAST in filter (line 202) | No | Possible |
| `expr:single-attr` / `expr:*` | SUBSTRING, COALESCE, etc. (line 209-211) | No | Yes: query019_spj (SUBSTRING) |
| `output:agg-type` (PG only) | PG `BuildOutputDescsFromIR` rejects INT64/DOUBLE outputs (line 1064-1066 postgres_adapter.cpp) | No (JOB has no aggs) | No (SPJ has no aggs) |
| `output:unsupported-dtype` (PG only) | PG non-agg output rejects non-INT32/non-VARCHAR (line 1082-1084 postgres_adapter.cpp) | Possible (bigint columns mapped to INT32, so no) | Possible (decimal as VARCHAR, so no INT64/DOUBLE outputs) |

**Which queries can QJIT compile today?**

- **JOB** (113 queries): All 113 are pure SPJ (inner joins, no GROUP BY, no ORDER BY, no LIMIT, no outer joins). All 113 are QJIT-compilable on both DuckDB and PG.
- **DSB SPJ** (13 queries):
  - 10 queries are pure SPJ with inner joins only → QJIT-compilable
  - `query040_spj`: LEFT OUTER JOIN → rejected by `join:Left`
  - `query072_spj`: LEFT OUTER JOIN → rejected by `join:Left`
  - `query019_spj`: uses `SUBSTRING(ca_zip,1,5)` in filter → rejected by unsupported expression type
- **DSB non-SPJ** (agg queries): ALL have GROUP BY → rejected by `agg:grouped`. Out of scope for now.

**Feature priority for maximizing QJIT coverage:**

| Priority | Feature | Queries unblocked | DuckDB status | PG status |
|----------|---------|-------------------|---------------|-----------|
| 1 | LEFT JOIN in QJIT | query040_spj, query072_spj | Not implemented | Not implemented |
| 2 | SUBSTRING/function calls in QJIT filters | query019_spj | Not implemented | Not implemented |
| 3 | GROUP BY in QJIT | All DSB non-SPJ | Not implemented | Not implemented |
| Low | ORDER BY/LIMIT post-processing | None in JOB/DSB SPJ | Already implemented (duckdb_adapter.cpp:1141-1182) | Missing (PG ExecuteQueryJitFinal has no sort/limit) |

**Feature gap details:**

**1. LEFT JOIN support** — Currently `Walk()` at `query_jit_steps.cpp:270` rejects all non-Inner
joins. Supporting LEFT JOIN requires:
- Changing the analyzer to accept `SimplestJoinType::Left`
- Modifying `BuildExecutionSteps` to mark join steps as LEFT (null-extend unmatched probe rows)
- Modifying `ir_to_llvm.cpp` codegen to emit null-extension logic for unmatched probe-side rows
- Both adapters' output desc builders need no change (LEFT JOIN doesn't change output types)

**2. SUBSTRING/function call support** — `CheckExpr()` at `query_jit_steps.cpp:179-212` whitelists
only: VarConstComparison, IsNull, In, VarComparison, LogicalExpr. Any function call
(SUBSTRING, COALESCE, CASE WHEN) hits the default reject. Supporting this requires:
- Adding function-call expression nodes to the whitelist
- Implementing codegen for each function in `ir_to_llvm.cpp`
- This is open-ended; each function needs its own implementation

**3. GROUP BY support** — `Walk()` at `query_jit_steps.cpp:296` rejects grouped aggregates.
Supporting this requires hash-based grouping in the JIT executor, which is a major feature.

**4. ORDER BY/LIMIT post-processing (PG adapter)** — QJIT's `BuildExecutionSteps` already peels
ORDER BY/LIMIT nodes and records them in `QjitQueryPlan::order_by` / `QjitQueryPlan::limit`
(query_jit_steps.cpp:1352-1396). The DuckDB adapter applies post-QJIT `stable_sort` + `resize`
at duckdb_adapter.cpp:1141-1182. The PG adapter's `ExecuteQueryJitFinal` (postgres_adapter.cpp:
1234-1275) does NOT have this post-processing. While no current JOB/DSB query triggers this,
it should be added for completeness.

---

#### 5B. PG adapter output type gaps (affects future features)

PG `BuildOutputDescsFromIR` (postgres_adapter.cpp:1035-1089) is more restrictive than DuckDB's
`BuildQjitOutputDescs` (duckdb_adapter.cpp:3737-3862):

| Output type | DuckDB adapter | PG adapter |
|------------|---------------|------------|
| INT32 | Accepted | Accepted |
| VARCHAR | Accepted | Accepted |
| INT64 (Count/CountStar) | Accepted | Accepted |
| INT64 (non-agg, e.g. BIGINT column) | Accepted | Rejected (`output:unsupported-dtype`, line 1082) |
| DOUBLE (Average result) | Accepted (AQP_DTYPE_DOUBLE) | Rejected (`output:agg-type`, line 1065) |
| Average agg cell dtype | `QjitAggDType::I64` or `F64` | `QjitAggDType::I64` only (line 1076) |

**Current impact**: None. JOB has no aggs. DSB SPJ has no aggs. No JOB/DSB column produces
INT64/DOUBLE output through the current PG type mapping (bigint→INT32, decimal→VARCHAR).

**When this matters**: If GROUP BY or AVG is added to QJIT, or if PG type mapping is changed
to map bigint→INT64 or decimal→INT32/INT64. Then PG `BuildOutputDescsFromIR` needs:
- Accept INT64 for non-agg outputs (line 1082: add `dt == AQP_DTYPE_INT64`)
- Accept DOUBLE for Average agg cells (line 1057-1066: add Average→DOUBLE case)
- Handle `QjitAggDType::F64` for Average cells (line 1073-1077: add F64 branch)
- `ExecuteQueryJitFinal` already handles INT64 output (line 1264), but needs DOUBLE added

---

#### 5C. PG type mapping: DECIMAL stored as VARCHAR

PG `LoadFromPostgreSQL` (storage_plan.cpp:475-482) maps `numeric`/`decimal` columns to
FlatColumnType::VARCHAR. DuckDB `LoadFromDuckDB` (storage_plan.cpp:297-326) maps DECIMAL to
INT32/INT64 based on precision (<=9 → INT32, >9 → INT64) as scaled integers.

**Why PG can't simply copy DuckDB's approach**: IR constants from `duckdb_plan_to_ir.cpp`
(lines 1097-1101) convert DECIMAL constants as unscaled `FloatVar` values (e.g., `123.45` as
`double 123.45`). QJIT's `EmitVarConst` in `ir_to_llvm.cpp:1175-1176` casts this to integer:
`(int64_t)cv->GetFloatValue()` → `123`, but DuckDB stores as scaled integer `12345`. The
comparison works because DuckDB's `LoadFromDuckDB` stores the raw scaled value AND the IR
constant was produced from DuckDB's plan (same scale). For PG, if we stored decimal as scaled
INT32/INT64, the IR constants (still from DuckDB's plan) would be unscaled → filter comparison
would be WRONG (comparing unscaled constant `123` against scaled stored value `12345`).

**Impact on DSB SPJ**: 12 of 13 DSB SPJ queries use decimal columns (ss_sales_price,
cs_net_profit, etc.) in filter predicates. These are stored as VARCHAR strings. QJIT compiles
VARCHAR filters as string comparisons. Correctness is preserved but performance is suboptimal
(string compare vs integer compare).

**Proper fix** (future work, not blocking):
1. Track column scale in `FlatColumn` metadata
2. In `ExpectedDtype` (query_jit_steps.cpp:574), map FloatVar to INT32/INT64 (already done)
3. In `EmitVarConst` (ir_to_llvm.cpp:1175), scale the constant: `(int64_t)(cv->GetFloatValue() * pow(10, scale))`
4. In PG `LoadFromPostgreSQL`, parse decimal text with scale awareness: `"123.45"` → INT32 `12345`
5. This requires passing column scale through the pipeline (FlatColumn → QJIT source → codegen)

---

#### 5D. PG performance issues (existing, not blocked by feature gaps)

**Primary issue: full table scan vs PG indexes.** Query-JIT reads from `StoragePlan` flat
column arrays (full scans + hash joins). PG's native path uses B-tree indexes for selective
queries. DuckDB also does full scans with query-JIT but doesn't regress because DuckDB's
native path is also full-scan (columnar engine, no B-tree indexes).

Current performance (JOB, node-based, NB+tmpl best config):
- **2.42x** vs no-split+no-jit
- **1.25x** vs NB+no-jit (64 winners, 46 losers)
- Big losers: 12b (0.10x), 4c (0.12x) — selective queries where PG uses index scan but JIT does full scan

Potential fixes:
1. SortedIndex for PG tables — currently `kSortedCols` in storage_plan.cpp:764-777 is hardcoded
   for JOB tables only, and `sorted_index.cpp:22-23` only supports INT32/VARCHAR. Needs
   generalization for DSB tables and INT64 support.
2. Selective JIT bypass using EXPLAIN cost estimates (skip JIT when PG would use index scan)
3. Better block-skipping with min/max predicate pushdown

**Secondary issue: MaterializeQjitTempToPostgreSQL overhead.** Row-by-row `PQputCopyData`
with text serialization. Measured: 5-430ms. Potential fix: binary COPY format or row batching.

---

#### 5E. PG ORDER BY/LIMIT post-processing TODO

The DuckDB adapter applies ORDER BY and LIMIT after QJIT execution (duckdb_adapter.cpp:1141-1182):
```
if (!compiled->order_by.empty()) {
    stable_sort(result.rows, comparator);  // supports INT32, INT64, DOUBLE, VARCHAR
}
if (compiled->limit >= 0 && result.rows.size() > compiled->limit) {
    result.rows.resize(compiled->limit);
}
```

The PG adapter's `ExecuteQueryJitFinal` (postgres_adapter.cpp:1234-1275) is missing this.
No current JOB/DSB query triggers this (none have ORDER BY/LIMIT), but it should be added.

**Implementation**: Copy the DuckDB sort+limit logic into PG `ExecuteQueryJitFinal`, after the
row-building loop (after line 1272). The sort comparator uses string-based comparison (same as
DuckDB) so it's engine-agnostic.

---

#### 5F. TopDown splitter outer join handling

TopDown splitter already handles outer/semi/anti joins by "locking" the subtree tables
(topdown_splitter.cpp:176-194). When a LEFT/RIGHT/FULL/SEMI/ANTI join is detected, all tables
in that subtree are added to the `locked` set, preventing the splitter from splitting across
the non-inner join. This is correct — QJIT will receive the entire outer-join subtree and
reject it (falling back to engine execution), but the split boundary won't break join semantics.

Node-based splitter (node_based_splitter.cpp) has no explicit outer join handling — it relies
on DuckDB's `MiddleOptimize` to determine split points, which implicitly preserves join
semantics.

---

#### 5G. DistinctCache PG paths

All three DistinctCache methods use PG catalog lookups (restored from jit branch):
- `Get()`: `SELECT CASE WHEN n_distinct >= 0 THEN n_distinct ELSE -n_distinct * reltuples END FROM pg_stats JOIN pg_class` — O(1) catalog lookup, handles negative n_distinct (fraction of table)
- `GetCorrelation()`: `SELECT abs(correlation) FROM pg_stats` — uses PG's pre-computed correlation stat
- `GetRowCount()`: `SELECT reltuples FROM pg_class` — O(1) catalog lookup

DuckDB paths use full SQL queries (`COUNT(DISTINCT col)`, `corr(rowid, col)`, `COUNT(*)`).
Engine dispatch via `adapter.GetEngineName() == "PostgreSQL"` in distinct_cache.cpp.

---

#### 5H. Summary of actionable TODOs

| # | TODO | Scope | Queries affected | Priority |
|---|------|-------|-----------------|----------|
| 5.1 | Add ORDER BY/LIMIT post-processing to PG `ExecuteQueryJitFinal` | ~20 lines, port from DuckDB adapter | None currently, future-proofing | Low |
| 5.2 | Add INT64/DOUBLE output support to PG `BuildOutputDescsFromIR` | ~15 lines | None currently (needs agg or type mapping changes) | Low |
| 5.3 | Add DOUBLE case to PG `ExecuteQueryJitFinal` result building | ~3 lines | None currently | Low |
| 5.4 | LEFT JOIN support in QJIT analyzer + codegen | Major: analyzer + BuildExecutionSteps + ir_to_llvm | DSB query040_spj, query072_spj | Medium |
| 5.5 | SUBSTRING support in QJIT expression codegen | Medium: CheckExpr whitelist + ir_to_llvm | DSB query019_spj | Medium |
| 5.6 | Fix DECIMAL constant scaling in EmitVarConst for PG | Medium: FlatColumn scale metadata + codegen | 12/13 DSB SPJ queries (perf, not correctness) | Medium |
| 5.7 | Selective JIT bypass for index-friendly queries | Analysis + heuristic | JOB 12b, 4c, etc. (PG perf) | Medium |
| 5.8 | GROUP BY support in QJIT | Major: hash grouping in JIT executor | All DSB non-SPJ queries | Low (large effort) |
| 5.9 | Generalize SortedIndex for DSB tables + INT64 | storage_plan.cpp + sorted_index.cpp | DSB tables (PG perf) | Low |

TODOs 5.1-5.3 are small, safe, and can be done independently. TODOs 5.4-5.5 unblock specific
DSB queries. TODO 5.6 improves DSB SPJ performance on PG. TODO 5.7 addresses the biggest PG
regression. TODOs 5.8-5.9 are larger efforts for future work.

## Key Code Locations (Reference)

### Splitter Layer (adapter-agnostic)

| File | What |
|------|------|
| `include/split/topdown_splitter.h` | TopDown/SDS splitter header; `V17Mode()`, `NoPlanCtor()` static methods |
| `src/split/topdown_splitter.cpp` | SDS `PlanNext()`, `SplitIR()`, `EstimateFilteredCardinality()` |
| `include/split/ir_join_optimizer.h` | DP join-order optimizer (pure algorithm, no engine deps) |
| `src/split/ir_join_optimizer.cpp` | DPccp on JoinRel/JoinEdge with tdom cardinality model |
| `include/split/distinct_cache.h` | Per-(table,column) stats cache |
| `src/split/distinct_cache.cpp` | `Get()`, `GetCorrelation()`, `GetRowCount()` -- PG uses `pg_stats`/`pg_class`, DuckDB uses SQL |
| `src/split/fk_based_splitter.cpp` | Sub-IR construction, mark-join cloning, condition dedup |

### Orchestrator

| File | What |
|------|------|
| `include/split/ir_query_splitter.h` | `IRQuerySplitter` header; `SpeculativeCompilation` struct, `CrossQueryPrepResult` |
| `src/split/ir_query_splitter.cpp` | `ExecuteWithSplit`, `ExecuteSplitLoop`, `ExecuteOneIteration` -- engine+strategy dispatch |

### Adapters

| File | What |
|------|------|
| `include/adapters/db_adapter.h` | `EngineAdapter` base class; `ApplyEngineSetting` (no-op default), `GenerateSQL` |
| `include/adapters/duckdb_adapter.h` | DuckDB adapter; `SetQjitPendingIR(ir, use_engine_plan=true)`, `TryBuildBinaryPlanFromIR` |
| `src/adapters/duckdb_adapter.cpp` | `ExecuteSQLandCreateTempTable` with plan-ctor fast path + SQL fallback |
| `include/adapters/postgres_adapter.h` | PG adapter; `PgCachedSubquery`, `PgCachedQueryPlan`, `ReplayQjitSubquery`, `ReplayQjitFinal` |
| `src/adapters/postgres_adapter.cpp` | PG `ExecuteSQLandCreateTempTable`, `AnnotateBuildSidesByCard`, `SpeculativeQueryJitCompile`, `ResetQueryState` |

### SimplestIR (intermediate representation)

| File | What |
|------|------|
| `third_party/IR_SQL_Converter/inc/simplest_ir.h` | All IR node class definitions |
| `third_party/IR_SQL_Converter/src/duckdb_plan_to_ir.cpp` | DuckDB logical plan -> SimplestIR |
| `third_party/IR_SQL_Converter/src/ir_to_sql.cpp` | SimplestIR -> SQL string |
| `third_party/IR_SQL_Converter/inc/cpp_interface.h` | Public API: ConvertDuckDBPlanToIR, ConvertIRToSQL |

### Query JIT

| File | What |
|------|------|
| `src/qjit/query_jit_steps.cpp` | `PlanBuilder` with `proj_remap_` for DuckDB compressed-mat projections |
| `src/jit/ir_to_llvm.cpp` | IR-to-LLVM codegen |

## Repositories

- **AQP_middleware**: `/home/pei/Project/AQP_middleware` (branches: `topdown_fix`, `jit`)
- **DuckDB (patched)**: `/home/pei/Project/duckdb`
- **PostgreSQL**: `/home/pei/Project/Postgresql-18.3`
- **JOB queries**: `/home/pei/Project/benchmarks/imdb_job-postgres/queries/`
- **JOB schema**: `/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql`
- **DuckDB database**: `/home/pei/Project/duckdb/measure/imdb.db`

Build commands:
```bash
# Middleware (debug / release)
cd /home/pei/Project/AQP_middleware && cmake --build build_debug -j$(nproc)
cd /home/pei/Project/AQP_middleware && cmake --build build_release -j$(nproc)
# DuckDB
cd /home/pei/Project/duckdb && cmake --build build/release -j 16
```

Build hazards:
- build_debug and build_release share `lib/*.a` outputs. If a debug build
  poisons the release archive (ASan), rebuild with
  `--target IR_SQL_Converter_C_static --clean-first`.

## Verification Workflow

### Single query end-to-end

Take a simple JOB query (e.g., 1a -- single join), run with
`./build_release/aqp_middleware ...` (check reference in measure/run_job.sh). Compare result
to golden output.

### Full JOB correctness

Run all 113 JOB queries with the correct flags, compare against golden files:
- DuckDB: `measure/duckdb_job_no-split_golden.txt`, `measure/duckdb_job_node-based_golden.txt`
- PostgreSQL: `measure/pg_job_no-split_golden.txt`, `measure/pg_job_node-based_golden.txt`, `measure/pg_job_topdown_golden.txt`

For more flags, check `measure/correctness_test.sh` (DuckDB) and `measure/correctness_test_pg.sh` (PG).

### Analysis scripts (measure/*.py)
- `tune_per_subquery.py [split]` -- pick best config per (query, sub-query)
- `show_all_configs.py [split]` -- summary table across all configs
- `find_top_queries.py [path] [--top=N]` -- rank queries by slowest median
- `verify_tuned_detail.py` -- verify tuned config results
