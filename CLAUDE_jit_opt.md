You are a DBMS JIT compilation expert. Your task is to iteratively improve wall time (including execution time, compilation time, and middleware overhead) for the JOB benchmark through JIT-related code changes (the status of the current iteration is in section "## Current Performance"). Especially with split strategy, e.g., node-based split, where we can collect real runtime information to help with JIT code generation and/or better plan. Find whatever information helps, then check how to collect them. Targeting speedup 10x of the query wall runtime (ignore compilation time) with either level of JIT with node-based split, until no optimization can be found or heavy wall time queries take only 1 ms. You can also improve the split strategy. Approach this as Prof. Thomas Neumann or Matthias Jasny would: apply every low-level technique available, leave nothing on the table. Focus on the bottlenecks identified in the tracing output, but also reconsider the overall approach if the remaining gap is large.
Always check ## Helper Section for optimization techinique.
When discussion, think in Prof. Thomas Neumann or Matthias Jasny way.

## Goal

Use runtime information collected during node-based split subquery execution to improve
JIT compilation — either faster compilation or faster execution of subsequent subqueries.
The node-based split executes subqueries iteratively: each iteration produces a temp table
whose actual cardinality and data are available before the next subquery compiles/runs.

---

## Current Performance (DuckDB query-jit, node-based, jitcache=full, LLVM)

| Rank | Query | Median (ms) | Pattern |
|------|-------|-------------|---------|
| 1 | 16b | 181.6 | 8-way join on cast_info/movie_companies/etc |
| 2 | 8c | 159.1 | Multi-join with large fact scans |
| 3 | 10c | 122.1 | 7-way join, LIKE '%(producer)%' on cast_info.note |
| 4 | 9d | 111.2 | Multi-join with string predicates |
| 5 | 7c | 97.1 | Join chain with dimension lookups |
| 6 | 17f | 97.1 | Multi-join with keyword filters |
| 7 | 19d | 95.4 | Large probe chain |
| 8 | 17e | 75.3 | Multi-join |
| 9 | 6d | 70.9 | Join with company filters |
| 10 | 6f | 69.5 | Similar to 6d |

DSB (DuckDB query-jit, node-based, jitcache=full, LLVM):

| Rank | Query | Median (ms) |
|------|-------|-------------|
| 1 | query050_spj_0 | 323.9 |
| 2 | query102_0 | 155.6 |
| 3 | query102_spj_0 | 143.3 |
| 4 | query085_0 | 28.1 |

PostgreSQL query-jit is 20-100× slower (10c=27s) due to mandatory row materialization
via COPY FROM STDIN to PG temp tables — an architectural constraint, not a JIT issue.
See "Verified Assumptions" section for details.

---

## Analysis: Runtime Information Available at Subquery Boundaries

In node-based split, the splitter loop (`ir_query_splitter.cpp:1236`) iterates:
```
while (!splitter->IsComplete()) {
  extract sub-IR → compile (JIT) → execute → create temp table → next iteration
}
```

After each iteration, the middleware has:

1. **Actual temp table cardinality** — stored via `override_cardinality` in the DuckDB
   catalog (`duckdb_adapter.cpp:100-108`), fed to DuckDB's optimizer for the next subquery.
2. **Actual data in FlatTable / QjitTable** — the temp table's full columnar data is
   available in memory (`qjit_temps_` map in `duckdb_adapter.cpp`).
3. **Hash table build statistics** — after `QjitHashTable::Finalize()`, the HT has:
   `NumEntries()`, `Key0Min()`, `Key0Max()`, `DirSize()`.
4. **Block statistics** — per-2048-row min/max already computed by `GetBlockStats()`.
5. **Build-side PK metadata** — `FlatTable::dense_pk`, `FlatTable::pk_to_row`,
   `FlatTable::max_pk` are available from the storage plan.

**Key insight:** All of items 3-5 are available BEFORE the probe-side morsel body
compiles. The query-JIT currently ignores items 2, 3 (partially uses), and 5 entirely.

---

## Optimization Opportunities (Ordered by Expected Impact)

All optimizations below use runtime information from prior subquery iterations or from
the storage plan to improve JIT code generation. Each is self-contained and can be
implemented independently.

---

### OPT-1: Dense Array Joins (Replace Hash Table with Direct-Addressed Array)

**Category:** Runtime info → execution time improvement
**Expected speedup:** 2–5× on join-heavy queries (16b, 8c, 7c, 10c)
**Complexity:** Medium

#### What runtime info is used

The `FlatTable` for each base table carries `dense_pk` (bool) and `pk_to_row`
(int→row mapping). These are computed at storage-plan load time
(`storage_plan.cpp:365-399`). For node-based split, when a build-side table has a dense
integer PK, the query-JIT can replace the chained hash table with a direct-addressed
array: `row_ptr = array[fk_value]`.

#### Why it helps

Current query-JIT always builds a `QjitHashTable` (chained, bloom-tagged) for every
join. Each probe requires: Murmur hash (5 multiplies+xors), directory load (likely L3
miss), bloom-tag check, chain walk with pointer chasing, key comparison. For PK-FK
equi-joins against dimension tables with dense integer PKs, a direct array lookup
eliminates ALL of these — just a bounds check + single array dereference.

IMDB tables with integer PKs (`id integer NOT NULL PRIMARY KEY`): name, aka_name,
kind_type, title, aka_title, char_name, role_type, cast_info, comp_cast_type,
company_name, company_type, complete_cast, info_type, keyword, link_type,
movie_companies, movie_info, movie_info_idx, movie_keyword, movie_link, person_info.

The kernel already implements this pattern via `KernelJoinStep::use_direct_pk`
(`sub_query_plan.cpp:1555-1568`). Query-JIT does NOT use it.

#### Implementation plan

1. **Detect eligibility in `BuildExecutionSteps`** (`query_jit_steps.cpp`):
   - When a probe has a single integer key, and the build-side source is a base table
     (not temp) whose FlatTable has `dense_pk=true` or `!pk_to_row.empty()`, mark the
     step's probe op as `DenseProbe` instead of `Probe`.
   - Need to pass FlatTable metadata into `BuildExecutionSteps` (currently it only sees IR).
   - Add a new `QjitStepOp::Kind::DenseProbe` variant.

2. **New `QjitHtDesc` field**: `bool dense_eligible = false; bool dense_pk = false;
   int32_t dense_max_pk = -1;`. Set during step planning.

3. **Codegen in `CompileQuerySteps`** (`ir_to_llvm.cpp:~9060`):
   - For `DenseProbe`: instead of hash+directory+chain, emit:
     ```llvm
     %key = load i32, col_data[key_col][row]
     %in_bounds = icmp ult %key, %dense_max_pk+1
     br i1 %in_bounds, %lookup, %miss
     %lookup:
       %row_ptr = load i8*, dense_array[%key]
       %not_null = icmp ne %row_ptr, null
       br i1 %not_null, %match, %miss
     ```
   - Dense array is built at HT-build time: `dense_rows[pk] = entry_row_ptr`.
   - For `dense_pk=true`: `row_in_table = key - 1` (no array needed, arithmetic).
   - For `dense_pk=false`: use `pk_to_row` to map `key → row`, then load payloads
     directly from FlatTable columns.

4. **Runtime**: Add a `QjitDenseArray` class (simpler than QjitHashTable):
   ```cpp
   struct QjitDenseArray {
     uint8_t **rows;  // rows[pk_value] = pointer to build-side row, or nullptr
     int32_t max_pk;
   };
   ```
   Or, for `dense_pk` tables: skip the array entirely and load directly from FlatTable
   column arrays using `row = key - 1`.

5. **Executor changes**: In `QjitExecutor::Run`, when a HT descriptor is marked
   `dense_eligible`, allocate `QjitDenseArray` instead of `QjitHashTable`. The build
   morsel writes `rows[key] = row_ptr`. The probe morsel does array lookup.

#### Key files to modify
- `include/qjit/query_jit_steps.h` — add DenseProbe kind, dense fields to QjitHtDesc
- `src/qjit/query_jit_steps.cpp` — detection logic in Decompose()
- `src/jit/ir_to_llvm.cpp` — codegen in CompileQuerySteps (~line 9060)
- `include/qjit/query_jit_runtime.h` — QjitDenseArray struct
- `src/qjit/query_jit_executor.cpp` — allocate dense array, pass to context
- `include/qjit/query_jit_executor.h` — accept FlatTable metadata

#### Verification
- Build: `cmake --build build_release -j$(nproc)`
- Correctness: `bash measure/correctness_test_job_duckdb.sh`
- Performance: `bash measure/measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all full off llvm`

#### Queries to verify (run in IMDB DuckDB database)
```sql
-- Check which tables have dense PKs:
SELECT 'title' AS tbl, COUNT(*) AS n, MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END AS pk_type
FROM title
UNION ALL
SELECT 'name', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END
FROM name
UNION ALL
SELECT 'cast_info', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END
FROM cast_info
UNION ALL
SELECT 'keyword', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END
FROM keyword
UNION ALL
SELECT 'company_name', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END
FROM company_name
UNION ALL
SELECT 'role_type', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END
FROM role_type
UNION ALL
SELECT 'company_type', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END
FROM company_type
UNION ALL
SELECT 'info_type', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END
FROM info_type
UNION ALL
SELECT 'kind_type', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END
FROM kind_type
UNION ALL
SELECT 'char_name', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END
FROM char_name
UNION ALL
SELECT 'movie_companies', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END
FROM movie_companies
UNION ALL
SELECT 'movie_keyword', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END
FROM movie_keyword;
```

---

### OPT-2: Probe-Side Prefetching in Query-JIT Morsel Body

**Category:** Runtime info → execution time improvement
**Expected speedup:** 1.3–2× on large-HT probes (16b, 8c, 19d)
**Complexity:** Medium

#### What runtime info is used

After each build step, `QjitHashTable::Finalize()` produces `NumEntries()` and
`DirSize()`. These tell the probe morsel whether the HT fits in L2 (≤256KB) or is an
L3/memory-resident structure. For large HTs, software prefetching hides the ~40ns L3
latency per probe.

#### Why it helps

Pipeline-JIT already has probe prefetching (`CompileFilterProbeProjectFusion`,
`ir_to_llvm.cpp:7524-7587`) with configurable `prefetch_entry_distance` (default 24)
and `prefetch_row_distance` (default 12). **Query-JIT's morsel body has ZERO
prefetching** — verified by searching `CompileQuerySteps` (lines 8877-10200) for
`prefetch`: no matches. The probe chain walk (`ir_to_llvm.cpp:~9838`) is purely scalar
with no look-ahead.

The ROF (Reorder-Filter) two-phase scan exists in query-JIT but Phase 2 also has
no prefetching — it iterates the selection buffer serially with no look-ahead.

#### Implementation plan

1. **Group-prefetch probe** (Redshift circular-buffer style):
   - In the morsel body, after computing hash for a probe row, issue
     `llvm.prefetch` for `dir[hash & mask]` N rows ahead.
   - Requires buffering: accumulate (row_idx, hash) pairs in a small circular
     buffer (e.g., 32 entries = 512 bytes, fits L1). Process buffer entries
     whose prefetch has had time to complete.

2. **Two-level prefetch** (matching pipeline-JIT's approach):
   - Stage 1: prefetch HT directory entry (entry distance = 24)
   - Stage 2: after loading the directory entry, prefetch the build-side row
     pointer (row distance = 12)

3. **Conditional activation**: Only emit prefetch when `DirSize() * 8 > L2_SIZE`
   (the HT directory doesn't fit in L2). For small HTs, prefetch overhead exceeds
   benefit.

4. **Runtime check**: The morsel entry block can load `qjit_ht_entries(ht_ptr)` and
   branch: if entries < threshold (e.g., 32K), skip prefetch path.

#### Key files to modify
- `src/jit/ir_to_llvm.cpp` — CompileQuerySteps, around probe chain walk (~line 9838)
- Reuse existing `prefetch_entry_distance_` and `prefetch_row_distance_` settings

#### Verification
- Same as OPT-1

---

### OPT-3: Extended Zonemaps (Filter Predicates + Multi-Key Block Skip)

**Category:** Runtime info → execution time improvement
**Expected speedup:** 1.2–1.5× on filtered large-table scans (10c, 9d, 16b)
**Complexity:** Low-Medium

#### What runtime info is used

`GetBlockStats()` (`query_jit_executor.cpp:21-50`) already computes per-2048-row
min/max for one column. This is used for block-skip on key0 of guards[0]. The runtime
info is: after the storage plan loads a FlatTable, the actual data distribution
(min/max per block) is known.

#### Why it helps

Current block-skip is limited:
1. Only key0 of the first guard — multi-key joins and secondary probes have no zonemap.
2. Only for probe-side join keys — filter predicates like `t.production_year > 1990`
   (query 10c) are NOT block-skipped, even though they could eliminate thousands of
   blocks.

Data in FlatTable IS ordered by PK (`ORDER BY id` in `storage_plan.cpp:343-347`),
so consecutive blocks contain consecutive ID ranges. Block-level min/max on ordered
integer columns has near-perfect selectivity.

#### Implementation plan

1. **Filter-predicate block skip**: In `BuildExecutionSteps`, when a filter op uses
   a `VarConstComparison` on an INT32 column with `<, >, <=, >=, BETWEEN`, mark the
   step for filter-level block skip. Compute min/max stats for that column.

2. **Multi-key block skip**: For each guarded probe, compute block stats for all
   join key columns (not just key0). Emit a compound range check: skip block if
   ANY key's block range is disjoint from the build-side key range.

3. **Block stats for temp tables**: Currently `GetBlockStats` only works for
   FlatTable base tables. For QjitTable temps produced by prior iterations,
   compute block stats lazily on first access.

#### Key files to modify
- `src/qjit/query_jit_executor.cpp` — extend GetBlockStats for filter columns
- `src/qjit/query_jit_steps.cpp` — mark filter predicates for block skip
- `src/jit/ir_to_llvm.cpp` — emit block-level filter checks in CompileQuerySteps

#### Verification
- Same as OPT-1

---

### OPT-4: SIMD Vectorized Filter in Query-JIT Morsel Body

**Category:** Runtime info → execution time improvement (uses data type info)
**Expected speedup:** 1.1–1.3× on numeric-filter queries (10c, 9d)
**Complexity:** Medium

#### What runtime info is used

Column data types are known from the IR and FlatTable schema. For integer columns,
SIMD comparison can process 4 (SSE2), 8 (AVX2), or 16 (AVX-512) rows per instruction.

#### Why it helps

Query-JIT morsel filter is fully scalar — verified: `CompileQuerySteps` uses
`EmitShortCircuitFilter` for all filter ops (scalar per-row evaluation).
`BuildFilterFunctionSIMD` (line 2887) exists but is only used in pipeline-JIT.
The ROF Phase 1 does scalar guard+filter+hash, not SIMD.

For queries with numeric predicates on large tables (10c: `production_year > 1990`
on 2.5M-row title), vectorizing the filter scan can process 8-16 rows per cycle
instead of 1.

#### Implementation plan

1. Port `BuildFilterFunctionSIMD` logic into the query-JIT morsel body:
   - In ROF Phase 1, replace scalar guard+filter with SIMD vectorized version.
   - Process `vec_width` rows at a time, compute a bitmask of passing rows.

2. For non-ROF path: add a SIMD pre-filter phase before the scalar row loop.
   Evaluate cheap numeric predicates in SIMD, produce a selection vector,
   then iterate only selected rows for expensive ops (string predicates, probes).

#### Key files to modify
- `src/jit/ir_to_llvm.cpp` — CompileQuerySteps, ROF Phase 1 and scalar row loop
- Reuse existing SIMD infrastructure (`vec_width`, `SimdISA`, `BuildFeatureStr`)

#### Verification
- Same as OPT-1

---

### OPT-5: Specialize Compilation Based on Actual HT Size

**Category:** Runtime info → compilation time + execution time improvement
**Expected speedup:** 1.05–1.2× on small-HT joins
**Complexity:** Low

#### What runtime info is used

After a build step executes, `QjitHashTable::NumEntries()` is known. This is
available before the probe-side morsel compiles (in speculative JIT or template
recompile mode).

#### Why it helps

When the build-side HT is very small (e.g., < 16 entries after filtering a dimension
table like `kind_type` with 7 rows), the chained HT probe is overkill. The JIT could:

1. **Inline the HT as constants**: For ≤8 entries, emit a linear scan over
   compile-time-constant key values (no HT data structure at all).
2. **Skip bloom-tag check**: For very small HTs, the bloom tag adds cost but no
   selectivity benefit.
3. **Skip the hash function**: For single-key integer joins with ≤16 entries,
   emit a switch/case or binary search on the key values.

#### Implementation plan

1. In the speculative/template-recompile path, when the probe morsel compiles,
   check `QjitHashTable::NumEntries()`.
2. If entries ≤ 8: emit inline linear scan (load key constants from params buffer).
3. If entries ≤ 64: emit a compact open-addressed HT with known size (no resize).
4. Otherwise: use the standard chained HT probe.

This requires the speculative-JIT or template-cache-mode-2 path where recompilation
is expected. It does NOT work with full object cache (mode 3) since the cached code
must handle arbitrary HT sizes.

#### Key files to modify
- `src/jit/ir_to_llvm.cpp` — CompileQuerySteps, probe codegen
- `src/qjit/query_jit_steps.h` — add size hint to QjitHtDesc

#### Verification
- Same as OPT-1

---

### OPT-6: Inline String Comparison Functions

**Category:** Compile-time optimization (code quality)
**Expected speedup:** 1.05–1.1× on LIKE-heavy queries (10c, 9d)
**Complexity:** Low

#### What runtime info is used

Not strictly runtime info, but string function implementations (`aqp_like_match`,
`aqp_str_eq`, `aqp_str_cmp`) are currently external C symbols resolved by ORC JIT
at link time. They are not inlined by the LLVM optimizer because they live outside
the LLVM module.

#### Why it helps

By emitting string functions as LLVM IR within the compiled module (or by providing
LLVM bitcode for them), the optimizer can:
- Eliminate redundant null checks inside the string function
- Inline short-string comparisons (strings ≤ 12 bytes can be compared with two i64 loads)
- Specialize LIKE patterns at compile time (prefix/suffix/contains are simpler than
  general multi-segment LIKE)

Note: the LIKE pattern classification (`ClassifyLikePatternEx`) already exists and
generates specialized code paths. This optimization is about giving LLVM visibility
into the called function bodies.

#### Implementation plan

1. Write LLVM IR (or C compiled to bitcode) for `aqp_str_eq`, `aqp_str_cmp`,
   and `aqp_like_contains` / `aqp_like_prefix` / `aqp_like_suffix`.
2. Load the bitcode module and link it into the query module before optimization.
3. Mark these functions with `alwaysinline` attribute.

#### Key files to modify
- `src/jit/ir_to_llvm.cpp` — module setup, string function declarations
- New file: `src/jit/string_builtins.ll` or `string_builtins.cpp` compiled to bitcode

#### Verification
- Same as OPT-1

---

## Verified Assumptions (Confirmed by Source Code Reading)

### 1. FlatTable data IS ordered by PK ✓
`storage_plan.cpp:343-347`: every column query includes `ORDER BY id` if an "id" column
exists. This makes zonemaps effective — consecutive blocks contain consecutive ID ranges.
Same for PostgreSQL path (`storage_plan.cpp:501-506`). Binary cache preserves ordering.

### 2. dense_pk and pk_to_row exist but are NOT used by query-JIT ✓
- `dense_pk` condition: PKs must be exactly [1, row_count], checked at
  `storage_plan.cpp:376-389`. Does NOT handle min_id ≠ 1.
- `pk_to_row` is allocated for non-dense tables (`storage_plan.cpp:391-398`).
- **Usage**: only in `KernelJoinStep::use_direct_pk` (`sub_query_plan.cpp:1555-1568`).
  Zero usage in `src/qjit/` or `src/jit/`. This is the gap OPT-1 fills.

### 3. Build side is determined by DuckDB's physical planner ✓
`AnnotateBuildSides()` (`duckdb_adapter.cpp:3070-3151`) matches DuckDB's physical plan
join sides to IR join children by table-index set equality. Sets `build_child = 0 or 1`.
For large tables like cast_info (36M rows), DuckDB typically makes them the probe side.
The query-JIT rejects joins with `build_child = -1` ("join:build-side-unannotated").

### 4. PostgreSQL query-jit slowness is NOT a JIT execution problem ✓
PG query-jit DOES execute in the middleware using FlatTables (`postgres_adapter.cpp:292-432`),
but every temp table must be materialized to PG via `MaterializeQjitTempToPostgreSQL()`
(`postgres_adapter.cpp:1318-1428`): row-by-row COPY FROM STDIN with string escaping +
1 syscall per row. This is 20-100× slower than DuckDB's in-memory QjitTable passing.
**Optimizing query-JIT probe/scan won't fix PG's bottleneck.**

### 5. Query-JIT morsel body has NO prefetching ✓
Confirmed: `CompileQuerySteps` (lines 8877-10200) contains zero `prefetch` calls.
Pipeline-JIT has prefetching in `CompileFilterProbeProjectFusion` (line 7524-7587).
ROF Phase 2 also has no prefetching — it iterates the selection buffer serially.

### 6. Query-JIT morsel body uses scalar filters only ✓
`BuildFilterFunctionSIMD` (line 2887) is only called from `CompileFilter` and
`CompileExpr` — both pipeline-JIT paths. Query-JIT's `CompileQuerySteps` uses
`EmitShortCircuitFilter` (scalar). ROF Phase 1 is also scalar.

---

## Items Needing Verification by Running Queries

Run the following SQL in the IMDB DuckDB database to confirm dense-PK status for all
tables. This determines which joins qualify for OPT-1 (dense array):

```sql
-- Run in: /home/pei/Project/duckdb/measure/imdb.db
-- Expected: most dimension tables (kind_type, role_type, company_type, info_type,
-- link_type, comp_cast_type) are DENSE. Large fact tables (cast_info, movie_companies,
-- movie_keyword) may or may not be dense.

SELECT 'title' AS tbl, COUNT(*) AS n, MIN(id) AS min_id, MAX(id) AS max_id,
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END AS pk_type
FROM title
UNION ALL SELECT 'name', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM name
UNION ALL SELECT 'cast_info', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM cast_info
UNION ALL SELECT 'keyword', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM keyword
UNION ALL SELECT 'company_name', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM company_name
UNION ALL SELECT 'role_type', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM role_type
UNION ALL SELECT 'company_type', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM company_type
UNION ALL SELECT 'info_type', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM info_type
UNION ALL SELECT 'kind_type', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM kind_type
UNION ALL SELECT 'char_name', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM char_name
UNION ALL SELECT 'movie_companies', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM movie_companies
UNION ALL SELECT 'movie_keyword', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM movie_keyword
UNION ALL SELECT 'movie_info', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM movie_info
UNION ALL SELECT 'movie_info_idx', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM movie_info_idx
UNION ALL SELECT 'person_info', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM person_info
UNION ALL SELECT 'aka_name', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM aka_name
UNION ALL SELECT 'aka_title', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM aka_title
UNION ALL SELECT 'complete_cast', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM complete_cast
UNION ALL SELECT 'movie_link', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM movie_link
UNION ALL SELECT 'link_type', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM link_type
UNION ALL SELECT 'comp_cast_type', COUNT(*), MIN(id), MAX(id),
       CASE WHEN MIN(id)=1 AND MAX(id)=COUNT(*) THEN 'DENSE' ELSE 'SPARSE' END FROM comp_cast_type;
```

---

## Key Code Locations (Reference)

### Query-JIT Pipeline

| File | What |
|------|------|
| `src/jit/ir_to_llvm.cpp:8877-10224` | `CompileQuerySteps()` — emits one morsel body per step |
| `src/jit/ir_to_llvm.cpp:9060-9079` | `can_skip_ehash` — skip-hash-cmp decision |
| `src/jit/ir_to_llvm.cpp:9170-9188` | Guard range + membership gate (loop-invariant) |
| `src/jit/ir_to_llvm.cpp:9262-9284` | ROF decision logic |
| `src/jit/ir_to_llvm.cpp:9295-9496` | ROF Phase 1+2 implementation |
| `src/jit/ir_to_llvm.cpp:9498-9575` | Scalar row loop with block skip |
| `src/jit/ir_to_llvm.cpp:9838-9872` | Probe chain walk (no prefetch) |
| `src/jit/ir_to_llvm.cpp:2887-3300` | `BuildFilterFunctionSIMD` (pipeline-JIT only) |
| `src/jit/ir_to_llvm.cpp:7524-7587` | Pipeline-JIT prefetch (entry+row distances) |

### Query-JIT Runtime

| File | What |
|------|------|
| `include/qjit/query_jit_runtime.h` | QjitHashTable, QjitTable, QjitAggState, bloom helpers |
| `include/qjit/query_jit_abi.h` | C ABI structs (QjitQueryContext, QjitTableView, QjitColView) |
| `include/qjit/query_jit_steps.h` | QjitQueryPlan, QjitStep, QjitStepOp, QjitHtDesc |
| `src/qjit/query_jit_steps.cpp` | BuildExecutionSteps — strict IR → step plan builder |
| `src/qjit/query_jit_executor.cpp` | QjitExecutor::Run, ResolveSource, GetBlockStats |

### Storage Plan

| File | What |
|------|------|
| `include/storage/flat_table.h:103-122` | FlatTable struct (dense_pk, pk_to_row, max_pk) |
| `include/storage/storage_plan.h` | StoragePlan class (tables, CSR, sorted/inverted indices) |
| `src/storage/storage_plan.cpp:343-399` | LoadFromDuckDB with ORDER BY id + PK detection |
| `include/storage/csr_index.h` | CSRIndex (FK→PK row mapping) |
| `include/storage/inverted_index.h` | InvertedIndex (dim_pk→target_pk via bridge) |

### Kernel (Reference for Dense PK Pattern)

| File | What |
|------|------|
| `include/kernel/sub_query_plan.h:135-154` | KernelJoinStep with use_direct_pk |
| `src/kernel/sub_query_plan.cpp:1555-1568` | Dense PK lookup execution (row = key - 1) |
| `src/kernel/sub_query_plan.cpp:1475-1479` | Dense PK planning logic |

### Adapters and Splitter

| File | What |
|------|------|
| `src/adapters/duckdb_adapter.cpp:3070-3151` | AnnotateBuildSides (physical→IR matching) |
| `src/adapters/duckdb_adapter.cpp:1470-1940` | ExecuteSQLandCreateTempTable (query-jit path) |
| `src/adapters/postgres_adapter.cpp:1318-1428` | MaterializeQjitTempToPostgreSQL (PG bottleneck) |
| `src/split/ir_query_splitter.cpp:1096-1271` | ExecuteSplitLoop (main node-based iteration) |

### SimplestIR

| File | What |
|------|------|
| `third_party/IR_SQL_Converter/inc/simplest_ir.h` | All IR node class definitions (1633 lines) |
| `third_party/IR_SQL_Converter/src/duckdb_plan_to_ir.cpp` | DuckDB logical plan → SimplestIR |
| `third_party/IR_SQL_Converter/src/ir_to_sql.cpp` | SimplestIR → SQL string |
| `third_party/IR_SQL_Converter/inc/cpp_interface.h` | Public API: ConvertDuckDBPlanToIR, ConvertIRToSQL |

## Repositories

- **AQP Middleware**: `/home/pei/Project/AQP_middleware` (this repo)
- **DuckDB (patched)**: `/home/pei/Project/duckdb`
- **JOB queries**: `/home/pei/Project/benchmarks/JOB4AQP/` and `/home/pei/Project/duckdb/benchmark/imdb_plan_cost/queries/`
- **JOB schema**: `/home/pei/Project/benchmarks/JOB4AQP/schema.sql`
- **DuckDB database**: `/home/pei/Project/duckdb/measure/imdb.db`

Build commands:
```bash
# Middleware (debug / release)
cd /home/pei/Project/AQP_middleware && cmake --build build_debug -j$(nproc)
cd /home/pei/Project/AQP_middleware && cmake --build build_release -j$(nproc)
# DuckDB — the middleware links the PREBUILT libduckdb.so from
# ${DUCKDB_ROOT}/build/release/src (find_library + dynamic link).
cd /home/pei/Project/duckdb && cmake --build build/release -j 16
```

Build hazards:
- build_debug and build_release share `lib/*.a` outputs. If a debug build
  poisons the release archive (ASan), rebuild with
  `--target IR_SQL_Converter_C_static --clean-first`.

## Verification Workflow

### Step 2: Single query end-to-end

Take a simple JOB query that related to our changes (e.g., 1a — single join), run with
`./build_release/aqp_middleware ...` (check reference in measure/run_aqp.sh job). Compare result to DuckDB golden output.

### Step 4: Full JOB correctness

Run all 113 JOB queries with the correct flags, compare against golden files:
- `measure/duckdb_job_no-split_golden.txt`
- `measure/duckdb_job_node-based_golden.txt`

For more flags, check measure/correctness_test.sh.

### Analysis scripts (measure/*.py)
- `tune_per_subquery.py [split]` — pick best config per (query, sub-query)
- `show_all_configs.py [split]` — summary table across all configs
- `find_top_queries.py [path] [--top=N]` — rank queries by slowest median

---

## Implementation Status: 


### Verification

- Build: passes (release)
- Correctness: all 113 JOB queries pass for all related configs
- CSV format: unchanged (no new columns), parseable by /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py
- No new print statements; existing LIKE debug trace guarded by `#ifndef NDEBUG`
- Check if need changes to measure/*.sh or measure/*.py
- Performance (top 15 worst LIKE queries): measure/measure_breakdown_time_aqp.sh job with correct configs

---

## Helper Section

### Measurement commands
```bash
# Top queries for DuckDB query-jit (cached LLVM, node-based):
cd /home/pei/Project/AQP_middleware/measure
python3 find_top_queries.py --top=15 job_result/duckdb_node-based_query_none_jitcache_full_llvm_breakdown_time_log.csv

# Run a single measurement config (12 args):
bash measure/measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all full off llvm

# Correctness test:
bash measure/correctness_test_job_duckdb.sh
```

### Key CLI flags
```
--jit-level=query        # Query-JIT mode
--split=node-based       # Node-based split
--jit-cache=full         # Persistent disk cache
--compile-mode=llvm      # Full LLVM O2 (alternatives: fastisel, tpde)
--jit-skip-hash-cmp=all  # Skip hash equality for integer keys
--storage-plan           # Load FlatTable column arrays
--storage-cache=<path>   # Binary cache for storage plan
--timing --repeat=15     # 5 warmup + 10 measured iterations
```
