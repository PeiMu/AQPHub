You are a DBMS JIT compilation expert. Your task is to iteratively improve execution time (ignore compilation time) for the JOB benchmark through JIT-related code changes (the status of the current iteration is in section "## Current Performance"). Especially with split strategy, e.g., node-based split, where we can collect real runtime information to help with JIT code generation and/or better plan. Find whatever information helps, then check how to collect them. Targeting speedup 10x of the query wall runtime (ignore compilation time) with either level of JIT with node-based split, until no optimization can be found or heavy wall time queries take only 1 ms. You can also improve the split strategy. Approach this as Prof. Thomas Neumann or Matthias Jasny would: apply every low-level technique available, leave nothing on the table. Focus on the bottlenecks identified in the tracing output, but also reconsider the overall approach if the remaining gap is large.
Always check ## Helper Section for optimization.

## Codebase
- AQP middleware: /home/pei/Project/AQP_middleware (split execution, JIT compilation, IR-to-LLVM)
- DuckDB (forked): /home/pei/Project/duckdb (JIT dispatch in physical operators)
- JIT code generation: /home/pei/Project/AQP_middleware/src/jit/ir_to_llvm.cpp
- JIT dispatch (DuckDB side): physical_filter.cpp, physical_hash_join.cpp, physical_table_scan.cpp, physical_projection.cpp, physical_ungrouped_aggregate.cpp in /home/pei/Project/duckdb/src/execution/operator/
- JIT ABI: /home/pei/Project/duckdb/src/include/duckdb/execution/aqp_jit.hpp
- Split execution loop: /home/pei/Project/AQP_middleware/src/split/ir_query_splitter.cpp
- DuckDB adapter (JIT registration): /home/pei/Project/AQP_middleware/src/adapters/duckdb_adapter.cpp
- Storage plan design: /home/pei/Project/AQP_middleware/storage_plan_design.md
- Storage plan code (NEW): /home/pei/Project/AQP_middleware/src/storage/ (flat tables, CSR indexes, sorted indices, dimension cache, kernels)
- Benchmark queries: /home/pei/Project/benchmarks/imdb_job-postgres/queries/ (113 SQL files)
- Reference implementations: /home/pei/Project/BespokeOLAP/output/ (compiled queries), /home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries/ (generated queries)

## Current Performance (already measured, don't re-derive)

Measurement data is in /home/pei/Project/AQP_middleware/measure/job_result/. We drop the first 5 runs as warm up and take average of the other 10 runs. The measurement uses `--repeat=15` (in-process iteration, single DuckDB connection).

## Constraints
- Only modify AQP middleware code (JIT in src/jit/, storage plan in src/storage/, split strategies in src/split/, adapters in src/adapters/).
- Do NOT touch native DuckDB or PostgreSQL code, except for existing JIT dispatch code in DuckDB operators (physical_filter.cpp, physical_hash_join.cpp, etc.).
- Native DuckDB/PostgreSQL must NOT be affected by storage plan changes — all auxiliary structures live in middleware memory.
- Must return identical results to the original query.
- Ignore compilation time -- only execution time matters.

## Workflow Per Iteration

### 0. Profile the bottleneck
Use `perf_analysis.md` as the step-by-step guide. First, run `python3 measure/find_top_queries.py` to identify the top-10 heaviest queries from the latest `breakdown_time_log.csv`. Always check the heaviest queries in jit_optimization_claude.md. Profile both baseline (none-split/none-jit) and best JIT config on the same queries: the gap shows where split+JIT overhead is; then also profile best JIT alone to find the remaining bottleneck to optimize. Think fundamentally what is the correct optimization technique for the current bottleneck. Think in the Umbra way and in the Thomas Neumann way.

**Profiling tool**:
1. **`perf stat`** — First step. Classifies bottleneck type (compute vs memory bound) via IPC, L1/LLC cache miss rates, branch misprediction, CPU utilization.
2. **`perf record` + `perf report`** — Second step. Shows function-level CPU hotspots. Directly answers "which operator is the bottleneck."
3. **DuckDB EXPLAIN ANALYZE** — Operator-level cardinality + time. Use selectively (adds overhead).
4. **`perf record -e cache-misses` / `perf mem`** — Only if `perf stat` shows memory-bound. Pinpoints which data structures cause cache misses.

### 1. Read relevant source code for the optimization you're implementing

### 2. Make the code change and write unit gtests in unit_test/ dir. If add new modules or code changes related to breakdown timer, decide a reasonable timer position and ask user to confirm. Confirm analyze_middleware_breakdown, analyze_none_split_breakdown in /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py still work. Otherwise, update the analyze_middleware_breakdown and/or analyze_none_split_breakdown, and confirm with the user.

### 3. Build and quick-test
Build:
```bash
cd /home/pei/Project/AQP_middleware/build_release && make -j12
# If DuckDB files changed:
cd /home/pei/Project/duckdb/build/release && make -j12
```
Run the current heaviest query (~1 second):
```bash
cd /home/pei/Project/AQP_middleware/measure
../build_release/aqp_middleware \
  --engine=duckdb \
  --db="/home/pei/Project/duckdb/measure/imdb.db" "" \
  --schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
  --fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
  --split="node-based" --no-analyze \
  --jit-level=pipeline --jit-opt=o1 --jit-simd=none \
  --timing \
  /home/pei/Project/benchmarks/imdb_job-postgres/queries/{query_id}.sql
```
Check time_log.csv for per-phase timing. Verify by unit tests first, then measure performance on 2-3 target queries. For breakdown csv, use analyze_middleware_breakdown, analyze_none_split_breakdown in /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py to parse it.

### 4. Correctness check
Quick (~2 min):
```bash
cd /home/pei/Project/AQP_middleware/measure
bash run_job.sh duckdb node-based pipeline o1 none
filter='grep -v "^Running\|^==\|^Execution\|^$\|^waiting\|^server\|^ANALYZ\|^duckdb runs:\|^(base)"'
diff <(eval $filter job_result/aqp_middleware_duckdb_node-based_pipeline_o1_none_job.txt) \
     <(eval $filter duckdb_job_node-based_golden.txt)
```
Full:
```bash
cd /home/pei/Project/AQP_middleware/measure
bash ./correctness_test.sh
```

### 5. If faster: full breakdown measurement (~28 min, only for final validation)
Do NOT do ANYTHING when running performance measurement to avoid noise.
```bash
cd /home/pei/Project/AQP_middleware/measure
bash measure_breakdown_time_job.sh duckdb node-based pipeline o1 none
```

### 6. If slower or neutral
Analyze why, check the slowdown queries if it is noise, revert or adjust (fundamentally change direction, algorithm, or guard by condition), try again.

### 7. Mixed results (some queries speed up, others slow down)
Check what's different. Fix fundamentally or at least guard by condition of the slowdown query's pattern.

### 8. Compare with baselines
Compare JIT with split strategy (e.g., node-based) speedup or slowdown vs none-split+none-JIT and vs the last version of JIT. Find the fundamental reason.

### 9. Summarize and update
Report: what changed, which queries improved/regressed, by how much, net effect on total JOB time. Update ## Current Status in this file. Then compact the conversation within Claude Code and continue the next step.

## Helper
- Hardware feature: {"cpu_cores": 12, "simd": "avx2", "total_memory_gb": 63, "disk_type": "hdd", "l3_cache_mb": 12}. You can run any tool to detect more on this server.
- Follow expert knowledge in /home/pei/Project/BespokeOLAP/conversations/prompts/expert_knowledge.txt and /home/pei/Project/GenDB/src/gendb/agents/code-generator/prompt.md and /home/pei/Project/GenDB/src/gendb/agents/query-optimizer/prompt.md
- Think fundamentally what information from the previous sub-plan execution can you use to generate better JIT code for the next sub-plan. Then check how to get this information from the codebase.
- We care about the whole JOB benchmark, so prioritize the heaviest queries with the heaviest bottlenecks.
- You can check well-optimized queries under /home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries and under /home/pei/Project/BespokeOLAP/output to find the gap, and learn from it.

## Current Status (Iter 28 — Lazy Loading Reverted, Back to Step 6)

See jit_optimization_claude.md for full per-iteration details.

**Performance** (in-memory, warm, avg of 10 runs — execution time only, ignoring JIT compile):

| Config                              | Execution (s) | Middleware (s) | JIT Compile (s) | Wall (s) | vs ns/nojit |
|--------------------------------------|-------------:|---------------:|----------------:|---------:|------------:|
| none-split / none-jit                |         9.44 |          0.003 |               — |     9.45 | baseline    |
| none-split / pipeline-jit (no SIMD)  |         8.86 |          0.013 |            8.79 |    17.66 | -0.59 (-6.2%) |
| none-split / pipeline-jit (auto SIMD)|         8.90 |          0.014 |            9.62 |    18.53 | -0.54 (-5.8%) |
| node-based / none-jit                |        10.17 |          0.82  |               — |    10.99 | +0.73 (+7.7%) |
| **node-based / pipeline-jit (no SIMD)**|       **6.98**|       **4.62** |         **1.19**|  **12.79**| **-2.46 (-26.1%)**|
| node-based / pipeline-jit (auto SIMD)|        22.16 |          5.05  |            1.06 |    28.26 | +12.71 (+135%) |

**Key result**: node-based/pipeline-jit execution **6,980ms** — **26.1% faster** than none-split/none-jit baseline (9,444ms). First time node-based split beats the baseline on execution time.

Auto-SIMD regression (22.16s) is a known issue — SIMD codegen has bugs on the kernel path. Use `--jit-simd=none`.

Comparison with Step 3 (Iter 25, `--csr-support=inner`): 6,980ms vs 6,437ms — 543ms gap from 213 kernel-invalid iterations (31.4%) still falling back to DuckDB.

### What JIT currently covers vs. the bottleneck

**Active JIT paths** (with `--jit-level=pipeline`):
1. FILTER → expr-level compiled filter (`expr_fns`) — dispatched in `physical_filter.cpp`
2. TABLE_SCAN → scan+filter fusion (`scan_filter_fns`) — FILTER becomes pass-through
3. FILTER → pipeline fused filter+projection (`pipeline_fns`) — row-at-a-time, skips LIKE
4. PROJECTION → zero-copy column mapping (`proj_col_maps`) — `Vector::Reference()`
5. TABLE_SCAN → Bloom filter scan push-down (`bloom_scan_filters`)
6. HASH_JOIN → JIT-gated software prefetching for probe bucket, build, BF (Iter 22)
7. HASH_JOIN → JIT-gated chain-walk prefetching in AdvancePointers (Iter 23)
8. HASH_JOIN → JIT-gated row-data prefetch before Match (Iter 24)
9. HASH_JOIN → ScanInnerJoin/ScanKeyMatches equality fast path (Iter 24, JIT-gated)
10. HASH_JOIN → Skip found_match writes for INNER/RIGHT joins (Iter 24, JIT-gated)
11. HASH_JOIN → ResolvePredicates memcpy optimization (Iter 24, JIT-gated)

**Disabled JIT paths** (compiled but never dispatched):
1. HASH_JOIN probe fusion — `if (false && ...)` at `physical_hash_join.cpp:1097`. Disabled in Iter 9: DuckDB's vectorized batch probe was 1.6% faster than scalar JIT probe.
2. HASH_JOIN Bloom filter — only consumed inside disabled probe block

**Never compiled:**
1. UNGROUPED_AGGREGATE — `DISABLE_AGG_JIT=1` (JOB uses only MIN on VARCHAR)
2. SIMD pipeline — `if (false && ...)` guard (incomplete implementation)
3. HASH_JOIN build — intentionally not JIT'd (DuckDB native is already tight)

## Next Steps

### Direction A (NEW): Auxiliary Storage Plan + Kernel Execution (high impact, est. 5-20x)

Build auxiliary in-memory data structures in the AQP middleware and execute sub-queries directly via optimized kernel functions, bypassing DuckDB's hash join and decompression overhead. **DuckDB/PostgreSQL remain completely unchanged** — they serve as data source at startup and SQL parser at query time. Native DuckDB/PostgreSQL does NOT benefit from these changes; all structures live in middleware memory. The reference code is from Bespoke (the fastest) in /home/pei/Project/BespokeOLAP/output/ and GenDB (a bit slower) in /home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries/ dirs.

**Full design**: `/home/pei/Project/AQP_middleware/storage_plan_design.md`

**Performance estimate**: 5-20x over current DuckDB, not the full 50-170x of BespokeOLAP. The gap: (1) temp table materialization between sub-queries — BespokeOLAP has zero intermediate materialization via single fused loop (cross-table bitmaps/byte-maps are part of this, see Step 7 loop fusion), (2) generic interpreter vs. per-query compiled code, (3) no dimension-partitioned flat tables. To close the gap: Step 7 sub-query loop fusion (eliminates temp materialization, biggest win) + Step 7 kernel compilation (eliminates interpreter overhead) + Direction B (fewer/smaller temps).

**Storage components** (each verified independently via breakdown measurement):
1. **Flat Column Arrays**: Decompress all base tables into plain C arrays at startup. Eliminates fsst_decompress (3-15%) and segment management overhead. Direct `array[row_id]` access. Code: `src/storage/flat_table.h/.cpp`
2. **CSR Indexes**: Compressed Sparse Row indexes on all FK columns (from `--fkeys`). O(1) FK→PK lookup, replaces hash table build+probe (InsertHashes 5-17%, AdvancePointers 6-8%). Code: `src/storage/csr_index.h/.cpp`
3. **Runtime CSR on Temp Tables**: After each sub-query, build CSR on temp result's join key. Next sub-query uses CSR lookup instead of DuckDB hash join. This is **runtime information** from split execution. Eliminates node-based split's +6.6% regression.
4. **Dimension Constants**: Cache tiny tables (<200 rows), resolve joins to constant predicates at parse time. Eliminates unnecessary joins with kind_type, info_type, etc. Code: `src/storage/dimension_cache.h/.cpp`
5. **Sorted Indices**: Sorted permutation arrays for MIN/MAX early termination. Scan in sorted order, stop at first match → O(k) instead of O(n). Code: `src/storage/sorted_index.h/.cpp`

**Execution model**: A `SubQueryPlan` struct describes each sub-query (scan table, filters, CSR join steps, projection, aggregation). A single generic `ExecuteSubQuery()` function handles all patterns in one scan loop — no separate primitive calls, no intermediate materialization within a sub-query. See `storage_plan_design.md` §Execution Model.

Sub-query pattern analysis (113 JOB queries, ~640 sub-queries from node-based split):
- 6 structural patterns cover 70% of sub-queries
- 15 structural patterns cover 97% of sub-queries
- Most common: `1base+1temp` (26%), `1base+1dim` (12%), `2base` (10%), `1base+1temp+FINAL` (9%)
- Multi-base patterns (`2base`, `3base`, `4base`) = 16% — these fall back to DuckDB or require hash join

Code: `src/storage/sub_query_plan.h/.cpp` (plan struct + executor)

**Interpreter vs. compiled kernel**: Steps 1-6 use a generic interpreter (`ExecuteSubQueryPlan`) that evaluates `SubQueryPlan` at runtime. Step 7 replaces the interpreter with JIT-compiled native code for each sub-query, eliminating per-row interpretation overhead. The interpreter is sufficient for v1 to validate correctness and measure the benefit of storage structures alone. Step 7 then isolates the additional gain from compilation.

**Interaction with Direction B (split strategy)**: A better split strategy would shift patterns toward more `1base+N_temp` and fewer multi-base patterns (`2base`/`3base`/`4base`). This is strictly better for the executor — more CSR-joinable, fewer DuckDB fallbacks. In particular, if we improve the split strategy to always start with a single filtered dimension→base join, every sub-query becomes `1base+N_temp` or `1dim+N_temp`, which are pure CSR-joinable. Direction B is NOT a prerequisite — implement storage plan first, then Direction B further improves it. After Direction B, verify if new sub-query patterns need executor support.

**Implementation order**: Step-by-step, with breakdown measurement after each component to verify its individual effect:
- Step 1: Flat Column Arrays + basic ScanFilter → measure scan speedup (**DONE**)
- Step 2: CSR Indexes on base tables + CSRSemiJoin → measure single FK join speedup (**DONE**)
- Step 3: Runtime CSR on temp tables + SubQueryPlan executor → measure node-based split execution (target: 16b, 8c) (**DONE** — exe 9.60s → 6.44s, -33%. mw 1.01s → 2.60s due to FlatTable loading + runtime CSR build.)
- Step 3.5: Filter support in kernel (**DONE** — 6/6 correctness pass, kernel coverage 0% → 58.7% (304/518 iterations). Supports =, !=, <, >, <=, >=, IN, NOT IN, IS NULL/NOT NULL, AND/OR/NOT. LIKE/BETWEEN → DuckDB fallback. pk_to_row for inner-join bitset. Timing CSV format unified: 6 cols/iter for both kernel and DuckDB paths. Fallback overhead logged to `fallback_waste_time.csv`. Code moved from `storage/` to `kernel/` dir.)
- Step 4: Dimension Constants → cache tiny tables (<200 rows), resolve joins to constant predicates at kernel analysis time. (**DONE** — 6/6 correctness pass. DimensionCache built at startup from FlatTables. AnalyzeSubIR resolves filtered dim tables to FK IN-filters, eliminates dim leaf+edge. Handles single-table filtered scan path (all joins eliminated) and 2-table join path (dim on scan or lookup side). Guards: skip unfiltered dims; skip when dim resolution leaves 2 base tables (DuckDB hash join faster than linear scan). join_filters on KernelJoinStep for dim filters targeting lookup table. Code: `src/storage/dimension_cache.h/.cpp`, `src/kernel/sub_query_plan.h/.cpp`.)
- Step 5: Sorted Indices → sorted permutation arrays for MIN early termination on final sub-query. (**DONE** — 3/3 correctness pass. SortedIndex built on 11 columns at startup (~10s), cached in --storage-cache v2. AnalyzeFinalIR handles Projection→Aggregate→child pattern, maps MIN columns through Projection's column_index to agg_fns. ExecuteFinalAggregate: Phase 1 = sorted scan with early termination for MIN columns on scan table with sorted index (O(k)), Phase 2 = running-min full scan for unsorted columns or lookup-table columns. Handles duplicate MIN columns, NULL output. Extended to N-table star joins (2-5 tables) with PK bitset fallback when no CSR exists. Guards: bail if scan table is base with >5M rows and 2+ join steps. Code: `src/storage/sorted_index.h/.cpp`, `src/kernel/sub_query_plan.h/.cpp`, `src/split/ir_query_splitter.cpp`.)
- Step 6: Full integration + end-to-end JOB benchmark (**DONE** — Iter 27. Kernel threshold `num_joins >= 1`: skip kernel for pure scan+filter, use kernel for 1+ CSR joins. exe 9,444→6,980ms (−26.1% vs baseline). 543ms slower than Step 3's 6,437ms due to 213 kernel-invalid iterations (31.4%) falling back to DuckDB. Middleware overhead 4,619ms from FlatTable loading + runtime CSR build.)
- Step 6.5: Pre-Step-7 optimizations. See **Known issues** below, ordered by priority.
- Step 7 (JIT): Kernel Compilation — compile SubQueryPlan into native code via LLVM JIT, replacing the interpreter. See details below.

**Known issues (ordered by estimated impact)**:

1. **Middleware overhead (4,619ms — 66% of wall time)**. Main cost: loading DuckDB temp results into FlatTable + building runtime CSR after each sub-query iteration. Top contributors: 16b (242ms), 9d (172ms), 8c (172ms), 19d (147ms), 12c (103ms). **Lazy loading attempted and reverted** (Iter 28): deferring FlatTable/CSR construction until needed was net-negative (+195ms exe, only −35ms MW) because most temps are used immediately by the next iteration. Remaining paths: (a) inverted indexes / precomputed bitmaps to eliminate entire sub-query iterations (reducing the number of temps created); (b) Step 7 loop fusion eliminates temp materialization entirely.

2. **Inverted indices (67/113 queries use `keyword_to_movies`)**: BespokeOLAP pre-computes 6 inverted index types: `keyword_to_movies` (67 queries), `person_to_rows` (10), `country_to_ids` (9), `combo_to_movies` (8), `company_to_rows` (3), `linked_movie_to_rows` (2). Our dim cache resolves dimension filters to PK values, then uses CSR to find matching FK rows. The inverted index provides direct `dim_value → vector<fk_row_id>` mapping, skipping both the dim cache PK resolution and CSR lookup. The `keyword_to_movies` index is the most impactful — it converts the `keyword IN(...) → movie_keyword → title` join pattern from a 2-base-table problem (currently hitting the base×base guard, +113.4ms regression on 6d+6b) into a direct lookup producing a movie_id bitset. Implementation: build `keyword_to_movies` at startup by scanning `movie_keyword` FK CSR; for each keyword_id, collect all movie_ids. Same pattern for other inverted indices. Could be combined with dimension cache as `dim_value → FK_row_set`. **Addresses the base×base guard** for queries like 6d, 6b without needing to lift the guard.

3. **LIKE support in kernel (+275.5ms regression from 70 LIKE-only fallback queries)**: Currently `CompileOnePredicate` returns empty for LIKE → `AnalyzeSubIR` bails → DuckDB fallback. Three fallback patterns: (1) **base JOIN dim** with LIKE on base → 0 joins after dim resolution → DuckDB by threshold anyway (majority, no kernel coverage gain); (2) **base JOIN base(+dim)** with LIKE → retains 1+ joins, LIKE support increases kernel coverage (e.g., q5a: `movie_companies JOIN title WHERE note LIKE ...`); (3) **base JOIN temp(+dim)** with LIKE → same. Pattern (1) is the majority; patterns (2)+(3) are a smaller subset where LIKE support genuinely increases kernel coverage but the regression comes from middleware overhead on all three patterns. Implementation: add `strstr()`/`memmem()` to `CompileOnePredicate` for `%pattern%` LIKE. BespokeOLAP handles LIKE more aggressively via dictionary memoization (`note_dict` + `note_memo` arrays, 12 queries) — pre-scanning unique string values and caching match results per dict-id, then checking `note_memo[dict_id]` per row instead of full string match. The `note_csr` inverted index (12 queries) further avoids scanning rows that don't match.

4. **Dimension-partitioned flat tables (67/113 BespokeOLAP queries)**: BespokeOLAP partitions large tables by dimension FK (e.g., `movie_info.partitions[info_type_id]`, `cast_info.role_movie_csr[role_id]`). Our kernel scans the full flat table with a runtime filter. Scan reduction: `cast_info` 36M→4M (9x), `movie_info` 14.8M→2M (7x). Estimated savings: 65-320ms per query touching large tables. Most impactful after Step 7 loop fusion (fused loops scan base tables directly, so partitioning avoids scanning non-matching rows). Implementation: at startup, for each large table with a low-cardinality FK column (role_id, info_type_id), build per-partition row lists. The kernel scans only the matching partition.

5. **Dictionary encoding for strings (26/113 BespokeOLAP queries)**: BespokeOLAP converts string comparisons to integer dict-id comparisons (~10x faster per row). Used for `movie_info.info` genre matching (Drama/Horror/etc.), `company_name.country_code` ("[us]"), and `cast_info.note` LIKE memoization. Impact limited for equality filters already resolved via dim cache, but significant for non-dim string columns on large tables (movie_info 14.8M rows with genre filters, cast_info 36M with note filters). Implementation: build per-column dictionary at FlatTable load time; `CompileOnePredicate` compiles string constants to dict-id integer comparisons.

6. **Cross-table bitmaps/byte-maps (14/113 BespokeOLAP queries)**: BespokeOLAP uses `us_movie_bitmap` (14 queries) for O(1) movie existence checks. These are NOT pre-computed structures — they are intermediate results built inside fused loops. Our per-sub-query architecture materializes the equivalent as temp table + runtime CSR. The bitmap optimization is part of Step 7 loop fusion. Single-table PK bitsets are already handled by `BuildFilteredPKBitset`.

7. **Base×base guard (+113.4ms on 6d+6b)**: When dim resolution reduces from 3+ tables to 2 base tables (no temps), kernel falls back to DuckDB. Reason: kernel linear scan of large base tables (cast_info 36M) is slower than DuckDB's parallel vectorized hash join. The guard is correct for generic base×base patterns. The specific regression pattern (6d, 6b: keyword IN→movie_keyword→title) is better solved by inverted indices (issue #2 above), which convert the 2-base problem into a bitset lookup. Revisit after Step 7 adds parallelism for remaining base×base cases.

8. **Latent CSR direction bug in `AnalyzeSubIR`** (`sub_query_plan.cpp` ~line 742): `GetCSR(scan_leaf->name, scan_col_name)` can return a CSR where `fk_table == scan_leaf->name`, meaning `Lookup()` returns row indices in the scan table instead of the lookup table. **Currently silent** because runtime CSR search (line 734) is checked first and succeeds for all temp-lookup cases; base×base pairs are guarded out. The same bug was fixed in `AnalyzeFinalIR` (CSR candidates validated with `csr->fk_table == lookup_leaf->name`). **Risk**: any future change that removes the base×base guard or changes CSR search order could activate this bug. **Fix path**: add the direction check to `AnalyzeSubIR` only for the `storage_plan->GetCSR()` calls (lines 742, 747), not the runtime CSR calls (lines 734, 754).

9. **`std::vector<uint64_t> dummy` per-row allocation** (`sub_query_plan.cpp:457`): Minor inefficiency in semi-join path — should pass scan_row directly to EmitRow for FROM_SCAN-only output. Low priority.

10. **OpenMP parallelism (4/113 GenDB queries, 0 BespokeOLAP)**: GenDB uses OpenMP in Q16c, Q19a, Q31b, Q33a. BespokeOLAP uses zero OpenMP — its fused single-threaded loops avoid scanning large tables directly. Low priority; Step 7 loop fusion is the better approach.

**Step 7 — Kernel Compilation (JIT)**:

The current `ExecuteSubQueryPlan()` in `src/kernel/sub_query_plan.cpp` is an **interpreter**: it takes a `SubQueryPlan` struct and loops through `join_steps`, `scan_filters`, `output_cols` at runtime, branching on `step.use_bitset`, `col.source == FROM_SCAN`, `col.type == INT32`, etc. Every row pays the cost of these generic checks (std::function calls for filters, type dispatch for column access, join step iteration).

The reference implementations show the performance ceiling of **compiled queries**:
- **BespokeOLAP** (`/home/pei/Project/BespokeOLAP/output/query_q*.cpp`): Per-query C++ source files compiled AOT to native binaries. Each query is a specialized `run_q*()` function with direct array/CSR access, hardcoded join order, inlined constants, query-specific bitmaps, sorted indices for early termination, and zero intermediate materialization. 50-170x faster than DuckDB.
- **GenDB** (`/home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries/Q*/iter_0/q*.cpp`): Similar approach — per-query C++ with mmap'd flat arrays, CSR lookups, and fused scan loops.

Both generate C++ at build time (AOT). We cannot do AOT at query time (~seconds compile), but we can achieve the same effect via **LLVM JIT** (~0.1-1ms compile per sub-query):
1. `AnalyzeSubIR()` produces a `SubQueryPlan` (plan generation, analogous to "compilation frontend")
2. **NEW**: `CompileSubQueryPlan()` translates the plan into LLVM IR — a specialized function with no per-row branching, inlined filter constants, direct array pointers, unrolled join steps. This is true compilation.
3. LLJIT compiles LLVM IR to native machine code
4. Execute the compiled function pointer — no interpretation overhead

This eliminates: `std::function` call overhead for filters (~5-10ns/row), type dispatch branches, join step iteration, and output column source checks. On large scans (1M+ rows), even 5ns/row overhead adds up to milliseconds.

Code: `src/jit/kernel_codegen.cpp` (LLVM IR generation from SubQueryPlan). Est. +10-30% on top of the generic interpreter, closing the gap toward BespokeOLAP's compiled performance.

**Step 7 sub-features**:
- **`__builtin_prefetch` for CSR accesses**: Insert prefetch instructions before CSR `row_ptr[]` and `col_idx[]` array accesses in the compiled kernel, same technique as Iter 22-24 DuckDB hash join prefetching. BespokeOLAP uses `__builtin_prefetch(&genre_csr_offsets[gkey], 0, 0)` with PF_DIST=8.
- **Sub-query loop fusion** (single biggest remaining optimization): Analyze the full query's sub-query sequence and generate a single fused function that chains CSR lookups without materializing intermediate temp tables. Eliminates temp allocation + memory writes (~1-2s middleware overhead) and runtime CSR building (~0.5s). BespokeOLAP processes entire queries in one fused loop with zero intermediate materialization.

  **BespokeOLAP fused loop patterns** (from examining q6f, q8c, q9b, q16b):
  - **Bitmap/byte-map as intermediate**: Instead of materializing a temp table from sub-query 1 and building runtime CSR for sub-query 2, a fused loop builds a byte-map (e.g., `us_movie_bytemap[movie_id]=1`) during the first scan pass, then checks it with `O(1)` array access in the second pass. This replaces both temp materialization AND runtime CSR build. Example: q8c scans `movie_companies` filtered by company_name → byte-map → scan `title` with byte-map check → sorted MIN.
  - **Nested CSR probes in single loop**: q9b scans `name` → CSR probe `cast_info` → CSR probe `movie_info` → CSR probe `movie_info_idx` in a single nested loop body. No intermediate tables between join steps.
  - **K-way merge for multi-FK patterns**: q6f identifies qualifying `movie_id`s via k-way sorted merge of CSR results from multiple dimension tables, then probes into `cast_info` and scans `name` — all in one function with zero intermediate materialization.
  - **Always scan small/filtered side, CSR-probe large side**: BespokeOLAP always iterates the smaller/filtered table and uses CSR to look up into the larger table. Single-threaded because fused loops never scan 36M-row tables directly (they CSR-probe into them).

  **Implementation approach**: Run the full split plan to identify the sub-query DAG. For each connected chain of sub-queries where the output of one feeds as input to the next, generate a single fused loop. The intermediate temp table becomes either (a) a byte-map if only existence is needed (semi-join), or (b) a small array if values are needed (projection). Requires analyzing which output columns of sub-query N are consumed by sub-query N+1.
- **Dimension-partitioned flat tables**: See known issue #4. Can be done pre-Step-7 or as part of Step 7. Most impactful after loop fusion (fused loops scan base tables directly).
- **Dictionary encoding**: See known issue #5. Can be done pre-Step-7 or as part of Step 7.

**Code organization**:
- Storage structures: `src/storage/` (flat tables, CSR indexes, storage plan)
- Kernel interpreter + plan analysis: `src/kernel/sub_query_plan.cpp`
- Kernel JIT compilation (Step 7): `src/jit/kernel_codegen.cpp`
- Integration: `src/adapters/duckdb_adapter.cpp` (loading), `src/split/ir_query_splitter.cpp` (sub-query plans reference flat tables + CSR)

**Memory budget**: ~3.8 GB (flat arrays ~3.1 GB + CSR ~0.7 GB) on 63 GB machine.

### Direction A-JIT (old, nearly exhausted): AMAC AdvancePointers

One remaining feasible optimization within the current pipeline-jit architecture. Deprioritized in favor of Direction A (storage plan).

**AMAC-style interleaved probing in AdvancePointers** (~20-40ms potential)

After Iter 24, AdvancePointers remains 6-8% on memory-bound queries. ScanInnerJoin now absorbs ResolvePredicates work and shows 7.8-10.5% on probe-heavy queries. True AMAC maintains a group of 8-16 probe keys simultaneously at different pipeline stages, switching between keys to hide memory latency.

Target queries (Iter 24 perf record): 16b (AdvancePointers 8.1%, ScanInnerJoin 3.7%), 8c (AdvancePointers 6.0%, ScanInnerJoin 7.8%), 17f (AdvancePointers 3.7%, ScanInnerJoin 2.4%), 19d (AdvancePointers 2.6%, ScanInnerJoin 10.5%).
Files: `join_hashtable.cpp` — restructure `ScanInnerJoin`/`AdvancePointers` loop.
Complexity: High — requires converting the probe loop from per-key sequential to state-machine interleaved.

**ResolvePredicates** — DONE (Iter 24, JIT-gated). Eliminated from JIT perf profile via equality-only fast path. TemplatedMatch<int> at 2.9% on 8c via row-data prefetch (was 3.1% in Iter 23).

### Direction B: Reduce split overhead on heavy queries

Node-based split produces unnecessarily large temp tables. Split order doesn't consider selectivity. Independent and complementary to Direction A — better split order produces smaller temp tables, making runtime CSR builds faster.

**Root cause**: 16b first subplan joins `movie_companies` (2.6M) × `company_name` (85K) → 1.15M rows. 8c same pattern.

**Approach**: Reorder subplans by estimated selectivity, or merge remaining subplans when temp cardinality exceeds threshold.

**Files**: `node_based_splitter.cpp`, `ir_query_splitter.cpp`
**Target queries**: 16b, 8c, 16c, 6d, 9d (top 5 regressions, total +597ms vs none-split)

## What I Expect
- Direction A (storage plan) is the next major effort. Implement step by step, verify each component's effect independently.
- Direction A-JIT is nearly exhausted (AMAC ~20-40ms, high complexity). Deprioritized.
- Direction B (split strategy) is complementary and can be done before or after Direction A.
- After each change, quick-test on the target queries (16b, 8c, 19d), then correctness check.
- After implementation, re-run `python3 measure/analyze_breakdown.py` to compare.
- Show uncertainty clearly. Flag assumptions. Verify with measurements.
- After each iteration, report: what changed, which queries improved, which regressed, net effect on total JOB time.
