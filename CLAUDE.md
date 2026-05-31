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

### 1. Read relevant source code for the optimization you're implementing

### 2. Check if we need use plan mode to design code implementation. Then make the code change and write unit gtests in unit_test/ dir. If add new modules or code changes related to breakdown timer, decide a reasonable timer position and ask user to confirm. Confirm analyze_middleware_breakdown, analyze_none_split_breakdown in /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py still work. Otherwise, update the analyze_middleware_breakdown and/or analyze_none_split_breakdown, and confirm with the user.

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

## Profile the bottleneck
Use `perf_analysis.md` as the step-by-step guide. First, run `python3 measure/find_top_queries.py` to identify the top-10 heaviest queries from the latest `breakdown_time_log.csv`. Always check the heaviest queries in jit_optimization_claude.md. Profile both baseline (none-split/none-jit) and best JIT config on the same queries: the gap shows where split+JIT overhead is; then also profile best JIT alone to find the remaining bottleneck to optimize. Think fundamentally what is the correct optimization technique for the current bottleneck. Think in the Umbra way and in the Thomas Neumann way.

**Profiling tool**:
1. **`perf stat`** — First step. Classifies bottleneck type (compute vs memory bound) via IPC, L1/LLC cache miss rates, branch misprediction, CPU utilization.
2. **`perf record` + `perf report`** — Second step. Shows function-level CPU hotspots. Directly answers "which operator is the bottleneck."
3. **DuckDB EXPLAIN ANALYZE** — Operator-level cardinality + time. Use selectively (adds overhead).
4. **`perf record -e cache-misses` / `perf mem`** — Only if `perf stat` shows memory-bound. Pinpoints which data structures cause cache misses.

## Current Status (Iter 30 — Step 6.5.3: LIKE Support + Bug Fixes + Issue Cleanup)

See jit_optimization_claude.md for full per-iteration details.

**Performance** (in-memory, warm, avg of 10 runs — execution time only, ignoring JIT compile):

| Config                              | Execution (s) | Middleware (s) | JIT Compile (s) | Wall (s) | vs ns/nojit |
|--------------------------------------|-------------:|---------------:|----------------:|---------:|------------:|
| none-split / none-jit                |         9.53 |          0.003 |               — |     9.53 | baseline    |
| **node-based / pipeline-jit (no SIMD)**|       **7.07**|       **5.19** |         **1.11**|  **13.40**| **-2.46 (-25.8%)**|

Step 6.5 → Step 6.5.3 (measured in step_6_fix_all vs step_6_5_perf):

| Metric | Step 6.5 | Step 6.5.3 | Delta |
|--------|-------:|---------:|------:|
| Execution | 7,297ms | 7,069ms | **-228ms (-3.1%)** |
| Middleware | 5,266ms | 5,190ms | -76ms |
| JIT Compile | 1,024ms | 1,108ms | +84ms |
| Wall | 13,587ms | 13,403ms | **-184ms (-1.4%)** |

**Key result**: Execution is **25.8% faster** than none-split/none-jit (9,530→7,069ms). But wall time is **40.6% slower** (9,530→13,403ms) due to middleware overhead (5,190ms) + JIT compile (1,108ms). All pre-Step-7 known issues are resolved — the remaining bottleneck is middleware overhead (temp materialization + runtime CSR building), which requires Step 7 loop fusion.

**Pre-Step-7 known issues — all resolved**:
- #1 MW overhead: partially addressed, fundamental fix = Step 7 loop fusion
- #2 Inverted indices: DONE (Iter 29)
- #3 LIKE support: DONE (Iter 30)
- #4 Dim-partitioned tables: deferred to after Step 7 (most impactful after loop fusion)
- #5 Dictionary encoding: deferred to after Step 7
- #6 Cross-table bitmaps: part of Step 7 loop fusion
- #7 Base×base guard: SKIP (addressed by inverted indices)
- #8 CSR direction bug: FIXED (Iter 30)
- #9 Dummy vector alloc: FIXED (Iter 30)
- #10 OpenMP: SKIP (loop fusion is better approach)

Auto-SIMD regression is a known issue — SIMD codegen has bugs on the kernel path. Use `--jit-simd=none`.

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
- Step 6.5: Pre-Step-7 optimizations (**DONE** — Iter 29. Unfiltered dim elimination converts 3-leaf sub-queries with no-filter dim tables to 2-leaf. 3-table inverted index resolution handles `source(filtered) + bridge + target` patterns via inverted index lookup for keyword→title, name→title, company_name→title. Inverted index specs reduced 11→3 (8 dead specs removed, ~100MB memory saved). exe 7,453→7,297ms (-2.1%), wall 13,796→13,587ms (-1.5%). Base×base guard KEPT due to IR column name mismatch bug in movie_link.)
- Step 7 (JIT): Kernel Compilation — compile SubQueryPlan into native code via LLVM JIT, replacing the interpreter. See details below.

**Known issues (ordered by estimated impact)**:

1. **Middleware overhead (5,266ms)** — **PARTIALLY ADDRESSED** (Iter 29). Inverted index resolution eliminates some sub-query iterations (fewer temps created). Unfiltered dim elimination converts 3-leaf to 2-leaf (kernel-handled, no DuckDB fallback). Net MW savings: -17ms. **Lazy loading attempted and reverted** (Iter 28): net-negative. Remaining paths: (a) more inverted index patterns (precomputed bitmaps, person_has_aka_bits); (b) Step 7 loop fusion eliminates temp materialization entirely (the fundamental fix).

2. **Inverted indices** — **DONE** (Iter 29, Step 6.5). Built 3 inverted indices at startup: `keyword→title` (via movie_keyword), `name→title` (via cast_info), `company_name→title` (via movie_companies). 3-table inverted index resolution in `AnalyzeSubIR` handles `source(filtered) + bridge + target` pattern: scans source with compiled filters, inverted index lookup → target PK IN-filter, eliminates source+bridge leaves. Bridge column remapping for output projections. Selectivity guard (<50% of target rows). Dead specs removed (11→3, ~100MB memory saved). Base×base guard KEPT due to IR column name mismatch bug.

3. **LIKE support in kernel** — **DONE** (Iter 30, Step 6.5.3). Added TextLike/Text_Not_Like to `CompileOnePredicate` with 6 pattern kinds: EQUALITY, PREFIX, SUFFIX, CONTAINS (memmem), MULTI_SEGMENT (sequential memmem), COMPLEX (DP LikeMatch). Single-table LIKE guard: base tables without inverted-index PK filter fall back to DuckDB (vectorized scan faster for large tables). Fixed dangling pointer bug in 3-table inverted index resolution (leaves.erase invalidated pointers used for dim_derived_filters cleanup). Fixed AnalyzeFinalIR LIKE guard (sorted-MIN path produced wrong results with LIKE on lookup tables). Net: exe -228ms (-3.1%), wall -218ms (-1.6%).

4. **Dimension-partitioned flat tables (67/113 BespokeOLAP queries)**: BespokeOLAP partitions large tables by dimension FK (e.g., `movie_info.partitions[info_type_id]`, `cast_info.role_movie_csr[role_id]`). Our kernel scans the full flat table with a runtime filter. Scan reduction: `cast_info` 36M→4M (9x), `movie_info` 14.8M→2M (7x). Estimated savings: 65-320ms per query touching large tables. Most impactful after Step 7 loop fusion (fused loops scan base tables directly, so partitioning avoids scanning non-matching rows). Implementation: at startup, for each large table with a low-cardinality FK column (role_id, info_type_id), build per-partition row lists. The kernel scans only the matching partition.

5. **Dictionary encoding for strings (26/113 BespokeOLAP queries)**: BespokeOLAP converts string comparisons to integer dict-id comparisons (~10x faster per row). Used for `movie_info.info` genre matching (Drama/Horror/etc.), `company_name.country_code` ("[us]"), and `cast_info.note` LIKE memoization. Impact limited for equality filters already resolved via dim cache, but significant for non-dim string columns on large tables (movie_info 14.8M rows with genre filters, cast_info 36M with note filters). Implementation: build per-column dictionary at FlatTable load time; `CompileOnePredicate` compiles string constants to dict-id integer comparisons.

6. **Cross-table bitmaps/byte-maps (14/113 BespokeOLAP queries)**: BespokeOLAP uses `us_movie_bitmap` (14 queries) for O(1) movie existence checks. These are NOT pre-computed structures — they are intermediate results built inside fused loops. Our per-sub-query architecture materializes the equivalent as temp table + runtime CSR. The bitmap optimization is part of Step 7 loop fusion. Single-table PK bitsets are already handled by `BuildFilteredPKBitset`.

7. **Base×base guard (+113.4ms on 6d+6b)**: When dim resolution reduces from 3+ tables to 2 base tables (no temps), kernel falls back to DuckDB. Reason: kernel linear scan of large base tables (cast_info 36M) is slower than DuckDB's parallel vectorized hash join. The guard is correct for generic base×base patterns. The specific regression pattern (6d, 6b: keyword IN→movie_keyword→title) is better solved by inverted indices (issue #2 above), which convert the 2-base problem into a bitset lookup. Revisit after Step 7 adds parallelism for remaining base×base cases.

8. **Latent CSR direction bug in `AnalyzeSubIR`** — **FIXED** (Iter 30). Added `fk_table` direction validation to `FindCSR` lambda in `AnalyzeSubIR`: both runtime CSR and `storage_plan->GetCSR()` paths now check `c->fk_table == l->name`, matching the pattern already used in `AnalyzeFinalIR`. Prevents silent wrong results from future code changes.

9. **`std::vector<uint64_t> dummy` per-row allocation** — **FIXED** (Iter 30). Hoisted from per-row allocation inside `ScanRow` lambda to a single pre-allocated `semi_dummy` outside the scan loop.

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
