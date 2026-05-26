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
- Schema: /home/pei/Project/benchmarks/imdb_job-postgres/schema.sql
- Foreign keys: /home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql
- Dataset: /home/pei/Project/benchmarks/imdb_job-postgres/csv
- Reference implementations: /home/pei/Project/BespokeOLAP/output/ (compiled queries), /home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries/ (generated queries)

## Build
```bash
# DuckDB (only if you modify DuckDB files):
cd /home/pei/Project/duckdb/build/release && make -j12

# AQP middleware (after any middleware change):
cd /home/pei/Project/AQP_middleware/build_release && make -j12
```

## Current Performance (already measured, don't re-derive)

Measurement data is in /home/pei/Project/AQP_middleware/measure/job_result/. We drop the first 5 runs as warm up and take average of the other 10 runs. The measurement uses `--repeat=15` (in-process iteration, single DuckDB connection).

### Understanding `breakdown_time_log.csv`

Per-sub-plan wall-clock times in **milliseconds** (chrono_tic/toc). Each query produces one row per iteration, with comma-separated timing columns. The column layout differs by config:

**none-split** (no splitting, direct DuckDB execution):
- No-JIT row: `[prepare_middleware, read_sql, execute, show_output]`
- JIT row: `[prepare_middleware, read_sql, jit_compile, execute, show_output]`

**split (node-based / relationship-center)**:
- Preamble (once): `prepare_middleware, read_sql, parse_sql, preprocess`
  - node-based has NO `convert_plan_to_ir` step; other strategies insert it after preprocess
- Per split iteration (repeated K times):
  - No-JIT: `[extract_sub-IR, generate_sub-SQL, execute_sub-SQL, extra_materialization, update_IR]` — 5 columns
  - JIT: `[extract_sub-IR, generate_sub-SQL, jit_compile, execute_sub-SQL, extra_materialization, update_IR]` — 6 columns
- Possibly one extra `extract_sub-IR` column at the end (if the final SplitIR returns no more sub-plans)
- Tail:
  - No-JIT: `[generate_final_sub-SQL, final_execute, show_output]` — 3 columns
  - JIT: `[generate_final_sub-SQL, jit_compile_final, final_execute, show_output]` — 4 columns

Reference parser: `analyze_middleware_breakdown()` in `/home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py`
- Quick analysis script: `python3 /home/pei/Project/AQP_middleware/measure/analyze_breakdown.py [job_result_dir]` — parses all 4 configs, reports totals, top-20 heaviest queries, regressions, and JIT effect

**Three time components** (from the parser):
- **Execution** = sum of all `execute_sub-SQL` + `final_execute` — the actual DuckDB query runtime
- **JIT Compile** = sum of all `jit_compile` + `jit_compile_final` — LLVM IR generation + compilation (includes DuckDB `Prepare()` time)
- **Middleware Overhead** = everything else (prepare_middleware, read_sql, parse_sql, preprocess, generate_sub-SQL, extra_materialization, extract_sub-IR, update_IR, show_output)

See current numbers and per-iteration history in /home/pei/Project/AQP_middleware/jit_optimization_claude.md.

## Constraints
- Only modify AQP middleware code (JIT in src/jit/, storage plan in src/storage/, split strategies in src/split/, adapters in src/adapters/).
- Do NOT touch native DuckDB or PostgreSQL code, except for existing JIT dispatch code in DuckDB operators (physical_filter.cpp, physical_hash_join.cpp, etc.).
- Native DuckDB/PostgreSQL must NOT be affected by storage plan changes — all auxiliary structures live in middleware memory.
- Must return identical results to the original query.
- You may design new split strategies in AQP_middleware if they enable better execution.
- Ignore compilation time -- only execution time matters.

## Fast Iteration Workflow

### Quick test (single query, ~1 second):
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
  /home/pei/Project/benchmarks/imdb_job-postgres/queries/7c.sql
```
Check time_log.csv for per-phase timing.

### Heavy test queries (focus on these for iteration):
You should always check the heaviest queries from the current version of jit in jit_optimization_claude.md.

### Profiling tool selection guide:
When analyzing bottlenecks of heavy queries, choose tools based on what you need to learn:

1. **`perf stat`** — First step. Classifies bottleneck type (compute vs memory bound) via IPC, L1/LLC cache miss rates, branch misprediction, CPU utilization. Run on all heavy queries as a batch.
2. **`perf record` + `perf report`** — Second step. Shows function-level CPU hotspots (e.g., `JoinHashTable::Probe` vs `PhysicalTableScan::GetData`). Directly answers "which operator is the bottleneck" without DuckDB's internal profiler.
3. **DuckDB EXPLAIN ANALYZE** — Operator-level cardinality + time. Reveals per-operator row counts and plan shape. Use selectively (adds overhead).
4. **`perf record -e cache-misses` / `perf mem`** — Only if `perf stat` shows memory-bound. Pinpoints which data structures cause cache misses.

**Do NOT use:**
- `strace`: Not useful for in-memory queries (no I/O syscalls on warm buffer pool)
- Intel VTune: `perf` gives sufficient insight; VTune is slower to set up
- eBPF/bpftrace: Overkill for CPU profiling; only useful for I/O or scheduler analysis

### Correctness check (~2 min):
For a quick test:
```bash
cd /home/pei/Project/AQP_middleware/measure
bash run_job.sh duckdb node-based pipeline o1 none
# Compare output with golden:
filter='grep -v "^Running\|^==\|^Execution\|^$\|^waiting\|^server\|^ANALYZ\|^duckdb runs:\|^(base)"'
diff <(eval $filter job_result/aqp_middleware_duckdb_node-based_pipeline_o1_none_job.txt) \
     <(eval $filter duckdb_job_node-based_golden.txt)
```
For whole test:
```bash
cd /home/pei/Project/AQP_middleware/measure
bash ./correctness_test.sh
```

### Full breakdown measurement (~28 min, only for final validation):
do NOT do ANYTHING when running performance measurement to avoid any noise.
```bash
cd /home/pei/Project/AQP_middleware/measure
bash measure_breakdown_time_job.sh duckdb node-based pipeline o1 none
```

## Workflow Per Iteration
0. **Profile the bottleneck**: Use `perf_analysis.md` as the step-by-step guide. First, run `python3 measure/find_top_queries.py` to identify the top-10 heaviest queries from the latest `breakdown_time_log.csv`. Then follow `perf_analysis.md` Steps 2-5 to profile those queries with `perf stat`, `perf record`, Intel VTune, and eBPF/bpftrace. Profile both baseline (none-split/none-jit) and best JIT config on the same queries: the gap shows where split+JIT overhead is; then also profile best JIT alone to find the remaining bottleneck to optimize. Think fundamentally what is the correct optimization technique for the current bottleneck. Think in the Umbra way and in the Thomas Neumann way.
 - Check detailed bottleneck in ### Profiling tool selection guide section.
1. Read relevant source code for the optimization you're implementing
2. Make the code change
3. Build (`make -j12` in the appropriate build_release dir)
4. Quick-test on 2-3 target queries, check execution time in time_log.csv
5. If faster: run correctness check on full JOB, then measure breakdown
6. If slower or neutral: analyze why, check the slowdown queries if it is noise, revert or adjust, try again. 
7. If some queries speedup but some others slowdown, check what's the differences and can we fix it fundamentally or at least guard it by some condition of the slowdown query's pattern.
8. Comparing queries where JIT with split strategy (e.g., node-based) speedup or slowdown compared to the none-split+none-JIT and compared to the last version of JIT, find out the fundamental reason.
9. Summarize: what changed, which queries improved/regressed, by how much, and update ## Current Status in this file. Then stop compact the conversation within Claude Code and continue the next step.

## Helper
- Hardware feature: {"cpu_cores": 12, "simd": "avx2", "total_memory_gb": 63, "disk_type": "hdd", "l3_cache_mb": 12}. You can run any tool to detect more on this server.
- Follow expert knowledge in /home/pei/Project/BespokeOLAP/conversations/prompts/expert_knowledge.txt and /home/pei/Project/GenDB/src/gendb/agents/code-generator/prompt.md and /home/pei/Project/GenDB/src/gendb/agents/query-optimizer/prompt.md
- Think fundamentally what information from the previous sub-plan execution can you use to generate better JIT code for the next sub-plan. Then check how to get this information from the codebase.
- It is okay that one optimization speedup some queries but slowdown the others and overall slowdown. Summarize the common pattern for improved queries and add a guard condition.  we'd better not have magic number in the code. can we calculate these thresholds with a formular or worst case a cost model by, e.g., selectivity, temp_card, span, etc from runtime information after execution the previous sub-query? but we cannot specify which query use which optimization or not use it.
- We care about the whole JOB benchmark, so prioritize the heaviest queries with the heaviest bottlenecks.
- You can check well-optimized queries under /home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries and under /home/pei/Project/BespokeOLAP/output to find the gap, and learn from it.

## Current Status (Iter 24 — Probe path fast paths + row-data prefetching)

Iterations 1-18 optimized JIT. Iter 19 fixed measurement infrastructure. Iter 20 re-measured with pure in-memory baseline. Iter 21 profiled bottlenecks. Iter 22 added JIT-gated software prefetching for hash join probe, build, and bloom filter operations. Iter 23 added chain-walk prefetching in AdvancePointers (JIT-gated) and unrolled BF GetMask. Iter 24 added ScanInnerJoin/ScanKeyMatches equality-only fast paths, found_match skip for INNER/RIGHT joins, ResolvePredicates memcpy, chain-node data prefetch, and row-data prefetch before Match — all JIT-gated. None-jit path is unchanged from original DuckDB. See jit_optimization_claude.md for full per-iteration details.

**Performance** (in-memory, warm, avg of 10 runs — execution time only, ignoring JIT compile):

| Config                              | Execution (s) | Middleware (s) | JIT Compile (s) | Wall (s) | vs ns/nojit |
|--------------------------------------|-------------:|---------------:|----------------:|---------:|------------:|
| none-split / none-jit                |         9.52 |          0.003 |               — |     9.53 | baseline    |
| none-split / pipeline-jit (no SIMD)  |         8.83 |          0.013 |            8.76 |    17.60 | -0.69 (-7.3%) |
| none-split / pipeline-jit (auto SIMD)|         8.85 |          0.014 |            9.59 |    18.45 | -0.67 (-7.1%) |
| node-based / none-jit                |        10.15 |          0.99  |               — |    11.14 | +0.63 (+6.6%) |
| node-based / pipeline-jit (no SIMD)  |         9.60 |          1.01  |            2.18 |    12.79 | +0.07 (+0.8%) |
| node-based / pipeline-jit (auto SIMD)|         9.60 |          1.01  |            2.22 |    12.82 | +0.07 (+0.8%) |

None-jit re-measured after JIT-gating all Iter 24 changes (none-jit path = original DuckDB). JIT numbers from previous session. Cross-session JIT effect: -7.3% (none-split), -5.5% (node-based). True same-session effect estimated ~-6.5% to -7.3%.

### None-split pipeline-JIT: Top 10 heavy queries (perf stat + perf record, Iter 24)

| Rank | Query | Exe (ms) | IPC  | Cache miss% | LLC miss% | CPU/Wall | Bottleneck Type |
|------|-------|----------|------|-------------|-----------|----------|-----------------|
| 1  | 7c  | 433.3 | 0.39 | 41.2% | 25.9% | 6.6x | Severely memory-bound (string memcpy) |
| 2  | 16b | 306.6 | 0.86 | 17.0% | 15.2% | 7.6x | Memory-bound (HJ probe + BF) |
| 3  | 11d | 266.9 | 1.02 | 15.9% | 34.4% | 1.8x | Parallelism-limited + mutex contention |
| 4  | 22d | 231.5 | 0.78 | 16.0% | 26.4% | 4.4x | Memory-bound (HJ build, many joins) |
| 5  | 8c  | 219.7 | 0.76 | 21.5% | 15.2% | 6.2x | Memory-bound (HJ probe+build balanced) |
| 6  | 17f | 207.0 | 1.04 |  7.3% |  4.8% | 7.6x | Compute-bound (BF lookup + HJ probe) |
| 7  | 19d | 197.8 | 0.91 | 12.7% | 12.2% | 4.7x | Mixed (HJ probe + string decompression) |
| 8  | 6f  | 197.7 | 0.80 | 13.6% | 12.7% | 8.3x | Memory-bound (HJ build dominant) |
| 9  | 25c | 186.2 | 1.12 |  7.9% | 11.9% | 5.9x | Compute-bound (string scan + BF lookup) |
| 10 | 17d | 175.1 | 1.10 |  5.6% |  2.8% | 7.7x | Compute-bound (BF lookup + filter) |

Top 10 sum: 2421.8ms (27.4% of total 8831.3ms execution).

**Per-query function hotspots (perf record top functions, Iter 24 — all JIT-gated):**

- **16b**: `AdvancePointers` **8.1%**, `BF::LookupHashes` 8.1%, `InsertHashes` 5.9%, `FastMemcpy` 5.2%, `FilterSwitch` 3.8%, `Hash` 3.8%, `ScanInnerJoin` 3.7%, `Finalize` 3.2%, `GetRowPointers` 3.1%, `Gather<string>` 2.6%, `Gather<int>` 2.5%.
- **8c**: `InsertHashes` 9.6%, `ScanInnerJoin` **7.8%**, `AdvancePointers` 6.0%, `BF::LookupHashes` 3.7%, `Finalize` 3.6%, `BF::InsertHashes` 3.4%, `GetRowPointers` 3.3%, `Gather<int>` 3.0%, `TemplatedMatch<int>` 2.9%, `BuildPartitionSel` 2.5%.
- **17f**: `BF::LookupHashes` 8.7%, `InsertHashes` 7.2%, `Hash` 5.1%, `FilterSwitch` 4.7%, `fsst_decompress` 4.0%, `AdvancePointers` 3.7%, `BuildPartitionSel` 3.1%, `ScanInnerJoin` 2.4%, `Gather<int>` 2.3%.
- **19d**: `ScanInnerJoin` **10.5%**, `InsertHashes` 5.7%, `BF::LookupHashes` 4.6%, `fsst_decompress` 3.9%, `StringScanPartial` 3.4%, `AdvancePointers` 2.6%, `Finalize` 2.5%, `FastMemcpy` 2.3%, `Hash` 2.2%.
- **6f**: `InsertHashes` **18.0%**, `BF::LookupHashes` 7.1%, `Finalize` 6.4%, `fsst_decompress` 5.7%, `BF::InsertHashes` 5.2%, `FilterSwitch` 4.5%, `Hash` 4.0%. Unchangeable — build-dominated.
- **7c**: Unchangeable — VARCHAR memcpy dominates.
- **11d**: Unchangeable — mutex contention.
- **22d**: Unchangeable — build-dominated.
- **25c**: Unchangeable — string decompression dominated.
- **17d**: Unchangeable — compute-bound BF+filter.

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

### Direction A-JIT optimization status (Iters 1-24, nearly exhausted)

| Technique | Status | Iters | Effect |
|-----------|--------|-------|--------|
| Scalar JIT hash probe | Done (disabled) | 9 | -1.6%, vectorized beats scalar |
| Naive 2-pass software prefetch | Done (reverted) | 12 | +3.8%, destroyed pipelining |
| Chain-walk skip (unique keys) | Done (kept) | 6 | Neutral (~60ms, below noise) |
| JIT-gated probe bucket prefetch | Done | 22 | Part of -6.5% on none-split |
| JIT-gated build prefetch (InsertHashes) | Done | 22 | Part of -6.5% on none-split |
| JIT-gated BF prefetch | Done | 22 | Part of -6.5% on none-split |
| JIT-gated chain-walk prefetch (AdvancePointers) | Done | 23 | -218ms on node-based |
| BF GetMask unroll | Done | 23 | Neutral (compiler already optimized) |
| ScanInnerJoin/ScanKeyMatches equality fast path | Done (JIT-gated) | 24 | Eliminates ResolvePredicates from profile |
| Skip found_match for INNER/RIGHT joins | Done (JIT-gated) | 24 | Reduces cache traffic |
| ResolvePredicates memcpy | Done (JIT-gated) | 24 | Replaces element-by-element loop |
| Row-data prefetch before Match | Done (JIT-gated) | 24 | LLC miss -2.8pp on 8c, TemplatedMatch -1.2pp |
| Chain-node data prefetch in AdvancePointers | Done (JIT-gated) | 24 | Warms key data for next ScanInnerJoin |
| **AMAC-style interleaved probe (AdvancePointers)** | **NOT done** | — | Est. ~20-40ms. AdvancePointers still 6-8%. High complexity. |
| BF inlining into probe loop | Deferred | — | Requires whole-pipeline JIT (BF hash ≠ probe hash) |
| Hash build JIT | Not worth it | — | DuckDB native is tight, ~1-2% achievable |

### Bottleneck function assessment (what's addressable vs not)

| Function | CPU% range | Addressable? | Notes |
|----------|-----------|-------------|-------|
| InsertHashes | 5-17% | No | Already prefetched. DuckDB's build is tight. |
| BF::LookupHashes | 3-10% | No | Already prefetched. Compute-bound queries are cache-resident. |
| **AdvancePointers** | 2-8% | **Maybe (AMAC)** | Data prefetch added (Iter 24). AMAC could hide remaining latency but high complexity. |
| ResolvePredicates | 0% (was 3-7%) | Done (Iter 24) | Eliminated via equality-only fast path in ScanInnerJoin. |
| Finalize | 3-10% | No | Calls InsertHashes internally. Already prefetched. |
| VectorOperations::Hash | 2-5% | No | Would need whole-pipeline JIT to share with probe hash. |
| Gather\<int\>/Gather\<string\> | 3-9% | No | Whole-pipeline JIT territory. |
| memmove/FastMemcpy | 3-28% | No | DuckDB row-layout architecture. |
| fsst_decompress/StringScanPartial | 2-15% | No | DuckDB internal FSST codec. |
| mutex | 6% | No | DuckDB scheduler. |

### Analysis tool
```bash
python3 /home/pei/Project/AQP_middleware/measure/analyze_breakdown.py [job_result_dir]
```

## Next Steps

### Direction A (NEW): Auxiliary Storage Plan + Kernel Execution (high impact, est. 5-20x)

Build auxiliary in-memory data structures in the AQP middleware and execute sub-queries directly via optimized kernel functions, bypassing DuckDB's hash join and decompression overhead. **DuckDB/PostgreSQL remain completely unchanged** — they serve as data source at startup and SQL parser at query time. Native DuckDB/PostgreSQL does NOT benefit from these changes; all structures live in middleware memory.

**Full design**: `/home/pei/Project/AQP_middleware/storage_plan_design.md`

**Performance estimate**: 5-20x over current DuckDB, not the full 50-170x of BespokeOLAP. The gap: (1) temp table materialization between sub-queries (BespokeOLAP has zero — single fused loop), (2) no query-specific pre-computed bitmaps, (3) generic executor vs. per-query compiled code. To close the gap further: kernel fusion (Step 7 JIT) + Direction B (fewer/smaller temps).

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

**DSL / multi-stage programming**: Not needed for v1. The generic `ExecuteSubQuery()` handles all patterns via the plan struct. If interpretation overhead in the inner loop becomes a bottleneck (checking join count, branching on agg type), consider either (a) JIT-compiling SubQueryPlan into specialized code (Step 7), or (b) a lightweight DSL that generates specialized C++ at compile time. Measure first.

**Interaction with Direction B (split strategy)**: A better split strategy would shift patterns toward more `1base+N_temp` and fewer multi-base patterns (`2base`/`3base`/`4base`). This is strictly better for the executor — more CSR-joinable, fewer DuckDB fallbacks. In particular, if we improve the split strategy to always start with a single filtered dimension→base join, every sub-query becomes `1base+N_temp` or `1dim+N_temp`, which are pure CSR-joinable. Direction B is NOT a prerequisite — implement storage plan first, then Direction B further improves it. After Direction B, verify if new sub-query patterns need executor support.

**Implementation order**: Step-by-step, with breakdown measurement after each component to verify its individual effect:
- Step 1: Flat Column Arrays + basic ScanFilter → measure scan speedup
- Step 2: CSR Indexes on base tables + CSRSemiJoin → measure single FK join speedup
- Step 3: Runtime CSR on temp tables + SubQueryPlan executor → measure node-based split execution (target: 16b, 8c)
- Step 4: Dimension Constants → measure join elimination effect
- Step 5: Sorted Indices → measure MIN/MAX aggregation speedup
- Step 6: Full integration + end-to-end JOB benchmark
- Step 7 (optional, JIT): Kernel Fusion — JIT-compile SubQueryPlan into specialized loops. Code: `src/jit/kernel_codegen.cpp`. Est. +10-30% on top of generic executor.

**Code organization**:
- Non-JIT components: `src/storage/` (new directory) — general middleware feature
- JIT kernel fusion (Step 7 only): `src/jit/kernel_codegen.cpp`
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
