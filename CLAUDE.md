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

## Workflow Per Iteration

### 0. Profile the bottleneck
Use `perf_analysis.md` as the step-by-step guide. First, run `python3 measure/find_top_queries.py` to identify the top-10 heaviest queries from the latest `breakdown_time_log.csv`. Always check the heaviest queries in jit_optimization_claude.md. Profile both baseline (none-split/none-jit) and best JIT config on the same queries: the gap shows where split+JIT overhead is; then also profile best JIT alone to find the remaining bottleneck to optimize. Think fundamentally what is the correct optimization technique for the current bottleneck. Think in the Umbra way and in the Thomas Neumann way.

**Profiling tool**:
1. **`perf stat`** — First step. Classifies bottleneck type (compute vs memory bound) via IPC, L1/LLC cache miss rates, branch misprediction, CPU utilization.
2. **`perf record` + `perf report`** — Second step. Shows function-level CPU hotspots. Directly answers "which operator is the bottleneck."
3. **DuckDB EXPLAIN ANALYZE** — Operator-level cardinality + time. Use selectively (adds overhead).
4. **`perf record -e cache-misses` / `perf mem`** — Only if `perf stat` shows memory-bound. Pinpoints which data structures cause cache misses.

**Do NOT use:** `strace` (no I/O syscalls in-memory), Intel VTune (`perf` suffices), eBPF/bpftrace (overkill for CPU profiling).

### 1. Read relevant source code for the optimization you're implementing

### 2. Make the code change and write unit gtests in unit_test/ dir. If add new modules or code changes related to breakdown timer, decide a reasonable timer position and ask user to confirm. Confirm analyze_middleware_breakdown, analyze_none_split_breakdown in /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py still work. Otherwise, update the analyze_middleware_breakdown and/or analyze_none_split_breakdown, and confirm with the user.

### 3. Build and quick-test
Build:
```bash
cd /home/pei/Project/AQP_middleware/build_release && make -j12
# If DuckDB files changed:
cd /home/pei/Project/duckdb/build/release && make -j12
```
Run a single query (~1 second):
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

Build auxiliary in-memory data structures in the AQP middleware and execute sub-queries directly via optimized kernel functions, bypassing DuckDB's hash join and decompression overhead. **DuckDB/PostgreSQL remain completely unchanged** — they serve as data source at startup and SQL parser at query time. Native DuckDB/PostgreSQL does NOT benefit from these changes; all structures live in middleware memory. The reference code is from Bespoke (the fastest) in /home/pei/Project/BespokeOLAP/output/ and GenDB (a bit slower) in /home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries/ dirs.

**Full design**: `/home/pei/Project/AQP_middleware/storage_plan_design.md`

**Performance estimate**: 5-20x over current DuckDB, not the full 50-170x of BespokeOLAP. The gap: (1) temp table materialization between sub-queries — BespokeOLAP has zero intermediate materialization via single fused loop, (2) no query-specific pre-computed bitmaps (~negligible, see Step 4 notes), (3) generic interpreter vs. per-query compiled code, (4) no dimension-partitioned flat tables. To close the gap: Step 7 sub-query loop fusion (eliminates temp materialization, biggest win) + Step 7 kernel compilation (eliminates interpreter overhead) + Direction B (fewer/smaller temps).

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
- Step 6: Full integration + end-to-end JOB benchmark
- Step 7 (JIT): Kernel Compilation — compile SubQueryPlan into native code via LLVM JIT, replacing the interpreter. See details below.

**Known issues (Step 3)**:
- `std::vector<uint64_t> dummy` allocated per row in semi-join path (`sub_query_plan.cpp:457`). Minor inefficiency — should pass scan_row directly to EmitRow for FROM_SCAN-only output.
- Middleware overhead increased from 1.01s to 2.60s. Main cost: loading DuckDB temp results into FlatTable + building runtime CSR. Optimization opportunity: avoid FlatTable copy for DuckDB temps that won't be CSR-looked-up (i.e., only build CSR on columns that will actually be used as join keys in subsequent iterations).

**Known issues (Step 4)**:
- **Pre-computed bitmaps**: BespokeOLAP uses query-specific bitmaps (e.g., `us_movie_bitmap`) for O(1) existence checks before CSR probes. Our CSR semi-join lookup is already O(1) (~two array accesses). Estimated savings: ~7ms/query — **negligible** for now. Revisit if profiling shows CSR lookup as a bottleneck.
- **Dictionary encoding for strings**: BespokeOLAP converts string comparisons to integer dict-id comparisons (~10x faster per row). No current Direction A step covers this. Impact limited because most string filters are already resolved via dim cache (dimension tables) or are on large tables where memory access dominates comparison cost. Could add as part of Step 7 (JIT could compile string constants to dict-id lookups) or as a separate FlatTable enhancement.
- **Dimension-partitioned flat tables**: BespokeOLAP partitions tables by dimension FK (e.g., `movie_info.partitions[info_type_id]`). Our kernel scans the full flat table with a runtime filter. Estimated savings: 65-320ms per query touching large tables (see Step 7 sub-features). Only benefits kernel path — DuckDB/PostgreSQL SQL paths unaffected. Folded into Step 7.
- **Base×base guard**: When dim resolution reduces from 3+ tables to 2 base tables (no temps), we skip kernel and fall back to DuckDB. Reason: kernel linear scan of large base tables (e.g., cast_info 36M) is much slower than DuckDB's parallel vectorized hash join. The guard is conservative — revisit after Step 7 adds loop fusion + parallelism.

**Known issues (Step 5)**:
- **Latent CSR direction bug in `AnalyzeSubIR`** (`sub_query_plan.cpp` ~line 742): `GetCSR(scan_leaf->name, scan_col_name)` can return a CSR where `fk_table == scan_leaf->name`, meaning `Lookup()` returns row indices in the scan table instead of the lookup table. When `ExecuteSubQueryPlan` uses these indices to read FROM_JOIN columns from `step.joined_table`, it indexes into the wrong table — producing wrong values or heap-buffer-overflow. **Currently silent** because runtime CSR search (line 734) is checked first and succeeds for all temp-lookup cases; base×base pairs are guarded out. The same bug was fixed in `AnalyzeFinalIR` (CSR candidates now validated with `csr->fk_table == lookup_leaf->name`). Fixing `AnalyzeSubIR` the same way broke 3/3 configs because runtime CSRs have `fk_table=temp_name` and the column-name matching between IR and FlatTable doesn't always align. **Risk**: any future change that removes the base×base guard, adds new leaf types, or changes CSR search order could silently activate this bug. **Fix path**: add the direction check to `AnalyzeSubIR` but only for the `storage_plan->GetCSR()` calls (lines 742, 747), not the runtime CSR calls (lines 734, 754) which use a different naming convention.

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
- **Sub-query loop fusion**: Analyze the full query's sub-query sequence and generate a single fused function that chains CSR lookups without materializing intermediate temp tables. Eliminates temp allocation + memory writes (~1-2s middleware overhead) and runtime CSR building (~0.5s). This is the **single biggest remaining optimization** — BespokeOLAP processes entire queries in one fused loop with zero intermediate materialization. Requires analyzing the full split plan upfront to determine which sub-query outputs feed into which subsequent sub-query inputs.
- **Dimension-partitioned flat tables**: Partition large base tables by dimension FK columns at load time (e.g., `movie_info` by `info_type_id`, `cast_info` by `role_id`). The kernel scans only the matching partition instead of the full table. Scan reduction: `cast_info` 36M→4M (9x), `movie_info` 14.8M→2M (7x). Estimated savings: 65-320ms per query. Most impactful after loop fusion (fused loops scan base tables directly, so partitioning avoids scanning non-matching rows). Only benefits kernel path — DuckDB/PostgreSQL SQL paths use their own storage and are unaffected.
- **Dictionary encoding**: Compile string constant comparisons to integer dict-id lookups in the flat table.

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
