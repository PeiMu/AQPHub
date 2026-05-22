You are a DBMS JIT compilation expert. Your task is to iteratively improve execution time (ignore compilation time) for the JOB benchmark through JIT-related code changes. Especially with split strategy, e.g., node-based split, where we can collect real runtime information to help with JIT code generation and/or better plan. Find whatever information helps, then check how to collect them. Speedup 10x of the query wall runtime (ignore compilation time) with either level of JIT with node-based split, until no optimization can be found or heavy wall time queries take only 1 ms. You can also improve the split strategy. Approach this as Prof. Thomas Neumann or Matthias Jasny would: apply every low-level technique available, leave nothing on the table. Focus on the bottlenecks identified in the tracing output, but also reconsider the overall approach if the remaining gap is large.
Always check ## Helper Section again if got stuck.

## Codebase
- AQP middleware: /home/pei/Project/AQP_middleware (split execution, JIT compilation, IR-to-LLVM)
- DuckDB (forked): /home/pei/Project/duckdb (JIT dispatch in physical operators)
- JIT code generation: /home/pei/Project/AQP_middleware/src/jit/ir_to_llvm.cpp
- JIT dispatch (DuckDB side): physical_filter.cpp, physical_hash_join.cpp, physical_table_scan.cpp, physical_projection.cpp, physical_ungrouped_aggregate.cpp in /home/pei/Project/duckdb/src/execution/operator/
- JIT ABI: /home/pei/Project/duckdb/src/include/duckdb/execution/aqp_jit.hpp
- Split execution loop: /home/pei/Project/AQP_middleware/src/split/ir_query_splitter.cpp
- DuckDB adapter (JIT registration): /home/pei/Project/AQP_middleware/src/adapters/duckdb_adapter.cpp
- Benchmark queries: /home/pei/Project/benchmarks/imdb_job-postgres/queries/ (113 SQL files)
- Schema: /home/pei/Project/benchmarks/imdb_job-postgres/schema.sql
- Dataset: /home/pei/Project/benchmarks/imdb_job-postgres/csv

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

### Understanding `operator_exe.csv`

Per-operator CPU-seconds from DuckDB's QueryProfiler. Format: `query_file,subplan_idx,op_idx,op_type,op_category,time_sec,cardinality`. Iterations are separated by `# iter-N` markers. Each query's iterations use `# iter-1` through `# iter-15` (resetting per query since each query runs in its own process).

**op_category** (column 4) is the grouping key: TABLE_SCAN, HASH_JOIN, FILTER, PROJECTION, UNGROUPED_AGGREGATE, INVALID, COLUMN_DATA_SCAN, EMPTY_RESULT. DuckDB reports CPU-seconds summed across all 12 worker threads. For wall-clock comparison, divide by ~7-9x parallelization factor.

See current numbers and per-iteration history in /home/pei/Project/AQP_middleware/jit_optimization_claude.md.

## Constraints
- Only modify JIT-related code (ir_to_llvm.cpp, duckdb_adapter.cpp, JIT dispatch in DuckDB operators, param_config, split strategies in AQP_middleware).
- Do NOT touch non-JIT DuckDB execution code.
- Must return identical results to the original query.
- You may design new split strategies in AQP_middleware if they enable better JIT.
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
Check time_log.csv and operator_exe.csv for per-operator timing.

### Heavy test queries (focus on these for iteration):
You should always check the heaviest queries from the current version of jit in jit_optimization_claude.md.
Use tools like strace, ebpf, intel vtune profiler, perf, etc. to breakdown the performance and focus on the bottleneck.

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
0. Think fundamentally what is the correct optimization technique for the current bottleneck, e.g., memory-bound or I/O bound. Think in the Umbra way and in the Thomas Neumann way.
1. Read relevant source code for the optimization you're implementing
2. Make the code change
3. Build (`make -j12` in the appropriate build_release dir)
4. Quick-test on 2-3 target queries, check execution time in time_log.csv and operator_exe.csv
5. If faster: run correctness check on full JOB, then measure breakdown
6. If slower or neutral: analyze why, check the slowdown queries if it is noise, revert or adjust, try again. 
7. If some queries speedup but some others slowdown, check what's the differences and can we fix it fundamentally or at least guard it by some condition of the slowdown query's pattern.
8. Comparing queries where JIT with split strategy (e.g., node-based) speedup or slowdown compared to the none-split+none-JIT and compared to the last version of JIT, find out the fundamental reason.
9. Summarize: what changed, which queries improved/regressed, by how much, and update ## Current Status in this file. Then stop compact the conversation within Claude Code and continue the next step.

## Helper
- Hardware feature: {"cpu_cores": 12, "simd": "avx2", "total_memory_gb": 63, "disk_type": "hdd", "l3_cache_mb": 12}. You can run any tool to detect more on this server.
- You can use tools like strace, ebpf, intel vtune profiler, perf, and etc to gain more detailed information (e.g., compute bound or memory bound) when analyzing specific heavy query.
- Follow expert knowledge in /home/pei/Project/BespokeOLAP/conversations/prompts/expert_knowledge.txt and /home/pei/Project/GenDB/src/gendb/agents/code-generator/prompt.md and /home/pei/Project/GenDB/src/gendb/agents/query-optimizer/prompt.md
- Think fundamentally what information from the previous sub-plan execution can you use to generate better JIT code for the next sub-plan. Then check how to get this information from the codebase.
- It is okay that one optimization speedup some queries but slowdown the others and overall slowdown. Summarize the common pattern for improved queries and add a guard condition.  we'd better not have magic number in the code. can we calculate these thresholds with a formular or worst case a cost model by, e.g., selectivity, temp_card, span, etc from runtime information after execution the previous sub-query? but we cannot specify which query use which optimization or not use it.
- We care about the whole JOB benchmark, so prioritize the heaviest queries with the heaviest bottlenecks.
- You can check well-optimized queries under /home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries to find the gap, and learn from it.

## Current Status (Iter 21 — Step 1 complete: bottleneck re-profiled)

Iterations 1-18 optimized JIT. Iter 19 fixed measurement infrastructure. Iter 20 re-measured with pure in-memory baseline. Iter 21 completed Step 1 (bottleneck profiling). See jit_optimization_claude.md for full details.

**Performance** (in-memory, warm, avg of 10 runs — execution time only, ignoring JIT compile):

| Config                      | Execution (s) | Middleware (s) | JIT Compile (s) | Wall (s) | vs ns/nojit |
|-----------------------------|-------------:|---------------:|----------------:|---------:|------------:|
| none-split / none-jit       |         9.77 |          0.002 |               — |     9.77 | baseline    |
| none-split / pipeline-jit   |         9.39 |          0.012 |            8.80 |    18.20 | -0.38 (-3.9%) |
| node-based / none-jit       |        10.59 |          0.98  |               — |    11.57 | +0.82 (+8.4%) |
| node-based / pipeline-jit   |        10.33 |          0.99  |            2.19 |    13.51 | +0.57 (+5.8%) |

**perf stat (warm in-memory, repeat=3, heaviest queries)**:

| Query | Config          | IPC  | Cache miss% | LLC miss% | Kernel% | CPU/Wall |
|-------|-----------------|------|------------|-----------|---------|----------|
| 16b   | nb-jit          | 0.82 | 17.2%      | 18.8%     | 8.5%    | 4.2x     |
| 16b   | ns-nojit        | 0.77 | 17.5%      | 22.3%     | 8.1%    | 8.4x     |
| 8c    | nb-jit          | 0.68 | 26.6%      | 31.2%     | 11.8%   | 4.3x     |
| 8c    | ns-nojit        | 0.67 | 22.5%      | 28.2%     | 10.6%   | 7.6x     |
| 19d   | nb-jit          | 0.89 | 19.0%      | 22.7%     | 7.9%    | 3.4x     |

**Key findings (Iter 21)**:
1. **Old bottleneck was disk I/O**: kernel% dropped from 40% → 8-12%, IPC rose from 0.64 → 0.68-0.89. Confirmed: Iter 0-18 measurements were contaminated.
2. **Memory-bound, not compute-bound**: IPC 0.68-0.89, LLC miss 19-31%. Hash join dominates (40-60% of CPU time). Cache-line pressure from large hash tables.
3. **Split halves parallelism**: CPU/Wall is ~4x for node-based vs ~8x for none-split. Sub-plans execute sequentially with fewer parallel opportunities per sub-plan. This is a fundamental architectural cost of splitting.
4. **Large temp tables are the root cause**: Top regressions (16b +162ms, 8c +155ms, 6d +96ms, 16c +93ms, 9d +91ms) all produce 1-4M row intermediates. The final subplan then hash-joins these massive temps, which is slower than DuckDB's single-pass optimized plan.
5. **JIT helps modestly**: -0.25s total across all queries (-2.4%). Biggest wins: 9a (-19ms), 26c (-14ms), 9d (-14ms). Biggest losses: 8c (+13ms), 13d (+8ms).

**Top 5 heaviest queries (nb-jit exe time, with operator breakdown)**:
- **16b** (483ms): subplan 5 alone = 186ms wall (hash-join 3.71M rows). Temp buildup: 42K→1.15M→68K→2.83M→3.71M.
- **8c** (395ms): subplan 3 = 183ms wall (hash-join 2.49M rows). Temp buildup: 1.15M→2.34M→2.49M.
- **19d** (293ms): distributed across 8 subplans, each moderate.
- **9d** (245ms): 6 subplans, subplan 4 heaviest (36ms wall).
- **10c** (244ms): subplan 2 = 142ms wall (filter+scan 12.7M rows).

### Analysis tool
```bash
python3 /home/pei/Project/AQP_middleware/measure/analyze_breakdown.py [job_result_dir]
```

## Next Steps

### Step 2: Reduce split overhead on heavy queries (Direction 1)

The #1 bottleneck is now clear: **node-based split produces unnecessarily large temp tables** that cause 2x more hash-join work in later subplans. The split order doesn't consider selectivity — it joins tables in graph-traversal order, not selectivity order.

**Root cause analysis**:
- 16b: first subplan joins `movie_companies` (2.6M) × `company_name` (85K) → 1.15M rows temp. This is only 44% selective. Better: start from the most selective join first.
- 8c: same pattern. First subplan produces 1.15M rows, cascading to 2.34M and 2.49M intermediates.
- None-split avoids this entirely because DuckDB's optimizer picks optimal join order for the whole query.

**Approach**:
1. Read the split strategy code (`node_based_splitter.cpp`, `ir_query_splitter.cpp`) to understand how split points are chosen
2. Identify where selectivity information (estimated or runtime) could influence split order
3. Option A: Reorder subplans by estimated selectivity (using DuckDB's cardinality estimates)
4. Option B: After each subplan, check temp cardinality; if too large, consider merging remaining subplans
5. Option C: Skip splitting for queries where DuckDB's optimizer already picks a good plan (heuristic: if estimated total cardinality < threshold, run as single plan)

**Files**: `node_based_splitter.cpp`, `ir_query_splitter.cpp`, DuckDB's `QuerySplit`

**Target queries**: 16b, 8c, 16c, 6d, 9d (top 5 regressions, total +597ms overhead vs none-split)

### Step 3: Re-enable/test previously disabled JIT optimizations

With warm buffer pool, the bottleneck profile changed (compute→memory). Re-test disabled optimizations:
- Fused build/probe (disabled in Iter 14-15)
- Inline hash (disabled in Iter 16)
- Prefetch (disabled in Iter 12)
These were disabled when TABLE_SCAN dominated (disk I/O). Now that HASH_JOIN dominates (40-60% of CPU), they may help.

### Direction 2: Whole-Pipeline JIT with Hybrid DuckDB Runtime (architectural, high impact — needs further discussion)

**Status**: Deferred until Step 2 and Step 3 show diminishing returns. The current gap is +0.57s (+5.8%) for node-based/pipeline-jit vs none-split/none-jit. If Step 2 can reduce the split overhead (currently +0.82s), we may reach parity or better without architectural changes. However, this is the most promising path to the 10x goal.

**Goal**: Bypass DuckDB's vectorized execution engine for sub-plans where compiled code wins. Compile entire pipeline stages into single tight loops — no virtual dispatch, no per-tuple function calls, no operator boundaries.

**Why this is the 10x path**: GenDB-style compiled queries (same approach) routinely beat DuckDB by 3-10x because:
- Zero runtime overhead: no query parser, no buffer pool lookups, no type dispatch
- Entire pipeline compiles into one function: scan → filter → hash-build → probe → aggregate
- Compiler sees full data flow: constant folding, branch elimination, auto-vectorization
- Direct memory access to columnar data (mmap or pointer)

**Hybrid design with DuckDB (hide compilation latency)**:

```
Sub-plan 1: Execute in DuckDB (immediate, no compilation wait)
             Meanwhile: LLVM-compile sub-plans 2..N in background thread
Sub-plan 2: If compiled binary ready AND plan unchanged after MiddleOptimize → use compiled
            Else → fall back to DuckDB
Sub-plan 3..N: Same check — compiled if ready, DuckDB otherwise
```

**Storage bridge — dual-format loading (Option C)**:

The compiled JIT code needs direct pointer access to flat column arrays (like GenDB's `mmap` columns or BespokeOLAP's Arrow tables). DuckDB's internal storage is compressed — even with `:memory:`, data is in compressed column segments that JIT code cannot scan with raw pointers.

Solution: load data into two formats at startup:
1. DuckDB `:memory:` tables — for the fallback DuckDB vectorized execution path (already implemented in Direction 0)
2. Flat column arrays — for the JIT compiled path. Options:
   - `mmap` per-column binary files (GenDB style, see `/home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries/Q10a/iter_0/q10a.cpp`)
   - Arrow arrays via DuckDB's built-in Arrow export
   - Or register flat arrays as a custom DuckDB table function (like existing `scan_temp_collection` but for base tables)

Memory cost: ~2x for base tables (~3.5 GB extra on 63 GB machine — negligible).

For temp tables: already `ColumnDataCollection` (uncompressed flat vectors) — both DuckDB and JIT code can access directly. This is already solved.

**Plan stability**: After `QuerySplit::Split()`, all remaining sub-plan structures are known. `MiddleOptimize` after each iteration mainly adjusts join order based on cardinality. We can:
1. Compile a parameterized version (scan/probe sides as parameters)
2. After re-optimization, check if structure changed — if yes, fall back to DuckDB
3. Inject runtime constants (BF data, range bounds, temp cardinality) into the compiled code via LLVM constant folding

**What we already have**:
- `ir_to_llvm.cpp`: LLVM IR generation from our IR representation
- Sub-plan IR: available immediately after `SplitIR()`
- LLVM OrcJIT: supports concurrent background compilation
- Runtime info bridge: temp table cardinality, min/max, BF data
- `AQPJITContext` / `aqp_jit.hpp`: JIT dispatch ABI between middleware and DuckDB

**What needs to be built**:
1. Flat column array loading alongside DuckDB `:memory:` tables (Direction 0 gives us the DuckDB side; this adds the flat arrays for JIT)
2. Extend `ir_to_llvm.cpp` to compile whole pipeline stages (scan+filter+probe+aggregate) instead of per-operator scalar functions
3. Build a morsel-driven parallel scan that reads flat column arrays directly
4. Implement hash table build/probe in compiled code (can reuse DuckDB's hash table format or build our own cache-friendly one)
5. Background compilation thread with validity checking
6. Fallback path to DuckDB when compilation isn't ready or plan changed

**Key design principles** (from expert knowledge):
- Critical path = scan of largest table. Minimize per-row work on this path (expert_knowledge #1, #6, GenDB code-generator Step 2)
- Data structure fitness: hash tables must fit LLC (12MB). If not, partition or use BF pre-filter (GenDB query-optimizer Q2, Category B)
- Work elimination: BF before probe, zone maps to skip segments, late materialization (GenDB query-optimizer Category C)
- Keep hot code compact for icache (expert_knowledge #9, #18 — proven in Iter 18)

## What I Expect
- Start with Step 2: read the split strategy code, understand how subplans are ordered, and find where to inject selectivity-based ordering.
- After each change, quick-test on 16b and 8c (the two biggest regressions), then correctness check.
- After Step 2, re-run `python3 measure/analyze_breakdown.py` to compare.
- Show uncertainty clearly. Flag assumptions. Verify with measurements.
- After each iteration, report: what changed, which queries improved, which regressed, net effect on total JOB time.
