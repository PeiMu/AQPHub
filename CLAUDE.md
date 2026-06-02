You are a DBMS JIT compilation expert. Your task is to iteratively improve wall time (including execution time, compilation time, and middleware overhead) for the JOB benchmark through JIT-related code changes (the status of the current iteration is in section "## Current Performance"). Especially with split strategy, e.g., node-based split, where we can collect real runtime information to help with JIT code generation and/or better plan. Find whatever information helps, then check how to collect them. Targeting speedup 10x of the query wall runtime (ignore compilation time) with either level of JIT with node-based split, until no optimization can be found or heavy wall time queries take only 1 ms. You can also improve the split strategy. Approach this as Prof. Thomas Neumann or Matthias Jasny would: apply every low-level technique available, leave nothing on the table. Focus on the bottlenecks identified in the tracing output, but also reconsider the overall approach if the remaining gap is large.
Always check ## Helper Section for optimization techinique.
When discussion, think in Prof. Thomas Neumann or Matthias Jasny way.

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

## Current Status (Iter 30 — Steps 1-6.5 DONE, wall time optimization phase)

See `jit_optimization_claude.md` for full per-iteration history (Iter 1-30).

**Performance** (step_7_2, in-memory, warm, avg of 10 runs):

| Config | Execution | MW Overhead | JIT Compile | Wall | vs baseline |
|--------|----------:|------------:|------------:|-----:|------------:|
| none-split / none-jit (baseline) | 9,529ms | 3ms | — | 9,532ms | — |
| node-based / pipeline-jit | 7,140ms | 6,467ms | 596ms | 14,202ms | +49% wall |

Execution is 25% faster than baseline, but MW overhead (6,467ms) makes wall time 49% worse. The bottleneck is now MW overhead, not execution.

**MW overhead breakdown** (6,467ms):
- CSR build on temp results: 3,291ms (50.9%)
- Lazy FlatTable+CSR for DuckDB fallback: 1,752ms (27.1%) — in generate_sub-SQL + generate_final columns
- SplitIR traversal: 661ms (10.2%)
- DuckDB JIT compile: 596ms (9.2%)
- Other: 167ms (2.6%)

**Storage plan components** (Steps 1-6, all DONE): Flat column arrays, base CSR indexes, runtime CSR on temps, dimension cache, sorted indices, inverted indices, LIKE support. See `storage_plan_design.md` for design details.

**What works**: Query-level kernel handles ~69% of sub-query iterations via CSR-based joins on flat arrays. ~31% fall back to DuckDB (patterns with 2+ base tables, missing CSR relationships, complex projections).

## Next Steps — Wall Time Optimization

**Full plan with analysis, cost breakdown, and design details**: `kernel_path_opt.md`

**Goal**: Reduce wall time from 14,202ms to below 9,532ms (DuckDB baseline).

Implementation order (from `kernel_path_opt.md`):

1. **Step A — Pipeline-level kernel** (est. -2,000 to -3,000ms): Compile DuckDB fallback iterations into native code with hash join on flat arrays, eliminating DuckDB SQL generation (966ms), DuckDB JIT (596ms), and lazy FlatTable+CSR build (1,752ms). Handles the 31% fallback iterations.

2. **Step B — Sparse CSR for small temps** (est. -1,500 to -2,000ms): Use hash-based CSR for temps < 50K rows. Eliminates 19-31MB memset per CSR build on small temps (90%+ of iterations).

3. **Step C — Byte-map for semi-join patterns** (est. -500ms): When runtime CSR is only used for existence checks (1-column temp), replace with byte-map.

4. **Step D — Skip last-iteration CSR** (est. -500ms): Don't build CSR on last iteration's output.

5. **Step E — Multi-threaded pipeline** (est. -300ms): Overlap CSR build with next iteration's extract+analyze.

6. **Step F — Kernel JIT compilation** (est. -300ms): LLVM-compile SubQueryPlan to eliminate interpreter overhead.

7. **Step G — Additional**: Dimension-partitioned flat tables, dictionary encoding, better split strategy.

## What I Expect
- Follow `kernel_path_opt.md` step by step, verify each step's effect independently via breakdown measurement.
- After each change, quick-test on target queries (16b, 29c, 8c, 19d, 9d), then correctness check.
- Show uncertainty clearly. Flag assumptions. Verify with measurements.
- After each step, report: what changed, which queries improved/regressed, net effect on total JOB wall time.
