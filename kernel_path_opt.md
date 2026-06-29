# Kernel Path Optimization Plan

Consolidated analysis and implementation plan for reducing wall time from 14.2s to below 9.5s (DuckDB baseline). The kernel path already achieves 25.8% faster execution (7.1s vs 9.5s), but middleware overhead (6.5s) makes wall time 49% worse. This document defines the optimization steps.

## Current State

### Baselines

**Vanilla DuckDB** (none-split / none-jit) = 9,532ms wall (all execution, no MW overhead).

**Step 7.2** (node-based / query-kernel CSR-based) = 14,202ms wall:
- execute: 7,140ms, MW overhead: 6,467ms, JIT compile: 596ms
- 69 queries pure kernel, 44 mixed (DuckDB fallback). 678 total iterations.

### After Pipeline Kernel + Phase 1 (A+B+C) Optimizations

**Pipeline kernel** (node-based / pipeline-jit, interpreted C++) = **13,343ms wall**:

| Column | Time (ms) | vs Step 7.2 |
|--------|-------:|-------:|
| prepare_middleware | 8.0 | -0.9 |
| extract_next_sub-IR | 700.6 | — |
| generate_sub-SQL | 300.2 | +78.5 |
| jit_compile | 152.8 | -15,875 |
| **execute_sub-SQL** | **9,987.7** | — |
| extra_materialization | 31.3 | -5.2 |
| update_IR | 14.4 | -3.0 |
| generate_final_sub_sql | 524.4 | +54.4 |
| jit_compile_final | 85.0 | -3.5 |
| **final_exe** | **1,536.3** | +63.1 |
| show_output | 2.4 | -3.0 |
| **TOTAL WALL** | **13,343** | -30,135 vs old PK |
| **TOTAL EXEC** | **11,524** | -14,376 vs old PK |
| **TOTAL COMPILE** | **238** | -15,879 vs old PK |
| **TOTAL MW OVERHEAD** | **1,581** | +120 vs old PK |

Note: "vs old PK" compares against the interpreted pipeline kernel A/B baseline (`pk_jit_ab/interpreted.csv`), which included 16,116ms of wasted `CompilePipelineKernel()` time in the jit_compile column.

**vs DuckDB baseline**: 13,343ms / 9,532ms = 1.40x slower (gap: 3,811ms).

**Top 15 queries by wall time**: 8c(396), 9d(319), 17a(286), 29c(239), 19a(237), 19d(236), 30a(230), 16b(224), 24b(213), 7c(212), 19c(211), 24a(207), 22d(206), 22a(204), 29a(202).

**Execution improvement breakdown (Phase 1)**:
- 87 queries improved (>2%), 17 regressed, 9 neutral
- 3a/3b/3c: 4,400ms → 65ms each (66-988x) — dimension table probes via DIRECT method
- 29a/29b/29c: 30-48ms improvement (1.2-1.3x) — semi-join Contains via POINT/LINEAR
- 8c: +96ms regression (0.75x) — needs investigation
- 6a/6e: +28ms each — needs investigation

## Key Findings from Analysis

### 0. Pipeline Kernel LLVM JIT — Failed, Removed

**A/B test** (2026-06-03): LLVM-compiled probe loops (O1) were 24% slower on execution, 16% slower on wall time than interpreted C++ (GCC -O2). 107/113 queries regressed. Top regressions: 20a 0.24x, 10c 0.27x, 20b 0.31x. Causes: poor LLVM O1 code quality, excessive memory indirection, missing auto-vectorization, function call overhead per-row.

**Decision**: All pipeline kernel optimizations go into interpreted C++ compiled by GCC -O2. `CompilePipelineKernel()`, `AQPPipelineKernelView`, and all LLVM JIT code for the pipeline kernel have been removed. Compile time dropped from 16,116ms to 238ms (remaining is DuckDB expr/operator JIT for fallback iterations).

**A/B data**: `measure/job_result/pk_jit_ab/{interpreted,jit}.csv`.

### 1. DuckDB JIT Levels (expr / operator / pipeline)

Measured on none-split (113 JOB queries, avg of 10 warm runs):

| Config | Execution | JIT compile | Wall |
|--------|-----------|-------------|------|
| none-jit | 9,529ms | 0 | 9,532ms |
| expr-jit | 8,849ms (-7.1%) | 1,579ms | 10,436ms |
| operator-jit | 8,804ms (-7.6%) | 1,574ms | 10,386ms |
| pipeline-jit | 8,857ms (-7.1%) | 8,786ms | 17,656ms |

- expr→operator: -45ms execution (-0.5%), same compile time. Marginal.
- operator→pipeline: +53ms execution (+0.6%), +7,200ms compile. **Net negative.**
- Pipeline-jit recompiles every repeat (JIT context reset between repeats, cache disabled in measurements). The 8,786ms is real per-query cost.
- **Conclusion**: expr-jit is the sweet spot for the DuckDB path. operator-jit adds negligible value. pipeline-jit is harmful. In node-based split, the kernel handles ~69% of iterations (no DuckDB JIT needed), so DuckDB JIT matters only for the ~31% fallback iterations.

### 2. Runtime CSR Build — Query Kernel Path

CSR build was the dominant MW cost (3,291ms, 56.8%) in the query-kernel path (`--jit-level=query`). The pipeline kernel (Step A) eliminated this entirely by using hash tables instead. See `query_jit_opt.md` for the full CSR cost analysis and optimization proposals (Steps B/C/D).

### 3. Reference Implementations (BespokeOLAP, GenDB)

Both use: flat column arrays + CSR indexes + no hash joins + single fused function per query (AOT compiled C++). Zero intermediate materialization. 50-170x faster than DuckDB.

**Neither resembles "pipeline-level kernel"**. They are query-level: one function handles the entire query with nested CSR probes. No pipeline breakers, no hash tables.

**Our advantage over BespokeOLAP/GenDB**: runtime adaptivity. They assume correct statistics at code-generation time (fixed join order, hardcoded table sizes). We collect actual cardinality after each sub-plan and re-optimize. For new/skewed workloads, our approach is more general.

**Our disadvantage**: MW overhead from split boundary (temp materialization + CSR build between sub-queries).

### 4. Threading

- DuckDB uses all 12 cores for vectorized execution (parallel hash join, parallel scan).
- Kernel uses OpenMP (up to 12 threads) for scans ≥ 10K rows (`sub_query_plan.cpp:26`).
- Adding MW threads conflicts with both. Resolution: pin DuckDB to N-2 threads, or only run MW threads during kernel execution (single-threaded for small scans).

## Implementation Plan

### Step A: Pipeline-Level Kernel with Hash Join — DONE

**Status**: Fully implemented and measured. All 470 pipeline kernel iterations handled. 0 DuckDB fallback for kernel-eligible sub-plans. Pipeline kernel code: `src/kernel/pipeline_kernel.cpp`, `include/kernel/pipeline_kernel.h`.

**JIT-level routing** (implemented in `ir_query_splitter.cpp`):

| `--jit-level=` | Middleware kernel | DuckDB fallback JIT |
|---|---|---|
| `expr` | none | expr-jit |
| `operator` | none | operator-jit |
| `pipeline` | **pipeline kernel** (hash build/probe, no CSR) | expr/operator-jit |
| `query` | **query kernel** (CSR-based, `AnalyzeSubIR`) | expr/operator-jit |

**Key design**:
- **Analyze** (`AnalyzePipelineKernel`): Reuses `CollectLeaves`, `CollectJoinEdges`, `CompileAllLeafFilters`, `ResolveDimensions`. Rejects aggregates, non-inner joins, missing FlatTables, non-INT32 keys. Picks scan table = largest by row count. BFS from scan table to order join steps. Each non-scan leaf → `PipelineJoinStep` with build table, key columns, filters, semi/inner flag.
- **Build** (`ExecutePipelineKernel`): Build chained `HashJoinTable` per join step. Fibonacci hash, power-of-2 capacity, ~50% load factor.
- **Probe**: Scan largest table with OpenMP (`dynamic, 8192` schedule). Semi-join steps use `Contains()`, inner-join steps use iterative DFS with explicit stack (`ProbeStepsIterative`). Output via per-thread `FlatTableBuilder` + `MergeBuilders`.
- **No CSR built on output**: `RegisterKernelResult(result, name, false)`. Future iterations build hash tables on temps at execution time.

**Join patterns handled**: single-table filtered scan, 2+ table inner joins, star joins (all edges to scan), chain joins (A→B→C), snowflake (star with branches). Handles all patterns the query kernel handles plus the ~31% that previously fell back to DuckDB.

**Result**: Eliminated 3,291ms CSR build overhead, 966ms DuckDB SQL generation, 596ms DuckDB JIT compile for fallback iterations. MW overhead dropped from 6,467ms to 1,581ms. But execution increased from 7,140ms (CSR O(1) lookups) to 11,524ms (hash probes) — net wall improvement of ~860ms vs step_7_2.

### Steps B/C/D: Query Kernel CSR Optimizations — SUPERSEDED

Moved to `query_jit_opt.md`. These targeted CSR build overhead in the query-kernel path (`--jit-level=query`), which the pipeline kernel (Step A) eliminated entirely.

### Step E: Pipeline Kernel Probe Optimizations — Phase 1 DONE, Phases 3-4 NEXT

The remaining execution gap (11,524ms vs DuckDB 9,529ms) is in the hash probe hot loops. These optimizations improve probe throughput without changing the algorithm.

#### Phase 1 (A+B+C): Adaptive Join + Direct Map + Unique Key Skip — DONE

**Implemented in**: `src/kernel/pipeline_kernel.cpp`, `include/kernel/pipeline_kernel.h`

**A. Adaptive join method based on actual HT size** — after HT build, select probe method:

| HT size | `ProbeMethod` | Why faster |
|---------|--------------|-----------|
| 0 | `SKIP` | Return immediately |
| 1 | `POINT` | Direct key compare, no hash, no chain walk |
| 2-15 | `LINEAR` | Linear scan of keys array, sequential access |
| 16+ | `HASH` | Standard chained hash probe |

**B. Direct-mapped array for small key ranges** — when `max_key - min_key < 10,000`:

Allocate `uint32_t direct_map[range]`, map `key → row_id`. Probe = `direct_map[key - min]`. No hash, no chain. Fits in L1 cache (< 40KB). Triggered for dimension tables:
- kind_type: 7 rows, range [1,7] → 28 bytes
- company_type: 4 rows, range [1,4] → 16 bytes
- role_type: 12 rows, range [1,12] → 48 bytes
- info_type: 113 rows, range [1,113] → 452 bytes

JOB impact: these tables are probed millions of times per query. `direct_map[key]` vs hash chain walk = ~5x throughput.

**C. Unique key (PK) → skip chain walk after first match** — for inner joins where build key is a PK (detected via CSR metadata: `csr.pk_table == build_table && csr.pk_column == build_col`):
- In `ProbeStepsIterative`: use single `Lookup()` instead of chain iteration
- In `AdvanceStep`: return false immediately (no backtracking needed)
- Field: `PipelineJoinStep::build_key_unique` set during `AnalyzePipelineKernel`

**Code changes**:
- `HashJoinTable`: added `TryBuildDirectMap()`, `SelectProbeMethod()`, per-method `Contains`/`Lookup`/`ForEach` variants, `ForEachHashUnique` (early-exit)
- `PipelineJoinStep`: added `ProbeMethod probe_method`, `bool build_key_unique`
- `AnalyzePipelineKernel()`: iterates `StoragePlan::GetCSRMap()` to detect PK uniqueness
- `ExecutePipelineKernel()`: calls `TryBuildDirectMap()` + `SelectProbeMethod()` after HT build
- `ProbeStepsIterative()`: dispatches on `probe_method`, uses `Lookup()` for unique/non-HASH inner joins
- `StoragePlan`: added `GetCSRMap()` accessor

**Measured result**: execution improved 2.25x (25,900ms → 11,524ms). 87 queries improved, 17 regressed. 3a/3b/3c improved 66-988x via DIRECT method. 8c regressed 0.75x (needs investigation).

#### Phase 3 (E): Software prefetch — DONE

For HASH probe on larger HTs, prefetch the bucket entry N rows ahead:

| HT memory (size*12 + capacity*4) | Prefetch distance |
|-----------------------------------|-------------------|
| < 32KB (L1) | 0 (none) |
| 32KB - 256KB (L2) | 4 rows ahead |
| 256KB - 8MB (L3) | 8 rows ahead |
| > 8MB (DRAM) | 16 rows ahead |

Implementation:
- Compute HT memory footprint after build
- Select prefetch distance per join step
- In probe loop, at iteration i, prefetch bucket for row i+distance:
  `__builtin_prefetch(&ht_buckets[hash(scan_keys[row + dist]) & mask], 0, 1);`
- Only for HASH method (POINT/LINEAR/DIRECT don't need prefetch)

Files: `src/kernel/pipeline_kernel.cpp` — add prefetch distance field to step, add prefetch in scan loop.

**Measured savings**: -503ms wall (-3.8%). Data: `measure/job_result/kernel_process_5/`.

#### Phase 4 (D): Vectorized batch processing + SIMD — DONE

Restructure `ExecutePipelineKernel` from row-at-a-time to batch processing:

**Current** (row-at-a-time):
```
for each scan_row:
  apply scan filters
  for each join step: hash probe → match/no match
  if all matched: emit output row
```

**Vectorized** (batch-at-a-time, BATCH_SIZE = 256):
```
for each batch of 256 scan rows:
  1. Evaluate scan filters on batch → qualifying_indices[]
  2. For join step 0:
     a. Gather keys from qualifying rows
     b. Compute hashes (SIMD: 8x i32 multiply with AVX2)
     c. Prefetch HT buckets for all keys
     d. Probe chains for all keys → match_indices[]
     e. Compact matches → input for step 1
  3. For join step 1..N: repeat (2) on surviving rows
  4. Emit output for all surviving rows
```

SIMD usage (AVX2, `--jit-simd=auto`):
- Hash: `_mm256_mullo_epi32(keys, fib_constant)` → 8 hashes at once
- Key compare: `_mm256_cmpeq_epi32(ht_keys, probe_keys)`
- Filter: `_mm256_cmpgt_epi32(col_values, threshold)`
- Compaction: `_mm256_movemask_epi8` + `pext` for gathering qualifying indices

Key design decisions:
- BATCH_SIZE = 256 (fits in L1 with working buffers)
- Semi-join: batch produces bitmap → compact for next step
- Inner-join with 1:N: handle fan-out within batch (expand matches)
- Fallback: if plan has unsupported features (complex inner-join DFS), fall back to row-at-a-time

Files: `src/kernel/pipeline_kernel.cpp` — new `ExecutePipelineKernelVectorized()`, batch filter eval, batch hash probe with SIMD.

**Measured savings**: -714ms wall (-5.5%) vs Phase 3. Cumulative Phase 3+4: -1,217ms wall (-9.1%) vs Phase 1. Data: `measure/job_result/kernel_process_6/`.

### Step F: Additional Optimizations (Priority 7+)

After Phase 3-4, re-measure and prioritize:
- **Dimension-partitioned flat tables**: Partition large tables by low-cardinality FK. Scan reduction: cast_info 36M→4M (9x). Est. 65-320ms per query.
- **Dictionary encoding**: String comparisons → integer comparisons. Est. significant for movie_info/cast_info filters.
- **Better split strategy**: Reorder sub-plans by selectivity. Fewer/smaller temps → less overhead.
- **Multi-threaded MW pipeline**: Overlap HT build with next iteration's extract+analyze on a separate thread.
- **Kernel JIT compilation**: LLVM-compile SubQueryPlan to eliminate interpreter overhead (~300ms). A/B test showed this is negative with current approach — needs fundamentally different IR generation strategy to beat GCC -O2.

## Wall Time Reduction Projection

| Step | Target | Status | Measured Wall |
|------|--------|--------|----------:|
| step_7_2 baseline | Query kernel (CSR) | Done | 14,202ms |
| A: Pipeline kernel | Eliminate DuckDB fallback + CSR build | Done | — |
| Phase 1 (A+B+C) | Adaptive join + direct map + unique key | Done | **13,343ms** |
| Phase 3 (E) | Software prefetch for large HTs | **Done** | **12,889ms** |
| Phase 4 (D) | Vectorized batch + SIMD | **Done** | **12,175ms** |
| Step F | Additional (partitioning, dict encoding, etc.) | Next | est. ~10,000ms |

**Current gap**: 12,175ms vs DuckDB baseline 9,532ms = 2,643ms (27.7% slower).
**Breakdown**: execution 10,479ms (vs 9,529ms baseline, +950ms) + MW overhead 1,462ms + compile 234ms.
**Target**: wall time < 9,532ms (DuckDB baseline).

## Code Organization

| Directory | Purpose |
|-----------|---------|
| `src/storage/` | Flat tables, CSR indexes, sorted indices, dimension cache, storage plan |
| `src/kernel/sub_query_plan.cpp` | Query-level kernel: AnalyzeSubIR + ExecuteSubQueryPlan + AnalyzeFinalIR |
| `src/kernel/pipeline_kernel.cpp` | Pipeline kernel: AnalyzePipelineKernel + ExecutePipelineKernel + HashJoinTable |
| `include/kernel/pipeline_kernel.h` | PipelineKernelPlan, PipelineJoinStep, ProbeMethod enum |
| `src/jit/ir_to_llvm.cpp` | DuckDB JIT: expr/operator compilation for DuckDB fallback path |
| `src/split/ir_query_splitter.cpp` | Split loop: extract → analyze → execute → materialize → update |
| `src/adapters/duckdb_adapter.cpp` | DuckDB integration: loading, JIT registration, temp management |

## Measurement & Validation

**Measurement data**:
- Phase 1 results: `measure/job_result/duckdb_node-based_pipeline_o1_none_breakdown_time_log.csv`
- LLVM JIT A/B data (historical): `measure/job_result/pk_jit_ab/{interpreted,jit}.csv`
- Step 7.2 data: `measure/job_result/step_7_2/`

**Breakdown CSV parser**: `analyze_middleware_breakdown(csv, has_jit=True, is_node_based=True)` in `/home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py`

**Per-row CSV format**: head (4 cols) + N iterations (6 cols each) + tail (4 cols).

Head: `prepare_middleware, read_sql, parse_sql, preprocess`

Per iteration (6 cols):
1. `extract_next_sub-IR`: SplitIR()
2. `generate_sub-SQL`: kernel = EnsureReferencedTemps + Analyze; DuckDB = failed kernel attempts + GenerateSQL
3. `jit_compile`: kernel = 0.000; DuckDB = LLVM compile (written by DuckDB adapter)
4. `execute_sub-SQL`: kernel = ExecutePipelineKernel; DuckDB = Execute (written by DuckDB adapter)
5. `extra_materialization`: kernel = RegisterKernelResult; DuckDB = extra_materialize (written by DuckDB adapter)
6. `update_IR`: UpdateRemainingIR + temp registration

Tail: `generate_final_sub_sql, jit_compile_final, final_exe, show_output`

Timer uses `chrono_toc` which resets `*start_time = current_time` — each column is a lap timer, no overlap.

**Quick test**: `bash run_job.sh duckdb node-based pipeline o1 none` + diff against golden.

**Full measurement**: `bash measure_breakdown_time_job.sh duckdb node-based pipeline o1 none` (~28 min).

**Target queries**: 8c (worst regression), 9d, 16b, 29c (many iters), 19d.
