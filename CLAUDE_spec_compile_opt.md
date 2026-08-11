You are a DBMS JIT compilation expert. Your task is to iteratively improve wall time (including execution time, compilation time, and middleware overhead) for the JOB benchmark through JIT-related code changes (the status of the current iteration is in section "## Current Performance"). Especially with split strategy, e.g., node-based split, where we can collect real runtime information to help with JIT code generation and/or better plan. Find whatever information helps, then check how to collect them. Targeting speedup 10x of the query wall runtime (ignore compilation time) with either level of JIT with node-based split, until no optimization can be found or heavy wall time queries take only 1 ms. You can also improve the split strategy. Approach this as Prof. Thomas Neumann or Matthias Jasny would: apply every low-level technique available, leave nothing on the table. Focus on the bottlenecks identified in the tracing output, but also reconsider the overall approach if the remaining gap is large.
Always check ## Helper Section for optimization techinique.
When discussion, think in Prof. Thomas Neumann or Matthias Jasny way.

## Goal

## Analysis 

### Root cause

### Fix

## Implementation Plan

## Key Code Locations (Reference)

### AQPHub Core

| File | What |
|------|------|

### SimplestIR (AQPHub's intermediate representation)

| File | What |
|------|------|
| `third_party/IR_SQL_Converter/inc/simplest_ir.h` | All IR node class definitions (1633 lines) |
| `third_party/IR_SQL_Converter/src/duckdb_plan_to_ir.cpp` | DuckDB logical plan → SimplestIR |
| `third_party/IR_SQL_Converter/src/ir_to_sql.cpp` | SimplestIR → SQL string (reference for how IR is walked) |
| `third_party/IR_SQL_Converter/inc/cpp_interface.h` | Public API: ConvertDuckDBPlanToIR, ConvertIRToSQL |

## Repositories

- **AQPHub**: `/home/pei/Project/AQPHub` (branch: `topdown_fix`)
- **DuckDB (patched)**: `/home/pei/Project/duckdb`
- **JOB queries**: `/home/pei/Project/benchmarks/imdb_job-postgres/queries/`
- **JOB schema**: `/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql`
- **DuckDB database**: `/home/pei/Project/duckdb/measure/imdb.db`

Build commands:
```bash
# Middleware (debug / release)
cd /home/pei/Project/AQPHub && cmake --build build_debug -j$(nproc)
cd /home/pei/Project/AQPHub && cmake --build build_release -j$(nproc)
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

## Implementation Status: Spec-JIT Compilation Time Reduction

### Current state: V6 (+4-8% overhead, down from +94.9%)

Config: `duckdb topdown query-jit tpde cache=single-run-template spec=recompile`

### Active changes (V6)

1. **Spec bg compile uses config compile-mode** (was hardcoded LLVM O2):
   - `src/split/ir_query_splitter.cpp`: `int spec_backend = spec_compile_mode` at 2 sites
     (node-based LaunchSpeculativeCompile, PG LaunchSpeculativeCompilePG).
     Third site (topdown PrecomputeNextExtraction) uses `config_.compile_mode`.
   - Effect: +94.9% → +10.3%

2. **Persistent object cache when spec-jit active** (don't clear between iterations):
   - `src/aqp_middleware.cpp`: skip `ClearObjCache()` when `config.spec_jit != 0`
     (2 sites: RunBenchmark loop and single-query repeat loop)
   - Effect: bg compile hits cache on iteration 1+ → ~0ms codegen instead of ~0.66ms

3. **Spec compiler cache mode matches inline** (was hardcoded to strict=1):
   - `src/split/ir_query_splitter.cpp`: `spec_compiler->SetCache(duck->GetJitCache())`
     (2 sites: EnsureSpecCompiler and EnsureCrossCompiler)
   - Ensures spec and inline share cache key format (template mode=2)

4. **Deferred CheckSpec for topdown** (check inside adapter, not before ExecSQL):
   - `src/split/ir_query_splitter.cpp`: topdown sets `DeferredSpecCheckFn` via adapter
     when `pending_spec_ && strategy == TOP_DOWN`
   - `src/adapters/duckdb_adapter.cpp`: adapter calls deferred check at start of
     query-jit block in `ExecuteSQLandCreateTempTable`, before `qjit_spec_hit_` check
   - `include/adapters/duckdb_adapter.h`: added `DeferredSpecCheckFn` type + `deferred_spec_check_` member

5. **`--no-cross-query-prep` flag** for measurement breakdown:
   - `include/util/param_config.h`: `bool no_cross_query_prep = false`
   - `src/util/param_config.cpp`: parses `--no-cross-query-prep`
   - `src/aqp_middleware.cpp`: gates `duck_for_cross` and `pg_for_cross` on the flag
   - `measure/measure_breakdown_time_aqp.sh`: 15th positional arg `disable_compile_opts`
     with `_nocrossqprep` filename suffix

6. **Measurement script**: `measure/breakdown_compile_time_reduction.sh`
   - Figure A: compiler backend (llvm/fastisel/tpde)
   - Figure B: JIT cache mode (off/strict/template)
   - Figure C: latency hiding (spec-off/spec-on/spec-on+cross-query)

### Dead code from reverted experiments (kept but unused)

7. **Build-side annotation cache** (`src/adapters/duckdb_adapter.cpp`):
   - `s_build_side_cache` with order-independent keys, `BuildSideCacheEntry` struct
   - `CollectIRTableNames`, `BuildSideCacheKey`, `CacheBuildSideAnnotations`
   - `HeuristicAnnotateBuildSides`, `CachedOrHeuristicAnnotateBuildSides`,
     `TryCacheOnlyAnnotateBuildSides`
   - `ClearBuildSideCache()` static method in DuckDBAdapter
   - `CacheBuildSideAnnotations()` called after each `AnnotateBuildSides()` (3 sites)
   - Currently populates cache but spec doesn't use it (full planning path annotates directly)

8. **`SpeculativeQueryJitCompileFast`** (Direction A — not called):
   - Implementation in `src/adapters/duckdb_adapter.cpp`, declaration in header
   - Skips DuckDB planning, uses spec IR directly
   - ABANDONED: peeked IR lacks DuckDB optimizer transformations → execution regressions

9. **`TryBuildBinaryPlanFromIR` overload with `duckdb::Connection &`**:
   - In header and implementation — not called from spec path

10. **`hint_ir` parameter on `SpeculativeQueryJitCompile`**:
    - Defaulted to nullptr, no callers pass it

### Key findings from exploration

- **Spec-jit overhead sources** (for topdown + TPDE + cache=template):
  - Spec machinery on main thread: ~0.15ms/subquery (peek, SQL gen, compare)
  - Bg compile: ~0.5-0.7ms (DuckDB planning dominates, codegen ~0ms on cache hit)
  - Wait: main thread blocks until bg finishes
  - Phase B launches AFTER Execute(i), only ~0.2ms before CheckSpec

- **Why spec can't beat no-spec for TPDE+cache**:
  - Inline compile with TPDE+cache is already ~0.5-0.7ms (fast enough)
  - Spec-compiled code differs from inline (full Optimize vs TryBuildBinaryPlanFromIR+FilterOptimize) → execution regressions up to 2-6x on ~30% of subqueries
  - Phase B's keep-on-match comparison acts as quality filter — only 79/342 subqueries get spec HITs (the safe ones)
  - Persistent cache helps bg finish faster but also helps inline → net benefit is small

- **Persistent cache alone (no spec) gives -5.8%**: just not clearing the object cache between iterations saves all compile time on iter 1+

- **Spec-jit IS valuable for LLVM compile-mode** (5-50ms compile time to hide)

- **Build-side heuristic accuracy**: 96% match with DuckDB on JOB (19/455 mismatches, all close-ratio cardinalities)

### Verification

- Build: passes (release)
- Correctness: all 113 JOB queries pass for all related configs
- CSV format: unchanged (no new columns), parseable by /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py
- No new print statements; existing LIKE debug trace guarded by `#ifndef NDEBUG`
- Check if need changes to measure/*.sh or measure/*.py
- Performance (top 15 worst LIKE queries): measure/measure_breakdown_time_aqp.sh job with correct configs 
