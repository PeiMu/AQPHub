# Task: Pipelined Speculative JIT Compilation for Node-Based Split

You are a DBMS JIT compilation expert. Approach this as Prof. Thomas Neumann or Matthias Jasny would: apply every low-level technique available, leave nothing on the table.

## Goal

Hide operator-JIT compilation latency by speculatively compiling the *next* subquery on a background thread while the *current* subquery executes via DuckDB interpretation. Target: the SQL fallback path in `ExecuteOneIteration` where `RegisterJIT` compiles LLVM IR for filters, projections, aggregates, and fused pipelines.

## Architecture Overview

### Current Sequential Flow (per iteration in `ExecuteOneIteration`)

```
SplitIR() → GenerateSQL() → SetJITPendingIR() → ExecuteSQLandCreateTempTable()
                                                   ↳ Prepare() → RegisterJIT() → Execute()
                                                     [compile]    [compile LLVM]   [run]
```

All JIT compilation blocks execution. The DuckDB `ExecuteSQL` path (line 812 of `duckdb_adapter.cpp`) calls `Prepare()` then `RegisterJIT()` then `Execute()` synchronously.

### Target Pipelined Flow

```
Iter i:   SplitIR → [Execute sub_i with DuckDB]  ←  bg thread: speculative compile for sub_{i+1}
Iter i+1: SplitIR → check speculative result:
            HIT (SQL matches):  use pre-compiled JIT, skip RegisterJIT → Execute
            MISS (plan changed): discard, fall through to normal compile+execute path
```

After `UpdateRemainingIR` completes (actual cardinalities known), run a speculative `RunMiddleOptimize + SplitIR` on a **separate DuckDB connection** to predict the next subquery, then compile its JIT code on the background thread.

## Key Design Decisions (Confirmed)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Separate DuckDB connection** for speculative Prepare | DuckDB `ClientContext` uses a mutex (`ClientContextLock`). Prepare() and Execute() on the same connection cannot run concurrently — they'd deadlock. Create a second `Connection` on the same `DuckDB` instance (shares catalog, temp tables visible). |
| 2 | **Focus on next subquery only**, not siblings | Sibling speculation is deferred. The next subquery is the primary target. |
| 3 | **Separate `jit_compile_pool_`** (1 worker) | Dedicated to speculative JIT. Does not compete with `bg_pool_` (used for async CSR builds). Both increment `g_bg_active_threads` for OpenMP coordination. |
| 4 | **`future::wait_for(0s)`** to check completion | Don't cancel LLVM mid-compilation. If not ready when needed, interpret normally. If ready and matches, use it. If ready but plan changed, discard. |
| 5 | **OS scheduler handles DuckDB thread competition** | The speculative compilation thread competes for CPU with DuckDB's internal threads. Let the OS handle it. **NOTE**: If this slows DuckDB interpretation, modify DuckDB code (`~/Project/duckdb`) to accept a runtime thread count hint (e.g., `SET threads=N-1` during speculative compilation). Monitor this in measurements. |
| 6 | **Only target operator-jit** (the `RegisterJIT` path) | Not kernel paths (PIPELINE/QUERY), not expr-only JIT. The `AQP_JIT_LEVEL_MASK` gated path in `DuckDBAdapter::ExecuteSQL` and `ExecuteSQLandCreateTempTable`. |
| 7 | **Separate `IrToLlvmCompiler` instance** for speculative thread | The compiler's `impl_->current_tracker` and `TrackerGuard` are not thread-safe. Each instance has its own LLJIT + ResourceTracker. The `s_filter_counter` atomic is already safe across instances. |

## Verified Facts (confirmed by code inspection)

- **A1**: DuckDB execution uses its internal thread pool, not OpenMP. OpenMP is only in `sub_query_plan.cpp` and `pipeline_kernel.cpp` (kernel paths). ✓
- **A2**: `IrToLlvmCompiler` is NOT thread-safe for concurrent compilation. `TrackerGuard` mutates shared `impl_->current_tracker`. No mutex. ✓
- **A3**: The SQL fallback path is the only path that uses LLVM JIT compilation. Kernel paths don't use it. ✓
- **A4**: DuckDB `ClientContext` uses `ClientContextLock` (a `lock_guard<mutex>`). `Prepare()` and `Execute()` both acquire it. Cannot run concurrently on the same connection. ✓ (verified at `duckdb/src/include/duckdb/main/client_context.hpp:347`)
- **A5**: LLVM ORC `LLJIT::addIRModule` is thread-safe when using separate `ResourceTracker`s. Each `Compile*` call creates its own `LLVMContext` + `Module` + `ThreadSafeModule`. ✓
- **A6**: `DuckDBAdapter` creates `DuckDB` instance + `Connection`. A second `Connection(*db)` shares the catalog. `db` is private — need to add `DuckDB &GetDB()` getter. ✓
- **A7**: `s_filter_counter` (unique function name generator) is `std::atomic<uint64_t>` — safe across threads. ✓

## Files to Modify

### AQPHub (middleware)

| File | Changes |
|------|---------|
| `include/split/ir_query_splitter.h` | Add `jit_compile_pool_`, `SpeculativeCompilation` struct, debug counters |
| `src/split/ir_query_splitter.cpp` | Speculative compile kickoff (after `UpdateRemainingIR`), hit/miss check (before execute), debug print |
| `include/adapters/duckdb_adapter.h` | Add `DuckDB &GetDB()` public getter. Add method to perform speculative compile on a separate connection+compiler |
| `src/adapters/duckdb_adapter.cpp` | Implement speculative compile method: Prepare on separate connection → RegisterJIT with separate compiler |
| `include/split/node_based_splitter.h` | May need to expose state for speculative SplitIR peek (or add a `SpeculativeSplitIR` method) |
| `src/split/node_based_splitter.cpp` | Speculative split: snapshot splitter state → tentative MiddleOptimize + SplitIR → restore state |

### DuckDB (if needed — Decision 5 note)

| File | Changes |
|------|---------|
| `~/Project/duckdb/...` | Only if DuckDB interpretation slows down due to CPU competition with bg compile thread. Would add a way to dynamically set thread count. |

## Data Structures

```cpp
// In ir_query_splitter.h:

struct SpeculativeCompilation {
    std::future<bool> future;              // true = compilation succeeded
    std::string speculative_sql;           // SQL string for match comparison
    // The compiled JIT context (AQPJITContext) lives in the speculative
    // DuckDB connection's ClientContext. On hit, swap it into the main adapter.
    std::unique_ptr<duckdb::Connection> spec_conn;        // separate connection
    std::unique_ptr<aqp_jit::IrToLlvmCompiler> spec_compiler;  // separate LLVM instance
};

// New members in IRQuerySplitter:
std::unique_ptr<ThreadPool> jit_compile_pool_;  // 1 worker, separate from bg_pool_
std::unique_ptr<SpeculativeCompilation> pending_spec_;

// Debug counters (printed under #ifndef NDEBUG or enable_debug_print)
int spec_subqueries_interpreted_ = 0;   // executed without speculative JIT
int spec_subqueries_compiled_ = 0;      // used speculative JIT (hits)
int spec_misses_ = 0;                   // speculative JIT discarded (plan changed)
int spec_not_ready_ = 0;               // speculative JIT still compiling when needed
```

## Speculative Compilation Flow (Detailed)

### Phase A: Kick off speculation (end of each iteration, after UpdateRemainingIR)

```
// After UpdateRemainingIR completes (actual cardinalities are in the plan):
// 1. Snapshot the node-based splitter state
// 2. On the bg thread: tentative MiddleOptimize + SplitIR → generate SQL → Prepare → RegisterJIT
// 3. Store the SQL string and future in pending_spec_

pending_spec_ = make_unique<SpeculativeCompilation>();
pending_spec_->spec_conn = make_unique<Connection>(duck->GetDB());
pending_spec_->spec_compiler = make_unique<IrToLlvmCompiler>(...);

jit_compile_pool_->Submit([&]() {
    // Tentative split on snapshot state
    auto spec_extraction = splitter_->SpeculativeSplitIR();
    if (!spec_extraction || spec_extraction->is_final) return false;

    // Generate SQL
    string sql = adapter_->GenerateSQL(*spec_extraction->sub_ir, ...);
    pending_spec_->speculative_sql = sql;

    // Prepare on separate connection (gets physical plan)
    auto prepared = pending_spec_->spec_conn->Prepare(sql);
    if (prepared->HasError()) return false;

    // Compile JIT on separate compiler
    RegisterJITSpeculative(prepared->physical_plan, *spec_extraction->sub_ir,
                           *pending_spec_->spec_compiler);
    return true;
});
```

### Phase B: Check speculation (start of next iteration, after SplitIR)

```
// After SplitIR produces the actual next subquery:
bool use_speculative = false;

if (pending_spec_ && pending_spec_->future.valid()) {
    auto status = pending_spec_->future.wait_for(chrono::seconds(0));
    if (status == future_status::ready) {
        string actual_sql = GenerateSQL(*extraction->sub_ir, ...);
        if (actual_sql == pending_spec_->speculative_sql && pending_spec_->future.get()) {
            // HIT: transfer compiled JIT context to main adapter
            use_speculative = true;
            spec_subqueries_compiled_++;
        } else {
            // MISS: plan changed
            spec_misses_++;
        }
    } else {
        // NOT READY: still compiling, don't wait
        spec_not_ready_++;
    }
    pending_spec_.reset();
}

if (use_speculative) {
    // Execute with pre-compiled JIT (skip RegisterJIT)
} else {
    // Normal path: compile inline + execute
    spec_subqueries_interpreted_++;
}
```

## Critical Design Challenges

### Challenge 1: Speculative SplitIR modifies NodeBasedSplitter state

`NodeBasedSplitter::SplitIR` calls `RunMiddleOptimize`, `qs_->Split`, etc., which mutate `plan_`, `subqueries_`, `proj_expr_`. A speculative split would corrupt the real state.

**Solutions (choose one):**
- **(a) Deep-copy the LogicalOperator plan and all splitter state before speculative split, run on the copy.** DuckDB's `LogicalOperator` supports `Copy()`. This is the safest approach.
- **(b) Run speculative split on the main thread synchronously (fast — just the split, not compile), then kick off only the compilation to the bg thread.** This avoids the state problem entirely: the main thread does `MiddleOptimize + SplitIR` (which it would do anyway next iteration), generates SQL, and hands off only the `Prepare + RegisterJIT` to the background. The bg thread needs the SQL string and sub-IR only.

**Recommended**: **(b)** — it's simpler and the split itself is fast. The expensive part is `Prepare + RegisterJIT(LLVM compile)`, which is what we want on the bg thread. The split + SQL generation can happen synchronously at the end of each iteration.

But note: doing the real SplitIR early (at the end of iter i instead of the start of iter i+1) means iter i+1 should NOT call SplitIR again. This requires restructuring the loop.

**Alternative (b')**: Don't do a speculative SplitIR at all. Instead, at the end of iter i, just generate SQL for the *current* remaining IR as if it were the next subquery (a rough approximation). This has a higher miss rate but zero state mutation risk.

### Challenge 2: Transferring JIT context from speculative connection to main

On a hit, we need the compiled function pointers from the speculative compiler to be used by the main DuckDB execution. The `AQPJITContext` holds function pointers that point into the speculative `IrToLlvmCompiler`'s LLJIT memory.

**Solution**: Keep the speculative compiler alive until the subquery finishes executing. Transfer the `AQPJITContext` from the speculative connection's `ClientContext` to the main connection's `ClientContext`. The function pointers remain valid as long as the speculative compiler lives.

### Challenge 3: Temp table visibility on speculative connection

The speculative `Prepare()` call needs to see temp tables created by previous iterations. Since both connections share the same `DuckDB` instance, temp tables in the catalog are visible. But `IN_MEM_TMP_TABLE` mode stores data in `temp_collections_` (a per-adapter map, not in the catalog).

**Solution**: The speculative connection only needs to `Prepare()` (not `Execute()`). `Prepare()` only needs schema info, not data. If DuckDB's `Prepare()` fails because it can't find the temp table, the speculation fails gracefully (miss).

**Verify**: Whether DuckDB `Prepare()` on a separate connection can see temp tables created by `ExecuteSQLandCreateTempTable` on the main connection when `IN_MEM_TMP_TABLE=1`. If temp tables are registered via replacement scans bound to the main adapter's `temp_collections_`, the speculative connection won't see them. May need to register a replacement scan on the speculative connection too, or use `CREATE TEMP TABLE` instead.

## Debug Output

Under `#ifndef NDEBUG` (matching existing pattern in `duckdb_adapter.cpp`):

```cpp
#ifndef NDEBUG
std::cerr << "[AQP-SPECJIT] interpreted=" << spec_subqueries_interpreted_
          << " compiled=" << spec_subqueries_compiled_
          << " misses=" << spec_misses_
          << " not_ready=" << spec_not_ready_ << "\n";
#endif
```

Also print per-iteration decisions:
```cpp
#ifndef NDEBUG
std::cerr << "[AQP-SPECJIT] iter=" << iteration_count_
          << " decision=" << (use_speculative ? "HIT" : "INTERPRET") << "\n";
#endif
```

## Workflow Per Iteration

### 1. Read relevant source code for the optimization you're implementing

### 2. Check if we need use plan mode to design code implementation. Then make the code change and write unit gtests in unit_test/ dir. If add new modules or code changes related to breakdown timer, decide a reasonable timer position and ask user to confirm. Confirm the timer functions cover everything and no overlap. Confirm analyze_middleware_breakdown, analyze_none_split_breakdown in /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py still work. Otherwise, update the analyze_middleware_breakdown and/or analyze_none_split_breakdown, and confirm with the user.

### 3. Build and quick-test
Build:
```bash
cd /home/pei/Project/AQPHub/build_release && make -j12
# If DuckDB files changed:
cd /home/pei/Project/duckdb/build/release && make -j12
```
Run the current heaviest query (~1 second):
```bash
cd /home/pei/Project/AQPHub/measure
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
cd /home/pei/Project/AQPHub/measure
bash run_aqp.sh job duckdb node-based pipeline o1 none
filter='grep -v "^Running\|^==\|^Execution\|^$\|^waiting\|^server\|^ANALYZ\|^duckdb runs:\|^(base)"'
diff <(eval $filter job_result/aqp_middleware_duckdb_node-based_pipeline_o1_none_job.txt) \
     <(eval $filter duckdb_job_node-based_golden.txt)
```
Full:
```bash
cd /home/pei/Project/AQPHub/measure
bash ./correctness_test.sh
```
If any error, debug with ../build_debug/aqp_middleware (or c++ -O0 -g) first, then re-test with ../build_release/aqp_middleware.

### 5. If faster: full breakdown measurement (~28 min, only for final validation)
Do NOT do ANYTHING when running performance measurement to avoid noise.
```bash
cd /home/pei/Project/AQPHub/measure
bash measure_breakdown_time_aqp.sh job duckdb node-based pipeline o1 none
```

### 6. If slower or neutral
Analyze why, check the slowdown queries if it is noise, revert or adjust (fundamentally change direction, algorithm, or guard by condition), try again.

### 7. Mixed results (some queries speed up, others slow down)
Check what's different. Fix fundamentally or at least guard by condition of the slowdown query's pattern.

### 8. Compare with baselines
Compare JIT with split strategy (e.g., node-based) speedup or slowdown vs none-split+none-JIT and vs the last version of JIT. Find the fundamental reason.

### 9. Summarize and update
Report: what changed, which queries improved/regressed, by how much, net effect on total JOB time.

## Codebase Quick Reference

| Component | Header | Implementation |
|-----------|--------|----------------|
| Split loop orchestrator | `include/split/ir_query_splitter.h` | `src/split/ir_query_splitter.cpp` |
| Node-based splitter | `include/split/node_based_splitter.h` | `src/split/node_based_splitter.cpp` |
| Top-down splitter | `include/split/topdown_splitter.h` | `src/split/topdown_splitter.cpp` |
| DuckDB adapter | `include/adapters/duckdb_adapter.h` | `src/adapters/duckdb_adapter.cpp` |
| JIT compiler (LLVM) | `include/jit/ir_to_llvm.h` | `src/jit/ir_to_llvm.cpp` (6642 lines) |
| JIT ABI constants | `include/jit/aqp_jit_abi.h` | — |
| DuckDB JIT dispatch | — | `~/Project/duckdb/src/execution/operator/physical_*.cpp` |
| DuckDB JIT context | — | `~/Project/duckdb/src/include/duckdb/execution/aqp_jit.hpp` |
| Thread pool | `include/util/thread_pool.h` | `src/util/thread_pool.cpp` |
| Config/flags | `include/util/param_config.h` | `src/util/param_config.cpp` |
| Pipeline kernel | `include/kernel/pipeline_kernel.h` | `src/kernel/pipeline_kernel.cpp` |
| Sub-query plan | `include/kernel/sub_query_plan.h` | `src/kernel/sub_query_plan.cpp` |

### Key Line References

- `ExecuteOneIteration`: `src/split/ir_query_splitter.cpp:533`
- SQL fallback path (where JIT compile happens): `src/split/ir_query_splitter.cpp:871-943`
- `SetJITPendingIR` call: `src/split/ir_query_splitter.cpp:911`
- `DuckDBAdapter::ExecuteSQL` (JIT path): `src/adapters/duckdb_adapter.cpp:812-987`
- `RegisterJIT` entry: `src/adapters/duckdb_adapter.cpp:2312`
- `EnsureJITCompiler`: `src/adapters/duckdb_adapter.cpp:2293`
- `IrToLlvmCompiler::CompileFilter`: `src/jit/ir_to_llvm.cpp:4860+`
- `CompileFilterProbeProjectFusion`: `src/jit/ir_to_llvm.cpp:5719+`
- `IrToLlvmCompiler::ResetModules`: `src/jit/ir_to_llvm.cpp:4614`
- `NodeBasedSplitter::SplitIR`: `src/split/node_based_splitter.cpp:68`
- `NodeBasedSplitter::UpdateRemainingIR`: `src/split/node_based_splitter.cpp:178`
- `RunMiddleOptimize`: `src/split/node_based_splitter.cpp:43`
- OpenMP thread reduction: `src/kernel/pipeline_kernel.cpp:1019-1048`, `src/kernel/sub_query_plan.cpp:1670-1691`
- `g_bg_active_threads`: `include/util/thread_pool.h:18`, `src/util/thread_pool.cpp:5`
- `DuckDB` private field: `include/adapters/duckdb_adapter.h:316`
- `Connection` constructor: `duckdb/src/include/duckdb/main/connection.hpp:42`
- `ClientContextLock`: `duckdb/src/include/duckdb/main/client_context.hpp:347`
- JIT level flags: `include/jit/aqp_jit_abi.h:71-87`
- `--jit-level` parsing: `src/util/param_config.cpp:135-149`

### OpenMP Coordination

All OpenMP regions follow this pattern:
```cpp
int bg = g_bg_active_threads.load(std::memory_order_relaxed);
int nthreads = std::min(std::max(1, 12 - bg), omp_get_max_threads());
#pragma omp parallel num_threads(nthreads)
```
Both `bg_pool_` and `jit_compile_pool_` workers increment/decrement `g_bg_active_threads` via the `ThreadPool::WorkerLoop`. No additional coordination needed.

### Temp Table Visibility (IN_MEM_TMP_TABLE)

When `IN_MEM_TMP_TABLE=1`, temp tables are stored in `DuckDBAdapter::temp_collections_` (a per-adapter map) and accessed via a registered `scan_temp_collection` table function bound to that map. A speculative connection created with `Connection(adapter->GetDB())` will NOT automatically see these temp tables because the replacement scan is bound to the main adapter's map.

**Workaround**: Either register the same replacement scan on the speculative connection (pointing to the same `temp_collections_`), or only use speculative compilation on iterations where `Prepare()` can resolve all referenced tables. If `Prepare()` fails, treat as a speculation miss.

### Build

```bash
cd /home/pei/Project/AQPHub/build_release && make -j12
# If DuckDB files changed:
cd /home/pei/Project/duckdb/build/release && make -j12
```

### Test Queries (heaviest, most likely to benefit)

16b, 29c, 8c, 19d, 9d — these have multiple split iterations with DuckDB fallback (JIT compilation).
