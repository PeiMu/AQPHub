# Middleware Overhead Analysis

## Summary

**After 7.3 + 7.3b optimization** (skip CreateTempFromFlatTable on kernel path + lazy CSR on DuckDB path):

| Metric | Step 6 Baseline | Step 7.3+7.3b | Improvement |
|--------|----------------:|-------------:|------------:|
| MW Overhead | 5,811ms* | 5,789ms | -22ms |
| Wall Time | 13,403ms* | 13,524ms | +121ms (noise) |
| Materialize | 4,539ms | 3,291ms | -1,248ms (-27.5%) |

*Step 6 baseline had ~1,294ms of untimed DuckDB-path FlatTable+CSR cost. Real wall time was ~14,700ms.

7.3 eliminated CreateTempFromFlatTable on kernel path (-1,237ms materialize).
7.3b made DuckDB-path FlatTable+CSR cost visible in timing (moved ~1,294ms from untimed to analyze+gen_final columns). Real performance unchanged.

**vs DuckDB none-split** (no AQP, pure DuckDB execution):

| Metric | DuckDB (none-split) | AQP 7.3b |
|--------|-------------------:|---------:|
| Execution | 9,529ms | 7,735ms (-18.8%) |
| MW Overhead | 3ms | 5,789ms |
| Wall Time | 9,532ms | 13,524ms (+41.9%) |

AQP kernel is 1.23x faster in execution, but MW overhead (+5,789ms) causes +42% wall time.
AQP wins 25/113 queries on wall time; DuckDB wins 88/113.

Baseline data: `measure/job_result/step_6_fix_all/` and `measure/job_result/step_6_perf/` (none-split)
Current (7.3b): `measure/job_result/step_7_2/`

## Per-Query Timing CSV Format

Each query logs: `prepare_mw, read_sql, parse_sql, preprocess` (one-time), then per iteration 6 columns, then final phase columns.
Check csv parser in /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py

### Per-Iteration Columns (6 per iteration)

| Col | Timer label | Kernel path | DuckDB fallback path |
|-----|-------------|-------------|---------------------|
| 1 | `extract` | `SplitIR()` — find next sub-query to extract | same |
| 2 | `gen_sub / analyze` | lazy DuckDB-temp FlatTable+CSR build (7.3b) + `AnalyzeSubIR()` | `GenerateSQL()` — convert IR to SQL string |
| 3 | `jit_compile` | 0 (kernel is pre-compiled C++) | JIT compilation time (LLVM) |
| 4 | `execute` | `ExecuteSubQueryPlan()` — kernel scan+filter+join | `ExecuteSQL()` — DuckDB vectorized execution |
| 5 | `materialize` | Runtime CSR build + `RegisterTempMetadata` (7.3: CreateTemp skipped) | metadata + stats scan (small) |
| 6 | `update_ir` | `UpdateRemainingIR()` — rewrite remaining IR indices | same |

### Final Phase Columns

| Col | Description |
|-----|-------------|
| 1 | `gen_final` — generate final SQL or AnalyzeFinalIR |
| 2 | `jit_final` — JIT compile final query (or 0 for kernel) |
| 3 | `final_exe` — execute final query |
| 4 | `show_output` — format and display results |

## Middleware Overhead = Everything Except Execution

Middleware overhead = all columns EXCEPT col 3 (jit_compile) and col 4 (execute).

Per iteration: `extract + gen_sub/analyze + materialize + update_ir`
One-time: `prepare_mw + read_sql + parse_sql + preprocess + gen_final + show_output`

## Column 5 Breakdown: Materialization (Dominant Cost)

The materialization column is the **single biggest middleware cost** per iteration on the kernel path. It contains 3 sequential operations.

Sub-timers added in `ir_query_splitter.cpp` (lines 667–725), output to `materialize_breakdown.csv`.
Format: `query, repeat, iteration, rename_ms, csr_build_ms, create_temp_ms, cardinality, int_cols, varchar_cols`

### Measured sub-phase breakdown (query 16b, heaviest query, 5 iterations)

| Iter | Rows | INT cols | VARCHAR cols | CSR build (ms) | CreateTemp (ms) | Total (ms) |
|------|-----:|:--------:|:------------:|---------------:|----------------:|-----------:|
| 1 | 42K | 1 | 0 | 4.7 | 0.1 | 4.8 |
| 2 | 1.15M | 1 | 0 | 18.7 | 1.1 | 19.8 |
| 3 | 68K | 1 | 0 | 4.7 | 0.1 | 4.8 |
| 4 | 2.83M | 2 | 0 | 37.4 | 4.6 | 42.0 |
| 5 | 3.71M | 2 | 1 | 38.3 | 112.2 | 150.5 |

Column renaming is ~0.001ms in all cases (negligible).

### Key finding: two distinct bottlenecks

**CSR build dominates INT-only temps** (iterations 1–4): 90%+ of materialization time. Cost scales with `rows × num_int_cols` and domain size (max_val determines `row_ptr` allocation).

**CreateTempFromFlatTable dominates when VARCHAR columns exist** (iteration 5): per-row `StringVector::AddString` costs 112ms for 3.71M rows with 1 VARCHAR column. For INT-only temps, CreateTemp is <5ms (just memcpy).

### 1. Column renaming (~0ms)
```cpp
// Lines 668-672 in ir_query_splitter.cpp
for (size_t c = 0; c < result_flat->columns.size(); c++) {
    result_flat->column_names[c] =
        ComputeColumnAlias(attr->GetTableIndex(), attr->GetColumnName());
}
```
Trivial string assignment. Negligible.

### 2. Runtime CSR build (total across all queries: ~3,500ms estimated)
```cpp
// Lines 674-696 in ir_query_splitter.cpp
for each INT32 column in result_flat:
    scan all rows to find max_val           // O(rows)
    if max_val > 10M: skip
    BuildCSR(flat, col_idx, max_val, ...)   // O(rows + max_val)
```
`BuildCSR` does:
- Allocate `row_ptr[max_val + 2]` (up to 40MB for cast_info PKs)
- Two-pass insertion: count phase + placement phase, both O(rows)
- Per INT32 column, so a temp with 3 INT32 cols = 3× CSR builds

**Why expensive**: Large domain (max_val up to 10M) means big allocation + cache-unfriendly random writes during placement. Multiple INT32 columns multiply the cost.

### 3. `CreateTempFromFlatTable` (total across all queries: ~1,000ms estimated)
```cpp
// Lines 437-505 in duckdb_adapter.cpp
// Copies FlatTable → DuckDB ColumnDataCollection
while (offset < row_count):
    for each column:
        if INT32: memcpy 2048 rows          // Fast
        if VARCHAR: per-row StringVector::AddString  // Slow
    collection->Append(chunk)
```
**Cost depends on column types**:
- INT-only temps: <5ms (memcpy path, negligible)
- VARCHAR temps: 30–112ms per VARCHAR column (per-row `StringVector::AddString` copies each string into DuckDB's string heap)

**Why it exists**: When the next iteration falls back to DuckDB (not kernel-handleable), DuckDB needs the data in its native `ColumnDataCollection` format. Also, DuckDB's optimizer uses the temp table's cardinality (`temp_table_card_`) for join ordering.

### Redundancy analysis

**`CreateTempFromFlatTable` is redundant when the next phase uses the kernel path.** The kernel reads from `kernel_temp_ptrs_` (FlatTable pointer), not DuckDB's `ColumnDataCollection`. DuckDB registration is only needed when a subsequent iteration or final phase falls back to DuckDB SQL execution.

**Last-iteration CSR build is often wasted.** The last iteration's CSR is only useful if the final phase uses the kernel path (`AnalyzeFinalIR` succeeds). Total last-iteration materialization: **883ms** across 104 multi-iteration queries. For many queries, the final phase is trivial (just reading a temp) or uses the DuckDB SQL path — CSR is not needed.

## Other Middleware Components

### Column 1: Extract / SplitIR (662ms total across all queries)
- Calls `splitter_->SplitIR(remaining_ir)` to find the next leaf/subtree to extract
- Node-based: traverses IR tree, selects leaf with best heuristic score
- Cost scales with IR tree depth × number of leaves

### Column 6: UpdateRemainingIR (18ms total)
- Rewrites remaining IR after extracting a sub-query
- Replaces executed table references with temp table reference
- Updates all column indices (table_idx, col_idx) in joins, filters, projections
- Recursive tree traversal + reconstruction

### Column 2: AnalyzeSubIR / GenerateSQL (966ms total after 7.3b)
- Kernel path: `EnsureReferencedTempsReady` (lazy FlatTable+CSR build, ~934ms) + `AnalyzeSubIR` (~32ms)
- DuckDB path: `GenerateSQL` converts IR tree to SQL string (recursive visitor)

### One-time: Preprocess (0.06ms total — negligible)
- Converts DuckDB logical plan to middleware IR (`AQPStmt`)
- DuckDB plan re-optimization (join reordering with temp cardinalities)

### One-time: prepare_mw (77ms total)
- Initial middleware setup

### One-time: parse_sql (48ms total)
- SQL parsing

### One-time: gen_final (786ms total after 7.3b)
- Generates final SQL from remaining IR
- On kernel path: `EnsureReferencedTempsReady` (lazy build, ~360ms) + `AnalyzeFinalIR` builds final aggregate plan

## Why Materialization is Needed (Adaptive QP)

The adaptive query processing loop requires materialization because:

1. **DuckDB join order optimization**: DuckDB's optimizer sees real temp table cardinalities (via `temp_table_card_`) and uses them for cost-based join reordering in subsequent iterations. Without registering temps, DuckDB can't optimize joins involving temp tables.

2. **Kernel CSR-based joins**: The runtime CSR built on temp results enables the kernel to do O(1) FK→PK lookups in subsequent iterations. Without CSR, the kernel can't handle joins on temp columns.

3. **DuckDB fallback execution**: When a subsequent iteration is not kernel-handleable (31.4% of iterations), DuckDB needs the data in `ColumnDataCollection` format to execute SQL against it.

## Optimization Opportunities (Preserving Adaptive QP)

### 7.1: Byte-maps as runtime artifacts
After kernel produces a temp, build a `vector<uint8_t>` indexed by PK value (byte-map). Next iteration's kernel can use byte-map as O(1) scan filter instead of CSR lookup. Saves runtime CSR build per temp for semi-join patterns where only PK membership matters.

**What it replaces**: Runtime CSR build (part of column 5)
**What it preserves**: Real cardinality (byte-map population count = exact cardinality), adaptive split decisions

### 7.2: Lazy FlatTable materialization
Only materialize columns actually referenced by downstream iterations. If remaining IR only needs PK existence (semi-join), skip VARCHAR columns entirely — saves `StringVector::AddString` overhead.

**What it replaces**: Full `CreateTempFromFlatTable` (part of column 5)
**What it preserves**: Cardinality info, columns that ARE referenced

### 7.3: Skip DuckDB registration for kernel-only chains
After executing iteration N, check `AnalyzeSubIR` (or `AnalyzeFinalIR` for the last iteration) on the remaining IR. If the next phase is kernel-handleable, skip `CreateTempFromFlatTable` entirely — the kernel reads from `kernel_temp_ptrs_` (our own FlatTable), not DuckDB's `ColumnDataCollection`. Similarly, if the next phase is kernel path, skip the JIT compilation entirely.

**What it replaces**: Entire `CreateTempFromFlatTable` call (part of column 5)
**What it preserves**: kernel_temp_ptrs_ for kernel access, cardinality for splitter decisions
**Risk**: Misprediction — if next iteration unexpectedly falls back to DuckDB, we'd need retroactive registration. But the splitter already knows the remaining IR at this point, so prediction should be reliable.

### 7.4: Skip last-iteration CSR build
The last iteration's CSR is only useful if the final phase uses the kernel path (`AnalyzeFinalIR` succeeds). For many queries, the final is trivial (just reading a temp, final_exe ~0ms) or uses DuckDB SQL — CSR is not needed. Can check `IsComplete()` after `UpdateRemainingIR` and skip CSR build if the remaining IR is trivial or `AnalyzeFinalIR` would fail.

**What it replaces**: CSR build on last iteration (part of column 5)
**Measured waste**: 883ms total across 104 multi-iteration queries

### 7.5: Pipelined materialization (background thread)

Overlap CSR build and CreateTempFromFlatTable with the next iteration's Extract + Analyze steps using a background thread.

**Dependency analysis** (traced through code):

Current sequential flow per iteration:
```
Execute → ColRename → CSR build → CreateTemp → UpdateRemainingIR → [next iter] Extract → Analyze → Execute
```

`UpdateRemainingIR` (node_based_splitter.cpp:179) depends on:
- `GetTempTableIndex()` → set by `CreateTempFromFlatTable` (duckdb_adapter.cpp:489)
- `temp_table_types` → set by `CreateTempFromFlatTable` (duckdb_adapter.cpp:490)
- `temp_table_cardinality` → known from kernel result (`result_flat->row_count`)

But `UpdateRemainingIR` does NOT depend on the actual data copy. It creates its own
empty `ColumnDataCollection` (node_based_splitter.cpp:210-211) just for DuckDB optimizer
cardinality tracking. The heavy `collection->Append(chunk)` loop in `CreateTempFromFlatTable`
(duckdb_adapter.cpp:458-484) is only needed when a future iteration actually executes SQL
against the temp.

**Solution**: Split `CreateTempFromFlatTable` into two phases:
1. **RegisterTempMetadata** (~0ms): assign `temp_table_index_`, `temp_table_types`,
   `chunk_col_names_`, `table_column_mappings`, `temp_table_card_`. This is just index
   assignment and map insertions — no data copy.
2. **CopyTempData** (expensive): build `ColumnDataCollection` from FlatTable, store in
   `temp_collections_`. Run on background thread.

Pipelined flow:
```
Main thread:    Execute(N) → Rename → RegisterMeta(N) → UpdateIR(N) → Extract(N+1) → Analyze(N+1) → [wait] → Execute(N+1)
Background:                  CSR(N) + CopyTempData(N) ────────────────────────────────────────────→ done
```

The wait point is just before `Execute(N+1)`:
- If N+1 is kernel: wait for CSR build to complete (kernel needs `runtime_csrs_`)
- If N+1 is DuckDB: wait for CopyTempData to complete (DuckDB needs `temp_collections_`)

**Overlap savings**: `min(CSR + CreateTemp, UpdateIR + Extract + Analyze)` per iteration.
UpdateIR is 18ms total, Extract is 662ms total, Analyze is 32ms total across 113 queries.
With ~580 total iterations, average Extract + Analyze per iteration ≈ 1.2ms.
So overlap saves ~1.2ms per iteration × 580 iterations ≈ **~700ms** in the best case.

CSR build + CreateTemp can also run in parallel with each other (both read from `result_flat`,
write to independent destinations), turning `CSR + CreateTemp` into `max(CSR, CreateTemp)`.
For 16b iter 5: max(38, 112) = 112ms instead of 38 + 112 = 150ms → saves 38ms on that iteration.

**Combines well with 7.3/7.4**: When 7.3 decides to skip CreateTemp, the background thread
only builds CSR. When 7.4 decides to skip CSR on the last iteration, the background thread
only does CreateTemp (if needed). The pipeline hides whatever work remains.

### 7.6: Dimension-partitioned flat tables
Partition large tables by low-cardinality FK at startup. Kernel scans only matching partition (e.g., cast_info filtered by role_id scans 4M instead of 36M). Reduces kernel execution time (column 4) by 7-9×.

**What it affects**: Column 4 (execution), not materialization directly. But faster execution → smaller temp results → faster materialization of those results.

## DuckDB-Path FlatTable + CSR Build (7.3b: Now Timed)

After a DuckDB JIT iteration produces a temp, the kernel needs a FlatTable + CSR to join against it in subsequent iterations. Before 7.3b, this was built eagerly after every DuckDB iteration and was **untimed** (~1,294ms hidden cost). 7.3b made it **lazy and timed**:

1. Removed eager build (was lines 795-917 in ir_query_splitter.cpp)
2. `EnsureReferencedTempsReady()` builds FlatTable+CSR on demand before `AnalyzeSubIR`/`AnalyzeFinalIR`
3. Cost now captured in `analyze` column (+934ms) and `gen_final` column (+360ms)

DuckDB-path iteration path sequences (K=kernel, D=DuckDB):

| Category | Count | Example | FlatTable+CSR needed? |
|----------|------:|---------|-------------|
| D followed by at least one K | 101 | `DKK`, `DKDK` | Yes (built lazily when K first references temp) |
| D with no K after | 26 | `KDD`, `D` (last iter) | No — lazy build skips these |
| **Total DuckDB iterations** | **127** | | |

## Optimization Progress

| # | Optimization | Status | Measured savings |
|:-:|-------------|--------|----------------:|
| 7.3 | Skip CreateTemp on kernel path | **DONE** | **-1,237ms materialize** |
| 7.3b | Lazy CSR on DuckDB path | **DONE** | Made ~1,294ms hidden cost visible; saves ≥26 wasted builds |
| 7.4 | Skip last-iteration CSR build | Attempted, reverted (node-based terminal detection issue) | est. ~883ms |
| 7.5 | Pipelined materialization (background thread) | Planned | est. ~700ms |
| 7.2 | Lazy materialization (skip VARCHAR cols) | Planned | est. ~400ms |
| 7.1 | Byte-maps (replace CSR build) | Planned | est. ~800ms |

## Remaining MW Overhead Breakdown (5,789ms after 7.3b — all costs now timed)

| Component | Time (ms) | Share | Target optimization |
|-----------|----------:|------:|---------------------|
| CSR build — kernel path (materialize col) | 3,291 | 56.8% | 7.1 byte-maps / 7.4 skip last-iter |
| FlatTable+CSR — DuckDB path (analyze col) | 934 | 16.1% | 7.1 byte-maps (replace CSR) |
| FlatTable+CSR — final phase (gen_final col) | 360 | 6.2% | 7.1 byte-maps (replace CSR) |
| Extract/SplitIR | 661 | 11.4% | — |
| Gen_final (pure analysis) | 426 | 7.4% | — |
| Other (parse, update_ir, show, prepare) | 117 | 2.0% | — |

To reach wall-time parity with DuckDB none-split (~9,532ms), need ~3,992ms MW reduction.
CSR/FlatTable build is the dominant target: 3,291 + 934 + 360 = **4,585ms** (79.2% of MW).
