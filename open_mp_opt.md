# OpenMP Optimization Opportunities

## Current OpenMP Usage (2 locations)

1. `ExecuteSubQueryPlan` main scan loop (`src/kernel/sub_query_plan.cpp:1500`) — 12 threads, `schedule(dynamic, 4096)`, threshold 10K rows
2. `ExecuteFinalAggregate` scan loop (`src/kernel/sub_query_plan.cpp:2121`) — same config

## Opportunity 1: Runtime CSR Build (est. -1,500 to -2,000ms wall time)

**Location**: `ir_query_splitter.cpp:676-696` (kernel path) and `ir_query_splitter.cpp:892-911` (DuckDB fallback path), calling `BuildCSR` in `src/storage/csr_index.cpp`

**Impact**: 573 runtime CSR builds across 113 JOB queries. Total `extra_materialization` = 4,040ms. CSR build dominates this timer.

**Evidence** (16b, single query, 5 sub-query iterations):

| Iter | Kernel Exe (ms) | CSR+Mat (ms) | Scan table | Result rows |
|------|---------------:|-------------:|------------|------------:|
| 1 | 27.6 | 20.2 | keyword | 41,840 |
| 2 | 11.7 | 37.2 | company_name | 1,153,798 |
| 3 | 0.6 | 12.2 | temp1 | 68,316 |
| 4 | 42.5 | 105.0 | temp3 | 2,832,555 |
| 5 | 211.2 | 211.4 | aka_name | 3,710,592 |
| **Total** | **325.1** | **385.0** | | |

CSR+Mat (385ms) > kernel execution (325ms) for 16b alone.

**Phase-level breakdown** (microbenchmark, 3.7M rows, domain 4M):

| Phase | Time (ms) | Parallelizable |
|-------|----------:|:--------------:|
| `alloc + memset` (20MB row_ptr) | 12.4 | Yes (memset) |
| Pass 1: count (random increments into row_ptr) | 32.9 | Yes (thread-local counts + merge) |
| Pass 2: prefix sum | 2.6 | No (sequential) |
| `cursors` vector copy (20MB) | 18.4 | Yes (memcpy segments) |
| Pass 3: scatter (random writes into col_idx) | 265.2 | Yes (partitioned scatter) |
| **Total** | **331.5** | |

Pass 3 (scatter) dominates at 80%. It's memory-bound with random writes — parallelization helps because 12 threads spread the write pressure across L3 cache lines.

**Implementation approach**:
- Pass 1: each thread counts into a thread-local `row_ptr` array, then merge (sum) across threads. Avoids atomic increments.
- Pass 3: partition rows by thread, each thread scatters into its own cursor range. Requires per-thread cursor arrays or partitioned approach.
- Alternative: replace `std::vector<uint64_t> cursors` copy with direct pointer arithmetic from `row_ptr` (avoids 20MB allocation).

**Estimated speedup**: 4-6x on Pass 1+3 with 12 threads (memory-bound, not compute-bound, so not full 12x). Total CSR build: ~330ms -> ~80ms per large temp. Across all queries: ~1,500-2,000ms savings.

## Opportunity 2: CreateTempFromFlatTable (est. -200 to -500ms wall time)

**Location**: `src/adapters/duckdb_adapter.cpp:437-505`

**What it does**: Copies kernel FlatTable results back into DuckDB `ColumnDataCollection` format (chunks of 2048 rows). Required so DuckDB's IR update mechanism can reference the temp table.

**Current implementation**: Sequential loop over 2048-row chunks, memcpy for INT32 columns, `StringVector::AddString` for VARCHAR columns. For large temps (1-3.7M rows), this takes 10-50ms per temp.

**Parallelization**: Build chunks in parallel, then append sequentially (DuckDB's `ColumnDataCollection::Append` is not thread-safe). Or pre-allocate all chunks and fill in parallel.

**Estimated speedup**: Modest (2-3x) because the bottleneck is memory bandwidth (memcpy) and DuckDB string allocation. VARCHAR columns involve small allocations per string which limit parallelism.

## Opportunity 3: Startup CSR Build (est. -1 to -2s startup, one-time)

**Location**: `src/storage/storage_plan.cpp:290-392` — `BuildCSRIndexes` builds 12+ CSR indexes sequentially.

**Current**: Each FK relationship's CSR is built one at a time. cast_info CSRs (36M rows) take ~300ms each.

**Parallelization**: Build each FK's CSR in parallel (`#pragma omp parallel for` over the FK list). No data dependencies between different FK CSRs.

**Impact**: One-time startup cost, cached to disk via `--storage-cache`. Only matters on first run or cache miss. Reduces startup from ~5s to ~1-2s.

## Opportunity 4: Startup Sorted Index Build (est. -3 to -5s startup, one-time)

**Location**: `src/storage/storage_plan.cpp:394-434` — `BuildSortedIndices` builds 11 sorted indices sequentially.

**Current**: Each `BuildSortedIndex` calls `std::sort` on a permutation array (up to 36M entries for cast_info). `std::sort` is single-threaded.

**Parallelization**: Two levels:
1. Build different indices in parallel (`#pragma omp parallel for` over the 11 specs)
2. Use parallel sort within each index (GNU `__gnu_parallel::sort` or manual merge-sort)

**Impact**: One-time startup, cached to disk. Reduces sorted index build from ~10s to ~2-3s.

## Opportunity 5: BuildFilteredPKBitset (est. -5 to -20ms per query)

**Location**: `src/kernel/sub_query_plan.cpp:360-403`

**What it does**: Scans a table with filters, sets bits in a PK bitset. Used during `AnalyzeSubIR` for join validation.

**Current**: Sequential scan with filter evaluation per row. Called for lookup tables (up to 2.6M rows for movie_companies).

**Parallelization**: Thread-local bitsets, merge with OR. Filter evaluation is independent per row.

**Impact**: Small — most calls are on dimension tables (<200 rows) or temp tables (<100K rows). Only movie_companies (2.6M) would benefit significantly. And this runs during plan analysis, not the hot execution path.

## Opportunity 6: Inverted Index Source Scan (est. negligible)

**Location**: `src/kernel/sub_query_plan.cpp:981` — scans source table with compiled filters during 3-table inverted index resolution.

**Tables**: keyword (134K), name (4.2M), company_name (235K).

**Impact**: Only name (4.2M) is large enough to benefit. Runs during `AnalyzeSubIR`, not execution. Marginal improvement.

## Summary

| Opportunity | Location | Est. Savings | Type |
|-------------|----------|-------------|------|
| 1. Runtime CSR Build | csr_index.cpp / ir_query_splitter.cpp | 1,500-2,000ms | Per-run |
| 2. CreateTempFromFlatTable | duckdb_adapter.cpp | 200-500ms | Per-run |
| 3. Startup CSR Build | storage_plan.cpp | 1-2s | One-time |
| 4. Startup Sorted Index | storage_plan.cpp | 3-5s | One-time |
| 5. BuildFilteredPKBitset | sub_query_plan.cpp | 5-20ms | Per-run |
| 6. Inverted Index Scan | sub_query_plan.cpp | <5ms | Per-run |

**Priority**: Opportunity 1 is the clear winner — it targets the largest component of the 5,266ms middleware overhead (4,040ms extra_materialization) and applies to every query execution, not just startup.

## Timer Bug Found During Investigation

The `extra_materialization` and `execute_sub-SQL` columns in the breakdown CSV were previously misattributed in analysis. The `analyze_middleware_breakdown` parser in `plot_middleware_jit.py` correctly parses the columns (verified), but earlier manual column-counting was wrong due to the 5 header columns (`prepare_mw, read_sql, parse_sql, preprocess, convert_plan_to_ir`) before the per-iteration groups.

Verified correct values for 16b (single run):
- Kernel execution total: 325ms (previously appeared as ~0.15ms due to stale CSV)
- CSR+Mat total: 385ms (previously appeared as 0ms)
- These sum to 710ms, consistent with 16b's ~213ms average `extra_materialization` per run (the difference: 710ms is single cold run, 213ms is avg of 10 warm runs with memory reuse)
