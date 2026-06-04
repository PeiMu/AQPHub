# Query-Level Kernel Optimization Plan

Optimization plan for `--jit-level=query` (CSR-based kernel with node-based split).

## Current State (Step 4 measurement, June 2025)

**Measurement**: `measure/job_result/perf_setp_4/`, parsed with `analyze_middleware_breakdown` from
`/home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py`
(skip 5 warmup, average runs 6-15).

### Total Times (ms, summed across 113 JOB queries)

| Component | Original | Step 4 | Change |
|-----------|----------|--------|--------|
| MW Overhead | 5,597 | 2,289 | **-3,308 (-59.1%)** |
| Execution | 6,836 | 8,397 | +1,561 (+22.8%) |
| JIT Compile | 181 | 327 | +146 (+80.6%) |
| **Total** | **12,614** | **11,013** | **-1,601 (-12.7%)** |

Vanilla (none-split, no-JIT) execution: **9,444ms**.
Step 4 vs vanilla: +1,569ms (+16.6%).

### Per-Component Breakdown (Step 4)

| Component | ms | % of MW | Category |
|-----------|---:|--------:|----------|
| generate_sub-SQL | 915 | 40.0% | MW (AnalyzeSubIR + EnsureReferencedTempsReady) |
| extract_next_sub-IR | 689 | 30.1% | MW (IR tree traversal + copy) |
| generate_final_sub_sql | 519 | 22.7% | MW (final iteration SQL/analysis) |
| extra_materialization | 87 | 3.8% | MW (async CSR, near-eliminated) |
| parse_sql | 47 | 2.1% | MW |
| Other (prepare, read, show, update) | 32 | 1.4% | MW |
| **MW Total** | **2,289** | — | |
| execute_sub-SQL | 7,202 | — | Execution |
| final_exe | 1,195 | — | Execution |
| jit_compile + jit_compile_final | 327 | — | JIT |

### Per-Query Summary

- 81 of 113 queries improved vs original, 16 regressed
- 39 queries faster than vanilla, 74 slower
- Kernel execution saves 1,047ms vs vanilla but MW+JIT costs 2,616ms

### Gap Decomposition vs Vanilla

| Source | Cost |
|--------|------|
| MW overhead | +2,289ms |
| JIT compile | +327ms |
| Execution savings | -1,047ms |
| **Net gap** | **+1,569ms** |

Break-even requires MW+JIT < 1,047ms. Currently at 2,616ms.

## Completed Optimizations

### CSR/MW Optimizations (this branch)

| Optimization | What | Effect |
|---|---|---|
| **Opt A: Direct PK Index** | Base+Temp joins: PK value = row index, skip CSR entirely | Eliminated dominant runtime CSR for ~25% of iterations |
| **Opt B: Async CSR** | ThreadPool background CSR build for Temp+Temp joins | `extra_materialization` 3,372ms → 87ms |
| **BuildFilteredPKBitset guard** | Skip kernel when filtered PK lookup on >500K row base table | Fixed 3,324ms → 915ms `generate_sub-SQL` regression from Opt B |
| **ThreadPool utility** | Reusable thread pool in `include/util/thread_pool.h` | Supports async CSR |
| **Per-task ResourceTracker** | JIT ResourceTracker per ThreadPool task | Memory safety for async JIT |
| **OpenMP thread cap** | Limit kernel OpenMP threads | Avoid contention with DuckDB threads |

### Kernel Execution Optimizations (Iter 25-30)

| Optimization | What | Effect |
|---|---|---|
| Flat column arrays | Decompress base tables to C arrays at startup | Direct `array[row_id]`, no FSST decompression |
| CSR indexes | Pre-built CSR on all FK columns | O(1) FK→PK lookup replaces hash join |
| Runtime CSR on temps | CSR on temp results for next iteration | Kernel handles ~77% of iterations |
| Filter support | Compile IR filters to `RowPredicate` closures | `=, !=, <, >, IN, NOT IN, IS NULL, LIKE` |
| Dimension cache | Cache tiny tables (<200 rows) as lookup maps | Resolve dim joins at analysis time |
| Sorted indices | Sorted permutation for MIN early termination | O(k) for sorted columns in final aggregate |
| Inverted indexes | 3 indexes: keyword→title, name→title, company→title | Convert 3-table fallback to kernel path |
| LIKE support | 6 optimized pattern kinds in kernel | Kernel handles LIKE natively |
| Unfiltered dim elimination | Remove no-op dim joins | Convert 3-leaf to 2-leaf patterns |

### DuckDB-Side JIT Optimizations (Iter 22-24, affect fallback iterations)

| Optimization | Effect |
|---|---|
| JIT-gated probe/build/BF prefetch | -4.7% execution (none-split) |
| Chain-walk prefetch in AdvancePointers | -218ms (node-based) |
| Row-data prefetch before Match | LLC miss rate -2.8pp |
| ScanInnerJoin equality fast path | Eliminated ResolvePredicates (5.5% → 0%) |
| Skip found_match for INNER/RIGHT | Fewer memory writes |
| Selective LIKE fusion skip | -2.9% (avoid slow JIT LIKE in scan) |
| Range predicate injection | -1.6% (BETWEEN on narrow-range temps) |
| BF scan push-down with match-rate guard | -34% on 29c, -30% on 8b |

## Remaining Optimizations

### Optimization 1: Dim-Partitioned Flat Tables

**BespokeOLAP equivalent**: `db.movie_info.partitions[info_type_id]`, `db.person_info.partitions[info_type_id]`.
BespokeOLAP partitions movie_info (14.8M rows) and person_info by `info_type_id`, scanning only the matching partition.

**What**: Partition large fact tables by low-cardinality FK column:
- `cast_info` (36M rows) by `role_id` (12 values) → avg 3M per partition
- `movie_info` (14.8M rows) by `info_type_id` (113 values) → avg 131K per partition
- `movie_info_idx` by `info_type_id`
- `person_info` by `info_type_id`

**Execution effect**: 7-9x scan reduction. Kernel scans only matching partition instead of full table.

**MW overhead effect**: Smaller temp tables from reduced scan → cheaper runtime CSR build (`extra_materialization`). Fewer rows through the pipeline → less data in `EnsureKernelTempReady` FlatTable copy.

**Estimated impact**: Large — affects the heaviest queries (10c, 19d, 9d where TABLE_SCAN dominates).

**Implementation**: Build partitioned flat tables at startup. In `AnalyzeSubIR`, when a leaf has filter on partition key, select matching partition(s). Kernel scan loop iterates only the partition's rows.

### Optimization 2: Dictionary Encoding

**BespokeOLAP equivalent**: Partial. BespokeOLAP uses `keyword_to_id` hash map + partition-scoped inline string comparison. They avoid full dictionary encoding by partitioning first (reducing scan to ~100K rows).

**What**: Encode string columns as integer dict-IDs. LIKE/equality filters become integer comparisons:
1. Build dictionary at startup: string → dict_id mapping
2. Store dict_id column alongside string column
3. For LIKE filter: pre-scan dictionary for matching entries → set of matching dict_ids
4. Kernel filter: `dict_id IN matching_set` (integer comparison, ~10x faster than `memmem`)

**Execution effect**: ~10x filter speed for LIKE predicates. Currently ~70 queries with LIKE fall back to DuckDB or use slow `memmem` in kernel.

**MW overhead effect**: Eliminates DuckDB fallback iterations for LIKE queries. Each avoided fallback saves per-iteration MW: no SQL generation (`generate_sub-SQL`), no DuckDB JIT compile (`jit_compile`), no lazy FlatTable+CSR build. Estimated 5-15ms MW saved per eliminated fallback iteration.

**Estimated impact**: Large — 70 queries affected. Combined with Opt 1 (partition first, then dict filter on partition) matches BespokeOLAP's approach.

### Optimization 3: Kernel JIT Compilation

**BespokeOLAP equivalent**: AOT-compiled C++ per query with `-O3`. Zero interpreter overhead.

**What**: Compile `SubQueryPlan` to LLVM IR → native function pointer. Eliminates:
- `std::function` dispatch per row (~5-10ns/row)
- Type dispatch in filter evaluation
- Join step iteration overhead
- Virtual call overhead in `ScanRow` lambda

**Execution effect**: -300-400ms across 113 queries (50M+ total rows processed by kernel).

**MW overhead effect**: Adds JIT compile time to `jit_compile` column. Can be hidden by async/speculative compilation on a background thread (compile sub-plan N+1 while executing sub-plan N).

**Estimated impact**: Medium. Pure execution improvement. Compile latency can be hidden.

**Implementation**: `src/jit/kernel_codegen.cpp` — new file. Takes `SubQueryPlan` → emits LLVM IR for the scan+filter+join+emit loop → LLJIT → function pointer.

### Optimization 4: CSR Prefetch in Kernel

**BespokeOLAP equivalent**: No explicit prefetch visible. Relies on compiler `-O3` auto-prefetch for sequential CSR access and tight loop optimization.

**What**: Add `__builtin_prefetch` before CSR `row_ptr`/`col_idx` access in `ExecuteSubQueryPlan`:
```cpp
// Before CSR lookup:
__builtin_prefetch(&csr->row_ptr[key + PF_DIST], 0, 1);
// Before iterating col_idx:
__builtin_prefetch(&csr->col_idx[begin + PF_DIST], 0, 1);
```

**Execution effect**: 5-15% improvement on kernel iterations. CSR access is random (key-dependent), so hardware prefetcher cannot predict it.

**MW overhead effect**: None — pure execution improvement within `execute_sub-SQL`.

**Estimated impact**: Small-medium. Only affects kernel iterations, not DuckDB fallback.

### Optimization 5: Precomputed Bitmaps

**BespokeOLAP equivalent**: `keyword_to_movies` inverted map (functionally equivalent to bitmap — eliminates keyword→movie_keyword→title chain in one lookup). Also implicitly: any dim→fact chain that BespokeOLAP resolves at build time.

**What**: Pre-build bitmaps/bytemaps at startup for common filter patterns:
- `us_movie_bitmap`: company_name.country_code="[us]" → movie_companies → title (14 queries)
- `person_has_aka_bits`: persons that have an aka_name entry (12 queries)

**Execution effect**: Eliminates entire sub-query iterations. Each eliminated iteration saves both execution AND MW overhead.

**MW overhead effect**: Each eliminated iteration saves full per-iteration MW cost:
- `extract_next_sub-IR`: ~1ms
- `generate_sub-SQL`: ~1-5ms
- `execute_sub-SQL`: variable (the main saving)
- `extra_materialization`: ~5-30ms (CSR build on output)
- `update_IR`: ~0.2ms
- Total: 7-36ms MW per eliminated iteration, plus execution time saved.

**Estimated impact**: Medium. 14+12 = 26 queries affected. Each saves 1-2 iterations.

### Optimization 6: More Inverted Indexes

**BespokeOLAP equivalent**: Rich inverted access — `movie_keyword.keyword_to_movies[kid]`, `cast_info.movie_csr`, `aka_name.person_csr`, per-partition CSRs. BespokeOLAP has many more index paths than our 3 inverted indexes.

**What**: Add more inverted indexes beyond the current 3 (keyword→title, name→title, company→title):
- `note_csr`: inverted index on movie_info/cast_info note columns (for LIKE on notes)
- Additional dim→fact mappings for less common patterns
- Per-partition inverted indexes (e.g., within movie_info partition for info_type="genres", index genre→movie_id)

**Execution effect**: Converts 3-table DuckDB fallback patterns to 1-2 table kernel path. Eliminates iterations.

**MW overhead effect**: Fewer iterations → proportionally less MW (same mechanism as Opt 5). Simpler sub-queries → faster `AnalyzeSubIR` in `generate_sub-SQL`.

**Estimated impact**: Medium. Depends on how many additional patterns can be captured.

## Implementation Order

### Step 5: CSR Prefetch in Kernel (next)

Start here. Lowest complexity (~20 lines), pure execution improvement, no MW impact.
The kernel's CSR lookups are random-access (key-dependent) so hardware prefetcher cannot help.
We already proved inline prefetch works in DuckDB's hash join (Iter 22-24, -6.5% execution).
The result also reveals whether the kernel is memory-bound (prefetch helps 10%+) or compute-bound
(prefetch helps <5%), which informs whether dim-partitioning (reduces working set) is worth pursuing next.

### Step 6: Dim-Partitioned Flat Tables

Biggest structural improvement. Matches BespokeOLAP's primary technique. Reduces scan volume
7-9x for the largest tables (cast_info, movie_info). Also reduces temp table sizes → cheaper
runtime CSR. Combined with Step 7 (dictionary encoding), this matches BespokeOLAP's approach
of partition-first, then inline string comparison on the small partition.

### Step 7: Dictionary Encoding

Eliminates DuckDB fallback for ~70 LIKE queries. Each avoided fallback saves 5-15ms MW per
iteration. Best done after dim-partitioning — with partitioned data, the dictionary per partition
is small and cache-friendly.

### Step 8: Precomputed Bitmaps

Low complexity, eliminates entire iterations for 26 queries. `us_movie_bitmap` (14 queries) and
`person_has_aka_bits` (12 queries). Each eliminated iteration saves 7-36ms MW + execution time.

### Step 9: More Inverted Indexes

Extends the 3 existing inverted indexes with additional patterns. Per-partition inverted indexes
become possible after Step 6. `note_csr` for LIKE-on-notes patterns.

### Step 10: Kernel JIT Compilation

Highest complexity, moderate return. Compile `SubQueryPlan` to LLVM native code. Best done last —
the interpreted kernel is correct and functional, and compile latency can be hidden by async
compilation on a background thread.

### Summary

| Step | Optimization | Execution | MW | Complexity |
|------|-------------|-----------|-----|-----------|
| 5 | CSR prefetch | Small-medium | None | Low |
| 6 | Dim-partitioned tables | Large | Medium reduction | Medium |
| 7 | Dictionary encoding | Large | Medium reduction | Medium |
| 8 | Precomputed bitmaps | Medium | Medium reduction | Low |
| 9 | More inverted indexes | Medium | Medium reduction | Medium |
| 10 | Kernel JIT compilation | Medium | Adds JIT time | High |

Steps 5-7 target execution time (the dominant cost at 8,397ms).
Steps 8-9 target both execution and MW (eliminate iterations).
Step 10 targets execution only (interpreter overhead).

## BespokeOLAP Comparison

BespokeOLAP achieves 50-170x faster than DuckDB. Key differences:

| Aspect | BespokeOLAP | Our Query Kernel |
|--------|-------------|------------------|
| Architecture | Single AOT-compiled function per query | Iterative split-execute loop |
| MW overhead | 0ms (no split) | 2,289ms |
| Temp materialization | None (loop fusion) | Per-iteration FlatTable + CSR |
| Join method | CSR only (all pre-built) | CSR + runtime CSR + DuckDB hash join fallback |
| Partitioning | Yes (movie_info, person_info by info_type_id) | Not yet |
| String handling | Inline comparison on partitioned data | `memmem` or DuckDB fallback |
| Inverted indexes | Rich (keyword_to_movies, per-partition CSR) | 3 indexes |
| Compilation | AOT C++ with `-O3` | Interpreted `std::function` dispatch |
| Adaptiveness | None (fixed plan at code-gen time) | Adaptive (real runtime info per iteration) |

Our advantage: adaptive query processing. BespokeOLAP assumes correct statistics at code-gen time. We collect actual cardinality after each sub-plan and adapt. For skewed/new workloads, our approach is more general.

## Assessed and Skipped Optimizations

Previously documented in `kernel_path_opt.md` (deleted) and `temp_csr_opt.txt` (deleted).
All content from those files is captured in this document.

| Optimization | Why Skipped |
|---|---|
| Bloom Filter + Min/Max for CSR probe | Temp+Temp probe overhead is negligible (87ms total). CSR prefetch (Step 5) is better |
| Pipeline kernel with hash join | Different `--jit-level` (`pipeline`), not query-level. Separate optimization direction |
| Sparse CSR for small temps | Superseded by async CSR (Opt B). Small temp CSR build is now hidden behind execution |
| Byte-map for semi-joins | Minor — semi-join iterations are rare after direct PK optimization |
| Skip last-iteration CSR | Async CSR already makes this near-free. Remaining cost is in `generate_sub-SQL` analysis, not CSR build |
| Loop fusion | Attempted and reverted (jit_optimization_claude.md Iter 31) — conflicts with adaptive QP |

## Measurement & Validation

**Measurement data**: `measure/job_result/perf_setp_4/` (latest), `original_perf/`, `perf_setp_1-3/`

**Breakdown CSV parser**: `analyze_middleware_breakdown(csv_file, has_jit=True, is_node_based=True)` in
`/home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py`

**Vanilla baseline**: `duckdb_none_none_o1_none_nofusbuild_nofusprobe_noinlhash_nopayprune_noprefetch_nobatchprobe_breakdown_time_log.csv`
parsed with `analyze_none_split_breakdown(csv_file, has_jit=False)`

**Quick test**:
```bash
cd measure
bash run_job.sh duckdb node-based query o1 none
```

**Full correctness**:
```bash
cd measure
bash check_correctness_duckdb_jit.sh
```

**Full benchmark** (~28 min):
```bash
cd measure
bash measure_breakdown_time_job.sh duckdb node-based query o1 none
```

**Target queries**: 16b (6 iters, large temps), 29c (12 iters, many small temps), 8c (4 iters, DuckDB fallback), 19d (8 iters), 10c (mixed).

## Code Organization

| File | Purpose |
|------|---------|
| `src/kernel/sub_query_plan.cpp` | Query kernel: AnalyzeSubIR + ExecuteSubQueryPlan + AnalyzeFinalIR |
| `include/kernel/sub_query_plan.h` | SubQueryPlan, KernelJoinStep structs |
| `src/split/ir_query_splitter.cpp` | Split loop: extract → analyze → execute → materialize → update |
| `include/split/ir_query_splitter.h` | EnsureReferencedTempsReady, async CSR support |
| `src/storage/storage_plan.cpp` | Flat tables, CSR indexes, inverted indexes, dimension cache |
| `include/storage/storage_plan.h` | StoragePlan API |
| `include/util/thread_pool.h` | Reusable thread pool for async CSR |
| `src/adapters/duckdb_adapter.cpp` | DuckDB integration, temp management, RegisterTempMetadata |
| `src/jit/ir_to_llvm.cpp` | DuckDB JIT: expr/operator compilation for fallback path |
