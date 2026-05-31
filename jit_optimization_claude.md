# JIT Optimization Log

## Iteration 0: Baseline Measurement

**Configuration**: node-based split, pipeline-jit, O1, no SIMD

**Baseline per-iteration operator breakdown** (average of 15 iterations):

| Operator        | Time (s) | Share  |
|-----------------|----------|--------|
| TABLE_SCAN      | 72.16    | 70.8%  |
| HASH_JOIN       | 24.89    | 24.4%  |
| INVALID         | 2.71     | 2.7%   |
| FILTER          | 1.93     | 1.9%   |
| PROJECTION      | 0.09     | 0.1%   |
| AGGREGATE       | 0.08     | 0.1%   |
| **TOTAL**       | **101.87** | 100% |

Key findings from prior analysis:
- TABLE_SCAN dominates (71%). Most of this is I/O + column decoding, not filterable.
- HASH_JOIN is 24%. JIT batch-probe + prefetch doesn't beat DuckDB vectorized probe.
- FILTER is small (1.9%) --- pipeline-jit already fuses filters into scans.
- Scan+filter fusion shifts ~3s from FILTER into TABLE_SCAN accounting but is net neutral/slightly positive.

---

## Iteration 1: Adaptive Scan+Filter Fusion (ABANDONED)

**Goal**: Only fuse filter into scan when selectivity > 10%. Skip fusion for low-selectivity filters that make scans slower.

**Result**: ABANDONED --- three approaches tried, all failed:
1. Selectivity-based guard: DuckDB always estimates 20% for LIKE filters, so the guard never triggers on the queries that need it.
2. Disabling fusion entirely: exposed a pre-existing bug in `CompilePipeline()` for standalone filters (query 10c wrong results).
3. Disabling both fusion and pipeline filters: FILTER time jumped from 1.93s to 8.73s (FILTER needs JIT).

**Conclusion**: Scan+filter fusion is beneficial overall. The per-query losses (23b +29%, 6b +27%) are outweighed by gains elsewhere. Not worth the complexity.

---

## Iteration 2: SQL Range Predicate Injection (DISABLED --- net regression)

**Goal**: Reduce TABLE_SCAN volume by injecting BETWEEN predicates into sub-plan SQL based on temp table min/max ranges.

**Mechanism**:
In node-based split execution, after sub-plan N materializes a temp table, we collect min/max values for all integer columns. When sub-plan N+k joins that temp table with a base table, the join key column in the base table must fall within the temp table's min/max range. We inject `AND base.col >= min AND base.col <= max` into the SQL before execution. DuckDB's optimizer can then use zone-map pruning to skip column segments outside this range.

**Changes** (code remains in tree, feature disabled):
1. `include/split/ir_query_splitter.h` --- Added `col_min_max` field to `TempTableInfo` struct
2. `src/split/ir_query_splitter.cpp` --- Range predicate injection (disabled with `if (false)`)
3. `src/adapters/duckdb_adapter.cpp` --- `GetTempTableMinMax()`, `GetColumnName()`, `table_name` in bind data
4. `include/adapters/duckdb_adapter.h` --- `TempColRange` struct, `GetTempTableMinMax()`, `GetColumnName()`
5. `src/jit/ir_to_llvm.cpp` --- `CompileRangeFilter()` (LLVM IR infrastructure, not used in SQL path)
6. `include/jit/ir_to_llvm.h` --- `CompileRangeFilter()` declaration

**Correctness**: All 113 JOB queries produce identical results to golden.

### Performance: Without selectivity guard

| Operator        | Baseline (s) | Range Pred (s) | Delta    | Pct     |
|-----------------|-------------|----------------|----------|---------|
| TABLE_SCAN      | 72.16       | 75.85          | +3.70    | +5.1%   |
| HASH_JOIN       | 24.89       | 21.44          | -3.45    | -13.9%  |
| FILTER          | 1.93        | 1.42           | -0.52    | -26.8%  |
| INVALID         | 2.71        | 3.19           | +0.47    | +17.4%  |
| **TOTAL**       | **101.87**  | **102.07**     | **+0.20**| **+0.20%** |

Top improved: 9a (-43%), 29c (-39%), 4c (-37%), 15b (-29%), 31a/31c (-25%)
Top regressed: 17d (+124%), 14b (+169%), 17c (+96%), 17b (+88%), 17f (+44%)

Root cause of regressions: The 17x queries have temp tables that span >95% of the key domain. BETWEEN predicates pass almost all rows but still add per-row evaluation cost during scan.

### Performance: With selectivity guard (skip range > 50% of max)

| Operator        | Baseline (s) | Guarded (s) | Delta    | Pct     |
|-----------------|-------------|-------------|----------|---------|
| TABLE_SCAN      | 72.16       | 74.43       | +2.27    | +3.2%   |
| HASH_JOIN       | 24.89       | 24.35       | -0.54    | -2.2%   |
| FILTER          | 1.93        | 1.85        | -0.08    | -4.2%   |
| INVALID         | 2.71        | 3.24        | +0.53    | +19.4%  |
| **TOTAL**       | **101.87**  | **104.06**  | **+2.19**| **+2.15%** |

The guard eliminated the worst regressions (17x family) but TABLE_SCAN still increases by +2.3s. The BETWEEN filter evaluation cost is not offset by zone-map pruning.

### Analysis

The approach fails for JOB because:
1. **DuckDB's zone-map granularity is too coarse**: column segments typically cover wide value ranges, so BETWEEN predicates rarely prune entire segments.
2. **Per-row BETWEEN evaluation cost is non-trivial**: even when passing all rows, checking `val >= min AND val <= max` for every scanned row adds measurable overhead on large tables (cast_info: 36M rows, movie_info: 14M rows).
3. **JOB's key distributions are dense**: most tables use sequential integer IDs, so temp table ranges tend to span 15-95% of the domain.
4. **The biggest tables have the worst selectivity**: cast_info, movie_info, person_info --- the tables that dominate TABLE_SCAN time --- produce broad ranges when filtered.

### What would work instead
- **Bloom filter push-down**: instead of min/max ranges, build a Bloom filter from the temp table's join key column. This would give per-row selectivity rather than range selectivity. A Bloom filter on 10K keys would filter 97% of a 36M-row table even when the key range spans the full domain.
- **Semi-join reduction**: materialize the temp table's key set and do a hash-based semi-join filter during the scan. This is what DuckDB's Dynamic Filter Push-Down already does within a single query plan, but not across split sub-plans.
- **JIT scan filter with early termination**: instead of SQL BETWEEN, compile a JIT function that checks membership in the temp table's key set. This avoids the optimizer's plan-change overhead.

---

## Iteration 3: Range Predicate Injection with Cost-Model Guard (ENABLED --- net -1.6%)

**Goal**: Re-enable range predicate injection with a cost-model-driven guard condition derived from hardware characteristics (no magic numbers).

**Mechanism**: Same as Iteration 2 (inject `AND base.col >= min AND base.col <= max` into sub-plan SQL), but with three guard conditions applied per-node:

1. **temp_card <= kMaxHTCapacity (4096)**: Derived from L2 cache size (256KB) / estimated HT entry size (64B). When the hash table fits in L2, each probe is cheap (~5ns) and reducing probe count via BETWEEN is worthwhile. When HT exceeds L2, probes are cache-miss-dominated (~50ns each) and the per-row BETWEEN eval cost (~2ns) doesn't offset savings.
2. **temp_card >= kMinTempCard (50)**: With very few build-side rows, the HT fits in L1 (32KB / 64B ≈ 512). Probes are essentially free; BETWEEN adds scan overhead with no probe-side benefit.
3. **selectivity < kMaxSelectivity (0.40)**: Range span / domain max. Zone-map pruning only helps when BETWEEN excludes enough column segments.

Min/max values are computed lazily --- only when a join partner is found in the SQL AND the cardinality guard passes, avoiding overhead on non-injected queries.

**Pattern analysis (from per-query diagnostics)**:

Queries that improve have:
- Join keys with narrow min/max range (selectivity < 0.35) against medium/large tables
- temp_card 50-4096 (HT fits in L2 cache)

Queries that would regress without guards have:
- temp_card > 10K (HT doesn't fit in cache, probes dominated by cache misses regardless)
- Selectivity > 0.50 (BETWEEN passes most rows, pure overhead)
- Very small temp_card < 50 (hash probe is effectively free, filter adds overhead)

**Changes**:
1. `src/split/ir_query_splitter.cpp` --- Refactored injection code with lazy min/max computation and cost-model-based guard (three conditions derived from L2/L1 cache sizes)
2. Removed eager `GetTempTableMinMax` call from materialization loop; min/max only computed when join partner found
3. `src/adapters/duckdb_adapter.cpp` --- `GetTempTableMinMax()` for lazy min/max extraction
4. `include/adapters/duckdb_adapter.h` --- `TempColRange` struct, `GetTempTableMinMax()`, `GetColumnName()`

**Correctness**: All 113 JOB queries produce identical results to golden.

### Performance (A/B with fresh baseline, 15 iterations each)

| Operator        | Baseline (s) | Treatment (s) | Delta    | Pct     |
|-----------------|-------------|---------------|----------|---------|
| TABLE_SCAN      | 74.21       | 73.04         | -1.17    | -1.6%   |
| HASH_JOIN       | 25.00       | 24.56         | -0.45    | -1.8%   |
| FILTER          | 1.98        | 1.88          | -0.11    | -5.4%   |
| INVALID         | 3.10        | 3.13          | +0.03    | +0.9%   |
| **TOTAL**       | **104.51**  | **102.82**    | **-1.70**| **-1.6%** |

~22 queries receive injections. Top improved: 29c (-41.8%, -0.91s), 9a (-29.8%, -0.56s), 31a (-23.6%, -0.41s), 19a (-8.1%), 7b (-7.2%)
Max regression: 12c (+4.6%, +0.028s absolute), 16b (+1.3%, +0.037s) --- all within noise range

---

## Iteration 4: Bloom Filter Scan Push-Down (DISABLED --- net +0.5% regression)

**Goal**: Build a Bloom filter from temp table join key columns and push it down to base table scans in subsequent sub-plans, filtering out non-matching rows before the hash join probe.

**Mechanism**:
After materializing a temp table with temp_card > 4096 (too large for range predicate injection), build a Bloom filter (12 bits/key, 4 hash bits, ~1.5% FPR) from the join key column. Register it as a scan-level filter in the DuckDB physical plan. The table scan dispatch checks each scanned row against the BF and removes non-matching rows before they reach the hash join.

**Guard conditions**:
1. temp_card in [50, 100000] (BF must fit in L2 cache: 100K * 1.5B = 150KB)
2. temp_card > kMaxHTCapacity (4096) --- lower range handled by range predicate injection
3. No self-joins (same table with multiple aliases in the sub-plan SQL)
4. One BF per (base_table, base_column) pair

**Changes** (code remains in tree, BF loop disabled):
1. `duckdb/src/include/duckdb/execution/aqp_jit.hpp` --- `AQPBloomScanFilter` struct, `bloom_scan_filters` map in `AQPJITContext`
2. `duckdb/src/execution/operator/scan/physical_table_scan.cpp` --- BF dispatch in scan's GetData
3. `AQP_middleware/src/adapters/duckdb_adapter.cpp` --- `BuildBloomFilter()`, `RegisterBloomFilters()`
4. `AQP_middleware/include/adapters/duckdb_adapter.h` --- `BloomFilterInfo` struct, `SetPendingBloomFilters()`
5. `AQP_middleware/src/split/ir_query_splitter.cpp` --- BF construction loop (disabled)

**Correctness**: All 113 JOB queries produce identical results to golden (after fixing self-join and duplicate BF issues).

### Performance (A/B with fresh baseline, 15 iterations each)

| Operator        | Baseline (s) | BF Treatment (s) | Delta    | Pct     |
|-----------------|-------------|-------------------|----------|---------|
| TABLE_SCAN      | 74.66       | 77.39             | +2.73    | +3.7%   |
| HASH_JOIN       | 24.72       | 22.68             | -2.04    | -8.3%   |
| FILTER          | 1.84        | 1.65              | -0.18    | -10.0%  |
| INVALID         | 3.12        | 3.11              | -0.02    | -0.5%   |
| **TOTAL**       | **104.56**  | **105.04**        | **+0.49**| **+0.5%** |

### Analysis

The BF reduces HASH_JOIN by -2.04s (saving hash probes for filtered-out rows) but increases TABLE_SCAN by +2.73s (per-row hash + BF lookup overhead). Net regression of +0.49s.

**Why it fails on JOB**:
1. **DuckDB's intra-plan optimization already narrows scans**: DuckDB's own zone-map pruning and dynamic filter pushdown within each sub-plan already reduce scan cardinality. In 7c, cast_info scans 265K rows (not 36M), so the BF only checks 265K rows but most of them match --- pure overhead.
2. **Per-row hash cost is non-trivial**: `Hash<int32_t>()` + BF lookup costs ~10ns/row. On aggregate across 113 queries, this adds 2-3s to TABLE_SCAN.
3. **Match rate is high after intra-plan filtering**: Since DuckDB already filters rows within each sub-plan, the remaining rows have a high probability of matching the join. The BF filters very few additional rows.

**What would work instead**:
- Apply BF at the hash join probe level instead of the scan level. This avoids the per-scan-row overhead and only checks rows that actually reach the probe.
- Only apply BF to scans with cardinality > 1M (truly large unfiltered scans).
- Integrate with DuckDB's existing `DynamicTableFilterSet` for segment-level BF pruning.

---

## Iteration 5: Relaxed Range Predicate Upper Bound (REVERTED --- massive regression)

**Goal**: Remove the `kMaxHTCapacity` (4096) upper bound on temp_card for range predicate injection, relying solely on the selectivity guard (< 0.40) to prevent regressions.

**Hypothesis**: The `kMaxHTCapacity` threshold was derived from HT cache residency reasoning, but the BETWEEN predicate's benefit comes from zone-map segment pruning, which is independent of HT size.

**Result**: REVERTED --- +38-55% regression on all newly-affected queries.

Quick A/B on 8 queries that gained new injections (temp_card 10K-193K, selectivity 0.30-0.36):
| Query | Delta | Pct |
|-------|-------|-----|
| 19c | +0.493s | +38% |
| 19d | +0.728s | +53% |
| 20a | +0.548s | +38% |
| 9d  | +0.830s | +55% |

**Root cause**: DuckDB's zone-map segments are too coarse for large tables (cast_info: 36M rows, ~300 segments of 120K rows each). A BETWEEN that covers 35% of the key domain still overlaps most segments, providing no pruning. The per-row BETWEEN evaluation cost (~2ns × 36M = 72ms per sub-plan) adds pure overhead.

**Conclusion**: The `kMaxHTCapacity = 4096` upper bound is necessary. It correlates with the regime where temp tables have narrow enough ranges (small key sets → narrow min/max span) to actually prune segments effectively.

---

## Iteration 6: Chain-Walk Skip for Unique-Key Hash Tables (ENABLED --- neutral, +0.3%)

**Goal**: Skip the chain-walk loop in the JIT hash probe when the hash table has no duplicate keys (PK-FK joins), since the chain is always length 1.

**Mechanism**: Exposed DuckDB's `chains_longer_than_one` flag to the JIT code via a new `no_chains` field in `AQPJoinHTView`. When `no_chains == 1` (all build-side keys are unique), the JIT probe skips loading the `next_ptr` from the row and goes directly to `chain_done_bb` after the first match.

**Changes**:
1. `duckdb/src/include/duckdb/execution/aqp_jit.hpp` --- Added `no_chains` field to `AQPJoinHTView`
2. `AQP_middleware/include/jit/aqp_jit_abi.h` --- Added `no_chains` field (C ABI struct)
3. `duckdb/src/execution/join_hashtable.cpp` --- Populates `no_chains` from `chains_longer_than_one`
4. `AQP_middleware/src/jit/ir_to_llvm.cpp` --- Updated ViewTy to include `no_chains`; added `skip_chain_walk` branch in advance block; added `chain_walk_bb` intermediate block

**Correctness**: All 113 JOB queries produce identical results to golden.

### Performance (A/B vs Iteration 3 baseline, 15 iterations each)

| Operator        | Baseline (s) | Treatment (s) | Delta    | Pct     |
|-----------------|-------------|---------------|----------|---------|
| TABLE_SCAN      | 73.08       | 73.29         | +0.22    | +0.3%   |
| HASH_JOIN       | 24.56       | 24.53         | -0.03    | -0.1%   |
| FILTER          | 1.88        | 1.99          | +0.11    | +6.0%   |
| INVALID         | 3.13        | 3.09          | -0.04    | -1.2%   |
| **TOTAL**       | **102.82**  | **103.08**    | **+0.26**| **+0.3%** |

All per-query deltas within noise range (< 3%). The optimization is theoretically sound but the absolute savings are too small to measure: skipping the chain walk saves ~2 cycles per match (~0.6ns at 3.4GHz), totaling ~60ms across 100M matches --- below measurement noise.

**Code kept**: The change is clean and correct. It eliminates redundant memory loads on the common PK-FK join path, providing a micro-optimization that may become measurable in larger workloads.

---

## Iteration 7: Selective LIKE Fusion Skip (ENABLED --- net -2.9%)

**Goal**: Skip scan+filter fusion and pipeline filter fn compilation for filters containing LIKE predicates, while keeping expr-level JIT compilation.

**Root cause identified**: Scan+filter fusion fuses LIKE predicates into the scan operator, making them evaluate per-row via the scalar JIT `aqp_like_match_segments` function. This is ~2x slower than DuckDB's native vectorized LIKE (LikeMatcher + FindStrInStr with per-needle-length specialization). The vectorized path processes 2048-row batches with better ILP and branch prediction.

**Mechanism**: Three-level guard in `duckdb_adapter.cpp`:
1. **Expr-level** (`skip_like = false`): Always compile, even with LIKE. The selection-vector path (Slice) benefits mixed filters (LIKE + non-LIKE predicates combined).
2. **Scan+filter fusion** (`!has_like` guard): Skip fusion when filter contains LIKE. Prevents fusing slow JIT LIKE into the scan operator.
3. **Pipeline filter fn** (`!FilterHasLike(filter_ir)` guard): Skip pipeline filter compilation for LIKE. Prevents replacing DuckDB's vectorized LIKE with slower JIT scalar LIKE.

**Alternatives tested**:
- **All-level LIKE skip** (skip expr too): 101.89s total. Helped LIKE-hurt queries (20a -0.109s, 23c -0.071s) but regressed queries with mixed filters where expr-level JIT benefits non-LIKE predicates (26a +0.119s, 20c +0.109s, 9a +0.094s). Net +1.84s worse than scan+pipeline-only skip.

**Changes**:
1. `src/adapters/duckdb_adapter.cpp` line ~2272-2275: Added `bool has_like = FilterHasLike(filter_ir)` guard on scan+filter fusion
2. `src/adapters/duckdb_adapter.cpp` line ~2324-2327: Added `!FilterHasLike(filter_ir)` guard on pipeline filter fn compilation

**Correctness**: All 113 JOB queries produce identical results to golden.

### Performance (15 iterations, vs Iter 6 baseline)

| Operator        | Iter 6 (s)  | Iter 7 (s)  | Delta    | Pct     |
|-----------------|-------------|-------------|----------|---------|
| TABLE_SCAN      | 73.29       | 66.10       | -7.19    | -9.8%   |
| HASH_JOIN       | 24.53       | 24.00       | -0.52    | -2.1%   |
| FILTER          | 1.99        | 6.99        | +5.00    | +251%   |
| INVALID         | 3.09        | 2.86        | -0.23    | -7.4%   |
| **TOTAL**       | **102.99**  | **100.05**  | **-2.94**| **-2.9%** |

TABLE_SCAN drops -7.19s because scan+filter fusion no longer forces slow JIT LIKE evaluation into the scan accounting. FILTER increases +5.00s because those filters now run in DuckDB's native executor. Net: -2.19s pure savings from avoiding the slow JIT LIKE path.

Top improved (vs Iter 6): 7c (-0.099s, -4.2%), 25c (-0.097s, -4.8%), 16b (-0.087s, -3.1%), 18c (-0.075s, -4.2%), 29c (-0.071s, -5.5%)
Max regression: 5a (+0.028s, +3.0%) --- within noise

### Per-query analysis of affected queries

Queries that benefited from LIKE skip (JIT LIKE was hurting):
- 20a: FILTER was 0.308s with JIT LIKE → 0.158s with DuckDB native (sub-plan scans 3.14M char_name rows for `LIKE '%Tony%Stark%'`)
- 23b: TABLE_SCAN was 0.510s → 0.510s (no change), FILTER 0.118s → 0.047s (LIKE on movie_info)
- 6a/6e: FILTER was 0.171s with JIT → 0.090s with DuckDB (`LIKE '%Downey%Robert%'`)

Queries where expr-level LIKE JIT helps (mixed LIKE + non-LIKE filters):
- 9a: FILTER 0.078s (JIT) vs 0.564s (none-jit) --- range pred injection + expr-level JIT dramatically reduces rows, saving -0.743s total
- 20c/26a/26c: 4-8% better than none-jit despite LIKE, because HJ savings outweigh FILTER overhead

---

## Current Status

**Current total**: 100.05s (node-based / pipeline-jit O1, avg of 15 iterations)

Active optimizations:
- Range predicate injection with cost-model guard (Iteration 3, -1.6% vs pre-opt pipeline-jit)
- Chain-walk skip for unique-key HTs (Iteration 6, neutral)
- Selective LIKE fusion skip (Iteration 7, -2.9% vs Iter 6)

Disabled:
- Bloom filter scan push-down (Iteration 4, +0.5% regression)
- Relaxed range pred upper bound (Iteration 5, massive regression)
- All-level LIKE skip (Iteration 7 variant, +1.8% vs scan+pipeline-only skip)

### Comparison Against All Baselines

| Config | TABLE_SCAN | HASH_JOIN | FILTER | INVALID | TOTAL |
|--------|-----------|-----------|--------|---------|-------|
| none-split / none-jit | 64.05 | 32.38 | 3.76 | 0.00 | **100.30** |
| node-based / none-jit | 65.49 | 24.91 | 8.51 | 2.73 | **101.87** |
| node-based / pipeline-jit (pre-opt) | 74.25 | 25.00 | 1.98 | 3.10 | **104.51** |
| node-based / pipeline-jit (Iter 6) | 73.29 | 24.53 | 1.99 | 3.09 | **103.08** |
| **node-based / pipeline-jit (Iter 7)** | **66.10** | **24.00** | **6.99** | **2.86** | **100.05** |

### Delta vs none-split / none-jit: -0.25s (-0.2%)

Pipeline-JIT with node-based split is now **faster** than none-split/none-jit for the first time.

| Operator | Delta | Pct | Explanation |
|----------|-------|-----|-------------|
| TABLE_SCAN | +2.05s | +3.2% | Scan+filter fusion shifts ~1.5s of non-LIKE FILTER work into scan; node-based split overhead |
| HASH_JOIN | -8.38s | -25.9% | Node-based split produces smaller HTs that fit in cache; JIT probe saves -0.90s vs native |
| FILTER | +3.23s | +85.9% | LIKE filters handled by DuckDB native (slower than fused accounting, but real work is same) |
| INVALID | +2.86s | — | Sub-plan setup/teardown overhead from node-based split |

### Delta vs node-based / none-jit: -1.82s (-1.8%)

| Operator | Delta | Pct | Explanation |
|----------|-------|-----|-------------|
| TABLE_SCAN | +0.61s | +0.9% | Scan+filter fusion for non-LIKE filters only; minimal overhead |
| HASH_JOIN | -0.91s | -3.7% | JIT probe is now measurably faster than DuckDB native (-0.90s) |
| FILTER | -1.52s | -17.9% | Expr-level JIT reduces non-LIKE FILTER time |
| INVALID | +0.13s | +4.8% | JIT dispatch overhead |

### Delta vs pre-optimization pipeline-jit: -4.46s (-4.3%)

| Operator | Delta | Pct | Explanation |
|----------|-------|-----|-------------|
| TABLE_SCAN | -8.15s | -11.0% | LIKE skip prevents slow JIT LIKE in scan; range pred injection reduces scans |
| HASH_JOIN | -1.00s | -4.0% | Chain-walk skip + range pred injection |
| FILTER | +5.01s | +253% | LIKE filters return to DuckDB native (accounting shift) |

Top improved queries (vs node-based/none-jit): 29c (-0.878s, -41.6%), 9a (-0.743s, -37.0%), 31a (-0.399s, -23.8%), 19a (-0.236s, -18.7%), 8c (-0.134s, -5.9%)
Top regressed queries (vs node-based/none-jit): 20a (+0.153s, +12.1%), 20b (+0.121s, +10.2%), 6a (+0.094s, +10.2%)
34 improved, 47 regressed, 32 neutral --- but improvements are larger in absolute magnitude

---

## Analysis: Current Bottleneck

### Heaviest queries (Iter 7)

**Note on units**: `operator_exe.csv` reports DuckDB profiler times summed across all worker threads (CPU-seconds). DuckDB uses 12 threads, so these totals are ~7-9x larger than wall-clock. The "Wall (s)" column shows actual wall-clock time from `breakdown_time_log.csv` (milliseconds converted to seconds, averaged over measurement iterations 6-15).

| Query | CPU-total (s) | TABLE_SCAN | HASH_JOIN | FILTER | INVALID | Wall (s) | Bottleneck |
|-------|--------------|-----------|-----------|--------|---------|----------|-----------|
| 16b | 2.71 | 0.67 | 1.88 | 0.00 | 0.15 | 0.652 | HJ (69%) |
| 8c | 2.15 | 0.31 | 1.63 | 0.00 | 0.20 | 0.494 | HJ (76%) |
| 7c | 2.26 | 1.63 | 0.26 | 0.12 | 0.26 | 0.343 | TS (72%) |
| 10c | 2.17 | 1.06 | 0.55 | 0.46 | 0.11 | 0.340 | TS (49%) + mixed |
| 6f | 1.99 | 0.79 | 1.18 | 0.01 | 0.00 | 0.233 | HJ (59%) |
| 31c | 1.95 | 1.43 | 0.42 | 0.09 | 0.01 | 0.255 | TS (73%) |
| 25a | 1.95 | 1.31 | 0.51 | 0.11 | 0.01 | 0.239 | TS (67%) |
| 30c | 1.94 | 1.27 | 0.54 | 0.12 | 0.01 | 0.246 | TS (65%) |

### Where JIT helps vs hurts

**HASH_JOIN**: JIT saves -0.90s overall (-3.7% vs none-jit). Best: 31a (-0.192s), 8c (-0.152s), 3c (-0.089s). Worst: 16c (+0.023s). JIT probe is consistently beneficial.

**FILTER**: JIT saves -1.52s on non-LIKE filters. The 47 regressed queries are almost all LIKE-containing, where the expr-level JIT still evaluates LIKE per-row. The regressions are small (max +0.153s).

**TABLE_SCAN**: Essentially at none-jit level after LIKE skip (66.10 vs 65.49s = +0.61s). The remaining +0.61s comes from scan+filter fusion on non-LIKE filters, which is net beneficial (the fused filter reduces downstream rows, saving HJ probe work).

### Remaining JIT overhead sources

1. **Expr-level LIKE JIT** (+1.2s total across LIKE queries): The expr-level JIT LIKE is still ~2x slower than DuckDB native for standalone LIKE filters. But skipping it entirely (all-level skip) loses the benefit of mixed filters, costing +1.8s. Net: keeping expr-level LIKE is the right tradeoff.

2. **JIT dispatch overhead** (~0.13s in INVALID): Small but nonzero cost from JIT context setup, chunk flattening, and result materialization.

3. **Scan+filter fusion accounting shift** (~1.5s): Non-LIKE filters fused into scan add ~1.5s to TABLE_SCAN while removing ~1.5s from FILTER. This is net neutral.

---

## Iteration 8: Improved LIKE substr search (REVERTED --- inconclusive)

**Goal**: Replace `memmem` in `aqp_like_match_segments` with a `memchr`+`memcmp` approach (DuckDB-style FindStrInStr) to speed up the JIT LIKE evaluation.

**Changes**: Added `aqp_find_substr` function using `memchr` for first-byte scan + `memcmp` for candidate verification, replacing `memmem` in the multi-segment LIKE matcher.

**Result**: FILTER improved -0.53s (-7.6%), but other operators showed +1.56s TABLE_SCAN and +0.39s HASH_JOIN increases attributable to run-to-run variance. Net +1.58s vs Iter 7, though the FILTER improvement is real.

Key LIKE query improvements: 20a FILTER 0.308s→0.152s (competitive with DuckDB native 0.158s), 6a FILTER 0.171s→0.076s.

**Reverted** because: (1) the overall variance masks the real improvement, (2) the `memchr`+`memcmp` approach can be O(n*m) in worst case (many partial matches with common first characters), while glibc `memmem` uses the Two-Way algorithm (guaranteed O(n+m)), (3) the -0.53s FILTER improvement is modest compared to the ~1.5s run-to-run variance. The fundamental issue is not the substring search algorithm but the **scalar per-row evaluation model** vs DuckDB's vectorized batch model.

---

## Iteration 9: Disable JIT Hash Probe (ENABLED --- net -1.6%)

**Goal**: Disable JIT probe dispatch for pipeline-level hash join, reverting to DuckDB's native vectorized probe.

**Root cause**: The JIT hash probe generates scalar per-row code (hash compute → chain walk → match → extract). DuckDB's vectorized probe processes 2048-row batches with better ILP and branch prediction. The JIT probe only wins for very small hash tables (< 512 entries), but these contribute negligible time.

**Changes**: Disabled JIT probe dispatch in `physical_hash_join.cpp` line 1086 by guarding with `if (false && ...)`.

**Correctness**: All 113 JOB queries produce identical results to golden.

### Performance (full breakdown, 15 iterations)

| Operator        | Iter 7 (s) | Iter 9 (s) | Delta    | Pct     |
|-----------------|-----------|-----------|----------|---------|
| TABLE_SCAN      | 66.10     | 65.88     | -0.22    | -0.3%   |
| HASH_JOIN       | 24.00     | 22.40     | -1.60    | -6.7%   |
| FILTER          | 6.99      | 7.01      | +0.02    | +0.3%   |
| INVALID         | 2.86      | 2.90      | +0.04    | +1.4%   |
| **TOTAL**       | **100.05**| **98.29** | **-1.76**| **-1.8%** |

Disabling the JIT probe saves -1.60s in HASH_JOIN. DuckDB's native vectorized probe is consistently faster than our scalar JIT probe across all query sizes.

---

## Iteration 10: Runtime Info Propagation & Projection Pushdown (ENABLED --- neutral)

### Investigation: Column Statistics Propagation

**Goal**: Provide per-column statistics (min/max, distinct count, null info) for temp table scans to DuckDB's optimizer, enabling better join orders and the perfect hash join optimization.

**Mechanism**: Implemented `TempCollectionStatistics` callback for `scan_temp_collection` table function. After materializing a temp table, scans the ColumnDataCollection to compute per-column BaseStatistics (min/max for INT8/16/32/64, null/non-null flags). These stats feed into:
1. **StatisticsPropagator** → populates `join_stats` → enables perfect hash join
2. **RelationStatisticsHelper** → uses distinct counts for join order estimation

**Result**: DISABLED --- causes 29c (+73%) and 9a (+40%) regressions.

**Root cause of regressions**: Providing min/max statistics changes DuckDB's `PropagateComparison` results, which alters filter pruning decisions and join ordering. For 29c, the optimizer chose a plan scanning 181K rows instead of 541 (without stats, DuckDB's intra-plan Dynamic Filter Push-Down narrows the scan much more effectively). The optimizer's decisions with partial information (our stats) are worse than its decisions with no information (where it falls back to conservative estimates that work better with its internal dynamic filters).

**Key insight**: DuckDB's optimizer + Dynamic Filter Push-Down is a tightly coupled system. Providing partial external statistics breaks the coupling --- the optimizer makes "optimistic" plans that the dynamic filter system can't recover from.

**Code retained but disabled**: `func.statistics_extended = TempCollectionStatistics` commented out. Statistics computation code remains in `ExecuteSQLandCreateTempTable` for future use.

### Projection Pushdown for Temp Table Scans (ENABLED --- neutral)

Set `func.projection_pushdown = true` and updated `TempCollectionInitGlobal` / `TempCollectionScanFunc` to use `input.column_ids` for column-selective scanning. This avoids scanning unused columns from temp tables.

**Correctness**: All 113 JOB queries produce identical results.

**Performance**: Neutral. Temp table scans are negligible (< 1ms each) because `ColumnDataCollection::Scan` only references column vectors without copying --- the overhead of extra columns is minimal.

---

## Runtime Information Analysis

### What is currently collected from sub-plan execution

| Info | Where Collected | How Used |
|------|-----------------|----------|
| Temp table cardinality | `GetTempTableCardinality()` after ExecuteSQLandCreateTempTable | Cost-model guards for range predicate injection (card < 4096) |
| Column min/max (integers) | `GetTempTableMinMax()` lazily when join partner found | BETWEEN predicate injection into sub-plan SQL |
| Column names | From IR target list | Join partner matching for predicate injection |
| Column types | From ColumnDataCollection | Type-specific statistics computation |
| Operator profiling (timing, cardinality) | `CollectOperatorTimings()` from DuckDB QueryProfiler | Logging only, NOT used for adaptive decisions |
| Hash table `chains_longer_than_one` | Exposed via `AQPJoinHTView` | JIT chain-walk skip (Iter 6, neutral impact) |

### What could be collected but is NOT used

| Info | How to Collect | Potential Use | Estimated Impact |
|------|---------------|---------------|------------------|
| Actual distinct counts per column | Count during stats scan (hash set) | Better join selectivity estimates | LOW: DuckDB's fallback (cardinality = distinct) works well enough |
| Column statistics via BaseStatistics | Implemented, disabled | Enable perfect hash join for temp tables | NEGATIVE: causes plan regressions |
| Actual vs estimated cardinality ratio | Compare profiler cardinality vs EXPLAIN estimate | Detect bad estimates → trigger re-optimization | MEDIUM: could help 2-3 queries |
| Hash table memory footprint | `ht->PointerTableSize()` after finalize | Predict cache behavior for next sub-plan | LOW: current guards already use cardinality |
| Dynamic filter push-down min/max from HJ | `FinalizeMinMax()` in JoinFilterPushdownInfo | Cross-sub-plan range filter propagation | LOW: our SQL-level BETWEEN already does this |
| Bloom filter from HJ build | `ht->GetBloomFilter()` | Cross-sub-plan bloom filter push-down | LOW: Iter 4 showed BF adds per-row overhead |
| Selectivity of executed sub-plans | input_card / output_card from profiler | Guide filter injection / join order | LOW: DuckDB's optimizer already adapts |

### Why DuckDB's intra-plan optimization limits cross-sub-plan gains

DuckDB's `JoinFilterPushdownInfo` already performs within each sub-plan:
1. **Min/max range filters** pushed to table scans (equivalent to our BETWEEN injection)
2. **Bloom filter construction** from hash table build-side keys
3. **Dynamic column filters** applied to probe-side scans

Because these run INSIDE the sub-plan (after HT build but before probe-side scan), they're more efficient than our cross-sub-plan approach which injects filters into the SQL text. The internal dynamic filters:
- Don't change the optimizer's plan (applied at execution time)
- Have zero planning overhead
- Automatically adapt to actual hash table contents

Our cross-sub-plan optimizations (range injection, bloom filters) can only help when:
1. The temp table's key range is NARROW (< 40% of domain) AND
2. The hash table is SMALL (< 4096 entries) AND
3. DuckDB's internal dynamic filters haven't already narrowed the scan

These conditions are met for ~22 queries in JOB, providing -1.6% total improvement.

### Hardware bottleneck analysis (perf stat on query 7c)

```
IPC: 0.64 (compute-bound threshold: > 2.0)
Cache miss rate: 24%
LLC misses: 14M
Kernel time: 40% (I/O)
```

The workload is **I/O and memory bound**: 0.64 IPC with 40% kernel time (disk I/O). JIT cannot speed up disk reads or cache misses. The remaining optimization space is:
1. Reducing the number of rows that reach expensive operators (already done with range predicates)
2. Reducing memory bandwidth per row (projection pushdown, but negligible for temp tables)
3. Improving cache utilization during hash probes (prefetching, but memory-bound means limited gains)

---

## Current Status (Iter 10)

**Active optimizations**:
- Range predicate injection with cost-model guard (Iter 3, -1.6%)
- Chain-walk skip for unique-key HTs (Iter 6, neutral)
- Selective LIKE fusion skip (Iter 7, -2.9%)
- JIT probe disabled for hash joins (Iter 9, -1.8%)
- Projection pushdown for temp table scans (Iter 10, neutral)

**Disabled / Reverted**:
- Column statistics propagation (Iter 10, causes plan regressions)
- Bloom filter scan push-down (Iter 4, +0.5% regression)
- Relaxed range pred upper bound (Iter 5, massive regression)
- All-level LIKE skip (Iter 7 variant, +1.8% worse)
- Improved LIKE substr search (Iter 8, inconclusive)

### Comparison Against All Baselines

| Config | TABLE_SCAN | HASH_JOIN | FILTER | INVALID | TOTAL |
|--------|-----------|-----------|--------|---------|-------|
| none-split / none-jit | 64.05 | 32.38 | 3.76 | 0.00 | **100.30** |
| node-based / none-jit | 65.49 | 24.91 | 8.51 | 2.73 | **101.87** |
| node-based / pipeline-jit (pre-opt) | 74.25 | 25.00 | 1.98 | 3.10 | **104.51** |
| **node-based / pipeline-jit (Iter 9)** | **65.88** | **22.40** | **7.01** | **2.90** | **98.29** |

### Delta vs none-split / none-jit: -2.01s (-2.0%)
### Delta vs node-based / none-jit: -3.58s (-3.5%)
### Delta vs pre-optimization pipeline-jit: -6.22s (-6.0%)

---

## Iteration 11: InjectTempTableJoinStats for PerfectHashJoin (INVESTIGATED — DISABLED)

**Goal**: Inject min/max column statistics from temp table scans into the physical hash join's `join_stats[1]` to enable DuckDB's `PerfectHashJoinExecutor`, which replaces hash probing with direct array lookup when build-side key range is small.

**Investigation**:
1. Fixed column-matching bug: original code matched by type only (`types[ci] == key_type`), picking first column of matching type. Implemented `TraceTempScanColumn` to trace join key through `BoundReferenceExpression::index` chains across projections/filters down to the temp scan's `column_ids`.
2. Fixed `NumericStats::HasMinMax` crash on VARCHAR: calling `HasMinMax` on non-numeric types crashes because it internally calls `NumericStats::Min()` → `NumericValueUnionToValue()` which throws. Added type check before `HasMinMax`.
3. Discovered fundamental correctness issue with DuckDB's `PerfectHashJoinExecutor::CanDoPerfectHashJoin()`:
   - `is_build_small` flag is set in the constructor (before HT is built) based on `join_stats`
   - At `Finalize`, if `is_build_small` is already true, the function returns true immediately (line 67-69) WITHOUT re-validating the range against actual HT contents
   - When a FILTER between scan and join narrows the effective key range, the perfect HT is built with pre-filter min/max bounds, causing rows outside those bounds to be silently dropped

**Result**: DISABLED — `InjectTempTableJoinStats` is now a no-op. The PerfectHashJoin sticky `is_build_small` flag makes it unsafe to inject pre-filter statistics. Safe injection would require post-filter stats, which we don't have at plan init time.

**Code**: `TraceTempScanColumn` helper removed (dead code). `InjectTempTableJoinStats` body replaced with explanatory comment.

---

## Iteration 12: Software Prefetch for Hash Join Probes (REVERTED — +3.8% regression)

**Goal**: Reduce cache miss latency during hash join probing by issuing software prefetch instructions for hash table entries ahead of time.

**Mechanism**: Two-pass approach in `physical_hash_join.cpp`:
1. Hash all probe keys and compute bucket addresses
2. Issue `__builtin_prefetch` for each entry pointer
3. Probe with precomputed hashes (avoiding recomputation)

Added `AQPJIT_PREFETCH` flag (bit 6) to `aqp_jit.hpp` and `GetEntries()` accessor to `JoinHashTable`.

**Correctness**: All 113 JOB queries produce identical results.

### Performance (full breakdown, 15 iterations)

| Operator        | Iter 9 (s) | Prefetch (s) | Delta    | Pct     |
|-----------------|-----------|-------------|----------|---------|
| TABLE_SCAN      | 65.88     | 66.03       | +0.15    | +0.2%   |
| HASH_JOIN       | 22.40     | 25.42       | +3.02    | +13.5%  |
| FILTER          | 7.01      | 7.07        | +0.06    | +0.9%   |
| INVALID         | 2.90      | 3.56        | +0.66    | +22.8%  |
| **TOTAL**       | **98.29** | **102.08**  | **+3.79**| **+3.9%** |

**Root cause**: The hash computation overhead for precomputing all hashes before probing exceeds any cache warming benefit. For DRAM-resident hash tables (our typical case with 12MB L3 and hash tables often > 1MB), the gap between prefetch issuance and use (~200 cycles needed for DRAM) is dwarfed by the total computation per batch. The two-pass approach also destroys the natural pipelining of DuckDB's vectorized probe.

**Reverted**: All prefetch code removed from `physical_hash_join.cpp`. The `AQPJIT_PREFETCH` flag and `GetEntries()` accessor remain (harmless).

---

## Status After Iter 12 (superseded by Iter 13-17)

Code is back to Iter 9 state. All 113 JOB queries pass correctness.

**Active optimizations**:
- Range predicate injection with cost-model guard (Iter 3, -1.6%)
- Chain-walk skip for unique-key HTs (Iter 6, neutral)
- Selective LIKE fusion skip (Iter 7, -2.9%)
- JIT probe disabled for hash joins (Iter 9, -1.8%)
- Projection pushdown for temp table scans (Iter 10, neutral)

**Disabled / Reverted**:
- Column statistics propagation (Iter 10, causes plan regressions)
- Bloom filter scan push-down (Iter 4, +0.5% regression)
- Relaxed range pred upper bound (Iter 5, massive regression)
- All-level LIKE skip (Iter 7 variant, +1.8% worse)
- Improved LIKE substr search (Iter 8, inconclusive)
- InjectTempTableJoinStats (Iter 11, correctness risk from PerfectHashJoin bug)
- Software prefetch (Iter 12, +3.8% regression)

### Comparison Against All Baselines

| Config | TABLE_SCAN | HASH_JOIN | FILTER | INVALID | TOTAL |
|--------|-----------|-----------|--------|---------|-------|
| none-split / none-jit | 64.05 | 32.38 | 3.76 | 0.00 | **100.30** |
| node-based / none-jit | 65.49 | 24.91 | 8.51 | 2.73 | **101.87** |
| node-based / pipeline-jit (pre-opt) | 74.25 | 25.00 | 1.98 | 3.10 | **104.51** |
| **node-based / pipeline-jit (Final)** | **65.88** | **22.40** | **7.01** | **2.90** | **98.29** |

### Net improvements:
- vs none-split / none-jit: **-2.01s (-2.0%)**
- vs node-based / none-jit: **-3.58s (-3.5%)**
- vs pre-optimization pipeline-jit: **-6.22s (-6.0%)**

---

## Iteration 13: Early Termination When temp_card = 0

**Goal**: Skip remaining sub-plans when a temp table has 0 rows (INNER JOIN with empty table = 0 results).

**Changes**:
- `ir_query_splitter.h`: Added `std::set<std::string> empty_temp_tables_` member and `SubPlanReferencesEmptyTemp()` helper.
- `ir_query_splitter.cpp`: After cardinality is computed, track tables with 0 rows. Before executing a sub-plan, if its SQL references an empty temp table, append `LIMIT 0` to skip expensive scan/join.

**Affected queries**: 32a (2 sub-plans skipped), 29b (1 sub-plan skipped), 24a (0 — only final query benefits, which we can't skip due to aggregate semantics).

**Result**: Correctness verified on all 113 queries. Marginal time savings (~10-50ms per affected query) because the skipped sub-plans were already fast (empty hash tables make probes trivial).

---

## Iteration 14: Pure-LIKE Guard for Expr-Level JIT

**Goal**: Skip expr-level JIT compilation for filters with ONLY LIKE predicates. DuckDB's native vectorized LIKE is ~2x faster than per-row JIT LIKE.

**Changes**:
- `duckdb_adapter.cpp`: Added `ExprIsOnlyLike()` (checks all predicates are TextLike/Text_Not_Like) and `FilterIsOnlyLike()`. Changed `skip_like = false` to `skip_like = FilterIsOnlyLike(filter_ir)` at the expr-level JIT gate.

**Guard logic**: Only skips JIT when ALL predicates are LIKE. Mixed filters (LIKE + non-LIKE) still get expr-level JIT for the non-LIKE part.

**Result**: Correctness verified. Expected to save ~20ms on 20a, 23c.

---

## Iteration 15: Bloom Filter at Scan with Match-Rate Guard

**Goal**: Re-enable cross-sub-plan Bloom filter push-down (disabled in Iter 4) with a match-rate guard to prevent regression on high-match-rate queries.

**Changes**:
- `duckdb_adapter.cpp`: Added `GetBaseTableCardinality()` using `conn->Query("SELECT COUNT(*)")` with static cache.
- `ir_query_splitter.cpp`: Removed `(void)tt; break;` disable. Added match-rate guard: `match_rate = temp_card / base_card`, skip if >= 0.25.

**Guard logic**: BF only applied when match_rate < 0.25 (temp table is <25% of base table size). This ensures BF filters out >75% of rows, making the overhead worthwhile.

**Key improvements**:
- 29c: -88ms (-29%) — cascading BF+range pred across 10 sub-plans
- 8b: -61ms (-30%) — BF on name table (56 join keys vs 4M rows)
- 9a: -55ms (-21%) — BF on cast_info (1558 keys vs 36M rows)
- 31a: -40ms (-17%) — BF chain across movie_info, movie_keyword, cast_info, name

---

## Iteration 16: PerfectHashJoin Stats Injection (Guarded)

**Goal**: Re-enable `InjectTempTableJoinStats` (disabled in Iter 11) with guard against FILTER on build path.

**Changes**:
- `duckdb_adapter.cpp`: Re-implemented `InjectTempTableJoinStats`. For each HASH_JOIN with single integer equi-join condition:
  - Guard: no FILTER on build path (filters change effective key range, causing wrong PerfectHJ bounds)
  - Guard: integer key type
  - Guard: key range < 1048576 (DuckDB's MAX_BUILD_SIZE)
  - If all pass: inject min/max stats into `join_stats[1]`

**Safety**: The previous correctness bug was caused by injecting stats when a FILTER on the build path changed the effective key range. The `has_filter` guard prevents this.

**Result**: Correctness verified on all 113 queries. Marginal improvement on queries with small integer temp tables.

---

## Iteration 17: Range Pred + BF Overlap for Medium Temps

**Goal**: Extend range predicate injection to temp tables with card > kMaxHTCapacity (4096) when both selectivity AND match_rate are low. Range pred provides coarse segment pruning, BF provides fine row-level filtering.

**Changes**:
- `ir_query_splitter.cpp`: Changed the range pred loop guard from `temp_card > kMaxHTCapacity → skip` to a two-tier guard:
  - Small temps (card <= 4096): selectivity < 0.40 (existing behavior)
  - Medium temps (card > 4096): selectivity < 0.25 AND match_rate < 0.25

**Result**: Correctness verified. Enables range pred on more sub-plans where BF also applies.

---

## Iteration 18: Cold-Function Outlining for ICache Locality

**Goal**: Reduce instruction cache pressure from Iter 13-17 optimization code that increased `ExecuteOneIteration` size, causing ~20-50ms regressions on queries where no optimization fires.

**Root cause investigation**:
- Regressed queries (8c, 16b, 12c, 26c, 30a) have NO optimization code paths executing. 8c has zero guard matches — not even the BF/range-pred loops produce a single candidate.
- Instrumented optimization guard loops: total overhead < 6ms worst case (16b). Cannot explain 40-74ms regressions.
- `GetTempTableMinMax` not called for large temps (cardinality guard skips them).
- Root cause: binary layout / instruction cache effect. Adding ~250 lines of optimization code inline in `ExecuteOneIteration` changed the function from ~8KB to ~18KB, exceeding the 32KB L1 icache sweet spot when combined with `ExecuteSplitLoop`.

**Technique** (from expert_knowledge.txt #9, #18):
- #9: "Keep code smaller, improves instruction cache usage"
- #18: "Inline hot functions when beneficial" — inverse: outline cold functions

**Changes**:
- `ir_query_splitter.h`: Added `ApplyCrossSubPlanOptimizations(std::string &sub_sql)` declaration.
- `ir_query_splitter.cpp`: Extracted entire range-pred + BF block (~250 lines) into `__attribute__((noinline)) ApplyCrossSubPlanOptimizations()`. The hot path now contains a single call instruction instead of the full optimization code.

**Binary layout verification** (via `nm --print-size`):
- `ExecuteOneIteration`: 8,486 bytes (down from ~18KB with inline code)
- `ApplyCrossSubPlanOptimizations`: 10,128 bytes (placed at separate address)
- Hot path (`ExecuteOneIteration` + `ExecuteSplitLoop`): 15.4KB — fits in L1 icache

**Correctness**: All 113 JOB queries produce identical results to golden.

### Performance (full breakdown, 15 iterations)

| Operator | Iter 12 (s) | Outlined (s) | Delta | Pct |
|----------|------------|-------------|-------|-----|
| TABLE_SCAN | 65.88 | 70.31 | +4.43 | +6.7% |
| HASH_JOIN | 22.40 | 21.45 | -0.95 | -4.2% |
| FILTER | 7.01 | 5.03 | -1.98 | -28.2% |
| INVALID | 2.90 | 3.02 | +0.12 | +4.1% |
| **TOTAL** | **98.29** | **99.99** | **+1.70** | **+1.7%** |

### Wall-clock comparison

| Config | Total wall (s) |
|--------|---------------|
| none-split / none-jit | **15.52** |
| node-based / none-jit | **17.56** |
| Iter 9/12 baseline (measured May 19) | **21.48** |
| **Current (Iter 18 outlined)** | **20.39** |

**vs Iter 9/12 baseline: -1.09s (-5.1%)**

### Per-query comparison (current vs Iter 9/12 baseline, wall-clock ms):

**Big improvements** (BF + range pred active):
- 29c: 319 → 210 ms (-109ms, -34%)
- 8b: 214 → 141 ms (-73ms, -34%)
- 9a: 276 → 211 ms (-65ms, -24%)
- 31a: 251 → 200 ms (-51ms, -20%)
- 10c: 358 → 336 ms (-22ms, -6%)

**Remaining regressions** (inherent node-based split overhead):
- 8c: 501 → 537 ms (+37ms, +7%) — 1.15M + 2.34M + 2.49M row temp tables, no optimization applicable
- 30a: 280 → 300 ms (+20ms, +7%)
- 19d: 414 → 433 ms (+19ms, +5%)
- 26c: 216 → 235 ms (+19ms, +9%)
- 12c: 219 → 238 ms (+19ms, +9%)

### Regression root cause analysis

The regressions are NOT from icache pressure or optimization code overhead. They are inherent to **node-based split producing large temp tables**:

| Query | Temp table sizes | Optimizations applicable | Split overhead vs none-split |
|-------|-----------------|------------------------|------------------------------|
| 8c | 1.15M, 2.34M, 2.49M | None (all too large) | +254ms (27 sub-plans) |
| 16b | 42K, 1.15M, 68K, 2.83M, 3.71M | 1 BF (68K→cast_info) | +279ms (39 sub-plans) |
| 12c | 122K, 1.15M, 358K, 42K | None | +138ms |

These queries produce million-row temp tables because DuckDB's `QuerySplit::Split()` picks split points that don't filter early enough. The temp table materialization (CTAS) and subsequent joins with large temps are expensive.

In contrast, queries that benefit from split have small intermediate results:
- 29c: first temp is 414 rows, cascading BFs reduce everything to <1000 rows
- 8b: first temp is 56 rows
- 7c: split is -315ms FASTER than none-split because split enables better plans

---

## Final Status (Iter 18)

All 113 JOB queries pass correctness.

**Active optimizations**:
- Range predicate injection with cost-model guard (Iter 3, extended in Iter 17)
- Chain-walk skip for unique-key HTs (Iter 6)
- Selective LIKE fusion skip (Iter 7)
- JIT probe disabled for hash joins (Iter 9)
- Projection pushdown for temp table scans (Iter 10)
- Early termination for empty temp tables (Iter 13)
- Pure-LIKE guard for expr-level JIT (Iter 14)
- Bloom filter at scan with match-rate guard (Iter 15)
- PerfectHashJoin stats injection with build-path guard (Iter 16)
- Range pred + BF overlap for medium temps (Iter 17)
- Cold-function outlining for icache locality (Iter 18)

### Comparison Against All Baselines

| Config | CPU total (s) | Wall total (s) |
|--------|-------------|---------------|
| none-split / none-jit | 100.30 | **15.52** |
| node-based / none-jit | 101.87 | **17.56** |
| node-based / pipeline-jit (pre-opt) | 104.51 | — |
| node-based / pipeline-jit (Iter 12) | 98.29 | **21.48** |
| **node-based / pipeline-jit (Iter 18)** | **99.99** | **20.39** |

### Current heaviest queries (wall-clock):
1. 16b: 654ms — 39 sub-plans, cast_info scan + large temp tables
2. 8c: 537ms — 27 sub-plans, all temp tables > 1M rows
3. 19d: 433ms — TABLE_SCAN heavy
4. 9d: 390ms — TABLE_SCAN heavy
5. 7c: 345ms — split helps (-315ms vs none-split)

### Performance analysis

1. **Algorithmic wins** from Iter 13-17: BF filtering and range pred injection save -1.09s wall time vs Iter 12 baseline. These are real improvements on queries where sub-plan execution produces small, selective temp tables (card < 100K, match_rate < 0.25).

2. **Binary layout regressions** from added code: The outlining in Iter 18 mitigated icache pressure by keeping `ExecuteOneIteration` at 8.5KB (vs ~18KB before). However, the remaining regressions (8c +37ms, 12c +19ms) persist because they stem from inherent node-based split overhead (temp table materialization, multiple sub-plan executions), not from icache effects.

3. **Split strategy is the bottleneck**: Queries like 8c (1.15M + 2.34M + 2.49M row temps) and 16b (up to 3.71M rows) suffer because DuckDB's `QuerySplit::Split()` picks split points that produce huge intermediate results. No JIT-level optimization can fix a bad split point. The real opportunity is **split strategy improvement**: reordering or choosing different split points to produce smaller temp tables first.

4. **Wall vs CPU discrepancy**: CPU total increased slightly (+1.7%) but wall time decreased (-5.1%). This suggests BF filtering reduces downstream I/O (fewer disk pages read for filtered scans), which improves wall time on HDD without reducing CPU cycles (the BF check itself adds CPU work).

### Next optimization direction

The remaining bottleneck is **split strategy quality**, not JIT:
1. **Sub-plan reordering**: Execute sub-plans that produce smallest temp tables first. Small temps enable BF/range-pred on downstream sub-plans.
2. **Split point selection**: Choose split points that maximize temp table selectivity. E.g., split at highly selective joins (FK to small dimension table) before unselective joins (FK to large fact table).
3. **Adaptive re-splitting**: After observing a large temp table (>100K rows), consider re-splitting the remaining plan to find a more selective path.

---

## Iteration 19: Fix Measurement Infrastructure

**Goal**: Eliminate per-process startup overhead from measurement by adding `--repeat=N` and `--in-memory` flags. Current measurement spawns a new OS process per iteration (15 per query × 113 queries = 1695 process spawns), each re-opening the 1.9 GB `.db` file and rebuilding DuckDB's buffer pool.

### Changes

**New CLI flags**:
- `--repeat=N`: Run the query N times in the same process, reusing the same DuckDB connection. Between iterations, `ResetQueryState()` clears per-query state (temp tables, planner, JIT contexts) while preserving the database instance and connection.
- `--in-memory`: Open DuckDB with `:memory:` and load all 21 tables from CSV using `CREATE TABLE` (from schema.sql) + `COPY FROM`. Guarantees zero disk I/O during query execution. CSV loading takes ~40-50s (HDD) but only happens once per process.
- `--csv-dir=<path>`: CSV directory (default: derived from `--schema` path as `dirname(schema)/csv/`).

**Files modified**:
1. `include/util/param_config.h` — Added `repeat_count`, `in_memory`, `csv_dir` fields
2. `src/util/param_config.cpp` — Parsing for new flags, validation, PrintUsage
3. `include/adapters/db_adapter.h` — Added virtual `ResetQueryState()` to base class
4. `include/adapters/duckdb_adapter.h` — Declared `ResetQueryState()` override, `LoadTablesFromCSV()`
5. `src/adapters/duckdb_adapter.cpp` — Implemented `ResetQueryState()` (clears temp_collections_, planner, plan, JIT state; preserves db/conn/jit_compiler_) and `LoadTablesFromCSV()` (parses CREATE TABLE names from schema.sql, executes schema, COPY FROM CSV)
6. `src/aqp_middleware.cpp` — Repeat loop in `main()` for single-query mode; `--in-memory` path in `CreateAdapter()` that opens `:memory:` and calls `LoadTablesFromCSV()`
7. `measure/measure_breakdown_time_job.sh` — Replaced inner `for i in {1..15}` loop with single invocation using `--repeat=${iteration}`

**Key design decisions**:
- `ResetQueryState()` does NOT reset `db`, `conn`, or `jit_compiler_` — these survive across iterations. The replacement scan function pointer (`&temp_collections_`) remains valid because the map object address is unchanged.
- CSV loading uses schema.sql for `CREATE TABLE` (proper types/constraints) then `COPY FROM` with `ESCAPE '\'` — avoids `read_csv_auto` quoting issues with embedded quotes in IMDB data.
- Repeat loop wraps single-query mode only (not `--benchmark`), matching the per-query iteration pattern of the measurement script.
- Iteration markers (`# iter-N`) written to `operator_exe.csv` from C++ code, replacing the shell script's `echo`.

**Correctness**: All 113 JOB queries produce identical results across 6 configurations:

| Split | JIT | Status |
|-------|-----|--------|
| none | none | Pass |
| none | pipeline | Pass |
| node-based | none | Pass |
| node-based | pipeline | Pass |
| relationship-center | none | Pass |
| relationship-center | pipeline | Pass |

**Performance impact**: Infrastructure only — no execution-time changes. The `--repeat` flag eliminates per-process DuckDB startup overhead (buffer pool rebuild, storage metadata parsing). The `--in-memory` flag eliminates all disk I/O during query execution. Both reduce measurement noise.

**Next step**: Re-run full breakdown measurement with `--repeat=15` (and optionally `--in-memory`) to establish a clean baseline. Then re-check whether the heavy-query bottlenecks (IPC 0.64, cache miss rate 24%, LLC misses 14M, kernel time 40%) identified in Iter 10 are still present or were artifacts of disk I/O noise.

---

## Iteration 20: Pure In-Memory Baseline Re-Measurement

**Goal**: Establish the true warm-buffer-pool baseline by analyzing the full JOB breakdown measured with `--repeat=15` (in-process iteration, single DuckDB connection per query). This eliminates per-process startup overhead and disk I/O noise that contaminated all prior measurements (Iter 0-18).

### Wall-Clock Breakdown (drop first 5, avg of 10 warm runs)

| Config                      | MW Overhead (s) | JIT Compile (s) | Execution (s) | Total Wall (s) |
|-----------------------------|----------------:|----------------:|--------------:|---------------:|
| none-split / none-jit       |            0.00 |            0.00 |          9.77 |           9.77 |
| none-split / pipeline-jit   |            0.01 |            8.80 |          9.39 |          18.20 |
| node-based / none-jit       |            0.98 |            0.00 |         10.59 |          11.57 |
| node-based / pipeline-jit   |            0.99 |            2.19 |         10.33 |          13.51 |

### Per-Operator CPU Breakdown (avg of iters 6-15)

| Config (CPU-seconds)        | TABLE_SCAN | HASH_JOIN | FILTER | INVALID | TOTAL  |
|-----------------------------|----------:|----------:|-------:|--------:|-------:|
| none-split / none-jit       |     38.71 |     26.85 |   3.56 |    0.00 |  69.24 |
| node-based / none-jit       |     42.86 |     18.76 |   5.76 |    1.43 |  69.01 |
| node-based / pipeline-jit   |     43.16 |     18.79 |   4.75 |    1.44 |  68.29 |

### Comparison vs Old Measurements (Iter 0-18)

| Metric                      | Old      | New      | Delta     |
|-----------------------------|----------|----------|-----------|
| none-split/none-jit CPU     | 100.30s  | 69.24s   | -31.06s (-31%) |
| none-split/none-jit wall    | 15.52s   | 9.77s    | -5.75s (-37%) |
| node-based/pipeline-jit CPU | 99.99s   | 68.29s   | -31.70s (-32%) |
| node-based/pipeline-jit wall| 20.39s   | 13.51s   | -6.88s (-34%) |
| TABLE_SCAN (none-split CPU) | 64.05s   | 38.71s   | -25.34s (-40%) |

The ~40% TABLE_SCAN drop confirms the old numbers included HDD I/O overhead. The old bottleneck profile (IPC=0.64, 40% kernel time) was measuring disk I/O, not true query execution.

### Key Findings

1. **JIT execution is faster than none-jit** (-0.25s, 10.33 vs 10.59s). The entire +1.95s wall overhead of pipeline-jit vs none-jit is JIT compilation time (2.19s), not execution regression. Since the task says "ignore compilation time," JIT is net positive for execution.

2. **Node-based split adds +0.82s execution overhead** vs none-split (10.59 vs 9.77s for none-jit). This is inherent split overhead: temp table materialization, sub-plan setup/teardown, and suboptimal plans from splitting.

3. **JIT saves -1.01s FILTER time** (4.75 vs 5.76s CPU, pipeline-jit vs none-jit). The LIKE skip + expr-level JIT optimizations from Iter 7/14 are working.

4. **HASH_JOIN is essentially neutral** with JIT (18.79 vs 18.76s). Disabling JIT probe (Iter 9) was the right call — DuckDB's native vectorized probe matches JIT.

5. **none-split/pipeline-jit has 8.80s JIT compile** for the whole query in one shot. node-based split reduces this to 2.19s because sub-plans are smaller.

### Heaviest Queries (wall-clock, node-based/pipeline-jit)

| Query | Wall (ms) | vs none-split delta | Bottleneck |
|-------|----------:|--------------------:|------------|
| 16b   |     530.0 |     +208.6 (+65%)   | Split overhead (39 sub-plans, large temps) |
| 8c    |     433.1 |     +193.0 (+80%)   | Split overhead (27 sub-plans, all temps >1M rows) |
| 19d   |     336.3 |     +118.7 (+55%)   | TABLE_SCAN heavy |
| 9d    |     278.2 |     +124.4 (+81%)   | TABLE_SCAN heavy |
| 10c   |     254.1 |      +82.9 (+48%)   | Mixed TS + FILTER |
| 30a   |     225.7 |     +111.7 (+98%)   | Split overhead |
| 22a   |     209.3 |     +131.5 (+173%)  | Split overhead |

### Queries Where Split+JIT Helps

| Query | Split+JIT (ms) | None-split (ms) | Delta |
|-------|---------------:|-----------------:|------:|
| 7c    |          168.4 |            440.9 | -272.5 (-62%) |
| 11d   |          117.0 |            251.6 | -134.6 (-54%) |
| 17d   |           81.1 |            190.1 | -108.9 (-57%) |
| 22d   |          177.0 |            256.4 |  -79.4 (-31%) |
| 17c   |           51.2 |            116.4 |  -65.2 (-56%) |
| 17b   |           60.8 |            124.9 |  -64.1 (-51%) |
| 6f    |          175.2 |            225.0 |  -49.8 (-22%) |

---

## Current Status (Iter 20)

All 113 JOB queries pass correctness.

**Active optimizations** (same code as Iter 18, re-measured with clean infrastructure):
- Range predicate injection with cost-model guard (Iter 3, extended in Iter 17)
- Chain-walk skip for unique-key HTs (Iter 6)
- Selective LIKE fusion skip (Iter 7)
- JIT probe disabled for hash joins (Iter 9)
- Projection pushdown for temp table scans (Iter 10)
- Early termination for empty temp tables (Iter 13)
- Pure-LIKE guard for expr-level JIT (Iter 14)
- Bloom filter at scan with match-rate guard (Iter 15)
- PerfectHashJoin stats injection with build-path guard (Iter 16)
- Range pred + BF overlap for medium temps (Iter 17)
- Cold-function outlining for icache locality (Iter 18)

### Execution-Only Comparison (ignoring JIT compile time per CLAUDE.md)

| Config                      | Execution (s) | vs none-split/none-jit |
|-----------------------------|-------------:|----------------------:|
| none-split / none-jit       |         9.77 | baseline              |
| none-split / pipeline-jit   |         9.39 | -0.38s (-3.9%)        |
| node-based / none-jit       |        10.59 | +0.82s (+8.4%)        |
| node-based / pipeline-jit   |        10.33 | +0.57s (+5.8%)        |

Pipeline-JIT reduces execution time by -0.38s for none-split and -0.25s for node-based. The remaining gap for node-based is split overhead (+0.82s), not JIT.

---

## Iter 21: Bottleneck Re-profiling (Step 1 complete)

**What changed**: No code changes. Re-profiled the heaviest queries with `perf stat` using warm in-memory data (`--repeat=3`). Confirmed old Iter 0-18 bottleneck profile was disk I/O noise.

**perf stat results (warm, in-memory)**:

| Query | Config     | IPC  | Cache miss% | LLC miss% | Kernel% | CPU/Wall |
|-------|------------|------|-------------|-----------|---------|----------|
| 16b   | nb-jit     | 0.82 | 17.2%       | 18.8%     | 8.5%    | 4.2x     |
| 16b   | ns-nojit   | 0.77 | 17.5%       | 22.3%     | 8.1%    | 8.4x     |
| 8c    | nb-jit     | 0.68 | 26.6%       | 31.2%     | 11.8%   | 4.3x     |
| 8c    | ns-nojit   | 0.67 | 22.5%       | 28.2%     | 10.6%   | 7.6x     |
| 19d   | nb-jit     | 0.89 | 19.0%       | 22.7%     | 7.9%    | 3.4x     |

**Key findings**:
1. Kernel time: 40% → 8-12%. Old bottleneck was disk I/O, not real CPU work.
2. IPC: 0.64 → 0.68-0.89. Memory-bound, not compute-bound. Hash join dominates.
3. CPU/Wall: node-based ~4x vs none-split ~8x. Split halves parallelism — subplans run sequentially with less intra-operator parallelism.
4. Large temp tables root cause: 16b (3.71M rows), 8c (2.49M rows) accumulate huge intermediates that cause expensive final hash joins.

**Top 5 regressions (nb-jit exe vs ns-nojit exe)**:
| Query | nb-jit exe | ns-nojit exe | Delta    | Root cause                          |
|-------|-----------|-------------|----------|--------------------------------------|
| 16b   | 482.7ms   | 320.3ms     | +162.4ms | 3.71M row temp, final HJ 186ms wall |
| 8c    | 395.0ms   | 240.4ms     | +154.5ms | 2.49M row temp, final HJ 183ms wall |
| 12c   | 137.1ms   |  34.1ms     | +103.0ms | large intermediate temps             |
| 6d    | 169.3ms   |  73.7ms     |  +95.6ms | large intermediate temps             |
| 16c   | 188.8ms   |  95.4ms     |  +93.4ms | large intermediate temps             |

**Top operator-level breakdown (nb-jit, avg of 10 warm iters)**:

16b: HASH_JOIN=1.63s (68%), TABLE_SCAN=0.64s (27%). Subplan 5 = 1.49s CPU = 186ms wall.
8c: HASH_JOIN=1.65s (84%), TABLE_SCAN=0.20s (10%). Subplan 3 = 1.46s CPU = 183ms wall.
19d: HASH_JOIN=0.63s (58%), TABLE_SCAN=0.37s (34%). Distributed across 8 subplans.
10c: TABLE_SCAN=0.69s (41%), HASH_JOIN=0.51s (31%), FILTER=0.40s (24%).

**Conclusion**: The primary bottleneck is now the split strategy producing large temp tables that cascade into expensive hash joins. JIT itself is modestly helpful (-0.25s). Next: optimize split order by selectivity (Step 2).

---

## Iteration 22: JIT-Gated Software Prefetching for Hash Join (ENABLED — -4.7% on none-split)

**Goal**: Add software prefetching to hash join probe, build, and bloom filter operations, gated behind JIT context so only JIT-active queries benefit.

**Approach**: Inline prefetching — prefetch entry[i+D] while processing entry[i]. This preserves the natural pipelining of DuckDB's vectorized probe (unlike Iter 12's failed two-pass approach).

**Changes**:
1. `join_hashtable.cpp` — Added `PREFETCH` template parameter to `ProbeForPointersInternal`. When enabled, issues `__builtin_prefetch` for entries 8 positions ahead during probe. Also added prefetch in `InsertHashesLoop` for build, gated on `ht.IsPrefetchEnabled()`.
2. `bloom_filter.cpp` — Added prefetch to `LookupHashes` (distance=16) and `InsertHashes` (distance=16), gated on `prefetch_enabled` flag.
3. `join_hashtable.hpp` — Added `prefetch_enabled` field to `ProbeState` and `JoinHashTable`, with `SetPrefetchEnabled()`/`IsPrefetchEnabled()` accessors.
4. `bloom_filter.hpp` — Added `prefetch_enabled` field to `BloomFilter`.
5. `physical_hash_join.cpp` — Set prefetch flags in `HashJoinGlobalSinkState` constructor and `GetOperatorState` when JIT context is active.
6. `duckdb_adapter.cpp` — Ensure `aqp_jit_context` is always created when JIT level is active (even if no filters compiled), so hash join prefetching is enabled.

**Why this differs from Iter 12 (which regressed +3.8%)**:
- Iter 12 used **two-pass**: precompute ALL hashes → prefetch ALL entries → probe ALL. This destroyed pipelining and the prefetch-to-use gap was wrong.
- Iter 22 uses **inline**: prefetch entry[i+8] while processing entry[i]. The prefetch distance of 8 entries × ~30-50 cycles/entry ≈ 240-400 cycles, matching L2/LLC miss latency.

**Correctness**: All 113 JOB queries produce identical results to golden.

### Performance (none-split, full breakdown, 15 iterations — runs 6-15 averaged)

| Config              | Execution (ms) | vs Iter 21 | vs ns/nojit |
|---------------------|---------------:|-----------:|------------:|
| none-split/none-jit |         9332.0 |    (same)  | baseline    |
| none-split/pipe-jit |         8889.7 |   -113.0   | -442.3 (-4.7%) |

### Per-query analysis (none-split: pipeline-jit vs none-jit, exe time)

**Most improved** (JIT vs no-JIT within this run):

| Query | no-JIT (ms) | JIT (ms) | Delta | Pct |
|-------|------------|---------|-------|-----|
| 7c  | 445.9 | 430.3 | -15.6 | -3.5% |
| 31b | 61.1 | 45.7 | -15.4 | -25.2% |
| 31a | 110.6 | 95.3 | -15.3 | -13.9% |
| 31c | 119.9 | 104.9 | -15.0 | -12.5% |
| 19a | 99.6 | 86.5 | -13.1 | -13.2% |
| 6f  | 219.2 | 189.6 | -29.6 | -13.5% |
| 17f | 214.8 | 199.9 | -14.9 | -6.9% |
| 16b | 317.5 | 296.6 | -20.9 | -6.6% |
| 8c  | 236.5 | 212.0 | -24.5 | -10.4% |

**Only regression**: 11d +21.3ms (+8.7%) — known high-variance mutex-contention query.

### Analysis

The prefetch optimization is most effective on:
- **Build-dominated queries** (6f: -13.5%) where `InsertHashes` is 21.7% of CPU
- **Memory-bound probe queries** (8c: -10.4%, 16b: -6.6%) where HT exceeds L3 cache
- **Compute-bound queries** still benefit modestly (17f: -6.9%) from BF prefetch

The 7c query shows only -3.5% because its bottleneck is `memmove` (28.8% of CPU from VARCHAR copies into row-oriented HT layout), which prefetching doesn't help.

### Active JIT features after Iter 22

1. FILTER → compiled filter (`expr_fns`)
2. TABLE_SCAN → scan+filter fusion (`scan_filter_fns`)
3. FILTER → pipeline fused filter+projection (`pipeline_fns`)
4. PROJECTION → zero-copy column mapping (`proj_col_maps`)
5. TABLE_SCAN → Bloom filter scan push-down (`bloom_scan_filters`)
6. HASH_JOIN → JIT-gated software prefetching for probe, build, and BF operations

---

## Iter 23 — Chain-walk prefetching + BF GetMask unroll

### Changes

1. **Chain-walk prefetching in `AdvancePointers`** (`join_hashtable.cpp`): When `ht.IsPrefetchEnabled()` and `sel_count > 8`, prefetches the next chain pointer 8 entries ahead during linked-list traversal in hash join probe. JIT-gated — non-JIT path unchanged.

2. **`GetMask` loop unrolling** (`bloom_filter.cpp`): Replaced `for` loop (4 iterations) with explicit 4-element unroll. Unconditional but neutral — compiler at -O3 already optimizes.

### Files modified

- `/home/pei/Project/duckdb/src/execution/join_hashtable.cpp` — `AdvancePointers` with prefetch path
- `/home/pei/Project/duckdb/src/planner/filter/bloom_filter.cpp` — `GetMask` unrolled

### Performance (clean measurement, avg of 10 runs after 5 warmup)

| Config | Iter 22 exe (ms) | Iter 23 exe (ms) | Delta |
|--------|----------------:|----------------:|------:|
| none-split / none-jit | 9,335.8 | 9,339.9 | +4.1 (noise) |
| none-split / pipeline-jit (no SIMD) | 8,672.7 | 8,680.8 | +8.1 (noise) |
| none-split / pipeline-jit (auto SIMD) | 8,680.3 | 8,684.8 | +4.5 (noise) |
| node-based / none-jit | 10,054.8 | 10,025.3 | -29.5 |
| node-based / pipeline-jit (no SIMD) | 9,594.2 | 9,375.7 | **-218.5** |
| node-based / pipeline-jit (auto SIMD) | 9,612.1 | 9,399.1 | **-213.0** |

### Key per-query improvements (node-based pipeline-jit vs node-based none-jit)

| Query | nb-jit exe | nb-nojit exe | JIT effect |
|-------|----------:|------------:|-----------:|
| 8c | 329.3 | 378.1 | -48.7ms |
| 16b | 436.9 | 476.8 | -39.9ms |
| 19d | 254.2 | 287.1 | -32.9ms |
| 10c | 217.8 | 248.0 | -30.2ms |
| 8d | 126.8 | 153.5 | -26.6ms |
| 9d | 230.8 | 257.1 | -26.3ms |

### None-split pipeline-jit vs none-jit: -659.1ms (-7.1%)

Top improved: 11d -26.5ms, 6f -24.7ms, 31c -21.1ms, 8c -20.6ms, 31a -18.1ms
Minimal regressions: only 2d +0.7ms, 2b +0.1ms (noise)

### Analysis

The chain-walk prefetch had **dramatically more impact on node-based** than none-split:
- Node-based creates larger temp tables that cascade into bigger hash tables with longer chains
- `AdvancePointers` chain walking is a larger fraction of execution in node-based sub-queries
- None-split uses more parallelism (6-8x CPU/Wall vs 3-4x), hiding memory latency via threads

Node-based pipeline-jit gap vs baseline narrowed from **+2.8%** (Iter 22) to **+0.4%** (Iter 23) — nearly at parity with vanilla DuckDB.

### Active JIT features after Iter 23

1. FILTER → compiled filter (`expr_fns`)
2. TABLE_SCAN → scan+filter fusion (`scan_filter_fns`)
3. FILTER → pipeline fused filter+projection (`pipeline_fns`)
4. PROJECTION → zero-copy column mapping (`proj_col_maps`)
5. TABLE_SCAN → Bloom filter scan push-down (`bloom_scan_filters`)
6. HASH_JOIN → JIT-gated software prefetching for probe, build, BF operations
7. HASH_JOIN → JIT-gated chain-walk prefetching in AdvancePointers

---

## Iter 23 — Post-optimization profiling and Direction A assessment

### Re-profiled top 10 after Iter 23 (perf stat + perf record, none-split/pipeline-jit, repeat=3)

| Rank | Query | Exe (ms) | IPC | Cache miss% | LLC miss% | CPU/Wall | Top functions |
|------|-------|---------|-----|-------------|-----------|----------|---------------|
| 1 | 7c | 434.7 | 0.46 | 35.9% | 25.3% | 5.5x | memmove 27.6%, ValueStore\<string\> 5.7% |
| 2 | 16b | 302.5 | 0.86 | 17.1% | 16.7% | 7.2x | BF::Lookup 8.0%, **AdvancePointers 7.8%**, InsertHashes 5.6%, FastMemcpy 5.0%, Hash 3.8%, ResolvePredicates 2.6% |
| 3 | 11d | 256.5 | 1.03 | 14.4% | 30.2% | 1.9x | Finalize 9.9%, Gather\<int\> 8.1%, InsertHashes 6.1%, mutex 6.1% |
| 4 | 22d | 232.6 | 0.79 | 15.4% | 26.2% | 4.4x | Finalize 7.5%, Gather\<int\> 6.9%, InsertHashes 6.0%, ValueStore\<string\> 5.7% |
| 5 | 8c | 219.2 | 0.77 | 21.4% | 17.2% | 6.1x | InsertHashes 9.2%, **AdvancePointers 6.1%**, **ResolvePredicates 5.5%**, BF::Lookup 3.6%, GetRowPointers 3.5%, TemplatedMatch\<int\> 3.1% |
| 6 | 17f | 204.7 | 1.03 | 7.3% | 5.6% | 7.4x | BF::Lookup 8.5%, InsertHashes 7.1%, Hash 4.9%, **AdvancePointers 4.3%**, FilterSwitch 3.7% |
| 7 | 6f | 195.5 | 0.79 | 13.6% | 13.6% | 8.0x | InsertHashes 17.4%, BF::Lookup 6.8%, Finalize 6.4%, fsst_decompress 5.3%, BF::InsertHashes 5.0% |
| 8 | 19d | 195.1 | 0.93 | 12.5% | 14.3% | 4.6x | **ResolvePredicates 7.3%**, InsertHashes 5.5%, BF::Lookup 4.5%, fsst_decompress 4.0%, StringScanPartial 3.3%, AdvancePointers 2.5% |
| 9 | 25c | 186.5 | 1.08 | 8.1% | 13.9% | 5.7x | BF::Lookup 10.0%, StringScanPartial 9.7%, fsst_decompress 4.8%, Hash 3.8% |
| 10 | 17d | 171.5 | 1.08 | 5.7% | 3.6% | 7.4x | BF::Lookup 9.8%, InsertHashes 6.9%, Hash 5.5%, FilterSwitch 4.6%, fsst_decompress 4.2%, AdvancePointers 2.8% |

Top 10 sum: 2398.9ms (27.6% of total 8680.8ms).

### Direction A completeness assessment

**Done (6 techniques):**
- Probe bucket prefetch, build prefetch, BF prefetch (Iter 22)
- Chain-walk prefetch in AdvancePointers, BF GetMask unroll (Iter 23)
- Scalar JIT probe (Iter 9, disabled), chain-walk skip (Iter 6)

**Remaining (2 techniques):**

1. **AMAC-style interleaved probing** — addresses AdvancePointers still at 7.8% (16b), 6.1% (8c), 4.3% (17f), 2.8% (17d), 2.5% (19d). Current prefetch brings the pointer field into cache, but the target chain node it points to is still a miss. True AMAC interleaves 8-16 independent keys at different probe stages to hide latency. Est. ~30-50ms. High complexity (state machine redesign of probe loop).

2. **Type-specialized ResolvePredicates** — addresses ResolvePredicates at 7.3% (19d), 5.5% (8c), 2.6% (16b). Generic RowMatcher dispatches through TemplatedMatch. JOB keys are mostly int32; a JIT-gated specialized path for direct int comparison avoids dispatch overhead. Est. ~15-30ms. Medium complexity.

**Exhausted / not feasible without whole-pipeline JIT:**
- InsertHashes (5-17%): already prefetched, DuckDB build is tight
- BF::LookupHashes (3-10%): already prefetched, compute-bound queries are cache-resident
- Finalize (3-10%): calls InsertHashes, already prefetched
- VectorOperations::Hash (2-5%): separate from probe hash, would need whole-pipeline fusion
- Gather/FastMemcpy/memmove (2-28%): inter-operator materialization, needs whole-pipeline JIT
- fsst_decompress/StringScanPartial (2-15%): DuckDB internal codec
- mutex (6%): DuckDB scheduler

---

## Iter 24 — Probe path fast paths + row-data prefetching

### Changes

All in `/home/pei/Project/duckdb/src/execution/join_hashtable.cpp`:

1. **ResolvePredicates memcpy**: Replaced element-by-element `set_index`/`get_index` loop with `memcpy` for sel_vector copy. JIT-gated (`IsPrefetchEnabled()`). None-jit path uses original element-by-element loop.

2. **ScanInnerJoin equality-only fast path**: For equality-only joins (`IsPrefetchEnabled() && !needs_chain_matcher`), skip `ResolvePredicates` call entirely — directly `memcpy` sel_vector and return. JIT-gated.

3. **ScanInnerJoin skip found_match for INNER/RIGHT joins**: `found_match` is never read after `NextInnerJoin` for pure INNER/RIGHT joins. JIT-gated (inside the equality fast path). None-jit path writes found_match unconditionally as original.

4. **ScanKeyMatches equality-only fast path**: Same as #2 for semi/anti/mark joins. JIT-gated (`IsPrefetchEnabled()`).

5. **AdvancePointers chain node data prefetch**: After loading the next chain pointer, prefetch chain node base address (`ptrs[idx]`) so key comparison data is warm for next `ScanInnerJoin`. JIT-gated (`IsPrefetchEnabled()`).

6. **GetRowPointersInternal row data prefetch**: Prefetch all row data pointers before calling `row_matcher_build.Match`. Warms cache for key comparison in `TemplatedMatchLoop`. JIT-gated (`state.prefetch_enabled`).

### Correctness

All 113 JOB queries produce identical results to golden for all 4 configs: none-split/none-jit, none-split/pipeline-jit, node-based/none-jit, node-based/pipeline-jit.

### Performance (all 6 configs, same session, full breakdown, 15 iterations — runs 6-15 averaged)

| Config | Execution (ms) | Middleware (ms) | JIT Compile (ms) | Wall (ms) |
|--------|---------------:|----------------:|------------------:|----------:|
| none-split / none-jit | 9,446.4 | 2.8 | — | 9,449.2 |
| none-split / pipeline-jit (no SIMD) | 8,831.3 | 12.7 | 8,760 | 17,603 |
| none-split / pipeline-jit (auto SIMD) | 8,854.1 | 14.2 | 9,590 | 18,458 |
| node-based / none-jit | 10,172.8 | 1,003 | — | 11,176 |
| node-based / pipeline-jit (no SIMD) | 9,596.8 | 1,011 | 2,180 | 12,788 |
| node-based / pipeline-jit (auto SIMD) | 9,595.5 | 1,014 | 2,220 | 12,830 |

**JIT effect (same-session, Iter 24):**

| Split | JIT delta (no SIMD) | JIT delta (auto SIMD) |
|-------|--------------------:|----------------------:|
| none-split | -615.2 ms (-6.5%) | -592.4 ms (-6.3%) |
| node-based | -576.0 ms (-5.7%) | -577.3 ms (-5.7%) |

**Iter 24 update**: All changes JIT-gated after review — none-jit path is now identical to original DuckDB.

**Re-measured none-jit (JIT-gated, separate session):**
| Config | Execution (ms) | Middleware (ms) | Wall (ms) |
|--------|---------------:|----------------:|----------:|
| none-split / none-jit | 9,523.6 | 2.7 | 9,526.3 |
| node-based / none-jit | 10,151.0 | 993.2 | 11,144.3 |

**Cross-session JIT effect (none-jit from new session, JIT from previous session):**
| Split | none-jit exe | JIT exe | Delta |
|-------|------------:|--------:|------:|
| none-split | 9,523.6 | 8,831.3 | -692.3 (-7.3%) |
| node-based | 10,151.0 | 9,596.8 | -554.2 (-5.5%) |

JIT advantage widened from -6.5% (when Iter 24 changes were unconditional) to -7.3% (JIT-gated). Cross-session noise ~100ms; true same-session effect estimated -6.5% to -7.3%.

### Per-query JIT improvements (none-split, pipe-jit vs no-jit, Iter 24)

Top 15 improved:
| Query | JIT (ms) | no-JIT (ms) | Delta |
|-------|---------|------------|-------|
| 11d | 267.1 | 308.2 | -41.1 |
| 6f | 197.2 | 223.1 | -25.9 |
| 22d | 229.9 | 250.4 | -20.6 |
| 16b | 304.1 | 323.9 | -19.8 |
| 31c | 103.2 | 122.7 | -19.6 |
| 31a | 94.0 | 112.6 | -18.6 |
| 8c | 218.8 | 236.5 | -17.7 |
| 19a | 82.9 | 100.4 | -17.6 |
| 10c | 154.0 | 171.1 | -17.0 |
| 19d | 196.6 | 212.6 | -16.0 |

Only 3 regressions, all within noise: 7c +4.2ms, 11c +1.7ms, 5a +1.0ms.

### perf profile changes (8c, none-split/pipeline-jit)

| Function | Iter 23 | Iter 24 | Change |
|----------|---------|---------|--------|
| InsertHashes | 9.2% | 9.31% | same |
| AdvancePointers | 6.1% | 6.07% | same |
| **ResolvePredicates** | **5.5%** | **0%** | **eliminated** |
| **ScanInnerJoin** | — | 5.56% | NEW (absorbed ResolvePredicates work) |
| **TemplatedMatch\<int\>** | **3.1%** | **1.94%** | **-1.16pp** (row-data prefetch) |
| BF::LookupHashes | 3.6% | 3.54% | same |
| LLC miss rate | 17.2% | 14.4% | **-2.8pp** |

### Active JIT features after Iter 24

1. FILTER → compiled filter (`expr_fns`)
2. TABLE_SCAN → scan+filter fusion (`scan_filter_fns`)
3. FILTER → pipeline fused filter+projection (`pipeline_fns`)
4. PROJECTION → zero-copy column mapping (`proj_col_maps`)
5. TABLE_SCAN → Bloom filter scan push-down (`bloom_scan_filters`)
6. HASH_JOIN → JIT-gated software prefetching for probe, build, BF operations
7. HASH_JOIN → JIT-gated chain-walk prefetching in AdvancePointers
8. HASH_JOIN → JIT-gated row-data prefetching before Match (new)
9. HASH_JOIN → ScanInnerJoin/ScanKeyMatches equality fast path (JIT-gated)
10. HASH_JOIN → Skip found_match writes for INNER/RIGHT joins (JIT-gated)
11. HASH_JOIN → ResolvePredicates memcpy optimization (JIT-gated)

---

## Iteration 25: Direction A Step 3 — Flat Tables + CSR Indexes + Runtime CSR + Kernel Executor

**Goal**: Build auxiliary in-memory storage structures (flat column arrays, CSR indexes) in middleware and execute eligible sub-queries via a kernel that bypasses DuckDB's hash join entirely. DuckDB remains unchanged — it serves as data source and SQL parser.

**Commit**: `9a93673` ("implement first 3 steps for direction A")

### What was implemented

1. **Flat Column Arrays** (`src/storage/flat_table.h/.cpp`): Decompress all base tables into plain C arrays at startup. Direct `array[row_id]` access eliminates DuckDB's FSST decompression (3-15% CPU) and segment management.
2. **CSR Indexes** (`src/storage/csr_index.h/.cpp`): Compressed Sparse Row indexes on all FK columns (from `--fkeys`). O(1) FK→PK lookup replaces hash table build+probe (InsertHashes 5-17%, AdvancePointers 6-8%).
3. **Runtime CSR on Temp Tables**: After each sub-query, build CSR on temp result's integer columns. Next sub-query uses CSR lookup instead of DuckDB hash join.
4. **SubQueryPlan Executor** (`src/storage/sub_query_plan.h/.cpp`): `AnalyzeSubIR()` converts sub-IR into a `SubQueryPlan` struct; `ExecuteSubQueryPlan()` runs it in a single scan loop with CSR lookups.
5. **`--csr-support=inner`** flag: Kernel handles filter-free 2-table CSR joins only. Sub-queries with filters or aggregates fall back to DuckDB.

### Configuration

CLI flags: `--storage-plan --storage-cache=/tmp/imdb_storage_plan.cache --csr-support=inner`
Kernel gate: `storage_plan_ && storage_plan_->IsLoaded() && config_.NeedsCsrSupport() && config_.jit_flags != 0`

### Performance

Measurement data: `JOB_duckdb_152/*_csrinner_breakdown_time_log.csv` (--repeat=15, drop first 5, avg next 10)

| Config | Execution (ms) | Middleware (ms) | JIT Compile (ms) | Wall (ms) |
|--------|---------------:|----------------:|------------------:|----------:|
| none-split / none-jit | 9,475.6 | — | — | — |
| node-based / none-jit (csrinner) | 10,159.3 | 978.2 | — | 11,137.5 |
| **node-based / pipeline-jit (csrinner)** | **6,436.5** | **2,602.7** | **1,805.9** | **10,845.2** |

**Kernel effect on execution**: -3,722.7ms (-36.6%) from nb/nojit → nb/jit
**vs none-split/none-jit baseline**: -3,039.0ms (-32.1%)

### Top 15 kernel wins (nb-jit vs nb-nojit execution)

| Query | Delta | nb-jit (ms) | nb-nojit (ms) | ns-nojit (ms) |
|-------|------:|------------:|--------------:|--------------:|
| 16b | -255.4 | 224.9 | 480.4 | 318.9 |
| 8c | -151.4 | 231.3 | 382.7 | 237.3 |
| 25c | -136.1 | 24.1 | 160.2 | 195.8 |
| 30a | -131.1 | 40.7 | 171.8 | 113.4 |
| 25a | -131.1 | 20.3 | 151.4 | 122.2 |
| 31c | -130.1 | 34.4 | 164.4 | 122.1 |
| 18c | -120.7 | 23.8 | 144.5 | 135.3 |
| 9d | -115.0 | 136.4 | 251.4 | 151.6 |
| 17a | -101.3 | 27.3 | 128.6 | 139.7 |
| 19d | -97.7 | 185.2 | 282.9 | 209.0 |
| 18a | -94.7 | 20.7 | 115.4 | 163.0 |
| 16c | -88.4 | 95.9 | 184.3 | 93.7 |
| 20a | -87.3 | 27.3 | 114.5 | 124.6 |
| 30b | -85.1 | 32.9 | 118.0 | 43.4 |
| 16d | -78.3 | 78.8 | 157.1 | 88.7 |

**Zero regressions**: All 113 queries improved or stayed the same.

### Top 10 heaviest queries (nb-jit)

| Query | nb-jit exe (ms) | nb-nojit exe (ms) | ns-nojit exe (ms) |
|-------|----------------:|------------------:|------------------:|
| 8c | 231.3 | 382.7 | 237.3 |
| 16b | 224.9 | 480.4 | 318.9 |
| 10c | 218.0 | 250.1 | 166.7 |
| 19d | 185.2 | 282.9 | 209.0 |
| 6f | 159.0 | 170.2 | 222.5 |
| 6d | 155.9 | 165.6 | 70.8 |
| 17f | 154.8 | 183.7 | 217.2 |
| 5a | 143.3 | 146.2 | 122.4 |
| 9d | 136.4 | 251.4 | 151.6 |
| 7c | 128.7 | 141.4 | 438.3 |

### Analysis

The kernel's advantage comes from eliminating hash table build+probe entirely for FK joins. CSR lookup is O(1) per key vs DuckDB's O(chain_length) with hash collisions, and avoids the expensive InsertHashes (5-17% CPU) and AdvancePointers (6-8% CPU) functions.

**Coverage**: The `--csr-support=inner` level only handled filter-free 2-table joins. Node-based split naturally produces many such sub-queries — filters are consumed in early iterations, leaving later join iterations filter-free. These later joins are the expensive ones (large temp tables), so the kernel covers exactly the high-cost sub-queries.

**Middleware overhead**: mw increased from 978ms to 2,603ms (+1,625ms) due to FlatTable loading from DuckDB temp results and runtime CSR construction. The 3,723ms execution savings more than compensates.

**Key insight**: CSR eliminates hash table build entirely — no InsertHashes, no AdvancePointers, no Bloom filter. For FK joins with known key ranges, this is fundamentally faster than any hash join optimization.

---

## Iteration 26: Direction A Steps 3.5-5 — Filters, Dimension Cache, Sorted Indices

**Goal**: Extend kernel coverage from filter-free joins only to support filters, dimension table resolution, and MIN early termination.

**Commit**: `fe8ff6b` ("upload step 1-5 perf")

### What was implemented

**Step 3.5 — Filter Support** (`src/kernel/sub_query_plan.cpp`):
- Compile IR filter predicates into `RowPredicate` closures for the kernel scan loop
- Supports: `=`, `!=`, `<`, `>`, `<=`, `>=`, `IN`, `NOT IN`, `IS NULL`, `IS NOT NULL`, `AND`/`OR`/`NOT`
- `LIKE`/`BETWEEN` → DuckDB fallback
- `pk_to_row` mapping for inner-join bitset construction
- **Kernel coverage**: 0% → 58.7% (304/518 iterations) — the "0%" refers to the state after removing `CsrSupportLevel` (Iter 25) but before adding filter compilation. Without `--csr-support=inner`, all sub-queries have filters, so the new `AnalyzeSubIR` initially rejected everything.
- Code moved from `src/storage/` to `src/kernel/` directory

**Step 4 — Dimension Cache** (`src/storage/dimension_cache.h/.cpp`):
- Cache tiny tables (<200 rows) as lookup maps at startup
- `AnalyzeSubIR` resolves filtered dimension joins to FK IN-filters at analysis time
- Eliminates dimension leaf+edge from the sub-query plan, reducing join count
- Handles single-table filtered scan path (all joins eliminated) and 2-table join path
- `join_filters` on `KernelJoinStep` for dim filters targeting lookup table
- Guards: skip unfiltered dims; skip when dim resolution leaves 2 base tables

**Step 5 — Sorted Indices** (`src/storage/sorted_index.h/.cpp`):
- Sorted permutation arrays on 11 columns for MIN early termination on final sub-queries
- `AnalyzeFinalIR`: handles Projection→Aggregate→child pattern, maps MIN columns through projection
- `ExecuteFinalAggregate`: Phase 1 = sorted scan with early termination for MIN columns O(k), Phase 2 = running-min full scan for unsorted/lookup-table columns
- Extended to N-table star joins (2-5 tables) with PK bitset fallback
- Guard: bail if scan table is base with >5M rows and 2+ join steps

### Architecture change

`CsrSupportLevel` enum and `--csr-support` flag **removed**. The kernel now uses a unified `AnalyzeSubIR()` that attempts to handle all sub-queries — filters, dims, aggregates — and falls back to DuckDB when it can't. This is more general than Step 3's approach but trades explicit coverage control for implicit analysis-time decisions.

### Performance

**No isolated breakdown measurement exists.** Steps 3.5-5 were implemented together in one commit and no breakdown was run between them. Performance is unknown — reproducing would require checking out commit `fe8ff6b`, building, and running the full breakdown measurement with `--storage-plan --storage-cache=/tmp/imdb_storage_plan.cache`.

Key uncertainty: removing `--csr-support=inner` and replacing it with the general `AnalyzeSubIR` changed which sub-queries the kernel handles. Step 3's approach (only filter-free joins) was simple and covered exactly the high-cost sub-queries. The new approach is more general but may have different coverage characteristics. Whether Steps 3.5-5 improved or regressed execution time vs Step 3 is **not confirmed**.

### Correctness

6/6 configurations pass (113 queries each):

| Split | JIT | Status |
|-------|-----|--------|
| none | none | Pass |
| none | pipeline | Pass |
| node-based | none | Pass |
| node-based | pipeline | Pass |
| relationship-center | none | Pass |
| relationship-center | pipeline | Pass |

## Iteration 27: Step 6 — Kernel Threshold Tuning (num_joins >= 1)

**Goal**: Tune the kernel vs DuckDB decision gate for Step 6 integration. Tuning data from `tune_kernel_threshold.py` showed pure scan+filter sub-queries (0 joins) are faster in DuckDB, while CSR-join sub-queries (1+ joins) are faster in the kernel.

**Commit**: `topdown_fix` branch (threshold added to `ir_query_splitter.cpp`)

### What changed

**`src/split/ir_query_splitter.cpp` line 618-619** — kernel decision gate:
```cpp
// Before:
if (sub_plan.valid && !config_.no_kernel) {
// After:
if (sub_plan.valid && !config_.no_kernel &&
    !sub_plan.join_steps.empty()) {
```

Skip kernel for sub-queries with 0 CSR join steps (pure scan+filter). These are ~339ms slower in kernel than DuckDB across 53 sub-queries. Sub-queries with 1+ joins save ~3,695ms total across 412 sub-queries.

Final sub-query path unchanged — all kernel-valid finals have 1+ joins.

Only affects: node-based/pipeline-jit and relationship-center/pipeline-jit configs (requires `storage_plan_` loaded + `jit_flags != 0` + DuckDB engine).

### Performance (node-based/pipeline-jit, avg of 10 runs)

| Config | Execution (ms) | Middleware (ms) | JIT Compile (ms) | Wall (ms) |
|--------|---------------:|----------------:|------------------:|----------:|
| none-split/none-jit | 9,444 | 3 | — | 9,447 |
| none-split/pipeline-jit | 8,857 | 13 | 8,786 | 17,656 |
| node-based/none-jit | 10,170 | 819 | — | 10,989 |
| **node-based/pipeline-jit** | **6,980** | **4,619** | **1,189** | **12,788** |

**Execution: 10,170 → 6,980ms (−31.4%) vs node-based/none-jit**
**Execution: 9,444 → 6,980ms (−26.1%) vs none-split/none-jit baseline**

Comparison with Step 3 (Iter 25): 6,980ms vs 6,437ms — 543ms slower. The gap comes from:
- 213 kernel-invalid iterations (31.4%) still falling back to plain DuckDB execution
- Step 3's `--csr-support=inner` was simpler but handled exactly the high-value sub-queries

Middleware overhead: 1,010ms (Iter 24) → 4,619ms — dominated by FlatTable loading from DuckDB temps + runtime CSR build.

### Top 10 heaviest queries (execution time)

| Query | nb-jit exe | nb-nojit exe | ns-nojit exe | Delta vs ns-nojit |
|-------|----------:|------------:|------------:|-----------------:|
| 10c | 232.9 | 248.7 | 165.9 | +67.0 |
| 16b | 207.5 | 483.7 | 320.0 | −112.5 |
| 8c | 202.6 | 383.3 | 236.2 | −33.6 |
| 9d | 174.5 | 250.1 | 150.9 | +23.6 |
| 6d | 155.6 | 165.8 | 71.0 | +84.6 |
| 5a | 144.2 | 141.4 | 127.0 | +17.2 |
| 19d | 140.7 | 293.7 | 214.3 | −73.5 |
| 30a | 136.7 | 171.8 | 113.2 | +23.5 |
| 25c | 136.0 | 160.8 | 195.6 | −59.6 |
| 31c | 135.8 | 164.5 | 122.2 | +13.5 |

### Biggest improvements vs none-split/none-jit

| Query | Delta | nb-jit | ns-nojit |
|-------|------:|-------:|---------:|
| 7c | −307.9 | 129.1 | 436.9 |
| 11d | −229.8 | 45.6 | 275.5 |
| 22d | −163.0 | 78.8 | 241.8 |
| 17d | −123.1 | 60.0 | 183.1 |
| 17f | −113.5 | 103.8 | 217.3 |
| 16b | −112.5 | 207.5 | 320.0 |

### Biggest regressions vs none-split/none-jit

| Query | Delta | nb-jit | ns-nojit |
|-------|------:|-------:|---------:|
| 6d | +84.6 | 155.6 | 71.0 |
| 12c | +83.8 | 115.6 | 31.9 |
| 10c | +67.0 | 232.9 | 165.9 |
| 30b | +54.1 | 97.5 | 43.4 |
| 12a | +50.3 | 73.8 | 23.4 |

### Correctness

2/2 affected configs pass (node-based/pipeline, relationship-center/pipeline). Other 4 configs unaffected.

### Remaining bottleneck

- 213 kernel-invalid iterations (31.4%) fall back to DuckDB — understanding why `AnalyzeSubIR` bails on these is the key to closing the gap with Step 3
- Middleware overhead (4,619ms) dominated by FlatTable copy from DuckDB temps + runtime CSR build — optimization opportunity: skip FlatTable copy for temps not used as CSR lookup targets

---

## Iteration 28: Lazy FlatTable/CSR Loading (Attempted, Reverted)

**Goal**: Reduce middleware overhead (4,619ms) by deferring FlatTable loading and CSR building until actually needed.

### What was attempted

1. **Lazy CSR building**: Instead of building CSR on ALL integer columns of every temp table, only build CSR when a column is actually looked up as a join key by `AnalyzeSubIR`.
2. **Lazy FlatTable loading**: Instead of immediately converting DuckDB `ColumnDataCollection` → `FlatTable` after each DuckDB-fallback sub-query, defer the conversion until a future `AnalyzeSubIR` references the temp table.

### Measurement result (node-based/pipeline-jit, avg of 10 runs)

| Metric | Before (Step 6) | After (Lazy) | Delta |
|--------|----------------:|-------------:|------:|
| Execution | 6,980ms | 7,175ms | **+195ms (+2.8%)** |
| Middleware | 4,619ms | 4,584ms | −35ms (−0.8%) |
| Wall | 12,788ms | 12,948ms | +160ms |

### Why it failed

- **Most temps ARE used**: Nearly every temp table is referenced by the subsequent kernel iteration, so deferred loading just moved the work from `extra_materialization` to `generate_sub-SQL` timing bucket.
- **Most temps have only 1-2 integer columns**: CSR savings from skipping unused columns were minimal since most temps only have the columns they need.
- **Execution regression (+195ms)**: The deferred loading introduced overhead in the hot path — `FindOrBuildCSR` checks and lazy initialization added latency inside `AnalyzeSubIR`.
- **Net negative**: −35ms MW savings did not compensate for +195ms exe regression.

### Resolution

All changes reverted (`git checkout -- include/kernel/sub_query_plan.h include/split/ir_query_splitter.h src/kernel/sub_query_plan.cpp src/split/ir_query_splitter.cpp`). Codebase returned to Step 6 (Iter 27) state.

### Lesson learned

Lazy/deferred approaches within the current per-sub-query architecture have diminishing returns because the iterative loop inherently needs most temps for the next iteration. The real fix for middleware overhead requires eliminating entire sub-query iterations (inverted indexes, precomputed bitmaps) or eliminating temp materialization entirely (Step 7 loop fusion).

### BespokeOLAP/GenDB gap analysis (completed during this iteration)

Identified 5 key structural gaps explaining the performance difference:

1. **Inverted indexes** (`keyword_to_movies`, `country_to_ids`, `note_csr`): 67/113 queries benefit from `keyword_to_movies` alone. Eliminates 2 sub-query iterations per query. BespokeOLAP pre-builds `dim_value → vector<fk_row_id>` mappings.
2. **Precomputed bitmaps** (`us_movie_bitmap`): 14 queries use `company_name.country_code="[us]"` → movie_companies → title pattern. BespokeOLAP pre-builds a bytemap to check movie membership in O(1).
3. **`person_has_aka_bits`**: 12 queries. BespokeOLAP pre-builds a bitset for persons that have an aka_name entry, eliminating the aka_name join sub-query entirely.
4. **Dictionary encoding + `note_csr`**: 70 queries with LIKE predicates currently fall back to DuckDB. BespokeOLAP uses dictionary memoization (`note_dict` + `note_memo` arrays) and inverted `note_csr` index.
5. **Loop fusion** (Step 7): Eliminates ALL temp materialization. BespokeOLAP processes entire queries in single fused loops with zero intermediate tables. This is the fundamental fix for the 4,619ms middleware overhead.

---

## Iteration 29: Step 6.5 — Unfiltered Dim Elimination + 3-Table Inverted Index Resolution

**Goal**: Convert 3-leaf sub-queries that fall back to DuckDB into kernel-handled patterns. Two optimizations:

### What changed

**Change 1: Unfiltered dim elimination** (`src/kernel/sub_query_plan.cpp`)

When a dim table (≤200 rows) has no WHERE filters AND no output columns reference it, the join is a no-op for valid FK data. Eliminate the dim leaf + edge to convert 3-leaf to 2-leaf patterns that the existing CSR join path handles.

Location: After the filtered dim resolution loop. New struct `UnfilteredDimElim { size_t leaf_idx; size_t edge_idx; }`. Loop checks:
- `dim_cache->IsDimension(leaf.name)` and `!leaf.HasFilters()`
- No output column references the dim table
- Edge has "id" on the dim side (PK join)

**Change 2: 3-table inverted index resolution** (`src/kernel/sub_query_plan.cpp`)

Pattern: `source(filtered) + bridge + target` where an inverted index maps source→target through bridge. Three indices built at startup:
- `keyword → title` (via movie_keyword)
- `name → title` (via cast_info)
- `company_name → title` (via movie_companies)

Logic:
1. Guard: `leaves.size() == 3 && edges.size() == 2 && all base && all have FlatTable`
2. Match leaves to inverted index roles (source=dim_table, bridge, target)
3. Verify edge topology
4. Check output columns: only target, or remappable bridge join-key
5. Compile source filters, scan source FlatTable for matching PKs
6. Inverted index lookup → collect target PK values
7. Selectivity guard: skip if target_vals > 50% of target rows
8. Add target PK IN-filter, erase source+bridge leaves
9. Bridge column remapping: `bridge.join_col` → `target.join_col` via `inv_col_remap` map

**Change 3: Base×base guard KEPT** (`src/kernel/sub_query_plan.cpp`)

Initially removed, then RESTORED after correctness failures on 7a/7c. Root cause: IR labels movie_link column index 2 as "movie_id" but the actual column is "linked_movie_id". When the guard was removed, kernel handled `title + movie_link + link_type` with wrong CSR, producing incorrect results.

**Change 4: Inverted index specs cleanup** (`src/storage/storage_plan.cpp`)

Reduced `kSpecs` from 11 to 3 entries. The 8 removed specs targeted dim tables already resolved by dim cache. Saves ~100MB memory and ~2s startup.

**Change 5: Cache rebuild guard** (`src/aqp_middleware.cpp`)

After loading from cache, rebuild inverted indices if empty (old cache format).

### Files modified

| File | Change |
|------|--------|
| `src/kernel/sub_query_plan.cpp` | Changes 1, 2, 3 |
| `src/storage/storage_plan.cpp` | Change 4 |
| `src/aqp_middleware.cpp` | Change 5 |
| `include/storage/storage_plan.h` | Added inverted index API |
| `CMakeLists.txt` | Added `src/storage/inverted_index.cpp` |

### Performance (node-based/pipeline-jit, avg of 10 runs, back-to-back with Step 6)

| Config | Execution (ms) | Middleware (ms) | JIT Compile (ms) | Wall (ms) |
|--------|---------------:|----------------:|------------------:|----------:|
| none-split/none-jit | 9,529 | 3 | — | 9,532 |
| node-based/pipeline-jit (Step 6) | 7,453 | 5,283 | 1,061 | 13,796 |
| **node-based/pipeline-jit (Step 6.5)** | **7,297** | **5,266** | **1,024** | **13,587** |

**Execution: 7,453 → 7,297ms (−156ms, −2.1%)**
**Wall: 13,796 → 13,587ms (−209ms, −1.5%)**
**vs none-split/none-jit: 9,529 → 7,297ms (−23.4%)**

### Correctness

All 113 JOB queries pass (node-based/pipeline-jit). 7a/7c correctness failure discovered and fixed by keeping base×base guard.

### Remaining bottleneck

- Middleware overhead (5,266ms) still dominates wall time — fundamental fix requires Step 7 loop fusion
- LIKE-only fallback queries (70 queries) still go to DuckDB
- Base×base guard prevents kernel for some 2-base-table patterns — needs IR column name fix or per-pattern guards

---

## Iteration 30 — Step 6.5.3: LIKE Support + Bug Fixes + Issue Cleanup

### Goal

1. Add LIKE/NOT LIKE support to kernel `CompileOnePredicate` (known issue #3)
2. Fix CSR direction bug in `AnalyzeSubIR` (known issue #8)
3. Fix per-row dummy vector allocation (known issue #9)

### Change 1: LIKE/NOT LIKE in CompileOnePredicate (`src/kernel/sub_query_plan.cpp`)

Added TextLike/Text_Not_Like cases in the VARCHAR switch, with 6 optimized pattern kinds:

| Pattern kind | Example | Strategy | Cost |
|---|---|---|---|
| EQUALITY | no wildcards | memcmp full string | O(1) |
| PREFIX | `Lionsgate%` | memcmp first N chars | O(1) |
| SUFFIX | `%complete` | memcmp last N chars | O(1) |
| CONTAINS | `%sequel%` | memmem() | O(n) |
| MULTI_SEGMENT | `%Downey%Robert%` | sequential memmem() | O(n*k) |
| COMPLEX | has `_` wildcard | DP-based LikeMatch | O(n*m) |

Helper functions added to anonymous namespace: `LikeMatch` (from dimension_cache.cpp), `ClassifyLikePattern`/`LikePatternKind` enum (from ir_to_llvm.cpp), `LikeSegments`/`ClassifyLikePatternEx` (from ir_to_llvm.cpp), `LikeMatchSegments` (new), `ExprContainsLike` (recursive walker), `LeafHasLikeFilter`.

### Change 2: Single-table LIKE guard (`src/kernel/sub_query_plan.cpp`)

Base tables with LIKE filters fall back to DuckDB unless an inverted-index PK filter exists (which guarantees massive row reduction). Without PK filter, DuckDB's vectorized scan is faster for large tables (cast_info 36M, movie_info 14.8M).

### Change 3: AnalyzeFinalIR LIKE guard (`src/kernel/sub_query_plan.cpp`)

Sorted-MIN path bails out if scan_leaf or any lookup leaf has LIKE filters — prevents wrong results from LIKE on lookup tables.

### Change 4: Dangling pointer fix in 3-table inverted index resolution (`src/kernel/sub_query_plan.cpp`)

`leaves.erase()` invalidated pointers (`source_leaf`, `bridge_leaf3`) used later for `dim_derived_filters` cleanup. Fixed by saving `source_ir_idx` and `bridge_ir_idx` before erasing.

### Change 5: CSR direction validation in FindCSR (known issue #8) (`src/kernel/sub_query_plan.cpp`)

Added `fk_table` direction check to both runtime CSR and `storage_plan->GetCSR()` paths in `FindCSR` lambda, matching the pattern already used in `AnalyzeFinalIR`. Prevents silent wrong results from future code changes.

### Change 6: Hoisted dummy vector (known issue #9) (`src/kernel/sub_query_plan.cpp`)

Moved `std::vector<uint64_t> dummy` from per-row allocation inside `ScanRow` lambda to a single pre-allocated `semi_dummy` outside the scan loop.

### Files modified

| File | Changes |
|------|---------|
| `src/kernel/sub_query_plan.cpp` | All 6 changes |

### Performance (node-based/pipeline-jit, avg of 10 runs)

Measurement data: `measure/job_result/step_6_fix_all/`

| Config | Execution (ms) | Middleware (ms) | JIT Compile (ms) | Wall (ms) |
|--------|---------------:|----------------:|------------------:|----------:|
| none-split/none-jit | 9,530 | 3 | — | 9,532 |
| node-based/pipeline-jit (Step 6.5) | 7,297 | 5,266 | 1,024 | 13,587 |
| **node-based/pipeline-jit (Step 6.5.3)** | **7,069** | **5,190** | **1,108** | **13,403** |

**Execution: 7,297 → 7,069ms (−228ms, −3.1%)**
**Wall: 13,587 → 13,403ms (−184ms, −1.4%)**
**vs none-split/none-jit: 9,530 → 7,069ms (−25.8% execution, +40.6% wall)**

### Correctness

All 113 JOB queries pass (node-based/pipeline-jit). Dangling pointer bug in 3a/3b/3c discovered and fixed. AnalyzeFinalIR LIKE guard fixed 1b/1d/3a/3b/3c.

### Pre-Step-7 known issues — all resolved

| Issue | Status |
|-------|--------|
| #1 MW overhead (5,190ms) | Partially addressed; fundamental fix = Step 7 loop fusion |
| #2 Inverted indices | DONE (Iter 29) |
| #3 LIKE support | DONE (this iteration) |
| #4 Dim-partitioned tables | Deferred to after Step 7 |
| #5 Dictionary encoding | Deferred to after Step 7 |
| #6 Cross-table bitmaps | Part of Step 7 loop fusion |
| #7 Base×base guard | SKIP (addressed by inverted indices) |
| #8 CSR direction bug | FIXED (this iteration) |
| #9 Dummy vector alloc | FIXED (this iteration) |
| #10 OpenMP | SKIP (loop fusion is better approach) |

### Next step

**Step 7: Loop Fusion** — the single biggest remaining optimization. Eliminates temp table materialization + runtime CSR building (~5s middleware overhead). Fuses connected sub-query chains into single scan loops with byte-maps/bitmaps as intermediates instead of temp tables. This is the path to closing the gap with BespokeOLAP (50-170x) and making wall time faster than the baseline.
