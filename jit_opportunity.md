# JIT Execution-Time Optimization Opportunities

Analysis based on JOB benchmark (113 queries) on DuckDB 1.5.2, measured across
none / relationship-center / node-based split strategies with
none / expr / operator / pipeline JIT levels.

---

## 1  Current Bottleneck Summary

Total operator execution time is ~100 s across all 113 JOB queries regardless
of JIT level.  Two operators dominate:

| Operator    | no-jit (s) | share | JIT impact today |
|-------------|-----------|-------|------------------|
| TABLE_SCAN  | 64.1      | 64 %  | none (I/O bound) |
| HASH_JOIN   | 32.4      | 32 %  | none (memory bound) |
| FILTER      |  3.8      |  4 %  | operator-jit -32 %, pipeline-jit -79 % (but net F+TS is flat) |
| Others      | <0.3      | <1 %  | negligible |

Key finding: **current JIT reduces FILTER time but the scan+filter fusion
merely shifts work into TABLE_SCAN**.  Combined FILTER + TABLE_SCAN time is
unchanged (+0.35 s for none-split, +0.07 s for node-based).

HASH_JOIN probe has JIT infrastructure (batch probe + prefetch in
`ir_to_llvm.cpp:6092-6280`).  The pipeline-jit measurements already have
`--jit-batch-probe` and `--jit-prefetch` enabled (defaults are `on` in
`measure_breakdown_time_aqp.sh job`), yet hash join time is unchanged.  The
prefetch infrastructure is active but **not effective at current default
distances (entry=24, row=12)** on this workload.

---

## 2  Per-Operator JIT Impact

### 2.1  FILTER

| JIT level | Total FILTER (s) | vs no-jit |
|-----------|-----------------|-----------|
| no-jit    | 3.77            | baseline  |
| expr-jit  | 3.67            | -2.7 %    |
| operator-jit | 2.55         | -32 %     |
| pipeline-jit | 0.79         | -79 %     |

- expr-jit compiles the WHERE clause into a selection-vector function
  (`AQPExprFn`).  Modest gain because DuckDB's vectorized executor is already
  efficient for simple predicates.
- operator-jit adds scan+filter fusion (`physical_table_scan.cpp:182-203`):
  compiled filter runs inside the scan operator, eliminating the separate
  PhysicalFilter call.
- pipeline-jit further fuses filter + hash-join probe or filter + projection
  into a single compiled function.

**Observation**: the FILTER decrease is almost exactly offset by a TABLE_SCAN
increase because the profiler attributes fused-filter work to the scan
operator.

### 2.2  HASH_JOIN

| JIT level | Total HASH_JOIN (s) | vs no-jit |
|-----------|-------------------|-----------|
| no-jit    | 32.37             | baseline  |
| expr-jit  | 32.48             | +0.3 %    |
| operator-jit | 32.50          | +0.4 %    |
| pipeline-jit | 32.35          | -0.1 %    |

Pipeline-JIT compiles the probe path (`physical_hash_join.cpp:1085-1112`)
with inlined MurmurHash64, salt check, and typed key comparison against
DuckDB's JoinHashTable via `AQPJoinHTView`.  Zero measurable improvement
despite batch-probe + prefetch being enabled (default `on`), because:
- DuckDB's native `ScanStructure` probe is already vectorized (2048-row
  batches with selection vectors).
- The JIT probe is scalar (row-by-row), even with batch-probe stage-1
  decoupling hash computation from memory access.
- Default prefetch distances (entry=24, row=12) may not be optimal for the
  actual HT sizes in JOB queries.

### 2.3  TABLE_SCAN

Purely I/O-bound (reading column segments from DuckDB storage).  JIT cannot
help the scan itself.  Pipeline-jit makes it appear slower because fused-filter
work is attributed to the scan operator.

### 2.4  PROJECTION

operator-jit replaces `ExpressionExecutor::Execute` with zero-copy column
remapping via `proj_col_maps`.  Real but negligible (0.018 s -> 0.011 s).

### 2.5  UNGROUPED_AGGREGATE

operator-jit compiles the aggregate update loop.  Real but negligible
(0.088 s across all queries).

---

## 3  Scan + Filter Fusion: When It Hurts

Pipeline-jit fuses filters into table scans.  This **hurts** when filter
selectivity is low (few rows pass), because the scan still reads all pages
but now also evaluates the filter per-row inside the scan operator.

Observed slowdowns (none-split):

| Query | Selectivity | no-jit F+TS (s) | pipe-jit F+TS (s) | Slowdown |
|-------|------------|-----------------|-------------------|----------|
| 23b   | 0.17 %     | 0.352           | 0.453             | +29 %    |
| 6b    | 0.00005 %  | 0.302           | 0.382             | +27 %    |
| 10c   | 4.2 %      | 1.391           | 1.529             | +10 %    |

Pattern: filters with selectivity < ~5 % on large base tables produce net
slowdowns when fused.

### Optimization O1 -- Adaptive Scan+Filter Fusion

Disable fusion when estimated selectivity is below a threshold.

- **Static path**: use DuckDB's cardinality estimates from EXPLAIN before
  compilation.  If `filter_output_card / scan_card < 0.05`, skip
  `scan_filter_fns` registration for that scan.
- **Runtime path (split only)**: after sub-plan N, if the same filter
  pattern appears in sub-plan N+1 (same table, same predicate structure),
  use observed selectivity from N to decide.

Implementation: in the middleware's `RegisterJIT()`, check estimated
selectivity before adding to `jit->scan_filter_fns`.

---

## 4  Hash Join Optimization

HASH_JOIN consumes 32 s (32 % of total).  This is the highest-leverage
target.

### 4.1  Existing Infrastructure (Not Yet Measured)

The JIT compiler already generates a two-stage ROF (Result-Oriented Fusion)
probe (`ir_to_llvm.cpp:6092-6280`):

**Stage 1** (`batch_probe_`): for all rows in chunk, evaluate filter, compute
hash + `ht_offset`, store in stack arrays.  No dependent memory accesses.

**Stage 2**: for each row, issue software prefetch for row i+D ahead, then
do the actual probe using precomputed hash.  Prefetch hides memory latency.

Prefetch granularity:
- `prefetch_entry_distance_`: prefetch `entries[ht_off[i+D]]` (hash table
  entry cache line)
- `prefetch_row_distance_`: speculatively dereference entry to get row
  pointer, prefetch row data

**These flags are OFF in all current measurements** (`--no-jit-batch-probe
--no-jit-prefetch`).

### Optimization O2 -- Tune Prefetch Distances

Batch probe + prefetch are already ON in pipeline-jit measurements (defaults
`on` in `measure_breakdown_time_aqp.sh job`), but show no improvement at the
default distances (entry=24, row=12).  Sweep a wider range of distances on
representative queries with large HTs to find the effective range, or
determine that the DuckDB-native vectorized probe is fundamentally hard to
beat with scalar prefetch-based code.

### Optimization O3 -- Runtime-Adaptive Prefetch Distance

Optimal prefetch distance depends on hash table size relative to cache:

| HT size vs cache  | Prefetch strategy |
|-------------------|-------------------|
| HT fits in L2 (<256 KB)  | distance = 0 (no prefetch; hits are cheap) |
| HT fits in L3 (<8-32 MB) | distance = 8-16 |
| HT exceeds L3            | distance = 16-32 |

For split execution, build-side cardinality is known after temp-table
materialization.  Compute `ht_size = build_card * tuple_size * 1.43` and
select distance from a lookup table calibrated by the prefetch sweep.

Implementation: in `ExecuteSQLandCreateTempTable()`, after materialization,
compute HT size estimate and set `prefetch_entry_distance_` /
`prefetch_row_distance_` on the JIT context before the next sub-plan's probe.

### Optimization O4 -- SIMD-Vectorized Probe

Current JIT probe is scalar (one row at a time).  DuckDB's native probe
processes 2048-row batches.  To beat it:

- Use AVX2/AVX-512 gather to load 8-16 probe keys at once
- Compute 8-16 hashes in parallel
- Use SIMD comparison to check keys against hash table entries
- Use scatter to write matching output rows

This requires extending the LLVM IR generation to emit vector instructions
for the probe loop.

### Optimization O5 -- Partition-Aware Probe (Large HT)

When HT exceeds L3, even prefetching saturates memory bandwidth.  Alternative:

1. Partition probe side by `hash & (P-1)` where `P = ceil(ht_size / L3_size)`
2. Process one partition at a time; each partition's HT region fits in L3
3. Turns random accesses into sequential-within-partition accesses

Natural fit for split execution: temp tables are already partial results,
and further partitioning by hash is cheap.

---

## 5  Table Scan Optimization

TABLE_SCAN consumes 64 s (64 %).  The bottleneck is data movement, not
computation.  JIT specialization of the scan itself has minimal impact.

### Optimization O6 -- Range Predicate Injection (Split Only)

After sub-plan N produces a temp table, collect min/max of join key columns.
Inject range predicates into sub-plan N+1's base-table scans:

```
Sub-plan 0: scan cast_info -> temp0 (movie_id range [120, 45000])
Sub-plan 1: scan title WHERE title.id BETWEEN 120 AND 45000
            JOIN temp0 ON title.id = temp0.movie_id
```

DuckDB can use the range predicate for zone-map pruning, skipping entire
column segments whose min/max fall outside the range.

Implementation:
- After `ExecuteSQLandCreateTempTable()`, scan the temp table's join key
  column for min/max (cheap: data is in memory).
- Store in `TempTableInfo`:
  ```cpp
  std::unordered_map<std::string, std::pair<int64_t, int64_t>> col_min_max;
  ```
- In `UpdateRemainingIR()`, add `BETWEEN` predicates to base-table scan
  nodes that join against this temp table.

Helps when temp table's key range covers a small fraction of the base table's
domain.  No-op when the range spans the full domain.

### Optimization O7 -- Bloom Filter Injection (Split Only)

Stronger than range predicates: build a Bloom filter on the temp table's
join key column and push it into the next sub-plan's scan as a semi-join
filter.

```
Sub-plan 0 -> temp0 (8790 rows)
Build Bloom filter on temp0.movie_id (~1 KB for 8790 elements)

Sub-plan 1: scan title WITH bloom_check(title.id)
-> ~1 % false positive rate, eliminates most non-matching rows before join
```

This is DuckDB's Dynamic Filter Push-Down, but done explicitly between
sub-plans using actual materialized data.

Implementation:
- Build Bloom filter after temp-table materialization (simple bit array +
  k hash functions)
- Register as a `scan_filter_fn` in the JIT context for the next sub-plan's
  scan operator
- Or inject as a SQL predicate if the engine supports UDF-based filters

### Optimization O8 -- Late Materialization for Fused Scan+Filter

When scan+filter fusion is enabled, verify that the scan only reads columns
needed for the filter predicate, and fetches remaining columns only for
qualifying rows.  If the JIT flattens the entire chunk before filtering,
it forces all columns to be materialized upfront, negating the benefit.

---

## 6  Per-Pipeline JIT Scheduling

Not all pipelines benefit from JIT.  A scheduling strategy can
enable/disable JIT per pipeline to avoid overhead on unhelpful pipelines.

### Optimization O9 -- Pattern-Based JIT Scheduling

| Pipeline Pattern | Decision | Rationale |
|-----------------|----------|-----------|
| FILTER (selectivity > 10 %) + TABLE_SCAN | Enable scan+filter fusion | Filter work reduces downstream rows |
| FILTER (selectivity < 5 %) + TABLE_SCAN | Disable fusion; use operator-jit filter only | Fusion overhead exceeds savings |
| HASH_JOIN dominated (> 50 % of query time) | Enable batch probe + prefetch if HT > L2 | Memory-latency hiding |
| HASH_JOIN with small HT (< L2) | Skip JIT | Probes are fast anyway |
| PROJECTION only | Enable operator-jit | Cheap zero-copy optimization |
| TABLE_SCAN only (no filter above) | Skip JIT | Nothing to compile |
| Complex string predicates in FILTER | Enable expr-jit or operator-jit | Compiled string comparison > interpreted |

### Optimization O10 -- Runtime-Adaptive Scheduling (Split Only)

For split execution, use first sub-plan as a profiling run:

1. Execute sub-plan 0 with full JIT enabled + timing collection
2. Observe per-operator times, filter selectivities, HT sizes
3. For sub-plans 1..N, enable/disable JIT features based on observed data:
   - If filter selectivity was < 5 %: disable fusion
   - If HT size > L3: enable prefetch with calibrated distance
   - If HT size < L2: disable hash-join JIT

This avoids the "compile everything" overhead for pipelines where JIT
provides no execution benefit.

---

## 7  Split-Specific Execution Improvements

### Optimization O11 -- Cardinality-Guided Join Reordering

Already partially implemented via `ReorderBeforeSplit()` in TopDownSplitter.
Extend it: after sub-plan N, if temp table cardinality differs significantly
from the optimizer's estimate (> 2x), re-optimize remaining sub-plans' join
orders using actual cardinalities.

### Optimization O12 -- Reduce Temp-Table Materialization

The `extra_materialization` cost (writing intermediate results to
`ColumnDataCollection`) is part of middleware overhead.  Opportunities:

- **Columnar projection push-down**: only materialize columns needed by
  downstream sub-plans, not the full SELECT list.
- **Pipeline chaining**: if sub-plan N+1 only reads temp_N once and temp_N
  is not needed again, stream directly instead of materializing.

---

## 8  Measurement Priorities

Ranked by expected impact and effort:

| Priority | Experiment | Expected Impact |
|----------|-----------|-----------------|
| **P0** | Sweep prefetch distances on HJ-heavy queries (batch-probe + prefetch are already ON at defaults but show no gain) | Medium: may reveal that DuckDB's vectorized probe is hard to beat with scalar JIT |
| **P1** | Implement adaptive scan+filter fusion (selectivity threshold) | Medium: eliminates 10-30 % slowdowns on affected queries |
| **P2** | Implement range-predicate injection between sub-plans | Medium-High: reduces scan volume for selective joins |
| **P3** | Implement Bloom filter injection between sub-plans | High: stronger than range predicates |
| **P4** | Implement runtime-adaptive prefetch distance | Medium: auto-tunes to HT size |
| **P5** | SIMD-vectorized probe | High but high effort: requires vector IR generation |
| **P6** | Per-pipeline JIT scheduling | Medium: avoids compilation overhead |
| **P7** | Partition-aware probe | Medium: helps only for very large HTs |

---

## 9  Data Sources

All analysis based on:
- `duckdb_{split}_{jit}_{opt}_{simd}_{flags}_breakdown_time_log.csv` --
  per-phase timing
- `duckdb_{split}_{jit}_{opt}_{simd}_{flags}_operator_exe.csv` --
  per-operator per-iteration timing
- Source code: `/home/pei/Project/duckdb/src/execution/operator/` (JIT
  dispatch), `/home/pei/Project/AQP_middleware/src/jit/ir_to_llvm.cpp` (code
  generation), `/home/pei/Project/AQP_middleware/src/split/ir_query_splitter.cpp`
  (split execution loop)
