# JIT Optimization Analysis

## Current JIT Execution Improvement (none-split, all 113 JOB queries)

| Config | Exec (ms) | vs baseline | Queries improved |
|--------|-----------|-------------|-----------------|
| none/none (baseline) | 9,363 | — | — |
| none/expr | 8,685 | **−677ms (−7.2%)** | 110/113 |
| none/operator | 8,719 | −643ms (−6.9%) | same as expr |
| none/pipeline | 8,696 | −667ms (−7.1%) | 106/113 |

Expr-level JIT captures nearly all the execution benefit. Operator and pipeline levels add no measurable execution improvement over expr.

## What expr-JIT already optimizes

| Optimization | Mechanism | Execution effect |
|---|---|---|
| Bloom filter prefetch | JIT-gated `__builtin_prefetch` for BF lookup entries | BloomFilter::LookupHashes 15.6% → 8.4% (16b) |
| ResolvePredicates fast-path | ScanInnerJoin equality inlining + memcpy elim | ResolvePredicates 2.5–6.8% → 0% (absorbed) |
| Chain-walk prefetch | JIT-gated prefetch in AdvancePointers | Marginal (memory-bound) |
| Row-data prefetch before Match | Prefetch row bytes before key comparison | TemplatedMatch overhead reduced |
| LLC miss reduction | Combined prefetch effects | 53M → 23M LLC misses on 16b (−57%) |

## Why operator-JIT and pipeline-JIT add nothing over expr-JIT

- **Operator-level projection**: Already zero-copy (`Vector::Reference`). No execution gain.
- **Operator-level aggregate**: Disabled for JOB (`DISABLE_AGG_JIT=1`); JOB only uses MIN on VARCHAR.
- **Pipeline filter→projection fusion**: Saves one `DataChunk` allocation per chunk (~0.1μs) — undetectable.
- **Pipeline hash probe fusion**: **Disabled** (`if (false && ...)` at `physical_hash_join.cpp:1097`). Was 1.6% slower than DuckDB's vectorized batch probe because the JIT probe was scalar row-at-a-time.

## Remaining bottlenecks after expr-JIT (query 16b)

Measured with `perf record -F 4999`, 5 iterations, none-split/expr-jit.

| Function | % of cycles | Category | Description |
|---|---|---|---|
| BloomFilter::LookupHashes | 8.4% | Hash Join | Bloom filter bit-test per probe key. Already prefetched; residual cost is the bit-test loop itself. |
| AdvancePointers | 7.5% | Hash Join Probe | Following `next` pointers in collision chains. Memory-bound random access; prefetching helps marginally. |
| InsertHashes | 6.2% | Hash Join Build | Hash table slot insertion during build phase. Not JIT'd; DuckDB native. |
| FastMemcpy | 4.9% | Materialization | Tuple gather/scatter between row-store and column-store representations. Structural overhead. |
| ScanInnerJoin | 4.2% | Hash Join Probe | JIT fast-path: inlined equality check + direct memcpy. Already near-optimal. |
| FilterSelectionSwitch\<int\> | 4.0% | Filter | DuckDB's native filter applying selection vector. Runs after JIT filter; applies the resulting selection. |
| GetRowPointers | 3.2% | Hash Join Probe | Computing hash → entry pointer (hash & bitmask → entries[slot]). Memory-random. |
| VectorOperations::Hash | 3.2% | Hash Join | Hash key computation for probe side. Called per-chunk. |
| fsst_decompress | 3.0% | Scan | FSST string decompression during table scan. Structural; not JIT-able within DuckDB. |
| Finalize | 2.7% | Hash Join Build | Build-side hash table finalization (merging thread-local partitions). |
| BuildPartitionSel\<true\> | 2.4% | Hash Join Build | Radix-partitioning selection during build. |
| TupleDataTemplatedGather\<string_t\> | 2.4% | Materialization | Gathering VARCHAR columns from row-store into vectors. |
| TupleDataTemplatedGather\<int\> | 2.3% | Materialization | Gathering INT columns from row-store into vectors. |
| VectorOperations::Copy | 1.4% | Materialization | Copying vectors between operators. |
| TemplatedMatch\<true,int,Equals\> | 1.2% | Hash Join Probe | Equality comparison during key matching. |

### Grouped by category

| Category | Combined % | Functions |
|---|---|---|
| **Hash Join Probe** | **24.5%** | AdvancePointers, ScanInnerJoin, GetRowPointers, TemplatedMatch |
| **Hash Join Build** | **11.3%** | InsertHashes, Finalize, BuildPartitionSel |
| **Hash Join Bloom Filter** | **8.4%** | BloomFilter::LookupHashes |
| **Hash Join Hashing** | **3.2%** | VectorOperations::Hash |
| **Materialization** | **11.0%** | FastMemcpy, TupleDataGather\<string_t\>, TupleDataGather\<int\>, VectorOperations::Copy |
| **Filter** | **4.0%** | FilterSelectionSwitch |
| **Scan/Decompress** | **3.0%** | fsst_decompress |
| **Total accounted** | **65.4%** | |

### Cross-query comparison (top 3 heaviest queries)

| Function | 16b (307ms) | 8c (226ms) | 19d (209ms) |
|---|---|---|---|
| BloomFilter::LookupHashes | 8.4% | 3.7% | 4.8% |
| AdvancePointers | 7.5% | 5.8% | 2.7% |
| InsertHashes | 6.2% | 9.4% | 5.8% |
| ScanInnerJoin | 4.2% | 8.7% | 10.8% |
| ResolvePredicates | 0% | 0% | 0% |
| GetRowPointers | 3.2% | 3.1% | 1.8% |
| FilterSelectionSwitch | 4.0% | 1.4% | 1.4% |
| fsst_decompress | 3.0% | 2.0% | 4.4% |

Pattern: hash join dominates all heavy queries (35–47% of cycles). The specific bottleneck shifts between build (InsertHashes on 8c) and probe (AdvancePointers on 16b, ScanInnerJoin on 19d) depending on table sizes and join selectivity.

## Why OpenMP cannot help

DuckDB already uses **morsel-driven parallelism** with 12 threads:
- threads=1: 1.542s → threads=12: 0.341s (4.5× speedup, 7.7 CPUs utilized)
- All scan, filter, build, and probe operations are already parallelized
- JIT functions are called per-chunk (2048 rows) from DuckDB's parallel pipeline — OpenMP inside would conflict with DuckDB's thread pool
- Node-based split sub-queries have data dependencies and must run sequentially

## Potential JIT optimizations for remaining bottlenecks

### 1. Re-enable vectorized hash probe fusion (pipeline-level)

**Target**: AdvancePointers (7.5%) + GetRowPointers (3.2%) + Hash (3.2%) + BloomFilter (8.4%) = ~22%

The disabled `CompileFilterProbeProjectFusion` was scalar row-at-a-time. DuckDB's vectorized batch probe was faster. The fix: use the existing `batch_probe_` two-stage approach (stage 1: hash + prefetch all keys, stage 2: probe with warm cache) with inlined hash computation, key comparison, and output materialization in a single compiled loop — eliminating 5+ function call boundaries per row.

**Est. impact**: 5–10% execution reduction if the compiled batch probe matches DuckDB's vectorized performance while eliminating function call overhead.

### 2. SIMD Bloom filter lookup (expr or pipeline level)

**Target**: BloomFilter::LookupHashes (8.4% on 16b)

The bloom filter check is 3 bit-tests per hash. With AVX2, could test 8 hashes in parallel using `vpgatherdd` + `vpand` + `vpcmpeqd`. The current scalar loop processes one hash at a time.

**Est. impact**: 2–4% execution reduction on bloom-filter-heavy queries.

### 3. AMAC-style interleaved AdvancePointers (pipeline-level)

**Target**: AdvancePointers (7.5% on 16b)

Maintain 8–16 probe keys at different stages of the chain walk. While one key waits for cache-line fill, process the next key. Hides L3 latency (~40ns) behind useful work.

**Est. impact**: 2–3% execution reduction. High implementation complexity (state machine conversion).

### 4. Scan+filter+hash fusion (pipeline-level)

**Target**: FilterSelectionSwitch (4.0%) + Hash (3.2%)

Fuse the scan filter evaluation with hash key computation in a single pass. Currently: scan → filter (selection vector) → hash (reads filtered rows). Fused: scan → evaluate filter + compute hash for passing rows in one loop, avoiding re-reading filtered columns.

**Est. impact**: 1–2% execution reduction from better cache locality.
