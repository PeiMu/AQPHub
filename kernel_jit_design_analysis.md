# Kernel JIT Design Analysis

Analysis of execution strategies for the AQP middleware kernel path, covering CSR vs hash join tradeoffs, kernel code patterns, compilation approaches, and the role of runtime statistics in split execution.

## Core Principle

Split execution's key value is **collecting runtime information** (e.g., actual cardinalities, surviving row sets) between sub-queries to improve subsequent sub-query execution. This rules out loop fusion (which merges sub-queries into one pass and loses the runtime info collection point). All optimizations must preserve the split boundary.

---

## 1. CSR vs Hash Join

### Hash Join (DuckDB's approach)
- **Build phase**: Scan build-side table, insert each row's join key into a hash table. O(n), involves hash computation, memory allocation, collision chain handling.
- **Probe phase**: For each probe-side row, hash the join key, follow collision chain to find matches.
- **Cost profile**: Hash computation + random memory access (pointer-chasing through collision chains) + build phase is a pipeline breaker.

### CSR (Compressed Sparse Row)
- Pre-built index on a FK column. Two arrays:
  - `row_ptr[pk_value]` → start position in `col_idx`
  - `col_idx[start..end]` → FK-side row IDs with this PK value
- **Lookup**: Given PK value `x`, matching FK rows = `col_idx[row_ptr[x]..row_ptr[x+1]]`. No hashing, no collision chains. O(1) to find range, O(k) to iterate k matches.
- **CSR chain**: Multiple CSR lookups in sequence (A→B→C), each as a nested loop with direct array access.

### Comparison

| Aspect | Hash Join | CSR |
|--------|-----------|-----|
| Build cost | Per-query: O(n) hash + insert | One-time startup: O(n + max_pk) |
| Probe cost | Hash + chain walk (random access) | Array index (sequential access) |
| Pipeline breaker | Yes (build must complete before probe) | No (index pre-exists) |
| Memory pattern | Random (hash table + chains) | Sequential (contiguous arrays) |
| Runtime build (temps) | O(n), DuckDB does this anyway | O(n + max_pk), middleware builds |
| Startup cost | None | ~10s for base tables (amortized, cacheable) |

### For temp tables specifically
The build cost is comparable: hash table O(n) vs CSR O(n + max_pk). Both are built after each sub-query. CSR probe is faster (array access vs hash+chain), but CSR uses more memory when PK space is sparse.

---

## 2. Kernel Code Patterns

Every kernel-handled sub-query reduces to one of three patterns:

### Pattern A: Scan + Filter + CSR chain → materialize temp
```
for row in scan_table:
    if !filter(row): continue
    for j1 in csr[0].lookup(row.fk0):
        // optional: filter on j1
        for j2 in csr[1].lookup(j1.fk1):
            emit(row, j1, j2)  → temp table
```
Variation: 0-4 CSR join steps, various filter types (=, IN, LIKE, IS NULL, ...), optional bitset semi-join checks.

### Pattern B: Scan + Filter + CSR chain → final aggregate (last sub-query)
```
for row in sorted_order(scan_table, min_col):
    if !filter(row): continue
    for j1 in csr[0].lookup(row.fk0):
        if !bitset[j1]: continue   // semi-join against temp
        result = row[min_col]
        return result               // early termination via sorted index
```

### Pattern C: Single-table scan + filter (no joins)
```
for row in table:
    if filter(row):
        emit(row)
```

**Structural invariant**: One outer scan loop over the "scan table," with nested CSR probes into "lookup tables." No hash tables, no hash probing.

---

## 3. Compilation Approaches

### Current architecture (interpretation)
1. **Frontend** (done): `AnalyzeSubIR()` analyzes sub-query IR → produces `SubQueryPlan` struct (scan table, filters, CSR join steps, output columns, aggregation).
2. **Backend** (interpreter): `ExecuteSubQueryPlan()` walks the `SubQueryPlan` at runtime — branching on `step.use_bitset`, dispatching `std::function` for filters, type-switching for column access.

### Approach A: Compile SubQueryPlan → LLVM native code
- Same `AnalyzeSubIR()` frontend → same `SubQueryPlan`.
- Replace interpreter with `CompileSubQueryPlan()` → LLVM IR → LLJIT → native function pointer.
- Eliminates: `std::function` call overhead (~5-10ns/row), type dispatch branches, join step iteration, output column source checks.
- Uses CSR for joins. No pipeline breakers.

### Approach B: Pipeline-level kernel JIT with hash join (no CSR)
- Compile each DuckDB pipeline into a tight loop, keeping hash tables as the join mechanism.
- Build phase: compiled scan → compiled hash insert (still a pipeline breaker).
- Probe phase: compiled scan → compiled filter → compiled hash probe → compiled projection.
- This is the Hyper/Umbra approach.

### Approach A vs B comparison

| Aspect | A: CSR + compiled kernel | B: Hash join + compiled pipeline |
|--------|-------------------------|----------------------------------|
| Join mechanism | CSR (pre-built, array access) | Hash table (per-query build, hash+chain) |
| Pipeline breakers | None (CSR pre-exists) | Yes (hash build) |
| Startup cost | ~10s CSR build (amortized) | None |
| Per-query probe | Faster (sequential array) | Slower (random hash) |
| Complexity | Own plan analysis + codegen | Hook into DuckDB's plan + codegen |
| Pattern coverage | ~69% (31% fall back to DuckDB) | 100% (all DuckDB patterns) |
| Works within DuckDB | No, bypasses DuckDB | Yes, replaces DuckDB's executor |

**Key observation**: Approach B saves startup CSR build time but is slower at every probe. In benchmark settings (113 queries × 15 runs), CSR's amortized startup is negligible. Approach B's main advantage is 100% pattern coverage (no fallback).

### Hybrid: CSR for base tables, hash join for unsupported patterns
Use CSR where the kernel supports the pattern (~69%), fall back to DuckDB's hash join for unsupported patterns (~31%). This is the current architecture. Improving pattern coverage (reducing the 31% fallback) is more valuable than switching to hash-join-only compilation.

---

## 4. Runtime Statistics in Split Execution

### What runtime information is available after each sub-query

1. **Actual output cardinality**: `temp_table.size()` — exact row count, no estimation error.
2. **Surviving row set**: Which PK values appear in the temp table — enables bitset/byte-map for O(1) existence checks.
3. **Value distribution**: Actual min/max/distinct counts of output columns — enables range-based optimizations for downstream filters.
4. **Per-join fanout**: Observed fanout at each CSR join step — informs join order for next sub-query.
5. **Filter selectivity**: Fraction of rows passing each filter — informs whether to apply filters early or late in next sub-query.

### How runtime statistics can improve next sub-query execution

#### 4.1 Scan direction decision
After sub-query 1 produces temp T1, sub-query 2 joins T1 with base table B:
- If `T1.size()` is small (e.g., 100): iterate T1, CSR-probe into B → 100 lookups.
- If `T1.size()` is large (e.g., 2M out of 2.5M possible): scan B, check bitset → sequential scan.
- Breakeven at roughly `T1.size() ≈ B.size() / average_CSR_fanout`.
- DuckDB's estimated cardinality may be wrong by orders of magnitude; actual cardinality is exact.

#### 4.2 Join order optimization
With multiple CSR join steps in a sub-query, order matters:
- Step with fanout 1→2 before step with fanout 1→100 reduces intermediate rows.
- Runtime fanout from previous sub-queries (or from CSR `row_ptr` statistics) can inform this ordering.
- Both BespokeOLAP and GenDB fix join order at compile time. We can do better at runtime.

#### 4.3 Bitset vs CSR for semi-join
When only existence is needed (semi-join pattern):
- Dense temp (>50% of PK space): dense bitset, scan base table.
- Sparse temp (<1% of PK space): iterate temp rows, CSR-probe.
- Runtime cardinality + PK space size determines which is better.

#### 4.4 Kernel vs DuckDB fallback decision
If temp table is very large (low selectivity from previous sub-query), the kernel's single-threaded scan may be slower than DuckDB's parallel vectorized execution. Runtime cardinality informs when to fall back.

### A/B test design for runtime statistics

**Goal**: Quantify the impact of using actual runtime cardinality vs DuckDB's estimated cardinality for execution strategy decisions.

#### Step 1: Logging (no behavior change)
Log for every sub-query execution:
```
query_id, sub_query_idx, estimated_card (from DuckDB), actual_card, scan_table, scan_table_size, num_join_steps
```
DuckDB's estimate: accessible via `PhysicalOperator::estimated_cardinality` (public member on every physical operator).
Actual cardinality: `temp_table.size()` after execution.

This step alone reveals **how wrong DuckDB's estimates are** for split sub-queries and whether the error is large enough to affect strategy decisions.

#### Step 2: Strategy selector
Add a cardinality-based strategy selector in `AnalyzeSubIR()` / `ExecuteSubQueryPlan()`:
```
if (cardinality < SCAN_THRESHOLD) {
    // Iterate temp rows, CSR-probe into base
    plan.scan_table = temp;
    plan.join_steps = [{csr: base_csr, ...}];
} else {
    // Scan base table, check bitset built from temp
    plan.scan_table = base;
    plan.use_bitset_for_temp = true;
}
```

#### Step 3: A/B comparison

| Variant | Cardinality source | When decided |
|---------|--------------------|--------------|
| A (baseline) | DuckDB's `estimated_cardinality` | Before sub-query execution |
| B (runtime) | Actual `temp.size()` from previous sub-query | After previous sub-query, before next |

Both use identical strategy logic and thresholds. Only the cardinality input differs.

#### Step 4: Measurement
Run full JOB benchmark with both variants. Compare per-query execution time. Queries where A ≠ B reveal where DuckDB's estimates are wrong AND the wrong estimate led to suboptimal strategy.

### Expected impact assessment
For JOB queries with node-based split, impact is likely **moderate**:
- Most temp tables are small (<10K rows) due to selective starting filters → scan direction is almost always "iterate temp, CSR-probe base" regardless of estimate.
- The cases where estimates matter most: queries with low-selectivity sub-queries producing large temps, where the wrong scan direction causes unnecessary full scans.
- Larger impact expected in real workloads with less predictable cardinalities.

---

## 5. Reference Implementation Findings

### BespokeOLAP (`/home/pei/Project/BespokeOLAP/output/`)
- **Zero hash joins** for join operations. Everything uses CSR probes in nested loops.
- **No pipeline breakers**: queries are fully pipelined (one exception: Q20 deduplicates intermediate results).
- **Join order is fixed at code generation time**, no runtime adaptation.
- **Scan large table, CSR-probe small**: predominant pattern (e.g., cast_info 30M+ rows full scan with CSR probes into smaller tables).
- **Hash maps used only for filtering** (keyword_to_id, keyword_to_movies), not for joins.
- **Dimension-partitioned tables**: partitions by FK for reduced scan range (e.g., `cast_info.role_movie_csr[role_id]`).

### GenDB (`/home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries/`)
- **Pure CSR-based**: all joins use CSR index structures, no hash tables.
- **Streaming push-down**: no build/probe phases; semi-joins via pre-computed bitsets.
- **Static join order**: determined at code generation time, with selective filters early.
- **Cardinality baked into code**: table sizes hardcoded (e.g., `TITLE_N = 2528312`).
- **Morsel-parallel execution**: some queries use OpenMP with morsel chunking.

### Key gap vs our approach
Both references use AOT compilation (per-query C++ compiled to native). We use split execution + runtime interpretation/JIT. Our advantage: **runtime adaptivity** — we can adjust strategy based on actual cardinalities that AOT approaches must estimate at compile time. Our disadvantage: interpretation overhead and temp materialization between sub-queries.

---

## 6. Design Decisions and Next Steps

### Decided
- **Keep CSR** for base tables (amortized startup cost, faster probe than hash join).
- **Keep split execution** boundary (enables runtime info collection — the core value proposition).
- **No loop fusion** (would eliminate split boundary and runtime info collection point).

### To investigate
1. **A/B test for runtime statistics** (Section 4): Start with logging (Step 1) to quantify estimate error, then implement strategy selector (Steps 2-3) if error is significant.
2. **Compile SubQueryPlan to LLVM** (Approach A in Section 3): Replace interpreter with compiled kernel. Orthogonal to runtime statistics — both can be combined.
3. **Improve pattern coverage**: Reduce the 31% fallback-to-DuckDB rate by handling more sub-query patterns in the kernel.
4. **Runtime CSR vs hash table for temp tables**: If CSR build on sparse PK space is expensive, consider hash table as alternative for temps with sparse keys.

### Open questions
- What is the actual estimate error distribution across JOB sub-queries? (Answer via Step 1 logging.)
- At what cardinality threshold does scan-base-with-bitset beat iterate-temp-with-CSR? (Answer via microbenchmark.)
- Is compile time for SubQueryPlan → LLVM acceptable per sub-query? (Estimate: 0.1-1ms per sub-query based on existing JIT compile times.)
