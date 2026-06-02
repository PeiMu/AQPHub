# Kernel Path Optimization Plan

Consolidated analysis and implementation plan for reducing wall time from 14.2s to below 9.5s (DuckDB baseline). The kernel path already achieves 25.8% faster execution (7.1s vs 9.5s), but middleware overhead (6.5s) makes wall time 49% worse. This document defines the optimization steps.

## Current State (step_7_2 measurement)

**Baseline**: none-split / none-jit = 9,532ms wall (all execution, no MW overhead).

**Current**: node-based / pipeline-jit = 14,202ms wall:

| Component | Time (ms) | Share |
|-----------|-------:|------:|
| execute_sub-SQL | 6,149 | 43.3% |
| **extra_materialization (CSR build)** | **3,291** | **23.2%** |
| final_exe | 990 | 7.0% |
| generate_sub-SQL (analyze + lazy FlatTable) | 966 | 6.8% |
| generate_final_sub_sql | 786 | 5.5% |
| extract_next_sub-IR | 661 | 4.7% |
| jit_compile (DuckDB JIT, 44/113 queries) | 596 | 4.2% |
| Other (parse, update_ir, show, prepare) | 85 | 0.6% |

113 queries, 678 total iterations, avg 6.0 iters/query. 69 queries pure kernel, 44 mixed (have DuckDB fallback iterations).

**MW overhead = 6,467ms** = everything except execute (7,140ms) and JIT compile (596ms).

**Top 15 queries by wall time**: 8c(429), 16b(365), 7c(314), 17a(276), 19d(272), 9d(262), 22d(230), 30a(228), 13d(227), 10c(223), 31c(219), 22a(211), 26c(208), 25a(205), 25c(197).

## Key Findings from Analysis

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

### 2. Runtime CSR Build — The Dominant MW Cost

CSR build = 3,291ms (56.8% of MW). After each sub-query, CSR is built on ALL INT32 columns of the temp result for future iterations to probe into.

**CSR build cost model**: `BuildCSR(flat, col_idx, max_val)` does:
1. Allocate `row_ptr[max_val + 2]` as `uint64_t[]` (19-31MB for title/name domains)
2. `memset(row_ptr, 0, ...)` — zeroing 19-31MB
3. Pass 1: count FK values (O(rows))
4. Pass 2: prefix sum (O(domain))
5. Pass 3: scatter row IDs (O(rows), random writes)

**Two cost profiles**:
- **Small temps (90%+ of iterations)**: e.g., 414 rows with domain 2.5M. Cost dominated by `memset(19MB)` ~3-5ms. The data passes over 414 rows are negligible. Query 29c builds 20+ CSRs across 12 iterations, mostly on tiny temps.
- **Large temps (<10% of iterations)**: e.g., 2.83M rows with domain 4M. Cost dominated by 3 data passes over millions of rows ~30-40ms per CSR.

**How CSR is used**: Kernel scans one table, does `csr->Lookup(fk_value)` to find matching rows in the lookup table. Returns `{begin, end}` pointer pair into `col_idx[]` array. The kernel iterates `col_idx[begin..end]` to access lookup table row indices.

**When CSR is NOT needed**: When the join is a pure semi-join (only existence check, all output columns from scan table). In this case, a byte-map (`uint8_t[domain]`) suffices. But many sub-queries output columns FROM the lookup table (FROM_JOIN), requiring actual row indices.

Example (16b, 6 iterations):
```
iter 1: keyword→mk: output mk.movie_id (FROM_JOIN) → CSR on temp1
iter 2: cn→mc: output mc.movie_id (FROM_JOIN) → CSR on temp2
iter 3: temp1→temp2: semi-join on movie_id → byte-map could work
iter 4: temp3→cast_info: output ci.person_id,ci.movie_id (FROM_JOIN, base CSR)
iter 5: aka_name→temp4: output an.*,temp4.movie_id (FROM_JOIN) → CSR on temp5
```

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

### Step A: Pipeline-Level Kernel with Hash Join (Priority 1)

**Goal**: Execute sub-queries on flat column arrays using hash build/probe pipelines instead of CSR-based joins. This avoids building runtime CSR on temp results (3,291ms, the dominant MW cost) while retaining all other kernel benefits (flat-array access, compiled filters, no DuckDB overhead). The pipeline kernel is a general interpreter — no query-specific code generation or training needed.

**Core idea**: The pipeline kernel executes sub-queries the same way as the current query kernel (`AnalyzeSubIR` → `ExecuteSubQueryPlan`) — scan flat columns, apply filters, join, emit to `FlatTableBuilder` — but replaces CSR lookups with hash build/probe pipelines. Crucially, **no CSR is built on output FlatTables**. Future iterations that reference a temp also use hash tables at execution time, eliminating the per-iteration CSR build cost entirely.

**Reference implementations** (take design cues from these):
- BespokeOLAP: `/home/pei/Project/BespokeOLAP/output/query_q*.cpp` — flat column arrays, CSR-based joins, byte-maps/bitsets for semi-joins, prefetching, OpenMP. Single fused function per query. E.g., `query_q8c.cpp` builds a `us_movie` byte-map via `company_csr`, then scans `cast_info` role-CSR probing the byte-map.
- GenDB: `/home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries/` — mmap'd flat columns, CSR offsets (`_idx/*__offsets.bin` + `__rowids.bin`), bitsets for filtering, OpenMP parallel scan. E.g., `Q16c/iter_0/q16c.cpp` builds a `us_company` bitset, enumerates candidate movies via CSR, then probes cast_info/aka_name CSRs.
- Both use **the same pattern**: scan flat arrays → index-based join lookups → emit. Our pipeline kernel follows this pattern but uses hash tables where CSR is unavailable.

#### JIT-Level Architecture Redesign

Redefine `--jit-level` to map each level to a distinct execution strategy. Remove the current DuckDB pipeline-jit codepath (which was net negative: +7,200ms compile for 0ms execution gain).

| `--jit-level=` | Middleware kernel | DuckDB fallback JIT |
|---|---|---|
| `expr` | none | expr-jit |
| `operator` | none | operator-jit |
| `pipeline` | **pipeline kernel** (hash build/probe on flat arrays, no CSR) | operator-jit |
| `query` | **query kernel** (CSR-based joins, existing `AnalyzeSubIR`) | operator-jit |

- **`--jit-level=pipeline`**: NEW. `AnalyzePipelineKernel()` → `ExecutePipelineKernel()`. Interprets build phase (scan → filter → hash insert) then probe phase (scan → filter → hash probe → emit) on flat column arrays. No CSR built or used. No CSR built on output. Handles all inner-join patterns (2+ tables, star/chain/snowflake). DuckDB fallback (for patterns the pipeline kernel cannot handle, e.g., non-inner joins, aggregates) uses operator-jit.
- **`--jit-level=query`**: NEW flag name for the existing query-level kernel. `AnalyzeSubIR()` → `ExecuteSubQueryPlan()`. Uses pre-built and runtime CSR indexes. Builds runtime CSR on output. DuckDB fallback uses operator-jit.
- **`--jit-level=operator`**: No middleware kernel. DuckDB operator-level JIT only.
- **`--jit-level=expr`**: No middleware kernel. DuckDB expression-level JIT only.

**Future: per-sub-plan scheduler** (Step E+). Instead of a single global `--jit-level`, a cost model decides per sub-plan which strategy to use: `{pipeline-kernel, query-kernel, operator-jit, expr-jit}`. Factors: number of tables, CSR availability, scan table size, filter selectivity, join graph topology.

#### Config & Flag Changes

- Add `AQP_JIT_QUERY (1u << 3)` flag in `aqp_jit_abi.h` (reuse bit 3, currently `AQP_JIT_OPT3` legacy alias — relocate OPT3 if needed).
- `--jit-level=pipeline` sets `AQP_JIT_PIPELINE` → triggers pipeline kernel path.
- `--jit-level=query` sets `AQP_JIT_QUERY` → triggers query kernel path (existing CSR-based).
- DuckDB fallback from either kernel: `SetJITPendingIR()` with `AQP_JIT_EXPR | AQP_JIT_OPERATOR` (operator-jit).
- Remove DuckDB pipeline-jit compilation in `ir_to_llvm.cpp` and DuckDB-side pipeline dispatch code.

#### Pipeline Kernel Design

**Terminology**:
- **Query kernel**: existing `AnalyzeSubIR()` → `SubQueryPlan` → `ExecuteSubQueryPlan()`. Uses CSR for joins. Builds runtime CSR on output for future iterations. Handles ~69% of iterations (2-table FK→PK joins with CSR).
- **Pipeline kernel**: NEW `AnalyzePipelineKernel()` → `ExecutePipelineKernel()`. Uses hash tables for joins. **No CSR built on output** — future iterations also use hash tables. Handles all inner-join patterns including the ~31% that currently fall back to DuckDB.

**Why pipeline kernel eliminates CSR overhead**:

The query kernel's dominant cost is runtime CSR build on temp results: 3,291ms (50.9% of MW overhead). For a 414-row temp with domain 2.5M, CSR costs ~5ms (19MB `memset`). A hash table on 414 rows costs ~0.01ms. Since 90%+ of iterations produce small temps, the pipeline kernel eliminates most of this cost.

| Component | Query kernel (CSR) | Pipeline kernel (hash) |
|---|---|---|
| Join index for base tables | Pre-built CSR (free at query time) | Hash table built per-execution (~5-10ms for 901K rows) |
| Join index for temp results | Runtime CSR: **~5ms per small temp** (19MB memset) | Hash table: **~0.01ms** (proportional to rows, not domain) |
| Index on output for next iteration | **Yes** — the 3,291ms cost | **None** — no CSR built |
| Lookup cost per key | O(1) direct array access | O(1) amortized hash probe |
| Total CSR overhead (113 queries) | **3,291ms** | **0ms** |

**Why no training needed**: The algorithm is general — it reuses the same IR analysis as the query kernel (`CollectLeaves`, `CollectJoinEdges`, `CompileOnePredicate`, `FlatTableBuilder`) but replaces CSR lookup with hash probe. No query-specific code generation. All patterns handled by the same interpreter.

**Execution model** — same as query kernel but with build/probe phases:

1. **Analyze** (`AnalyzePipelineKernel`): Parse sub-IR → collect leaf tables, join edges, filters (reuse `CollectLeaves`, `CollectJoinEdges`). Reject if: has aggregate, non-inner join, any leaf missing FlatTable, unsupported filter types, non-INT32 join keys. Build join graph. Pick scan table (largest by row count). BFS from scan table to order join steps. Each non-scan table becomes a `PipelineJoinStep` with build table, key column, filters, and probe source (scan table for star, previous step for chain).

2. **Build phase** (`ExecutePipelineKernel`): For each join step, scan the build table's flat columns → apply build filters → insert qualifying `(key, row_id)` pairs into a chained hash table. This is the "build pipeline" — one pipeline per join step, each a pipeline breaker (Hyper/Umbra approach).

3. **Probe phase** (`ExecutePipelineKernel`): Scan the scan table's flat columns with OpenMP parallelism (per-thread `FlatTableBuilder`). For each qualifying scan row, probe each join step's hash table in BFS order:
   - **Star pattern**: all probe keys come from scan row (e.g., `cast_info.role_id → role_type HT`, `cast_info.person_id → aka_name HT`)
   - **Chain pattern**: probe key for step i+1 comes from matched row in step i's build table (e.g., scan A → probe B HT → from matched B row, probe C HT)
   - **Semi-join steps** (no output columns from build table): `Contains(key)` — existence check only
   - **Inner join steps** (output columns from build table): `ForEach(key, callback)` — iterate all matches

4. **Emit**: Output rows → `FlatTableBuilder` → `FlatTable`. **No CSR built on output.** The FlatTable is registered as a kernel temp (`kernel_temp_ptrs_`) for future iterations. Future pipeline kernel iterations build hash tables on it at execution time.

**Hash table** (`HashJoinTable`, in anonymous namespace):
```cpp
struct HashJoinTable {
  static constexpr uint32_t EMPTY = UINT32_MAX;
  std::vector<uint32_t> buckets_;   // bucket[hash & mask] → head of chain
  std::vector<uint32_t> next_;      // next_[i] → next entry, or EMPTY
  std::vector<int32_t>  keys_;      // keys_[i]
  std::vector<uint32_t> row_ids_;   // row_ids_[i] → row in build table
  uint32_t mask_ = 0;

  void Build(const FlatTable &table, int key_col,
             const std::vector<RowPredicate> &filters);
  template<typename Fn> void ForEach(int32_t key, Fn &&fn) const;
  bool Contains(int32_t key) const;
};
```
Build: O(qualifying_rows). Probe: O(1) amortized per key. Fibonacci hash (`key * 2654435761u`), power-of-2 capacity, ~50% load factor.

**Example: query 8c, iteration 2** (currently DuckDB fallback):
```
Sub-IR: aka_name JOIN cast_info JOIN role_type
  - cast_info.person_id = aka_name.person_id
  - cast_info.role_id = role_type.id
  - Filter: role_type.role = 'writer'
  - Output: aka_name.person_id, aka_name.name, cast_info.movie_id
```
Pipeline kernel execution:
1. **Build phase**: Build HT on role_type (12 rows, filtered to ~1 row for 'writer', key=id). Build HT on aka_name (901K rows, key=person_id).
2. **Probe phase**: Scan cast_info (36M rows) with OpenMP. For each row: probe role_type HT on `role_id` (semi-join, ~90% filtered out) → probe aka_name HT on `person_id` (inner join, emit `person_id`, `name`, `movie_id`).
3. **No CSR built** on the 2.3M-row output. Next iteration builds a hash table on it.

**Join patterns handled**:
- 3+ tables with inner joins (e.g., 8c above)
- 2 base tables without CSR relationship (base×base joins)
- Star joins (all edges to scan table) and chain joins (A→B→C)
- Snowflake patterns (star with branches)
- All patterns the query kernel handles (2-table with CSR) — but using hash instead of CSR

**What this eliminates**:
- Runtime CSR build on temp results: **3,291ms** (entirely eliminated)
- DuckDB SQL generation for fallback iterations: 966ms
- DuckDB JIT compile for fallback iterations: 596ms
- DuckDB vectorized execution overhead: operator dispatch, selection vectors, segment decompression
- Lazy FlatTable+CSR build for DuckDB path: 1,752ms

**Estimated savings**: 3,000-5,000ms total MW overhead reduction.

**Code**: Extend `sub_query_plan.cpp/h` (reuse `CollectLeaves`, `CollectJoinEdges`, `CompileOnePredicate`, `CompileAllLeafFilters`, `FlatTableBuilder`, `MergeBuilders` from anonymous namespace).

**Files to modify**:
- `include/kernel/sub_query_plan.h`: Add `PipelineJoinStep`, `PipelineKernelPlan`, function declarations.
- `src/kernel/sub_query_plan.cpp`: Add `HashJoinTable`, `AnalyzePipelineKernel()`, `ExecutePipelineKernel()`.
- `src/split/ir_query_splitter.cpp`: Route based on JIT level — `AQP_JIT_PIPELINE` → try pipeline kernel (no CSR build on output), fallback to DuckDB with operator-jit. `AQP_JIT_QUERY` → try query kernel (with CSR), fallback to DuckDB with operator-jit.
- `include/jit/aqp_jit_abi.h`: Add `AQP_JIT_QUERY` flag, update `AQP_JIT_LEVEL_MASK`.
- `src/util/param_config.cpp`: Add `--jit-level=query` parsing.
- `src/jit/ir_to_llvm.cpp`: Remove DuckDB pipeline-jit compilation code.
- DuckDB-side: Remove pipeline dispatch in `physical_*.cpp` (or leave as dead code gated by the removed flag).

**Files to study & reference**:
- Current query kernel: `AnalyzeSubIR()` + `ExecuteSubQueryPlan()` in `src/kernel/sub_query_plan.cpp` — the pipeline kernel follows the same pattern (scan → join → emit) but with hash tables
- BespokeOLAP: `/home/pei/Project/BespokeOLAP/output/query_q*.cpp` — flat-array access patterns, byte-maps/bitsets for semi-joins, CSR-based probes, OpenMP parallelism. E.g., `query_q8c.cpp` shows the scan → byte-map check → CSR probe → emit pattern
- GenDB: `/home/pei/Project/GenDB/output/imdb-job-sf1/runs/latest/queries/` — mmap'd flat columns, CSR offsets for joins, bitsets for filtering, OpenMP. E.g., `Q16c/iter_0/q16c.cpp` shows build bitset → enumerate via CSR → semi-join probe → MIN aggregation
- Existing hash table: `src/jit/aqp_jit_hashtable.cpp`, `include/jit/aqp_jit_abi.h`
- DuckDB SQL generation (to be eliminated): `GenerateSQL()` in `src/adapters/duckdb_adapter.cpp`

### Step B: Sparse CSR for Small Temps (Priority 2)

**Goal**: Eliminate the dominant CSR build cost for small temp tables (90%+ of iterations).

**Problem**: Building dense `row_ptr[2.5M]` for a temp with 414 rows wastes 19MB memset. The dense CSR is designed for base tables with millions of rows; for small temps it's grossly oversized.

**Solution**: When temp has < 50K rows, use a hash-based CSR instead of dense CSR:
```cpp
struct HashCSR {
  std::unordered_map<int32_t, std::vector<uint32_t>> index;
  std::pair<const uint32_t*, const uint32_t*> Lookup(int32_t key) const;
};
```
- Build: O(rows), no domain-sized allocation. For 414 rows: ~0.01ms vs ~5ms.
- Lookup: O(1) amortized (hash lookup). Same interface as dense CSR.
- Memory: proportional to rows, not domain.

For large temps (≥ 50K rows), keep dense CSR but optimize the build:

**Large-temp CSR optimizations**:
1. **Pre-allocated row_ptr pool**: At startup, allocate reusable `row_ptr` buffers for common PK domains (title.id = 2.5M → 19MB, name.id = 4.2M → 31MB). Reuse across iterations instead of malloc/free each time. The `memset` is still needed but avoids allocator overhead.
2. **OpenMP-parallel data passes**: The 3 data passes (count, prefix-sum, scatter) over millions of rows are embarrassingly parallel. For 2.83M rows (16b iter 4): split rows across threads for count and scatter passes; prefix-sum is sequential but O(domain) only. Expected: ~2x speedup on the data passes, reducing 30-40ms to 15-20ms per large-temp CSR build.

**Estimated savings**: ~1,500-2,000ms from hash CSR on small temps + ~200-400ms from parallelized large-temp builds.

**Files to modify**:
- `include/storage/csr_index.h`: add `HashCSR` struct with same `Lookup` interface; add `CSRPool` for pre-allocated buffers
- `src/storage/csr_index.cpp`: add `BuildHashCSR()` function; add `BuildCSRParallel()` with OpenMP
- `src/split/ir_query_splitter.cpp:680-700`: choose `BuildHashCSR` vs `BuildCSRParallel` based on row count

### Step C: Byte-Map for Semi-Join Patterns (Priority 3)

**Goal**: When a runtime CSR is only used for existence checks (semi-join), replace it with a cheaper byte-map.

**When byte-map works**: The future iteration that probes into this temp only checks existence (all output columns come FROM_SCAN, not FROM_JOIN). The temp has 1 column (just the join key).

**When byte-map doesn't work**: Future iteration outputs columns FROM the lookup temp (FROM_JOIN). Need actual row indices from CSR.

**Solution**:
```cpp
struct ByteMap {
  std::unique_ptr<uint8_t[]> data;
  uint64_t size;
  bool Contains(int32_t key) const { return key >= 0 && key < size && data[key]; }
};
```
Build: allocate `uint8_t[max_val+1]`, memset to 0, mark surviving values. For domain 2.5M: 2.4MB memset (vs 19MB for CSR). Plus O(rows) to mark.

**Detection**: After producing temp N, check if temp N has only 1 column (the join key). If yes, build byte-map instead of CSR. The kernel's join step uses `bytemap.Contains(key)` instead of `csr->Lookup(key)`.

**Estimated savings**: ~500ms additional (on top of Step B).

**Files to modify**:
- `include/storage/csr_index.h`: add `ByteMap` struct
- `src/split/ir_query_splitter.cpp`: build byte-map for 1-column temps
- `src/kernel/sub_query_plan.cpp`: add byte-map check path in join step execution

### Step D: Skip Last-Iteration CSR Build (Priority 4)

**Goal**: Don't build CSR on the last iteration's output — it's only useful if the final phase uses the kernel path.

**Measured waste**: ~883ms across 104 multi-iteration queries.

**Solution**: After `UpdateRemainingIR`, check if remaining IR is trivial (final phase). If so, skip CSR build. If `AnalyzeFinalIR` succeeds, the final aggregate reads from the temp's FlatTable directly (no CSR needed for final MIN scan).

**Files to modify**: `src/split/ir_query_splitter.cpp` (post-UpdateIR check)

### Step E: Multi-Threaded Latency-Hiding Pipeline (Priority 5)

**Goal**: Hide JIT compilation and CSR build latency by overlapping them with execution on separate threads.

**3-thread design**:
```
Main thread:    interpret[0] → update_info → interpret[1] → update_info → interpret[2] → ...
JIT thread:                     compile[1] ─→ compile[2] ─→ ...
CSR thread:     ← build CSR for sub-plan[N] or [N-1] ──────────────────→
```

**Detailed flow per iteration**:
1. Main thread **interprets** sub-plan[0] immediately (no wait for JIT).
2. After sub-plan[0]'s result is known (runtime info: actual cardinality, surviving rows), main thread calls `AnalyzeSubIR()` for sub-plan[1].
3. **JIT thread** starts operator-jit compiling sub-plan[1] (or kernel JIT if kernel-handleable) concurrently with main thread's interpretation of sub-plan[0].
4. **CSR thread** builds CSR (or hash CSR / byte-map) for sub-plan[N] or [N-1]'s output, overlapping with main thread's next extract+analyze.
5. When main thread is ready to execute sub-plan[1]:
   - If JIT thread has finished compiling sub-plan[1]: use the compiled native code.
   - If JIT thread is still compiling: **abandon JIT**, interpret sub-plan[1] instead. Start JIT-compiling sub-plan[2] on the JIT thread.
6. Before executing sub-plan[N+1], **wait** for CSR thread if sub-plan[N+1] needs the CSR from sub-plan[N]'s output.

**Operator-jit (not pipeline-jit) for the DuckDB fallback path**: Pipeline-jit adds 7,200ms compile overhead for 0ms execution gain. Operator-jit adds 1,574ms compile for 45ms execution gain. With speculative compilation on a separate thread, the compile cost is hidden — only the execution gain matters. Use operator-jit for the ~31% DuckDB fallback iterations.

**Key constraints**:
- Execute(N+1) depends on CSR(N): CSR thread must finish before execution starts.
- `AnalyzeSubIR(N+1)` depends on sub-plan[N]'s result: JIT thread can only start after analyze completes.
- Thread conflicts: DuckDB uses all 12 cores internally. Use `std::thread` for MW threads (2 threads max). Remove kernel OpenMP (`sub_query_plan.cpp:1750`, threshold 10K rows) to free CPU cores — OpenMP is low-priority per CLAUDE.md issue #10.

**Overlap savings**: `min(JIT_compile, interpret_time)` per iteration for JIT thread + `min(CSR_build, Extract + Analyze)` per iteration for CSR thread. After Steps B+C reduce CSR build time, CSR thread overlap is ~300ms. JIT thread overlap depends on sub-plan size.

**Estimated total savings**: ~500-800ms (JIT latency hiding + CSR build overlap).

**Note on incremental CSR**: If runtime info from sub-plan[0] changes the join order for subsequent sub-plans (DuckDB re-optimizes), the CSR built speculatively by the CSR thread may target the wrong column. This is rare in practice (split order is determined by the splitter's heuristic, not by DuckDB's re-optimization), but the CSR thread should validate that its output is still needed before the main thread consumes it. If invalidated, rebuild (no worse than sequential).

### Step F: Kernel JIT Compilation (Priority 6)

**Goal**: Replace the query-level kernel interpreter with LLVM-compiled native code.

Current interpreter overhead: `std::function` calls (~5-10ns/row), type dispatch, join step iteration. For 50M total rows across all queries: ~500ms.

**Solution**: `CompileSubQueryPlan()` → LLVM IR → LLJIT → native function pointer. This is the compilation target for the JIT thread in Step E — the JIT thread compiles sub-plan[N+1] while the main thread interprets sub-plan[N].

**Estimated savings**: ~300-400ms (execution improvement from eliminating interpreter overhead; compile latency hidden by Step E's JIT thread).

**Code**: `src/jit/kernel_codegen.cpp`

### Step G: Additional Optimizations (Priority 7+)

After Steps A-F, re-measure and prioritize:
- **Dimension-partitioned flat tables**: Partition large tables by low-cardinality FK. Scan reduction: cast_info 36M→4M (9x). Est. 65-320ms per query.
- **Dictionary encoding**: String comparisons → integer comparisons. Est. significant for movie_info/cast_info filters.
- **Better split strategy (Direction B)**: Reorder sub-plans by selectivity. Fewer/smaller temps → less CSR build overhead.

## Wall Time Reduction Projection

| Step | Target | Savings | Cumulative Wall |
|------|--------|---------|----------------|
| Current | — | — | 14,202ms |
| A: Pipeline-level kernel | Eliminate DuckDB fallback | -2,000 to -3,000ms | ~11,500ms |
| B: Sparse + parallel CSR | Hash CSR for small temps + OpenMP for large temps | -1,700 to -2,400ms | ~9,500ms |
| C: Byte-map | Semi-join CSR elimination | -500ms | ~9,000ms |
| D: Skip last-iter CSR | Wasted CSR builds | -500ms | ~8,500ms |
| E: Multi-threaded pipeline | 3-thread: interpret + JIT compile + CSR build overlap | -500 to -800ms | ~7,900ms |
| F: Kernel JIT | Interpreter overhead (compile latency hidden by Step E) | -300ms | ~7,600ms |

**Target**: wall time < 9,532ms (DuckDB baseline). Steps A+B should get there; C-F provide further headroom.

## Code Organization

| Directory | Purpose |
|-----------|---------|
| `src/storage/` | Flat tables, CSR indexes, sorted indices, dimension cache, storage plan |
| `src/kernel/sub_query_plan.cpp` | Query-level kernel: AnalyzeSubIR + ExecuteSubQueryPlan + AnalyzeFinalIR |
| `src/kernel/pipeline_kernel.cpp` | Pipeline-level kernel (Step A, new file) |
| `src/jit/ir_to_llvm.cpp` | DuckDB JIT: expr/operator/pipeline compilation for DuckDB path |
| `src/jit/kernel_codegen.cpp` | Kernel JIT: SubQueryPlan → LLVM native code (Step F, new file) |
| `src/jit/aqp_jit_hashtable.cpp` | Hash table for JIT (used by pipeline-level kernel) |
| `src/split/ir_query_splitter.cpp` | Split loop: extract → analyze → execute → materialize → update |
| `src/adapters/duckdb_adapter.cpp` | DuckDB integration: loading, JIT registration, temp management |

## Measurement & Validation

**Measurement data**: `/home/pei/Project/AQP_middleware/measure/job_result/step_7_2/`

**Breakdown CSV parser**: `analyze_middleware_breakdown` in `/home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py`

**Per-iteration CSV format** (6 columns per iteration):
1. `extract`: SplitIR()
2. `gen_sub/analyze`: kernel path = lazy FlatTable+CSR + AnalyzeSubIR; DuckDB path = GenerateSQL
3. `jit_compile`: kernel path = 0; DuckDB path = LLVM compile
4. `execute`: kernel or DuckDB execution
5. `materialize`: CSR build + RegisterTempMetadata
6. `update_ir`: UpdateRemainingIR

**Quick test**: `bash run_job.sh duckdb node-based pipeline o1 none` + diff against golden.

**Full measurement**: `bash measure_breakdown_time_job.sh duckdb node-based pipeline o1 none` (~28 min).

**Target queries**: 16b (6 iters, large temps), 29c (12 iters, many small temps), 8c (4 iters, DuckDB fallback), 19d (8 iters), 9d (6 iters).
