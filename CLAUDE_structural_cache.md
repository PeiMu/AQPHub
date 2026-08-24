You are a DBMS JIT compilation expert. Your task is to iteratively improve wall time (including execution time, compilation time, and middleware overhead) for the JOB benchmark through JIT-related code changes (the status of the current iteration is in section "## Current Performance"). Especially with split strategy, e.g., node-based split, where we can collect real runtime information to help with JIT code generation and/or better plan. Find whatever information helps, then check how to collect them. Targeting speedup 10x of the query wall runtime (ignore compilation time) with either level of JIT with node-based split, until no optimization can be found or heavy wall time queries take only 1 ms. You can also improve the split strategy. Approach this as Prof. Thomas Neumann or Matthias Jasny would: apply every low-level technique available, leave nothing on the table. Focus on the bottlenecks identified in the tracing output, but also reconsider the overall approach if the remaining gap is large.
Always check ## Helper Section for optimization techinique.
When discussion, think in Prof. Thomas Neumann or Matthias Jasny way.

## Goal

## Analysis — Template Cache (Plan-Level Structural Reuse)

### Terminology

- **Parameterize-cache** (existing, `cache_mode_ == 2`, `--jit-cache=single-run-template`): Caches compiled code for a query-JIT plan where filter constants are replaced by type tags. Two plans match only when they have **identical topology AND identical table/column identities** — only the filter literal values differ. Runtime params buffer supplies constants via `QjitUserData::params`.

- **Template-cache** (proposed): Caches compiled code at the **structural shape** level. Two plans match when they have the same dtype signature, operation sequence, and HT layout — regardless of which specific tables or columns are referenced. Column/table identity becomes a **runtime binding** resolved at execution setup time, not a compile-time constant.

### Key Insight: Compiled Code Does Not Embed Table/Column Identity

Verified by tracing the codegen path in `CompileQuerySteps` (`ir_to_llvm.cpp:9090+`):

1. **Source columns**: The compiled code accesses `ctx->sources[step_idx].cols[i].data` where `i` is the **positional index** within `QjitStep::cols`. The original `(table_index, column_index)` from the IR is resolved at compile time by `FindColIdx` (`ir_to_llvm.cpp:909-912`) to this positional index. The generated LLVM instruction is `cc.col_data[i]` — a compile-time constant index.

2. **Hash table probes**: The compiled code uses `qjit_ht_dir(ctx->hash_tables[ht_id])` and `qjit_ht_mask(...)` — runtime loads from the context. HT identity is not embedded.

3. **HT payload access**: Byte offsets (`hc.offset`) are embedded as compile-time constants via `cc.c64(hc.offset)` (lines 9643, 9656, 9672, 9775, 9861, 10064, 10074). Two plans with different HT payload layouts produce different code.

4. **Filter expressions**: `QjitStepOp::filter` raw pointers are used only at LLVM IR generation time (`EmitShortCircuitFilter` at line 9793). They are never passed to the compiled function at runtime. The executor's `Run()` method receives only resolved sources, tuple sizes, and a params buffer — no plan or filter pointers.

5. **Guards and block-skip**: `guard_pos` indexes into `step.ops`, `op_index` indexes into `step.ops`, `block_skip_col` indexes into `step.cols`. All step-local positional indices.

### What CAN Be Cached (Code Cache)

| Component | Baked into machine code | Runtime parameter |
|-----------|------------------------|-------------------|
| Number of steps, their order | Yes | — |
| Per-step: ncols, dtype sequence | Yes | — |
| Per-step: op sequence (Filter/Probe) | Yes | — |
| Per-step: filter structure + type tags | Yes (parameterize-cache: constants via params buffer) | Filter constant values |
| Per-step: probe key count + dtypes | Yes | — |
| Per-step: HT payload layout (byte offsets) | Yes (`cc.c64(hc.offset)`) | — |
| Per-step: output column sources (src_col/ht_id+layout_col) | Yes | — |
| Per-step: sink kind (Result/HtBuild/Agg) | Yes | — |
| Per-step: guard structure | Yes | Guard min/max values (runtime via `qjit_ht_key0_min/max`) |
| Table data pointers | — | Yes (ctx->sources) |
| Which physical columns are bound | — | Yes (executor setup) |
| Hash table data | — | Yes (built fresh per query) |
| Row counts | — | Yes (ctx->sources[k].nrows) |

### What CANNOT Be Cached

1. **Hash table data** — Each query's HTs contain query-specific rows. Must rebuild per execution.
2. **Plans with different HT byte offsets** — `hc.offset` is a compile-time constant. Different column orders or dtype mixes produce different offsets → different code. The `AssignOffsets` function (`query_jit_steps.cpp:1180-1203`) lays out: prefix_bytes (validity, 8-aligned), then keys (i64 each), then VARCHAR payloads (16 bytes each), then INT32 payloads (4 bytes each). Plans with the same dtype set in the same order get the same offsets.
3. **Aggregation cell layout changes** — Different aggregate functions/columns → different emitted code.

### Build/Probe Side Annotation

The engine's physical plan annotates which join child is build vs. probe (`SimplestJoin::GetBuildChild()`). This is critical: it determines step ordering, HT layout, and probe key resolution (`query_jit_steps.cpp:737-825`). However, the structural template key captures this **implicitly** — the sink type, HT column types, and probe key locations fully encode the build/probe assignment. No explicit flag needed, but swapping build/probe sides invalidates the cache.

### Filter-Optional Templates (w/wo filter)

A filter compiles to inline LLVM IR (comparison instructions, string operations, branch targets). Without the filter, those basic blocks don't exist in the module.

**Approach: runtime `has_filter` flag in params buffer — ACCEPTED.**

When the flag is constant for the entire morsel (always 0 or always 1), the branch predictor sees a 100% one-outcome branch. A perfectly-predicted branch costs **~0 cycles per row** after warmup (~2-3 iterations). There is no misprediction penalty.

Implementation: add a `uint8_t filter_enabled` at a known offset in the params buffer. The compiled code loads it once per morsel (hoist outside row loop) and branches around the filter block:
```
filter_enabled = load_from_params(...)  // once per morsel, outside loop
for each row:
  if (filter_enabled) {
    // existing filter code (comparisons, LIKE, etc.)
    if (!passes) goto next_row;
  }
  // probes, output, etc.
```

For no-filter subqueries: set `filter_enabled = 0`. The branch is perfectly predicted, filter code is never executed, icache impact is marginal (dead code stays in binary but CPU never fetches after warmup).

The same flag must also guard filter evaluation in the ROF Phase 1 loop (`ir_to_llvm.cpp:9340-9347`) if ROF is active.

**Benefit**: doubles hit rate for patterns where same structure appears with and without filter. Example: `Scan → [maybe Filter(VARCHAR=$S)] → BuildHT(INT32 key, VARCHAR payload)` shares one compiled function across dimension lookup builds (filtered) and temp table builds (unfiltered).

### Hash Table Data Cache (conditionally recommended — small HTs only)

Measured HT finalize times (chain-linking only, excludes scan+filter+hash+append):

| Query | Largest HT | Finalize | All HTs total |
|-------|-----------|----------|---------------|
| 1a | 250 rows | 0.002 ms | ~0.005 ms |
| 13a | 71K rows | 1.2 ms | ~2.0 ms |
| 17a | 2.8M rows | 14.2 ms | ~17.6 ms |
| 25a | 337K rows | 0.4 ms | ~1.5 ms |

Total build cost (scan+hash+append+finalize) is 2-4x the finalize alone. For the 2.8M-row HT in 17a, total build is ~30-60 ms.

**Measured reuse across 57 cast_info JOB queries** (via `AQP_DUMP_CACHE_KEYS`):

| HT content | n rows | Appearances | Est. build cost each | Savable |
|---|---|---|---|---|
| movie_info_idx (info_type='votes') | 1,381,453 | 6× (30a/b/c, 31a/b/c) | ~45 ms | 5× = ~225 ms |
| aka_name (full scan) | 901,343 | 3× | ~28 ms | 2× = ~56 ms |
| cast_info (filtered subsets) | 2,832,555 | 3× (16b, 17a, 17e) | ~60 ms | 2× = ~120 ms |
| complete_cast (full scan) | 135,086 | 3× (20a/b/c) | ~5 ms | 2× = ~10 ms |
| **Total** | | | | **~410 ms** |

Total savable across 57 queries: ~410 ms. Average per query: ~7 ms.

**Memory cost is the blocker for large HTs**: a 2.8M-row HT with 64-byte rows = ~170 MB. Caching multiple large HTs could consume 500+ MB.

**Recommendation**: implement for **small HTs only** (< 100K rows, < 10 MB memory), which covers dimension table builds (complete_cast, info_type, company_type, etc.). These are cheap individually (~0.01-5 ms) but appear many times. Large HTs are not worth the memory cost.

**Cache key**: `(table_name, filter_predicates_serialized, key_cols, payload_cols)`. An exact match means identical HT data (assuming base data unchanged within the benchmark run).

**Memory budget (verified dataset sizes)**:

| Dataset | DB file | Total rows | Largest table |
|---------|---------|-----------|---------------|
| JOB (IMDB) | 2.0 GB | 74M | cast_info 36M |
| DSB SF50 | 25 GB | 366M | store_sales 144M |

JOB: all unique repeated HTs from 57 cast_info queries total ~914 MB. The entire IMDB dataset is 2 GB — caching all repeated HTs is feasible.

DSB SF50: a single `store_sales` HT could exceed 10 GB. Only dimension table HTs are cacheable.

**Threshold design:**
```cpp
struct HtDataCacheConfig {
    size_t max_total_bytes = 1ULL << 30;  // 1 GB total cache budget
    size_t max_per_ht_bytes = 256ULL << 20;  // 256 MB per individual HT
    size_t max_per_ht_rows = 5'000'000;  // 5M rows per HT
};
```
- 1 GB total: fits JOB comfortably. For DSB, only dimension tables cached.
- 256 MB per HT: allows movie_info_idx (74 MB), aka_name (57 MB), cast_info filtered up to ~3M rows (150 MB). Rejects full cast_info (2.1 GB).
- 5M row limit: practical sanity check for build time vs. memory tradeoff.
- LRU eviction when total budget exceeded.

**Not cacheable**:
- Temp table HTs: contents are query-instance-specific, no cross-query reuse
- HTs exceeding per-HT size limit

### Structural Template Key Design

Current parameterize-cache key includes concrete table/column indices:
```
S{title#5B|5.0.INT32|5.3.VARCHAR;FT:5.3:6$S;P:0(s0);K0:-1|s0|p0.1}
```

Proposed structural template key normalizes away identity:
```
S{_#_B|INT32|VARCHAR;FT:_.c0:6$S;P:0(s0);K0:-1|s0|p0.1}
H{1,8,28|INT32@8|VARCHAR@16}
```

Dtype sequence, operation sequence, HT layout (including byte offsets), filter structure, guard structure are preserved. Table names and column indices are erased.

### Cache Timing: Lazy / On-Demand

The template-cache uses the same timing as the existing parameterize-cache — compile on first encounter, reuse on subsequent matches:

1. Subquery arrives → `BuildExecutionSteps` → `QjitQueryPlan`
2. Serialize plan → compute structural cache key → `TryCacheLoad`
3. **Cache hit**: load compiled binary, build params buffer + bindings, execute
4. **Cache miss**: compile via LLVM, `TryCacheSave`, execute

No pre-compilation. The flow hooks into the same code path at `duckdb_adapter.cpp:4205`.

### Filter Merging: Two-Tier Lookup with Upgrade

We cannot pre-compile a filter placeholder without knowing the filter structure (which column, comparison type, dtype determine the generated instructions). The solution is a two-tier cache:

```
Cache structure: map<shape_key, map<filter_key, CompiledEntry>>
  filter_key = "none" for no-filter plans, or filter type template string
```

**Lookup algorithm:**
```
1. Compute shape_key (dtype signature, ops, HT layout — no table/column identity)
2. Compute filter_key (filter type template, or "none")
3. Exact lookup: cache[shape_key][filter_key]
   → HIT: use directly (filter_enabled=1 if filtered, build params)
4. If MISS and filter_key == "none":
   → Secondary lookup: any entry in cache[shape_key]
   → HIT: use that binary with filter_enabled=0 in params buffer
   → The filter code exists in the binary but never executes
     (perfectly-predicted branch, ~0 cycles/row)
5. If still MISS:
   → Compile:
     - Plan HAS filter: compile WITH if(filter_enabled) guard, cache as (shape, filter)
     - Plan has NO filter: compile WITHOUT filter code, cache as (shape, "none")
   → Execute
```

**Upgrade semantics**: once a filtered variant is compiled (with `if(filter_enabled)` guard), it serves all future no-filter requests for the same shape via step 4. The no-filter-only binary remains in cache but is less useful. No eviction needed.

**No need to "remove" the filter condition at runtime.** When `filter_enabled=0`:
- The branch is perfectly predicted (constant outcome for entire morsel)
- CPU instruction decoder sees the branch, predicts not-taken, skips filter code
- Zero per-row execution cost for dead filter code
- The `filter_enabled` flag is a `uint8_t` in the params buffer, loaded once per morsel (outside row loop)

**ROF (Reorder Filter) path**: the same `filter_enabled` flag must guard filter evaluation in ROF Phase 1 loop (`ir_to_llvm.cpp:9340-9347`), not just the main row loop.

### Implementation Design

**Phase 1: Positional Resolution Before Serialization**

Add `SerializeQjitPlanStructuralTemplate(plan)`:
1. Pre-resolve all filter expression attrs to positional indices (via `FindColIdx` against the step's schema)
2. Serialize step structure with dtypes only (no table/column names, no T/B distinction)
3. HT layout: serialize `(num_keys, col_dtypes, offsets, tuple_size)` — offsets are deterministic from `AssignOffsets`
4. Probe keys and outputs: already use positional `src_col` / `layout_col` in `QjitValueLoc`
5. Filter: serialize structure with type tags (same as `SerializeExprTemplate`), positionally resolved

**Phase 2: Split Cache Key from Column Binding**

```cpp
struct StructuralCacheResult {
    std::string shape_key;           // structural signature (ops, dtypes, HT layout)
    std::string filter_key;          // filter type template or "none"
    std::vector<uint8_t> params_buf; // filter constant values + filter_enabled flag
    // Per step: which (table, col) to bind to each positional slot
    // Used by executor to set up QjitTableView
    std::vector<std::vector<QjitColumnRef>> step_bindings;
    // Per step: source_is_temp (for executor binding path)
    std::vector<bool> source_is_temp;
};
```

On cache hit: load compiled function, executor uses `step_bindings` + `source_is_temp` to wire up `QjitTableView`, params buffer supplies filter constants + `filter_enabled` flag.

**Phase 3: Normalize HT Layout** (optional, for maximum hit rate)

`AssignOffsets` already has a deterministic layout: keys → VARCHAR payloads → INT32 payloads. As long as the dtype set and count are the same, offsets match. No normalization needed beyond what `AssignOffsets` provides — it already sorts by type.

### Temp vs Base Table Source: No Distinction in Compiled Code

Verified: `source_is_temp` is used only in serialization (`ir_to_llvm.cpp:5449, 5559`) and executor setup (`duckdb_adapter.cpp:4052-4059`). The compiled morsel body accesses `ctx->sources[step_idx].cols[i].data` identically for both. The executor uses `source_is_temp` to choose `ResolveTempSource` vs `ResolveSource` — a setup-time decision, not in the hot loop.

**Decision**: Drop T/B distinction from the structural template key. Temp and base sources with the same dtype signature share compiled code. The executor already has `QjitStep::source_is_temp` to choose the right binding path. This means `ScanTemp → BuildHT(INT32, VARCHAR)` reuses the same compiled function as `Scan(info_type) → Filter → BuildHT(INT32, VARCHAR)` (with `filter_enabled=0` for the temp case).

### Expected Reuse in JOB

JOB has 113 queries × ~3-5 sub-queries each ≈ 400-500 sub-queries.

**Pattern A: Dimension lookup builds** (most common)
- `Scan(small_table) → Filter(VARCHAR=$S) → BuildHT(key=INT32, payload=VARCHAR)`
- Applies to: company_type, info_type, kind_type, keyword, role_type, link_type
- All share one compiled function (same dtype signature, same filter structure)

**Pattern B: Large table builds**
- `Scan(cast_info/movie_info/...) → Filter(INT32=$I) → BuildHT(...)`
- ~10-15 structural variants based on key/payload dtype mix

**Pattern C: Probe chains**
- `Scan(table) → Probe(HT₀) → Probe(HT₁) → ... → Result`
- ~30-50 variants based on probe count + output signature

**Pattern D: Temp table builds**
- `ChunkGet(temp) → BuildHT(keys, payloads)` or `ChunkGet(temp) → Probe(HT) → Result`
- Shares compiled code with Pattern A/C when dtype matches (no T/B distinction)
- With filter-optional flag: `ScanTemp → BuildHT` reuses `Scan → Filter → BuildHT` code (filter_enabled=0)

**Estimated reduction**: 40-60 structural templates to cover all ~500 sub-queries (8-12x reduction vs. parameterize-cache). Higher than initial estimate because filter-optional merging and T/B merging increase hit rate.

### Key Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Current parameterize-cache key | `src/jit/ir_to_llvm.cpp` | 5555-5602 (`SerializeQjitPlanTemplate`) |
| Expression template serialization | `src/jit/ir_to_llvm.cpp` | 5494-5553 (`SerializeExprTemplate`) |
| Params buffer builder | `src/jit/ir_to_llvm.cpp` | 5659-5668 (`BuildParamsBuffer`) |
| CompileQuerySteps cache logic | `src/jit/ir_to_llvm.cpp` | 8881-8939 |
| Column resolution (FindColIdx) | `src/jit/ir_to_llvm.cpp` | 891-915 |
| Plan builder (Decompose) | `src/qjit/query_jit_steps.cpp` | 620-863 |
| HT offset assignment | `src/qjit/query_jit_steps.cpp` | 1180-1203 (`AssignOffsets`) |
| HT finalize (runtime) | `src/qjit/query_jit_runtime.cpp` | 118-270 |
| Compiled entry function | `src/jit/ir_to_llvm.cpp` | 10133-10171 |
| QjitQueryPlan structure | `include/qjit/query_jit_steps.h` | 170-191 |
| QjitStep structure | `include/qjit/query_jit_steps.h` | 123-162 |
| Query-JIT ABI (runtime) | `include/qjit/query_jit_abi.h` | 1-225 |

## Implementation Plan

## Key Code Locations (Reference)

### SimplestIR (AQPHub's intermediate representation)

| File | What |
|------|------|
| `third_party/IR_SQL_Converter/inc/simplest_ir.h` | All IR node class definitions (1633 lines) |
| `third_party/IR_SQL_Converter/src/duckdb_plan_to_ir.cpp` | DuckDB logical plan → SimplestIR |
| `third_party/IR_SQL_Converter/src/ir_to_sql.cpp` | SimplestIR → SQL string (reference for how IR is walked) |
| `third_party/IR_SQL_Converter/inc/cpp_interface.h` | Public API: ConvertDuckDBPlanToIR, ConvertIRToSQL |

## Repositories

- **AQPHub**: `/home/pei/Project/AQP_middleware`
- **DuckDB (patched)**: `/home/pei/Project/duckdb`
- **JOB queries**: `/home/pei/Project/benchmarks/imdb_job-postgres/queries/`
- **DSB queries**: `/home/pei/Project/benchmarks/dsb-postgres/code/tools/1_instance_out_aqp/1/`
- **TPC-H queries**: `/home/pei/Project/benchmarks/tpch-postgres/dbgen/out_50/queries/`
- **JOB schema**: `/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql`
- **DSB schema**: `/home/pei/Project/benchmarks/dsb-postgres/scripts/create_tables.sql`
- **TPC-H schema**: `/home/pei/Project/benchmarks/tpch-postgres/dbgen/dss.ddl`
- **DuckDB JOB database**: `/home/pei/Project/duckdb/measure/imdb.db`
- **DuckDB DSB database**: `/home/pei/Project/duckdb/measure/dsb_50.db`
- **DuckDB TPC-H database**: `/home/pei/Project/duckdb/measure/tpch_50.db`

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

### Step 1: Single query end-to-end

Take a simple JOB query that related to our changes (e.g., 1a — single join), run with
`./build_release/aqp_middleware ...` (check reference in measure/run_aqp.sh job). Compare result to DuckDB golden output.

### Step 2: Full JOB and DSB correctness

check measure/correctness_test_job_duckdb.sh and measure/correctness_test_dsb_duckdb.sh.

### Analysis scripts (measure/*.py)
CSV parser is: /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py
---

## Implementation Status: 


### Verification

- Build: passes (release)
- Correctness: all 113 JOB queries pass for all related configs
- CSV format: unchanged (no new columns), parseable by /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py
- No new print statements; existing LIKE debug trace guarded by `#ifndef NDEBUG`
- Check if need changes to measure/*.sh or measure/*.py
- Performance (top 15 worst LIKE queries): measure/measure_breakdown_time_aqp.sh job with correct configs 
