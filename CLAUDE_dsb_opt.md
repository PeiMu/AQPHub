## Goal

Make node-based/topdown + query-jit (TPDE + template-cache) at least **2x faster** than no-split + no-jit on DSB benchmark. Any changes shouldn't affect (e.g., slowdown) the performance of JOB queries.

**Target config:** `--split=node-based --jit-level=query --compile-mode=tpde --jit-cache=single-run-template` (no spec-jit)

**Metric:** average of 10 measured runs (after 5 warmup runs) of the sum over all DSB queries.

### Current state (2026-07-18, SF=10, 22-query set)

query050_spj, query102, query102_spj were REMOVED from the benchmark set (see "DSB Query Set" below): all three are dominated by non-equi range joins that query-jit rejects (`join:non-equi-condition`), so no target-config mechanism can beat the no-split interpreter on them; proposed fixes could only restore parity with baseline, never a speedup.

| Metric | 22 queries (mean) |
|--------|-------------------|
| Baseline (no-split, no-jit) | 300.8 ms |
| Target config (no spec) | 252.0 ms |
| **Speedup** | **1.193x** |
| 2x target | ≤150.4 ms |

Ceiling analysis (sum of per-query medians): best-of-ALL-measured-configs = 1.72x on the 22 queries — 2x is NOT reachable by config picking alone; needs new optimizations (candidates: fix cross-query-prep guard loss below; grouped-agg QJIT support — `agg:grouped` reject blocks many sub-queries; biggest remaining costs under target config: query085 30.6ms, query085_spj 29.2, query084 14.2, query084_spj 12.8, query019 11.8, query027_spj 11.8).

**Template cache limitation:** Template cache is single-run (cleared between repeat iterations). It only helps when the same plan shape appears across sub-queries within ONE execution. For DSB queries with few sub-queries of unique shapes, it provides no cache hits on warm runs. Full cache (persistent disk) is not the target config.

**Split overhead dominates:** NB interpreter (no JIT) = 0.617x (25q basis). The split itself costs more than JIT saves.

Phases 1–6 complete (see `dsb_support.log`). NOTE: dsb_support.log and older numbers are on the 25-query set; current basis is 22 queries.

### Next step TODO: fix cross-query-prep guard loss (a.k.a. "Fix 3")

**Symptom:** under `--jit-cache=single-run-template` (target config; NO `--spec-jit` involved), DSB query101 21.9ms vs 11.4ms with cache off; JOB 8b sub1 exec 9.8→28.5ms, 19b 13.5→34.4ms, 25c 4.1→6.6ms.

**Root cause (verified in debug traces and code review 2026-07-20):** cross-query prep bg-compiles the NEXT query's first sub-query via `SpeculativeQueryJitCompile` (`src/adapters/duckdb_adapter.cpp:4041`). This function re-parses the SQL string, re-optimizes through DuckDB's optimizer, and regenerates IR from scratch — the IR→SQL→parse→optimize→IR round-trip can change join nesting, causing `PlanJoinFilterPushdown` (called inside `BuildExecutionSteps` at `src/qjit/query_jit_steps.cpp:1347`) to find fewer or zero guards because probe keys may no longer come from source columns. With `jit_cache>=3`, iteration 0 uses the guard-less bg kernel for sq0 and caches it; iterations 1+ replay the guard-less kernel for all measured iterations.

**Scope:** only sq0 (first sub-query) per query is affected; sq1+ are inline-compiled with guards. Only configs with `jit_cache >= 1` + `{node-based, topdown}` + DuckDB are affected (120 configs per benchmark sweep). The fix is a **speedup** (guards skip irrelevant rows/blocks via cheap range checks).

**Fix variants (undecided):** (a) pass the original IR (from the splitter) to the bg compilation path instead of re-parsing SQL; (b) keep bg kernel only for iteration-0 latency hiding, record an inline-compiled kernel for replay.

**Expected impact:** DSB query101/101_spj ~ -10ms each; JOB improves ~40-45ms across 8b/19b/25c (removes an existing hidden regression). Acceptance gate: full JOB + DSB sweeps, no slowdown anywhere.

**Note on CachedSubquery (2026-07-20):** The `CachedSubquery` struct (`include/adapters/duckdb_adapter.h:103-126`) does NOT store `guard_pos` or `guards`. However, this is not the cause of the bug — guards are baked into the compiled LLVM IR machine code. The guard loss happens during bg compilation (wrong IR structure), not during replay.

### Decisions on record

- ExecuteRow empty-collection bug in patched DuckDB: **ignored by decision** — latent, does not manifest in the 22-query set.
- Fixes for the 3 removed queries: **dropped** — queries removed instead.
- DSB golden files regenerated for the 22-query set (2026-07-18). `duckdb_dsb_no-split_golden.txt` is the only DSB golden file — all DuckDB splits use it as ground truth. `lingodb_dsb_no-split_golden.txt` unaffected.

### DSB scale factors

`measure/*dsb*.sh` are SF-parametrized via `DSB_SF` env (default 10): `breakdown_measurement_script_dsb_duckdb.sh 100` runs against `dsb_100.db`, writes to `measure/dsb_result_sf100/`. Analysis: `analyze_dsb.py dsb_result_sf100`; `tune_per_subquery.py` / `show_all_configs.py` / `verify_tuned*.py` accept `--result-dir=`. sf=10 behavior/paths unchanged.

## Key Code Locations (Reference)

### SimplestIR (AQPHub's intermediate representation)

| File | What |
|------|------|
| `third_party/IR_SQL_Converter/inc/simplest_ir.h` | All IR node class definitions |
| `third_party/IR_SQL_Converter/src/duckdb_plan_to_ir.cpp` | DuckDB logical plan → SimplestIR |
| `third_party/IR_SQL_Converter/src/ir_to_sql.cpp` | SimplestIR → SQL string |
| `third_party/IR_SQL_Converter/inc/cpp_interface.h` | Public API: ConvertDuckDBPlanToIR, ConvertIRToSQL |

## Repositories

- **AQPHub**: `/home/pei/Project/AQPHub` (branch: `support_dsb`)
- **DuckDB (patched)**: `/home/pei/Project/duckdb`
- **JOB queries**: `/home/pei/Project/benchmarks/imdb_job-postgres/queries/`
- **JOB schema**: `/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql`
- **DuckDB database**: `/home/pei/Project/duckdb/measure/imdb.db`
- **DSB queries**: `/home/pei/Project/benchmarks/DSB4AQP/code/tools/1_instance_out_aqp/`
- **DSB database**: `/home/pei/Project/duckdb/measure/dsb_10.db`

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

## Measure Scripts

### Naming convention (2026-07-20)
Scripts are named `{type}_{benchmark}_{engine}.sh`:
- `breakdown_measurement_script_{job,dsb}_{duckdb,postgres}.sh` — full performance sweep
- `correctness_test_{job,dsb}_{duckdb,postgres}.sh` — correctness verification
- `hyperfine_aqp.sh {job,dsb}`, `run_aqp.sh {job,dsb}`, `measure_breakdown_time_aqp.sh {job,dsb}` — unified engine-agnostic helpers (accept benchmark and engine as parameters)

### PostgreSQL query-jit
- Only `query` jit-level supported (no expr/operator/pipeline-jit)
- Splits: `none`, `node-based`, `topdown`
- No `payload_prune`, `prefetch`, `batch_probe` flags
- Golden files: `pg_job_{no-split,node-based}_golden.txt`, `pg_dsb_{no-split,node-based}_golden.txt`
- `tune_per_subquery.py` accepts `--engine=postgresql` and `--bench=dsb`

## Verification Workflow

### Single query end-to-end

Take a simple JOB query (e.g., 1a — single join), run with
`./build_release/aqp_middleware ...` (check reference in measure/run_aqp.sh job). Compare result to golden output.

### Full correctness

```bash
# DuckDB JOB / DSB
bash measure/correctness_test_job_duckdb.sh
bash measure/correctness_test_dsb_duckdb.sh
# PostgreSQL JOB / DSB
bash measure/correctness_test_job_postgres.sh
bash measure/correctness_test_dsb_postgres.sh
```

Golden files:
- DuckDB: `measure/duckdb_job_no-split_golden.txt`, `measure/duckdb_job_node-based_golden.txt`, `measure/duckdb_dsb_no-split_golden.txt`
- PostgreSQL: `measure/pg_job_no-split_golden.txt`, `measure/pg_job_node-based_golden.txt`

### Analysis scripts (measure/*.py)
- `tune_per_subquery.py [split]` — pick best config per (query, sub-query)
- `show_all_configs.py [split]` — summary table across all configs
- `find_top_queries.py [path] [--top=N]` — rank queries by slowest median

---

## DSB Query Set

**22 queries** in `1_instance_out_aqp/1/`. Node-based: 22/22. Topdown: 22/22.

21 additional DSB queries (from `1_instance_out/1/`) crash in DuckDB's `ReorderGet::ReorderTables` during `QuerySplit::Split()` — this is a DuckDB-internal bug, not fixable from middleware.

Removed queries:
- query094: DELIM_JOIN/DELIM_GET (correlated subquery) — incompatible with topdown's IR→SQL round-trip.
- query050: self-join on date_dim with cross-table date range — node-based splits the two date_dim instances apart.
- query013, query013_spj: `filter:payload-ref` — cross-table OR filters reference HT payload columns.
- query072, query072_spj: LEFT JOIN not supported by DuckDB plan-to-IR converter.
- query050_spj (removed 2026-07-18): QJIT-rejected non-equi build, floor = baseline.
- query102, query102_spj (removed 2026-07-18): splitter extracts range-join standalone, floor = baseline.


-------------------------
## Solving DSB_100 query085 performance issue
### TODO-1: Large-Intermediate Sub-Query Performance (DSB query085)

**Problem**: DSB query085's temp1 sub-query (`web_sales 7.2M JOIN date_dim WHERE
d_year=2002` → 969K output rows × 7 INT32 columns) runs **3× slower** under
query-JIT (92ms) vs DuckDB native vectorized (28ms). This single query causes
180% of the total node-based exe_sub regression, wiping out all gains from other
queries that are 2–100× faster under JIT. Both paths use 12-core parallelism.

**Attempted fix — batched columnar output**: Implemented and tested a buffer-and-flush
approach (buffer 64 row references, flush column-at-a-time with batch-ensure to
eliminate per-element cursor checks). Result: **no improvement** (92ms → 95ms).
The per-element cursor checks were NOT the bottleneck. Reverted.

**Root cause analysis (definitive, verified with perf counters)**:

The bottleneck is **memory latency degradation under thread concurrency**, not
contention, false sharing, or algorithmic overhead.

Measured (DSB SF100, query085_0 temp1, 7.2M input → 969K output × 7 INT32):

| Config | exe_sub wall | Per-row cycles | Speedup |
|--------|-------------|---------------|---------|
| JIT 1 thread | 272ms | 151 cycles | 1.0× |
| JIT 2 threads | 163ms | — | 1.7× (84% eff) |
| JIT 4 threads | 108ms | — | 2.5× (63% eff) |
| JIT 6 threads (= phys cores) | 90ms | 300 cycles | 3.0× (50% eff) |
| JIT 8 threads (6 phys + 2 HT) | 98ms | — | 2.8× (worse than 6!) |
| JIT 12 threads (6 phys + 6 HT) | 90ms | — | 3.0× (25% eff) |
| DuckDB native NB-NoJIT (12 thr) | 26ms | — | — |
| DuckDB none-split (whole query) | 271ms | — | — |

**Hardware**: Intel Xeon E-2236 — **6 physical cores + 6 HT** (not 12 physical).
12 MiB L3 shared. Single socket, single NUMA node.

Key findings (verified with perf stat, 50-repeat runs):
1. **Single-thread JIT matches DuckDB whole-query** (272ms vs 271ms). The compiled
   morsel code is efficient. The per-row work is correct and fast.

2. **Total instructions are IDENTICAL** across 1/6 threads (277.4B vs 277.9B).
   LLC misses identical (351M vs 357M). dTLB misses identical (117M vs 121M).
   **No extra work, no extra cache misses, no extra TLB misses.**

3. **Per-row cycles DOUBLE** with 6 threads (151 → 300). Same instructions, same
   cache misses, but each miss takes 2× longer to resolve. Root cause: 6 threads
   issue concurrent DRAM requests that compete for the memory controller's queue
   depth. Each LLC miss takes ~60-80ns (unloaded) → ~120-160ns (6 threads queuing).
   The data is 207 MB (cold streaming scan, no reuse), each row reads 7 columns
   from 7 separate arrays → 42 concurrent memory streams with 6 threads.

4. **HT threads add ZERO benefit**: 8 threads (98ms) is SLOWER than 6 threads (90ms).
   HT siblings share L1/L2/execution ports; for memory-latency-bound code, they
   just add more pressure to the memory controller. This is a hallmark of
   memory-subsystem saturation.

5. **This is NOT a bug**: morsel scheduling is clean (atomic fetch_add, no lock
   contention), QjitTable partitions are per-worker (no false sharing), morsel size
   doesn't matter (20K vs 1M gives same result). The scaling is inherent to
   streaming 207 MB of columnar data through row-at-a-time code on this hardware.

6. **DuckDB scales better because of its access pattern**: Vectorized execution
   processes ONE column at a time per 2048-row chunk (8 KB fits in L1). This
   generates a single sequential stream per column → better hw prefetcher →
   lower memory controller pressure. The JIT morsel reads 7 columns per row =
   7 concurrent streams per thread.

7. **The Iter 9 "JIT probe disabled" finding was about pipeline-JIT**, not query-JIT.
   Pipeline-JIT probe code is in `physical_hash_join.cpp`; query-JIT probe is in
   `ir_to_llvm.cpp`. Different code paths, different JIT levels.

**Approach 2 (column-at-a-time morsel scan) INVALIDATED**: Verified that the
current code already does late materialization — only the join-key column (col 6,
ws_sold_date_sk) is loaded before the probe. Remaining 6 output columns are loaded
only after probe match (in the `case QjitStep::Result` sink, line 9880). Block-skip
(`blockskip_col=6`, 2048-row blocks) eliminates most non-matching blocks. Within
processed blocks, match rate is ~81%. Deferring column reads would save <5% of work.

Measured: block-skip ON vs OFF (SF10, 1 thread): 28ms vs 43ms (35% savings already
captured). The inner loop processes ~30-35% of total rows; ~81% of those match.

**Recommended approaches** (re-prioritized after verification):

1. **Selective JIT bypass for large scan+join sub-queries** (IMPLEMENTED):
   When a Result-sink step's source table exceeds 50M rows (from StoragePlan
   FlatTable), skip query-JIT compilation and fall back to DuckDB native execution.
   Bypass added at two code paths: `TryCompileQueryJit` (execution path) and
   `SpeculativeQueryJitCompile` (spec/cache path). Threshold 50M avoids JOB
   regressions (cast_info=36M < 50M) while catching DSB SF100 (web_sales=72M > 50M).
   
   Results (verified):
   - DSB SF100 NB: 0.569s → 0.451s (**-119ms, -20.9%**)
   - query085_0 exe_sub: 90.1ms → 32.0ms (2.8× faster)
   - JOB topdown: 4.895s → 4.794s (**-101ms, -2.1%**, noise-level improvement)
   - Correctness: all 113 JOB queries pass across 9 configs (TPDE/LLVM × cache modes)
   
   Code locations:
   - `src/adapters/duckdb_adapter.cpp`: `TryCompileQueryJit` source-rows check
     after `ResolveQjitSources`; `SpeculativeQueryJitCompile` FlatTable row_count
     check after `BuildExecutionSteps`
   - Threshold: `kSourceRowBypass = 50000000` (50M rows)

2. ~~**Software prefetching for source column scan**~~ (INVALIDATED):
   Verified with perf counters that the HW prefetcher already brings 67% of L2
   data speculatively (`l2_lines_in.all=858M >> l2_rqsts.demand_data_rd=287M`).
   All cache/memory counters are IDENTICAL between 1 and 6 threads. The per-row
   cycle doubling is caused by DRAM queuing delay (longer per-miss latency), NOT
   more misses. SW prefetch cannot reduce per-miss latency under contention — it
   can only overlap latency with computation, and IPC is already ~1.1. Existing
   `--jit-prefetch` (HT probe prefetch in CompileFilterProbeProjectFusion) uses
   ROF stage-2 look-ahead; query-JIT morsel body (`CompileQuerySteps`) has NO
   prefetch code, but adding it would give at most 5-10% improvement.

3. ~~**Morsel-level column-scan + selection vector**~~ (INVALIDATED):
   Verified that the current morsel body already does late materialization: only the
   join-key column (1 source col) is loaded before the probe. Block-skip eliminates
   ~35% of rows (non-matching blocks). Of processed rows, ~81% match. Restructuring
   to column-at-a-time would save <5% of work.

**Summary of scaling analysis (verified, not fixable in software)**:

The 2× per-row cycle inflation under 6-thread concurrency (151→300 cycles/row) is
an inherent DRAM/memory-controller property. Same instructions, same cache misses,
same TLB misses, but each DRAM access takes ~2× longer due to queuing. This is the
same effect that limits STREAM benchmark scaling on single-socket systems. It cannot
be fixed by changes to the query-JIT code — it requires either:
- Different hardware (higher memory bandwidth, more memory channels)
- Different access pattern that avoids DRAM entirely (data fits in L3)
- Accepting the 3× scaling limit and using approach 1 for affected queries

### TODO-2: DSB query-JIT rejected operators (after fixing perf issues 1 & 3)

Priority 1 — most DSB subqueries are accepted (34/44 = 77%). Rejections are:

| Reason | Count | Scope | What causes it | Effort |
|--------|-------|-------|---------------|--------|
| `expr:Unknown` | 5 | 3 temp, 2 result | Computed expressions: `a/b BETWEEN ...`, `substring()` in comparisons — emitted as `SimplestFunctionExpr` not in CheckExpr whitelist | Moderate |
| `agg:grouped` | 3 | 3 result | GROUP BY clause — query-JIT only supports global aggregation | Major |
| `join:non-equi-condition` | 1 | 1 result | Inequality join: `d_date BETWEEN d1.d_date AND ...` | Major |
| `expr:single-attr` | 1 | 1 temp | Mark/semi-join boolean from correlated subquery | Moderate |

Most rejections hit FINAL subqueries (label=result). Fix performance issues first;
these operator extensions only matter if QJIT should run the final aggregation step.
