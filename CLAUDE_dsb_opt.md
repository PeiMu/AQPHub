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
- `hyperfine_{job,dsb}.sh`, `measure_{job,dsb}.sh`, `run_{job,dsb}.sh`, `measure_breakdown_time_{job,dsb}.sh` — engine-agnostic helpers (accept engine as parameter)

### PostgreSQL query-jit
- Only `query` jit-level supported (no expr/operator/pipeline-jit)
- Splits: `none`, `node-based`, `topdown`
- No `payload_prune`, `prefetch`, `batch_probe` flags
- Golden files: `pg_job_{no-split,node-based}_golden.txt`, `pg_dsb_{no-split,node-based}_golden.txt`
- `tune_per_subquery.py` accepts `--engine=postgresql` and `--bench=dsb`

## Verification Workflow

### Single query end-to-end

Take a simple JOB query (e.g., 1a — single join), run with
`./build_release/aqp_middleware ...` (check reference in measure/run_job.sh). Compare result to golden output.

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
