# jit-cache=template optimization

## Goal

## Analysis 

### Root cause

### Fix

## Implementation Plan

## Key Code Locations (Reference)

### AQPHub Core

| File | What |
|------|------|

### SimplestIR (AQPHub's intermediate representation)

| File | What |
|------|------|
| `third_party/IR_SQL_Converter/inc/simplest_ir.h` | All IR node class definitions (1633 lines) |
| `third_party/IR_SQL_Converter/src/duckdb_plan_to_ir.cpp` | DuckDB logical plan → SimplestIR |
| `third_party/IR_SQL_Converter/src/ir_to_sql.cpp` | SimplestIR → SQL string (reference for how IR is walked) |
| `third_party/IR_SQL_Converter/inc/cpp_interface.h` | Public API: ConvertDuckDBPlanToIR, ConvertIRToSQL |

## Repositories

- **AQPHub**: `/home/pei/Project/AQPHub` (branch: `topdown_fix`)
- **DuckDB (patched)**: `/home/pei/Project/duckdb`
- **Lingo-db**: `/home/pei/Project/lingo-db`
- **JOB queries**: `/home/pei/Project/benchmarks/imdb_job-postgres/queries/`
- **JOB schema**: `/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql`
- **DuckDB database**: `/home/pei/Project/duckdb/measure/imdb.db`
- **LingoDB CSV dir**: `/home/pei/Project/benchmarks/imdb_job-postgres/lingo_db_csv`

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

### Step 1: Verify optimizer-skip works

Build a hardcoded minimal MLIR string (scan one table, project one column),
feed it to LingoDB with `queryOptimizer = nullptr`, verify it executes
without crash and returns correct results.

### Step 2: Single sub-query end-to-end

Take a simple JOB query (e.g., 1a — single join), run with
`--engine=lingo-db-runtime`, verify the IR-to-RelAlg converter produces
valid MLIR that LingoDB can lower and execute. Compare result to DuckDB
golden output.

### Step 3: Multi-iteration query

Run a query with multiple sub-queries (e.g., 13a with node-based split),
verify temp table creation, DuckDB helper sync, and multi-iteration
execution all work correctly.

### Step 4: Full JOB correctness

Run all 113 JOB queries, compare against golden files:
- `measure/duckdb_job_no-split_golden.txt`
- `measure/duckdb_job_node-based_golden.txt`

### Analysis scripts (measure/*.py)
- `tune_per_subquery.py [split]` — pick best config per (query, sub-query)
- `show_all_configs.py [split]` — summary table across all configs
- `find_top_queries.py [path] [--top=N]` — rank queries by slowest median

---

## Implementation Status: DONE

All changes in `src/jit/ir_to_llvm.cpp` (+177, -93 lines):

1. **BuildParamsFromExpr** (~line 5515): LIKE patterns now allocate
   literal/segments into params buffer matching what codegen expects:
   - MULTI_SEGMENT: one AllocString per segment
   - CONTAINS: AllocString(literal) + AllocI32(first_char)
   - PREFIX/SUFFIX: AllocString(literal)
   - EQUALITY/COMPLEX: AllocString(raw pattern)

2. **EmitVarConst pat_ptr** (~line 1341): Skip raw pattern EmitParamString
   for LIKE in template mode; each specialized path does its own loads.

3. **LIKE codegen** (~line 1394): Removed blanket `aqp_like_match` fallback.
   Each LIKE kind now handles template mode with parameterized loads:
   - EQUALITY: inline EmitParamString → aqp_str_eq
   - PREFIX/SUFFIX: EmitParamString(literal) → inline memcmp
   - CONTAINS: EmitParamString + EmitParamI32 → inline memchr+memcmp loop
     (needle_len==1 check is now runtime branch)
   - MULTI_SEGMENT: alloca arrays filled from EmitParamString per segment →
     aqp_like_match_segments
   - COMPLEX: inline EmitParamString → aqp_like_match

### Verification

- Build: passes (release)
- Correctness: all 113 JOB queries pass for all template-mode configs:
  - `query-jit + single-run-template` (node-based, no-split)
  - `query-jit + single-run-template + spec-jit=recompile` (node-based)
  - `expr-jit + single-run-template` (node-based)
  - `operator-jit + single-run-template` (node-based)
  - `pipeline-jit + single-run-template` (node-based)
- CSV format: unchanged (no new columns), parseable by plot_middleware_jit.py
  (verified: 113 queries, all 17 columns present)
- No new print statements; existing LIKE debug trace guarded by `#ifndef NDEBUG`
- No changes to measure/*.sh or measure/*.py (not needed)
- Performance (top 15 worst LIKE queries): template/strict ratio 1.018x
  (was 1.30x before fix)
