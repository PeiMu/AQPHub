# LingoDB Analysis & Pipeline-JIT Findings

## LingoDB Compilation Architecture

LingoDB uses a 7-stage MLIR-based compilation pipeline:

```
SQL → RelAlg → SubOp → DB → Standard → LLVM IR → Native Code
```

Full optimizer pipeline has ~18 passes including pushdown, unnesting, column folding, transitive equality expansion (`src/compiler/Dialect/RelAlg/Passes.cpp:17-49`).

### How LingoDB Compiles Pipelines

LingoDB compiles **build and probe as separate execution steps within a single LLVM function**:

1. `SplitIntoExecutionSteps` pass (`src/compiler/Dialect/SubOperator/Transforms/SplitIntoExecutionSteps.cpp:14-224`) splits the query into pipeline stages via state read/write dependency analysis. Build and probe become separate `ExecutionStepOp`s.
2. `handleExecutionStepCPU()` (`src/compiler/Conversion/SubOpToControlFlow/SubOpToControlFlow.cpp:4334-4365`) inlines each `ExecutionStepOp` sequentially into the **parent function** — no separate functions per pipeline stage.
3. A single `IRMapping` is used across all steps (`SubOpToControlFlow.cpp:4483-4528`), enabling cross-step optimization.

This means the entire probe pipeline (scan → filter → probe₁ → probe₂ → ... → probeₙ → projection → output) is compiled as **one function**.

## LingoDB Limitations

- **INSERT INTO ... SELECT ... is NOT supported**: `sql_analyzer.cpp:982` explicitly requires `BOUND_VALUES` producer type only.
- **CREATE TABLE AS SELECT is NOT supported**: Parser syntax error.
- **No EXPLAIN command**: Zero references in codebase. Cost estimation requires running frontend+optimizer (`ExecutionMode::NONE` skips backend).
- **`getNumRows()` is real cardinality** (actual row count from in-memory storage at `LingoDBTable.h:70-72`), not estimated.

## Pipeline-JIT: What to Learn from LingoDB

### Why Current DuckDB Probe Dispatch Is Disabled

The current `CompileFilterProbeProjectFusion` (Level 3 fusion-jit) generates JIT code for a **single hash probe**, but the DuckDB executor still manages the pipeline between operators. The overhead of calling into/out of JIT per-tuple negates the benefit.

- `physical_hash_join.cpp:1097`: `if (false && jit ...)` — currently disabled.
- Filter JIT dispatch (`physical_filter.cpp:52-98`) is still active because it stays within one operator.

### LingoDB Optimizations to Apply

1. **Late materialization (PullGatherUp pass)**: Only materializes columns when actually needed, not at scan time. Reduces memory traffic through the pipeline.

2. **Hash table specialization (SpecializeSubOp pass)**: Generates specialized hash lookup code based on key types and count, avoiding generic dispatch overhead.

3. **Column folding (FoldColumns pass)**: Eliminates duplicate column computations across the pipeline, reducing redundant work.

4. **Single function for entire probe pipeline**: All probe stages are inlined into one LLVM function. This is the critical architectural difference — instead of JIT-compiling one probe at a time, compile the entire pipeline (scan → filter → probe₁ → probe₂ → ... → probeₙ → projection → aggregate/output) as a single function.

### Pipeline-JIT Design

Extend fusion-jit (Level 3) to handle **multiple sequential hash probes** in one compiled function:

- **DuckDB builds hash tables natively** via `PhysicalHashJoin::Sink()`, then `PopulateAQPJITView()` exposes HT internals (entries, bitmask, layout offsets) via `AQPJoinHTView`. JIT code probes directly.
- Pipeline-jit compiles the **entire probe chain** as a single function, eliminating per-operator dispatch overhead.
- Pipeline-jit and query-jit are the same thing (no benefit from query-level over pipeline-level), confirmed by lingo-db which also operates at the pipeline level only.
- Pipeline-jit is **completely unrelated to the kernel path** (`PipelineKernelPlan`/`SubQueryPlan`). It is a new JIT level for the DuckDB execution path with `kernel_path=NONE`.

### Confirmation Strategy

1. Implement pipeline-jit by extending fusion-jit to handle multiple sequential probes in one compiled function.
2. Apply the lingo-db optimizations above (late materialization, hash table specialization, column folding, single-function compilation).
3. Test none-split with DuckDB adapter + pipeline-jit.
4. Compare against none-split with lingo-db (which already compiles natively at pipeline level).
5. If DuckDB+pipeline-jit matches lingo-db's execution time, the approach is validated.

## Key Source Code References

| File | Lines | Description |
|------|-------|-------------|
| `lingo-db/src/compiler/Dialect/RelAlg/Passes.cpp` | 17-49 | Full optimizer pipeline (18 passes) |
| `lingo-db/src/compiler/Dialect/RelAlg/Transforms/OptimizeJoinOrder.cpp` | 120-127 | DPHyp + GOO join order optimizer |
| `lingo-db/src/compiler/Dialect/SubOperator/Transforms/SplitIntoExecutionSteps.cpp` | 14-224 | Pipeline splitting via state dependency analysis |
| `lingo-db/src/compiler/Conversion/SubOpToControlFlow/SubOpToControlFlow.cpp` | 4334-4365 | Single-function pipeline compilation |
| `lingo-db/src/compiler/frontend/sql_analyzer.cpp` | 982 | INSERT INTO SELECT limitation |
| `lingo-db/include/lingodb/runtime/storage/LingoDBTable.h` | 70-72 | Real cardinality via `getNumRows()` |
| `duckdb/src/execution/operator/join/physical_hash_join.cpp` | 1096-1129 | Disabled JIT probe dispatch |
| `duckdb/src/execution/join_hashtable.cpp` | 1549-1560 | `PopulateAQPJITView()` |
| `AQP_middleware/src/adapters/duckdb_adapter.cpp` | 2876-2904 | Level 3 CompilePipeline (gated on kernel_path) |
| `AQP_middleware/src/adapters/duckdb_adapter.cpp` | 3082-3600 | Level 3 CompileFilterProbeProjectFusion |
| `AQP_middleware/include/jit/aqp_jit_abi.h` | 135-137 | `AQPPipelineFn` signature |
