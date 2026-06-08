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

1. **Late materialization (PullGatherUp pass)** → mapped to `--jit-payload-prune` (already implemented). Only materializes columns when actually needed. Reduces memory traffic through the pipeline.

2. **Hash table specialization (SpecializeSubOp pass)** → mapped to `--jit-skip-hash-cmp` (plumbed, codegen TODO). For integer keys, lingo-db skips hash comparison during lookup because direct key comparison is cheaper (`SpecializeSubOpPass.cpp:116-117`). Our implementation: skip salt/hash check in `CompileFilterProbeProjectFusion` when all probe keys are integer types.

3. **Column folding (FoldColumns pass)** → implicit in codegen. Our `CompileFilterProbeProjectFusion` already only emits loads for columns referenced by the projection. No separate flag needed.

4. **Single function for entire probe pipeline** → `--jit-level=pipeline`. All probe stages are inlined into one LLVM function. This is the critical architectural difference — instead of JIT-compiling one probe at a time, compile the entire pipeline (scan → filter → probe₁ → probe₂ → ... → probeₙ → projection → aggregate/output) as a single function.

### LingoDB vs AQP Middleware Optimization Mapping

| LingoDB optimization | AQP middleware equivalent | Status |
|---|---|---|
| PullGatherUp (late materialization) | `--jit-payload-prune` | Implemented |
| FoldColumns (eliminate redundant cols) | Implicit in probe codegen | Implemented |
| SpecializeSubOp (skip hash cmp for int keys) | `--jit-skip-hash-cmp` | Implemented (codegen at `ir_to_llvm.cpp:5810-5824`) |
| Single-function pipeline compilation | `--jit-level=pipeline` | Single-probe done; multi-probe TODO |
| Software prefetch | `--jit-prefetch` | Implemented (AQP-only, lingo-db doesn't do this) |
| Batch/ROF probe | `--jit-batch-probe` | Implemented (AQP-only, lingo-db doesn't do this) |
| LLVM passes (InstCombine, Reassociate, GVN, SimplifyCFG) | Same 4 passes | Identical |

### LingoDB Join Ordering: DPHyp (produces bushy plans)

LingoDB uses **DPHyp** (`OptimizeJoinOrder.cpp:121`) for join ordering.
DPHyp is a dynamic programming algorithm that searches over both left-deep
and bushy plan shapes. For > 1000 subgraphs, falls back to GOO (greedy
operator ordering). The `SplitIntoExecutionSteps` pass
(`SplitIntoExecutionSteps.cpp`) then splits the plan into pipeline stages by
**read/write state dependency analysis** (topological sort), not by tree shape.
This means lingo-db's probe pipeline compilation is shape-agnostic — it
compiles all probe lookups sequentially in one function regardless of whether
the plan is left-deep or bushy.

### DuckDB Probe Pipeline Shape (JOB benchmark, none-split, verified)

DuckDB's physical plans for JOB have this probe pipeline HJ distribution:
- **90/113** queries: 1 HJ on probe pipeline (single-probe JIT sufficient)
- **21/113** queries: 2 HJ on probe pipeline (multi-probe fusion target)
- **2/113** queries: 3 HJ on probe pipeline (5a, 5c)

Multi-probe target queries (23 total): 1b, 1c, 1d, 3a, 3c, 5a, 5b, 5c,
6b, 6d, 6f, 11c, 11d, 12a, 12b, 12c, 13a, 13b, 13c, 13d, 21a, 21c, 27c.

Important: "bushy" in the physical plan tree (73/113 queries have HJ on
child[0] of another HJ) does NOT mean multiple HJ on the probe pipeline.
The build side (child[1]) runs in a separate pipeline. What matters for
multi-probe fusion is consecutive HJ operators in the same probe pipeline.

### Dead Configs Removed

- **`fusion_build`**: Was a shell-script-only parameter; `CompileFilterHashBuildFusion` was already removed. Cleaned from all scripts.
- **`fusion_probe`**: Probe fusion is now always-on when pipeline-jit or kernel-path=pipeline is active. No toggle needed.
- **`inline_hash`**: The active probe path (`CompileFilterProbeProjectFusion`) always inlines MurmurHash64 (matching DuckDB's `JoinHashTable`). The old FNV-1a `EmitHash`/`EmitInlineFNV1a` had zero call sites — dead code, removed.

### Pipeline-JIT Design

New JIT level: `--jit-level=pipeline` (`AQP_JIT_PIPELINE_JIT`, bit 6). Implies expr + operator levels. Mutually exclusive with `--kernel-path` (validated at parse time).

- **DuckDB builds hash tables natively** via `PhysicalHashJoin::Sink()`, then `PopulateAQPJITView()` exposes HT internals (entries, bitmask, layout offsets) via `AQPJoinHTView`. JIT code probes directly.
- Pipeline-jit compiles the **entire probe chain** as a single function, eliminating per-operator dispatch overhead.
- Pipeline-jit and query-jit are the same thing (no benefit from query-level over pipeline-level), confirmed by lingo-db which also operates at the pipeline level only.
- Pipeline-jit activates probe fusion in the **DuckDB execution path** (not through PipelineKernel). Uses the same `CompileFilterProbeProjectFusion` codegen as kernel-path=pipeline, but extended for multi-probe chains.
- Extra-jit-flags (`payload_prune`, `prefetch`, `batch_probe`, `skip_hash_cmp`) work under both `--jit-level=pipeline` and `--kernel-path=pipeline`.

### Confirmation Strategy

1. ~~Implement pipeline-jit by extending fusion-jit to handle multiple sequential probes in one compiled function.~~ **Done for single-probe**; multi-probe is next (see `lingo-db-plan.md`).
2. ~~Apply the lingo-db optimizations above (skip-hash-cmp codegen).~~ **Done** (`ir_to_llvm.cpp:5810-5824`).
3. ~~Add debug print for each optimization, and test it in debug mode.~~ **Done** (`[AQP-JIT] skip_hash_cmp:` under `#ifndef NDEBUG`).
4. Check the correctness of pipeline-jit with none-split, node-based, and relationship-center in release mode, and compare to golden result.
5. Measure performance diff with enable/disable each optimization for pipeline-jit with none-split, node-based, and relationship-center in release mode. 
6. If DuckDB+pipeline-jit matches lingo-db's execution time, the approach is validated.
7. Finally, per-query flag tuning: sweep JOB queries with flag combinations, store optimal configs in a JSON file.

### Implementation Plan

See `lingo-db-plan.md` for the full implementation plan covering:
- Phase 2–5: Multi-probe chain detection, codegen, and dispatch
- Phase 6: JIT compilation cache for `CompileFilterProbeProjectFusion`
- Phase 7: Per-query flag tuning

## Key Source Code References

| File | Lines | Description |
|------|-------|-------------|
| `lingo-db/src/compiler/Dialect/RelAlg/Passes.cpp` | 17-49 | Full optimizer pipeline (18 passes) |
| `lingo-db/src/compiler/Dialect/RelAlg/Transforms/OptimizeJoinOrder.cpp` | 121 | DPHyp join ordering (bushy plans) |
| `lingo-db/src/compiler/Dialect/SubOperator/Transforms/SplitIntoExecutionSteps.cpp` | 14-224 | Pipeline splitting via state dependency analysis |
| `lingo-db/src/compiler/Conversion/SubOpToControlFlow/SubOpToControlFlow.cpp` | 4334-4365 | Single-function pipeline compilation |
| `lingo-db/src/compiler/Conversion/DBToStd/LowerToStd.cpp` | 1071-1074 | Hash inlined as MLIR ops (hashInteger + combineHashes) |
| `lingo-db/src/compiler/Dialect/SubOperator/Transforms/SpecializeSubOpPass.cpp` | 116-117 | Skip hash cmp for integer keys |
| `lingo-db/src/execution/LLVMBackends.cpp` | 565-581 | Same 4 LLVM passes (InstCombine, Reassociate, GVN, SimplifyCFG) |
| `lingo-db/src/execution/Execution.cpp` | 122-133 | SubOp optimization pipeline (FoldColumns, PullGatherUp, etc.) |
| `duckdb/src/common/types/hash.cpp` | 36-43 | MurmurHash64 (matches our inline hash in probe codegen) |
| `duckdb/src/execution/operator/join/physical_hash_join.cpp` | 1079-1130 | JIT probe dispatch (enabled), `D_ASSERT(sink.finalized)` |
| `duckdb/src/execution/join_hashtable.cpp` | 1549-1560 | `PopulateAQPJITView()` |
| `duckdb/src/execution/aqp_jit.cpp` | 26-52 | `MakeChunkViewAt()` — thread-local col_buf for JIT dispatch |
| `duckdb/src/include/duckdb/execution/aqp_jit.hpp` | 131-229 | `AQPJITContext` (pipeline_fns, join_ht_views, pipeline_states) |
| `duckdb/src/parallel/pipeline_executor.cpp` | 407-470 | `PipelineExecutor::Execute()` — chains operators sequentially |
| `AQP_middleware/include/jit/aqp_jit_abi.h` | 79-83 | `AQP_JIT_PIPELINE_JIT` and `AQP_JIT_LEVEL_MASK` |
| `AQP_middleware/src/adapters/duckdb_adapter.cpp` | 2293-2305 | `EnsureJITCompiler()` — sets all JIT flags, cache in benchmark_mode |
| `AQP_middleware/src/adapters/duckdb_adapter.cpp` | 3234 | Probe fusion gate (`jit_flags_ & AQP_JIT_PIPELINE_JIT`) |
| `AQP_middleware/src/adapters/duckdb_adapter.cpp` | 3243-3256 | payload_prune reads `hj.payload_columns.col_idxs` |
| `AQP_middleware/src/adapters/duckdb_adapter.cpp` | 3451-3548 | Output column maps: `lhs_output_idxs`, `rhs_output_layout_idxs`, `lhs_key_chunk_idxs` |
| `AQP_middleware/src/jit/ir_to_llvm.cpp` | 5719-5808 | `CompileFilterProbeProjectFusion` entry (NO cache) |
| `AQP_middleware/src/jit/ir_to_llvm.cpp` | 5810-5824 | skip_hash_cmp codegen: `do_skip_salt` decision |
| `AQP_middleware/src/jit/ir_to_llvm.cpp` | 6214-6270 | ROF two-stage prelude (batch_probe) |
| `AQP_middleware/src/jit/ir_to_llvm.cpp` | 6299-6362 | Prefetch: stage-2 consumer-side look-ahead (requires batch_probe) |
