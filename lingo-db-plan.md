# Pipeline-JIT Implementation Plan

## Repositories

- **AQP middleware**: `/home/pei/Project/AQP_middleware` (branch: `topdown_fix`)
- **DuckDB (patched)**: `/home/pei/Project/duckdb`
- **Lingo-db (reference only)**: `/home/pei/Project/lingo-db`
- **JOB queries**: `/home/pei/Project/benchmarks/imdb_job-postgres/queries/`
- **DuckDB database**: `/home/pei/Project/duckdb/measure/imdb.db`
- **Analysis doc**: `lingo-db-pipeline-jit.md` (in AQP_middleware root)

Build commands:
```bash
# Debug build
cd /home/pei/Project/AQP_middleware && cmake --build build_debug
# Release build
cd /home/pei/Project/AQP_middleware && cmake --build build_release
```

## Context

The AQP middleware compiles individual hash-join probes via `CompileFilterProbeProjectFusion`, producing one JIT function per probe operator. Lingo-db compiles the **entire probe pipeline** as a single function, eliminating per-operator dispatch overhead. We replicate this under `--jit-level=pipeline`.

Key architectural clarification: `kernel-path=pipeline` uses `PipelineKernel`'s own `HashJoinTable` (called from `ir_query_splitter.cpp`), not DuckDB's `JoinHashTable`. The `CompileFilterProbeProjectFusion` probe codegen is pipeline-jit only. The `kernel_path_==PIPELINE` check in the probe gate was dead code and has been removed.

### How the existing single-probe pipeline-jit works (end-to-end)

**Registration (compile time)** — `duckdb_adapter.cpp:RegisterJIT()`:
1. DFS walk of the physical plan tree. For each HASH_JOIN operator:
2. Build `build_schema` from IR build child's `target_list` (line 3141-3169)
3. Build `probe_schema` from IR probe child's `target_list`, or fall back to
   DuckDB physical types (line 3301-3339)
4. Validate schema match: `probe_schema` dtypes must match DuckDB's
   `op.children[0].GetTypes()` (line 3351-3361)
5. Build `payload_schema` from `build_schema`, optionally pruned via
   `hj.payload_columns.col_idxs` (line 3367-3374)
6. Get output column maps from the `PhysicalHashJoin` object (line 3451-3463):
   - `lhs_output_idxs` = `hj.lhs_output_columns.col_idxs` (indices into probe child output)
   - `rhs_output_layout_idxs` = `hj.rhs_output_columns.col_idxs` (indices into HT [keys, payload])
   - `lhs_output_dtypes` / `rhs_output_dtypes` from DuckDB types
7. Get key positions from `hj.conditions[i].left` as `BoundReferenceExpression::index` (line 3518-3538)
8. Call `CompileFilterProbeProjectFusion(...)` with all of the above (line 3541-3548)
9. Register the compiled function in `ctx->aqp_jit_context->pipeline_fns[eid]` (line 3556)
10. Register an empty `AQPJoinHTView` in `ctx->aqp_jit_context->join_ht_views[eid]` (line 3434)
11. Set `pipeline_states[eid]` pointing to the same view (line 3438)

**Dispatch (runtime)** — `physical_hash_join.cpp:ExecuteInternal()`:
1. Check: `jit && AQPJIT_PIPELINE && !sink.external && Count()>0 && !perfect_join` (line 1097)
2. Look up `pipeline_fns[eid]` and `join_ht_views[eid]` (line 1100-1103)
3. `PopulateAQPJITView()` fills the view from the finalized hash table (line 1107)
4. `MakeChunkView(input)` → `in_cv` (probe chunk from child operator)
5. `MakeChunkViewAt(chunk, input.ColumnCount())` → `out_cv` (output chunk, offset in thread-local col_buf)
6. Call: `out_rows = pipeline_fn(&in_cv, &out_cv, pipeline_state)` (line 1120)
7. If `out_rows >= 0`: `chunk.SetCardinality(out_rows)`, return `NEED_MORE_INPUT`
8. If `out_rows < 0`: bail, increment `bail_count`, fall through to interpreter

**Output chunk layout**: `[lhs_output_columns..., rhs_output_columns...]`.
The JIT function writes columns 0..N into `out_cv`, which maps to the
chunk's columns starting at `col_buf[input.ColumnCount()]`. DuckDB expects
this to match `chunk`'s column layout exactly.

## Completed

- **Phase 1: skip_hash_cmp codegen** — `src/jit/ir_to_llvm.cpp:5836-5849, 6074, 6423, 6465`
- **DuckDB dispatch enabled** — `physical_hash_join.cpp:1097`: `if (false && ...)` → `if (...)`
- **Dead code removed** — `kernel_path_==PIPELINE` from probe fusion gates (`duckdb_adapter.cpp:3070, 3219`)
- **Debug build link error fixed** — extracted `util.cpp` into `util_lib` (`CMakeLists.txt`)
- **Debug print added** — `[AQP-JIT] skip_hash_cmp:` under `#ifndef NDEBUG`
- **Pipeline-jit configs added** to `correctness_test.sh` and `breakdown_measurement_script.sh`

## Verification Workflow (per phase)

Each phase follows this workflow:

### Step 1: Debug smoke test
Build debug, run 1 query (1a.sql) with all 3 splits to verify feature activates:
```bash
cd measure
# none-split
../build_debug/aqp_middleware --engine=duckdb --db=... --split=none \
  --jit-level=pipeline --jit-skip-hash-cmp --benchmark /tmp/test_1a 2>&1 | grep skip_hash_cmp
# node-based
../build_debug/aqp_middleware ... --split=node-based ... 2>&1 | grep skip_hash_cmp
# relationship-center
../build_debug/aqp_middleware ... --split=relationship-center ... 2>&1 | grep skip_hash_cmp
```

### Step 2: Correctness (release, all JOB queries)
Run pipeline-jit configs from `correctness_test.sh` and diff against golden files:
```bash
cd measure
# Runs all 113 JOB queries per config, diffs against golden
# Pipeline-jit configs in JIT_CONFIGS array:
#   duckdb|none|pipeline|none, duckdb|none|pipeline|auto
#   duckdb|node-based|pipeline|none, duckdb|node-based|pipeline|auto
#   duckdb|relationship-center|pipeline|none, duckdb|relationship-center|pipeline|auto
bash correctness_test.sh  # or run subset manually
```

### Step 3: Performance measurement (release)
Compare skip_hash_cmp on vs off:
```bash
cd measure
# skip_hash_cmp ON (default)
bash measure_breakdown_time_job.sh duckdb none pipeline none
bash measure_breakdown_time_job.sh duckdb node-based pipeline none
bash measure_breakdown_time_job.sh duckdb relationship-center pipeline none

# skip_hash_cmp OFF
bash measure_breakdown_time_job.sh duckdb none pipeline none on on on off
bash measure_breakdown_time_job.sh duckdb node-based pipeline none on on on off
bash measure_breakdown_time_job.sh duckdb relationship-center pipeline none on on on off
```
Results go to `job_result/` as CSV files. Compare per-query median times.

---

## Phase 2–5 (combined): Multi-Probe Chain Detection + State + Codegen + Dispatch

These four phases are combined into one because they are not independently
testable: chain detection produces only debug prints without codegen, and
codegen is useless without dispatch.

### DuckDB probe pipeline shape (verified)

**Critical correction**: the earlier "left-deep vs bushy" analysis was based
on tree shape (HJ's child[0] being another HJ), which is about the physical
plan tree structure. But what actually matters for multi-probe is how many
HASH_JOIN operators appear on the **probe pipeline** — i.e., the linear
operator chain that DuckDB's `PipelineExecutor` pushes chunks through.

In DuckDB's pipeline model:
- child[0] of HJ = probe side (receives chunks from previous pipeline operator)
- child[1] of HJ = build side (runs in a **separate pipeline** that completes
  before the probe pipeline starts)

Tested all 113 JOB queries (none-split) by tracing the child[0] path from
the plan root and counting HJ operators:

| HJ count on probe pipeline | Query count | Examples |
|:---:|:---:|:---|
| 1 | **90** | 7c, 29a, 33a, 17f — most queries |
| 2 | **21** | 1b, 1c, 1d, 3a, 3c, 5b, 6b/d/f, 11c/d, 12a-c, 13a-d, 21a/c, 27c |
| 3 | **2** | 5a, 5c |

Multi-probe fusion benefits only the **23 queries with 2+ HJ on the probe
pipeline**. The 90 single-HJ queries already get optimal single-probe JIT.

Representative probe pipelines (read right-to-left = execution order):
- 13a: `SCAN → HJ₁ → HJ₂ → PRJ → AGG` — two consecutive HJs, no gap
- 1b:  `SCAN → FLT → HJ₁ → HJ₂ → PRJ → AGG` — two consecutive HJs
- 5a:  `SCAN → HJ₁ → FLT → PRJ → HJ₂ → HJ₃ → PRJ → AGG` — FLT/PRJ between HJ₁ and HJ₂
- 11d: `SCAN → FLT → PRJ → HJ₁ → HJ₂ → PRJ → AGG` — two consecutive HJs

Execution times for multi-HJ queries (none-split, no JIT):
6f=228ms, 11d=259ms, 1c=125ms, 5a=126ms, 11c=87ms, 3c=78ms are heaviest.

### How lingo-db handles this

Lingo-db uses **DPHyp** (`OptimizeJoinOrder.cpp:121`) for join ordering,
which explicitly produces bushy plans. Its `SplitIntoExecutionSteps` pass
splits by **read/write state dependencies** (topological sort), not by tree
shape. The probe pipeline compiles all lookups sequentially within a single
function. The algorithm is **shape-agnostic**.

### DuckDB pipeline execution model (verified)

Confirmed by reading `pipeline_executor.cpp:407-470`:
1. `PipelineExecutor::Execute()` chains operators sequentially via
   `intermediate_chunks[]`. Each operator receives the previous operator's
   output and produces its own output DataChunk.
2. `D_ASSERT(sink.finalized)` at `physical_hash_join.cpp:1083` confirms
   all build sides are finalized before any probe starts.
3. The output DataChunk from each HJ has layout
   `[lhs_output_columns..., rhs_output_columns...]` where:
   - `lhs_output_columns.col_idxs` = indices into probe child's output
   - `rhs_output_columns.col_idxs` = indices into HT layout [keys, payload]

### Multi-probe dispatch strategy

For consecutive HJ operators on the probe pipeline (e.g., `→ HJ₁ → HJ₂ →`):

**Dispatch at the innermost HJ (HJ₁)**. The fused function:
1. Receives the input chunk (from SCAN/FILTER before HJ₁)
2. Probes HT₁, for each match extracts lhs+rhs columns per HJ₁'s output map
3. Using the intermediate result, probes HT₂
4. Writes the final output matching HJ₂'s output layout

**HJ₂ becomes a pass-through**: receives the already-complete output from HJ₁'s
fused function and passes it unchanged. The pass-through checks
`multi_probe_passthrough_eids` in the JIT context and returns
`NEED_MORE_INPUT` immediately.

Why dispatch at innermost (not outermost):
- HJ₁ receives the original scan input — matches the existing
  `CompileFilterProbeProjectFusion` input contract (probe chunk from child).
- HJ₂ as pass-through is trivial — just `chunk.Reference(input)`.
- If we dispatched at HJ₂, it would receive HJ₁'s output chunk, but we'd
  need HJ₁ to pass through the raw scan input — and HJ₁'s output schema
  differs from the scan schema, breaking downstream expectations.

### Chain detection algorithm (concrete)

**DFS visit order**: `RegisterJIT` does a DFS walk starting from the plan
root, visiting the current node first, then recursing into `children[0]`
(probe side), then `children[1]` (build side). See `duckdb_adapter.cpp:3631`.

For 13a's probe pipeline `SCAN(5d00) → HJ₁(1320) → HJ₂(82a8) → PRJ → AGG`,
the DFS visits **HJ₂ (outer) before HJ₁ (inner)** because HJ₁ is HJ₂'s
child[0]. Verified with debug trace output (see below).

**Algorithm**: When `RegisterJIT` visits a HASH_JOIN node, walk the
`children[0]` chain to detect consecutive HJs:

```
DetectChain(op):  // op is a HASH_JOIN
  chain = [op]    // outermost HJ first
  cursor = op.children[0]
  while cursor.type == HASH_JOIN:
    chain.append(cursor)
    cursor = cursor.children[0]
  if len(chain) < 2:
    return  // no multi-probe; single-probe will handle this HJ normally

  // chain = [HJ_outer, ..., HJ_inner], execution order is reversed
  // HJ_inner (chain.back()) is the dispatch point
  // HJ_outer..HJ_outer-1 are pass-through
  innermost_eid = ExpressionID(chain.back())
  for each hj in chain[0..n-2]:  // all except innermost
    passthrough_eids.insert(ExpressionID(hj))

  // Collect per-stage metadata for all HJs in chain (reversed to match
  // execution order: innermost first). For each HJ, extract the same
  // metadata that single-probe uses (probe_schema, payload_schema,
  // lhs/rhs output maps, conditions, key positions).
  stages = []
  for hj in reversed(chain):
    stages.append(CollectProbeStageInfo(hj))

  // Compile fused function and register for innermost eid
  fn = CompileMultiProbeChain(stages, ...)
  pipeline_fns[innermost_eid] = fn
```

**When to run**: This detection happens at the **outer HJ** (visited first
in DFS). When the inner HJ is visited later, check if its eid is already
in `pipeline_fns` (registered by the multi-probe compilation) and skip
redundant single-probe compilation.

**Prerequisite**: All HJs in the chain must individually pass the
single-probe eligibility checks (IR matching, schema validation, etc.).
If any HJ fails, break the chain at that point. Verified that the inner
HJ's `PhysicalHashJoin` metadata (`lhs_output_columns`, `conditions`,
etc.) is fully populated when the outer HJ is visited, because
`RegisterJIT` runs after the physical plan is completely constructed
(`duckdb_adapter.cpp:881`).

**Verified DFS visit order for 13a** (from `[AQP-JIT-TRACE] visit op=`):
```
AGG(8500) → PROJ(8450) → HJ(82a8) → HJ(1320) → SCAN(5d00)
                            ↑ outer     ↑ inner
                            visited 1st  visited 2nd
```

### Chain detection rules

Collect maximal runs of consecutive HJ operators on the probe pipeline
(the child[0] chain). A run breaks at any non-HASH_JOIN operator (FILTER,
PROJECTION, etc.) or at an HJ that fails eligibility. For each run of
length ≥ 2:
- HJ at the end of the child[0] chain (innermost) = dispatch point
- Remaining HJs = pass-through

For 5a's pipeline `SCAN → HJ₁ → FLT → PRJ → HJ₂ → HJ₃ → PRJ → AGG`:
- Physical tree child[0] chain from root: AGG→PRJ→HJ₃→HJ₂→FLT→PRJ→HJ₁→SCAN
- HJ₃→HJ₂ are consecutive (HJ₂ is HJ₃'s child[0]): fuse, HJ₂ dispatches, HJ₃ pass-through
- HJ₁ is alone (separated by FLT/PRJ): single-probe JIT

### OutColDesc encoding in existing codegen (reference)

The codegen at `ir_to_llvm.cpp:5847-5993` maps each output column to
either a PROBE source or a PAYLOAD source via `OutColDesc`:

```cpp
struct OutColDesc {
  enum { PROBE, PAYLOAD } source;
  int probe_col_idx;     // index into probe_schema (PROBE source only)
  int payload_col_idx;   // encoding depends on sign:
  //   >= 0: payload column at data_offsets[num_keys + payload_row_indices[k]]
  //   <  0: key column at data_offsets[num_keys + payload_col_idx]
  //          (= data_offsets[layout_idx] where layout_idx < num_keys)
  unsigned payload_offset;
  int32_t dtype;
  unsigned elem_size;
};
```

The `rhs_output_layout_idxs` from DuckDB index into the HT layout
`[key₀, key₁, ..., payload₀, payload₁, ...]`:
- `layout_idx < num_keys` → key column. `payload_col_idx = layout_idx - num_keys` (negative).
  Emitted as: `data_offsets[layout_idx]`.
- `layout_idx >= num_keys` → payload column. `payload_col_idx = layout_idx - num_keys`.
  Emitted as: `data_offsets[num_keys + payload_row_indices[payload_col_idx]]`.

Key code reference: `ir_to_llvm.cpp:5934-5965` (rhs output mapping),
`ir_to_llvm.cpp:6533-6553` (emit PAYLOAD: load from row via data_offsets).

### Output schema threading (concrete algorithm)

For a 2-probe chain `→ HJ₁ → HJ₂ →`:

**HJ₁ probe input**: original scan output, `probe_schema₁` with N₁ columns.
**HJ₁ output**: `[lhs_out₁..., rhs_out₁...]` where:
  - `lhs_out₁[i]` = `hj₁.lhs_output_columns.col_idxs[i]` (index into probe_schema₁)
  - `rhs_out₁[j]` = `hj₁.rhs_output_columns.col_idxs[j]` (index into HT₁ layout)
  - Output has `L₁ = len(lhs_out₁)` LHS columns then `R₁ = len(rhs_out₁)` RHS columns

**HJ₂ probe input**: HJ₁'s output, a virtual schema with `L₁ + R₁` columns.
**HJ₂ output**: `[lhs_out₂..., rhs_out₂...]` where:
  - `lhs_out₂[i]` = `hj₂.lhs_output_columns.col_idxs[i]` (index into HJ₁'s output)
  - `rhs_out₂[j]` = `hj₂.rhs_output_columns.col_idxs[j]` (index into HT₂ layout)

**Transitive resolution for final output columns**:

The fused function must produce HJ₂'s output layout directly from the
original probe input + HT₁ + HT₂. For each final output column:

```
resolve_final_output_columns(stages):
  // stages[0] = HJ₁ (innermost, receives scan input)
  // stages[1] = HJ₂ (outermost)

  final_out_cols = []
  L₁ = len(stages[0].lhs_output_idxs)

  // HJ₂'s LHS columns: trace back through HJ₁'s output
  for i in range(len(stages[1].lhs_output_idxs)):
    idx = stages[1].lhs_output_idxs[i]
    if idx < L₁:
      // This column came from HJ₁'s LHS → original probe chunk
      orig_probe_idx = stages[0].lhs_output_idxs[idx]
      final_out_cols.append(FinalCol(source=PROBE, probe_col_idx=orig_probe_idx))
    else:
      // This column came from HJ₁'s RHS → HT₁
      rhs_idx = idx - L₁
      ht1_layout_idx = stages[0].rhs_output_layout_idxs[rhs_idx]
      final_out_cols.append(FinalCol(source=HT1, ht_stage=0, layout_idx=ht1_layout_idx))

  // HJ₂'s RHS columns: direct from HT₂
  for j in range(len(stages[1].rhs_output_layout_idxs)):
    ht2_layout_idx = stages[1].rhs_output_layout_idxs[j]
    final_out_cols.append(FinalCol(source=HT2, ht_stage=1, layout_idx=ht2_layout_idx))

  return final_out_cols
```

For 3-probe chains (5a, 5c), this extends recursively: HJ₃'s LHS columns
trace back through HJ₂'s output, which may trace back through HJ₁'s output.

**In LLVM codegen**: No intermediate materialization needed. After probing
HT₁ and finding a match, the matched row pointer (`chain_ptr₁`) gives
access to HT₁ data. For columns from the original probe chunk, use
`cc.col_data[orig_probe_idx]` (SSA value from entry block). For HT₁
payload columns, use `GEP(chain_ptr₁, hoisted_offset₁[...])`. Then probe
HT₂ using keys extracted from these SSA values. On HT₂ match, emit all
final output columns from the resolved origins.

**Key positions for HT₂ probe**: HJ₂'s `conditions[i].left` is a
`BoundReferenceExpression` whose `.index` references HJ₁'s output (not the
original scan). Apply the same LHS/RHS resolution as above to find whether
each key comes from the probe chunk or HT₁, and load accordingly.

### ProbeStageInfo struct

```cpp
struct ProbeStageInfo {
  duckdb::PhysicalHashJoin *hj;   // the physical operator
  uint64_t eid;                    // ExpressionID(*hj)
  std::vector<aqp_jit::ColSchema> probe_schema;   // only for stage 0 (scan input)
  std::vector<aqp_jit::ColSchema> payload_schema;  // build-side columns in HT
  std::vector<int> payload_row_indices;
  std::vector<int> lhs_output_idxs;    // hj.lhs_output_columns.col_idxs
  std::vector<int> rhs_output_layout_idxs;  // hj.rhs_output_columns.col_idxs
  std::vector<int32_t> lhs_output_dtypes;
  std::vector<int32_t> rhs_output_dtypes;
  std::vector<int> lhs_key_chunk_idxs;  // from conditions[i].left.index
  std::vector<int32_t> lhs_key_dtypes;
  bool skip_hash_cmp_eligible;    // all keys are integer type
  // IR nodes for filter/join (needed by CompileMultiProbeChain)
  const ir_sql_converter::AQPStmt *filter_ir;
  const ir_sql_converter::AQPStmt *join_ir;
};
```

### AQPMultiProbeState struct (aqp_jit_abi.h)

```c
typedef struct {
  AQPJoinHTView *views[4];  // up to 4 probe stages (max 3 observed in JOB)
  uint32_t num_stages;
} AQPMultiProbeState;
```

At dispatch time, `physical_hash_join.cpp` for the innermost HJ fills all
views: `PopulateAQPJITView()` for its own HT (stage 0), plus look up the
outer HJs' `join_ht_views[eid]` for stages 1..N-1. The outer HJs' build
sides are already finalized (`D_ASSERT(sink.finalized)` at line 1083).

### Intermediate chunk column count problem (verified)

`PipelineExecutor` (`pipeline_executor.cpp:31-39`) pre-allocates one
`intermediate_chunk` per operator at pipeline construction time. Each
chunk's column count is fixed at `Initialize()` and cannot change after.
Specifically, `intermediate_chunks[k+1]` (the output chunk for operator `k`)
is initialized with `operators[k].GetTypes()` — the operator's own output
types (verified by tracing the initialization loop at lines 31-39).

For the multi-probe chain `→ HJ₁ → HJ₂ →`:
- HJ₁'s output chunk has `HJ₁.GetTypes().size()` columns
- HJ₂'s output chunk has `HJ₂.GetTypes().size()` columns
- These may differ (HJ₁ may project differently than HJ₂)

If HJ₁ dispatches the fused function that writes HJ₂'s output layout into
HJ₁'s chunk, the column count may not match:
- `HJ₂.output_cols > HJ₁.output_cols` → writes past end of chunk → crash
- `HJ₂.output_cols ≤ HJ₁.output_cols` → works but wastes memory

**Solution: Dispatch at the outermost HJ (HJ₂), not the innermost.**

Revised dispatch strategy:
1. HJ₁ (inner) is registered as pass-through. When `ExecuteInternal` is
   called for HJ₁, it does NOT run a JIT function. Instead, it executes
   the **normal DuckDB interpreter probe** for HJ₁ (the existing non-JIT
   code path after the JIT dispatch block). This produces HJ₁'s normal
   output into `intermediate_chunks[k+1]`.
2. HJ₂ (outer) receives HJ₁'s output as `input`. HJ₂ dispatches the fused
   function which takes HJ₁'s output and probes HT₂ only (single probe,
   but with HJ₁'s output as the probe schema instead of the scan schema).
3. The fused function writes HJ₂'s output layout into HJ₂'s chunk → column
   count matches.

This does NOT eliminate HJ₁'s probe overhead (HJ₁ still runs the
interpreter). The benefit is eliminating the **operator dispatch overhead
between HJ₁ and HJ₂** and compiling HJ₂'s probe as JIT.

**Alternative: Full fusion with chunk reinitialization.** If HJ₁ dispatches
a function that probes both HTs and writes HJ₂'s output layout, we must
reinitialize HJ₁'s output chunk to have HJ₂'s column types:

```cpp
// In HJ₁'s ExecuteInternal, multi-probe path:
auto &outer_types = jit->multi_probe_outer_types[eid];
if (chunk.ColumnCount() != outer_types.size()) {
    chunk.Destroy();
    chunk.Initialize(allocator, outer_types);
}
// ... dispatch fused fn writing HJ₂'s output layout
```

Then HJ₂'s pass-through does `chunk.Reference(input)` which works because
`input.ColumnCount() == chunk.ColumnCount()` (both have HJ₂'s layout).

**Trade-off**: Chunk reinitialization adds complexity and may interact
poorly with PipelineExecutor's internal assumptions. The simpler "dispatch
at outer" approach is safer and still eliminates one operator dispatch call.
Start with "dispatch at outer" and only add full fusion if measurements
show the remaining HJ₁ interpreter overhead is significant.

### Pass-through and dispatch in physical_hash_join.cpp

**For the innermost HJ (pass-through option, only if full fusion chosen)**:
```cpp
if (jit && jit->multi_probe_inner_eids.count(eid)) {
    // Skip JIT dispatch for this HJ — the outer HJ's fused function
    // will do the probing. Fall through to DuckDB interpreter.
    // (Or: reinitialize chunk + dispatch fused fn, see above.)
}
```

**For the outermost HJ (dispatch at outer approach)**:
```cpp
// Standard JIT dispatch path — the fused function for a multi-probe
// chain is registered at the outermost HJ's eid. It receives HJ₁'s
// output as input (the probe schema is HJ₁'s output types, not the
// scan types). It probes HT₂ (and potentially HT₃ for 3-probe chains)
// and writes HJ₂'s final output layout.
```

**For the outermost HJ (full fusion approach)**:
```cpp
if (jit && jit->multi_probe_passthrough_eids.count(eid)) {
    chunk.Reference(input);
    return OperatorResultType::NEED_MORE_INPUT;
}
```

### Recommended approach: Dispatch at outer (Phase 2-5 v1)

Start with the simpler "dispatch at outer" approach:
1. **No pass-through needed** — all HJs run normally except the outermost
   gets a JIT function that probes only its own HT (but compiled, not
   interpreted). The inner HJ(s) run the DuckDB interpreter.
2. The "multi-probe" benefit at this stage is: the outermost HJ gets JIT
   even when its probe input comes from another HJ (currently, the
   outermost HJ may fail single-probe IR matching because its probe
   schema comes from HJ₁'s output, not a scan).
3. Later optimization: add full fusion with chunk reinitialization if
   profiling shows the inner HJ interpreter is the bottleneck.

**Column counts verified for 13a**: HJ₁(1320) output = 4 cols,
HJ₂(82a8) output = 4 cols (happens to match). For general queries, they
may differ — the "dispatch at outer" approach avoids this problem entirely.

### Files to modify

**Dispatch-at-outer (v1, recommended)**:

**`src/adapters/duckdb_adapter.cpp`** — `RegisterJIT`
- At the HASH_JOIN block (line 3072), add chain detection: walk
  `op.children[0]` to find consecutive HJs. If chain length ≥ 2,
  the outermost HJ's probe schema = HJ₁'s output types (not scan types).
  Build the probe schema from DuckDB physical types (the same fallback
  path at line 3331-3339), compile a single-probe JIT for the outermost
  HJ using HJ₁'s output as the probe schema, and register it normally.
  The inner HJ(s) get their own single-probe JIT as usual.
- No `ProbeStageInfo` needed for v1 — it's just single-probe JIT for
  each HJ individually, but with correct probe schema for the outer HJ.

No changes to `aqp_jit_abi.h`, `aqp_jit.hpp`, or `physical_hash_join.cpp`
for v1. The only change is making the outer HJ's probe schema construction
work when the probe input comes from another HJ (not a scan).

**Full fusion (v2, follow-up if v1 shows insufficient speedup)**:

**`include/adapters/duckdb_adapter.h`**
- Add `ProbeStageInfo` struct (see above)

**`src/adapters/duckdb_adapter.cpp`** — `RegisterJIT`
- Chain detection + collect `ProbeStageInfo` for each HJ in chain.
  Compile fused function via `CompileMultiProbeChain`, register for
  innermost eid. Store outermost types in JIT context for chunk
  reinitialization.

**`include/jit/aqp_jit_abi.h`** — add `AQPMultiProbeState` struct

**`include/jit/ir_to_llvm.h`** — add `CompileMultiProbeChain()` declaration
**`src/jit/ir_to_llvm.cpp`** — implement `CompileMultiProbeChain()`

**`duckdb/.../aqp_jit.hpp`** — add `multi_probe_passthrough_eids` set,
  `multi_probe_outer_types` map, `multi_probe_states` map
**`duckdb/.../physical_hash_join.cpp`** — chunk reinitialization at
  innermost HJ, pass-through at outer HJ(s)

### Key design decisions
- Column threading via SSA values (no intermediate materialization)
- Per-probe `skip_hash_cmp` based on that probe's key types (already orthogonal)
- Per-probe `payload_prune` from each HJ's `payload_columns.col_idxs`
- **Disable** `batch_probe` and `prefetch` for multi-probe initially (scalar
  probe loop per stage). Add as follow-up.
- Bail to interpreter on output overflow (>2048 rows)
- Handle `perfect_join_executor` case: if any HJ in the chain uses perfect
  hash, break the chain at that point
- Break chain at any non-HJ operator (FILTER, PROJECTION) between HJs
- Multi-probe only when ALL HJs in chain pass single-probe eligibility
  (IR matching, schema validation). If any fails, fall back to per-HJ
  single-probe for the entire chain.

### Verification
- Debug: run 13a.sql (simplest 2-HJ case), verify chain detection + dispatch
- Correctness: run pipeline-jit configs from `correctness_test.sh`
- Performance: compare multi-probe vs single-probe for the 23 affected queries

---

## Phase 6: JIT Compilation Cache

**Goal**: Ensure all JIT compilation paths have in-memory caching, and expose
a standalone `--jit-cache` CLI flag.

### Gap analysis (verified)

| Compile function | Has cache? | Location |
|:---|:---:|:---|
| `CompileExpr` | Yes | `ir_to_llvm.cpp:4878` |
| `CompileFilter` | Yes | (shares CompileExpr path) |
| `CompileProjection` | Yes | `ir_to_llvm.cpp:5040` |
| `CompileAggUpdate` | Yes | `ir_to_llvm.cpp:5163` |
| `CompilePipeline` | Yes | `ir_to_llvm.cpp:5380` |
| `CompileFilterProbeProjectFusion` | **NO** | `ir_to_llvm.cpp:5719` |
| `CompileRangeFilter` | **NO** | (tiny functions, low priority) |
| `CompileMultiProbeChain` (new) | **NO** | (must add from the start) |

The most expensive compilation — `CompileFilterProbeProjectFusion` — has
zero cache logic, confirmed by reading the source. Every probe fusion is
recompiled even in benchmark mode (`--repeat=N`).

Currently, cache is implicitly enabled by `benchmark_mode_`
(`duckdb_adapter.cpp:2303`). There is no standalone `--jit-cache` flag.

### Files to modify

**`src/jit/ir_to_llvm.cpp`** — `CompileFilterProbeProjectFusion()`
- Add `TryCacheLoad` / `pending_cache_key` logic matching the pattern in
  `CompilePipeline`. Cache key must include: join IR, filter IR, projection
  IR, probe schema, payload schema, payload_row_indices, output idx vectors,
  and all flag state (skip_hash_cmp, batch_probe, prefetch distances).

**`src/jit/ir_to_llvm.cpp`** — `CompileMultiProbeChain()` (new)
- Add cache from the start, keyed on the concatenated per-probe descriptors.

**`src/jit/ir_to_llvm.cpp`** — `CompileRangeFilter()` (optional)
- Low priority — range filters are tiny and unique per range.

**`src/util/param_config.cpp`** — add `--jit-cache` CLI flag
- Independent of `--benchmark`; enables `SetCache(true)`.

### Verification
- Debug: verify `[AQP-JIT] cache HIT` messages on second run of same query
  in benchmark mode.
- Performance: compare first-run vs cached-run compilation time.

---

## Phase 7: Per-Query Flag Tuning

**Goal**: Sweep JOB queries with JIT flag combinations, find optimal config
per query, save to JSON.

### Flag interaction summary (verified from codegen)

The 4 extra-JIT flags are **largely orthogonal** with one dependency:

| Flag | Independent? | Notes |
|:---|:---:|:---|
| `skip_hash_cmp` | Yes | Local to probe loop branch structure |
| `payload_prune` | Yes | Controls output column set |
| `batch_probe` | Yes | Two-stage ROF (filter+hash, then probe) |
| `prefetch` | **Depends on batch_probe** | Stage-2 consumer-side prefetch only emitted when `batch_probe_` is true (`ir_to_llvm.cpp:6299`) |

Search space: 2^4 = 16 combinations per query (or 12 effective, since
prefetch-without-batch_probe is a no-op for probe path). With 113 queries
× 16 combos × ~0.1s each = ~3 min total.

### Files to modify
- New tuning script in `measure/` — sweep all flag combinations per query
- Output: JSON mapping query → optimal flag set
- Optional: runtime flag selection based on saved JSON

## Critical Files Summary

| File | Role |
|------|------|
| `CMakeLists.txt` | util_lib extraction (link fix) |
| `src/jit/ir_to_llvm.cpp:5836-5849` | skip_hash_cmp: do_skip_salt computation |
| `src/jit/ir_to_llvm.cpp:6074` | salt_ok_bb conditional creation |
| `src/jit/ir_to_llvm.cpp:6423` | probe_salt conditional computation |
| `src/jit/ir_to_llvm.cpp:6465` | Probe loop branch: skip salt or check salt |
| `src/jit/ir_to_llvm.cpp:5719` | CompileFilterProbeProjectFusion (missing cache) |
| `src/adapters/duckdb_adapter.cpp:2303` | Cache only enabled in benchmark_mode_ |
| `src/adapters/duckdb_adapter.cpp:3234` | Probe fusion gate (pipeline-jit only) |
| `src/adapters/duckdb_adapter.cpp:3243` | payload_prune reads hj.payload_columns |
| `duckdb/.../physical_hash_join.cpp:1097` | Probe dispatch (enabled) |
| `duckdb/.../aqp_jit.hpp:131` | AQPJITContext (pipeline_fns, join_ht_views) |
| `measure/correctness_test.sh` | Pipeline-jit correctness configs |
| `measure/breakdown_measurement_script.sh` | Pipeline-jit performance configs |

## Probe Pipeline Statistics (DuckDB, JOB benchmark, none-split)

HJ operators on the main probe pipeline (what matters for multi-probe fusion):
- **90/113** queries: 1 HJ (single-probe JIT sufficient)
- **21/113** queries: 2 HJ (multi-probe fusion possible)
- **2/113** queries: 3 HJ (5a, 5c — multi-probe fusion possible)

Multi-probe target queries (23 total):
1b, 1c, 1d, 3a, 3c, 5a, 5b, 5c, 6b, 6d, 6f, 11c, 11d,
12a, 12b, 12c, 13a, 13b, 13c, 13d, 21a, 21c, 27c.

Heaviest multi-probe targets: 11d (259ms), 6f (228ms), 1c (125ms), 5a (126ms).
