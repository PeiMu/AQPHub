# Performance Bottleneck Analysis Guide

Step-by-step guide for profiling operator-level bottlenecks using external tools (perf, Intel VTune, eBPF). Run at the start of each optimization iteration.

**Why not use DuckDB's internal `operator_exe.csv`?** It sums per-thread wall times via `MetricSum` in `QueryProfiler::Flush()`. A parallel operator running on N threads reports Nx its wall time, misidentifying bottlenecks under multi-threading.

For breakdown csv, use analyze_middleware_breakdown, analyze_none_split_breakdown in /home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts/plot_middleware_jit.py to parse it.

## Prerequisites: Build with Frame Pointers + Debug Symbols

Both AQPHub and DuckDB must be built with `-fno-omit-frame-pointer` and RelWithDebInfo so that `perf` and VTune can unwind call stacks accurately. This has ~1-3% overhead vs pure Release — negligible for memory-bound workloads.

### DuckDB

```bash
cd /home/pei/Project/duckdb
mkdir -p build/reldebug && cd build/reldebug
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" \
      -DCMAKE_C_FLAGS="-fno-omit-frame-pointer" \
      ../..
cmake --build . --config RelWithDebInfo -j$(nproc)
```

### AQPHub

`CMakeLists.txt` already searches `build/reldebug/src` for the DuckDB library.

```bash
cd /home/pei/Project/AQPHub
mkdir -p build_reldebug && cd build_reldebug
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" \
      -DCMAKE_C_FLAGS="-fno-omit-frame-pointer" \
      ..
make -j$(nproc)
```

Set `LD_LIBRARY_PATH` at runtime:

```bash
export LD_LIBRARY_PATH=/home/pei/Project/duckdb/build/reldebug/src:$LD_LIBRARY_PATH
```

Verify:

```bash
file build_reldebug/aqp_middleware   # should show "with debug_info, not stripped"
```

---

## Step 1: Identify Profiling Targets

Run the top-query finder on the latest breakdown CSV:

```bash
python3 measure/find_top_queries.py measure/job_result/
```

This parses all `*_breakdown_time_log.csv` files, drops 5 warmup iterations, and ranks queries by median total wall-clock time. Pick the top-5 for profiling.

---

## Step 2: Profile Baseline vs Best JIT (Find the Gap)

Profile the same top-5 queries under two configurations to understand **where split+JIT overhead comes from** and **what remains as the bottleneck to optimize**.

```bash
BINARY=build_reldebug/aqp_middleware
DB=/home/pei/Project/duckdb/measure/imdb.db
SCHEMA=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql
FKEYS=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql
QUERY_DIR=/home/pei/Project/benchmarks/imdb_job-postgres/queries
```

### 2a. Baseline (none-split / none-jit)

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,\
LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,\
dTLB-loads,dTLB-load-misses,\
L1-dcache-loads,L1-dcache-load-misses \
  $BINARY --engine=duckdb --db=$DB --schema=$SCHEMA --fkeys=$FKEYS \
    --split=none --repeat=5 --jit-level=none \
    $QUERY_DIR/<query>.sql
```

### 2b. Best JIT config (node-based / pipeline-jit)

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,\
LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,\
dTLB-loads,dTLB-load-misses,\
L1-dcache-loads,L1-dcache-load-misses \
  $BINARY --engine=duckdb --db=$DB --schema=$SCHEMA --fkeys=$FKEYS \
    --split=node-based --repeat=5 \
    --jit-level=pipeline --jit-opt=o1 --jit-simd=none \
    $QUERY_DIR/<query>.sql
```

### 2c. How to interpret the gap

Compare baseline vs best JIT on the same query:

| What to compare | What it tells you |
|-----------------|-------------------|
| Wall time delta | Total overhead of split+JIT vs native DuckDB |
| IPC delta | Whether JIT improved compute efficiency or made it worse |
| LLC miss ratio delta | Whether JIT improved or worsened cache behavior |
| dTLB miss ratio delta | Whether JIT changed memory access patterns |
| CPU/Wall ratio delta | Whether split reduced parallelism (lower = fewer threads active) |

**If baseline is faster**: the gap is split overhead (temp table materialization, sub-plan setup, reduced parallelism). Focus on split strategy improvements.

**If best JIT is faster**: JIT is working. Focus on remaining bottleneck within the JIT config (Step 3).

### 2d. Classify the bottleneck type (on best JIT config)

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| IPC (instructions/cycle) | < 1.0 | Memory-bound (CPU stalling for data) |
| IPC | 1.0 - 2.0 | Mixed |
| IPC | > 2.0 | Compute-bound |
| LLC-load-misses / LLC-loads | > 20% | LLC-miss-bound — data spilling to DRAM |
| L1-dcache-load-misses / L1-dcache-loads | > 10% | L1-miss-bound — poor spatial/temporal locality |
| dTLB-load-misses / dTLB-loads | > 1% | TLB-bound — random access over large memory |
| cache-misses / cache-references | > 30% | General cache pressure |

**Decision**:
- LLC-miss-bound (DRAM-bound) → prefetching, batch probing, data layout, reduce working set
- L1/L2-miss-bound → struct packing, loop tiling, fusion to keep data in L1/L2
- TLB-bound → huge pages (madvise), reduce pointer chasing
- Compute-bound → SIMD, expression compilation, branch reduction

---

## Step 3: `perf record` + Flamegraph — Find Hotspot Functions

After classifying the bottleneck type, use `perf record` to find **which functions** are responsible.

### 3a. Sample on CPU cycles (where is wall time spent?)

```bash
perf record -F 997 -g --call-graph dwarf \
  $BINARY --engine=duckdb --db=$DB --schema=$SCHEMA --fkeys=$FKEYS \
    --split=node-based --repeat=10 \
    --jit-level=pipeline --jit-opt=o1 --jit-simd=none \
    $QUERY_DIR/<heavy_query>.sql

perf report --stdio --no-children --sort=overhead,sym | head -50
```

### 3b. Sample on LLC cache misses (where are cache misses?)

Only needed if Step 2 shows LLC-miss-bound or DRAM-bound.

```bash
perf record -e LLC-load-misses -c 10000 -g --call-graph dwarf \
  $BINARY --engine=duckdb --db=$DB --schema=$SCHEMA --fkeys=$FKEYS \
    --split=node-based --repeat=10 \
    --jit-level=pipeline --jit-opt=o1 --jit-simd=none \
    $QUERY_DIR/<heavy_query>.sql

perf report --stdio --no-children --sort=overhead,sym | head -50
```

### 3c. Sample on L1 cache misses (if L1-miss-bound)

```bash
perf record -e L1-dcache-load-misses -c 50000 -g --call-graph dwarf \
  $BINARY ... $QUERY_DIR/<heavy_query>.sql

perf report --stdio --no-children --sort=overhead,sym | head -50
```

### 3d. Generate flamegraph

```bash
# Install FlameGraph tools (one-time)
git clone https://github.com/brendangregg/FlameGraph /tmp/FlameGraph

# Generate
perf script | /tmp/FlameGraph/stackcollapse-perf.pl | /tmp/FlameGraph/flamegraph.pl > flamegraph.svg
```

### 3e. Mapping functions to DuckDB operators

| Function pattern | Operator |
|-----------------|----------|
| `*HashJoin*` | HASH_JOIN (build + probe) |
| `*HashJoinProbe*`, `*ScanStructure*Probe*` | HASH_JOIN probe side |
| `*PerfectHashJoin*` | PERFECT_HASH_JOIN |
| `*JoinHashTable*Build*`, `*Finalize*` | HASH_JOIN build side |
| `*TableScan*`, `*SeqScan*`, `*ColumnData*Scan*` | SEQ_SCAN / TABLE_SCAN |
| `*Filter*Execute*`, `*ExpressionExecutor*` | FILTER |
| `*Aggregate*`, `*HashAggregate*` | HASH_GROUP_BY |
| `*Sort*`, `*MergeSorter*` | ORDER_BY |
| `*Projection*` | PROJECTION |
| `*NestedLoop*` | NESTED_LOOP_JOIN |
| `*TempCollection*` | SCAN_TEMP_COLLECTION (AQPHub temp tables) |

JIT-related:

| Function pattern | Meaning |
|-----------------|---------|
| `*aqp_jit*`, `*jit_compiled*` | JIT-compiled operator code |
| `*ir_to_llvm*`, `*LLVM*` | JIT compilation overhead |
| `*Prefetch*`, `*__builtin_prefetch*` | Software prefetch from JIT |

---

## Step 4: Intel VTune — Deep Memory and Threading Analysis

Use VTune when `perf` identifies the bottleneck function but you need to understand **why** it's slow (which cache level, bandwidth saturation, thread imbalance).

### 4a. Memory Access Analysis

Shows DRAM Bound vs L1/L2/L3 Bound breakdown per function, and memory bandwidth utilization over time.

```bash
vtune -collect memory-access \
  -result-dir vtune_mem_<query> \
  -- $BINARY --engine=duckdb --db=$DB --schema=$SCHEMA --fkeys=$FKEYS \
    --split=node-based --repeat=10 \
    --jit-level=pipeline --jit-opt=o1 --jit-simd=none \
    $QUERY_DIR/<heavy_query>.sql

vtune -report hotspots -r vtune_mem_<query> -format=csv > vtune_mem_hotspots.csv
vtune -report summary -r vtune_mem_<query>
```

Key metrics:
- **DRAM Bound %**: fraction of stall cycles waiting for DRAM — the true "memory-bound" indicator
- **L1/L2/L3 Bound %**: which cache level is the bottleneck per function
- **Memory Bandwidth utilization**: if close to peak, optimization must reduce data volume, not just latency
- **Top functions by LLC misses**: the real memory bottleneck

### 4b. Threading Analysis

Shows thread utilization timeline — directly answers "does split reduce parallelism?"

```bash
vtune -collect threading \
  -result-dir vtune_thread_<query> \
  -- $BINARY --engine=duckdb --db=$DB --schema=$SCHEMA --fkeys=$FKEYS \
    --split=node-based --repeat=10 \
    --jit-level=pipeline --jit-opt=o1 --jit-simd=none \
    $QUERY_DIR/<heavy_query>.sql

vtune -report summary -r vtune_thread_<query>
```

Key metrics:
- **Effective CPU Utilization**: how many cores are busy
- **Spin/Overhead Time**: time wasted in synchronization
- **Wait Time by function**: which operator causes thread stalls
- **Imbalance**: if one thread takes much longer in the same operator

### 4c. Compare baseline vs JIT with VTune

Run both 4a and 4b for baseline (no JIT) and best JIT config. Compare:
- Did JIT reduce DRAM Bound %?
- Did JIT improve thread utilization?
- Which functions still dominate LLC misses after JIT?

---

## Step 5: eBPF/bpftrace — Targeted Operator Latency

After Steps 3-4 identify the bottleneck operator, use bpftrace to measure per-invocation latency of that specific function.

### 5a. Trace a specific operator function

```bash
# Find the exact mangled symbol name
nm build_reldebug/aqp_middleware | grep -i "HashJoin" | grep -i "Execute\|Probe" | head -10
# or from DuckDB shared lib:
nm -D /home/pei/Project/duckdb/build/reldebug/src/libduckdb.so | grep -i "HashJoin" | grep -i "Execute\|Probe" | head -10
```

Then trace:

```bash
bpftrace -e '
uprobe:build_reldebug/aqp_middleware:<mangled_function_name> {
    @start[tid] = nsecs;
}
uretprobe:build_reldebug/aqp_middleware:<mangled_function_name> {
    if (@start[tid]) {
        @latency_us = hist((nsecs - @start[tid]) / 1000);
        @count = count();
        delete(@start[tid]);
    }
}' -c "$BINARY --engine=duckdb --db=$DB --schema=$SCHEMA --fkeys=$FKEYS \
    --split=node-based --repeat=5 \
    --jit-level=pipeline --jit-opt=o1 --jit-simd=none \
    $QUERY_DIR/<heavy_query>.sql"
```

If the function is in the DuckDB shared library, use the `.so` path instead of the binary.

### 5b. Compare operator latency baseline vs JIT

Run the same bpftrace script under both configs. The histogram shows:
- Whether JIT reduces per-invocation latency
- Whether the distribution is bimodal (cache hit/miss pattern)

### 5c. Count function calls

```bash
bpftrace -e '
uprobe:build_reldebug/aqp_middleware:<mangled_function_name> {
    @calls = count();
}' -c "$BINARY ... $QUERY_DIR/<heavy_query>.sql"
```

High call count x moderate latency can dominate over low call count x high latency.

---

## Interpreting Results and Optimization Mapping

### Decision tree

```
perf stat shows IPC < 1.0?
├── Yes (memory-bound)
│   ├── VTune shows DRAM Bound % high (or LLC miss ratio > 20%)
│   │   ├── perf record -e LLC-load-misses shows HashJoin*Probe*
│   │   │   → Try: --jit-prefetch, --jit-batch-probe, --jit-fusion-probe
│   │   ├── perf record shows TableScan*/ColumnData*
│   │   │   → Try: scan prefetching, columnar layout optimization
│   │   └── perf record shows Aggregate*/Sort*
│   │       → Try: reduce working set, partition aggregation
│   ├── VTune shows L1 Bound or L2 Bound % high (or L1 miss ratio > 10%)
│   │   → Try: struct packing, loop tiling, --jit-fusion-build, --jit-fusion-probe
│   ├── VTune shows L3 Bound % high
│   │   → Try: reduce working set to fit L3, --jit-payload-prune
│   └── TLB-bound (dTLB miss ratio > 1%)
│       → Try: huge pages (madvise), reduce pointer chasing
└── No (compute-bound, IPC > 2.0)
    ├── perf record shows ExpressionExecutor*
    │   → Try: --jit-level=expr or higher, --jit-simd=auto
    ├── perf record shows Filter*
    │   → Try: expression JIT compilation
    └── perf record shows ir_to_llvm/LLVM*
        → JIT compilation overhead too high; try caching (--jit-cache)
```

### JIT optimization flag reference

| Flag | What it targets | When to use |
|------|----------------|-------------|
| `--jit-prefetch` | Software prefetch for hash table probes | DRAM-bound / LLC-miss-bound on HASH_JOIN |
| `--jit-prefetch=<distance>` | Prefetch distance tuning | Fine-tune after confirming prefetch helps |
| `--jit-batch-probe` | Batch hash table lookups for locality | DRAM-bound, high dTLB misses on HASH_JOIN |
| `--jit-fusion-probe` | Fuse probe with downstream operator | L1/L2-miss-bound, reuse hot cache lines |
| `--jit-fusion-build` | Fuse build-side operations | L1/L2-miss-bound on HASH_JOIN build |
| `--jit-inline-hash` | Inline hash function | Compute-bound on hashing |
| `--jit-payload-prune` | Remove unused columns from hash table | L3-bound, reduce working set |
| `--jit-simd=auto` | Enable SIMD vectorization | Compute-bound on expression evaluation |
| `--jit-opt=o1/o2/o3` | LLVM optimization level | Balance compile time vs runtime improvement |
| `--jit-cache` | Cache compiled JIT code | High JIT compilation overhead |
