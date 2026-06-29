## Prerequisites

| Dependency | Minimum version | Install (Debian/Ubuntu) |
|------------|----------------|------------------------|
| CMake | 3.25 | `sudo apt install cmake` |
| Clang/LLVM | 20.1 | [apt.llvm.org](https://apt.llvm.org/) — `sudo apt install clang-20 llvm-20-dev libmlir-20-dev mlir-20-tools` |
| nlohmann-json | 3.x | `sudo apt install nlohmann-json3-dev` |
| Ninja (optional) | 1.10+ | `sudo apt install ninja-build` |
| Python | 3.9+ | `sudo apt install python3` (for analysis scripts in `measure/`) |
| Boost | 1.83+ | `sudo apt-get install libboost-context1.83-dev` (required by LingoDB adapter) |
| Apache Arrow | — | `sudo apt install libarrow-dev` (required by LingoDB adapter) |

Clang-20 and LLVM-20 are required because the JIT compiler and LingoDB adapter depend on LLVM 20 APIs and MLIR 20 dialects.

## Clone Project
```bash
git clone --recurse-submodules git@github.com:PeiMu/AQPHub.git
```

## How to compile
```bash
CC=clang-20 CXX=clang++-20 cmake -S . -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug -j$(nproc)

CC=clang-20 CXX=clang++-20 cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release -j$(nproc)
```

## Configuration

It can select different engines and split strategies.

### Config by engines

#### PostgreSQL
```bash
./build_release/aqp_middleware \
--engine=postgresql \
--db="host=localhost port=5432 dbname=imdb user=pei" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=relationship-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

#### DuckDB
```bash
./build_release/aqp_middleware \
--engine=duckdb \
--db="/home/pei/Project/duckdb/measure/imdb.db" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=relationship-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

#### Umbra
```bash
# start the docker
docker run \
--name umbra_middleware \
--network=host \
-v umbra-db:/var/db \
-v /tmp:/tmp \
--ulimit nofile=1048576:1048576 \
--ulimit memlock=8388608:8388608 \
umbradb/umbra:latest \
umbra-server --address 0.0.0.0 --port 15432 /var/db/imdb.db

# run the aqp_middleware
../build_release/aqp_middleware \
--engine=umbra \
--db="host=localhost port=15432 user=postgres password=postgres" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=relationship-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

#### MariaDB
```bash
../build_release/aqp_middleware \
--engine=mariadb \
--db="host=localhost dbname=imdb user=imdb" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=relationship-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

#### OpenGauss
We installed OpenGauss by `sudo apt install opengauss`, making it rely on some libs, e.g., `lib*_gauss.*` and `libpq.so.5.5`. But the PostgreSQL's libpq version we used is `libpq.so.5.12`.
To separate the libpq library from PostgreSQL to OpenGauss, we move the OpenGauss-related libraries to a separate directory, e.g., `$HOME/gauss_compat_libs`.
Then we need to add it to the `LD_LIBRARY_PATH`.

```bash
env LD_LIBRARY_PATH=$HOME/gauss_compat_libs ./build_release/aqp_middleware \
--engine=opengauss \
--db="host=localhost port=7654 dbname=imdb user=imdb password=imdb_132" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=relationship-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

When coding with IDEs, e.g., Clion, we need to specify the LD_LIBRARY_PATH as `LD_LIBRARY_PATH=$HOME/gauss_compat_libs`

### Config by split strategies

#### relationship-center
```bash
./build_release/aqp_middleware \
--engine=postgresql \
--db="host=localhost port=5432 dbname=imdb user=pei" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=relationship-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

Note: this split strategy depends on the estimation of each "cluster", but the MariaDB's estimator is very bad.
Thus we can specify it a helper estimator engine with path, 
e.g., `--estimator=postgres --helper-db-path="host=localhost port=5432 dbname=imdb user=pei"`

#### entity-center
```bash
./build_release/aqp_middleware \
--engine=postgresql \
--db="host=localhost port=5432 dbname=imdb user=pei" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=entity-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

#### min-subquery
```bash
./build_release/aqp_middleware \
--engine=postgresql \
--db="host=localhost port=5432 dbname=imdb user=pei" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=min-subquery \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

#### node-based
```bash
./build_release/aqp_middleware \
--engine=postgresql \
--db="host=localhost port=5432 dbname=imdb user=pei" \
--helper-db-path="/home/pei/Project/duckdb/measure/imdb.db" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=node-based \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```
Note: it is a bit tricky that there are some bugs with the `node-based` strategy, 
and we need to specify a duckdb's database path to avoid these bugs 

### JIT

Requires building with LLVM (`-DHAVE_LLVM=ON`). The JIT compiler compiles SQL operators from the AQP IR into native machine code using LLVM ORC LLJIT.

#### JIT Level

`--jit-level=<level>` selects the compilation granularity:

| Level | Flag value | Description |
|-------|-----------|-------------|
| None | `none` | JIT disabled (interpreter) |
| Expression | `expr` | Compile individual filter expressions |
| Operator | `operator` | Compile whole operators (filter, projection, hash build/probe, aggregate) |
| Pipeline | `pipeline` | Fuse adjacent operators into single compiled functions (multi-probe chain fusion) |
| Query | `query` | Compile entire sub-queries (scan→filters/probes/projections→sink in one LLVM module) |

#### Compile Mode

`--compile-mode=<mode>` selects the JIT backend:

| Flag value | Description |
|-----------|-------------|
| `llvm` | Full quality LLVM O2 (default) |
| `fastisel` | LLVM O0 + FastISel (skips optimization passes, faster compile) |
| `tpde` | TPDE fast codegen (fastest compile, lower code quality) |

#### Speculative JIT

`--spec-jit=<mode>` controls speculative background compilation (node-based split only):

| Flag value | Description |
|-----------|-------------|
| `off` | Disabled (default) |
| `recompile` | Bg compile during execute(i); on miss, inline recompile with TPDE |
| `interpret` | Bg compile during execute(i); on miss, skip JIT (interpreter) |

The bg spec compile always uses full quality LLVM O2 (compile time is free — overlaps execution). Miss recompile always uses TPDE regardless of `--compile-mode`.

#### JIT SIMD

`--jit-simd=<isa>` selects the SIMD instruction set:

| Flag value | Description |
|-----------|-------------|
| `off` | SIMD disabled |
| `sse2` | SSE2 (128-bit) |
| `avx` | AVX (256-bit) |
| `avx2` | AVX2 (256-bit with integer ops) |
| `avx512` | AVX-512 (512-bit) |
| `auto` | Auto-detect best available ISA |

#### Per-Optimization Flags

These flags control individual pipeline/query-level optimizations.

| Flag | Description |
|------|-------------|
| `--jit-payload-prune` / `--no-jit-payload-prune` | Hash build payload pruning. Only copies columns referenced downstream into the hash table payload. |
| `--jit-prefetch` / `--jit-prefetch=<distance>` / `--no-jit-prefetch` | Software prefetching for hash table access. Default distance is 8. |
| `--jit-batch-probe` / `--no-jit-batch-probe` | Batch/vectorized hash probe (ROF two-stage). |
| `--jit-skip-hash-cmp` / `--no-jit-skip-hash-cmp` | Skip stored-hash comparison for integer keys. |
| `--jit-cache` / `--no-jit-cache` | In-memory compiled-object cache (debug flag, default **off**). |

#### Example

```bash
./build_release/aqp_middleware \
  --engine=duckdb \
  --db="/home/pei/Project/duckdb/measure/imdb.db" \
  --schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
  --fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
  --split=node-based \
  --jit-level=query --compile-mode=tpde \
  --storage-plan --storage-cache=/tmp/imdb_storage_plan.cache \
  --no-analyze \
  /home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

### Whole benchmark

It can also run the whole benchmark. For now we only support JOB+IMDB.

E.g.,
```bash
--engine=postgresql
--db="host=localhost port=5432 dbname=imdb user=pei"
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql
--split=relationshipcenter
--check-correctness
--benchmark
--debug
/home/pei/Project/benchmarks/imdb_job-postgres/queries
```

### Disable updating cardinality

There's a configuration option `--no-update-temp-card` to disable the feature of updating cardinality. 
By default, the cardinality updater is enabled.

### Combine sub-sqls

We have the feature of combining all the sub-SQLs into a whole SQL, but keep their execution order, by enabling `--combine-sub-plans`.
We did this by first run the AQP with split strategies and get sub-SQLs. 
Then replace the "temp tables" with the corresponding sub-SQL by `CREATE TEMP TABLE temp AS xxx`.

### Disable analyze

By default, the middleware runs `ANALYZE` for each execution. 
When measuring performance, we have the `ANALYZE` at the beginning of the measurement script.
Thus no need to rerun the `ANALYZE` inside of the middleware, and we can disable it by `--no-analyze`.

### Print SQL only

User can use `--print-sql` to check the vanilla SQL and the generated sub-SQLs (or the combined whole SQL with `--combine-sub-plans`).

### Check correctness

Now we only check if there's any error report by having `--check-correctness`. 
We plan to collect the correct result by running the vanilla query first as golden, 
then compare it with the splitted result. 

### Performance breakdown

We use `std::chrono::high_resolution_clock::time_point` to measure the function-level performance breakdown.
you can enable this by `--timing`.
The time will be saved in `time_log.csv`. 
You can also run script `bash ./measure_breakdown_time_job.sh` to measure all engines with all split strategies.

### Debug print

You can use `--debug` to print out the necessary log.

### Help

You can check all the config options by `--help`.

## Support of different dataset

Currently we support two benchmarks: "JOB+IMDB" and "DSB".
All the examples above are using JOB benchmark. 
For "DSB", the config is, e.g.,


### PostgreSQL

```bash
--engine=postgresql \
--db="host=localhost port=5432 dbname=dsb_10 user=postgres" \
--schema=/home/pei/Project/benchmarks/dsb-postgres/scripts/create_tables.sql \
--fkeys=/home/pei/Project/benchmarks/dsb-postgres/code/tools/tpcds_ri.sql \
--split=relationship-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/dsb-postgres/code/tools/1_instance_out_wo_multi_block/1/query013/query013_0.sql
```

### DuckDB
```bash
--engine=duckdb \
--db="/home/pei/Project/duckdb/measure/dsb_10.db" \
--schema=/home/pei/Project/benchmarks/dsb-postgres/scripts/create_tables.sql \
--fkeys=/home/pei/Project/benchmarks/dsb-postgres/code/tools/tpcds_ri.sql \
--split=relationship-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/dsb-postgres/code/tools/1_instance_out_wo_multi_block/1/query013/query013_0.sql
```

### Umbra

```bash
# start the docker
docker run \
--name umbra_middleware \
--network=host \
-v umbra-db:/var/db \
-v /tmp:/tmp \
--ulimit nofile=1048576:1048576 \
--ulimit memlock=8388608:8388608 \
umbradb/umbra:latest \
umbra-server --address 0.0.0.0 /var/db/dsb_10.db

# run the aqp_middleware
--engine=umbra \
--db="host=localhost port=15432 user=postgres password=postgres" \
--schema=/home/pei/Project/benchmarks/dsb-postgres/scripts/create_tables.sql \
--fkeys=/home/pei/Project/benchmarks/dsb-postgres/scripts/tpcds_ri_umbra.sql \
--split=relationship-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/dsb-postgres/code/tools/1_instance_out_wo_multi_block/1/query013/query013_0.sql
```

### MariaDB

```bash
--engine=mariadb \
--db="host=localhost dbname=dsb_10 user=dsb_10" \
--schema=/home/pei/Project/benchmarks/dsb-postgres/scripts/create_tables.sql \
--fkeys=/home/pei/Project/benchmarks/dsb-postgres/scripts/tpcds_ri_mariadb \
--split=relationship-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/dsb-postgres/code/tools/1_instance_out_wo_multi_block/1/query013/query013_0.sql
```

## Measurement Scripts

Go to directory `measure/`. The scripts accept the following positional arguments:

```
$1   engine          duckdb / postgres / umbra / mariadb / opengauss / lingodb
$2   split           none / relationship-center / node-based
$3   jit_level       none / expr / operator / pipeline / query  (lingodb: llvm / tpde)
$4   jit_simd        off / none / auto                          (default: off)
$5   payload_prune   on / off                                   (default: on)
$6   prefetch        on / off / <distance>                      (default: on)
$7   batch_probe     on / off                                   (default: on)
$8   skip_hash_cmp   on / off                                   (default: on)
$9   jit_cache       off / on                                   (default: off)
$10  spec_jit        off / recompile / interpret                (default: off)
$11  compile_mode    off / fastisel / tpde                       (default: off = llvm)
$12  tune_config     <path to JSON>                             (default: none)
```

Arguments 5-12 are optional.

### Run a single benchmark pass

```bash
bash ./run_job.sh duckdb none pipeline none
```

### Measure performance breakdown (per-query timing)

```bash
bash ./measure_breakdown_time_job.sh duckdb node-based query none
```

### Examples with compile mode and spec-jit

```bash
# Query-jit with TPDE backend
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off off tpde

# Query-jit with spec-jit=recompile (bg=LLVM O2, miss=TPDE)
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off recompile tpde

# Per-subquery tuned config
bash ./measure_breakdown_time_job.sh duckdb node-based query none \
    on on on on off off off job_result/tuned_per_subquery_node-based.json
```

Log filenames encode the active flags, e.g., `duckdb_node-based_query_none_fctpde_breakdown_time_log.csv`.

### Analysis Scripts (measure/*.py)

| Script | Usage | Description |
|--------|-------|-------------|
| `tune_per_subquery.py` | `python3 tune_per_subquery.py [split]` | Pick best config per (query, sub-query), write tuned JSON |
| `show_all_configs.py` | `python3 show_all_configs.py [split]` | Summary table: overhead / jit / exe / total across all configs |
| `verify_tuned.py` | `python3 verify_tuned.py [split]` | Compare measured vs predicted per-query totals |
| `verify_tuned_detail.py` | `python3 verify_tuned_detail.py [query] [split]` | Per-sub-query drill-down for one query |
| `find_top_queries.py` | `python3 find_top_queries.py [path] [--top=N]` | Rank queries by slowest median total |
| `tune_kernel_threshold.py` | `python3 tune_kernel_threshold.py [dir]` | Kernel-path threshold analysis (PipelineKernel vs DuckDB) |

### Native engine scripts

We also provide scripts for running the native Umbra and MariaDB.

```bash
bash ./run_umbra.sh
bash ./run_mariadb.sh
```

Or measure their performance.

```bash
bash ./measure_umbra.sh
bash ./measure_mariadb.sh
```

## Tuning

### Kernel Threshold Tuning

The middleware includes a kernel execution path that uses pre-built flat column arrays and CSR indexes to execute sub-queries directly, bypassing the SQL engine's hash join and decompression. For each sub-query, the system decides whether to use the kernel or fall back to the SQL engine (e.g., DuckDB). The decision depends on sub-query features: scan table size, number of joins, number of filters, and number of output columns.

The optimal threshold must be tuned empirically because the kernel and SQL engine have different performance profiles: the kernel excels on filtered scans with CSR joins but may be slower on patterns with many joins or very small tables where DuckDB's vectorized execution has lower overhead.

#### How to tune

1. Build with storage plan support and run the tuning benchmark:

```bash
cd measure/
bash tune_kernel_threshold.sh
```

This runs all 113 JOB queries in 4 configurations: {node-based, relationship-center} x {kernel-enabled, kernel-disabled}, each with `--repeat=5` (first 2 as warmup). Output CSVs go to `measure/tuning_data/`.

2. Analyze the results:

```bash
python3 tune_kernel_threshold.py tuning_data/
```

This matches kernel vs DuckDB times per sub-query, reports which features predict kernel wins, and recommends a threshold formula.

#### When to retune

- **New split strategy**: Recommended but not required. The threshold is based on sub-query features (scan_rows, num_joins, etc.), not the strategy itself. A new strategy produces different sub-query patterns that may not be covered by existing tuning data. Add the new strategy to the `STRATEGIES` variable in `tune_kernel_threshold.sh` and rerun.
- **New engine**: Required if kernel support is extended beyond DuckDB. The kernel competes against the engine's own execution, so a slower engine (e.g., PostgreSQL) shifts the threshold in the kernel's favor. Currently the kernel path is DuckDB-only (`engine == BackendEngine::DUCKDB`).
- **New hardware**: Recommended. Cache sizes, core counts, and memory bandwidth affect the crossover point.
- **Schema/data changes**: Recommended if table sizes change significantly.

#### CLI flags

| Flag | Description |
|------|-------------|
| `--tuning` | Enable per-sub-query feature + timing logging (zero overhead when disabled) |
| `--no-kernel` | Force SQL engine path for all sub-queries (for collecting baseline comparison data) |
| `--storage-cache=<path>` | Binary cache file for flat arrays + CSR indexes (avoids rebuilding on each run) |

### Per-Subquery JIT Config Tuning

Different sub-queries within a split query benefit from different JIT configurations. For example, sub-query 0 might run fastest with query-jit + TPDE backend, while sub-query 1 runs fastest with operator-jit. Per-subquery tuning finds the optimal config for each (query, sub-query index) pair.

#### Tunable flags

| Flag | Values | Description |
|------|--------|-------------|
| jit level | none (interp), expr, operator, pipeline, query | Which compilation level to use |
| simd | off, auto | SIMD vectorization for expr/operator/pipeline-jit |
| compile_mode | llvm (0), fastisel (1), tpde (2) | Compile backend (--compile-mode=llvm, fastisel, or tpde) |
| payload_prune | on, off | Prune unused payload columns in pipeline-jit |
| prefetch | on, off | Software prefetch in pipeline-jit probe |
| batch_probe | on, off | ROF two-stage batch probe in pipeline-jit |
| skip_hash_cmp | on, off | Skip stored-hash comparison for integer keys |

#### How to tune

1. Collect breakdown CSVs for each candidate config (all with `--spec-jit=off`):

```bash
cd measure/
# Each config produces a CSV in job_result/
bash measure_breakdown_time_job.sh duckdb node-based none none       # interp
bash measure_breakdown_time_job.sh duckdb node-based expr none  on on on on off off  # expr
bash measure_breakdown_time_job.sh duckdb node-based expr auto  on on on on off off  # expr_simd
bash measure_breakdown_time_job.sh duckdb node-based operator none on on on on off off  # operator
bash measure_breakdown_time_job.sh duckdb node-based operator auto on on on on off off  # operator_simd
bash measure_breakdown_time_job.sh duckdb node-based pipeline none on on on on off off  # pipeline
bash measure_breakdown_time_job.sh duckdb node-based pipeline auto on on on on off off  # pipeline_simd
bash measure_breakdown_time_job.sh duckdb node-based query none on on on on off off  # query_full
bash measure_breakdown_time_job.sh duckdb node-based query none on on on on off off fastisel  # query_fastisel
bash measure_breakdown_time_job.sh duckdb node-based query none on on on on off off tpde  # query_tpde
```

2. Run the tuning script:

```bash
python3 tune_per_subquery.py [split]
```

This reads all available breakdown CSVs, picks the config with the lowest jit+execute time for each sub-query, and writes a JSON file to `job_result/tuned_per_subquery_<split>.json`.

3. Measure the tuned config:

```bash
bash measure_breakdown_time_job.sh duckdb node-based query none \
    on on on on off off off \
    job_result/tuned_per_subquery_node-based.json
```

4. Verify:

```bash
python3 show_all_configs.py         # compare suite totals across all configs
python3 verify_tuned.py             # compare measured vs predicted per query
python3 verify_tuned_detail.py 10a  # per-subquery detail for a specific query
```

#### JSON format

The tune JSON maps query name → sub-query index → config:

```json
{
  "10a": {
    "0": {"config": "query_tpde", "compile_mode": 2, "total_ms": 3.749},
    "1": {"config": "query_tpde", "compile_mode": 2, "total_ms": 3.963},
    "3": {"config": "interp", "total_ms": 0.738},
    "4": {"config": "operator", "total_ms": 0.914}
  }
}
```

Fields: `config` (required, matched by `ParseTuneLabel`), `total_ms` (predicted time), and optional flag overrides (`compile_mode`, `simd`, `payload_prune`, `prefetch`, `batch_probe`, `skip_hash_cmp`). Absent flags use the global CLI defaults.

#### Adding new configs

To sweep additional flag combinations:

1. Run the measurement with the desired flags (produces a new CSV in `job_result/`).
2. Add an entry to the `make_configs()` function in `tune_per_subquery.py` with the CSV filename and the flag values.
3. Re-run `python3 tune_per_subquery.py` to regenerate the JSON.

#### Combining with speculative JIT

The tune config is orthogonal to `--spec-jit`. The tune config determines *what* to compile for each sub-query; spec-jit determines *when* (eagerly in the background during the previous sub-query's execution). Using both together can hide the compile latency of config switches.

When both are active, the speculative compiler automatically looks ahead in the tune config to compile the *next* sub-query with its tuned flags (jit level, backend, SIMD). If the next sub-query is tuned to "interp" (no JIT), the speculative launch is skipped entirely. The main-thread JIT compiler is also recreated on-demand when the tune config switches between backends or SIMD modes.

## Web Interface

We provide a web interface at https://github.com/bitaasudeh/aqp-web-interface

## Citation

TBD
