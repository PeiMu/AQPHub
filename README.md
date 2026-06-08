## Clone Project
```bash
git clone --recurse-submodules git@github.com:PeiMu/AQPHub.git
```

## How to compile
```bash
sudo apt install nlohmann-json3-dev # we need json

mkdir -p build_debug && cd build_debug/
cmake -DCMAKE_BUILD_TYPE=Debug .. # requires CMake 4.0 or higher
make -j32

mkdir -p build_release && cd build_release/
cmake -DCMAKE_BUILD_TYPE=Release .. # requires CMake 4.0 or higher
make -j32
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

Requires building with LLVM 14 (`-DHAVE_LLVM=ON`). The JIT compiler compiles SQL operators from the AQP IR into native machine code using LLVM ORC LLJIT.

#### JIT Level

`--jit-level=<level>` selects the compilation granularity:

| Level | Flag value | Description |
|-------|-----------|-------------|
| None | `none` | JIT disabled |
| Expression | `expr` | Compile individual filter expressions |
| Operator | `operator` | Compile whole operators (filter, projection, hash build/probe, aggregate) |
| Pipeline | `pipeline` | Fuse adjacent operators into single compiled functions (filter+projection, filter+aggregate, filter+hash build, filter+probe+projection) |
| SQL | `sql` | Compile entire sub-plans / whole queries |

#### JIT Optimization Level

`--jit-opt=<opt>` selects the LLVM optimization level:

| Flag value | Description |
|-----------|-------------|
| `o0` | No optimization |
| `o1` | Basic optimization (default) |
| `o2` | Standard optimization |
| `o3` | Aggressive optimization |

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

These flags control individual pipeline-level optimizations. All default to **enabled** when `--jit-level` is `pipeline` or `sql`.

| Flag | Description |
|------|-------------|
| `--jit-fusion-build` / `--no-jit-fusion-build` | Filter + HashBuild fusion. Fuses filter evaluation and hash table insertion into a single loop, eliminating the intermediate DataChunk between them. |
| `--jit-fusion-probe` / `--no-jit-fusion-probe` | Filter + HashProbe + Projection fusion. Fuses filter, hash probe, and projection into a single loop on the probe side, eliminating two intermediate DataChunks. |
| `--jit-inline-hash` / `--no-jit-inline-hash` | Inline FNV-1a hash computation as LLVM IR instead of calling the `aqp_hash()` C function. Eliminates function call overhead and enables LLVM to keep the hash value in a register. |
| `--jit-payload-prune` / `--no-jit-payload-prune` | Hash build payload pruning. Only copies columns referenced downstream into the hash table payload instead of all input columns. Reduces hash table memory footprint. |
| `--jit-prefetch` / `--jit-prefetch=<distance>` / `--no-jit-prefetch` | Software prefetching for hash table access. Uses `llvm.prefetch` intrinsic to prefetch hash table slots ahead of the probe loop. Default distance is 8. |
| `--jit-batch-probe` / `--no-jit-batch-probe` | Batch/vectorized hash probe. Two-phase probe: Phase 1 computes all hashes and prefetches slots; Phase 2 probes with cache-hot slots. |
| `--jit-cache` / `--jit-cache=<path>` / `--no-jit-cache` | Cross-process compiled binary cache. Caches compiled ELF objects to disk so subsequent runs skip LLVM compilation. Default path: `~/.cache/aqp_jit/`. |

#### Example

```bash
./build_release/aqp_middleware \
  --engine=duckdb \
  --db="/home/pei/Project/duckdb/measure/imdb.db" \
  --schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
  --fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
  --split=none \
  --jit-level=pipeline --jit-opt=o2 --jit-simd=auto \
  --no-jit-batch-probe --jit-prefetch=16 --jit-cache \
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
$1  engine          duckdb / postgres / umbra / mariadb / opengauss
$2  split           none / relationship-center / entity-center / min-subquery / node-based
$3  jit_level       none / expr / operator / pipeline
$4  jit_simd        off / sse2 / avx / avx2 / avx512 / auto
$5  payload_prune   on / off                    (default: on)
$6  prefetch        on / off / <distance>       (default: on, distance=8)
$7  batch_probe     on / off                    (default: on)
$8  skip_hash_cmp   on / off                    (default: on)
$9  cache           off / on / <path>           (default: off)
```

Arguments 5-9 are optional and default to all optimizations enabled (cache off).

### Run a single benchmark pass

```bash
bash ./run_job.sh duckdb none sql o2 auto
```

### Measure performance with hyperfine

```bash
bash ./measure_job.sh duckdb none sql o2 auto
```

### Measure performance breakdown (per-query timing)

```bash
bash ./measure_breakdown_time_job.sh duckdb none pipeline o2 auto
```

### Examples with per-optimization flags

```bash
# All optimizations enabled (default)
bash ./measure_job.sh duckdb none sql o2 auto

# Disable fusion-build to isolate its contribution
bash ./measure_job.sh duckdb none sql o2 auto off

# Disable both fusion optimizations
bash ./measure_job.sh duckdb none sql o2 auto off off

# Custom prefetch distance of 16
bash ./measure_job.sh duckdb none sql o2 auto on on on on 16

# Enable disk cache
bash ./measure_job.sh duckdb none sql o2 auto on on on on on on on

# Disable batch probe only (pass preceding defaults)
bash ./run_job.sh duckdb none sql o2 auto on on on on on off

# Breakdown timing with batch probe disabled
bash ./measure_breakdown_time_job.sh duckdb none pipeline o2 auto on on on on on off
```

Log filenames encode the active flags, e.g., `aqp_middleware_duckdb_none_pipeline_auto_nopayprune_job.csv`.

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

## Kernel Threshold Tuning

The middleware includes a kernel execution path that uses pre-built flat column arrays and CSR indexes to execute sub-queries directly, bypassing the SQL engine's hash join and decompression. For each sub-query, the system decides whether to use the kernel or fall back to the SQL engine (e.g., DuckDB). The decision depends on sub-query features: scan table size, number of joins, number of filters, and number of output columns.

The optimal threshold must be tuned empirically because the kernel and SQL engine have different performance profiles: the kernel excels on filtered scans with CSR joins but may be slower on patterns with many joins or very small tables where DuckDB's vectorized execution has lower overhead.

### How to tune

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

### When to retune

- **New split strategy**: Recommended but not required. The threshold is based on sub-query features (scan_rows, num_joins, etc.), not the strategy itself. A new strategy produces different sub-query patterns that may not be covered by existing tuning data. Add the new strategy to the `STRATEGIES` variable in `tune_kernel_threshold.sh` and rerun.
- **New engine**: Required if kernel support is extended beyond DuckDB. The kernel competes against the engine's own execution, so a slower engine (e.g., PostgreSQL) shifts the threshold in the kernel's favor. Currently the kernel path is DuckDB-only (`engine == BackendEngine::DUCKDB`).
- **New hardware**: Recommended. Cache sizes, core counts, and memory bandwidth affect the crossover point.
- **Schema/data changes**: Recommended if table sizes change significantly.

### CLI flags

| Flag | Description |
|------|-------------|
| `--tuning` | Enable per-sub-query feature + timing logging (zero overhead when disabled) |
| `--no-kernel` | Force SQL engine path for all sub-queries (for collecting baseline comparison data) |
| `--storage-cache=<path>` | Binary cache file for flat arrays + CSR indexes (avoids rebuilding on each run) |

## Web Interface

We provide a web interface at https://github.com/bitaasudeh/aqp-web-interface

## Citation

TBD
