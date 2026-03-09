## Clone Project
```bash
git clone --recurse-submodules git@github.com:PeiMu/AQP_middleware.git
```

## How to compile
```bash
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
--db="/home/pei/Project/duckdb_132/measure/imdb.db" \
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
umbra-server --address 0.0.0.0 /var/db/imdb.db

# run the aqp_middleware
../build_release/aqp_middleware \
--engine=umbra \
--db="host=localhost port=5432 user=postgres password=postgres" \
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
```bash
./build_release/aqp_middleware \
--engine=opengauss \
--db="host=localhost port=7654 dbname=imdb user=imdb password=imdb_132" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=relationship-center \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

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
--helper-db-path="/home/pei/Project/duckdb_132/measure/imdb.db" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=node-based \
--check-correctness \
--debug \
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```
Note: it is a bit tricky that there are some bugs with the `node-based` strategy, 
and we need to specify a duckdb's database path to avoid these bugs 

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

## Measurement Scripts
Go to directory `measure/`, there is script to run with either engine/split_strategy

```bash
bash ./run_job.sh duckdb node-based
```

Or measure the performance.

```bash
bash ./measure_job.sh duckdb node-based
```


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