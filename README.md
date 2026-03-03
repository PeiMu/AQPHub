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
./build_release/aqp_middleware
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
./build_release/aqp_middleware
--engine=duckdb \
--db="/home/pei/Project/duckdb_132/measure/imdb.db" \
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
--fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
--split=node-based \
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
../build/aqp_middleware \
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
../build/aqp_middleware \
--engine=mariadb \
--db="host=localhost dbname=imdb user=imdb" \
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
./build_release/aqp_middleware
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
./build_release/aqp_middleware
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
./build_release/aqp_middleware
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
./build_release/aqp_middleware
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

It can also run the whole benchmark.

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