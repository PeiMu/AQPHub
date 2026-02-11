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

File structure:
```bash
├── CMakeLists.txt
├── examples
│   ├── test_duckdb.cpp
│   └── test_postgres.cpp
├── include
│   ├── db_adapter.h
│   ├── duckdb_adapter.h
│   └── postgres_adapter.h
└── src
    └── adapters
        ├── duckdb_adapter.cpp
        └── postgres_adapter.cpp
```


##Configuration

It can select different engines and split strategies.

E.g.,
```bash
--engine=postgresql
--db="host=localhost port=5432 dbname=imdb user=pei"
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql
--split=relationshipcenter
--check-correctness
--debug
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

Or
```bash
--engine=duckdb
--db="/home/pei/Project/duckdb_010/measure/imdb.db"
--schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql
--split=top_down
--check-correctness
--debug
/home/pei/Project/benchmarks/imdb_job-postgres/queries/1a.sql
```

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