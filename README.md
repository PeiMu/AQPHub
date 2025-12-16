## How to compile
```bash
mkdir -p build && cd build/
cmake .. # requires CMake 4.0 or higher
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
