# Unit Tests

## Prerequisites

Install Google Test:
```bash
sudo apt install libgtest-dev
cd /usr/src/googletest && sudo mkdir -p build && cd build && sudo cmake .. && sudo make -j12 && sudo make install
```

## Build

```bash
cd /home/pei/Project/AQP_middleware/build_release
cmake ..
make test_storage -j12
```

## Run

```bash
cd /home/pei/Project/AQP_middleware/build_release
./test_storage                          # run all tests
./test_storage --gtest_filter='CSR*'    # run only CSR tests
./test_storage --gtest_filter='SubQueryPlan.InnerJoin*'  # run inner join tests
ctest --test-dir . -R StorageTests      # run via ctest
```

## Test coverage

- `FlatColumn`: INT32/VARCHAR access, long strings
- `FlatTable`: FindColumn lookup
- `CSRIndex`: build, lookup, out-of-range, empty, all-same-FK, negative values, 100K-row scale
- `SubQueryPlan`: semi-join, inner join, mixed INT32/VARCHAR output, chained kernel executions, high fan-out, varchar stress, multi-step semi, bitset semi, string ownership after source destruction
- `Memory`: CSR independent of FlatTable lifetime, repeat loop simulation, dangling pointer bug pattern
- `Integration`: full repeat cycle (3 kernel iterations, 5 repeats)
