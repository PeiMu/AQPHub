#!/bin/bash

cd ../../duckdb_132/ && make clean && GEN=ninja ENABLE_QUERY_SPLIT=0 VERBOSE=1 make
cd ../PostgreSQL-12.3/build/ && make clean && make -j32 && sudo make install
cd ../../AQP_middleware

if [ -d "build" ]; then
  cd build/ && make clean && make -j32
else
  mkdir build/ && cd build/ && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j32
fi