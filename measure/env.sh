#!/usr/bin/env bash
#
# Machine-specific paths — Bita's PG 18.3 setup.
# See env.sh.template for documentation.

JOB_PATH="${JOB_PATH:-/home/bita/Project/benchmarks/JOB4AQP}"
DSB_PATH="${DSB_PATH:-}"
IMDB_BENCH="${IMDB_BENCH:-}"

DUCKDB_DB="${DUCKDB_DB:-/home/bita/Project/duckdb/measure/imdb.db}"
DSB_DUCKDB_DB="${DSB_DUCKDB_DB:-}"

PG_CONN="${PG_CONN:-host=localhost port=5436 dbname=imdb user=bita}"
PG_HOME="${PG_HOME:-/home/bita/Project/Postgres-18.3}"
PG_BIN="${PG_BIN:-/home/bita/Project/Postgres-18.3/install/bin}"
PG_DATA="${PG_DATA:-/home/bita/Project/Postgres-18.3/data}"
PG_LOG="${PG_LOG:-/home/bita/Project/Postgres-18.3/logfile}"

export LD_LIBRARY_PATH=/home/bita/Project/Postgres-18.3/install/lib:${LD_LIBRARY_PATH:-}

UMBRA_CONN="${UMBRA_CONN:-}"
MARIADB_CONN="${MARIADB_CONN:-}"
OPENGAUSS_CONN="${OPENGAUSS_CONN:-}"
