#!/usr/bin/env bash
#
# Machine-specific paths — Bita's PG 18.3 setup.
# See env.sh.template for documentation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${PROJECT}/build_release/aqp_middleware"

JOB_PATH="${JOB_PATH:-/home/bita/Project/benchmarks/JOB4AQP}"
DSB_PATH="${DSB_PATH:-}"
IMDB_BENCH="${JOB_PATH}"

STORAGE_CACHE="/tmp/imdb_storage_plan_pg.cache"

DUCKDB_DB="${DUCKDB_DB:-/home/bita/Project/duckdb/measure/imdb.db}"
DSB_DUCKDB_DB="${DSB_DUCKDB_DB:-}"

PG_CONN="${PG_CONN:-host=localhost port=5432 dbname=imdb user=bita}"
PG_HOME="${PG_HOME:-/home/bita/Project/project_bins}"
PG_BIN="${PG_BIN:-${PG_HOME}/bin}"
PG_DATA="${PG_DATA:-${PG_HOME}/data}"
PG_LOG="${PG_LOG:-${PG_HOME}/logfile}"

export LD_LIBRARY_PATH=${PG_HOME}/lib:${LD_LIBRARY_PATH:-}

UMBRA_CONN="${UMBRA_CONN:-}"
MARIADB_CONN="${MARIADB_CONN:-}"
OPENGAUSS_CONN="${OPENGAUSS_CONN:-}"
