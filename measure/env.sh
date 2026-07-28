#!/usr/bin/env bash
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PROJECT}/.." && pwd)"
BINARY="${PROJECT}/build_release/aqp_middleware"

#JOB_PATH="${JOB_PATH:-}"
#DSB_PATH="${DSB_PATH:-}"

########################################
# DSB scale factor (set before sourcing env.sh to override)
########################################
DSB_SF="${DSB_SF:-10}"

########################################
# Per-benchmark PostgreSQL connections
########################################
PG_CONN_JOB="${PG_CONN_JOB:-host=localhost port=5432 dbname=imdb user=imdb}"
PG_CONN_DSB="${PG_CONN_DSB:-host=localhost port=5432 dbname=dsb_${DSB_SF} user=postgres}"

########################################
# Per-benchmark DuckDB database paths
########################################
DUCKDB_DB_JOB="${DUCKDB_DB_JOB:-${PROJECT_ROOT}/duckdb/measure/imdb.db}"
DUCKDB_DB_DSB="${DUCKDB_DB_DSB:-${PROJECT_ROOT}/duckdb/measure/dsb_${DSB_SF}.db}"

########################################
# Per-benchmark storage plan caches (DuckDB engine)
########################################
STORAGE_CACHE_DUCKDB_JOB="${STORAGE_CACHE_DUCKDB_JOB:-/tmp/imdb_storage_plan.cache}"
STORAGE_CACHE_DUCKDB_DSB="${STORAGE_CACHE_DUCKDB_DSB:-/tmp/dsb_sf${DSB_SF}_storage_plan.cache}"

########################################
# Per-benchmark storage plan caches (PostgreSQL engine)
########################################
STORAGE_CACHE_PG_JOB="${STORAGE_CACHE_PG_JOB:-/tmp/imdb_storage_plan_pg.cache}"
STORAGE_CACHE_PG_DSB="${STORAGE_CACHE_PG_DSB:-/tmp/dsb_sf${DSB_SF}_storage_plan_pg.cache}"

PG_HOME="${PG_HOME:-${PROJECT_ROOT}/project_bins}"
PG_BIN="${PG_BIN:-${PG_HOME}/bin}"
PG_DATA="${PG_DATA:-${PG_HOME}/data_18_3}"
PG_LOG="${PG_LOG:-${PG_HOME}/logfile}"

export LD_LIBRARY_PATH=${PG_HOME}/lib:${LD_LIBRARY_PATH:-}

UMBRA_CONN="${UMBRA_CONN:-host=localhost port=15432 user=postgres password=postgres}"
MARIADB_CONN="${MARIADB_CONN:-}"
OPENGAUSS_CONN="${OPENGAUSS_CONN:-}"
