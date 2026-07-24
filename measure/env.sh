#!/usr/bin/env bash
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${PROJECT}/build_release/aqp_middleware"

#JOB_PATH="${JOB_PATH:-}"
#DSB_PATH="${DSB_PATH:-}"
IMDB_BENCH="${JOB_PATH}"

STORAGE_CACHE="/tmp/imdb_storage_plan_pg.cache"

DUCKDB_DB="${DUCKDB_DB:-/home/pei/Project/duckdb/measure/imdb.db}"
DSB_DUCKDB_DB="${DSB_DUCKDB_DB:-/home/pei/Project/duckdb/measure/dsb_10.db}"

PG_CONN="${PG_CONN:-host=localhost port=5432 dbname=imdb user=imdb}"
PG_HOME="${PG_HOME:-/home/pei/Project/project_bins}"
PG_BIN="${PG_BIN:-${PG_HOME}/bin}"
PG_DATA="${PG_DATA:-${PG_HOME}/data_18_3}"
PG_LOG="${PG_LOG:-${PG_HOME}/logfile}"

export LD_LIBRARY_PATH=${PG_HOME}/lib:${LD_LIBRARY_PATH:-}

UMBRA_CONN="${UMBRA_CONN:-}"
MARIADB_CONN="${MARIADB_CONN:-}"
OPENGAUSS_CONN="${OPENGAUSS_CONN:-}"
