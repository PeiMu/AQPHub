#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

mkdir -p job_result/
rm -rf compile.log

engine=$1
split=$2

########################################
# Start / Stop PostgreSQL
########################################
pg_start() {
  ${PG_BIN}/pg_ctl start -l "${PG_LOG}" -D "${PG_DATA}"
}
pg_stop() {
  ${PG_BIN}/pg_ctl stop -D "${PG_DATA}" -m smart -s
}
rm_pg_log() {
  rm -f "${PG_LOG}"
}

cleanup() {
    if [[ "$engine" == "postgres" ]]; then
        pg_stop
    fi
}
trap cleanup EXIT

########################################
# Start Engine
########################################
if [[ "$engine" == "postgres" ]]; then
    pg_start
else
    echo "ERROR: Only postgres engine supported (got: $engine)"
    exit 1
fi

########################################
# ANALYZE — refresh stats before benchmarking
########################################
echo "ANALYZING..."
${PG_BIN}/psql -d "${PG_CONN}" -c "ANALYZE;"
echo "ANALYZE done"

# Hand off to the hyperfine sweep
cd "${SCRIPT_DIR}" && bash ./hyperfine_job_bita.sh "${engine}" "${split}"
