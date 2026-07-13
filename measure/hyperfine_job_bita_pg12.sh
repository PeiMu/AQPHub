#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

engine=$1
split=$2

if [[ -z "$JOB_PATH" ]]; then
    echo "ERROR: JOB_PATH not set in env.sh"
    exit 1
fi

PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log_name=aqp_middleware_${engine}_${split}_job.csv
dir="$JOB_PATH/queries"

rm -rf temp.csv

########################################
# DB connection (PG 12 uses a different port)
########################################
PG12_CONN="${PG12_CONN:-host=localhost port=5433 dbname=imdb user=bita}"
if [[ "$engine" != "postgres" ]]; then
    echo "ERROR: Only postgres engine supported (got: $engine)"
    exit 1
fi

########################################
# Helper DB for node-based splitter
########################################
helper_db_arg=""
if [[ "$split" == "node-based" || "$split" == "nodebased" || "$split" == "node_based" ]]; then
    helper_db_arg="--helper-db-path=${DUCKDB_DB}"
fi

# Fresh CSV for this run
rm -f "${log_name}"
rm -f "job_result/${log_name}"

warmup=5
iteration=10

for sql in "${dir}"/*.sql; do
    echo "Running benchmark for ${sql}..."

    hyperfine --warmup ${warmup} --runs ${iteration} --export-csv temp.csv \
    "${PROJECT}/build_release/aqp_middleware --engine=${engine} \
    --db=\"${PG12_CONN}\" \
    ${helper_db_arg} \
    --schema=${JOB_PATH}/schema.sql \
    --fkeys=${JOB_PATH}/fkeys.sql \
    --split=\"${split}\" --no-analyze ${sql}"
    cat temp.csv >> "${log_name}"
done

mkdir -p job_result
mv "${log_name}" job_result/
rm -rf temp.csv

echo "===== Benchmark complete ====="
echo "Results: job_result/${log_name}"
