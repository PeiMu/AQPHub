#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# DSB scale factor: set DSB_SF=100 to run against the SF-100 database.
# Defaults to 10, keeping all existing paths/filenames unchanged.
DSB_SF=${DSB_SF:-10}
if [[ "$DSB_SF" == "10" ]]; then
    result_dir="dsb_result"
else
    result_dir="dsb_result_sf${DSB_SF}"
fi

engine=$1
split=$2

log_name=aqp_middleware_${engine}_${split}_dsb.csv
dir="$DSB_PATH/code/tools/1_instance_out_aqp/1/"

rm -rf temp.csv

########################################
# DB connection
########################################
if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
    db_conn="${PG_CONN}"

elif [[ "$engine" == "duckdb" ]]; then
    db_conn="${DSB_DUCKDB_DB:-/home/pei/Project/duckdb/measure/dsb_${DSB_SF}.db}"

elif [[ "$engine" == "umbra" ]]; then
    db_conn="${UMBRA_CONN}"

elif [[ "$engine" == "mariadb" ]]; then
    db_conn="${MARIADB_CONN:-host=localhost dbname=dsb_${DSB_SF} user=dsb_${DSB_SF}}"

elif [[ "$engine" == "opengauss" ]]; then
    db_conn="${OPENGAUSS_CONN:-host=localhost port=7654 dbname=dsb_${DSB_SF} user=dsb_${DSB_SF} password=dsb_${DSB_SF}}"

elif [[ "$engine" == "lingodb" ]]; then
    db_conn=""

else
    echo "Unknown engine: $engine"
    exit 1
fi

# For node-based split on non-DuckDB backends, pass the DuckDB helper DB
# for planning.  For DuckDB itself the flag is unused.
helper_db_arg=""
if [[ "$split" == "node-based" && "$engine" != "duckdb" ]]; then
    helper_db_path="${DSB_DUCKDB_DB:-/home/pei/Project/duckdb/measure/dsb_${DSB_SF}.db}"
    helper_db_arg="--helper-db-path=${helper_db_path}"
elif [[ "$engine" == "mariadb" ]]; then
    helper_db_path="${PG_CONN:-host=localhost port=5432 dbname=dsb_${DSB_SF} user=postgres}"
    helper_db_arg="--helper-db-path=${helper_db_path} --estimator=postgres"
fi

rm -f "${log_name}"
rm -f "${result_dir}/${log_name}"

cmd_prefix=""
if [[ "$engine" == "opengauss" ]]; then
    cmd_prefix="LD_LIBRARY_PATH=$HOME/gauss_compat_libs "
fi

if [[ "$engine" == "mariadb" ]]; then
    warmup=1
    iteration=3
else
    warmup=5
    iteration=10
fi

# LingoDB: in-memory with CSV loading instead of --db
db_arg="--db=\"${db_conn}\""
lingodb_flags=""
if [[ "$engine" == "lingodb" ]]; then
    db_arg="--in-memory"
    lingodb_flags="--csv-dir=$DSB_PATH/code/tools/out_${DSB_SF}/lingo_db_csv"
fi

for sql in $(find "$dir" -type f -name "*.sql" | sort); do
    echo "Running benchmark for ${sql}..."

    hyperfine --warmup ${warmup} --runs ${iteration} --export-csv temp.csv \
    "${cmd_prefix}${PROJECT}/build_release/aqp_middleware --engine=${engine} \
    --db=\"${db_conn}\" \
    \"${helper_db_arg}\" \
    --schema=${DSB_PATH}/scripts/create_tables.sql \
    --fkeys=${DSB_PATH}/scripts/tpcds_ri_umbra.sql \
    --split=\"${split}\" --no-analyze ${sql}"
    cat temp.csv >> "${log_name}"
done

mkdir -p "${result_dir}"
mv "${log_name}" "${result_dir}"/
rm -rf temp.csv
