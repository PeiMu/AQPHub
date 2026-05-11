#!/bin/bash

engine=$1
split=$2

log_name=aqp_middleware_${engine}_${split}_dsb.csv
dir_1="$DSB_PATH/code/tools/1_instance_out_wo_multi_block/1/"
dir_2="$DSB_PATH/code/tools/1_instance_out_wo_multi_block/2/"

rm -rf temp.csv

########################################
# DB connection
########################################
if [[ "$engine" == "postgres" ]]; then
    db_conn="host=localhost port=5432 dbname=dsb_10 user=postgres"

elif [[ "$engine" == "duckdb" ]]; then
    db_conn="/home/pei/Project/duckdb_132/measure/dsb_10.db"

elif [[ "$engine" == "umbra" ]]; then
    db_conn="host=localhost port=15432 user=postgres password=postgres"

elif [[ "$engine" == "mariadb" ]]; then
    db_conn="host=localhost dbname=dsb_10 user=dsb_10"

elif [[ "$engine" == "opengauss" ]]; then
    db_conn="host=localhost port=7654 dbname=dsb_10 user=dsb_10 password=dsb_10"

else
    echo "Unknown engine: $engine"
    exit 1
fi

# For node-based split on non-DuckDB backends, pass the DuckDB helper DB
# for planning.  For DuckDB itself the flag is unused.
helper_db_arg=""
if [[ "$split" == "node-based" && "$engine" != "duckdb" ]]; then
    helper_db_path="/home/pei/Project/duckdb_132/measure/dsb_10.db"
    helper_db_arg="--helper-db-path=${helper_db_path}"
elif [[ "$engine" == "mariadb" ]]; then
    helper_db_path="host=localhost port=5432 dbname=dsb_10 user=postgres"
    helper_db_arg="--helper-db-path=${helper_db_path} --estimator=postgres"
fi

rm -f "${log_name}"
rm -f "dsb_result/${log_name}"

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

for sql in $(find "$dir_1" "$dir_2" -type f -name "*.sql"); do
    echo "Running benchmark for ${sql}..."

    hyperfine --warmup ${warmup} --runs ${iteration} --export-csv temp.csv \
    "${cmd_prefix}../build/aqp_middleware --engine=${engine} \
    --db=\"${db_conn}\" \
    \"${helper_db_arg}\" \
    --schema=/home/pei/Project/benchmarks/dsb-postgres/scripts/create_tables.sql \
    --fkeys=/home/pei/Project/benchmarks/dsb-postgres/scripts/tpcds_ri_umbra.sql \
    --split=\"${split}\" --no-analyze ${sql}"
    cat temp.csv >> "${log_name}"
done

mkdir -p job_result
mv "${log_name}" job_result/
rm -rf temp.csv

