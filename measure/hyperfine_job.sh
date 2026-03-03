#!/bin/bash

engine=$1
split=$2

log_name=aqp_middleware_${engine}_job.csv
if [[ "$engine" == "mariadb" ]]; then
    dir="$JOB_PATH/mariadb_queries"
else 
    dir="$JOB_PATH/queries"
fi

rm -rf temp.csv

########################################
# DB connection
########################################
if [[ "$engine" == "postgres" ]]; then
    db_conn="host=localhost port=5432 dbname=imdb user=pei"

elif [[ "$engine" == "duckdb" ]]; then
    db_conn="/home/pei/Project/duckdb_132/measure/imdb.db"

elif [[ "$engine" == "umbra" ]]; then
    db_conn="host=localhost port=5432 user=postgres password=postgres"

elif [[ "$engine" == "mariadb" ]]; then
    db_conn="host=localhost dbname=imdb user=imdb"

else
    echo "Unknown engine: $engine"
    exit 1
fi

# For node-based split on non-DuckDB backends, pass the DuckDB helper DB
# for planning.  For DuckDB itself the flag is unused.
helper_db_arg=""
if [[ "$split" == "node-based" && "$engine" != "duckdb" ]]; then
    helper_db_path="/home/pei/Project/duckdb_132/measure/imdb.db"
    helper_db_arg="--helper-db-path=${helper_db_path}"
elif [[ "$engine" == "mariadb" ]]; then
    helper_db_path="host=localhost port=5432 dbname=imdb user=pei"
    helper_db_arg="--helper-db-path=${helper_db_path} --estimator=postgres"
fi

rm -f "${log_name}"
rm -f "job_result/${log_name}"

if [[ "$engine" == "mariadb" ]]; then
    warmup=1
    iteration=3
else
    warmup=5
    iteration=10
fi

for sql in "${dir}"/*.sql; do
    echo "Running benchmark for ${sql}..."

    hyperfine --warmup ${warmup} --runs ${iteration} --export-csv temp.csv \
    "../build/aqp_middleware --engine=${engine} \
    --db=\"${db_conn}\" \
    \"${helper_db_arg}\" \
    --schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
    --fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
    --split=\"${split}\" --no-analyze ${sql}"
    cat temp.csv >> "${log_name}"
done

mkdir -p job_result
mv "${log_name}" job_result/
rm -rf temp.csv

