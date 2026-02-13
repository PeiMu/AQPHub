#!/bin/bash
set -euo pipefail

engine=$1
log_name=aqp_middleware_${engine}_job.txt
dir="$JOB_PATH/queries"

# Set the DB connection string based on the engine
if [ "$engine" == "postgres" ]; then
    db_conn="host=localhost port=5432 dbname=imdb user=pei"
elif [ "$engine" == "duckdb" ]; then
    db_conn="/home/pei/Project/duckdb_010/measure/imdb.db"
elif [ "$engine" == "umbra" ]; then
    db_conn="host=localhost port=5432 user=postgres password=postgres"
else
    echo "Unknown engine: $engine"
    exit 1
fi

rm -f "${log_name}"
rm -f "job_result/${log_name}"

for sql in "${dir}"/*.sql; do
    echo "Running benchmark for ${sql}..."
    
    ../build/aqp_middleware --engine=${engine} \
    --db="${db_conn}" \
    --schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
    --fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
    --split=relationshipcenter --no-postgres-analyze ${sql} \
    2>&1 | tee "$log_name"
done

mkdir -p job_result
mv "${log_name}" job_result/
