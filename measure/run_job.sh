#!/usr/bin/env bash
set -euo pipefail

engine=$1
log_name=aqp_middleware_${engine}_job.txt
dir="$JOB_PATH/queries"
container_name="umbra_benchmark"

########################################
# DB connection
########################################
if [[ "$engine" == "postgres" ]]; then
    db_conn="host=localhost port=5432 dbname=imdb user=pei"

elif [[ "$engine" == "duckdb" ]]; then
    db_conn="/home/pei/Project/duckdb_010/measure/imdb.db"

elif [[ "$engine" == "umbra" ]]; then
    db_conn="host=localhost port=5432 user=postgres password=postgres"

else
    echo "Unknown engine: $engine"
    exit 1
fi

########################################
# Start / Stop Umbra
########################################
start_umbra() {
    echo "Starting Umbra docker..."

    docker run -d \
        --name "$container_name" \
        --network=host \
        -v umbra-db:/var/db \
        -v /tmp:/tmp \
        --ulimit nofile=1048576:1048576 \
        --ulimit memlock=8388608:8388608 \
        umbradb/umbra:latest \
        umbra-server --address 0.0.0.0 /var/db/imdb.db >/dev/null

    wait_for_umbra
}

stop_umbra() {
    echo "Stopping Umbra docker..."
    docker stop "$container_name" >/dev/null 2>&1 || true
    docker rm "$container_name" >/dev/null 2>&1 || true
}

########################################
# Start / Stop PostgreSQL
########################################
Project_path=/home/pei/Project/project_bins
pg_start() {
  pg_ctl start -l $Project_path/logfile -D $Project_path/data
}
pg_stop() {
  pg_ctl stop -D $Project_path/data -m smart -s
}
rm_pg_log() {
  rm $Project_path/logfile
}


cleanup() {
    if [[ "$engine" == "umbra" ]]; then
        stop_umbra
    else
	pg_stop
    fi
}
#trap cleanup EXIT

########################################
# Wait until Umbra is ready
########################################
wait_for_umbra() {
    echo "Waiting for Umbra to accept connections on port 5432..."
    until pg_isready -h localhost -p 5432 >/dev/null 2>&1; do
        sleep 1
    done
    echo "Umbra is ready."
}

########################################
# Prepare logs
########################################
rm -f "${log_name}"
rm -f "job_result/${log_name}"

mkdir -p job_result
shopt -s nullglob

echo "compiling..."
bash ./compile.sh >> compile.log 2>&1
echo "compilation done"

########################################
# Start Umbra if needed
########################################
#if [[ "$engine" == "umbra" ]]; then
#    start_umbra
#else
#    pg_start
#fi

########################################
# ANALYZE
########################################
echo "ANALYZING..."
if [[ "$engine" == "umbra" ]]; then
    PGPASSWORD=postgres psql -p 5432 -h localhost -U postgres -c "ANALYZE;"
else
    psql -U pei -d imdb -c "ANALYZE;"
fi
echo "ANALYZE done"

########################################
# Run benchmark
########################################
start=$(date +%s%N)
for sql in "$dir"/*.sql; do
    echo "Running benchmark for $sql..." | tee -a "$log_name"

    ../build/aqp_middleware \
        --engine="${engine}" \
        --db="${db_conn}" \
        --schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
        --fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
        --split=relationshipcenter \
        --no-analyze \
        "${sql}" \
        2>&1 | tee -a "$log_name"
done
end=$(date +%s%N)
elapsed_ns=$((end - start))
elapsed_ms=$((elapsed_ns / 1000000))

echo "${engine} runs: ${elapsed_ms} ms"

mv "${log_name}" job_result/
