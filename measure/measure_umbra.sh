#!/bin/bash

mkdir -p job_result/
rm -rf compile.log

log_name=umbra_official.csv
rm -rf ${log_name}
rm -rf temp.csv

########################################
# Start / Stop Umbra
########################################
container_name="umbra_benchmark"
start_umbra() {
    echo "Starting Umbra docker..."

    docker run -d \
        --name "$container_name" \
        --network=host \
	--cpuset-cpus="0" \
        -e OMP_NUM_THREADS=1 \
        -e OMP_THREAD_LIMIT=1 \
        -e OMP_PROC_BIND=TRUE \
        -v umbra-db:/var/db \
        -v /tmp:/tmp \
        --ulimit nofile=1048576:1048576 \
        --ulimit memlock=8388608:8388608 \
        umbradb/umbra:latest \
        umbra-server --address 0.0.0.0 --port 15432 /var/db/imdb.db >/dev/null

    wait_for_umbra
}

stop_umbra() {
    echo "Stopping Umbra docker..."
    docker stop "$container_name" >/dev/null 2>&1 || true
    docker rm "$container_name" >/dev/null 2>&1 || true
}

wait_for_umbra() {
    echo "Waiting for Umbra to accept connections on port 15432..."
    until pg_isready -h localhost -p 15432 >/dev/null 2>&1; do
        sleep 1
    done
    echo "Umbra is ready."
}

cleanup() {
    stop_umbra
}
trap cleanup EXIT

start_umbra

########################################
# ANALYZE
########################################
echo "ANALYZING..."
PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres -c "ANALYZE;"
echo "ANALYZE done"

dir="$JOB_PATH/queries"
iteration=10

for sql in "${dir}"/*.sql; do
  #echo "hyperfine run ${sql}" 2>&1|tee -a ${log_name}
  hyperfine --warmup 5 --runs ${iteration} --export-csv temp.csv "PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres -f ${sql}"
  cat temp.csv >> ${log_name}
done

mv ${log_name} job_result/.
rm -rf temp.csv

