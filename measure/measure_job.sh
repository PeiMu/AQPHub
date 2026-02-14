#!/bin/bash

mkdir -p job_result/
rm -rf compile.log

engine=$1

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

########################################
# Start / Stop Umbra
########################################
container_name="umbra_benchmark"
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

wait_for_umbra() {
    echo "Waiting for Umbra to accept connections on port 5432..."
    until pg_isready -h localhost -p 5432 >/dev/null 2>&1; do
        sleep 1
    done
    echo "Umbra is ready."
}

cleanup() {
    if [[ "$engine" == "umbra" ]]; then
        stop_umbra
    else
        pg_stop
    fi
}
trap cleanup EXIT


echo "compiling..."
bash ./compile.sh >> compile.log 2>&1
echo "compilation done"

########################################
# Start Umbra if needed
########################################
if [[ "$engine" == "umbra" ]]; then
    start_umbra
else
    pg_start
fi

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

cd ../measure && bash ./hyperfine_job.sh "${engine}"

mv compile.log job_result/.
