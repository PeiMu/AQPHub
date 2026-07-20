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

mkdir -p "${result_dir}"/
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

########################################
# Start / Stop Umbra
########################################
container_name="umbra_benchmark"
start_umbra() {
    echo "Starting Umbra docker (in-memory via tmpfs)..."

    docker run -d \
        --name "$container_name" \
        --network=host \
        --tmpfs /var/db:rw,size=16g \
        -v /tmp:/tmp \
        -v "$DSB_PATH/code/tools/out_${DSB_SF}/csv":/benchmark/csv:ro \
        --ulimit nofile=1048576:1048576 \
        --ulimit memlock=8388608:8388608 \
        umbradb/umbra:latest \
        umbra-server --address 0.0.0.0 --port 15432 /var/db/dsb_${DSB_SF}.db >/dev/null

    wait_for_umbra
    load_umbra_dsb_data
}

load_umbra_dsb_data() {
    echo "Loading schema and CSV data into Umbra..."
    PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres \
        -f "$DSB_SCHEMA"
    PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres \
        -f "$DSB_IMPORT_CSV"
    echo "Data loading done."
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

########################################
# Start / Stop MariaDB
########################################
mariadb_start() {
    sudo systemctl start mariadb
}

mariadb_stop() {
    sudo systemctl stop mariadb
}


########################################
# Start / Stop OpenGauss
########################################
opengauss_start() {
    sudo systemctl start opengauss
}

opengauss_stop() {
    sudo systemctl stop opengauss
}

cleanup() {
    if [[ "$engine" == "umbra" ]]; then
        stop_umbra
    elif [[ "$engine" == "mariadb" ]]; then
        mariadb_stop
        pg_stop
    elif [[ "$engine" == "postgres" ]]; then
        pg_stop
    elif [[ "$engine" == "opengauss" ]]; then
        opengauss_stop
    fi
}
trap cleanup EXIT


#echo "compiling..."
#bash ./compile.sh >> compile.log 2>&1
#echo "compilation done"

########################################
# Start Engine
########################################
if [[ "$engine" == "umbra" ]]; then
    start_umbra
elif [[ "$engine" == "mariadb" ]]; then
    mariadb_start
    pg_start
elif [[ "$engine" == "postgres" ]]; then
    pg_start
elif [[ "$engine" == "opengauss" ]]; then
    opengauss_start
fi

########################################
# ANALYZE
########################################
echo "ANALYZING..."
if [[ "$engine" == "umbra" ]]; then
    PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres -c "ANALYZE;"
elif [[ "$engine" == "mariadb" ]]; then
    mariadb -u dsb_${DSB_SF} -D dsb_${DSB_SF} < "${DSB_PATH}/analyze_mariadb_dsb_table.sql"
elif [[ "$engine" == "postgres" ]]; then
    psql -U postgres -d dsb_${DSB_SF} -c "ANALYZE;"
elif [[ "$engine" == "opengauss" ]]; then
    sudo -i -u opengauss gsql -d dsb_${DSB_SF} -U dsb_${DSB_SF} --host=localhost -p 7654 -W dsb_${DSB_SF} -c "ANALYZE;"
fi
echo "ANALYZE done"

cd "${SCRIPT_DIR}" && bash ./hyperfine_dsb.sh "${engine}" "${split}"

#mv compile.log job_result/.
