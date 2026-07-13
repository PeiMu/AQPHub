#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

engine=$1
split=$2
log_name=time_log.csv
dir_1="$DSB_PATH/code/tools/1_instance_out_wo_multi_block/1/"
dir_2="$DSB_PATH/code/tools/1_instance_out_wo_multi_block/2/"
container_name="umbra_benchmark"

iteration=15 # 5 warm up, 10 runs

########################################
# DB connection
########################################
if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
    db_conn="${PG_CONN}"

elif [[ "$engine" == "duckdb" ]]; then
    db_conn="${DSB_DUCKDB_DB}"

elif [[ "$engine" == "umbra" ]]; then
    db_conn="${UMBRA_CONN}"

elif [[ "$engine" == "mariadb" ]]; then
    db_conn="${MARIADB_CONN}"

elif [[ "$engine" == "opengauss" ]]; then
    db_conn="${OPENGAUSS_CONN}"

else
    echo "Unknown engine: $engine"
    exit 1
fi

# For node-based split on non-DuckDB backends, pass the DuckDB helper DB
# for planning.  For DuckDB itself the flag is unused.
helper_db_arg=""
if [[ "$split" == "node-based" && "$engine" != "duckdb" ]]; then
    helper_db_arg="--helper-db-path=${DSB_DUCKDB_DB}"
elif [[ "$engine" == "mariadb" ]]; then
    helper_db_arg="--helper-db-path=${PG_CONN} --estimator=postgres"
fi

########################################
# Start / Stop Umbra
########################################
start_umbra() {
    echo "Starting Umbra docker (in-memory via tmpfs)..."

    docker run -d \
        --name "$container_name" \
        --network=host \
        --tmpfs /var/db:rw,size=16g \
        -v /tmp:/tmp \
        -v "$DSB_CSV_DIR":/benchmark/csv:ro \
        --ulimit nofile=1048576:1048576 \
        --ulimit memlock=8388608:8388608 \
        umbradb/umbra:latest \
        umbra-server --address 0.0.0.0 --port 15432 /var/db/dsb_10.db >/dev/null

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
    elif [[ "$engine" == "opengauss" ]]; then
        opengauss_stop
    elif [[ "$engine" == "postgres" ]]; then
	      pg_stop
    fi
}
trap cleanup EXIT

########################################
# Wait until Umbra is ready
########################################
wait_for_umbra() {
    echo "Waiting for Umbra to accept connections on port 15432..."
    until pg_isready -h localhost -p 15432 >/dev/null 2>&1; do
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

#echo "compiling..."
#bash ./compile.sh >> compile.log 2>&1
#echo "compilation done"

########################################
# Start Umbra if needed
########################################
if [[ "$engine" == "umbra" ]]; then
    start_umbra
elif [[ "$engine" == "mariadb" ]]; then
    mariadb_start
elif [[ "$engine" == "opengauss" ]]; then
    opengauss_start
elif [[ "$engine" == "postgres" ]]; then
    pg_start
fi

########################################
# ANALYZE
########################################
echo "ANALYZING..."
if [[ "$engine" == "umbra" ]]; then
    PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres -c "ANALYZE;"
elif [[ "$engine" == "mariadb" ]]; then
    mariadb -u dsb_10 -D dsb_10 < "${DSB_PATH}/analyze_mariadb_dsb_table.sql"
elif [[ "$engine" == "postgres" ]]; then
    psql -U postgres -d dsb_10 -c "ANALYZE;"
elif [[ "$engine" == "opengauss" ]]; then
    sudo -i -u opengauss gsql -d dsb_10 -U dsb_10 --host=localhost -p 7654 -W dsb_10 -c "ANALYZE;"
fi
echo "ANALYZE done"

########################################
# Run benchmark
########################################
cmd_prefix=""
if [[ "$engine" == "opengauss" ]]; then
    cmd_prefix="env LD_LIBRARY_PATH=$HOME/gauss_compat_libs"
fi

for dsb_dir in "$dir_1" "$dir_2"; do
  $cmd_prefix "${PROJECT}/build_release/aqp_middleware" \
    --engine="${engine}" \
    --db="${db_conn}" \
    "${helper_db_arg}" \
    --schema="${DSB_PATH}/scripts/create_tables.sql" \
    --fkeys="${DSB_PATH}/scripts/tpcds_ri_umbra.sql" \
    --split="${split}" \
    --timing \
    --no-analyze \
    --repeat=${iteration} \
    --benchmark \
    "${dsb_dir}"
done

mv "${log_name}" job_result/${engine}_${split}_breakdown_"${log_name}"
