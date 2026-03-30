#!/usr/bin/env bash

engine=$1
split=$2
log_name=aqp_middleware_${engine}_${split}_dsb.txt
dir_1="$DSB_PATH/code/tools/1_instance_out_wo_multi_block/1/"
dir_2="$DSB_PATH/code/tools/1_instance_out_wo_multi_block/2/"
container_name="umbra_benchmark"

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
        umbra-server --address 0.0.0.0 --port 15432 /var/db/dsb_10.db >/dev/null

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
    else
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
rm -f "dsb_result/${log_name}"

mkdir -p dsb_result
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
else
    pg_start
fi

########################################
# ANALYZE
########################################
echo "ANALYZING..."
if [[ "$engine" == "umbra" ]]; then
    PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres -c "ANALYZE;"
elif [[ "$engine" == "mariadb" ]]; then
    mariadb -u dsb_10 -D dsb_10 < /home/pei/Project/benchmarks/dsb-postgres/analyze_mariadb_dsb_table.sql
elif [[ "$engine" == "postgres" ]]; then
    psql -U postgres -d dsb_10 -c "ANALYZE;"
elif [[ "$engine" == "opengauss" ]]; then
    sudo -i -u opengauss gsql -d dsb_10 -U dsb_10 --host=localhost -p 7654 -W dsb_10 -c "ANALYZE;"
fi
echo "ANALYZE done"

########################################
# Run benchmark
########################################
start=$(date +%s%N)
cmd_prefix=""
if [[ "$engine" == "opengauss" ]]; then
    cmd_prefix="env LD_LIBRARY_PATH=$HOME/gauss_compat_libs"
fi

for sql in $(find "$dir_1" "$dir_2" -type f -name "*.sql"); do
    echo "Running benchmark for $sql..." | tee -a "$log_name"

    $cmd_prefix ../build/aqp_middleware \
        --engine="${engine}" \
        --db="${db_conn}" \
        "${helper_db_arg}" \
        --schema=/home/pei/Project/benchmarks/dsb-postgres/scripts/create_tables.sql \
        --fkeys=/home/pei/Project/benchmarks/dsb-postgres/scripts/tpcds_ri_umbra.sql \
        --split="${split}" \
        --no-analyze \
        "${sql}" \
        2>&1 | tee -a "$log_name"
done
end=$(date +%s%N)
elapsed_ns=$((end - start))
elapsed_ms=$((elapsed_ns / 1000000))

echo "${engine} runs: ${elapsed_ms} ms"

mv "${log_name}" dsb_result/
