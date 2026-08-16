#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bench=$1
engine=$2
split=$3
jit_level=$4
jit_simd=$5
payload_prune=${6:-on}
prefetch=${7:-on}
batch_probe=${8:-on}
skip_hash_cmp=${9:-all}   # off | all (legacy: on=all)
jit_cache=${10:-off}
spec_jit=${11:-off}       # off | recompile | interpret
compile_mode=${12:-llvm}   # llvm | fastisel | tpde
tune_config=${13:-}       # path to per-subquery tune JSON
disable_runtime_opts=${14:-}  # comma-separated: range-pred,bloom-filter,range-guard,block-skip,membership,early-term

if [[ -z "$bench" || -z "$engine" || -z "$split" ]]; then
    echo "Usage: $0 <job|dsb_10|dsb_100> <engine> <split> <jit_level> <jit_simd> [flags...]"
    echo "  engine: duckdb | postgres | umbra | mariadb | opengauss | lingodb | lingo-db-runtime"
    echo "  split:  none | node-based | topdown | relationship-center | entity-center | min-subquery"
    exit 1
fi

########################################
# Parse dsb_<SF> bench argument
########################################
if [[ "$bench" == dsb_* ]]; then
    DSB_SF="${bench#dsb_}"
    bench="dsb"
fi
export DSB_SF

source "${SCRIPT_DIR}/env.sh"

########################################
# Benchmark-specific paths
########################################
if [[ "$bench" == "job" ]]; then
    result_dir="job_result"
    pg_analyze_db="${PG_CONN_JOB}"
    analyze_mariadb_cmd="mariadb -u imdb -D imdb < \"${JOB_PATH}/analyze_mariadb_table.sql\""
    opengauss_db="imdb"
    opengauss_user="imdb"
    opengauss_pw="imdb_132"
    umbra_csv_mount="$JOB_PATH/csv"
    umbra_db_name="imdb.db"
    umbra_schema="$JOB_PATH/schema.sql"
    umbra_import="$JOB_PATH/import_umbra_csv.sql"
elif [[ "$bench" == "dsb" ]]; then
    if [[ "$DSB_SF" == "10" ]]; then
        result_dir="dsb_result"
    else
        result_dir="dsb_result_sf${DSB_SF}"
    fi
    pg_analyze_db="${PG_CONN_DSB}"
    analyze_mariadb_cmd="mariadb -u dsb_${DSB_SF} -D dsb_${DSB_SF} < \"${DSB_PATH}/analyze_mariadb_dsb_table.sql\""
    opengauss_db="dsb_${DSB_SF}"
    opengauss_user="dsb_${DSB_SF}"
    opengauss_pw="dsb_${DSB_SF}"
    umbra_csv_mount="$DSB_PATH/code/tools/out_${DSB_SF}/csv"
    umbra_db_name="dsb_${DSB_SF}.db"
    umbra_schema="$DSB_SCHEMA"
    umbra_import="$DSB_IMPORT_CSV"
else
    echo "Unknown benchmark: $bench (expected job, dsb_10, or dsb_100)"
    exit 1
fi

mkdir -p "${result_dir}"/
rm -rf compile.log

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
    echo "Starting Umbra docker (pre-loaded volume)..."

    docker run -d \
        --name "$container_name" \
        --network=host \
        -v umbra-db:/var/db \
        -v /tmp:/tmp \
        --ulimit nofile=1048576:1048576 \
        --ulimit memlock=8388608:8388608 \
        umbradb/umbra:latest \
        umbra-server --address 0.0.0.0 --port 15432 /var/db/${umbra_db_name} >/dev/null

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
    elif [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
        pg_stop
    elif [[ "$engine" == "opengauss" ]]; then
        opengauss_stop
    fi
}
trap cleanup EXIT

########################################
# Start Engine
########################################
if [[ "$engine" == "umbra" ]]; then
    start_umbra
elif [[ "$engine" == "mariadb" ]]; then
    mariadb_start
    pg_start
elif [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
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
    eval "$analyze_mariadb_cmd"
elif [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
    ${PG_BIN}/psql -d "${pg_analyze_db}" -c "ANALYZE;"
elif [[ "$engine" == "opengauss" ]]; then
    sudo -i -u opengauss gsql -d "${opengauss_db}" -U "${opengauss_user}" \
        --host=localhost -p 7654 -W "${opengauss_pw}" -c "ANALYZE;"
fi
echo "ANALYZE done"

cd "${SCRIPT_DIR}" && bash ./hyperfine_aqp.sh "${bench}" "${engine}" "${split}" \
    "${jit_level}" "${jit_simd}" \
    "${payload_prune}" "${prefetch}" "${batch_probe}" "${skip_hash_cmp}" \
    "${jit_cache}" "${spec_jit}" "${compile_mode}" "${tune_config}" \
    "${disable_runtime_opts}"
