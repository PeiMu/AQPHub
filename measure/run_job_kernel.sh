#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

engine=$1
split=$2
kernel_path=$3        # none | pipeline | query
jit_simd=${4:-auto}
payload_prune=${5:-on}
prefetch=${6:-on}
batch_probe=${7:-on}
skip_hash_cmp=${8:-all}   # off | all (legacy: on=all)

# Build CLI flags from positional args
jit_extra_flags=""
[[ "$payload_prune"  == "off" ]] && jit_extra_flags+=" --no-jit-payload-prune"
if [[ "$prefetch" == "off" ]]; then
    jit_extra_flags+=" --no-jit-prefetch"
elif [[ "$prefetch" != "on" ]]; then
    jit_extra_flags+=" --jit-prefetch=${prefetch}"
fi
[[ "$batch_probe"    == "off" ]] && jit_extra_flags+=" --no-jit-batch-probe"
[[ "$skip_hash_cmp"  == "on" ]] && skip_hash_cmp="all"  # legacy compat
[[ "$skip_hash_cmp"  != "off" ]] && jit_extra_flags+=" --jit-skip-hash-cmp=${skip_hash_cmp}"

# Build a short suffix for the log filename
flag_suffix=""
[[ "$payload_prune"  == "off" ]] && flag_suffix+="_nopayprune"
[[ "$prefetch"       == "off" ]] && flag_suffix+="_noprefetch"
[[ "$prefetch" != "on" && "$prefetch" != "off" ]] && flag_suffix+="_pf${prefetch}"
[[ "$batch_probe"    == "off" ]] && flag_suffix+="_nobatchprobe"
[[ "$skip_hash_cmp"  == "off" ]] && flag_suffix+="_noskiphashcmp"

# Storage plan flags (always enabled for kernel path)
storage_flags="--storage-plan --storage-cache=${STORAGE_CACHE_DUCKDB_JOB}"

# Kernel path implies operator-level JIT
jit_level_flag="operator"
kernel_path_flag=""
if [[ "$kernel_path" != "none" ]]; then
    kernel_path_flag="--kernel-path=${kernel_path}"
fi

log_name=aqp_middleware_${engine}_${split}_kernel-${kernel_path}_${jit_simd}${flag_suffix}_job.txt
dir="$JOB_PATH/queries"
container_name="umbra_benchmark"

########################################
# DB connection
########################################
if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
    db_conn="${PG_CONN_JOB}"

elif [[ "$engine" == "duckdb" ]]; then
    db_conn="${DUCKDB_DB_JOB}"

elif [[ "$engine" == "umbra" ]]; then
    db_conn="${UMBRA_CONN}"

elif [[ "$engine" == "mariadb" ]]; then
    db_conn="${MARIADB_CONN}"

elif [[ "$engine" == "opengauss" ]]; then
    db_conn="${OPENGAUSS_CONN}"

elif [[ "$engine" == "lingodb" ]]; then
    db_conn=""

else
    echo "Unknown engine: $engine"
    exit 1
fi

# For node-based split on non-DuckDB backends, pass the DuckDB helper DB
# for planning.  For DuckDB itself the flag is unused.
helper_db_arg=""
if [[ "$split" == "node-based" && "$engine" != "duckdb" ]]; then
    helper_db_arg="--helper-db-path=${DUCKDB_DB_JOB}"
elif [[ "$engine" == "mariadb" ]]; then
    helper_db_arg="--helper-db-path=${PG_CONN_JOB} --estimator=postgres"
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
        -v "$JOB_PATH/csv":/benchmark/csv:ro \
        --ulimit nofile=1048576:1048576 \
        --ulimit memlock=8388608:8388608 \
        umbradb/umbra:latest \
        umbra-server --address 0.0.0.0 --port 15432 /var/db/imdb.db >/dev/null

    wait_for_umbra
    load_umbra_imdb_data
}

load_umbra_imdb_data() {
    echo "Loading schema and CSV data into Umbra..."
    PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres \
        -f "$JOB_PATH/schema.sql"
    PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres \
        -f "$JOB_PATH/import_umbra_csv.sql"
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
    elif [[ "$engine" == "duckdb" || "$engine" == "lingodb" ]]; then
        :
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
rm -f "job_result/${log_name}"

mkdir -p job_result
shopt -s nullglob

########################################
# Start engine if needed
########################################
if [[ "$engine" == "umbra" ]]; then
    start_umbra
elif [[ "$engine" == "mariadb" ]]; then
    mariadb_start
elif [[ "$engine" == "opengauss" ]]; then
    opengauss_start
elif [[ "$engine" == "duckdb" || "$engine" == "lingodb" ]]; then
    :
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
    mariadb -u imdb -D imdb < "${JOB_PATH}/analyze_mariadb_table.sql"
elif [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
    ${PG_BIN}/psql -d "${PG_CONN_JOB}" -c "ANALYZE;"
elif [[ "$engine" == "opengauss" ]]; then
    sudo -i -u opengauss gsql -d imdb -U imdb --host=localhost -p 7654 -W imdb_132 -c "ANALYZE;"
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

# LingoDB: in-memory with CSV loading instead of --db
db_arg="--db=${db_conn}"
lingodb_flags=""
if [[ "$engine" == "lingodb" ]]; then
    db_arg="--in-memory"
    lingodb_flags="--csv-dir=$JOB_PATH/lingo_db_csv"
fi

$cmd_prefix "${PROJECT}/build_release/aqp_middleware" \
    --engine="${engine}" \
    ${db_arg} \
    "${helper_db_arg}" \
    --schema=$JOB_PATH/schema.sql \
    --fkeys=$JOB_PATH/fkeys.sql \
    --split="${split}" \
    ${lingodb_flags} \
    --no-analyze \
    --jit-level=${jit_level_flag} --jit-simd=${jit_simd} \
    ${kernel_path_flag} \
    ${jit_extra_flags} \
    ${storage_flags} \
    --benchmark \
    "${dir}" \
    2>&1 | tee -a "$log_name"
end=$(date +%s%N)
elapsed_ns=$((end - start))
elapsed_ms=$((elapsed_ns / 1000000))

echo "${engine} runs: ${elapsed_ms} ms"

mv "${log_name}" job_result/
