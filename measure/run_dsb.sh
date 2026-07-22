#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# DSB scale factor: set DSB_SF=100 to run against dsb_100.db.
# Defaults to 10, keeping all existing paths/filenames unchanged.
DSB_SF=${DSB_SF:-10}
if [[ "$DSB_SF" == "10" ]]; then
    result_dir="dsb_result"
    storage_cache="/tmp/dsb_storage_plan.cache"
else
    result_dir="dsb_result_sf${DSB_SF}"
    storage_cache="/tmp/dsb_sf${DSB_SF}_storage_plan.cache"
fi

engine=$1
split=$2
jit_level=$3
jit_simd=$4
payload_prune=${5:-on}
prefetch=${6:-on}
batch_probe=${7:-on}
skip_hash_cmp=${8:-all}   # off | all (legacy: on=all)
jit_cache=${9:-off}
spec_jit=${10:-off}       # off | recompile | interpret (--spec-jit mode)
compile_mode=${11:-llvm}   # llvm | fastisel | tpde (--compile-mode backend)
tune_config=${12:-}       # path to per-subquery tune JSON (from tune_per_subquery.py)

# For lingodb/lingo-db-runtime, the 3rd arg selects the execution mode
# (llvm | tpde) instead of the DuckDB jit level.
lingodb_mode="llvm"
if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    lingodb_mode=${3:-llvm}
    jit_level=none
    jit_simd=none
fi

# Build CLI flags from positional args
jit_extra_flags=""
if [[ "$jit_cache" == "on" ]]; then
    jit_extra_flags+=" --jit-cache"
elif [[ "$jit_cache" != "off" ]]; then
    jit_extra_flags+=" --jit-cache=${jit_cache}"
fi
[[ "$jit_cache" == "full" ]] && jit_extra_flags+=" --repeat=2"
[[ "$payload_prune"  == "off" ]] && jit_extra_flags+=" --no-jit-payload-prune"
if [[ "$prefetch" == "off" ]]; then
    jit_extra_flags+=" --no-jit-prefetch"
elif [[ "$prefetch" != "on" ]]; then
    jit_extra_flags+=" --jit-prefetch=${prefetch}"
fi
[[ "$batch_probe"    == "off" ]] && jit_extra_flags+=" --no-jit-batch-probe"
[[ "$skip_hash_cmp"  == "on" ]] && skip_hash_cmp="all"  # legacy compat
[[ "$skip_hash_cmp"  != "off" ]] && jit_extra_flags+=" --jit-skip-hash-cmp=${skip_hash_cmp}"
[[ "$spec_jit"       != "off" ]] && jit_extra_flags+=" --spec-jit=${spec_jit}"
[[ "$compile_mode" != "off" && "$compile_mode" != "llvm" ]] && jit_extra_flags+=" --compile-mode=${compile_mode}"
[[ -n "$tune_config" ]]         && jit_extra_flags+=" --tune-config=${tune_config}"

# Build a short suffix for the log filename
flag_suffix=""
[[ "$payload_prune"  == "off" ]] && flag_suffix+="_nopayprune"
[[ "$prefetch"       == "off" ]] && flag_suffix+="_noprefetch"
[[ "$prefetch" != "on" && "$prefetch" != "off" ]] && flag_suffix+="_pf${prefetch}"
[[ "$batch_probe"    == "off" ]] && flag_suffix+="_nobatchprobe"
[[ "$skip_hash_cmp"  == "off" ]] && flag_suffix+="_noskiphashcmp"
if [[ "$jit_cache" == "on" ]]; then
    flag_suffix+="_jitcache"
elif [[ "$jit_cache" != "off" ]]; then
    flag_suffix+="_jitcache_${jit_cache//-/_}"
fi
[[ "$spec_jit"       != "off" ]] && flag_suffix+="_spec${spec_jit}"
[[ "$compile_mode" != "off" ]] && flag_suffix+="_${compile_mode}"
[[ -n "$tune_config" ]]         && flag_suffix+="_tuned"

# Storage plan flags. Only query-jit consumes the storage plan.
storage_flags=""
if [[ "$jit_level" == "query" ]]; then
    storage_flags="--storage-plan --storage-cache=${storage_cache}"
fi

if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    log_name=aqp_middleware_${engine}_${lingodb_mode}_${split}_dsb.txt
else
    log_name=aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${flag_suffix}_dsb.txt
fi
if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    dir="$DSB_PATH/code/tools/1_instance_out_lingo_db/1/"
else
    dir="$DSB_PATH/code/tools/1_instance_out_aqp/1/"
fi
container_name="umbra_benchmark"

########################################
# DB connection
########################################
if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
    db_conn="${PG_CONN}"

elif [[ "$engine" == "duckdb" ]]; then
    db_conn="${DSB_DUCKDB_DB:-/home/pei/Project/duckdb/measure/dsb_${DSB_SF}.db}"

elif [[ "$engine" == "umbra" ]]; then
    db_conn="${UMBRA_CONN}"

elif [[ "$engine" == "mariadb" ]]; then
    db_conn="${MARIADB_CONN:-host=localhost dbname=dsb_${DSB_SF} user=dsb_${DSB_SF}}"

elif [[ "$engine" == "opengauss" ]]; then
    db_conn="${OPENGAUSS_CONN:-host=localhost port=7654 dbname=dsb_${DSB_SF} user=dsb_${DSB_SF} password=dsb_${DSB_SF}}"

elif [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    db_conn=""

else
    echo "Unknown engine: $engine"
    exit 1
fi

# For node-based split on non-DuckDB backends, pass the DuckDB helper DB
# for planning.  For DuckDB itself the flag is unused.
helper_db_arg=""
if [[ "$engine" == "lingo-db-runtime" ]]; then
    helper_db_path="${DSB_DUCKDB_DB:-/home/pei/Project/duckdb/measure/dsb_${DSB_SF}.db}"
    helper_db_arg="--helper-db-path=${helper_db_path}"
elif [[ "$split" == "node-based" && "$engine" != "duckdb" ]]; then
    helper_db_path="${DSB_DUCKDB_DB:-/home/pei/Project/duckdb/measure/dsb_${DSB_SF}.db}"
    helper_db_arg="--helper-db-path=${helper_db_path}"
elif [[ "$engine" == "mariadb" ]]; then
    helper_db_path="${PG_CONN:-host=localhost port=5432 dbname=dsb_${DSB_SF} user=postgres}"
    helper_db_arg="--helper-db-path=${helper_db_path} --estimator=postgres"
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
rm -f "${result_dir}/${log_name}"

mkdir -p "${result_dir}"
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
    mariadb -u dsb_${DSB_SF} -D dsb_${DSB_SF} < "${DSB_PATH}/analyze_mariadb_dsb_table.sql"
elif [[ "$engine" == "postgres" ]]; then
    psql -U postgres -d dsb_${DSB_SF} -c "ANALYZE;"
elif [[ "$engine" == "opengauss" ]]; then
    sudo -i -u opengauss gsql -d dsb_${DSB_SF} -U dsb_${DSB_SF} --host=localhost -p 7654 -W dsb_${DSB_SF} -c "ANALYZE;"
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

# LingoDB / lingo-db-runtime: in-memory with CSV loading instead of --db
db_arg="--db=${db_conn}"
lingodb_flags=""
if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    db_arg="--in-memory"
    lingodb_flags="--csv-dir=$DSB_PATH/code/tools/out_${DSB_SF}/lingo_db_csv --lingodb-mode=${lingodb_mode}"
fi

for sql in $(find "$dir" -type f -name "*.sql" | sort); do
    echo "Running benchmark for $sql..." | tee -a "$log_name"

    $cmd_prefix "${PROJECT}/build_release/aqp_middleware" \
        --engine="${engine}" \
        --db="${db_conn}" \
        "${helper_db_arg}" \
        --schema="${DSB_PATH}/scripts/create_tables.sql" \
        --fkeys="${DSB_PATH}/scripts/tpcds_ri_umbra.sql" \
        --split="${split}" \
        --no-analyze \
        "${sql}" \
        2>&1 | tee -a "$log_name"
done
end=$(date +%s%N)
elapsed_ns=$((end - start))
elapsed_ms=$((elapsed_ns / 1000000))

echo "${engine} runs: ${elapsed_ms} ms"

mv "${log_name}" "${result_dir}"/
