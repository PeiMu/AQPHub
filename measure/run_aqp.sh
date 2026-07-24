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
spec_jit=${11:-off}       # off | recompile | interpret (--spec-jit mode)
compile_mode=${12:-llvm}   # llvm | fastisel | tpde (--compile-mode backend)
tune_config=${13:-}       # path to per-subquery tune JSON (from tune_per_subquery.py)

########################################
# Parse dsb_<SF> bench argument
########################################
if [[ "$bench" == dsb_* ]]; then
    DSB_SF="${bench#dsb_}"
    bench="dsb"
fi

source "${SCRIPT_DIR}/env.sh"

########################################
# Benchmark-specific paths
########################################
if [[ "$bench" == "job" ]]; then
    dir="$JOB_PATH/queries"
    schema="$JOB_PATH/schema.sql"
    fkeys="$JOB_PATH/fkeys.sql"
    result_dir="job_result"
    log_suffix="_job.txt"
    duckdb_db="${DUCKDB_DB_JOB}"
    storage_cache="${STORAGE_CACHE_DUCKDB_JOB}"
    storage_cache_pg="${STORAGE_CACHE_PG_JOB}"
    csv_dir="$JOB_PATH/lingo_db_csv"
    umbra_csv_mount="$JOB_PATH/csv"
    umbra_db_name="imdb.db"
    umbra_schema="$JOB_PATH/schema.sql"
    umbra_import="$JOB_PATH/import_umbra_csv.sql"
    analyze_mariadb_cmd="mariadb -u imdb -D imdb < \"${JOB_PATH}/analyze_mariadb_table.sql\""
    pg_analyze_db="${PG_CONN_JOB}"
    opengauss_db="imdb"
    opengauss_user="imdb"
    opengauss_pw="imdb_132"
elif [[ "$bench" == "dsb" ]]; then
    if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
        dir="$DSB_PATH/code/tools/1_instance_out_lingo_db/1/"
    else
        dir="$DSB_PATH/code/tools/1_instance_out_aqp/1/"
    fi
    schema="${DSB_PATH}/scripts/create_tables.sql"
    fkeys="${DSB_PATH}/scripts/tpcds_ri_umbra.sql"
    if [[ "$DSB_SF" == "10" ]]; then
        result_dir="dsb_result"
    else
        result_dir="dsb_result_sf${DSB_SF}"
    fi
    log_suffix="_dsb.txt"
    duckdb_db="${DUCKDB_DB_DSB}"
    storage_cache="${STORAGE_CACHE_DUCKDB_DSB}"
    storage_cache_pg="${STORAGE_CACHE_PG_DSB}"
    csv_dir="$DSB_PATH/code/tools/out_${DSB_SF}/lingo_db_csv"
    umbra_csv_mount="$DSB_PATH/code/tools/out_${DSB_SF}/csv"
    umbra_db_name="dsb_${DSB_SF}.db"
    umbra_schema="$DSB_SCHEMA"
    umbra_import="$DSB_IMPORT_CSV"
    analyze_mariadb_cmd="mariadb -u dsb_${DSB_SF} -D dsb_${DSB_SF} < \"${DSB_PATH}/analyze_mariadb_dsb_table.sql\""
    pg_analyze_db="${PG_CONN_DSB}"
    opengauss_db="dsb_${DSB_SF}"
    opengauss_user="dsb_${DSB_SF}"
    opengauss_pw="dsb_${DSB_SF}"
else
    echo "Usage: $0 <job|dsb_10|dsb_100> <engine> <split> <jit_level> <jit_simd> [flags...]"
    exit 1
fi

########################################
# LingoDB mode override
########################################
lingodb_mode="llvm"
if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    lingodb_mode=${4:-llvm}
    jit_level=none
    jit_simd=none
fi

########################################
# Build CLI flags from positional args
########################################
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

########################################
# Build a short suffix for the log filename
########################################
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
[[ "$compile_mode" != "off" && "$compile_mode" != "llvm" ]] && flag_suffix+="_${compile_mode}"
[[ -n "$tune_config" ]]         && flag_suffix+="_tuned"

########################################
# Storage plan flags
########################################
storage_flags=""
if [[ "$jit_level" == "query" ]]; then
    if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
        storage_flags="--storage-plan --storage-cache=${storage_cache_pg}"
    else
        storage_flags="--storage-plan --storage-cache=${storage_cache}"
    fi
fi

########################################
# Log name
########################################
if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    log_name=aqp_middleware_${engine}_${lingodb_mode}_${split}${log_suffix}
else
    log_name=aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${flag_suffix}${log_suffix}
fi
container_name="umbra_benchmark"

########################################
# DB connection
########################################
if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
    if [[ "$bench" == "job" ]]; then
        db_conn="${PG_CONN_JOB}"
    else
        db_conn="${PG_CONN_DSB}"
    fi
elif [[ "$engine" == "duckdb" ]]; then
    db_conn="${duckdb_db}"
elif [[ "$engine" == "umbra" ]]; then
    db_conn="${UMBRA_CONN}"
elif [[ "$engine" == "mariadb" ]]; then
    if [[ "$bench" == "job" ]]; then
        db_conn="${MARIADB_CONN}"
    else
        db_conn="${MARIADB_CONN:-host=localhost dbname=dsb_${DSB_SF} user=dsb_${DSB_SF}}"
    fi
elif [[ "$engine" == "opengauss" ]]; then
    if [[ "$bench" == "job" ]]; then
        db_conn="${OPENGAUSS_CONN}"
    else
        db_conn="${OPENGAUSS_CONN:-host=localhost port=7654 dbname=dsb_${DSB_SF} user=dsb_${DSB_SF} password=dsb_${DSB_SF}}"
    fi
elif [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    db_conn=""
else
    echo "Unknown engine: $engine"
    exit 1
fi

########################################
# Helper DB for non-DuckDB backends
########################################
helper_db_arg=""
if [[ "$engine" == "lingo-db-runtime" ]]; then
    helper_db_arg="--helper-db-path=${duckdb_db}"
elif [[ ("$split" == "node-based" || "$split" == "topdown") && "$engine" != "duckdb" ]]; then
    helper_db_arg="--helper-db-path=${duckdb_db}"
elif [[ "$engine" == "mariadb" ]]; then
    if [[ "$bench" == "job" ]]; then
        helper_db_arg="--helper-db-path=${PG_CONN_JOB} --estimator=postgres"
    else
        helper_db_arg="--helper-db-path=${PG_CONN_DSB} --estimator=postgres"
    fi
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
        -v "${umbra_csv_mount}":/benchmark/csv:ro \
        --ulimit nofile=1048576:1048576 \
        --ulimit memlock=8388608:8388608 \
        umbradb/umbra:latest \
        umbra-server --address 0.0.0.0 --port 15432 /var/db/${umbra_db_name} >/dev/null
    wait_for_umbra
    load_umbra_data
}

load_umbra_data() {
    echo "Loading schema and CSV data into Umbra..."
    PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres \
        -f "$umbra_schema"
    PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres \
        -f "$umbra_import"
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
mariadb_start() { sudo systemctl start mariadb; }
mariadb_stop()  { sudo systemctl stop mariadb; }

########################################
# Start / Stop OpenGauss
########################################
opengauss_start() { sudo systemctl start opengauss; }
opengauss_stop()  { sudo systemctl stop opengauss; }

cleanup() {
    if [[ "$engine" == "umbra" ]]; then
        stop_umbra
    elif [[ "$engine" == "mariadb" ]]; then
        mariadb_stop
    elif [[ "$engine" == "opengauss" ]]; then
        opengauss_stop
    elif [[ "$engine" == "duckdb" || "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
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
rm -f "${result_dir}/${log_name}"

mkdir -p "${result_dir}"
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
elif [[ "$engine" == "duckdb" || "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
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
    eval "$analyze_mariadb_cmd"
elif [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
    ${PG_BIN}/psql -d "${pg_analyze_db}" -c "ANALYZE;"
elif [[ "$engine" == "opengauss" ]]; then
    sudo -i -u opengauss gsql -d "${opengauss_db}" -U "${opengauss_user}" \
        --host=localhost -p 7654 -W "${opengauss_pw}" -c "ANALYZE;"
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

db_arg="--db=${db_conn}"
lingodb_flags=""
if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    db_arg="--in-memory"
    lingodb_flags="--csv-dir=${csv_dir} --lingodb-mode=${lingodb_mode}"
fi

$cmd_prefix "${PROJECT}/build_release/aqp_middleware" \
    --engine="${engine}" \
    "${db_arg}" \
    "${helper_db_arg}" \
    --schema="${schema}" \
    --fkeys="${fkeys}" \
    --split="${split}" \
    ${lingodb_flags} \
    --no-analyze --jit-level=${jit_level} --jit-simd=${jit_simd} \
    ${jit_extra_flags} \
    ${storage_flags} \
    --benchmark \
    "${dir}" \
    2>&1 | tee -a "$log_name"
end=$(date +%s%N)
elapsed_ns=$((end - start))
elapsed_ms=$((elapsed_ns / 1000000))

echo "${engine} runs: ${elapsed_ms} ms"

mv "${log_name}" "${result_dir}"/
