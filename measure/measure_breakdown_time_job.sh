#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

engine=$1
split=$2
jit_level=${3:-none}
jit_simd=${4:-off}
payload_prune=${5:-on}
prefetch=${6:-on}
batch_probe=${7:-on}
skip_hash_cmp=${8:-all}   # off | all
jit_cache=${9:-off}       # off | single-run-strict | single-run-template | full
spec_jit=${10:-off}       # off | recompile | interpret
compile_mode=${11:-llvm}   # llvm | fastisel | tpde 
tune_config=${12:-}       # path to per-subquery tune JSON (from tune_per_subquery.py)

# For lingodb/lingo-db-runtime, the 3rd arg selects the execution mode
# (llvm | tpde) instead of the DuckDB jit level.
lingodb_mode="llvm"
if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    lingodb_mode=${3:-llvm}
    jit_level=none
    jit_simd=off
fi

# Build CLI flags from positional args
jit_extra_flags=""
if [[ "$jit_cache" == "on" ]]; then
    jit_extra_flags+=" --jit-cache"
elif [[ "$jit_cache" != "off" ]]; then
    jit_extra_flags+=" --jit-cache=${jit_cache}"
fi
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

# Query-jit scans base tables through the storage plan; the binary cache file
# is built on first use if missing (--storage-plan is auto-enabled by the
# binary for --jit-level=query, kept explicit here for clarity).
if [[ "$jit_level" == "query" ]]; then
    jit_extra_flags+=" --storage-plan --storage-cache=/tmp/imdb_storage_plan.cache"
fi

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
[[ "$compile_mode" != "off" && "$compile_mode" != "llvm" ]] && flag_suffix+="_fc${compile_mode}"
[[ -n "$tune_config" ]]         && flag_suffix+="_tuned"

log_name=time_log.csv
dir="$JOB_PATH/queries"
container_name="umbra_benchmark"

iteration=15 # 5 warm up, 10 runs

########################################
# DB connection
########################################
if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
    db_conn="${PG_CONN}"

elif [[ "$engine" == "duckdb" ]]; then
    db_conn="${DUCKDB_DB}"

elif [[ "$engine" == "umbra" ]]; then
    db_conn="${UMBRA_CONN}"

elif [[ "$engine" == "mariadb" ]]; then
    db_conn="${MARIADB_CONN}"

elif [[ "$engine" == "opengauss" ]]; then
    db_conn="${OPENGAUSS_CONN}"

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
    helper_db_arg="--helper-db-path=${DUCKDB_DB}"
elif [[ "$split" == "node-based" && "$engine" != "duckdb" ]]; then
    helper_db_arg="--helper-db-path=${DUCKDB_DB}"
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
rm -f lingodb_compile_time.csv

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
    mariadb -u imdb -D imdb < "${IMDB_BENCH}/analyze_mariadb_table.sql"
elif [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
    ${PG_BIN}/psql -d "${PG_CONN}" -c "ANALYZE;"
elif [[ "$engine" == "opengauss" ]]; then
    sudo -i -u opengauss gsql -d imdb -U imdb --host=localhost -p 7654 -W imdb_132 -c "ANALYZE;"
fi
echo "ANALYZE done"

########################################
# Run benchmark
########################################
cmd_prefix=""
if [[ "$engine" == "opengauss" ]]; then
    cmd_prefix="env LD_LIBRARY_PATH=$HOME/gauss_compat_libs"
fi

# LingoDB / lingo-db-runtime: in-memory with CSV loading instead of --db
db_arg="--db=${db_conn}"
lingodb_flags=""
if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    db_arg="--in-memory"
    lingodb_flags="--csv-dir=$JOB_PATH/lingo_db_csv --lingodb-mode=${lingodb_mode}"
fi

# Clear disk JIT cache for clean cold-start on iter 0
if [[ "$jit_cache" == "full" ]]; then
    rm -rf /dev/shm/aqp_jit_cache/
fi

$cmd_prefix ../build_release/aqp_middleware \
  --engine="${engine}" \
  ${db_arg} \
  "${helper_db_arg}" \
  --schema=$JOB_PATH/schema.sql \
  --fkeys=$JOB_PATH/fkeys.sql \
  --split="${split}" \
  ${lingodb_flags} \
  --timing \
  --no-analyze \
  --repeat=${iteration} \
  --jit-level=${jit_level} --jit-simd=${jit_simd} \
  ${jit_extra_flags} \
  --benchmark \
  "${dir}"

if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    mv "${log_name}" job_result/${engine}_${lingodb_mode}_${split}_breakdown_"${log_name}"
    mv lingodb_compile_time.csv job_result/${engine}_${lingodb_mode}_${split}_compile_time.csv
else
    mv "${log_name}" job_result/${engine}_${split}_${jit_level}_${jit_simd}${flag_suffix}_breakdown_"${log_name}"
fi
