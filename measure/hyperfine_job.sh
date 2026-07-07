#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

engine=$1
split=$2
jit_level=$3
jit_simd=$4
payload_prune=${5:-on}    # on / off
prefetch=${6:-on}         # on / off / <distance>
batch_probe=${7:-on}      # on / off
skip_hash_cmp=${8:-all}   # off | single | all (legacy: on=all)

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
[[ "$skip_hash_cmp"  == "single" ]] && flag_suffix+="_skiphash1"
[[ "$skip_hash_cmp"  == "off" ]] && flag_suffix+="_noskiphashcmp"

log_name=aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${flag_suffix}_job.csv
if [[ "$engine" == "mariadb" ]]; then
    dir="$JOB_PATH/mariadb_queries"
else 
    dir="$JOB_PATH/queries"
fi

rm -rf temp.csv

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
    helper_db_arg="--helper-db-path=${DUCKDB_DB}"
elif [[ "$engine" == "mariadb" ]]; then
    helper_db_arg="--helper-db-path=${PG_CONN} --estimator=postgres"
fi

rm -f "${log_name}"
rm -f "job_result/${log_name}"

cmd_prefix=""
if [[ "$engine" == "opengauss" ]]; then
    cmd_prefix="LD_LIBRARY_PATH=$HOME/gauss_compat_libs "
fi

if [[ "$engine" == "mariadb" ]]; then
    warmup=1
    iteration=3
else
    warmup=5
    iteration=10
fi

# LingoDB: in-memory with CSV loading instead of --db
db_arg="--db=\"${db_conn}\""
lingodb_flags=""
if [[ "$engine" == "lingodb" ]]; then
    db_arg="--in-memory"
    lingodb_flags="--csv-dir=$JOB_PATH/lingo_db_csv"
fi

for sql in "${dir}"/*.sql; do
    echo "Running benchmark for ${sql}..."

    hyperfine --warmup ${warmup} --runs ${iteration} --export-csv temp.csv \
    "${cmd_prefix}${PROJECT}/build_release/aqp_middleware --engine=${engine} \
    ${db_arg} \
    \"${helper_db_arg}\" \
    --schema=$JOB_PATH/schema.sql \
    --fkeys=$JOB_PATH/fkeys.sql \
    --split=\"${split}\" ${lingodb_flags} --no-analyze --jit-level=${jit_level} --jit-simd=${jit_simd} ${jit_extra_flags} ${sql}"
    cat temp.csv >> "${log_name}"
done

mkdir -p job_result
mv "${log_name}" job_result/
rm -rf temp.csv

