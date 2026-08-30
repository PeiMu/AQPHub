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
    if [[ "$engine" == "mariadb" ]]; then
        dir="$JOB_PATH/mariadb_queries"
    else
        dir="$JOB_PATH/queries"
    fi
    schema="$JOB_PATH/schema.sql"
    fkeys="$JOB_PATH/fkeys.sql"
    result_dir="job_result"
    log_suffix="_job"
    duckdb_db="${DUCKDB_DB_JOB}"
    storage_cache="${STORAGE_CACHE_DUCKDB_JOB}"
    storage_cache_pg="${STORAGE_CACHE_PG_JOB}"
    csv_dir="$JOB_PATH/lingo_db_csv"
elif [[ "$bench" == "dsb" ]]; then
    if [[ "$engine" == "lingodb" ]]; then
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
    log_suffix="_dsb"
    duckdb_db="${DUCKDB_DB_DSB}"
    storage_cache="${STORAGE_CACHE_DUCKDB_DSB}"
    storage_cache_pg="${STORAGE_CACHE_PG_DSB}"
    csv_dir="$DSB_PATH/code/tools/out_${DSB_SF}/lingo_db_csv"
    if [[ ! -d "$csv_dir" ]]; then
        csv_dir="$DSB_PATH/code/tools/out_${DSB_SF}/csv"
    fi
else
    echo "Usage: $0 <job|dsb_10|dsb_100> <engine> <split> <jit_level> <jit_simd> [flags...]"
    exit 1
fi

########################################
# LingoDB mode / JIT level
########################################
lingodb_mode="llvm"
if [[ "$engine" == "lingodb" ]]; then
    if [[ "$jit_level" == "query" || "$jit_level" == "expr" ]]; then
        : # middleware JIT — keep jit_level as-is, lingodb_mode=llvm
    else
        lingodb_mode=${4:-llvm}
        jit_level=none
        jit_simd=none
    fi
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
if [[ -n "$disable_runtime_opts" ]]; then
    IFS=',' read -ra _dro <<< "$disable_runtime_opts"
    for _opt in "${_dro[@]}"; do
        case "$_opt" in
            range-pred)    jit_extra_flags+=" --no-range-predicate-injection" ;;
            bloom-filter)  jit_extra_flags+=" --no-bloom-filter-injection" ;;
            range-guard)   jit_extra_flags+=" --no-range-guard" ;;
            block-skip)    jit_extra_flags+=" --no-block-skip" ;;
            membership)    jit_extra_flags+=" --no-membership-preprobe" ;;
            early-term)    jit_extra_flags+=" --no-early-termination" ;;
            *) echo "Unknown runtime opt: $_opt"; exit 1 ;;
        esac
    done
fi

########################################
# Storage plan flags
########################################
storage_flags=""
if [[ "$jit_level" == "query" ]]; then
    if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
        storage_flags="--storage-plan --storage-cache=${storage_cache_pg}"
    elif [[ "$engine" == "lingodb" ]]; then
        storage_flags="--storage-plan --storage-cache=${storage_cache}"
    else
        storage_flags="--storage-plan --storage-cache=${storage_cache}"
    fi
fi

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
[[ "$disable_runtime_opts" == *"range-pred"* ]]   && flag_suffix+="_norangepred"
[[ "$disable_runtime_opts" == *"bloom-filter"* ]]  && flag_suffix+="_nobloomfilt"
[[ "$disable_runtime_opts" == *"range-guard"* ]]   && flag_suffix+="_norangeguard"
[[ "$disable_runtime_opts" == *"block-skip"* ]]    && flag_suffix+="_noblockskip"
[[ "$disable_runtime_opts" == *"membership"* ]]    && flag_suffix+="_nomembership"
[[ "$disable_runtime_opts" == *"early-term"* ]]    && flag_suffix+="_noearlyterm"

########################################
# Log name
########################################
if [[ "$engine" == "lingodb" ]]; then
    if [[ "$jit_level" == "none" ]]; then
        log_name=aqp_middleware_${engine}_${lingodb_mode}_${split}${log_suffix}.csv
    else
        log_name=aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${flag_suffix}${log_suffix}.csv
    fi
else
    log_name=aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${flag_suffix}${log_suffix}.csv
fi

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
elif [[ "$engine" == "lingodb" ]]; then
    db_conn=""
else
    echo "Unknown engine: $engine"
    exit 1
fi

########################################
# Helper DB for non-DuckDB backends
########################################
helper_db_arg=""
if [[ "$engine" == "lingodb" ]]; then
    helper_db_arg="--helper-db-path=${duckdb_db}"
elif [[ ("$split" == "node-based" || "$split" == "topdown" || "$split" == "auto") && "$engine" != "duckdb" ]]; then
    helper_db_arg="--helper-db-path=${duckdb_db}"
elif [[ "$engine" == "mariadb" ]]; then
    if [[ "$bench" == "job" ]]; then
        helper_db_arg="--helper-db-path=${PG_CONN_JOB} --estimator=postgres"
    else
        helper_db_arg="--helper-db-path=${PG_CONN_DSB} --estimator=postgres"
    fi
fi

########################################
# Prepare
########################################
rm -f temp.csv
rm -f "${log_name}"
rm -f "${result_dir}/${log_name}"

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

db_arg="--db=\"${db_conn}\""
lingodb_flags=""
if [[ "$engine" == "lingodb" ]]; then
    db_arg="--in-memory"
    lingodb_flags="--csv-dir=${csv_dir} --lingodb-mode=${lingodb_mode}"
fi

########################################
# Run benchmark per query
########################################
for sql in $(find "$dir" -type f -name "*.sql" | sort); do
    echo "Running benchmark for ${sql}..."

    hyperfine --warmup ${warmup} --runs ${iteration} --export-csv temp.csv \
    "${cmd_prefix}${PROJECT}/build_release/aqp_middleware --engine=${engine} \
    ${db_arg} \
    ${helper_db_arg} \
    --schema=${schema} \
    --fkeys=${fkeys} \
    --split=${split} ${lingodb_flags} --no-analyze \
    --jit-level=${jit_level} --jit-simd=${jit_simd} \
    ${jit_extra_flags} ${storage_flags} ${sql}"
    cat temp.csv >> "${log_name}"
done

mkdir -p "${result_dir}"
mv "${log_name}" "${result_dir}"/
rm -f temp.csv
