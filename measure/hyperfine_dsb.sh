#!/bin/bash

engine=$1
split=$2
jit_level=$3
jit_simd=$4
payload_prune=${5:-on}    # on / off
prefetch=${6:-on}         # on / off / <distance>
batch_probe=${7:-on}      # on / off
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

log_name=aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${flag_suffix}_dsb.csv
if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    dir="$DSB_PATH/code/tools/1_instance_out_lingo_db/1/"
else
    dir="$DSB_PATH/code/tools/1_instance_out_aqp/1/"
fi

rm -rf temp.csv

########################################
# DB connection
########################################
if [[ "$engine" == "postgres" ]]; then
    db_conn="host=localhost port=5432 dbname=dsb_10 user=postgres"

elif [[ "$engine" == "duckdb" ]]; then
    db_conn="/home/pei/Project/duckdb/measure/dsb_10.db"

elif [[ "$engine" == "umbra" ]]; then
    db_conn="host=localhost port=15432 user=postgres password=postgres"

elif [[ "$engine" == "mariadb" ]]; then
    db_conn="host=localhost dbname=dsb_10 user=dsb_10"

elif [[ "$engine" == "opengauss" ]]; then
    db_conn="host=localhost port=7654 dbname=dsb_10 user=dsb_10 password=dsb_10"

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
    helper_db_path="/home/pei/Project/duckdb/measure/dsb_10.db"
    helper_db_arg="--helper-db-path=${helper_db_path}"
elif [[ "$engine" == "mariadb" ]]; then
    helper_db_path="host=localhost port=5432 dbname=dsb_10 user=postgres"
    helper_db_arg="--helper-db-path=${helper_db_path} --estimator=postgres"
fi

rm -f "${log_name}"
rm -f "dsb_result/${log_name}"

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
    lingodb_flags="--csv-dir=$DSB_PATH/code/tools/out_10/lingo_db_csv"
fi

for sql in $(find "$dir" -type f -name "*.sql" | sort); do
    echo "Running benchmark for ${sql}..."

    hyperfine --warmup ${warmup} --runs ${iteration} --export-csv temp.csv \
    "${cmd_prefix}../build_release/aqp_middleware --engine=${engine} \
    ${db_arg} \
    \"${helper_db_arg}\" \
    --schema=$DSB_PATH/scripts/create_tables.sql \
    --fkeys=$DSB_PATH/code/tools/tpcds_ri.sql \
    --split=\"${split}\" ${lingodb_flags} --no-analyze --jit-level=${jit_level} --jit-simd=${jit_simd} ${jit_extra_flags} ${sql}"
    cat temp.csv >> "${log_name}"
done

mkdir -p dsb_result
mv "${log_name}" dsb_result/
rm -rf temp.csv
