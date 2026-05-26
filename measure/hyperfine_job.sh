#!/bin/bash

engine=$1
split=$2
jit_level=$3
jit_opt=$4
jit_simd=$5
fusion_build=${6:-on}     # on / off
fusion_probe=${7:-on}     # on / off
inline_hash=${8:-on}      # on / off
payload_prune=${9:-on}    # on / off
prefetch=${10:-on}        # on / off / <distance>
batch_probe=${11:-on}     # on / off
cache=${12:-off}          # off / on / <path>

# Build CLI flags from positional args
jit_extra_flags=""
[[ "$fusion_probe"  == "off" ]] && jit_extra_flags+=" --no-jit-fusion-probe"
[[ "$inline_hash"   == "off" ]] && jit_extra_flags+=" --no-jit-inline-hash"
[[ "$payload_prune" == "off" ]] && jit_extra_flags+=" --no-jit-payload-prune"
if [[ "$prefetch" == "off" ]]; then
    jit_extra_flags+=" --no-jit-prefetch"
elif [[ "$prefetch" != "on" ]]; then
    jit_extra_flags+=" --jit-prefetch=${prefetch}"
fi
[[ "$batch_probe" == "off" ]] && jit_extra_flags+=" --no-jit-batch-probe"
if [[ "$cache" == "off" ]]; then
    jit_extra_flags+=" --no-jit-cache"
elif [[ "$cache" == "on" ]]; then
    jit_extra_flags+=" --jit-cache"
else
    jit_extra_flags+=" --jit-cache=${cache}"
fi

# Build a short suffix for the log filename
flag_suffix=""
[[ "$fusion_build"  == "off" ]] && flag_suffix+="_nofusbuild"
[[ "$fusion_probe"  == "off" ]] && flag_suffix+="_nofusprobe"
[[ "$inline_hash"   == "off" ]] && flag_suffix+="_noinlhash"
[[ "$payload_prune" == "off" ]] && flag_suffix+="_nopayprune"
[[ "$prefetch"      == "off" ]] && flag_suffix+="_noprefetch"
[[ "$prefetch" != "on" && "$prefetch" != "off" ]] && flag_suffix+="_pf${prefetch}"
[[ "$batch_probe"   == "off" ]] && flag_suffix+="_nobatchprobe"
[[ "$cache"         != "off" ]] && flag_suffix+="_cache"

log_name=aqp_middleware_${engine}_${split}_${jit_level}_${jit_opt}_${jit_simd}${flag_suffix}_job.csv
if [[ "$engine" == "mariadb" ]]; then
    dir="$JOB_PATH/mariadb_queries"
else 
    dir="$JOB_PATH/queries"
fi

rm -rf temp.csv

########################################
# DB connection
########################################
if [[ "$engine" == "postgres" ]]; then
    db_conn="host=localhost port=5432 dbname=imdb user=pei"

elif [[ "$engine" == "duckdb" ]]; then
    db_conn="/home/pei/Project/duckdb/measure/imdb.db"

elif [[ "$engine" == "umbra" ]]; then
    db_conn="host=localhost port=15432 user=postgres password=postgres"

elif [[ "$engine" == "mariadb" ]]; then
    db_conn="host=localhost dbname=imdb user=imdb"

elif [[ "$engine" == "opengauss" ]]; then
    db_conn="host=localhost port=7654 dbname=imdb user=imdb password=imdb_132"

else
    echo "Unknown engine: $engine"
    exit 1
fi

# For node-based split on non-DuckDB backends, pass the DuckDB helper DB
# for planning.  For DuckDB itself the flag is unused.
helper_db_arg=""
if [[ "$split" == "node-based" && "$engine" != "duckdb" ]]; then
    helper_db_path="/home/pei/Project/duckdb/measure/imdb.db"
    helper_db_arg="--helper-db-path=${helper_db_path}"
elif [[ "$engine" == "mariadb" ]]; then
    helper_db_path="host=localhost port=5432 dbname=imdb user=pei"
    helper_db_arg="--helper-db-path=${helper_db_path} --estimator=postgres"
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

for sql in "${dir}"/*.sql; do
    echo "Running benchmark for ${sql}..."

    hyperfine --warmup ${warmup} --runs ${iteration} --export-csv temp.csv \
    "${cmd_prefix}../build_release/aqp_middleware --engine=${engine} \
    --db=\"${db_conn}\" \
    \"${helper_db_arg}\" \
    --schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
    --fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
    --split=\"${split}\" --no-analyze --jit-level=${jit_level} --jit-opt=${jit_opt} --jit-simd=${jit_simd} ${jit_extra_flags} ${sql}"
    cat temp.csv >> "${log_name}"
done

mkdir -p job_result
mv "${log_name}" job_result/
rm -rf temp.csv

