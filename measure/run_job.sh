#!/usr/bin/env bash

engine=$1
split=$2
jit_level=$3
jit_opt=$4
jit_simd=$5
fusion_build=${6:-on}
fusion_probe=${7:-on}
inline_hash=${8:-on}
payload_prune=${9:-on}
prefetch=${10:-on}
batch_probe=${11:-on}
cache=${12:-off}

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

log_name=aqp_middleware_${engine}_${split}_${jit_level}_${jit_opt}_${jit_simd}${flag_suffix}_job.txt
dir="$JOB_PATH/queries"
container_name="umbra_benchmark"

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
        umbra-server --address 0.0.0.0 --port 15432 /var/db/imdb.db >/dev/null

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
  pg_ctl start -l $Project_path/logfile -D $Project_path/data_18_3
}
pg_stop() {
  pg_ctl stop -D $Project_path/data_18_3 -m smart -s
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
rm -f "job_result/${log_name}"

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
    mariadb -u imdb -D imdb < /home/pei/Project/benchmarks/imdb_job-postgres/analyze_mariadb_table.sql
elif [[ "$engine" == "postgres" ]]; then
    psql -U pei -d imdb -c "ANALYZE;"
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

for sql in "$dir"/*.sql; do
    echo "Running benchmark for $sql..." | tee -a "$log_name"

    $cmd_prefix ../build_release/aqp_middleware \
        --engine="${engine}" \
        --db="${db_conn}" \
        "${helper_db_arg}" \
        --schema=/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql \
        --fkeys=/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql \
        --split="${split}" \
        --no-analyze --jit-level=${jit_level} --jit-opt=${jit_opt} --jit-simd=${jit_simd} \
        ${jit_extra_flags} \
        "${sql}" \
        2>&1 | tee -a "$log_name"
done
end=$(date +%s%N)
elapsed_ns=$((end - start))
elapsed_ms=$((elapsed_ns / 1000000))

echo "${engine} runs: ${elapsed_ms} ms"

mv "${log_name}" job_result/
