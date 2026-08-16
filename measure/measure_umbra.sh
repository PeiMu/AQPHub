#!/bin/bash
#
# Umbra benchmark — Way 2: all queries per iteration.
# Runs all queries once per iteration for N iterations (measured).
# Output: CSV matching measure_breakdown_time_aqp.sh format:
#   "Running benchmark for <path>..." header per query,
#   one row per iteration with a single wall-time value (ms).
#
# Usage: measure_umbra.sh <job|dsb_10|dsb_100>
#

# Disable pager so psql output doesn't block the script
export PAGER=cat

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bench=$1

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
    result_dir="job_result"
    umbra_db_name="imdb.db"
elif [[ "$bench" == "dsb" ]]; then
    dir="$DSB_PATH/code/tools/1_instance_out_aqp/1/"
    if [[ "$DSB_SF" == "10" ]]; then
        result_dir="dsb_result"
    else
        result_dir="dsb_result_sf${DSB_SF}"
    fi
    umbra_db_name="dsb_${DSB_SF}.db"
else
    echo "Usage: $0 <job|dsb_10|dsb_100>"
    exit 1
fi

mkdir -p "${result_dir}"

log_name=umbra_official.csv
rm -f "${log_name}"

########################################
# Start / Stop Umbra
########################################
container_name="umbra_benchmark"
start_umbra() {
    echo "Starting Umbra docker (pre-loaded volume)..."
    docker stop "$container_name" >/dev/null 2>&1 || true
    docker rm "$container_name" >/dev/null 2>&1 || true

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

cleanup() {
    stop_umbra
}
trap cleanup EXIT

start_umbra

########################################
# ANALYZE
########################################
echo "ANALYZING..."
PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres -c "ANALYZE;"
echo "ANALYZE done"

measured=15

# Collect sorted query list
mapfile -t sql_files < <(find "$dir" -type f -name "*.sql" | sort)

# Measured iterations — store times per query across iterations
declare -A query_times

for i in $(seq 1 ${measured}); do
    echo "Measured iteration ${i}/${measured}..."
    for sql in "${sql_files[@]}"; do
        qname=$(basename "$sql")
        start_ns=$(date +%s%N)
        PGPASSWORD=postgres psql -p 15432 -h localhost -U postgres -f "$sql"
        end_ns=$(date +%s%N)
        elapsed_us=$(( (end_ns - start_ns) / 1000 ))
        elapsed_ms="${elapsed_us%???}.${elapsed_us: -3}"
        query_times["${qname},${i}"]="${elapsed_ms}"
    done
done

# Write CSV in measure_breakdown_time_aqp.sh format
for sql in "${sql_files[@]}"; do
    qname=$(basename "$sql")
    echo "Running benchmark for ${sql}..." >> "${log_name}"
    for i in $(seq 1 ${measured}); do
        echo "${query_times["${qname},${i}"]}" >> "${log_name}"
    done
done

mv "${log_name}" "${result_dir}/"
echo "Done. Results in ${result_dir}/${log_name}"
