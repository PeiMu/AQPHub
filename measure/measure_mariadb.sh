#!/bin/bash
#
# MariaDB benchmark — Way 2: all queries per iteration.
# Runs all queries once per iteration for N iterations (warmup + measured).
# Output: CSV matching measure_breakdown_time_aqp.sh format:
#   "Running benchmark for <path>..." header per query,
#   one row per iteration with a single wall-time value (ms).
#
# Usage: measure_mariadb.sh <job|dsb_10|dsb_100>
#

# Disable pager so client output doesn't block the script
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
    dir="$JOB_PATH/mariadb_queries"
    result_dir="job_result"
    mariadb_user="imdb"
    mariadb_db="imdb"
    analyze_cmd="mariadb -u imdb -D imdb < \"${JOB_PATH}/analyze_mariadb_table.sql\""
elif [[ "$bench" == "dsb" ]]; then
    dir="$DSB_PATH/code/tools/1_instance_out_aqp/1/"
    if [[ "$DSB_SF" == "10" ]]; then
        result_dir="dsb_result"
    else
        result_dir="dsb_result_sf${DSB_SF}"
    fi
    mariadb_user="dsb_${DSB_SF}"
    mariadb_db="dsb_${DSB_SF}"
    analyze_cmd="mariadb -u ${mariadb_user} -D ${mariadb_db} < \"${DSB_PATH}/analyze_mariadb_dsb_table.sql\""
else
    echo "Usage: $0 <job|dsb_10|dsb_100>"
    exit 1
fi

mkdir -p "${result_dir}"

log_name=mariadb_official.csv
rm -f "${log_name}"

########################################
# Start / Stop MariaDB
########################################
mariadb_start() {
    sudo systemctl start mariadb
}

mariadb_stop() {
    sudo systemctl stop mariadb
}

cleanup() {
    mariadb_stop
}
trap cleanup EXIT

mariadb_start

########################################
# ANALYZE
########################################
echo "ANALYZING..."
eval "$analyze_cmd"
echo "ANALYZE done"

warmup=1
measured=3

# Collect sorted query list
mapfile -t sql_files < <(find "$dir" -type f -name "*.sql" | sort)

# Warmup iterations
for w in $(seq 1 ${warmup}); do
    echo "Warmup iteration ${w}/${warmup}..."
    for sql in "${sql_files[@]}"; do
        mariadb -u "${mariadb_user}" -D "${mariadb_db}" < "$sql"
    done
done

# Measured iterations
declare -A query_times

for i in $(seq 1 ${measured}); do
    echo "Measured iteration ${i}/${measured}..."
    for sql in "${sql_files[@]}"; do
        qname=$(basename "$sql")
        start_ns=$(date +%s%N)
        mariadb -u "${mariadb_user}" -D "${mariadb_db}" < "$sql"
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
