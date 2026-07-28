#!/bin/bash
#
# OpenGauss benchmark — Way 2: all queries per iteration.
# Runs all queries once per iteration for N iterations (warmup + measured).
# Output: CSV matching measure_breakdown_time_aqp.sh format:
#   "Running benchmark for <path>..." header per query,
#   one row per iteration with a single wall-time value (ms).
#
# Usage: measure_opengauss.sh <job|dsb_10|dsb_100>
#

# Drop OS page cache for reproducible cold-start measurement
sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

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
    dir="$JOB_PATH/queries"
    result_dir="job_result"
    opengauss_db="imdb"
    opengauss_user="imdb"
    opengauss_pw="imdb_132"
    analyze_cmd="sudo -i -u opengauss gsql -d imdb -U imdb --host=localhost -p 7654 -W imdb_132 < \"${JOB_PATH}/analyze_table.sql\""
elif [[ "$bench" == "dsb" ]]; then
    dir="$DSB_PATH/code/tools/1_instance_out_aqp/1/"
    if [[ "$DSB_SF" == "10" ]]; then
        result_dir="dsb_result"
    else
        result_dir="dsb_result_sf${DSB_SF}"
    fi
    opengauss_db="dsb_${DSB_SF}"
    opengauss_user="dsb_${DSB_SF}"
    opengauss_pw="dsb_${DSB_SF}"
    analyze_cmd="sudo -i -u opengauss gsql -d ${opengauss_db} -U ${opengauss_user} --host=localhost -p 7654 -W ${opengauss_pw} -c 'ANALYZE;'"
else
    echo "Usage: $0 <job|dsb_10|dsb_100>"
    exit 1
fi

mkdir -p "${result_dir}"

log_name=opengauss_official.csv
rm -f "${log_name}"

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
    opengauss_stop
}
trap cleanup EXIT

opengauss_start

########################################
# ANALYZE
########################################
echo "ANALYZING..."
eval "$analyze_cmd"
echo "ANALYZE done"

warmup=5
measured=10

# Collect sorted query list
mapfile -t sql_files < <(find "$dir" -type f -name "*.sql" | sort)

# Warmup iterations
for w in $(seq 1 ${warmup}); do
    echo "Warmup iteration ${w}/${warmup}..."
    for sql in "${sql_files[@]}"; do
        sudo -i -u opengauss gsql -d "${opengauss_db}" -U "${opengauss_user}" \
            --host=localhost -p 7654 -W "${opengauss_pw}" -f "$sql"
    done
done

# Measured iterations
declare -A query_times

for i in $(seq 1 ${measured}); do
    echo "Measured iteration ${i}/${measured}..."
    for sql in "${sql_files[@]}"; do
        qname=$(basename "$sql")
        start_ns=$(date +%s%N)
        sudo -i -u opengauss gsql -d "${opengauss_db}" -U "${opengauss_user}" \
            --host=localhost -p 7654 -W "${opengauss_pw}" -f "$sql"
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
