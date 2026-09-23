#!/usr/bin/env bash
#
# count_spec_jit_rate.sh — Measure speculative JIT correct rate (how often
# the speculated subquery matches and no recompile is needed).
#
# The script patches ir_query_splitter.cpp to make the [AQP-SPECJIT] summary
# line print unconditionally (removing the enable_timing/enable_debug_print
# guard), rebuilds, runs all configs once, parses the summary from stderr,
# aggregates per-config, then reverts the patch and rebuilds.
#
# Correct rate = hits / (hits + compensate_fast)
#   hits            = speculation matched + bg compile succeeded (no recompile)
#   compensate_fast = speculation missed or bg error (TPDE recompile applied)
#
# "not_ready" means the speculation was correct but the bg compile hadn't
# finished yet — the main thread waited for it. This is still a HIT (no
# recompile), so not_ready is included in hits, not compensate_fast.
#
# Configurations:
#   1. job    duckdb     node-based single-run-template       recompile  tpde
#   2. dsb_50 duckdb     node-based single-run-template       recompile  tpde
#   3. job    postgresql node-based single-run-strict          recompile  fastisel
#   4. dsb_50 postgresql node-based single-run-parameterized   recompile  tpde
#   5. job    duckdb     node-based single-run-template       recompile  llvm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_FILE="${PROJECT}/src/split/ir_query_splitter.cpp"
BUILD_DIR="${PROJECT}/build_release"
RESULTS_FILE="${SCRIPT_DIR}/spec_jit_rate_results.txt"
STDERR_DIR="/tmp/aqp_spec_jit_dumps"

# bench  engine  split  jit_cache  compile_mode
CONFIGS=(
    "job       duckdb      node-based single-run-template       tpde"
    "dsb_50    duckdb      node-based single-run-template       tpde"
    "job       postgresql  node-based single-run-strict          fastisel"
    "dsb_50    postgresql  node-based single-run-parameterized   tpde"
    "job       duckdb      node-based single-run-template       llvm"
)

########################################
# Step 1: Patch ir_query_splitter.cpp
########################################
patch_source() {
    echo "=== Patching ${SRC_FILE} to print spec-jit summary unconditionally ==="

    cp "$SRC_FILE" "${SRC_FILE}.spec_jit_backup"

    python3 - "$SRC_FILE" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    text = f.read()

old = '  if ((config_.enable_debug_print || config_.enable_timing) &&\n      spec_hits_ + spec_misses_ + spec_card_misses_ + spec_not_ready_ +\n              spec_bg_errors_ + spec_compensate_fast_ +\n              spec_compensate_interp_ > 0) {'
new = '  if (spec_hits_ + spec_misses_ + spec_card_misses_ + spec_not_ready_ +\n              spec_bg_errors_ + spec_compensate_fast_ +\n              spec_compensate_interp_ > 0) {'

if old in text:
    text = text.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(text)
    print("  Removed enable_timing/enable_debug_print guard from summary line")
else:
    print("  WARNING: guard pattern not found, summary may already be unconditional")
PYEOF
}

########################################
# Step 2: Rebuild
########################################
rebuild() {
    echo "=== Rebuilding aqp_middleware ==="
    cmake --build "$BUILD_DIR" --target aqp_middleware -j "$(nproc)" 2>&1 | tail -5
    echo "  Build complete."
}

########################################
# Step 3: Revert patch
########################################
revert_source() {
    echo "=== Reverting patch ==="
    if [[ -f "${SRC_FILE}.spec_jit_backup" ]]; then
        mv "${SRC_FILE}.spec_jit_backup" "$SRC_FILE"
        echo "  Reverted from backup."
    else
        echo "  WARNING: no backup found, cannot revert!"
    fi
}

########################################
# Source env.sh for paths and DB connections
########################################
source "${SCRIPT_DIR}/env.sh"

########################################
# Engine start/stop helpers
########################################
pg_running=false
pg_start() {
    if ! "$pg_running"; then
        echo "  Starting PostgreSQL..."
        ${PG_BIN}/pg_ctl start -l "${PG_LOG}" -D "${PG_DATA}"
        pg_running=true
    fi
}
pg_stop() {
    if "$pg_running"; then
        echo "  Stopping PostgreSQL..."
        ${PG_BIN}/pg_ctl stop -D "${PG_DATA}" -m smart -s
        pg_running=false
    fi
}

########################################
# ANALYZE (once per engine+bench)
########################################
declare -A analyzed=()
run_analyze() {
    local bench="$1" engine="$2"
    local key="${engine}_${bench}"
    if [[ -n "${analyzed[$key]:-}" ]]; then
        return
    fi
    echo "  ANALYZING ${engine} ${bench}..."
    if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
        local pg_db
        if [[ "$bench" == "job" ]]; then
            pg_db="${PG_CONN_JOB}"
        else
            pg_db="${PG_CONN_DSB}"
        fi
        ${PG_BIN}/psql -d "${pg_db}" -c "ANALYZE;"
    fi
    analyzed[$key]=1
    echo "  ANALYZE done."
}

########################################
# Resolve benchmark paths and flags
########################################
resolve_config() {
    local bench="$1" engine="$2" split="$3" jit_cache="$4" compile_mode="$5"

    local real_bench="$bench"
    if [[ "$bench" == dsb_* ]]; then
        DSB_SF="${bench#dsb_}"
        real_bench="dsb"
    fi
    unset DUCKDB_DB_DSB STORAGE_CACHE_DUCKDB_DSB STORAGE_CACHE_PG_DSB PG_CONN_DSB
    source "${SCRIPT_DIR}/env.sh"

    if [[ "$real_bench" == "job" ]]; then
        _dir="${JOB_PATH:-${PROJECT}/../benchmarks/imdb_job-postgres}/queries"
        _schema="${JOB_PATH:-${PROJECT}/../benchmarks/imdb_job-postgres}/schema.sql"
        _fkeys="${JOB_PATH:-${PROJECT}/../benchmarks/imdb_job-postgres}/fkeys.sql"
        _duckdb_db="${DUCKDB_DB_JOB}"
        _storage_cache="${STORAGE_CACHE_DUCKDB_JOB}"
        _storage_cache_pg="${STORAGE_CACHE_PG_JOB}"
        _pg_db="${PG_CONN_JOB}"
    else
        if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
            _dir="${DSB_PATH:-${PROJECT}/../benchmarks/dsb-postgres}/code/tools/1_instance_out_aqp_pg/1/"
        else
            _dir="${DSB_PATH:-${PROJECT}/../benchmarks/dsb-postgres}/code/tools/1_instance_out_aqp/1/"
        fi
        _schema="${DSB_PATH:-${PROJECT}/../benchmarks/dsb-postgres}/scripts/create_tables.sql"
        _fkeys="${DSB_PATH:-${PROJECT}/../benchmarks/dsb-postgres}/scripts/tpcds_ri_umbra.sql"
        _duckdb_db="${DUCKDB_DB_DSB}"
        _storage_cache="${STORAGE_CACHE_DUCKDB_DSB}"
        _storage_cache_pg="${STORAGE_CACHE_PG_DSB}"
        _pg_db="${PG_CONN_DSB}"
    fi

    if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
        _storage_flags="--storage-plan --storage-cache=${_storage_cache_pg}"
    else
        _storage_flags="--storage-plan --storage-cache=${_storage_cache}"
    fi

    if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
        _db_conn="${_pg_db}"
    else
        _db_conn="${_duckdb_db}"
    fi

    _helper_db_arg=""
    if [[ ("$split" == "node-based" || "$split" == "topdown") && "$engine" != "duckdb" ]]; then
        _helper_db_arg="--helper-db-path=${_duckdb_db}"
    fi

    _compile_mode_flag=""
    if [[ "$compile_mode" != "llvm" ]]; then
        _compile_mode_flag="--compile-mode=${compile_mode}"
    fi
}

########################################
# Run one configuration, parse spec-jit summary from stderr
########################################
run_one() {
    local bench="$1" engine="$2" split="$3" jit_cache="$4" compile_mode="$5" stderr_file="$6"

    resolve_config "$bench" "$engine" "$split" "$jit_cache" "$compile_mode"

    echo "  Running: bench=${bench} engine=${engine} split=${split} cache=${jit_cache} compile=${compile_mode}"

    "${PROJECT}/build_release/aqp_middleware" \
        --engine="${engine}" \
        --db="${_db_conn}" \
        ${_helper_db_arg} \
        --schema="${_schema}" \
        --fkeys="${_fkeys}" \
        --split="${split}" \
        --no-analyze --jit-level=query --jit-simd=none \
        --jit-cache="${jit_cache}" \
        --spec-jit=recompile \
        ${_compile_mode_flag} \
        ${_storage_flags} \
        --benchmark \
        "${_dir}" \
        > /dev/null 2> "$stderr_file" || {
            echo "    WARNING: aqp_middleware exited with non-zero status"
        }

    if [[ ! -s "$stderr_file" ]]; then
        echo "    => No stderr output (no spec-jit activity?)"
        echo -e "${bench}\t${engine}\t${split}\t${jit_cache}\t${compile_mode}\t0\t0\t0\t0\t0\t0\tN/A" >> "$RESULTS_FILE"
        return
    fi

    # Aggregate all [AQP-SPECJIT] summary lines across queries in this run.
    local total_hits=0 total_misses=0 total_card_misses=0
    local total_not_ready=0 total_bg_errors=0 total_compensate_fast=0

    while IFS= read -r line; do
        local h m cm nr bgr cf
        h=$(echo "$line"  | grep -oP 'hits=\K[0-9]+')
        m=$(echo "$line"  | grep -oP '(?<= )misses=\K[0-9]+')
        cm=$(echo "$line" | grep -oP 'card_misses=\K[0-9]+')
        nr=$(echo "$line" | grep -oP 'not_ready=\K[0-9]+')
        bgr=$(echo "$line" | grep -oP 'bg_errors=\K[0-9]+')
        cf=$(echo "$line"  | grep -oP 'compensate_fast=\K[0-9]+' || echo 0)
        total_hits=$((total_hits + h))
        total_misses=$((total_misses + m))
        total_card_misses=$((total_card_misses + cm))
        total_not_ready=$((total_not_ready + nr))
        total_bg_errors=$((total_bg_errors + bgr))
        total_compensate_fast=$((total_compensate_fast + cf))
    done < <(grep '\[AQP-SPECJIT\] summary:' "$stderr_file")

    local total=$((total_hits + total_compensate_fast))
    local rate
    if (( total > 0 )); then
        rate=$(python3 -c "print(f'{100*${total_hits}/${total}:.1f}')")
    else
        rate="N/A"
    fi
    echo "    => hits=${total_hits} (not_ready=${total_not_ready}) compensate=${total_compensate_fast} total=${total} rate=${rate}%"
    echo "       (misses=${total_misses} card_misses=${total_card_misses} bg_errors=${total_bg_errors})"
    echo -e "${bench}\t${engine}\t${split}\t${jit_cache}\t${compile_mode}\t${total_hits}\t${total_compensate_fast}\t${total_misses}\t${total_card_misses}\t${total_not_ready}\t${total_bg_errors}\t${rate}%" >> "$RESULTS_FILE"
}

########################################
# Main
########################################
main() {
    mkdir -p "$STDERR_DIR"
    rm -f "$RESULTS_FILE"
    echo -e "BENCH\tENGINE\tSPLIT\tCACHE\tCOMPILE\tHITS\tCOMPENSATE\tMISSES\tCARD_MISS\tNOT_READY\tBG_ERR\tRATE" > "$RESULTS_FILE"

    # Patch and rebuild
    patch_source
    rebuild

    trap 'echo "Caught signal, reverting..."; revert_source; rebuild; pg_stop; exit 1' INT TERM

    # First pass: duckdb configs
    for config in "${CONFIGS[@]}"; do
        read -r bench engine split jit_cache compile_mode <<< "$config"
        [[ "$engine" != "duckdb" ]] && continue
        run_analyze "$bench" "$engine"
        safe_name="${bench}_${engine}_${split}_${jit_cache}_${compile_mode}"
        safe_name="${safe_name//\//_}"
        run_one "$bench" "$engine" "$split" "$jit_cache" "$compile_mode" \
            "${STDERR_DIR}/${safe_name}.stderr"
    done

    # Second pass: postgres configs
    local has_pg=false
    for config in "${CONFIGS[@]}"; do
        read -r _b engine _s _c _m <<< "$config"
        if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
            has_pg=true
            break
        fi
    done
    if "$has_pg"; then
        pg_start
        for config in "${CONFIGS[@]}"; do
            read -r bench engine split jit_cache compile_mode <<< "$config"
            [[ "$engine" != "postgres" && "$engine" != "postgresql" ]] && continue
            run_analyze "$bench" "$engine"
            safe_name="${bench}_${engine}_${split}_${jit_cache}_${compile_mode}"
            safe_name="${safe_name//\//_}"
            run_one "$bench" "$engine" "$split" "$jit_cache" "$compile_mode" \
                "${STDERR_DIR}/${safe_name}.stderr"
        done
        pg_stop
    fi

    # Revert patch and rebuild
    revert_source
    rebuild

    # Print summary table
    echo ""
    echo "=========================================================================="
    echo "                   Speculative JIT Correct Rate Summary"
    echo "=========================================================================="
    echo "  HITS = speculation correct, no recompile (includes not_ready waits)"
    echo "  COMP = speculation wrong, TPDE recompile applied"
    echo "  RATE = HITS / (HITS + COMP)"
    echo "=========================================================================="
    printf "%-10s %-12s %-12s %-28s %-10s %5s %5s %5s %7s\n" \
        "BENCH" "ENGINE" "SPLIT" "CACHE" "COMPILE" "HITS" "NRDY" "COMP" "RATE"
    echo "--------------------------------------------------------------------------"
    tail -n +2 "$RESULTS_FILE" | while IFS=$'\t' read -r bench engine split cache compile hits comp misses cm nr bge rate; do
        printf "%-10s %-12s %-12s %-28s %-10s %5s %5s %5s %7s\n" \
            "$bench" "$engine" "$split" "$cache" "$compile" "$hits" "$nr" "$comp" "$rate"
    done
    echo "=========================================================================="
    echo ""
    echo "Detailed breakdown (misses / card_misses / bg_errors):"
    tail -n +2 "$RESULTS_FILE" | while IFS=$'\t' read -r bench engine split cache compile hits comp misses cm nr bge rate; do
        printf "  %-10s %-12s: misses=%s card_misses=%s bg_errors=%s\n" \
            "$bench" "$engine" "$misses" "$cm" "$bge"
    done
    echo ""
    echo "Raw stderr files: ${STDERR_DIR}/"
    echo "Results TSV:      ${RESULTS_FILE}"
}

main "$@"
