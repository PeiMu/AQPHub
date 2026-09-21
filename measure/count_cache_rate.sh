#!/usr/bin/env bash
#
# count_cache_rate.sh — Measure JIT cache hit rates across cache modes and configurations.
#
# Configurations tested:
#   1. duckdb   topdown    JOB
#   2. duckdb   none       JOB
#   3. duckdb   node-based DSB_50
#   4. duckdb   none       DSB_50
#   5. postgres node-based JOB
#   6. postgres node-based DSB_50
#
# Cache modes: single-run-strict, single-run-parameterized, single-run-template
#
# The script patches ir_to_llvm.cpp to enable AQP_DUMP_CACHE_KEYS in release
# builds, rebuilds, runs all configs, parses results, then reverts the patch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_FILE="${PROJECT}/src/jit/ir_to_llvm.cpp"
BUILD_DIR="${PROJECT}/build_release"
DUMP_DIR="/tmp/aqp_cache_rate_dumps"
RESULTS_FILE="${SCRIPT_DIR}/cache_rate_results.txt"

CACHE_MODES=("single-run-strict" "single-run-parameterized" "single-run-template")

CONFIGS=(
    "job       duckdb   topdown"
    "job       duckdb   none"
    "dsb_50    duckdb   node-based"
    "dsb_50    duckdb   none"
    "job       postgres node-based"
    "dsb_50    postgres node-based"
)

########################################
# Step 1: Patch ir_to_llvm.cpp
########################################
patch_source() {
    echo "=== Patching ${SRC_FILE} to enable AQP_DUMP_CACHE_KEYS in release builds ==="

    # Save a backup for clean revert
    cp "$SRC_FILE" "${SRC_FILE}.cache_rate_backup"

    python3 - "$SRC_FILE" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()

out = []
i = 0
removed = 0
while i < len(lines):
    line = lines[i].rstrip('\n')
    if line.strip() == '#ifndef NDEBUG':
        # Look ahead for AQP_DUMP_CACHE_KEYS or <fcntl.h> within the next 5 lines
        lookahead = ''.join(lines[i:i+6])
        if 'AQP_DUMP_CACHE_KEYS' in lookahead or '<fcntl.h>' in lookahead:
            # Skip this #ifndef NDEBUG line
            i += 1
            removed += 1
            # Now find and skip the matching #endif
            depth = 1
            block_lines = []
            while i < len(lines):
                l = lines[i].rstrip('\n')
                if l.strip().startswith('#ifndef') or l.strip().startswith('#ifdef'):
                    depth += 1
                    block_lines.append(lines[i])
                elif l.strip() == '#endif':
                    depth -= 1
                    if depth == 0:
                        # Skip this #endif line
                        i += 1
                        removed += 1
                        break
                    else:
                        block_lines.append(lines[i])
                else:
                    block_lines.append(lines[i])
                i += 1
            out.extend(block_lines)
            continue
    out.append(lines[i])
    i += 1

with open(path, 'w') as f:
    f.writelines(out)
print(f"  Removed {removed} #ifndef NDEBUG / #endif guard lines")
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
# Step 3: Revert patch (restore from backup, not git checkout)
########################################
revert_source() {
    echo "=== Reverting patch ==="
    if [[ -f "${SRC_FILE}.cache_rate_backup" ]]; then
        mv "${SRC_FILE}.cache_rate_backup" "$SRC_FILE"
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
    local bench="$1" engine="$2" split="$3" cache_mode="$4"

    local real_bench="$bench"
    if [[ "$bench" == dsb_* ]]; then
        DSB_SF="${bench#dsb_}"
        real_bench="dsb"
    fi
    # Unset derived vars so env.sh recomputes them with the current DSB_SF
    unset DUCKDB_DB_DSB STORAGE_CACHE_DUCKDB_DSB STORAGE_CACHE_PG_DSB PG_CONN_DSB
    source "${SCRIPT_DIR}/env.sh"

    # Query directory
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

    # Storage plan flags
    _storage_flags=""
    if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
        _storage_flags="--storage-plan --storage-cache=${_storage_cache_pg}"
    else
        _storage_flags="--storage-plan --storage-cache=${_storage_cache}"
    fi

    # DB connection
    if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
        _db_conn="${_pg_db}"
    else
        _db_conn="${_duckdb_db}"
    fi

    # Helper DB for non-DuckDB with split
    _helper_db_arg=""
    if [[ ("$split" == "node-based" || "$split" == "topdown") && "$engine" != "duckdb" ]]; then
        _helper_db_arg="--helper-db-path=${_duckdb_db}"
    fi
}

########################################
# Run one configuration (calls binary directly, no run_aqp.sh)
########################################
run_one() {
    local bench="$1" engine="$2" split="$3" cache_mode="$4" dump_file="$5"

    rm -f "$dump_file"

    resolve_config "$bench" "$engine" "$split" "$cache_mode"

    echo "  Running: bench=${bench} engine=${engine} split=${split} cache=${cache_mode}"

    AQP_DUMP_CACHE_KEYS="$dump_file" \
        "${PROJECT}/build_release/aqp_middleware" \
        --engine="${engine}" \
        --db="${_db_conn}" \
        ${_helper_db_arg} \
        --schema="${_schema}" \
        --fkeys="${_fkeys}" \
        --split="${split}" \
        --no-analyze --jit-level=query --jit-simd=avx2 \
        --jit-cache="${cache_mode}" \
        ${_storage_flags} \
        --benchmark \
        "${_dir}" \
        > /dev/null 2>&1 || {
            echo "    WARNING: aqp_middleware exited with non-zero status"
        }

    if [[ -f "$dump_file" ]]; then
        local total hit miss rate
        total=$(wc -l < "$dump_file")
        hit=$(grep -c 'HIT=1' "$dump_file" || true)
        miss=$(grep -c 'HIT=0' "$dump_file" || true)
        if (( total > 0 )); then
            rate=$(python3 -c "print(f'{100*${hit}/${total}:.1f}')")
        else
            rate="N/A"
        fi
        echo "    => total=${total} hit=${hit} miss=${miss} rate=${rate}%"
        echo -e "${bench}\t${engine}\t${split}\t${cache_mode}\t${total}\t${hit}\t${miss}\t${rate}%" >> "$RESULTS_FILE"
    else
        echo "    => No dump file produced (no JIT compilations?)"
        echo -e "${bench}\t${engine}\t${split}\t${cache_mode}\t0\t0\t0\tN/A" >> "$RESULTS_FILE"
    fi
}

########################################
# Main
########################################
main() {
    mkdir -p "$DUMP_DIR"
    rm -f "$RESULTS_FILE"
    echo -e "BENCH\tENGINE\tSPLIT\tCACHE_MODE\tTOTAL\tHIT\tMISS\tRATE" > "$RESULTS_FILE"

    # Patch and rebuild
    patch_source
    rebuild

    trap 'echo "Caught signal, reverting..."; revert_source; rebuild; pg_stop; exit 1' INT TERM

    # Group runs by engine to minimize start/stop cycles
    # First pass: duckdb configs (no engine to start)
    for config in "${CONFIGS[@]}"; do
        read -r bench engine split <<< "$config"
        [[ "$engine" != "duckdb" ]] && continue
        run_analyze "$bench" "$engine"
        for cache_mode in "${CACHE_MODES[@]}"; do
            safe_name="${bench}_${engine}_${split}_${cache_mode}"
            safe_name="${safe_name//\//_}"
            dump_file="${DUMP_DIR}/${safe_name}.tsv"
            run_one "$bench" "$engine" "$split" "$cache_mode" "$dump_file"
        done
    done

    # Second pass: postgres configs (start once, run all, stop once)
    local has_pg=false
    for config in "${CONFIGS[@]}"; do
        read -r _b engine _s <<< "$config"
        if [[ "$engine" == "postgres" || "$engine" == "postgresql" ]]; then
            has_pg=true
            break
        fi
    done
    if "$has_pg"; then
        pg_start
        for config in "${CONFIGS[@]}"; do
            read -r bench engine split <<< "$config"
            [[ "$engine" != "postgres" && "$engine" != "postgresql" ]] && continue
            run_analyze "$bench" "$engine"
            for cache_mode in "${CACHE_MODES[@]}"; do
                safe_name="${bench}_${engine}_${split}_${cache_mode}"
                safe_name="${safe_name//\//_}"
                dump_file="${DUMP_DIR}/${safe_name}.tsv"
                run_one "$bench" "$engine" "$split" "$cache_mode" "$dump_file"
            done
        done
        pg_stop
    fi

    # Revert patch and rebuild
    revert_source
    rebuild

    # Print summary table
    echo ""
    echo "=================================================================="
    echo "                   JIT Cache Hit Rate Summary"
    echo "=================================================================="
    printf "%-10s %-10s %-12s %-28s %6s %5s %5s %7s\n" \
        "BENCH" "ENGINE" "SPLIT" "CACHE_MODE" "TOTAL" "HIT" "MISS" "RATE"
    echo "------------------------------------------------------------------"
    tail -n +2 "$RESULTS_FILE" | while IFS=$'\t' read -r bench engine split cmode total hit miss rate; do
        printf "%-10s %-10s %-12s %-28s %6s %5s %5s %7s\n" \
            "$bench" "$engine" "$split" "$cmode" "$total" "$hit" "$miss" "$rate"
    done
    echo "=================================================================="
    echo ""
    echo "Raw dump files: ${DUMP_DIR}/"
    echo "Results TSV:    ${RESULTS_FILE}"
}

main "$@"
