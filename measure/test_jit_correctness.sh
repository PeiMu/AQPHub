#!/usr/bin/env bash
#
# test_jit_correctness.sh — Verify JIT flag combinations produce correct results
#
# For each split strategy (none, relationship-center, node-based), runs queries
# with --jit-level=none as golden data, then re-runs with various JIT flag
# combinations and compares query output row-by-row.
#
# Usage:
#   bash ./test_jit_correctness.sh [query_dir]
#
# Default query_dir: $JOB_PATH/queries

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${SCRIPT_DIR}/../build_release/aqp_middleware"
DB_PATH="/home/pei/Project/duckdb_132/measure/imdb.db"
SCHEMA="/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql"
FKEYS="/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql"

QUERY_DIR="${1:-${JOB_PATH:-/home/pei/Project/benchmarks/imdb_job-postgres}/queries}"

SPLITS=("none" "relationship-center" "node-based")

RESULT_DIR="${SCRIPT_DIR}/test_jit_results/correctness"
mkdir -p "${RESULT_DIR}"

PASS=0
FAIL=0
SKIP=0
FAIL_LIST=""

########################################
# Extract query result rows from stdout.
# Captures lines between "=== Query Results ===" and "=== Execution completed ==="
# (the Rows:/Columns: header and all pipe-delimited data rows).
########################################
extract_results() {
    sed -n '/^=== Query Results ===/,/^=== Execution completed ===/p' \
        | grep -v '=== Query Results ===' \
        | grep -v '=== Execution completed ===' \
        | sort
}

########################################
# Run a single query and save extracted results to a file.
# Args: split, sql_file, output_file, [extra_flags...]
########################################
run_query() {
    local split="$1"
    local sql_file="$2"
    local out_file="$3"
    shift 3
    local extra_flags=("$@")

    local helper_db_arg=""
    if [[ "$split" == "node-based" ]]; then
        helper_db_arg="--helper-db-path=${DB_PATH}"
    fi

    "${BIN}" \
        --engine=duckdb \
        --db="${DB_PATH}" \
        ${helper_db_arg} \
        --schema="${SCHEMA}" \
        --fkeys="${FKEYS}" \
        --split="${split}" \
        --no-analyze \
        "${extra_flags[@]}" \
        "${sql_file}" \
        2>/dev/null | extract_results > "${out_file}"
}

########################################
# JIT flag combinations to test.
# Each entry: "description|flags..."
# All combinations use jit_opt=o2 and jit_simd=auto as representative settings.
########################################
JIT_COMBOS=(
    # Level: expr
    "expr_defaults|--jit-level=expr --jit-opt=o2 --jit-simd=auto"

    # Level: operator
    "operator_defaults|--jit-level=operator --jit-opt=o2 --jit-simd=auto"

    # Level: pipeline — all optimizations on (default)
    "pipeline_all_on|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto"

    # Level: pipeline — disable one optimization at a time
    "pipeline_no_fusion_build|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-fusion-build"
    "pipeline_no_fusion_probe|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-fusion-probe"
    "pipeline_no_inline_hash|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-inline-hash"
    "pipeline_no_payload_prune|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-payload-prune"
    "pipeline_no_prefetch|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-prefetch"
    "pipeline_no_batch_probe|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-batch-probe"
    "pipeline_prefetch_16|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --jit-prefetch=16"

    # Level: pipeline — all pipeline optimizations disabled
    "pipeline_all_off|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-fusion-build --no-jit-fusion-probe --no-jit-inline-hash --no-jit-payload-prune --no-jit-prefetch --no-jit-batch-probe"

    # Level: sql
    "sql_all_on|--jit-level=sql --jit-opt=o2 --jit-simd=auto"
    "sql_no_fusion_build|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-fusion-build"
    "sql_no_fusion_probe|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-fusion-probe"
    "sql_no_inline_hash|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-inline-hash"
    "sql_no_payload_prune|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-payload-prune"
    "sql_no_prefetch|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-prefetch"
    "sql_no_batch_probe|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-batch-probe"
    "sql_all_off|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-fusion-build --no-jit-fusion-probe --no-jit-inline-hash --no-jit-payload-prune --no-jit-prefetch --no-jit-batch-probe"

    # Varying optimization levels
    "pipeline_o0|--jit-level=pipeline --jit-opt=o0 --jit-simd=auto"
    "pipeline_o1|--jit-level=pipeline --jit-opt=o1 --jit-simd=auto"
    "pipeline_o3|--jit-level=pipeline --jit-opt=o3 --jit-simd=auto"

    # SIMD variants
    "pipeline_simd_off|--jit-level=pipeline --jit-opt=o2 --jit-simd=off"
    "pipeline_simd_sse2|--jit-level=pipeline --jit-opt=o2 --jit-simd=sse2"

    # Cache (should not change results)
    "pipeline_cache|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --jit-cache"
)

########################################
# Collect query files
########################################
shopt -s nullglob
SQL_FILES=("${QUERY_DIR}"/*.sql)
shopt -u nullglob

if [[ ${#SQL_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No .sql files found in ${QUERY_DIR}"
    exit 1
fi

echo "========================================="
echo " JIT Correctness Test"
echo "========================================="
echo "Queries: ${#SQL_FILES[@]} files in ${QUERY_DIR}"
echo "Splits:  ${SPLITS[*]}"
echo "Combos:  ${#JIT_COMBOS[@]}"
echo "========================================="

########################################
# Phase 1: Generate golden data (jit-level=none)
########################################
echo ""
echo "=== Phase 1: Generating golden data (--jit-level=none) ==="

for split in "${SPLITS[@]}"; do
    golden_dir="${RESULT_DIR}/golden/${split}"
    mkdir -p "${golden_dir}"

    for sql_file in "${SQL_FILES[@]}"; do
        qname="$(basename "${sql_file}" .sql)"
        golden_file="${golden_dir}/${qname}.txt"

        if [[ -f "${golden_file}" ]]; then
            continue
        fi

        echo "  [golden] split=${split} query=${qname}"
        if ! run_query "${split}" "${sql_file}" "${golden_file}" "--jit-level=none"; then
            echo "  WARNING: golden run failed for split=${split} query=${qname}"
            echo "FAILED" > "${golden_file}"
        fi
    done
done
echo "Golden data generated."

########################################
# Phase 2: Test each JIT combo against golden
########################################
echo ""
echo "=== Phase 2: Testing JIT flag combinations ==="

for split in "${SPLITS[@]}"; do
    golden_dir="${RESULT_DIR}/golden/${split}"

    for combo_entry in "${JIT_COMBOS[@]}"; do
        combo_name="${combo_entry%%|*}"
        combo_flags_str="${combo_entry#*|}"

        # Split flags string into array
        read -ra combo_flags <<< "${combo_flags_str}"

        test_dir="${RESULT_DIR}/${split}/${combo_name}"
        mkdir -p "${test_dir}"

        for sql_file in "${SQL_FILES[@]}"; do
            qname="$(basename "${sql_file}" .sql)"
            golden_file="${golden_dir}/${qname}.txt"
            test_file="${test_dir}/${qname}.txt"

            # Skip if golden failed
            if [[ "$(cat "${golden_file}" 2>/dev/null)" == "FAILED" ]]; then
                ((SKIP++))
                continue
            fi

            echo -n "  [${split}] ${combo_name} / ${qname} ... "

            if ! run_query "${split}" "${sql_file}" "${test_file}" "${combo_flags[@]}"; then
                echo "CRASH"
                ((FAIL++))
                FAIL_LIST+="  CRASH: split=${split} combo=${combo_name} query=${qname}\n"
                continue
            fi

            if diff -q "${golden_file}" "${test_file}" > /dev/null 2>&1; then
                echo "OK"
                ((PASS++))
            else
                echo "MISMATCH"
                ((FAIL++))
                FAIL_LIST+="  MISMATCH: split=${split} combo=${combo_name} query=${qname}\n"

                # Save diff for inspection
                diff_file="${test_dir}/${qname}.diff"
                diff -u "${golden_file}" "${test_file}" > "${diff_file}" 2>&1 || true
            fi
        done
    done
done

########################################
# Summary
########################################
echo ""
echo "========================================="
echo " Summary"
echo "========================================="
echo "  PASS:  ${PASS}"
echo "  FAIL:  ${FAIL}"
echo "  SKIP:  ${SKIP}"
echo "  TOTAL: $((PASS + FAIL + SKIP))"
echo "========================================="

if [[ ${FAIL} -gt 0 ]]; then
    echo ""
    echo "Failed tests:"
    echo -e "${FAIL_LIST}"
    echo "Diffs saved in: ${RESULT_DIR}/"
    exit 1
else
    echo "All tests passed."
    exit 0
fi
