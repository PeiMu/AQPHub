#!/usr/bin/env bash
#
# correctness_test.sh — Test JIT flag combinations for correctness
#
# Golden data: duckdb with --jit-level=none for each split strategy.
# Then re-runs every JIT flag combo per split and compares row-by-row.
#
# Usage:
#   bash ./correctness_test.sh [--jit-level=expr|operator|pipeline|sql|all] [query_dir]
#
#   --jit-level   Only test combos for the given JIT level (default: all)
#   query_dir     Default: $JOB_PATH/queries
#
# Examples:
#   bash ./correctness_test.sh --jit-level=expr          # test expr-level only
#   bash ./correctness_test.sh --jit-level=pipeline      # test pipeline-level only
#   bash ./correctness_test.sh                           # test all levels

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${SCRIPT_DIR}/../build_release/aqp_middleware"
DB_PATH="/home/pei/Project/duckdb_132/measure/imdb.db"
SCHEMA="/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql"
FKEYS="/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql"

# Parse --jit-level argument
JIT_LEVEL_FILTER="all"
QUERY_DIR=""

for arg in "$@"; do
    if [[ "$arg" == --jit-level=* ]]; then
        JIT_LEVEL_FILTER="${arg#--jit-level=}"
    else
        QUERY_DIR="$arg"
    fi
done
QUERY_DIR="${QUERY_DIR:-${JOB_PATH:-/home/pei/Project/benchmarks/imdb_job-postgres}/queries}"

case "$JIT_LEVEL_FILTER" in
    expr|operator|pipeline|sql|all) ;;
    *) echo "ERROR: Unknown --jit-level='$JIT_LEVEL_FILTER' (valid: expr, operator, pipeline, sql, all)"; exit 1 ;;
esac

SPLITS=("none" "relationship-center" "node-based")

RESULT_DIR="${SCRIPT_DIR}/correctness_results"
mkdir -p "${RESULT_DIR}"

PASS=0
FAIL=0
SKIP=0
FAIL_LIST=""

########################################
# Extract query result rows from stdout.
# Captures lines between "=== Query Results ===" and "=== Execution completed ==="
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
# JIT flag combinations, organized by level.
# Format: "name|flags..."
########################################
COMBOS_EXPR=(
    "expr_o2_auto|--jit-level=expr --jit-opt=o2 --jit-simd=auto"
)

COMBOS_OPERATOR=(
    "operator_o2_auto|--jit-level=operator --jit-opt=o2 --jit-simd=auto"
)

COMBOS_PIPELINE=(
    # defaults
    "pipeline_o2_auto|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto"

    # optimization levels
    "pipeline_o0_auto|--jit-level=pipeline --jit-opt=o0 --jit-simd=auto"
    "pipeline_o1_auto|--jit-level=pipeline --jit-opt=o1 --jit-simd=auto"
    "pipeline_o3_auto|--jit-level=pipeline --jit-opt=o3 --jit-simd=auto"

    # SIMD variants
    "pipeline_o2_off|--jit-level=pipeline --jit-opt=o2 --jit-simd=off"
    "pipeline_o2_sse2|--jit-level=pipeline --jit-opt=o2 --jit-simd=sse2"
    "pipeline_o2_avx|--jit-level=pipeline --jit-opt=o2 --jit-simd=avx"
    "pipeline_o2_avx2|--jit-level=pipeline --jit-opt=o2 --jit-simd=avx2"
    "pipeline_o2_avx512|--jit-level=pipeline --jit-opt=o2 --jit-simd=avx512"

    # toggle: disable one at a time
    "pipe_no_fusion_build|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-fusion-build"
    "pipe_no_fusion_probe|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-fusion-probe"
    "pipe_no_inline_hash|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-inline-hash"
    "pipe_no_payload_prune|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-payload-prune"
    "pipe_no_prefetch|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-prefetch"
    "pipe_no_batch_probe|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-batch-probe"

    # all pipeline opts off
    "pipe_all_off|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --no-jit-fusion-build --no-jit-fusion-probe --no-jit-inline-hash --no-jit-payload-prune --no-jit-prefetch --no-jit-batch-probe"

    # prefetch distance
    "pipe_pf16|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --jit-prefetch=16"
    "pipe_pf32|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --jit-prefetch=32"

    # cache
    "pipeline_cache|--jit-level=pipeline --jit-opt=o2 --jit-simd=auto --jit-cache"
)

COMBOS_SQL=(
    "sql_o2_auto|--jit-level=sql --jit-opt=o2 --jit-simd=auto"

    # toggle: disable one at a time
    "sql_no_fusion_build|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-fusion-build"
    "sql_no_fusion_probe|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-fusion-probe"
    "sql_no_inline_hash|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-inline-hash"
    "sql_no_payload_prune|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-payload-prune"
    "sql_no_prefetch|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-prefetch"
    "sql_no_batch_probe|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-batch-probe"
    "sql_all_off|--jit-level=sql --jit-opt=o2 --jit-simd=auto --no-jit-fusion-build --no-jit-fusion-probe --no-jit-inline-hash --no-jit-payload-prune --no-jit-prefetch --no-jit-batch-probe"
)

########################################
# Build JIT_COMBOS based on --jit-level filter
########################################
JIT_COMBOS=()

if [[ "$JIT_LEVEL_FILTER" == "all" || "$JIT_LEVEL_FILTER" == "expr" ]]; then
    JIT_COMBOS+=("${COMBOS_EXPR[@]}")
fi
if [[ "$JIT_LEVEL_FILTER" == "all" || "$JIT_LEVEL_FILTER" == "operator" ]]; then
    JIT_COMBOS+=("${COMBOS_OPERATOR[@]}")
fi
if [[ "$JIT_LEVEL_FILTER" == "all" || "$JIT_LEVEL_FILTER" == "pipeline" ]]; then
    JIT_COMBOS+=("${COMBOS_PIPELINE[@]}")
fi
if [[ "$JIT_LEVEL_FILTER" == "all" || "$JIT_LEVEL_FILTER" == "sql" ]]; then
    JIT_COMBOS+=("${COMBOS_SQL[@]}")
fi

if [[ ${#JIT_COMBOS[@]} -eq 0 ]]; then
    echo "ERROR: No JIT combos selected for --jit-level=${JIT_LEVEL_FILTER}"
    exit 1
fi

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

TOTAL_TESTS=$((${#SQL_FILES[@]} * ${#SPLITS[@]} * ${#JIT_COMBOS[@]}))

echo "========================================="
echo " JIT Correctness Test"
echo "========================================="
echo " Binary:    ${BIN}"
echo " JIT level: ${JIT_LEVEL_FILTER}"
echo " Queries:   ${#SQL_FILES[@]} files in ${QUERY_DIR}"
echo " Splits:    ${SPLITS[*]}"
echo " Combos:    ${#JIT_COMBOS[@]}"
echo " Total:     ${TOTAL_TESTS} test cases (+ ${#SQL_FILES[@]} * ${#SPLITS[@]} golden)"
echo "========================================="
echo ""

# Verify binary exists
if [[ ! -x "${BIN}" ]]; then
    echo "ERROR: Binary not found or not executable: ${BIN}"
    exit 1
fi

########################################
# Phase 1: Generate golden data (jit-level=none)
########################################
echo "=== Phase 1: Generating golden data (--jit-level=none) ==="

for split in "${SPLITS[@]}"; do
    golden_dir="${RESULT_DIR}/golden/${split}"
    mkdir -p "${golden_dir}"

    for sql_file in "${SQL_FILES[@]}"; do
        qname="$(basename "${sql_file}" .sql)"
        golden_file="${golden_dir}/${qname}.txt"

        if [[ -f "${golden_file}" && "$(cat "${golden_file}" 2>/dev/null)" != "FAILED" ]]; then
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
echo ""

########################################
# Phase 2: Test each JIT combo against golden
########################################
echo "=== Phase 2: Testing JIT flag combinations ==="

tested=0
for split in "${SPLITS[@]}"; do
    golden_dir="${RESULT_DIR}/golden/${split}"

    for combo_entry in "${JIT_COMBOS[@]}"; do
        combo_name="${combo_entry%%|*}"
        combo_flags_str="${combo_entry#*|}"
        read -ra combo_flags <<< "${combo_flags_str}"

        test_dir="${RESULT_DIR}/${split}/${combo_name}"
        mkdir -p "${test_dir}"

        for sql_file in "${SQL_FILES[@]}"; do
            qname="$(basename "${sql_file}" .sql)"
            golden_file="${golden_dir}/${qname}.txt"
            test_file="${test_dir}/${qname}.txt"

            ((tested++)) || true

            # Skip if golden failed
            if [[ "$(cat "${golden_file}" 2>/dev/null)" == "FAILED" ]]; then
                ((SKIP++)) || true
                continue
            fi

            echo -n "  [${tested}/${TOTAL_TESTS}] ${split}/${combo_name}/${qname} ... "

            if ! run_query "${split}" "${sql_file}" "${test_file}" "${combo_flags[@]}"; then
                echo "CRASH"
                ((FAIL++)) || true
                FAIL_LIST+="  CRASH: split=${split} combo=${combo_name} query=${qname}\n"
                continue
            fi

            if diff -q "${golden_file}" "${test_file}" > /dev/null 2>&1; then
                echo "OK"
                ((PASS++)) || true
            else
                echo "MISMATCH"
                ((FAIL++)) || true
                FAIL_LIST+="  MISMATCH: split=${split} combo=${combo_name} query=${qname}\n"
                diff -u "${golden_file}" "${test_file}" > "${test_dir}/${qname}.diff" 2>&1 || true
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
echo "  TOTAL: ${TOTAL_TESTS}"
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
