#!/usr/bin/env bash
# Test script for --jit-opt and --jit-simd flags.
# Runs all JOB queries with various flag combinations and checks
# (1) no crash, (2) results match baseline (no JIT).

set -uo pipefail

BINARY=../build_release/aqp_middleware
DB=/home/pei/Project/duckdb_132/measure/imdb.db
QUERIES="$JOB_PATH/queries"
RESULTS_DIR=test_jit_results
BASELINE_DIR="$RESULTS_DIR/baseline"

mkdir -p "$RESULTS_DIR" "$BASELINE_DIR"
rm -rf "$RESULTS_DIR"/*.out "$RESULTS_DIR"/*.stderr

shopt -s nullglob
ALL_SQLS=("$QUERIES"/*.sql)
shopt -u nullglob

if [[ ${#ALL_SQLS[@]} -eq 0 ]]; then
    echo "No SQL files found in $QUERIES"
    exit 1
fi
echo "Found ${#ALL_SQLS[@]} queries in $QUERIES"

run_query() {
    local label=$1
    local jit_args=$2
    local sql=$3
    local outbase=$(basename "$sql" .sql)

    $BINARY \
        --engine=duckdb \
        --db="$DB" \
        --split=none \
        --no-analyze \
        $jit_args \
        "$sql" \
        2>"$RESULTS_DIR/${label}_${outbase}.stderr" \
        >"$RESULTS_DIR/${label}_${outbase}.out"
}

echo "========================================"
echo "Step 1: Generate baseline (no JIT)"
echo "========================================"
for sql in "${ALL_SQLS[@]}"; do
    outbase=$(basename "$sql" .sql)
    echo "  $outbase"
    run_query "baseline" "--jit-level=none" "$sql"
    cp "$RESULTS_DIR/baseline_${outbase}.out" "$BASELINE_DIR/${outbase}.txt"
done

echo ""
echo "========================================"
echo "Step 2: Test --jit-opt levels (expr)"
echo "========================================"
for opt in O0 O1 O2 O3; do
    echo "  --- --jit-opt=$opt ---"
    for sql in "${ALL_SQLS[@]}"; do
        outbase=$(basename "$sql" .sql)
        run_query "opt_${opt}" "--jit-level=expr --jit-opt=$opt" "$sql"
    done
done

echo ""
echo "========================================"
echo "Step 3: Test --jit-simd levels (expr)"
echo "========================================"
for simd in off sse2 avx avx2 avx512 auto; do
    echo "  --- --jit-simd=$simd ---"
    for sql in "${ALL_SQLS[@]}"; do
        outbase=$(basename "$sql" .sql)
        run_query "simd_${simd}" "--jit-level=expr --jit-opt=O2 --jit-simd=$simd" "$sql"
    done
done

echo ""
echo "========================================"
echo "Step 4: Test operator level"
echo "========================================"
for sql in "${ALL_SQLS[@]}"; do
    outbase=$(basename "$sql" .sql)
    run_query "level_operator" "--jit-level=operator --jit-opt=O2 --jit-simd=avx2" "$sql"
done

echo ""
echo "========================================"
echo "Step 5: Compare results"
echo "========================================"
pass=0
fail=0
skip=0
fail_list=""
for f in "$RESULTS_DIR"/*.out; do
    fname=$(basename "$f")
    # Extract query name: everything after the last _ before .out
    outbase=$(echo "$fname" | sed 's/.*_\([0-9][0-9]*[a-z]\.sql\?\)\.out$/\1/')
    # More robust: strip prefix up to last underscore before digits
    outbase=$(echo "$fname" | grep -oP '\d+[a-z]\.out' | sed 's/\.out$//')
    label=$(echo "$fname" | sed "s/_${outbase}\.out$//")

    if [[ "$label" == "baseline" ]]; then
        continue
    fi

    baseline="$BASELINE_DIR/${outbase}.txt"
    if [[ ! -f "$baseline" ]]; then
        echo "  SKIP: $fname (no baseline)"
        ((skip++))
        continue
    fi

    if diff -q "$f" "$baseline" >/dev/null 2>&1; then
        ((pass++))
    else
        echo "  FAIL: $fname"
        diff "$f" "$baseline" | head -3
        ((fail++))
        fail_list="$fail_list $fname"
    fi
done

echo ""
echo "========================================"
echo "Results: $pass passed, $fail failed, $skip skipped"
echo "========================================"
if [[ -n "$fail_list" ]]; then
    echo "Failed:"
    for f in $fail_list; do
        echo "  $f"
    done
fi

# Show SIMD debug output snippets
echo ""
echo "SIMD activation samples:"
grep -h "using SIMD\|SIMD pipeline\|SIMD filter\|SIMD agg\|VW=" "$RESULTS_DIR"/*.stderr 2>/dev/null | sort -u | head -20

echo ""
echo "CPU/feature detection:"
grep -h "\[AQP-JIT\] CPU=" "$RESULTS_DIR"/*.stderr 2>/dev/null | sort -u | head -5

exit $fail
