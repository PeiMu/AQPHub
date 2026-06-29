#!/bin/bash
# tune_kernel_threshold.sh — Collect per-sub-query kernel vs DuckDB timing data
# for threshold tuning.
#
# Runs JOB benchmark in two modes for each split strategy:
#   1. kernel enabled (default) — kernel handles eligible sub-queries
#   2. kernel disabled (--no-kernel) — all sub-queries go to DuckDB
#
# Output: CSV files in $OUT_DIR, one per (strategy, mode) combination.
# Then run: python3 tune_kernel_threshold.py to analyze.

set -euo pipefail

MEASURE_DIR="$(cd "$(dirname "$0")" && pwd)"
AQP_DIR="$(dirname "$MEASURE_DIR")"
BUILD="$AQP_DIR/build_release"
BENCHMARK_DIR="/home/pei/Project/benchmarks/imdb_job-postgres"
DB="/home/pei/Project/duckdb/measure/imdb.db"

REPEAT=${1:-5}   # default 5 repeats (drop first 2 as warmup in analysis)
OUT_DIR="$MEASURE_DIR/tuning_data"
mkdir -p "$OUT_DIR"

HEADER="query,repeat,iteration,type,kernel_valid,kernel_used,scan_table,scan_rows,num_joins,num_filters,num_output_cols,exe_time_ms"

STRATEGIES="node-based relationship-center"
MODES="kernel no-kernel"

for STRATEGY in $STRATEGIES; do
  for MODE in $MODES; do
    LOG_FILE="$OUT_DIR/${STRATEGY}_${MODE}.csv"
    echo "$HEADER" > "$LOG_FILE"

    echo "=== Running $STRATEGY / $MODE (repeat=$REPEAT) ==="

    for query in "$BENCHMARK_DIR/queries/"*.sql; do
      QNAME=$(basename "$query" .sql)

      EXTRA_FLAGS=""
      if [ "$MODE" = "no-kernel" ]; then
        EXTRA_FLAGS="--no-kernel"
      fi

      AQP_KERNEL_LOG_FILE="$LOG_FILE" AQP_QUERY_NAME="$QNAME" \
        "$BUILD/aqp_middleware" \
          --engine=duckdb \
          --db="$DB" "" \
          --schema="$BENCHMARK_DIR/schema.sql" \
          --fkeys="$BENCHMARK_DIR/fkeys.sql" \
          --storage-cache="/tmp/imdb_storage_plan.cache" \
          --split="$STRATEGY" --no-analyze \
          --jit-level=operator --kernel-path=pipeline --jit-simd=none \
          --tuning $EXTRA_FLAGS \
          --repeat="$REPEAT" \
          "$query" > /dev/null 2>&1

      echo "  $QNAME done"
    done

    echo "  -> $LOG_FILE"
  done
done

echo ""
echo "Data collected in $OUT_DIR/"
echo "Run: python3 $MEASURE_DIR/tune_kernel_threshold.py $OUT_DIR"
