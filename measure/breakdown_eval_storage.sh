#!/usr/bin/env bash
#
# Bi-directional storage layer evaluation.
#
# Evaluates the two features of bi-directional storage independently:
#
#   Scan forwarding  – intermediate results are served to DuckDB via
#                      in-memory replacement-scan table functions instead
#                      of being materialised into catalog temp tables.
#
#   Mode bridging    – GetOrLoadQjitTemp() converts interpreter-produced
#                      temps (ColumnDataCollection) into QjitTable so that
#                      subsequent subqueries can still use query-JIT.
#
# ============================================================
# Steps
# ============================================================
#
# Step 1: Both enabled (default / baseline)
#
# Step 2: Scan forwarding disabled (--no-scan-forwarding)
#   After each subquery the result is ALSO copied into a DuckDB catalog
#   temp table, measuring the round-trip overhead that the replacement-
#   scan path avoids.
#
# Step 3: Mode bridging evaluation (--force-interpreter-nth=N)
#   For N = 2..5, forces every N-th subquery to use the interpreter,
#   then compares mode bridging ON (subsequent JIT subqueries can
#   consume interpreter temps) vs OFF (cascade fallback).
#
# Run from measure/ directory.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

COMMON="job duckdb topdown query none on on on all single-run-template off tpde"
DEST_DIR="${SCRIPT_DIR}/job_result"
D="duckdb_topdown_query_none_jitcache_single_run_template_tpde"

# Step 1: Both enabled (default)
echo "=== Step 1: Bi-directional storage enabled (baseline) ==="
#bash ./measure_breakdown_time_aqp.sh $COMMON && \
cp "${DEST_DIR}/${D}_breakdown_time_log.csv" \
   "${DEST_DIR}/storage_step1_bidir_enabled.csv" && \

# Step 2: Scan forwarding disabled
echo "=== Step 2: Scan forwarding disabled ==="
bash ./measure_breakdown_time_aqp.sh $COMMON "" \
    "no-scan-forwarding" && \
mv "${DEST_DIR}/${D}_noscanfwd_breakdown_time_log.csv" \
   "${DEST_DIR}/storage_step2_no_scan_forwarding.csv" && \

# Step 3: Mode bridging evaluation (sweep N=2..5)
for N in 1 2 3 4 5; do
  echo "=== Step 3a: force-interpreter-nth=${N}, mode bridging ON ==="
  bash ./measure_breakdown_time_aqp.sh $COMMON "" \
      "force-interpreter-nth=${N}" && \
  mv "${DEST_DIR}/${D}_forceinterp${N}_breakdown_time_log.csv" \
     "${DEST_DIR}/storage_step3a_nth${N}_bridge_on.csv" && \

  echo "=== Step 3b: force-interpreter-nth=${N}, mode bridging OFF ==="
  bash ./measure_breakdown_time_aqp.sh $COMMON "" \
      "force-interpreter-nth=${N},no-mode-bridging" && \
  mv "${DEST_DIR}/${D}_nomodebridg_forceinterp${N}_breakdown_time_log.csv" \
     "${DEST_DIR}/storage_step3b_nth${N}_bridge_off.csv"
done

echo ""
echo "=== Bi-directional storage evaluation complete ==="
echo "Output:"
echo "  ${DEST_DIR}/storage_step1_bidir_enabled.csv"
echo "  ${DEST_DIR}/storage_step2_no_scan_forwarding.csv"
for N in 1 2 3 4 5; do
  echo "  ${DEST_DIR}/storage_step3a_nth${N}_bridge_on.csv"
  echo "  ${DEST_DIR}/storage_step3b_nth${N}_bridge_off.csv"
done
