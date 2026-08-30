#!/usr/bin/env bash
#
# Bi-directional storage layer evaluation.
#
# Compares query-JIT with the middleware's replacement-scan-based temp
# storage (bi-directional) against a baseline that also materializes
# every intermediate result into a DuckDB catalog temp table (the
# round-trip that would be required without bi-directional storage).
#
# ============================================================
# Steps
# ============================================================
#
# Step 1: Bi-directional storage enabled (default)
#   Intermediate results stay in middleware memory (QjitTable /
#   ColumnDataCollection) and are served to DuckDB via replacement
#   scan table functions (scan_qjit_temp / scan_temp_collection).
#
# Step 2: Bi-directional storage disabled
#   Same as Step 1, but after each subquery the result is ALSO
#   copied into a DuckDB catalog temp table (CREATE TABLE +
#   LocalAppend), measuring the round-trip overhead that the
#   replacement-scan path avoids.
#
# Run from measure/ directory.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

COMMON="job duckdb topdown query none on on on all single-run-structural off tpde"
DEST_DIR="${SCRIPT_DIR}/job_result"
D="duckdb_topdown_query_none_jitcache_single_run_template_tpde"

# Step 1: Bi-directional storage enabled (default)
echo "=== Step 1: Bi-directional storage enabled ==="
#bash ./measure_breakdown_time_aqp.sh $COMMON && \
cp "${DEST_DIR}/${D}_breakdown_time_log.csv" \
   "${DEST_DIR}/storage_step1_bidir_enabled.csv" && \

# Step 2: Bi-directional storage disabled
echo "=== Step 2: Bi-directional storage disabled ==="
bash ./measure_breakdown_time_aqp.sh $COMMON "" \
    "disable-bi-directional-storage" && \
mv "${DEST_DIR}/${D}_nobidirstorage_breakdown_time_log.csv" \
   "${DEST_DIR}/storage_step2_bidir_disabled.csv" && \

echo ""
echo "=== Bi-directional storage evaluation complete ==="
echo "Output: ${DEST_DIR}/storage_step[1-2]_*.csv"
