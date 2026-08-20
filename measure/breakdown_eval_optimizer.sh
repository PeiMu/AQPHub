#!/usr/bin/env bash
#
# Evaluate the effect of the DBMS engine's optimizer on subquery execution.
#
# Compares two configurations:
#   1. Baseline: DuckDB optimizer enabled (default)
#   2. No optimizer: --disable-optimizer skips DuckDB's optimizer for
#      subquery Prepare() and DuckDBAdapter::Optimize() calls.
#      FilterOptimize (init IR) and split-decision EXPLAIN calls are
#      unaffected — only per-subquery execution optimization is disabled.
#
# Target: duckdb topdown query-jit tpde cache=single-run-template
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

COMMON="job duckdb topdown query none on on on all single-run-template off tpde"
DEST_DIR="${SCRIPT_DIR}/job_result"
D="duckdb_topdown_query_none_jitcache_single_run_template_tpde"

# Step 1: Baseline — optimizer enabled (default)
echo "=== Step 1: Baseline (optimizer enabled) ==="
#bash ./measure_breakdown_time_aqp.sh $COMMON && \
cp "${DEST_DIR}/${D}_breakdown_time_log.csv" \
   "${DEST_DIR}/eval_optimizer_baseline.csv" && \

# Step 2: Optimizer disabled
echo "=== Step 2: Optimizer disabled ==="
bash ./measure_breakdown_time_aqp.sh $COMMON "" "disable-optimizer" && \
mv "${DEST_DIR}/${D}_nooptimizer_breakdown_time_log.csv" \
   "${DEST_DIR}/eval_optimizer_disabled.csv" && \

echo ""
echo "=== Engine optimizer evaluation complete ==="
echo "Output:"
echo "  ${DEST_DIR}/eval_optimizer_baseline.csv"
echo "  ${DEST_DIR}/eval_optimizer_disabled.csv"
