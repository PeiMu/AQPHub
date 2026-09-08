#!/usr/bin/env bash
#
# Performance breakdown: LinGo-DB with different DBMS optimizers on JOB benchmark.
# Compares lingo-db's own optimizer (default) vs DuckDB vs PostgreSQL optimizer.
# Split=none for all configs.
#
# For PG optimizer: 27 JOB queries are skipped (they timeout due to suboptimal
# PG join ordering). Default and DuckDB optimizer measure all 113 queries.
#
# Run from measure/ directory.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# 27 queries that timeout with PG optimizer
PG_SKIP="12a|14c|15d|16b|16c|16d|17a|17e|17f|18c|19d|20a|22c|22d|26a|26c|29c|2a|2b|2d|33b|4a|4c|6d|6f|9c|9d"

# ============================================================
# Step 1: LingoDB own optimizer (default, all 113 queries)
# ============================================================
echo "=== Step 1: LingoDB own optimizer ==="
bash ./measure_breakdown_time_aqp.sh job lingodb none llvm

# ============================================================
# Step 2: DuckDB optimizer (all 113 queries)
# ============================================================
echo "=== Step 2: DuckDB optimizer ==="
AQP_LINGODB_PLAN_OPTIMIZER=duckdb \
bash ./measure_breakdown_time_aqp.sh job lingodb none llvm

# ============================================================
# Step 3: PostgreSQL optimizer (skip 27 timeout queries)
# ============================================================
echo "=== Step 3: PostgreSQL optimizer (86 queries, 27 skipped) ==="
AQP_LINGODB_PLAN_OPTIMIZER=postgres \
AQP_SKIP_QUERIES="${PG_SKIP}" \
bash ./measure_breakdown_time_aqp.sh job lingodb none llvm

echo ""
echo "=== LinGo-DB diff-opt JOB breakdown complete ==="
echo "Results in job_result/:"
echo "  lingodb_llvm_none_breakdown_time_log.csv                (own optimizer, 113 queries)"
echo "  lingodb_llvm_none_planopt_duckdb_breakdown_time_log.csv (DuckDB optimizer, 113 queries)"
echo "  lingodb_llvm_none_planopt_postgres_breakdown_time_log.csv (PG optimizer, 86 queries)"
