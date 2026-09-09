#!/usr/bin/env bash
#
# Performance breakdown: LinGo-DB with different DBMS optimizers on JOB benchmark.
# Tests split={none,topdown} x optimizer={own,duckdb,postgres}.
# Uses tpde mode (faster compilation).
#
# The plan optimizer flag only affects split=none execution. For split paths,
# sub-queries always use lingo-db's own MLIR optimizer, so the flag is a no-op.
# We still measure all combinations to quantify the (non-)effect.
#
# For PG optimizer: 27 JOB queries are skipped (they timeout due to suboptimal
# PG join ordering). Default and DuckDB optimizer measure all 113 queries.
#
# Run from measure/ directory.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# 69 JOB queries that timeout or produce wrong results with PG optimizer.
# (27 timeout + 42 wrong results — PG's join ordering is suboptimal for lingo-db)
PG_SKIP="11c|11d|12a|12c|13a|14a|14b|14c|15d|16b|16c|16d|17a|17e|17f|18a|18b|18c|19a|19c|19d|20a|20c|21a|21c|22a|22b|22c|22d|24a|24b|25a|25b|25c|26a|26b|26c|28a|28b|28c|29a|29b|29c|2a|2b|2d|30a|30b|30c|31a|31b|31c|32b|33a|33b|33c|3a|3c|4a|4c|5c|6b|6d|6f|7c|9a|9b|9c|9d"

# ============================================================
# split=none: optimizer changes the query plan
# ============================================================

echo "=== Step 1: split=none, LingoDB own optimizer (113 queries) ==="
bash ./measure_breakdown_time_aqp.sh job lingodb none tpde

echo "=== Step 2: split=none, DuckDB optimizer (113 queries) ==="
AQP_LINGODB_PLAN_OPTIMIZER=duckdb \
bash ./measure_breakdown_time_aqp.sh job lingodb none tpde

echo "=== Step 3: split=none, PostgreSQL optimizer (44 queries, 69 skipped) ==="
AQP_LINGODB_PLAN_OPTIMIZER=postgres \
AQP_SKIP_QUERIES="${PG_SKIP}" \
bash ./measure_breakdown_time_aqp.sh job lingodb none tpde

# ============================================================
# split=topdown: optimizer flag is a no-op for split execution
# ============================================================

echo "=== Step 4: split=topdown, LingoDB own optimizer (113 queries) ==="
bash ./measure_breakdown_time_aqp.sh job lingodb topdown tpde

echo "=== Step 5: split=topdown, DuckDB optimizer (113 queries) ==="
AQP_LINGODB_PLAN_OPTIMIZER=duckdb \
bash ./measure_breakdown_time_aqp.sh job lingodb topdown tpde

echo "=== Step 6: split=topdown, PostgreSQL optimizer (44 queries, 69 skipped) ==="
AQP_LINGODB_PLAN_OPTIMIZER=postgres \
AQP_SKIP_QUERIES="${PG_SKIP}" \
bash ./measure_breakdown_time_aqp.sh job lingodb topdown tpde

echo ""
echo "=== LinGo-DB diff-opt JOB breakdown complete ==="
echo "Results in job_result/:"
echo "  split=none:"
echo "    lingodb_tpde_none_breakdown_time_log.csv                   (own optimizer, 113 queries)"
echo "    lingodb_tpde_none_planopt_duckdb_breakdown_time_log.csv    (DuckDB optimizer, 113 queries)"
echo "    lingodb_tpde_none_planopt_postgres_breakdown_time_log.csv  (PG optimizer, 44 queries)"
echo "  split=topdown:"
echo "    lingodb_tpde_topdown_breakdown_time_log.csv                   (own optimizer, 113 queries)"
echo "    lingodb_tpde_topdown_planopt_duckdb_breakdown_time_log.csv    (DuckDB optimizer, 113 queries)"
echo "    lingodb_tpde_topdown_planopt_postgres_breakdown_time_log.csv  (PG optimizer, 44 queries)"
