#!/usr/bin/env bash
#
# Performance breakdown: LinGo-DB with different DBMS optimizers on DSB benchmark.
# Tests split={none,topdown} x optimizer={own,duckdb,postgres}.
# Uses tpde mode (faster compilation).
#
# The plan optimizer flag only affects split=none execution. For split paths,
# sub-queries always use lingo-db's own MLIR optimizer, so the flag is a no-op.
# We still measure all combinations to quantify the (non-)effect.
#
# Usage: bash breakdown_measurement_script_dsb_lingodb_diff_opt.sh [scale_factor]
# Run from measure/ directory.
#
set -e

DSB_SF="${1:-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# Own optimizer: timeout queries
OWN_SKIP="query040|query050|query072|query101|query102"

# DuckDB optimizer: crash/abort/timeout
DUCKDB_SKIP="query018$|query019$|query025$|query038$|query040$|query050$|query072$|query084$|query085|query087$|query091$|query099$|query100$|query101$|query102"

# PG optimizer: crash/timeout
PG_SKIP="query018$|query038$|query040|query050|query072|query084$|query087$|query100$|query101|query102"

# ============================================================
# split=none: optimizer changes the query plan
# ============================================================

echo "=== Step 1: split=none, LingoDB own optimizer ==="
AQP_SKIP_QUERIES="${OWN_SKIP}" \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} lingodb none tpde

echo "=== Step 2: split=none, DuckDB optimizer ==="
AQP_LINGODB_PLAN_OPTIMIZER=duckdb \
AQP_SKIP_QUERIES="${DUCKDB_SKIP}" \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} lingodb none tpde

echo "=== Step 3: split=none, PostgreSQL optimizer ==="
AQP_LINGODB_PLAN_OPTIMIZER=postgres \
AQP_SKIP_QUERIES="${PG_SKIP}" \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} lingodb none tpde

# ============================================================
# split=topdown: optimizer flag is a no-op for split execution
# ============================================================

echo "=== Step 4: split=topdown, LingoDB own optimizer ==="
AQP_SKIP_QUERIES="${OWN_SKIP}" \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} lingodb topdown tpde

echo "=== Step 5: split=topdown, DuckDB optimizer ==="
AQP_LINGODB_PLAN_OPTIMIZER=duckdb \
AQP_SKIP_QUERIES="${DUCKDB_SKIP}" \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} lingodb topdown tpde

echo "=== Step 6: split=topdown, PostgreSQL optimizer ==="
AQP_LINGODB_PLAN_OPTIMIZER=postgres \
AQP_SKIP_QUERIES="${PG_SKIP}" \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} lingodb topdown tpde

echo ""
echo "=== LinGo-DB diff-opt DSB SF${DSB_SF} breakdown complete ==="
