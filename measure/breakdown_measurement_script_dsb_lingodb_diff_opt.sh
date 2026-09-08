#!/usr/bin/env bash
#
# Performance breakdown: LinGo-DB with different DBMS optimizers on DSB benchmark.
# Compares lingo-db's own optimizer (default) vs DuckDB vs PostgreSQL optimizer.
# Split=none for all configs. Uses 1_instance_out_aqp/1/ query directory.
#
# Usage: bash breakdown_measurement_script_dsb_lingodb_diff_opt.sh [scale_factor]
# Run from measure/ directory.
#
set -e

DSB_SF="${1:-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# Own optimizer: 10 timeout queries
OWN_SKIP="query040|query050|query072|query101|query102"

# DuckDB optimizer: 15 crash/abort + 3 timeout
DUCKDB_SKIP="query018$|query019$|query025$|query038$|query040$|query050$|query072$|query084$|query085|query087$|query091$|query099$|query100$|query101$|query102"

# PG optimizer: 5 crash + 10 timeout
PG_SKIP="query018$|query038$|query040|query050|query072|query084$|query087$|query100$|query101|query102"

# ============================================================
# Step 1: LingoDB own optimizer (default, skip 10 timeout queries)
# ============================================================
echo "=== Step 1: LingoDB own optimizer ==="
AQP_SKIP_QUERIES="${OWN_SKIP}" \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} lingodb none llvm

# ============================================================
# Step 2: DuckDB optimizer (skip crash/abort/timeout queries)
# ============================================================
echo "=== Step 2: DuckDB optimizer ==="
AQP_LINGODB_PLAN_OPTIMIZER=duckdb \
AQP_SKIP_QUERIES="${DUCKDB_SKIP}" \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} lingodb none llvm

# ============================================================
# Step 3: PostgreSQL optimizer (skip crash/timeout queries)
# ============================================================
echo "=== Step 3: PostgreSQL optimizer ==="
AQP_LINGODB_PLAN_OPTIMIZER=postgres \
AQP_SKIP_QUERIES="${PG_SKIP}" \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} lingodb none llvm

echo ""
echo "=== LinGo-DB diff-opt DSB SF${DSB_SF} breakdown complete ==="
