#!/usr/bin/env bash
#
# Runtime-statistics-guided optimization breakdown: 5-step waterfall.
# Target: duckdb topdown query-jit tpde cache=single-run-template spec=off
#
# Each step enables ONE additional optimization on top of the previous,
# following the code's execution order.
#
# ============================================================
# Paper names, execution phase, and descriptions
# ============================================================
#
# Step 1: Baseline (all disabled)
#   No runtime-statistics-guided optimizations.
#
# Step 2: +Runtime Range Predicate Injection  (pre-execution phase)
#   Injects WHERE key BETWEEN min AND max into sub-plan SQL using
#   materialized temp table min/max statistics.
#
# (commented out) +Empty-Intermediate Pruning  (pre- and post-execution phase)
#   Pre: appends LIMIT 0 when a sub-plan references an empty temp table.
#   Post: stops the split loop when a temp returns 0 rows (all-inner guarantee).
#   Measured improvement: ~10ms (noise-level on JOB). Absorbed into All Enabled.
#
# Step 3: +Per-Row Build-Key Range Guard  (codegen phase)
#   Skips hash probe when probe key falls outside build-side [min, max].
#
# Step 4: +Min/Max Block Skipping  (codegen phase)
#   Skips entire 2048-row blocks when block-level [min, max] doesn't
#   overlap build-key range.
#
# Step 5: All Enabled (+Existence Pre-Check + Empty-Intermediate Pruning)
#   Existence Pre-Check: walks hash chain to verify key existence before
#   evaluating expensive filter predicates (gated to small builds <= 64K).
#   Reuses existing all-enabled CSV.
#
# ============================================================
# Waterfall measurement order (4 measured + 1 reused)
# ============================================================
#
# Step  Enabled optimizations                disable_runtime_opts (14th arg)
# ----  -------------------------------------  -----------------------------------------------
# 1     (none)                               range-pred,early-term,range-guard,block-skip,membership,bloom-filter
# 2     range-pred                           early-term,range-guard,block-skip,membership,bloom-filter
# 3     range-pred,range-guard               early-term,block-skip,membership,bloom-filter
# 4     range-pred,range-guard,block-skip    early-term,membership,bloom-filter
# 5     (all enabled)                        — reuse existing CSV
#
# Run from measure/ directory.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

COMMON="job duckdb topdown query none on on on all single-run-template off tpde"

# Step 1: Baseline — all runtime opts disabled
bash ./measure_breakdown_time_aqp.sh $COMMON "" \
    "range-pred,early-term,range-guard,block-skip,membership,bloom-filter" && \

# Step 2: +Runtime Range Predicate Injection
bash ./measure_breakdown_time_aqp.sh $COMMON "" \
    "early-term,range-guard,block-skip,membership,bloom-filter" && \

# (commented out) +Empty-Intermediate Pruning — ~10ms on JOB, absorbed into All Enabled
# bash ./measure_breakdown_time_aqp.sh $COMMON "" \
#     "range-guard,block-skip,membership,bloom-filter" && \

# Step 3: +Per-Row Build-Key Range Guard
bash ./measure_breakdown_time_aqp.sh $COMMON "" \
    "early-term,block-skip,membership,bloom-filter" && \

# Step 4: +Min/Max Block Skipping
bash ./measure_breakdown_time_aqp.sh $COMMON "" \
    "early-term,membership,bloom-filter" && \

# Step 5: All Enabled — reuse existing CSV
echo "Step 5: reusing existing all-enabled CSV:"
echo "  job_result/duckdb_topdown_query_none_jitcache_single_run_template_tpde_breakdown_time_log.csv"

echo ""
echo "=== Runtime-guided optimization breakdown complete ==="
