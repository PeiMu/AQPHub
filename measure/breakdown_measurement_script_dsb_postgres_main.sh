#!/usr/bin/env bash
#
# PostgreSQL DSB full performance sweep.
# Run from measure/ directory.  Don't run anything else while measuring.
#
# Mirrors breakdown_measurement_script_dsb_duckdb.sh for PostgreSQL engine.
# Uses measure_breakdown_time_aqp.sh for each config (same format as DuckDB).
#
# measure_breakdown_time_aqp.sh dsb_${DSB_SF} arg order:
#   1=bench 2=engine 3=split 4=jit_level 5=jit_simd
#   6=payload_prune 7=prefetch 8=batch_probe 9=skip_hash_cmp
#   10=jit_cache 11=spec_jit 12=compile_mode 13=tune_config
#
# Usage: breakdown_measurement_script_dsb_postgres.sh [scale_factor]
#   scale_factor defaults to 10.
#
set -e

DSB_SF="${1:-10}"
if [[ "$DSB_SF" == "10" ]]; then
    RESULT_DIR="dsb_result"
else
    RESULT_DIR="dsb_result_sf${DSB_SF}"
fi
echo "=== DSB scale factor: ${DSB_SF} (results -> ${RESULT_DIR}/) ==="

# ============================================================
# Interpreter baseline (2 configs)
# ============================================================
# COMMENTED OUT: query032 times out with split=none on PostgreSQL DSB
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql none none && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based none && \

# ============================================================
# split=none: query-jit x 3 compile-mode x 2 cache(off/full) = 6
# ============================================================

# --- cache=off --- (COMMENTED OUT: query032 times out with split=none)
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql none query none on on on all off off llvm && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql none query none on on on all off off fastisel && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql none query none on on on all off off tpde && \

# --- cache=full --- (COMMENTED OUT: query032 times out with split=none)
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql none query none on on on all full off llvm && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql none query none on on on all full off fastisel && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql none query none on on on all full off tpde && \

# ============================================================
# node-based: (3 compile-mode + 1 tune) x 4 cache x 2 spec = 32
# Outer: spec -> cache -> { compile-mode + tune }
# ============================================================

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

# ---- cache=off, spec=off: 3 compile-mode + 1 tune = 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all off off llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all off off fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py --bench=dsb_${DSB_SF} --result-dir=${RESULT_DIR} --engine=postgresql && \

#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all off off llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

# ---- cache=single-run-strict, spec=off: 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-strict off tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all single-run-strict off llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

# ---- cache=single-run-template, spec=off: 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-template off tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all single-run-template off llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

## ---- cache=full, spec=off: 4 ----
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all full off llvm && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all full off fastisel && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all full off tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all full off llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

# ------------------------------------------------------------
# spec-jit=recompile
# ------------------------------------------------------------

# ---- cache=off, spec=recompile: 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all off recompile llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all off recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all off recompile llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

# ---- cache=single-run-strict, spec=recompile: 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-strict recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all single-run-strict recompile llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

# ---- cache=single-run-template, spec=recompile: 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all single-run-template recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all single-run-template recompile llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

## ---- cache=full, spec=recompile: 4 ----
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all full recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all full recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql node-based query none on on on all full recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all full recompile llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

# ============================================================
# topdown: mirrors node-based configs (query-jit only)
# 1 interpreter + (3 compile-mode + 1 tune) x 4 cache x 2 spec = 33
# ============================================================

# ---- interpreter baseline ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown none && \

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

# ---- cache=off, spec=off: 3 compile-mode + 1 tune = 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all off off llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all off off fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py --bench=dsb_${DSB_SF} --result-dir=${RESULT_DIR} --engine=postgresql && \

#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all off off llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

# ---- cache=single-run-strict, spec=off: 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-strict off tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all single-run-strict off llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

# ---- cache=single-run-template, spec=off: 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-template off tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all single-run-template off llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

## ---- cache=full, spec=off: 4 ----
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all full off llvm && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all full off fastisel && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all full off tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all full off llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

# ------------------------------------------------------------
# spec-jit=recompile
# ------------------------------------------------------------

# ---- cache=off, spec=recompile: 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all off recompile llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all off recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all off recompile llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

# ---- cache=single-run-strict, spec=recompile: 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-strict recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all single-run-strict recompile llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

# ---- cache=single-run-template, spec=recompile: 4 ----
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all single-run-template recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all single-run-template recompile llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

## ---- cache=full, spec=recompile: 4 ----
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all full recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all full recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql topdown query none on on on all full recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh dsb_${DSB_SF} postgresql auto query none on on on all full recompile llvm ${RESULT_DIR}/tuned_cross_split_postgresql.json && \

echo "=== All PostgreSQL DSB breakdown measurements complete ==="
