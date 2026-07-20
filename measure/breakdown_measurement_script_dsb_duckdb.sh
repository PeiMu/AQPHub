#!/usr/bin/env bash
#
# Full performance sweep for DSB benchmark.
# Run from measure/ directory.  Don't run anything else while measuring.
#
# Mirrors breakdown_measurement_script_job.sh with DSB-specific paths.
#
# measure_breakdown_time_dsb.sh arg order:
#   1=engine 2=split 3=jit_level 4=jit_simd
#   5=payload_prune 6=prefetch 7=batch_probe 8=skip_hash_cmp
#   9=jit_cache 10=spec_jit 11=compile_mode 12=tune_config
#
# Usage: breakdown_measurement_script_dsb.sh [scale_factor]
#   scale_factor defaults to 10 (dsb_10.db, results in dsb_result/).
#   E.g. `breakdown_measurement_script_dsb.sh 100` uses dsb_100.db and
#   writes results to dsb_result_sf100/.
#
set -e

export DSB_SF=${1:-10}
if [[ "$DSB_SF" == "10" ]]; then
    RESULT_DIR="dsb_result"
else
    RESULT_DIR="dsb_result_sf${DSB_SF}"
fi
echo "=== DSB scale factor: ${DSB_SF} (results -> ${RESULT_DIR}/) ==="

## ============================================================
## lingodb LLVM backend
## ============================================================
#bash ./measure_breakdown_time_dsb.sh lingodb none llvm && \
#bash ./measure_breakdown_time_dsb.sh lingodb node-based llvm && \
#
## ============================================================
## lingodb TPDE backend
## ============================================================
#bash ./measure_breakdown_time_dsb.sh lingodb none tpde && \
#bash ./measure_breakdown_time_dsb.sh lingodb node-based tpde && \
#
## ============================================================
## lingo-db-runtime (DuckDB optimizer + LingoDB JIT runtime)
## ============================================================
#bash ./measure_breakdown_time_dsb.sh lingo-db-runtime node-based llvm && \
#bash ./measure_breakdown_time_dsb.sh lingo-db-runtime node-based tpde && \
#bash ./measure_breakdown_time_dsb.sh lingo-db-runtime none llvm && \
#bash ./measure_breakdown_time_dsb.sh lingo-db-runtime none tpde && \

# ============================================================
# Interpreter baseline (2 configs)
# ============================================================
bash ./measure_breakdown_time_dsb.sh duckdb none none && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based none && \

# ============================================================
# split=none: 3 jit-level x 3 compile-mode x 2 cache(off/full) = 18
# Grouped by: jit-level -> compile-mode -> cache
# ============================================================

# --- expr-jit ---
bash ./measure_breakdown_time_dsb.sh duckdb none expr none on on on all off off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb none expr none on on on all full off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb none expr none on on on all off off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb none expr none on on on all full off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb none expr none on on on all off off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb none expr none on on on all full off tpde && \

# --- operator-jit ---
bash ./measure_breakdown_time_dsb.sh duckdb none operator none on on on all off off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb none operator none on on on all full off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb none operator none on on on all off off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb none operator none on on on all full off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb none operator none on on on all off off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb none operator none on on on all full off tpde && \

# --- pipeline-jit (deprioritized) ---
#bash ./measure_breakdown_time_dsb.sh duckdb none pipeline none on on on all off off llvm && \
#bash ./measure_breakdown_time_dsb.sh duckdb none pipeline none on on on all full off llvm && \
#bash ./measure_breakdown_time_dsb.sh duckdb none pipeline none on on on all off off fastisel && \
#bash ./measure_breakdown_time_dsb.sh duckdb none pipeline none on on on all full off fastisel && \
#bash ./measure_breakdown_time_dsb.sh duckdb none pipeline none on on on all off off tpde && \
#bash ./measure_breakdown_time_dsb.sh duckdb none pipeline none on on on all full off tpde && \

# --- query-jit ---
bash ./measure_breakdown_time_dsb.sh duckdb none query none on on on all off off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb none query none on on on all full off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb none query none on on on all off off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb none query none on on on all full off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb none query none on on on all off off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb none query none on on on all full off tpde && \

# ============================================================
# node-based: (3 level x 3 mode + 1 tune) x 4 cache x 2 spec = 80
# Outer: spec-jit -> cache-tier -> { level x mode + tune }
# ============================================================

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

# ---- cache=off, spec=off: JIT configs (9) ----
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all off off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all off off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all off off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all off off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all off off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all off off tpde && \
#bash ./measure_breakdown_time_dsb.sh duckdb node-based pipeline none on on on all off off llvm && \
#bash ./measure_breakdown_time_dsb.sh duckdb node-based pipeline none on on on all off off fastisel && \
#bash ./measure_breakdown_time_dsb.sh duckdb node-based pipeline none on on on all off off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all off off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all off off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py --bench=dsb --result-dir=${RESULT_DIR} node-based && \

# ---- cache=off, spec=off: tune (1) ----
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all off off llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-strict, spec=off (10) ----
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-strict off llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-template, spec=off (10) ----
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-template off llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=full, spec=off (10) ----
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all full off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all full off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all full off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all full off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all full off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all full off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all full off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all full off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all full off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all full off llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ------------------------------------------------------------
# spec-jit=recompile
# ------------------------------------------------------------

# ---- cache=off, spec=recompile (10) ----
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all off recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all off recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all off recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all off recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all off recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all off recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all off recompile llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-strict, spec=recompile (10) ----
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-strict recompile llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-template, spec=recompile (10) ----
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all single-run-template recompile llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=full, spec=recompile (10) ----
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all full recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based expr none on on on all full recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all full recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based operator none on on on all full recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all full recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all full recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb node-based query none on on on all full recompile llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ============================================================
# topdown: mirrors node-based configs including tune
# 1 interpreter + (3 level x 3 mode + 1 tune) x 4 cache x 2 spec = 81
# ============================================================

# ---- interpreter baseline ----
bash ./measure_breakdown_time_dsb.sh duckdb topdown none && \

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

# ---- cache=off, spec=off (9) ----
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all off off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all off off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all off off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all off off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all off off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all off off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all off off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all off off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py --bench=dsb --result-dir=${RESULT_DIR} topdown && \

# ---- cache=off, spec=off: tune (1) ----
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all off off llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-strict, spec=off (9) ----
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-strict off llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-template, spec=off (9) ----
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-template off llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=full, spec=off (9) ----
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all full off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all full off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all full off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all full off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all full off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all full off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all full off llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all full off fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all full off tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all full off llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ------------------------------------------------------------
# spec-jit=recompile
# ------------------------------------------------------------

# ---- cache=off, spec=recompile (9) ----
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all off recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all off recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all off recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all off recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all off recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all off recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all off recompile llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-strict, spec=recompile (9) ----
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-strict recompile llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-template, spec=recompile (9) ----
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all single-run-template recompile llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=full, spec=recompile (9) ----
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all full recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown expr none on on on all full recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all full recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown operator none on on on all full recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all full recompile llvm && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all full recompile tpde && \
bash ./measure_breakdown_time_dsb.sh duckdb topdown query none on on on all full recompile llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

## ============================================================
## pipeline kernel (deprioritized)
## ============================================================
#bash ./measure_breakdown_time_dsb_kernel.sh duckdb none pipeline none on on on all && \
#bash ./measure_breakdown_time_dsb_kernel.sh duckdb node-based pipeline none on on on all && \

echo "=== All DSB breakdown measurements complete ==="
