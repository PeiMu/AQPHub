#!/usr/bin/env bash
#
# Full performance sweep: 100 configs.
# Run from measure/ directory.  Don't run anything else while measuring.
#
# Formula:
#   2 interpreter
#   + 1 none-split × 3 jit-level × 3 compile-mode × 2 cache(off/full) = 18
#   + 1 node-based × (3 jit-level × 3 compile-mode + 1 tune) × 4 cache × 2 spec(off/recompile) = 80
#   = 100 configs
#
# measure_breakdown_time_aqp.sh arg order: job <engine> ...
#   1=bench 2=engine 3=split 4=jit_level 5=jit_simd
#   6=payload_prune 7=prefetch 8=batch_probe 9=skip_hash_cmp
#   10=jit_cache 11=spec_jit 12=compile_mode 13=tune_config
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

## ============================================================
## lingodb LLVM backend
## ============================================================
#bash ./measure_breakdown_time_aqp.sh job lingodb none llvm && \
#bash ./measure_breakdown_time_aqp.sh job lingodb node-based llvm && \
#
## ============================================================
## lingodb TPDE backend
## ============================================================
#bash ./measure_breakdown_time_aqp.sh job lingodb none tpde && \
#bash ./measure_breakdown_time_aqp.sh job lingodb node-based tpde && \
#
## ============================================================
## lingo-db-runtime (DuckDB optimizer + LingoDB JIT runtime)
## ============================================================
#bash ./measure_breakdown_time_aqp.sh job lingo-db-runtime node-based llvm && \
#bash ./measure_breakdown_time_aqp.sh job lingo-db-runtime node-based tpde && \
#bash ./measure_breakdown_time_aqp.sh job lingo-db-runtime none llvm && \
#bash ./measure_breakdown_time_aqp.sh job lingo-db-runtime none tpde && \

# ============================================================
# Interpreter baseline (2 configs)
# ============================================================
bash ./measure_breakdown_time_aqp.sh job duckdb none none && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based none && \

# ============================================================
# split=none: 3 jit-level × 3 compile-mode × 2 cache(off/full) = 18
# Grouped by: jit-level → compile-mode → cache
# ============================================================

## --- expr-jit ---
#bash ./measure_breakdown_time_aqp.sh job duckdb none expr none on on on all off off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb none expr none on on on all full off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb none expr none on on on all off off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb none expr none on on on all full off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb none expr none on on on all off off tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb none expr none on on on all full off tpde && \
#
## --- operator-jit ---
#bash ./measure_breakdown_time_aqp.sh job duckdb none operator none on on on all off off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb none operator none on on on all full off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb none operator none on on on all off off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb none operator none on on on all full off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb none operator none on on on all off off tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb none operator none on on on all full off tpde && \
#
## --- pipeline-jit (deprioritized) ---
##bash ./measure_breakdown_time_aqp.sh job duckdb none pipeline none on on on all off off llvm && \
##bash ./measure_breakdown_time_aqp.sh job duckdb none pipeline none on on on all full off llvm && \
##bash ./measure_breakdown_time_aqp.sh job duckdb none pipeline none on on on all off off fastisel && \
##bash ./measure_breakdown_time_aqp.sh job duckdb none pipeline none on on on all full off fastisel && \
##bash ./measure_breakdown_time_aqp.sh job duckdb none pipeline none on on on all off off tpde && \
##bash ./measure_breakdown_time_aqp.sh job duckdb none pipeline none on on on all full off tpde && \

# --- query-jit ---
bash ./measure_breakdown_time_aqp.sh job duckdb none query none on on on all off off llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb none query none on on on all full off llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb none query none on on on all off off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb none query none on on on all full off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb none query none on on on all off off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb none query none on on on all full off tpde && \

# ============================================================
# node-based: (3 level × 3 mode + 1 tune) × 4 cache × 2 spec = 80
# Outer: spec-jit → cache-tier → { level × mode + tune }
# ============================================================

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

## ---- cache=off, spec=off: JIT configs (9) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all off off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all off off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all off off tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all off off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all off off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all off off tpde && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all off off llvm && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all off off fastisel && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all off off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all off off llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all off off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py && \

# ---- cache=off, spec=off: tune (1) ----
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all off off llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=single-run-strict, spec=off (10) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-strict off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-strict off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-strict off tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-strict off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-strict off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-strict off tpde && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-strict off llvm && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-strict off fastisel && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-strict off llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=single-run-template, spec=off (10) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-template off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-template off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-template off tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-template off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-template off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-template off tpde && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-template off llvm && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-template off fastisel && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-template off llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=full, spec=off (10) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all full off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all full off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all full off tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all full off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all full off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all full off tpde && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all full off llvm && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all full off fastisel && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all full off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all full off llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all full off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all full off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all full off llvm job_result/tuned_cross_split_duckdb.json && \

# ------------------------------------------------------------
# spec-jit=recompile
# ------------------------------------------------------------

## ---- cache=off, spec=recompile (10) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all off recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all off recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all off recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all off recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all off recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all off recompile tpde && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all off recompile llvm && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all off recompile fastisel && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all off recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all off recompile llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all off recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all off recompile llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=single-run-strict, spec=recompile (10) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-strict recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-strict recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-strict recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-strict recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-strict recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-strict recompile tpde && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-strict recompile llvm && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-strict recompile fastisel && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-strict recompile llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=single-run-template, spec=recompile (10) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-template recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-template recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all single-run-template recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-template recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-template recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all single-run-template recompile tpde && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-template recompile llvm && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-template recompile fastisel && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-template recompile llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=full, spec=recompile (10) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all full recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all full recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based expr none on on on all full recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all full recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all full recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based operator none on on on all full recompile tpde && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all full recompile llvm && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all full recompile fastisel && \
##bash ./measure_breakdown_time_aqp.sh job duckdb node-based pipeline none on on on all full recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all full recompile llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on all full recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all full recompile llvm job_result/tuned_cross_split_duckdb.json && \

# ============================================================
# topdown: mirrors node-based configs including tune
# 1 interpreter + (3 level × 3 mode + 1 tune) × 4 cache × 2 spec = 81
# Outer: spec-jit → cache-tier → { level × mode + tune }
# ============================================================

# ---- interpreter baseline ----
bash ./measure_breakdown_time_aqp.sh job duckdb topdown none && \

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

## ---- cache=off, spec=off (9) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all off off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all off off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all off off tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all off off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all off off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all off off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all off off llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all off off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py && \

# ---- cache=off, spec=off: tune (1) ----
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all off off llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=single-run-strict, spec=off (9) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-strict off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-strict off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-strict off tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-strict off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-strict off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-strict off llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=single-run-template, spec=off (9) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-template off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-template off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-template off tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-template off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-template off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-template off llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=full, spec=off (9) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all full off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all full off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all full off tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all full off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all full off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all full off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all full off llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all full off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all full off tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all full off llvm job_result/tuned_cross_split_duckdb.json && \

# ------------------------------------------------------------
# spec-jit=recompile (NOTE: spec-jit currently gated to NODE_BASED;
# these run topdown without spec-jit, kept for future ungating)
# ------------------------------------------------------------

## ---- cache=off, spec=recompile (9) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all off recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all off recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all off recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all off recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all off recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all off recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all off recompile llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all off recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all off recompile llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=single-run-strict, spec=recompile (9) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-strict recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-strict recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-strict recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-strict recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-strict recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-strict recompile llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=single-run-template, spec=recompile (9) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-template recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-template recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all single-run-template recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-template recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-template recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-template recompile llvm job_result/tuned_cross_split_duckdb.json && \

## ---- cache=full, spec=recompile (9) ----
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all full recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all full recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown expr none on on on all full recompile tpde && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all full recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all full recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb topdown operator none on on on all full recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all full recompile llvm && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all full recompile tpde && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all full recompile llvm job_result/tuned_cross_split_duckdb.json && \

## ============================================================
## pipeline kernel (deprioritized)
## ============================================================
#bash ./measure_breakdown_time_job_kernel.sh duckdb none pipeline none on on on all && \
#bash ./measure_breakdown_time_job_kernel.sh duckdb node-based pipeline none on on on all && \

echo "=== All breakdown measurements complete ==="
