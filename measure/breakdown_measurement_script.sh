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
# measure_breakdown_time_job.sh arg order:
#   1=engine 2=split 3=jit_level 4=jit_simd
#   5=payload_prune 6=prefetch 7=batch_probe 8=skip_hash_cmp
#   9=jit_cache 10=spec_jit 11=compile_mode 12=tune_config
#
set -e

## ============================================================
## lingodb LLVM backend
## ============================================================
#bash ./measure_breakdown_time_job.sh lingodb none llvm && \
#bash ./measure_breakdown_time_job.sh lingodb node-based llvm && \
#
## ============================================================
## lingodb TPDE backend
## ============================================================
#bash ./measure_breakdown_time_job.sh lingodb none tpde && \
#bash ./measure_breakdown_time_job.sh lingodb node-based tpde && \
#
## ============================================================
## lingo-db-runtime (DuckDB optimizer + LingoDB JIT runtime)
## ============================================================
#bash ./measure_breakdown_time_job.sh lingo-db-runtime node-based llvm && \
#bash ./measure_breakdown_time_job.sh lingo-db-runtime node-based tpde && \
#bash ./measure_breakdown_time_job.sh lingo-db-runtime none llvm && \
#bash ./measure_breakdown_time_job.sh lingo-db-runtime none tpde && \

# ============================================================
# Interpreter baseline (2 configs)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none none && \
bash ./measure_breakdown_time_job.sh duckdb node-based none && \

# ============================================================
# split=none: 3 jit-level × 3 compile-mode × 2 cache(off/full) = 18
# Grouped by: jit-level → compile-mode → cache
# ============================================================

# --- expr-jit ---
bash ./measure_breakdown_time_job.sh duckdb none expr none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb none expr none on on on all full off llvm && \
bash ./measure_breakdown_time_job.sh duckdb none expr none on on on all off off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb none expr none on on on all full off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb none expr none on on on all off off tpde && \
bash ./measure_breakdown_time_job.sh duckdb none expr none on on on all full off tpde && \

# --- operator-jit ---
bash ./measure_breakdown_time_job.sh duckdb none operator none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb none operator none on on on all full off llvm && \
bash ./measure_breakdown_time_job.sh duckdb none operator none on on on all off off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb none operator none on on on all full off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb none operator none on on on all off off tpde && \
bash ./measure_breakdown_time_job.sh duckdb none operator none on on on all full off tpde && \

# --- pipeline-jit (deprioritized) ---
#bash ./measure_breakdown_time_job.sh duckdb none pipeline none on on on all off off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb none pipeline none on on on all full off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb none pipeline none on on on all off off fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb none pipeline none on on on all full off fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb none pipeline none on on on all off off tpde && \
#bash ./measure_breakdown_time_job.sh duckdb none pipeline none on on on all full off tpde && \

# --- query-jit ---
bash ./measure_breakdown_time_job.sh duckdb none query none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb none query none on on on all full off llvm && \
bash ./measure_breakdown_time_job.sh duckdb none query none on on on all off off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb none query none on on on all full off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb none query none on on on all off off tpde && \
bash ./measure_breakdown_time_job.sh duckdb none query none on on on all full off tpde && \

# ============================================================
# node-based: (3 level × 3 mode + 1 tune) × 4 cache × 2 spec = 80
# Outer: spec-jit → cache-tier → { level × mode + tune }
# ============================================================

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

# ---- cache=off, spec=off: JIT configs (9) ----
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all off off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all off off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all off off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all off off tpde && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all off off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all off off fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all off off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py node-based && \

# ---- cache=off, spec=off: tune (1) ----
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off off llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-strict, spec=off (10) ----
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-strict off tpde && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-strict off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-strict off fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-strict off llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-template, spec=off (10) ----
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-template off tpde && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-template off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-template off fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-template off llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=full, spec=off (10) ----
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all full off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all full off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all full off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all full off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all full off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all full off tpde && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all full off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all full off fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all full off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all full off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all full off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all full off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all full off llvm job_result/tuned_per_subquery_node-based.json && \

# ------------------------------------------------------------
# spec-jit=recompile
# ------------------------------------------------------------

# ---- cache=off, spec=recompile (10) ----
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all off recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all off recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all off recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all off recompile tpde && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all off recompile llvm && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all off recompile fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all off recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off recompile llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-strict, spec=recompile (10) ----
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-strict recompile tpde && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-strict recompile llvm && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-strict recompile fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-strict recompile llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-template, spec=recompile (10) ----
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all single-run-template recompile tpde && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-template recompile llvm && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-template recompile fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all single-run-template recompile llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=full, spec=recompile (10) ----
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all full recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all full recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all full recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all full recompile tpde && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all full recompile llvm && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all full recompile fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all full recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all full recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all full recompile tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all full recompile llvm job_result/tuned_per_subquery_node-based.json && \

## ============================================================
## pipeline kernel (deprioritized)
## ============================================================
#bash ./measure_breakdown_time_job_kernel.sh duckdb none pipeline none on on on all && \
#bash ./measure_breakdown_time_job_kernel.sh duckdb node-based pipeline none on on on all && \

echo "=== All breakdown measurements complete ==="
