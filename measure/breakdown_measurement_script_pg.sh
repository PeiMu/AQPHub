#!/usr/bin/env bash
#
# PostgreSQL full performance sweep.
# Run from measure/ directory.  Don't run anything else while measuring.
#
# Uses measure_breakdown_time_job.sh as the per-config runner (same as DuckDB).
# Arg order for measure_breakdown_time_job.sh:
#   1=engine 2=split 3=jit_level 4=jit_simd
#   5=payload_prune 6=prefetch 7=batch_probe 8=skip_hash_cmp
#   9=jit_cache 10=spec_jit 11=compile_mode 12=tune_config
#
# PostgreSQL-irrelevant flags (payload_prune, prefetch, batch_probe) are passed
# as their defaults (on, on, on) for consistency with the shared script.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ============================================================
# Interpreter baseline (3 configs: none, node-based, topdown)
# ============================================================
bash ./measure_breakdown_time_job.sh postgresql none none && \
bash ./measure_breakdown_time_job.sh postgresql node-based none && \
bash ./measure_breakdown_time_job.sh postgresql topdown none && \

# ============================================================
# split=none: query-jit × 3 compile-mode × 2 cache(off/full) = 6
# ============================================================

# --- cache=off ---
bash ./measure_breakdown_time_job.sh postgresql none query none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh postgresql none query none on on on all off off fastisel && \
bash ./measure_breakdown_time_job.sh postgresql none query none on on on all off off tpde && \

# --- cache=full ---
bash ./measure_breakdown_time_job.sh postgresql none query none on on on all full off llvm && \
bash ./measure_breakdown_time_job.sh postgresql none query none on on on all full off fastisel && \
bash ./measure_breakdown_time_job.sh postgresql none query none on on on all full off tpde && \

# ============================================================
# node-based: (3 compile-mode + 1 tune) × 4 cache × 2 spec = 32
# Outer: spec → cache → { compile-mode + tune }
# ============================================================

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

# ---- cache=off, spec=off: 3 compile-mode + 1 tune = 4 ----
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all off off fastisel && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py node-based --engine=postgresql && \

bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all off off llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-strict, spec=off: 4 ----
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-strict off llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-template, spec=off: 4 ----
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-template off llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=full, spec=off: 4 ----
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all full off llvm && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all full off fastisel && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all full off tpde && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all full off llvm job_result/tuned_per_subquery_node-based.json && \

# ------------------------------------------------------------
# spec-jit=recompile
# ------------------------------------------------------------

# ---- cache=off, spec=recompile: 4 ----
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all off recompile llvm && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all off recompile tpde && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all off recompile llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-strict, spec=recompile: 4 ----
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-strict recompile llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-template, spec=recompile: 4 ----
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all single-run-template recompile llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=full, spec=recompile: 4 ----
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all full recompile llvm && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all full recompile tpde && \
bash ./measure_breakdown_time_job.sh postgresql node-based query none on on on all full recompile llvm job_result/tuned_per_subquery_node-based.json && \

# ============================================================
# topdown: mirrors node-based configs
# (3 compile-mode + 1 tune) × 4 cache × 2 spec = 32
# ============================================================

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

# ---- cache=off, spec=off: 3 + 1 tune = 4 ----
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all off off fastisel && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py topdown --engine=postgresql && \

bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all off off llvm job_result/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-strict, spec=off: 4 ----
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-strict off llvm && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-strict off fastisel && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-strict off tpde && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-strict off llvm job_result/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-template, spec=off: 4 ----
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-template off llvm && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-template off fastisel && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-template off tpde && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-template off llvm job_result/tuned_per_subquery_topdown.json && \

# ---- cache=full, spec=off: 4 ----
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all full off llvm && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all full off fastisel && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all full off tpde && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all full off llvm job_result/tuned_per_subquery_topdown.json && \

# ------------------------------------------------------------
# spec-jit=recompile
# ------------------------------------------------------------

# ---- cache=off, spec=recompile: 4 ----
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all off recompile llvm && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all off recompile fastisel && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all off recompile tpde && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all off recompile llvm job_result/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-strict, spec=recompile: 4 ----
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-strict recompile llvm && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-strict recompile tpde && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-strict recompile llvm job_result/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-template, spec=recompile: 4 ----
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-template recompile llvm && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-template recompile fastisel && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-template recompile tpde && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all single-run-template recompile llvm job_result/tuned_per_subquery_topdown.json && \

# ---- cache=full, spec=recompile: 4 ----
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all full recompile llvm && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all full recompile fastisel && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all full recompile tpde && \
bash ./measure_breakdown_time_job.sh postgresql topdown query none on on on all full recompile llvm job_result/tuned_per_subquery_topdown.json && \

echo "=== All PostgreSQL breakdown measurements complete ==="
