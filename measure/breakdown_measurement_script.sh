#!/usr/bin/env bash
#
# Full performance sweep: all engine × split × jit-level × flag combos.
# Run from measure/ directory.  Don't run anything else while measuring.
#
# measure_breakdown_time_job.sh arg order:
#   1=engine 2=split 3=jit_level 4=jit_simd
#   5=payload_prune 6=prefetch 7=batch_probe 8=skip_hash_cmp
#   9=jit_cache 10=spec_jit 11=compile_mode 12=tune_config
#   13=hide_latency
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

# ============================================================
# duckdb backend
# ============================================================
#
# ============================================================
# Interpreter baseline (no JIT — spec is a no-op here)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center none && \
bash ./measure_breakdown_time_job.sh duckdb node-based none && \

# ============================================================
# expr-jit (spec-jit default=off)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none expr none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb none expr auto on on on all off off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center expr none on on on all off off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center expr auto on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr auto on on on all off off llvm && \

# ============================================================
# operator-jit (spec-jit default=off)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none operator none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb none operator auto on on on all off off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center operator none on on on all off off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center operator auto on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator auto on on on all off off llvm && \

# ============================================================
# pipeline-jit (spec-jit default=off)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none pipeline none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb none pipeline auto on on on all off off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline none on on on all off off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline auto on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based pipeline auto on on on all off off llvm && \

# ============================================================
# query-jit compile-mode=llvm (spec=off baseline)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none query none on on on all off off llvm && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center query none on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off off llvm && \

# ============================================================
# query-jit compile-mode=fastisel (spec=off baseline)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none query none on on on all off off fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center query none on on on all off off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off off fastisel && \

# ============================================================
# query-jit compile-mode=tpde (spec=off baseline)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none query none on on on all off off tpde && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center query none on on on all off off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off off tpde && \

# ============================================================
# query-jit + spec-jit (node-based only)
# 3 compile-modes × 2 spec modes = 6 combos
# ============================================================
# compile-mode=llvm + spec=recompile
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off recompile llvm && \
# compile-mode=llvm + spec=interpret
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off interpret llvm && \
# compile-mode=fastisel + spec=recompile
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off recompile fastisel && \
# compile-mode=fastisel + spec=interpret
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off interpret fastisel && \
# compile-mode=tpde + spec=recompile
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off recompile tpde && \
# compile-mode=tpde + spec=interpret
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off interpret tpde && \

# ============================================================
# query-jit SIMD (ROF two-phase split, compile-mode=llvm/fastisel only;
# TPDE excluded — ROF disabled for TPDE, falls back to scalar)
# ============================================================
# compile-mode=llvm (spec=off)
bash ./measure_breakdown_time_job.sh duckdb none query auto on on on all off off llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query auto on on on all off off llvm && \
# compile-mode=fastisel (spec=off)
bash ./measure_breakdown_time_job.sh duckdb none query auto on on on all off off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query auto on on on all off off fastisel && \
# spec-jit combos (node-based, llvm + fastisel)
bash ./measure_breakdown_time_job.sh duckdb node-based query auto on on on all off recompile llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query auto on on on all off interpret llvm && \
bash ./measure_breakdown_time_job.sh duckdb node-based query auto on on on all off recompile fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query auto on on on all off interpret fastisel && \

# ============================================================
# tuned config
# ============================================================
python3 tune_per_subquery.py node-based && \
# tuned + spec=off (measure tuned config without speculative JIT)
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off off llvm job_result/tuned_per_subquery_node-based.json && \
# tuned + spec=recompile
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off recompile llvm job_result/tuned_per_subquery_node-based.json && \

# ============================================================
# cross-query latency hiding (node-based only)
# Orthogonal flag: re-measure key configs with --hide-latency-across-queries
# ============================================================
# interpreter + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based none off on on on all off off llvm "" on && \
# expr-jit + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based expr none on on on all off off llvm "" on && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr auto on on on all off off llvm "" on && \
# operator-jit + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based operator none on on on all off off llvm "" on && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator auto on on on all off off llvm "" on && \
# pipeline-jit + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none on on on all off off llvm "" on && \
bash ./measure_breakdown_time_job.sh duckdb node-based pipeline auto on on on all off off llvm "" on && \
# query-jit llvm + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off off llvm "" on && \
# query-jit fastisel + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off off fastisel "" on && \
# query-jit tpde + hide-latency (primary config)
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off off tpde "" on && \
# query-jit llvm + spec=recompile + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off recompile llvm "" on && \
# query-jit llvm + spec=interpret + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off interpret llvm "" on && \
# query-jit fastisel + spec=recompile + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off recompile fastisel "" on && \
# query-jit fastisel + spec=interpret + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off interpret fastisel "" on && \
# query-jit tpde + spec=recompile + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off recompile tpde "" on && \
# query-jit tpde + spec=interpret + hide-latency
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on all off interpret tpde "" on && \

## ============================================================
## pipeline kernel
## ============================================================
#bash ./measure_breakdown_time_job_kernel.sh duckdb none pipeline none on on on all && \
#bash ./measure_breakdown_time_job_kernel.sh duckdb relationship-center pipeline none on on on all && \
#bash ./measure_breakdown_time_job_kernel.sh duckdb node-based pipeline none on on on all && \

echo "=== All breakdown measurements complete ==="
