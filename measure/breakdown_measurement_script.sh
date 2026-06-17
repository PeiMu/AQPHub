#!/usr/bin/env bash
#
# Full performance sweep: all engine × split × jit-level × flag combos.
# Run from measure/ directory.  Don't run anything else while measuring.
#
# measure_breakdown_time_job.sh arg order:
#   1=engine 2=split 3=jit_level 4=jit_simd
#   5=payload_prune 6=prefetch 7=batch_probe 8=skip_hash_cmp
#   9=jit_cache 10=spec_jit 11=compile_mode 12=tune_config
#
set -e

# ============================================================
# lingodb LLVM backend
# ============================================================
bash ./measure_breakdown_time_job.sh lingodb none llvm && \
bash ./measure_breakdown_time_job.sh lingodb node-based llvm && \

# ============================================================
# lingodb TPDE backend
# ============================================================
bash ./measure_breakdown_time_job.sh lingodb none tpde && \
bash ./measure_breakdown_time_job.sh lingodb node-based tpde

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
bash ./measure_breakdown_time_job.sh duckdb none expr none && \
bash ./measure_breakdown_time_job.sh duckdb none expr auto && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center expr none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center expr auto && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr none && \
bash ./measure_breakdown_time_job.sh duckdb node-based expr auto && \

# ============================================================
# operator-jit (spec-jit default=off)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none operator none && \
bash ./measure_breakdown_time_job.sh duckdb none operator auto && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center operator none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center operator auto && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator none && \
bash ./measure_breakdown_time_job.sh duckdb node-based operator auto && \

# ============================================================
# pipeline-jit (spec-jit default=off)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none pipeline none && \
bash ./measure_breakdown_time_job.sh duckdb none pipeline auto && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center pipeline auto && \
bash ./measure_breakdown_time_job.sh duckdb node-based pipeline none && \
bash ./measure_breakdown_time_job.sh duckdb node-based pipeline auto && \

# ============================================================
# query-jit compile-mode=llvm (spec=off baseline)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none query none && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center query none && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none && \

# ============================================================
# query-jit compile-mode=fastisel (spec=off baseline)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none query none on on on on off off fastisel && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center query none on on on on off off fastisel && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off off fastisel && \

# ============================================================
# query-jit compile-mode=tpde (spec=off baseline)
# ============================================================
bash ./measure_breakdown_time_job.sh duckdb none query none on on on on off off tpde && \
#bash ./measure_breakdown_time_job.sh duckdb relationship-center query none on on on on off off tpde && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off off tpde && \

# ============================================================
# query-jit + spec-jit (node-based only)
# 3 compile-modes × 2 spec modes = 6 combos
# ============================================================
# compile-mode=llvm + spec=recompile
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off recompile && \
# compile-mode=llvm + spec=interpret
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off interpret && \
# compile-mode=fastisel + spec=recompile
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off recompile fastisel && \
# compile-mode=fastisel + spec=interpret
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off interpret fastisel && \
# compile-mode=tpde + spec=recompile
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off recompile tpde && \
# compile-mode=tpde + spec=interpret
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off interpret tpde && \

# ============================================================
# tuned config
# ============================================================
# tuned + spec=off (measure tuned config without speculative JIT)
python3 tune_per_subquery.py node-based && \
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off off off job_result/tuned_per_subquery_node-based.json && \
# tuned + spec=recompile
bash ./measure_breakdown_time_job.sh duckdb node-based query none on on on on off recompile off job_result/tuned_per_subquery_node-based.json && \

# ============================================================
# pipeline kernel (none-split only)
# ============================================================
#bash ./measure_breakdown_time_job_kernel.sh duckdb none pipeline none && \
#bash ./measure_breakdown_time_job_kernel.sh duckdb relationship-center pipeline none && \
#bash ./measure_breakdown_time_job_kernel.sh duckdb node-based pipeline none && \

echo "=== All breakdown measurements complete ==="
