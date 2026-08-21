#!/usr/bin/env bash
#
# skip_hash_cmp=off sweep for query-jit + node-based, and retuning.
# Run from measure/ directory.  Don't run anything else while measuring.
# Requires JOB_PATH in the environment (set in ~/.bashrc).
#
# Motivation: skip_hash_cmp=off is reproducibly faster than =all on some
# queries under query-jit (e.g. 6a/6c/6e/26b, up to ~+24% for =all), while
# =all wins elsewhere (2d/29b/24b, up to ~-18%).  Suite total is a wash, so
# skip_hash_cmp becomes a per-subquery tuning knob.  The flag has no effect
# under expr/operator jit (gated on pipeline kernel-path / AQP_JIT_PIPELINE_JIT),
# so only query-jit configs need off-variants.
#
# Formula:
#   3 compile-mode × 4 cache × 2 spec (query-jit, skip_hash_cmp=off) = 24
#   + 8 tuned reruns (one per cache × spec tier, with regenerated JSON) = 32 runs
#
# measure_breakdown_time_aqp.sh arg order: job <engine> ...
#   1=bench 2=engine 3=split 4=jit_level 5=jit_simd
#   6=payload_prune 7=prefetch 8=batch_probe 9=skip_hash_cmp
#   10=jit_cache 11=spec_jit 12=compile_mode 13=tune_config
#
set -e

# ============================================================
# Step 1: cache=off, spec=off, skip_hash_cmp=off (3 configs)
# These CSVs feed tune_per_subquery.py (the tuner only uses
# cache=off / spec=off runs, where exe time is self-contained).
# ============================================================
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off off off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off off off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off off off tpde && \

# ============================================================
# Step 2: regenerate the tune JSON.
# tune_per_subquery.py now includes the *_noskiphashcmp_* CSVs as
# candidates (flags: skip_hash_cmp="off"), so the JSON can pick
# skip_hash_cmp=off per subquery.
# ============================================================
python3 tune_per_subquery.py && \

# ============================================================
# Step 3: rerun all 8 tuned configs with the new JSON
# (mirrors the tuned runs in breakdown_measurement_script.sh; the
# 8th arg stays "all" — the JSON overrides it per subquery).
# ============================================================
# spec=off
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all off off llvm job_result/tuned_cross_split_duckdb.json && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-strict off llvm job_result/tuned_cross_split_duckdb.json && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-template off llvm job_result/tuned_cross_split_duckdb.json && \
#bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all full off llvm job_result/tuned_cross_split_duckdb.json && \
# spec=recompile
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all off recompile llvm job_result/tuned_cross_split_duckdb.json && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-strict recompile llvm job_result/tuned_cross_split_duckdb.json && \
bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all single-run-template recompile llvm job_result/tuned_cross_split_duckdb.json && \
#bash ./measure_breakdown_time_aqp.sh job duckdb auto query none on on on all full recompile llvm job_result/tuned_cross_split_duckdb.json && \

# ============================================================
# Step 4: skip_hash_cmp=off variants for the remaining tiers
# (reporting only; not used by the tuner).
# ============================================================
# ---- cache=single-run-strict, spec=off (3) ----
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-strict off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-strict off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-strict off tpde && \

# ---- cache=single-run-template, spec=off (3) ----
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-template off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-template off fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-template off tpde && \

# ---- cache=full, spec=off (3) ----
# (not in the original request list, but lines 151-153 of
#  breakdown_measurement_script.sh measure this tier with =all;
#  remove if not wanted)
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off full off llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off full off fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off full off tpde && \

# ---- cache=off, spec=recompile (3) ----
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off off recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off off recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off off recompile tpde && \

# ---- cache=single-run-strict, spec=recompile (3) ----
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-strict recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-strict recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-strict recompile tpde && \

# ---- cache=single-run-template, spec=recompile (3) ----
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-template recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-template recompile fastisel && \
bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off single-run-template recompile tpde && \

# ---- cache=full, spec=recompile (3) ----
# (not in the original request list, mirrors lines 215-217 of
#  breakdown_measurement_script.sh; remove if not wanted)
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off full recompile llvm && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off full recompile fastisel && \
#bash ./measure_breakdown_time_aqp.sh job duckdb node-based query none on on on off full recompile tpde && \

echo "shc_off sweep done"
