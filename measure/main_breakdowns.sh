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

# ------------------------------------------------------------

#RQ1 & RQ2
# DuckDB JOB
bash ./measure_breakdown_time_aqp.sh job duckdb none none &&\
bash ./measure_breakdown_time_aqp.sh job duckdb none query none on on on all off off tpde &&\
bash ./measure_breakdown_time_aqp.sh job duckdb topdown none &&\
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all single-run-structural off tpde &&\

## DuckDB DSB 10
#bash ./measure_breakdown_time_aqp.sh dsb duckdb none none &&\
#bash ./measure_breakdown_time_aqp.sh dsb duckdb none query none on on on all off off tpde &&\
#bash ./measure_breakdown_time_aqp.sh dsb duckdb node-based none &&\
#bash measure_breakdown_time_aqp.sh dsb duckdb node-based query none on on on all single-run-template off tpde &&\

# PostgreSQL JOB
bash ./measure_breakdown_time_aqp.sh job postgresql none none &&\
bash ./measure_breakdown_time_aqp.sh job postgresql none query none on on on all off off tpde &&\
bash ./measure_breakdown_time_aqp.sh job postgresql node-based none &&\
bash ./measure_breakdown_time_aqp.sh job postgresql node-based query none on on on all single-run-structural recompile fastisel &&\

## PostgreSQL DSB 10
##bash ./measure_breakdown_time_aqp.sh dsb postgresql none none &&\
##bash ./measure_breakdown_time_aqp.sh dsb postgresql none query none on on on all off off tpde &&\
#bash ./measure_breakdown_time_aqp.sh dsb postgresql node-based none &&\
#bash ./measure_breakdown_time_aqp.sh dsb postgresql node-based query none on on on all single-run-template recompile tpde &&\

# DuckDB DSB 50
bash ./measure_breakdown_time_aqp.sh dsb_50 duckdb none none &&\
bash ./measure_breakdown_time_aqp.sh dsb_50 duckdb none query none on on on all off off tpde &&\
bash ./measure_breakdown_time_aqp.sh dsb_50 duckdb node-based none &&\
bash ./measure_breakdown_time_aqp.sh dsb_50 duckdb node-based query none on on on all single-run-structural off tpde &&\

# PostgreSQL DSB 50
bash ./measure_breakdown_time_aqp.sh dsb_50 postgresql none none &&\
bash ./measure_breakdown_time_aqp.sh dsb_50 postgresql none query none on on on all off off tpde &&\
bash ./measure_breakdown_time_aqp.sh dsb_50 postgresql node-based none &&\
bash ./measure_breakdown_time_aqp.sh dsb_50 postgresql node-based query none on on on all single-run-structural recompile tpde &&\

## DuckDB DSB 100
#bash ./measure_breakdown_time_aqp.sh dsb_100 duckdb none none &&\
#bash ./measure_breakdown_time_aqp.sh dsb_100 duckdb none query none on on on all off off tpde &&\
#bash ./measure_breakdown_time_aqp.sh dsb_100 duckdb node-based none &&\
#bash measure_breakdown_time_aqp.sh dsb_100 duckdb node-based query none on on on all single-run-template off tpde 

# PostgreSQL DSB 100
#bash ./measure_breakdown_time_aqp.sh dsb postgresql none none &&\
#bash ./measure_breakdown_time_aqp.sh dsb postgresql none query none on on on all off off tpde &&\
#bash ./measure_breakdown_time_aqp.sh dsb_100 postgresql node-based none &&\
#bash ./measure_breakdown_time_aqp.sh dsb_100 postgresql node-based query none on on on all single-run-template recompile tpde

#RQ3, TODO: measure pg; measure w/wo stats
bash ./breakdown_eval_storage.sh &&\
bash ./breakdown_runtime_guided_opt.sh &&\

#RQ4, TODO: need to check and fix
bash ./breakdown_compile_time_reduction.sh &&\

#RQ5
bash ./breakdown_eval_storage.sh
