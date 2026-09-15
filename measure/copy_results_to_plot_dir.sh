#!/usr/bin/env bash
#
# Copy measurement CSV results to the plotting script's expected directories.
# Run from measure/ directory after main_breakdowns.sh completes.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_DIR="/home/pei/Document/Evaluate-Query-Split-Method-Experiment-Analysis-Benchmark-/scripts"

JOB_RESULT="${SCRIPT_DIR}/job_result"
DSB_RESULT="${SCRIPT_DIR}/dsb_result"
DSB50_RESULT="${SCRIPT_DIR}/dsb_result_sf50"

########################################
# RQ1 & RQ2: Main breakdown CSVs
########################################
for bm_result in "$JOB_RESULT" "$DSB50_RESULT"; do
    if [[ "$bm_result" == "$JOB_RESULT" ]]; then
        dk_dest="${PLOT_DIR}/JOB_duckdb_152"
        pg_dest="${PLOT_DIR}/JOB_postgres_183"
    elif [[ "$bm_result" == "$DSB50_RESULT" ]]; then
        dk_dest="${PLOT_DIR}/DSB_50_duckdb_152"
        pg_dest="${PLOT_DIR}/DSB_50_postgres_183"
    fi
    mkdir -p "$dk_dest" "$pg_dest"

    for f in "${bm_result}"/duckdb_*_breakdown_time_log.csv; do
        [[ -f "$f" ]] && cp "$f" "$dk_dest/"
    done
    for f in "${bm_result}"/postgresql_*_breakdown_time_log.csv; do
        [[ -f "$f" ]] && cp "$f" "$pg_dest/"
    done
done

# Umbra CSVs
for f in "${JOB_RESULT}"/umbra_official.csv; do
    [[ -f "$f" ]] && cp "$f" "${PLOT_DIR}/umbra_job_official.csv"
done
for f in "${DSB50_RESULT}"/umbra_official.csv; do
    [[ -f "$f" ]] && cp "$f" "${PLOT_DIR}/umbra_dsb_official.csv"
done

########################################
# RQ3.1: Runtime statistics evaluation (q4)
########################################
STATS_DEST="${PLOT_DIR}/evaluate_stats"
mkdir -p "$STATS_DEST"
for f in eval_stats_nojit_nostats.csv eval_stats_nojit_stats.csv \
         eval_stats_queryjit_nostats.csv eval_stats_queryjit_stats.csv; do
    [[ -f "${JOB_RESULT}/$f" ]] && cp "${JOB_RESULT}/$f" "${STATS_DEST}/$f"
done

########################################
# RQ3.2: Runtime-guided optimization waterfall (q3)
########################################
RGO_DEST="${PLOT_DIR}/runtime_guided_opt_breakdown"
mkdir -p "$RGO_DEST"
for f in step1_baseline.csv step2_range_pred.csv step3_range_guard.csv \
         step4_block_skip.csv step5_all_enabled.csv \
         step1_interprete_baseline.csv step2-4_interprete_collect-stats.csv; do
    [[ -f "${JOB_RESULT}/$f" ]] && cp "${JOB_RESULT}/$f" "${RGO_DEST}/$f"
done

########################################
# RQ4: Compile-time reduction (q5, q6, q7)
########################################
CTR_DEST="${PLOT_DIR}/compile_time_reduction_breakdown"
mkdir -p "$CTR_DEST"
for f in figA1_llvm.csv figA2_fastisel.csv figA3_tpde.csv \
         figB1_cache_off_tpde.csv figB2_cache_strict_tpde.csv \
         figB3_cache_parameterized_tpde.csv figB4_cache_template_tpde.csv \
         figB5_cache_off_fastisel.csv figB6_cache_strict_fastisel.csv \
         figB7_cache_parameterized_fastisel.csv figB8_cache_template_fastisel.csv \
         figB9_cache_off_llvm.csv figB10_cache_strict_llvm.csv \
         figB11_cache_parameterized_llvm.csv figB12_cache_template_llvm.csv \
         figC1_no_hiding_tpde.csv figC2_cross_only_tpde.csv figC3_spec_and_cross_tpde.csv \
         figC4_no_hiding_llvm.csv figC5_cross_only_llvm.csv figC6_spec_and_cross_llvm.csv; do
    [[ -f "${JOB_RESULT}/$f" ]] && cp "${JOB_RESULT}/$f" "${CTR_DEST}/$f"
done

########################################
# RQ5: Bi-directional storage (q8)
########################################
STOR_DEST="${PLOT_DIR}/evaluate_storage"
mkdir -p "$STOR_DEST"
for f in storage_step1_bidir_enabled.csv storage_step2_bidir_disabled.csv; do
    [[ -f "${JOB_RESULT}/$f" ]] && cp "${JOB_RESULT}/$f" "${STOR_DEST}/$f"
done

########################################
# Engine optimizer evaluation (q9)
########################################
OPT_DEST="${PLOT_DIR}/evaluate_engine_optimizer"
mkdir -p "$OPT_DEST"
for f in eval_optimizer_baseline.csv eval_optimizer_disabled.csv; do
    [[ -f "${JOB_RESULT}/$f" ]] && cp "${JOB_RESULT}/$f" "${OPT_DEST}/$f"
done

echo "=== CSV results copied to ${PLOT_DIR} ==="
