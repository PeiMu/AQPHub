#!/usr/bin/env bash
#
# PostgreSQL DSB full performance sweep: 73 configs.
# Run from measure/ directory.  Don't run anything else while measuring.
#
# Formula:
#   3 interpreter                                            [none, node-based, topdown]
#   + 1 none-split x 3 compile-mode x 2 cache(off/full)   = 6
#   + 1 node-based x (3 compile-mode + 1 tune) x 4 cache
#                  x 2 spec(off/recompile)                 = 32
#   + 1 topdown   x (3 compile-mode + 1 tune) x 4 cache
#                  x 2 spec(off/recompile)                 = 32
#   = 73 configs
#
# Mirrors breakdown_measurement_script_job_postgres.sh for DSB queries.
# DSB_SF env var selects scale factor (default 10).
#
set -e

# ============================================================
# Environment -- source machine-specific paths from env.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

SF="${DSB_SF:-${1:-10}}"

HELPER_DB="${DSB_DUCKDB_DB}"
SCHEMA="${DSB_PATH}/scripts/create_tables.sql"
FKEYS="${DSB_PATH}/scripts/tpcds_ri_umbra.sql"
QUERY_DIR="${DSB_PATH}/code/tools/1_instance_out_aqp/1/"

STORAGE_CACHE_DSB="/tmp/dsb_sf${SF}_storage_plan_pg.cache"

ITERATION=15  # 5 warm-up, 10 measured

cd "${PROJECT}/measure"

RESULT_DIR="dsb_result"
if [[ "$SF" != "10" ]]; then
  RESULT_DIR="dsb_result_sf${SF}"
fi
mkdir -p "${RESULT_DIR}"

# ============================================================
# Helper: run one DSB breakdown config
# ============================================================
run_pg_dsb_breakdown() {
  local split="$1"
  local jit_level="${2:-none}"
  local jit_simd="${3:-off}"
  local skip_hash_cmp="${4:-all}"
  local jit_cache="${5:-off}"
  local spec_jit="${6:-off}"
  local compile_mode="${7:-llvm}"
  local tune_config="${8:-}"

  local jit_extra_flags=""
  if [[ "$jit_cache" == "on" ]]; then
    jit_extra_flags+=" --jit-cache"
  elif [[ "$jit_cache" != "off" ]]; then
    jit_extra_flags+=" --jit-cache=${jit_cache}"
  fi
  [[ "$skip_hash_cmp" != "off" ]] && jit_extra_flags+=" --jit-skip-hash-cmp=${skip_hash_cmp}"
  [[ "$spec_jit" != "off" ]] && jit_extra_flags+=" --spec-jit=${spec_jit}"
  [[ "$compile_mode" != "off" && "$compile_mode" != "llvm" ]] && jit_extra_flags+=" --compile-mode=${compile_mode}"
  [[ -n "$tune_config" ]] && jit_extra_flags+=" --tune-config=${tune_config}"

  local storage_flags=""
  if [[ "$jit_level" == "query" ]]; then
    storage_flags="--storage-plan --storage-cache=${STORAGE_CACHE_DSB}"
  fi

  local helper_flag=""
  if [[ "$split" == "node-based" || "$split" == "topdown" ]]; then
    helper_flag="--helper-db-path=${HELPER_DB}"
  fi

  # Build filename suffix
  local flag_suffix=""
  if [[ "$jit_cache" == "on" ]]; then
    flag_suffix+="_jitcache"
  elif [[ "$jit_cache" != "off" ]]; then
    flag_suffix+="_jitcache_${jit_cache//-/_}"
  fi
  [[ "$spec_jit" != "off" ]] && flag_suffix+="_spec${spec_jit}"
  [[ "$compile_mode" != "off" && "$compile_mode" != "llvm" ]] && flag_suffix+="_fc${compile_mode}"
  [[ -n "$tune_config" ]] && flag_suffix+="_tuned"

  local log_name="time_log.csv"
  rm -f "${log_name}"

  # Clear disk JIT cache for clean cold-start
  if [[ "$jit_cache" == "full" ]]; then
    rm -rf /dev/shm/aqp_jit_cache/
  fi

  echo ">>> postgresql-dsb ${split} ${jit_level} ${jit_simd}${flag_suffix}"

  "${BINARY}" \
    --engine=postgresql \
    --db="${PG_CONN}" \
    ${helper_flag} \
    --schema="${SCHEMA}" \
    --fkeys="${FKEYS}" \
    --split="${split}" \
    --timing \
    --no-analyze \
    --repeat=${ITERATION} \
    --jit-level="${jit_level}" --jit-simd="${jit_simd}" \
    ${jit_extra_flags} \
    ${storage_flags} \
    --benchmark \
    "${QUERY_DIR}"

  mv "${log_name}" \
    "${RESULT_DIR}/postgresql_${split}_${jit_level}_${jit_simd}${flag_suffix}_breakdown_time_log.csv"
}

# ============================================================
# PostgreSQL start/stop
# ============================================================
pg_start() {
  if ! ${PG_BIN}/pg_isready -h localhost >/dev/null 2>&1; then
    echo "Starting PostgreSQL..."
    ${PG_BIN}/pg_ctl start -l "${PG_LOG}" -D "${PG_DATA}"
    until ${PG_BIN}/pg_isready -h localhost >/dev/null 2>&1; do
      sleep 0.5
    done
    echo "PostgreSQL is ready."
  else
    echo "PostgreSQL already running."
  fi
}

pg_stop() {
  ${PG_BIN}/pg_ctl stop -D "${PG_DATA}" -m smart -s 2>/dev/null || true
}

cleanup() { pg_stop; }
trap cleanup EXIT

# ============================================================
# Start PG + ANALYZE
# ============================================================
pg_start

echo "ANALYZING..."
${PG_BIN}/psql -d "${PG_CONN}" -c "ANALYZE;"
echo "ANALYZE done"

# ============================================================
# Interpreter baseline (3 configs)
# ============================================================
run_pg_dsb_breakdown none none && \
run_pg_dsb_breakdown node-based none && \

# ============================================================
# split=none: query-jit x 3 compile-mode x 2 cache(off/full) = 6
# ============================================================

# --- cache=off ---
run_pg_dsb_breakdown none query none all off off llvm && \
run_pg_dsb_breakdown none query none all off off fastisel && \
run_pg_dsb_breakdown none query none all off off tpde && \

# --- cache=full ---
run_pg_dsb_breakdown none query none all full off llvm && \
run_pg_dsb_breakdown none query none all full off fastisel && \
run_pg_dsb_breakdown none query none all full off tpde && \

# ============================================================
# node-based: (3 compile-mode + 1 tune) x 4 cache x 2 spec = 32
# Outer: spec -> cache -> { compile-mode + tune }
# ============================================================

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

# ---- cache=off, spec=off: 3 compile-mode + 1 tune = 4 ----
run_pg_dsb_breakdown node-based query none all off off llvm && \
run_pg_dsb_breakdown node-based query none all off off fastisel && \
run_pg_dsb_breakdown node-based query none all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py --bench=dsb node-based --engine=postgresql --result-dir=${RESULT_DIR} && \

run_pg_dsb_breakdown node-based query none all off off llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-strict, spec=off: 4 ----
run_pg_dsb_breakdown node-based query none all single-run-strict off llvm && \
run_pg_dsb_breakdown node-based query none all single-run-strict off fastisel && \
run_pg_dsb_breakdown node-based query none all single-run-strict off tpde && \
run_pg_dsb_breakdown node-based query none all single-run-strict off llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-template, spec=off: 4 ----
run_pg_dsb_breakdown node-based query none all single-run-template off llvm && \
run_pg_dsb_breakdown node-based query none all single-run-template off fastisel && \
run_pg_dsb_breakdown node-based query none all single-run-template off tpde && \
run_pg_dsb_breakdown node-based query none all single-run-template off llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=full, spec=off: 4 ----
run_pg_dsb_breakdown node-based query none all full off llvm && \
run_pg_dsb_breakdown node-based query none all full off fastisel && \
run_pg_dsb_breakdown node-based query none all full off tpde && \
run_pg_dsb_breakdown node-based query none all full off llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ------------------------------------------------------------
# spec-jit=recompile
# ------------------------------------------------------------

# ---- cache=off, spec=recompile: 4 ----
run_pg_dsb_breakdown node-based query none all off recompile llvm && \
run_pg_dsb_breakdown node-based query none all off recompile fastisel && \
run_pg_dsb_breakdown node-based query none all off recompile tpde && \
run_pg_dsb_breakdown node-based query none all off recompile llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-strict, spec=recompile: 4 ----
run_pg_dsb_breakdown node-based query none all single-run-strict recompile llvm && \
run_pg_dsb_breakdown node-based query none all single-run-strict recompile fastisel && \
run_pg_dsb_breakdown node-based query none all single-run-strict recompile tpde && \
run_pg_dsb_breakdown node-based query none all single-run-strict recompile llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-template, spec=recompile: 4 ----
run_pg_dsb_breakdown node-based query none all single-run-template recompile llvm && \
run_pg_dsb_breakdown node-based query none all single-run-template recompile fastisel && \
run_pg_dsb_breakdown node-based query none all single-run-template recompile tpde && \
run_pg_dsb_breakdown node-based query none all single-run-template recompile llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ---- cache=full, spec=recompile: 4 ----
run_pg_dsb_breakdown node-based query none all full recompile llvm && \
run_pg_dsb_breakdown node-based query none all full recompile fastisel && \
run_pg_dsb_breakdown node-based query none all full recompile tpde && \
run_pg_dsb_breakdown node-based query none all full recompile llvm ${RESULT_DIR}/tuned_per_subquery_node-based.json && \

# ============================================================
# topdown: mirrors node-based configs (query-jit only)
# 1 interpreter + (3 compile-mode + 1 tune) x 4 cache x 2 spec = 33
# ============================================================

# ---- interpreter baseline ----
run_pg_dsb_breakdown topdown none && \

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

# ---- cache=off, spec=off: 3 compile-mode + 1 tune = 4 ----
run_pg_dsb_breakdown topdown query none all off off llvm && \
run_pg_dsb_breakdown topdown query none all off off fastisel && \
run_pg_dsb_breakdown topdown query none all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py --bench=dsb topdown --engine=postgresql --result-dir=${RESULT_DIR} && \

run_pg_dsb_breakdown topdown query none all off off llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-strict, spec=off: 4 ----
run_pg_dsb_breakdown topdown query none all single-run-strict off llvm && \
run_pg_dsb_breakdown topdown query none all single-run-strict off fastisel && \
run_pg_dsb_breakdown topdown query none all single-run-strict off tpde && \
run_pg_dsb_breakdown topdown query none all single-run-strict off llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-template, spec=off: 4 ----
run_pg_dsb_breakdown topdown query none all single-run-template off llvm && \
run_pg_dsb_breakdown topdown query none all single-run-template off fastisel && \
run_pg_dsb_breakdown topdown query none all single-run-template off tpde && \
run_pg_dsb_breakdown topdown query none all single-run-template off llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=full, spec=off: 4 ----
run_pg_dsb_breakdown topdown query none all full off llvm && \
run_pg_dsb_breakdown topdown query none all full off fastisel && \
run_pg_dsb_breakdown topdown query none all full off tpde && \
run_pg_dsb_breakdown topdown query none all full off llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ------------------------------------------------------------
# spec-jit=recompile
# ------------------------------------------------------------

# ---- cache=off, spec=recompile: 4 ----
run_pg_dsb_breakdown topdown query none all off recompile llvm && \
run_pg_dsb_breakdown topdown query none all off recompile fastisel && \
run_pg_dsb_breakdown topdown query none all off recompile tpde && \
run_pg_dsb_breakdown topdown query none all off recompile llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-strict, spec=recompile: 4 ----
run_pg_dsb_breakdown topdown query none all single-run-strict recompile llvm && \
run_pg_dsb_breakdown topdown query none all single-run-strict recompile fastisel && \
run_pg_dsb_breakdown topdown query none all single-run-strict recompile tpde && \
run_pg_dsb_breakdown topdown query none all single-run-strict recompile llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=single-run-template, spec=recompile: 4 ----
run_pg_dsb_breakdown topdown query none all single-run-template recompile llvm && \
run_pg_dsb_breakdown topdown query none all single-run-template recompile fastisel && \
run_pg_dsb_breakdown topdown query none all single-run-template recompile tpde && \
run_pg_dsb_breakdown topdown query none all single-run-template recompile llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

# ---- cache=full, spec=recompile: 4 ----
run_pg_dsb_breakdown topdown query none all full recompile llvm && \
run_pg_dsb_breakdown topdown query none all full recompile fastisel && \
run_pg_dsb_breakdown topdown query none all full recompile tpde && \
run_pg_dsb_breakdown topdown query none all full recompile llvm ${RESULT_DIR}/tuned_per_subquery_topdown.json && \

echo "=== All 73 PostgreSQL DSB breakdown measurements complete ==="
