#!/usr/bin/env bash
#
# PostgreSQL full performance sweep: 40 configs.
# Run from measure/ directory.  Don't run anything else while measuring.
#
# Formula:
#   2 interpreter                                            [none, node-based]
#   + 1 none-split × 3 compile-mode × 2 cache(off/full)   = 6
#   + 1 node-based × (3 compile-mode + 1 tune) × 4 cache
#                  × 2 spec(off/recompile)                 = 32
#   = 40 configs
#
# Uses measure_breakdown_time_job_pg.sh as the per-config runner.
# Arg order (same as measure_breakdown_time_job.sh):
#   1=engine 2=split 3=jit_level 4=jit_simd
#   5=payload_prune 6=prefetch 7=batch_probe 8=skip_hash_cmp
#   9=jit_cache 10=spec_jit 11=compile_mode 12=tune_config
#
set -euo pipefail

# ============================================================
# Environment — source machine-specific paths from env.sh
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${PROJECT}/build_release/aqp_middleware"
HELPER_DB="${DUCKDB_DB}"
SCHEMA="${JOB_PATH}/schema.sql"
FKEYS="${JOB_PATH}/fkeys.sql"
QUERY_DIR="${JOB_PATH}/queries"
STORAGE_CACHE="/tmp/imdb_storage_plan_pg.cache"

ITERATION=15  # 5 warm-up, 10 measured

cd "${PROJECT}/measure"
mkdir -p job_result

# ============================================================
# Helper: run one breakdown config
# ============================================================
run_pg_breakdown() {
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
    storage_flags="--storage-plan --storage-cache=${STORAGE_CACHE}"
  fi

  local helper_flag=""
  if [[ "$split" == "node-based" ]]; then
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

  echo ">>> postgresql ${split} ${jit_level} ${jit_simd}${flag_suffix}"

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
    "job_result/postgresql_${split}_${jit_level}_${jit_simd}${flag_suffix}_breakdown_time_log.csv"
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
# Interpreter baseline (2 configs)
# ============================================================
run_pg_breakdown none none && \
run_pg_breakdown node-based none && \

# ============================================================
# split=none: query-jit × 3 compile-mode × 2 cache(off/full) = 6
# ============================================================

# --- cache=off ---
run_pg_breakdown none query none all off off llvm && \
run_pg_breakdown none query none all off off fastisel && \
run_pg_breakdown none query none all off off tpde && \

# --- cache=full ---
run_pg_breakdown none query none all full off llvm && \
run_pg_breakdown none query none all full off fastisel && \
run_pg_breakdown none query none all full off tpde && \

# ============================================================
# node-based: (3 compile-mode + 1 tune) × 4 cache × 2 spec = 32
# Outer: spec → cache → { compile-mode + tune }
# ============================================================

# ------------------------------------------------------------
# spec-jit=off
# ------------------------------------------------------------

# ---- cache=off, spec=off: 3 compile-mode + 1 tune = 4 ----
run_pg_breakdown node-based query none all off off llvm && \
run_pg_breakdown node-based query none all off off fastisel && \
run_pg_breakdown node-based query none all off off tpde && \

# Generate tune JSON from cache=off CSVs produced above
python3 tune_per_subquery.py node-based --engine=postgresql && \


run_pg_breakdown node-based query none all off off llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-strict, spec=off: 4 ----
run_pg_breakdown node-based query none all single-run-strict off llvm && \
run_pg_breakdown node-based query none all single-run-strict off fastisel && \
run_pg_breakdown node-based query none all single-run-strict off tpde && \
run_pg_breakdown node-based query none all single-run-strict off llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-template, spec=off: 4 ----
run_pg_breakdown node-based query none all single-run-template off llvm && \
run_pg_breakdown node-based query none all single-run-template off fastisel && \
run_pg_breakdown node-based query none all single-run-template off tpde && \
run_pg_breakdown node-based query none all single-run-template off llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=full, spec=off: 4 ----
run_pg_breakdown node-based query none all full off llvm && \
run_pg_breakdown node-based query none all full off fastisel && \
run_pg_breakdown node-based query none all full off tpde && \
run_pg_breakdown node-based query none all full off llvm job_result/tuned_per_subquery_node-based.json && \

# ------------------------------------------------------------
# spec-jit=recompile
# ------------------------------------------------------------

# ---- cache=off, spec=recompile: 4 ----
run_pg_breakdown node-based query none all off recompile llvm && \
run_pg_breakdown node-based query none all off recompile fastisel && \
run_pg_breakdown node-based query none all off recompile tpde && \
run_pg_breakdown node-based query none all off recompile llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-strict, spec=recompile: 4 ----
run_pg_breakdown node-based query none all single-run-strict recompile llvm && \
run_pg_breakdown node-based query none all single-run-strict recompile fastisel && \
run_pg_breakdown node-based query none all single-run-strict recompile tpde && \
run_pg_breakdown node-based query none all single-run-strict recompile llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=single-run-template, spec=recompile: 4 ----
run_pg_breakdown node-based query none all single-run-template recompile llvm && \
run_pg_breakdown node-based query none all single-run-template recompile fastisel && \
run_pg_breakdown node-based query none all single-run-template recompile tpde && \
run_pg_breakdown node-based query none all single-run-template recompile llvm job_result/tuned_per_subquery_node-based.json && \

# ---- cache=full, spec=recompile: 4 ----
run_pg_breakdown node-based query none all full recompile llvm && \
run_pg_breakdown node-based query none all full recompile fastisel && \
run_pg_breakdown node-based query none all full recompile tpde && \
run_pg_breakdown node-based query none all full recompile llvm job_result/tuned_per_subquery_node-based.json && \

echo "=== All 40 PostgreSQL breakdown measurements complete ==="
