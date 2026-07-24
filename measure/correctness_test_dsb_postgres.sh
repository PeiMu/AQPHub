#!/usr/bin/env bash
#
# PostgreSQL DSB query-jit correctness test.
# Tests [none-split, node-based, topdown] x [none-jit, query-jit] against golden files.
#
# Usage:
#   bash correctness_test_dsb_postgres.sh [SF] [--test-only]
#   bash correctness_test_dsb_postgres.sh              # SF=10, generate golden + run all tests
#   bash correctness_test_dsb_postgres.sh 100           # SF=100, generate golden + run all tests
#   bash correctness_test_dsb_postgres.sh 10 --test-only  # SF=10, skip golden generation
#
set -uo pipefail

# ============================================================
# Environment -- source machine-specific paths from env.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSB_SF="${1:-10}"
source "${SCRIPT_DIR}/env.sh"

SCHEMA="${DSB_PATH}/scripts/create_tables.sql"
FKEYS="${DSB_PATH}/scripts/tpcds_ri_umbra.sql"
QUERY_DIR="${DSB_PATH}/code/tools/1_instance_out_aqp/1/"

STORAGE_CACHE_DSB="${STORAGE_CACHE_PG_DSB}"

# Output filter: strip log/timing/debug lines so only query results remain.
FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^NOTICE:|^\[AQP|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|^same engine|^embed data|no version information available"'

cd "${PROJECT}/measure"
if [[ "$DSB_SF" == "10" ]]; then
  RESULT_DIR="dsb_result"
else
  RESULT_DIR="dsb_result_sf${DSB_SF}"
fi
mkdir -p "${RESULT_DIR}"

GOLDEN_NOSPLIT="pg_dsb_no-split_golden.txt"
GOLDEN_NB="pg_dsb_node-based_golden.txt"

# ============================================================
# Helper: run all DSB queries with given flags, write output
# ============================================================
run_pg_dsb() {
  local split="$1"
  local jit_level="$2"
  local output_file="$3"
  local compile_mode="${4:-llvm}"
  local skip_hash_cmp="${5:-all}"
  local simd="${6:-off}"
  local jit_cache="${7:-off}"
  local spec_jit="${8:-off}"
  local tune_config="${9:-}"

  local split_flag="--split=${split}"
  local jit_flag="--jit-level=${jit_level}"
  local storage_flags=""
  local helper_flag=""
  local extra_flags=""

  if [[ "$jit_level" == "query" ]]; then
    storage_flags="--storage-plan --storage-cache=${STORAGE_CACHE_DSB}"
  fi

  if [[ "$split" == "node-based" || "$split" == "topdown" ]]; then
    helper_flag="--helper-db-path=${DUCKDB_DB_DSB}"
  fi

  if [[ "$compile_mode" != "llvm" ]]; then
    extra_flags="--compile-mode=${compile_mode}"
  fi

  if [[ "$skip_hash_cmp" != "all" ]]; then
    extra_flags+=" --jit-skip-hash-cmp=${skip_hash_cmp}"
  fi

  if [[ "$simd" != "off" ]]; then
    extra_flags+=" --jit-simd=${simd}"
  fi

  if [[ "$jit_cache" != "off" ]]; then
    if [[ "$jit_cache" == "on" ]]; then
      extra_flags+=" --jit-cache"
    else
      extra_flags+=" --jit-cache=${jit_cache}"
    fi
  fi
  [[ "$jit_cache" == "full" ]] && extra_flags+=" --repeat=2"

  if [[ "$spec_jit" != "off" ]]; then
    extra_flags+=" --spec-jit=${spec_jit}"
  fi

  if [[ -n "$tune_config" ]]; then
    extra_flags+=" --tune-config=${tune_config}"
  fi

  rm -f "$output_file"

  "${BINARY}" \
    --engine=postgresql \
    --db="${PG_CONN_DSB}" \
    ${helper_flag} \
    --schema="${SCHEMA}" \
    --fkeys="${FKEYS}" \
    ${split_flag} \
    ${jit_flag} \
    --no-analyze \
    ${storage_flags} \
    ${extra_flags} \
    --benchmark \
    "${QUERY_DIR}" \
    2>&1 | tee -a "$output_file"
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
${PG_BIN}/psql -d "${PG_CONN_DSB}" -c "ANALYZE;"
echo "ANALYZE done"

# ============================================================
# Step 1: Generate golden files (interpreter baseline)
# ============================================================
generate_golden=${2:-}
if [[ "$generate_golden" != "--test-only" ]]; then
  echo "========================================"
  echo "Generating golden: no-split, none-jit"
  echo "========================================"
  run_pg_dsb none none "${GOLDEN_NOSPLIT}"
  echo ""

  echo "========================================"
  echo "Generating golden: node-based, none-jit"
  echo "========================================"
  run_pg_dsb node-based none "${GOLDEN_NB}"
  echo ""
fi

# Verify golden files exist
for g in "$GOLDEN_NOSPLIT" "$GOLDEN_NB"; do
  if [[ ! -f "$g" ]]; then
    echo "ERROR: Golden file missing: $g"
    echo "Run without --test-only to generate golden files first."
    exit 1
  fi
done

# ============================================================
# Step 2: Test configs
# ============================================================
# Config format: split|jit_level|golden_file|compile_mode|skip_hash_cmp|simd|jit_cache|spec_jit
# Defaults: llvm, all, off, off, off
#
# PostgreSQL only supports query-jit (no expr/operator/pipeline).
# No payload_prune, prefetch, batch_probe flags.
CONFIGS=(
  # ============================================================
  # Interpreter baseline (should match golden exactly)
  # ============================================================
  "none|none|${GOLDEN_NOSPLIT}"
  "node-based|none|${GOLDEN_NB}"

  # ============================================================
  # Query-jit, compile_mode=llvm (default)
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}"
  "node-based|query|${GOLDEN_NB}"

  # ============================================================
  # Query-jit, compile_mode=fastisel
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|fastisel"
  "node-based|query|${GOLDEN_NB}|fastisel"

  # ============================================================
  # Query-jit, compile_mode=tpde
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|tpde"
  "node-based|query|${GOLDEN_NB}|tpde"

  # ============================================================
  # Query-jit, jit_cache=single-run-strict
  # (none-split omitted: single-run cache only helps across sub-queries)
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|all|off|single-run-strict"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|single-run-strict"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|single-run-strict"

  # ============================================================
  # Query-jit, jit_cache=single-run-template
  # (none-split omitted: single-run cache only helps across sub-queries)
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|all|off|single-run-template"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|single-run-template"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|single-run-template"

  # ============================================================
  # Query-jit, jit_cache=full (with --repeat=2 for cold+warm)
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|llvm|all|off|full"
  "none|query|${GOLDEN_NOSPLIT}|fastisel|all|off|full"
  "none|query|${GOLDEN_NOSPLIT}|tpde|all|off|full"
  "node-based|query|${GOLDEN_NB}|llvm|all|off|full"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|full"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|full"

  # ============================================================
  # Speculative JIT, recompile (TPDE on miss)
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|all|off|off|recompile"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|off|recompile"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|off|recompile"

  # ============================================================
  # Speculative JIT, recompile + jit_cache=single-run-strict
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|all|off|single-run-strict|recompile"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|single-run-strict|recompile"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|single-run-strict|recompile"

  # ============================================================
  # Speculative JIT, recompile + jit_cache=single-run-template
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|all|off|single-run-template|recompile"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|single-run-template|recompile"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|single-run-template|recompile"

  # ============================================================
  # Speculative JIT, recompile + jit_cache=full
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|all|off|full|recompile"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|full|recompile"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|full|recompile"

  # ============================================================
  # topdown (SDS) -- uses the same no-split golden (ground truth)
  # ============================================================

  # Interpreter baseline
  "topdown|none|${GOLDEN_NOSPLIT}"

  # Query-jit (llvm / fastisel / tpde)
  "topdown|query|${GOLDEN_NOSPLIT}"
  "topdown|query|${GOLDEN_NOSPLIT}|fastisel"
  "topdown|query|${GOLDEN_NOSPLIT}|tpde"

  # jit-cache=single-run-strict (query x llvm / fastisel / tpde)
  "topdown|query|${GOLDEN_NOSPLIT}|llvm|all|off|single-run-strict"
  "topdown|query|${GOLDEN_NOSPLIT}|fastisel|all|off|single-run-strict"
  "topdown|query|${GOLDEN_NOSPLIT}|tpde|all|off|single-run-strict"

  # jit-cache=single-run-template (query x llvm / fastisel / tpde)
  "topdown|query|${GOLDEN_NOSPLIT}|llvm|all|off|single-run-template"
  "topdown|query|${GOLDEN_NOSPLIT}|fastisel|all|off|single-run-template"
  "topdown|query|${GOLDEN_NOSPLIT}|tpde|all|off|single-run-template"

  # jit-cache=full (query x llvm / fastisel / tpde)
  "topdown|query|${GOLDEN_NOSPLIT}|llvm|all|off|full"
  "topdown|query|${GOLDEN_NOSPLIT}|fastisel|all|off|full"
  "topdown|query|${GOLDEN_NOSPLIT}|tpde|all|off|full"

  # Speculative JIT, recompile (query x llvm / fastisel / tpde)
  "topdown|query|${GOLDEN_NOSPLIT}|llvm|all|off|off|recompile"
  "topdown|query|${GOLDEN_NOSPLIT}|fastisel|all|off|off|recompile"
  "topdown|query|${GOLDEN_NOSPLIT}|tpde|all|off|off|recompile"

  # Speculative JIT, recompile + jit_cache=single-run-strict
  "topdown|query|${GOLDEN_NOSPLIT}|llvm|all|off|single-run-strict|recompile"
  "topdown|query|${GOLDEN_NOSPLIT}|fastisel|all|off|single-run-strict|recompile"
  "topdown|query|${GOLDEN_NOSPLIT}|tpde|all|off|single-run-strict|recompile"

  # Speculative JIT, recompile + jit_cache=single-run-template
  "topdown|query|${GOLDEN_NOSPLIT}|llvm|all|off|single-run-template|recompile"
  "topdown|query|${GOLDEN_NOSPLIT}|fastisel|all|off|single-run-template|recompile"
  "topdown|query|${GOLDEN_NOSPLIT}|tpde|all|off|single-run-template|recompile"

  # Speculative JIT, recompile + jit_cache=full
  "topdown|query|${GOLDEN_NOSPLIT}|llvm|all|off|full|recompile"
  "topdown|query|${GOLDEN_NOSPLIT}|fastisel|all|off|full|recompile"
  "topdown|query|${GOLDEN_NOSPLIT}|tpde|all|off|full|recompile"
)

passed=0
failed=0
total=${#CONFIGS[@]}
declare -a FAILED_CONFIGS=()
FAIL_LOG="${RESULT_DIR}/correctness_pg_dsb_failures.log"
: > "$FAIL_LOG"

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r split jit_level golden compile_mode skip_hash_cmp simd jit_cache spec_jit <<< "$entry"
  compile_mode=${compile_mode:-llvm}
  skip_hash_cmp=${skip_hash_cmp:-all}
  simd=${simd:-off}
  jit_cache=${jit_cache:-off}
  spec_jit=${spec_jit:-off}
  config_label="split=${split} jit=${jit_level} compile=${compile_mode} shc=${skip_hash_cmp} simd=${simd} cache=${jit_cache} spec=${spec_jit}"
  echo "=== Testing: ${config_label} ==="

  fc_suffix=""
  [[ "$compile_mode" != "llvm" ]] && fc_suffix="_${compile_mode}"
  shc_suffix=""
  [[ "$skip_hash_cmp" != "all" ]] && shc_suffix="_shc${skip_hash_cmp}"
  simd_suffix=""
  [[ "$simd" != "off" ]] && simd_suffix="_simd${simd}"
  cache_suffix=""
  [[ "$jit_cache" != "off" ]] && cache_suffix="_cache_${jit_cache//-/_}"
  spec_suffix=""
  [[ "$spec_jit" != "off" ]] && spec_suffix="_spec_${spec_jit}"
  output="${RESULT_DIR}/pg_${split}_${jit_level}${fc_suffix}${shc_suffix}${simd_suffix}${cache_suffix}${spec_suffix}_dsb.txt"
  run_pg_dsb "${split}" "${jit_level}" "${output}" "${compile_mode}" "${skip_hash_cmp}" "${simd}" "${jit_cache}" "${spec_jit}"

  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("${config_label}  [output missing: $output]")
    echo "--- ${config_label} ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
    echo ""
    continue
  fi
  if [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("${config_label}  [golden missing: $golden]")
    echo "--- ${config_label} ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
    echo ""
    continue
  fi

  # For --jit-cache=full (--repeat=2): compare BOTH iterations against golden.
  # Iter 0 = normal path, iter 1 = replay path. Both must match.
  if [[ "$jit_cache" == "full" ]]; then
    d0=$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    d1=$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0 + iter1)"
      ((passed++))
    else
      echo "  FAIL: differences found"
      [[ -n "$d0" ]] && echo "  iter0 diff:" && echo "$d0" | head -10
      [[ -n "$d1" ]] && echo "  iter1 diff:" && echo "$d1" | head -10
      FAILED_CONFIGS+=("${config_label}")
      echo "--- ${config_label} ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "iter0 diff:" >> "$FAIL_LOG" && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "iter1 diff:" >> "$FAIL_LOG" && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -30
      FAILED_CONFIGS+=("${config_label}")
      echo "--- ${config_label} ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""
done

# --- Per-subquery tune-config correctness for node-based (if JSON exists) ---
TUNE_JSON="${RESULT_DIR}/tuned_per_subquery_node-based.json"
if [[ -f "$TUNE_JSON" ]]; then
  golden="${GOLDEN_NB}"

  # Tune + spec=off, cache=off
  echo "=== Testing: per-subquery tune-config (node-based, spec-jit off) ==="
  ((total++))
  output="${RESULT_DIR}/pg_node-based_query_tuned_dsb.txt"
  run_pg_dsb node-based query "${output}" llvm all off off off "$TUNE_JSON"
  config_label="tune-config node-based spec=off"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=off
  echo "=== Testing: per-subquery tune-config (node-based, cache=strict, spec-jit off) ==="
  ((total++))
  output="${RESULT_DIR}/pg_node-based_query_cache_single_run_strict_tuned_dsb.txt"
  run_pg_dsb node-based query "${output}" llvm all off single-run-strict off "$TUNE_JSON"
  config_label="tune-config node-based cache=strict spec=off"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=off
  echo "=== Testing: per-subquery tune-config (node-based, cache=template, spec-jit off) ==="
  ((total++))
  output="${RESULT_DIR}/pg_node-based_query_cache_single_run_template_tuned_dsb.txt"
  run_pg_dsb node-based query "${output}" llvm all off single-run-template off "$TUNE_JSON"
  config_label="tune-config node-based cache=template spec=off"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=off
  echo "=== Testing: per-subquery tune-config (node-based, cache=full, spec-jit off) ==="
  ((total++))
  output="${RESULT_DIR}/pg_node-based_query_cache_full_tuned_dsb.txt"
  run_pg_dsb node-based query "${output}" llvm all off full off "$TUNE_JSON"
  config_label="tune-config node-based cache=full spec=off"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d0=$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    d1=$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0 + iter1)"
      ((passed++))
    else
      echo "  FAIL: differences found"
      [[ -n "$d0" ]] && echo "  iter0 diff:" && echo "$d0" | head -10
      [[ -n "$d1" ]] && echo "  iter1 diff:" && echo "$d1" | head -10
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "iter0 diff:" >> "$FAIL_LOG" && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "iter1 diff:" >> "$FAIL_LOG" && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + spec-jit=recompile, cache=off
  echo "=== Testing: per-subquery tune-config (node-based, spec-jit=recompile) ==="
  ((total++))
  output="${RESULT_DIR}/pg_node-based_query_spec_recompile_tuned_dsb.txt"
  run_pg_dsb node-based query "${output}" llvm all off off recompile "$TUNE_JSON"
  config_label="tune-config node-based spec=recompile"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=recompile
  echo "=== Testing: per-subquery tune-config (node-based, cache=strict, spec-jit=recompile) ==="
  ((total++))
  output="${RESULT_DIR}/pg_node-based_query_cache_single_run_strict_spec_recompile_tuned_dsb.txt"
  run_pg_dsb node-based query "${output}" llvm all off single-run-strict recompile "$TUNE_JSON"
  config_label="tune-config node-based cache=strict spec=recompile"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=recompile
  echo "=== Testing: per-subquery tune-config (node-based, cache=template, spec-jit=recompile) ==="
  ((total++))
  output="${RESULT_DIR}/pg_node-based_query_cache_single_run_template_spec_recompile_tuned_dsb.txt"
  run_pg_dsb node-based query "${output}" llvm all off single-run-template recompile "$TUNE_JSON"
  config_label="tune-config node-based cache=template spec=recompile"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=recompile
  echo "=== Testing: per-subquery tune-config (node-based, cache=full, spec-jit=recompile) ==="
  ((total++))
  output="${RESULT_DIR}/pg_node-based_query_cache_full_spec_recompile_tuned_dsb.txt"
  run_pg_dsb node-based query "${output}" llvm all off full recompile "$TUNE_JSON"
  config_label="tune-config node-based cache=full spec=recompile"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d0=$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    d1=$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0 + iter1)"
      ((passed++))
    else
      echo "  FAIL: differences found"
      [[ -n "$d0" ]] && echo "  iter0 diff:" && echo "$d0" | head -10
      [[ -n "$d1" ]] && echo "  iter1 diff:" && echo "$d1" | head -10
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "iter0 diff:" >> "$FAIL_LOG" && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "iter1 diff:" >> "$FAIL_LOG" && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""
else
  echo "(skipping node-based tune-config test: $TUNE_JSON not found)"
fi

# --- Per-subquery tune-config correctness for topdown (if JSON exists) ---
TUNE_JSON_TD="${RESULT_DIR}/tuned_per_subquery_topdown.json"
if [[ -f "$TUNE_JSON_TD" ]]; then
  golden="${GOLDEN_NOSPLIT}"

  # Tune + spec=off, cache=off
  echo "=== Testing: per-subquery tune-config (topdown, spec-jit off) ==="
  ((total++))
  output="${RESULT_DIR}/pg_topdown_query_tuned_dsb.txt"
  run_pg_dsb topdown query "${output}" llvm all off off off "$TUNE_JSON_TD"
  config_label="tune-config topdown spec=off"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=off
  echo "=== Testing: per-subquery tune-config (topdown, cache=strict, spec-jit off) ==="
  ((total++))
  output="${RESULT_DIR}/pg_topdown_query_cache_single_run_strict_tuned_dsb.txt"
  run_pg_dsb topdown query "${output}" llvm all off single-run-strict off "$TUNE_JSON_TD"
  config_label="tune-config topdown cache=strict spec=off"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=off
  echo "=== Testing: per-subquery tune-config (topdown, cache=template, spec-jit off) ==="
  ((total++))
  output="${RESULT_DIR}/pg_topdown_query_cache_single_run_template_tuned_dsb.txt"
  run_pg_dsb topdown query "${output}" llvm all off single-run-template off "$TUNE_JSON_TD"
  config_label="tune-config topdown cache=template spec=off"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=off
  echo "=== Testing: per-subquery tune-config (topdown, cache=full, spec-jit off) ==="
  ((total++))
  output="${RESULT_DIR}/pg_topdown_query_cache_full_tuned_dsb.txt"
  run_pg_dsb topdown query "${output}" llvm all off full off "$TUNE_JSON_TD"
  config_label="tune-config topdown cache=full spec=off"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d0=$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    d1=$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0 + iter1)"
      ((passed++))
    else
      echo "  FAIL: differences found"
      [[ -n "$d0" ]] && echo "  iter0 diff:" && echo "$d0" | head -10
      [[ -n "$d1" ]] && echo "  iter1 diff:" && echo "$d1" | head -10
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "iter0 diff:" >> "$FAIL_LOG" && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "iter1 diff:" >> "$FAIL_LOG" && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + spec-jit=recompile, cache=off
  echo "=== Testing: per-subquery tune-config (topdown, spec-jit=recompile) ==="
  ((total++))
  output="${RESULT_DIR}/pg_topdown_query_spec_recompile_tuned_dsb.txt"
  run_pg_dsb topdown query "${output}" llvm all off off recompile "$TUNE_JSON_TD"
  config_label="tune-config topdown spec=recompile"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=recompile
  echo "=== Testing: per-subquery tune-config (topdown, cache=strict, spec-jit=recompile) ==="
  ((total++))
  output="${RESULT_DIR}/pg_topdown_query_cache_single_run_strict_spec_recompile_tuned_dsb.txt"
  run_pg_dsb topdown query "${output}" llvm all off single-run-strict recompile "$TUNE_JSON_TD"
  config_label="tune-config topdown cache=strict spec=recompile"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=recompile
  echo "=== Testing: per-subquery tune-config (topdown, cache=template, spec-jit=recompile) ==="
  ((total++))
  output="${RESULT_DIR}/pg_topdown_query_cache_single_run_template_spec_recompile_tuned_dsb.txt"
  run_pg_dsb topdown query "${output}" llvm all off single-run-template recompile "$TUNE_JSON_TD"
  config_label="tune-config topdown cache=template spec=recompile"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=recompile
  echo "=== Testing: per-subquery tune-config (topdown, cache=full, spec-jit=recompile) ==="
  ((total++))
  output="${RESULT_DIR}/pg_topdown_query_cache_full_spec_recompile_tuned_dsb.txt"
  run_pg_dsb topdown query "${output}" llvm all off full recompile "$TUNE_JSON_TD"
  config_label="tune-config topdown cache=full spec=recompile"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  else
    d0=$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    d1=$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0 + iter1)"
      ((passed++))
    else
      echo "  FAIL: differences found"
      [[ -n "$d0" ]] && echo "  iter0 diff:" && echo "$d0" | head -10
      [[ -n "$d1" ]] && echo "  iter1 diff:" && echo "$d1" | head -10
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "iter0 diff:" >> "$FAIL_LOG" && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "iter1 diff:" >> "$FAIL_LOG" && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""
else
  echo "(skipping topdown tune-config test: $TUNE_JSON_TD not found)"
fi

# ============================================================
# Summary
# ============================================================
echo "==============================="
echo "Results: ${passed}/${total} passed, ${failed} failed"
echo "==============================="
if (( ${#FAILED_CONFIGS[@]} > 0 )); then
  echo ""
  echo "Failed configs:"
  for i in "${!FAILED_CONFIGS[@]}"; do
    echo "  $((i+1)). ${FAILED_CONFIGS[$i]}"
  done
  echo ""
  echo "Full diffs saved to: $FAIL_LOG"
fi
exit $failed
