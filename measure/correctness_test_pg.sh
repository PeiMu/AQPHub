#!/usr/bin/env bash
#
# PostgreSQL query-jit correctness test.
# Tests [none-split, node-based] x [none-jit, query-jit] against golden files.
#
# Usage:
#   bash correctness_test_pg.sh              # generate golden + run all tests
#   bash correctness_test_pg.sh --test-only  # skip golden generation, run tests only
#
set -uo pipefail

# ============================================================
# Environment — source machine-specific paths from env.sh
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK="${JOB_PATH}"
BINARY="${PROJECT}/build_release/aqp_middleware"
STORAGE_CACHE="/tmp/imdb_storage_plan_pg.cache"

SCHEMA="${BENCHMARK}/schema.sql"
FKEYS="${BENCHMARK}/fkeys.sql"
QUERY_DIR="${BENCHMARK}/queries"

# Output filter: strip log/timing/debug lines so only query results remain.
FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^NOTICE:|^\[AQP|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|^same engine|^embed data|no version information available"'

cd "${PROJECT}/measure"
mkdir -p job_result

GOLDEN_NOSPLIT="pg_job_no-split_golden.txt"
GOLDEN_NB="pg_job_node-based_golden.txt"

# ============================================================
# Helper: run all JOB queries with given flags, write output
# ============================================================
run_pg_job() {
  local split="$1"
  local jit_level="$2"
  local output_file="$3"
  local compile_mode="${4:-llvm}"
  local skip_hash_cmp="${5:-all}"
  local simd="${6:-off}"
  local jit_cache="${7:-off}"
  local spec_jit="${8:-off}"

  local split_flag="--split=${split}"
  local jit_flag="--jit-level=${jit_level}"
  local storage_flags=""
  local helper_flag=""
  local extra_flags=""

  if [[ "$jit_level" == "query" ]]; then
    storage_flags="--storage-plan --storage-cache=${STORAGE_CACHE}"
  fi

  if [[ "$split" == "node-based" ]]; then
    helper_flag="--helper-db-path=${DUCKDB_DB}"
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

  rm -f "$output_file"

  "${BINARY}" \
    --engine=postgresql \
    --db="${PG_CONN}" \
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
    2>&1 | tee "$output_file"
}

# ============================================================
# Step 1: Generate golden files (interpreter baseline)
# ============================================================
generate_golden=${1:-}
if [[ "$generate_golden" != "--test-only" ]]; then
  echo "========================================"
  echo "Generating golden: no-split, none-jit"
  echo "========================================"
  run_pg_job none none "${GOLDEN_NOSPLIT}"
  echo ""

  echo "========================================"
  echo "Generating golden: node-based, none-jit"
  echo "========================================"
  run_pg_job node-based none "${GOLDEN_NB}"
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
# Flags relevant to query-jit (CompileQuerySteps in ir_to_llvm.cpp):
#   compile_mode  - llvm/fastisel/tpde: LLVM backend selection.
#   skip_hash_cmp - off/single/all: skip hash comparison for integer keys.
#   simd          - off/sse2/avx/avx2/avx512/auto: SIMD vectorization.
#   jit_cache     - off/single-run-strict/single-run-template/full: LLVM object cache.
#   spec_jit      - off/recompile/interpret: speculative JIT compilation.
#                   recompile = TPDE on miss, interpret = skip JIT on miss.
#                   Only meaningful with node-based split (needs PeekNextSubquery).
#
# Flags NOT relevant (expr/operator-jit only, not used by CompileQuerySteps):
#   payload_prune, prefetch, batch_probe
CONFIGS=(
  # ============================================================
  # Interpreter baseline (should match golden exactly)
  # ============================================================
  "none|none|${GOLDEN_NOSPLIT}"
  "node-based|none|${GOLDEN_NB}"

  # ============================================================
  # Query-jit, compile_mode=llvm (default), skip_hash_cmp=all (default)
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
  # Query-jit, skip_hash_cmp=off
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|llvm|off"
  "node-based|query|${GOLDEN_NB}|llvm|off"

  # ============================================================
  # Query-jit, skip_hash_cmp=single
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|llvm|single"
  "node-based|query|${GOLDEN_NB}|llvm|single"

  # ============================================================
  # Query-jit, simd=auto
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|llvm|all|auto"
  "node-based|query|${GOLDEN_NB}|llvm|all|auto"

  # ============================================================
  # Query-jit, jit_cache=single-run-strict
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|llvm|all|off|single-run-strict"
  "node-based|query|${GOLDEN_NB}|llvm|all|off|single-run-strict"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|single-run-strict"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|single-run-strict"

  # ============================================================
  # Query-jit, jit_cache=single-run-template
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|llvm|all|off|single-run-template"
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
)

passed=0
failed=0
total=${#CONFIGS[@]}
declare -a FAILED_CONFIGS=()
FAIL_LOG="job_result/correctness_pg_failures.log"
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
  [[ "$compile_mode" != "llvm" ]] && fc_suffix="_fc${compile_mode}"
  shc_suffix=""
  [[ "$skip_hash_cmp" != "all" ]] && shc_suffix="_shc${skip_hash_cmp}"
  simd_suffix=""
  [[ "$simd" != "off" ]] && simd_suffix="_simd${simd}"
  cache_suffix=""
  [[ "$jit_cache" != "off" ]] && cache_suffix="_cache_${jit_cache//-/_}"
  spec_suffix=""
  [[ "$spec_jit" != "off" ]] && spec_suffix="_spec_${spec_jit}"
  output="job_result/pg_${split}_${jit_level}${fc_suffix}${shc_suffix}${simd_suffix}${cache_suffix}${spec_suffix}_job.txt"
  run_pg_job "${split}" "${jit_level}" "${output}" "${compile_mode}" "${skip_hash_cmp}" "${simd}" "${jit_cache}" "${spec_jit}"

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
