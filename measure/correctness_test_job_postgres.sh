#!/usr/bin/env bash
#
# PostgreSQL query-jit correctness test.
# Tests [none-split, node-based, topdown] x [none-jit, query-jit] against golden files.
#
# Usage:
#   bash correctness_test_job_postgres.sh              # generate golden + run all tests
#   bash correctness_test_job_postgres.sh --test-only  # skip golden generation, run tests only
#
# Config format:
#   engine|split|jit_level|jit_simd|golden[|spec_jit[|jit_cache[|compile_mode[|skip_hash_cmp]]]]
#   compile_mode defaults to llvm, skip_hash_cmp defaults to on(=all) when omitted.
#
set -uo pipefail

FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^NOTICE:|^\[AQP|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|^same engine|^embed data|no version information available"'

GOLDEN_NOSPLIT="pg_job_no-split_golden.txt"
GOLDEN_NB="pg_job_node-based_golden.txt"

# ============================================================
# Step 1: Generate golden files (interpreter baseline)
# ============================================================
generate_golden=${1:-}
if [[ "$generate_golden" != "--test-only" ]]; then
  echo "========================================"
  echo "Generating golden: no-split, none-jit"
  echo "========================================"
  bash run_aqp.sh job postgresql none none none
  cp job_result/aqp_middleware_postgresql_none_none_none_job.txt "${GOLDEN_NOSPLIT}"
  echo ""

  echo "========================================"
#  echo "Generating golden: node-based, none-jit"
  echo "========================================"
#  bash run_aqp.sh job postgresql node-based none none
#  cp job_result/aqp_middleware_postgresql_node-based_none_none_job.txt "${GOLDEN_NB}"
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
JIT_CONFIGS=(
  # ============================================================
  # Interpreter baseline (should match golden exactly)
  # ============================================================
  "postgresql|none|none|none|${GOLDEN_NOSPLIT}"
#  "postgresql|node-based|none|none|${GOLDEN_NB}"

  # ============================================================
  # Query-jit, compile_mode=llvm (default)
  # ============================================================
  "postgresql|none|query|none|${GOLDEN_NOSPLIT}"
#  "postgresql|node-based|query|none|${GOLDEN_NB}"

  # ============================================================
  # Query-jit, compile_mode=fastisel
  # ============================================================
#  "postgresql|none|query|none|${GOLDEN_NOSPLIT}|||fastisel"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|||fastisel"

  # ============================================================
  # Query-jit, compile_mode=tpde
  # ============================================================
  "postgresql|none|query|none|${GOLDEN_NOSPLIT}|||tpde"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|||tpde"

  # ============================================================
  # jit-cache=single-run-strict
  # ============================================================
#  "postgresql|node-based|query|none|${GOLDEN_NB}|off|single-run-strict"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|off|single-run-strict|fastisel"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|off|single-run-strict|tpde"

  # ============================================================
  # jit-cache=single-run-template
  # ============================================================
#  "postgresql|node-based|query|none|${GOLDEN_NB}|off|single-run-template"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|off|single-run-template|fastisel"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|off|single-run-template|tpde"

  # ============================================================
  # jit-cache=full (with --repeat=2 for cold+warm)
  # ============================================================
  "postgresql|none|query|none|${GOLDEN_NOSPLIT}|off|full"
#  "postgresql|none|query|none|${GOLDEN_NOSPLIT}|off|full|fastisel"
#  "postgresql|none|query|none|${GOLDEN_NOSPLIT}|off|full|tpde"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|off|full"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|off|full|fastisel"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|off|full|tpde"

  # ============================================================
  # Speculative JIT, recompile
  # ============================================================
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile||fastisel"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile||tpde"

  # ============================================================
  # Speculative JIT, recompile + jit_cache=single-run-strict
  # ============================================================
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile|single-run-strict"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile|single-run-strict|fastisel"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile|single-run-strict|tpde"

  # ============================================================
  # Speculative JIT, recompile + jit_cache=single-run-template
  # ============================================================
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile|single-run-template"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile|single-run-template|fastisel"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile|single-run-template|tpde"

  # ============================================================
  # Speculative JIT, recompile + jit_cache=full
  # ============================================================
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile|full"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile|full|fastisel"
#  "postgresql|node-based|query|none|${GOLDEN_NB}|recompile|full|tpde"

  # ============================================================
  # topdown (SDS) -- uses the same no-split golden (ground truth)
  # ============================================================

  # Interpreter baseline
  "postgresql|topdown|none|none|${GOLDEN_NOSPLIT}"

  # Query-jit (llvm / fastisel / tpde)
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}"
#  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|||fastisel"
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|||tpde"

  # jit-cache=single-run-strict (query x llvm / fastisel / tpde)
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|off|single-run-strict"
#  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|off|single-run-strict|fastisel"
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|off|single-run-strict|tpde"

  # jit-cache=single-run-template (query x llvm / fastisel / tpde)
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|off|single-run-template"
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|off|single-run-structural"
#  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|off|single-run-template|fastisel"
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|off|single-run-template|tpde"
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|off|single-run-structural|tpde"

  # jit-cache=full (query x llvm / fastisel / tpde)
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|off|full"
#  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|off|full|fastisel"
#  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|off|full|tpde"

  # Speculative JIT, recompile (query x llvm / fastisel / tpde)
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile"
#  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile||fastisel"
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile||tpde"

  # Speculative JIT, recompile + jit_cache=single-run-strict
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile|single-run-strict"
#  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile|single-run-strict|fastisel"
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile|single-run-strict|tpde"

  # Speculative JIT, recompile + jit_cache=single-run-template
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile|single-run-template"
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile|single-run-structural"
#  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile|single-run-template|fastisel"
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile|single-run-template|tpde"
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile|single-run-structural|tpde"

  # Speculative JIT, recompile + jit_cache=full
  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile|full"
#  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile|full|fastisel"
#  "postgresql|topdown|query|none|${GOLDEN_NOSPLIT}|recompile|full|tpde"
)

passed=0
failed=0
total=${#JIT_CONFIGS[@]}

# Clear disk JIT cache to ensure clean slate
rm -rf /dev/shm/aqp_jit_cache/
declare -a FAILED_CONFIGS=()
FAIL_LOG="job_result/correctness_pg_failures.log"
mkdir -p job_result
: > "$FAIL_LOG"

# --- Run JIT-level configs via run_aqp.sh job ---
for entry in "${JIT_CONFIGS[@]}"; do
  IFS='|' read -r engine split jit_level jit_simd golden spec_jit_mode jit_cache_mode compile_mode skip_hash_cmp <<< "$entry"
  spec_jit_mode=${spec_jit_mode:-off}
  jit_cache_mode=${jit_cache_mode:-off}
  compile_mode=${compile_mode:-llvm}
  skip_hash_cmp=${skip_hash_cmp:-on}
  echo "=== Testing: engine=${engine} split=${split} jit=${jit_level} simd=${jit_simd} compile=${compile_mode} spec=${spec_jit_mode} cache=${jit_cache_mode} skip_hash_cmp=${skip_hash_cmp} ==="

  bash run_aqp.sh job "${engine}" "${split}" "${jit_level}" "${jit_simd}" \
       on on on "${skip_hash_cmp}" "${jit_cache_mode}" "${spec_jit_mode}" "${compile_mode}"

  shc_suffix=""
  [[ "$skip_hash_cmp" == "off" ]] && shc_suffix="_noskiphashcmp"
  spec_suffix=""
  [[ "$spec_jit_mode" != "off" ]] && spec_suffix="_spec${spec_jit_mode}"
  cache_suffix=""
  if [[ "$jit_cache_mode" == "on" ]]; then
    cache_suffix="_jitcache"
  elif [[ "$jit_cache_mode" != "off" ]]; then
    cache_suffix="_jitcache_${jit_cache_mode//-/_}"
  fi
  fc_suffix=""
  [[ "$compile_mode" != "llvm" ]] && fc_suffix="_${compile_mode}"
  output="job_result/aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${shc_suffix}${cache_suffix}${spec_suffix}${fc_suffix}_job.txt"

  config_label="engine=${engine} split=${split} jit=${jit_level} simd=${jit_simd} compile=${compile_mode} spec=${spec_jit_mode} cache=${jit_cache_mode} skip_hash_cmp=${skip_hash_cmp}"

  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
    continue
  fi
  if [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
    continue
  fi

  if [[ "$jit_cache_mode" == "full" ]]; then
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
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -30
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""
done

# --- Per-subquery tune-config correctness for node-based (if JSON exists) ---
TUNE_JSON="job_result/tuned_cross_split_postgresql.json"
if [[ -f "$TUNE_JSON" ]]; then
  golden="${GOLDEN_NB}"

  # Tune + spec=off, cache=off
  echo "=== Testing: per-subquery tune-config (auto, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on off off llvm "$TUNE_JSON"
  config_label="tune-config auto spec=off"
  output="job_result/aqp_middleware_postgresql_auto_query_none_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on single-run-strict off llvm "$TUNE_JSON"
  config_label="tune-config auto cache=strict spec=off"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_single_run_strict_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on single-run-template off llvm "$TUNE_JSON"
       on on on on single-run-structural off llvm "$TUNE_JSON"
  config_label="tune-config auto cache=template spec=off"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_single_run_template_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
#       on on on on full off llvm "$TUNE_JSON"
  config_label="tune-config auto cache=full spec=off"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_full_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on off recompile llvm "$TUNE_JSON"
  config_label="tune-config auto spec=recompile"
  output="job_result/aqp_middleware_postgresql_auto_query_none_specrecompile_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on single-run-strict recompile llvm "$TUNE_JSON"
  config_label="tune-config auto cache=strict spec=recompile"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_single_run_strict_specrecompile_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on single-run-template recompile llvm "$TUNE_JSON"
       on on on on single-run-structural recompile llvm "$TUNE_JSON"
  config_label="tune-config auto cache=template spec=recompile"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_single_run_template_specrecompile_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
#       on on on on full recompile llvm "$TUNE_JSON"
  config_label="tune-config auto cache=full spec=recompile"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_full_specrecompile_tuned_job.txt"
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
TUNE_JSON_TD="job_result/tuned_cross_split_postgresql.json"
if [[ -f "$TUNE_JSON_TD" ]]; then
  golden="${GOLDEN_NOSPLIT}"

  # Tune + spec=off, cache=off
  echo "=== Testing: per-subquery tune-config (auto, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on off off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto spec=off"
  output="job_result/aqp_middleware_postgresql_auto_query_none_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on single-run-strict off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=strict spec=off"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_single_run_strict_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on single-run-template off llvm "$TUNE_JSON_TD"
       on on on on single-run-structural off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=template spec=off"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_single_run_template_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
#       on on on on full off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=full spec=off"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_full_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on off recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto spec=recompile"
  output="job_result/aqp_middleware_postgresql_auto_query_none_specrecompile_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on single-run-strict recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=strict spec=recompile"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_single_run_strict_specrecompile_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
       on on on on single-run-template recompile llvm "$TUNE_JSON_TD"
       on on on on single-run-structural recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=template spec=recompile"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_single_run_template_specrecompile_tuned_job.txt"
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
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job postgresql auto query none \
#       on on on on full recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=full spec=recompile"
  output="job_result/aqp_middleware_postgresql_auto_query_none_jitcache_full_specrecompile_tuned_job.txt"
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
