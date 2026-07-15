#!/usr/bin/env bash
#
# PostgreSQL query-jit correctness test.
# Tests [none-split, node-based, topdown] x [none-jit, query-jit] against golden files.
#
# Uses run_job.sh as the per-config runner (same as DuckDB correctness_test.sh).
# Arg order for run_job.sh:
#   1=engine 2=split 3=jit_level 4=jit_simd
#   5=payload_prune 6=prefetch 7=batch_probe 8=skip_hash_cmp
#   9=jit_cache 10=spec_jit 11=compile_mode 12=tune_config
#
# Usage:
#   bash correctness_test_pg.sh              # generate golden + run all tests
#   bash correctness_test_pg.sh --test-only  # skip golden generation, run tests only
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${SCRIPT_DIR}"
mkdir -p job_result

# Output filter: strip log/timing/debug lines so only query results remain.
FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^NOTICE:|^\[AQP|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|^same engine|^embed data|no version information available"'

GOLDEN_NOSPLIT="pg_job_no-split_golden.txt"
GOLDEN_NB="pg_job_node-based_golden.txt"
GOLDEN_TD="pg_job_topdown_golden.txt"

# ============================================================
# Step 1: Generate golden files (interpreter baseline)
# ============================================================
generate_golden=${1:-}
if [[ "$generate_golden" != "--test-only" ]]; then
  echo "========================================"
  echo "Generating golden: no-split, none-jit"
  echo "========================================"
  bash ./run_job.sh postgresql none none off
  mv job_result/aqp_middleware_postgresql_none_none_off_job.txt "${GOLDEN_NOSPLIT}"
  echo ""

  echo "========================================"
  echo "Generating golden: node-based, none-jit"
  echo "========================================"
  bash ./run_job.sh postgresql node-based none off
  mv job_result/aqp_middleware_postgresql_node-based_none_off_job.txt "${GOLDEN_NB}"
  echo ""

  echo "========================================"
  echo "Generating golden: topdown, none-jit"
  echo "========================================"
  bash ./run_job.sh postgresql topdown none off
  mv job_result/aqp_middleware_postgresql_topdown_none_off_job.txt "${GOLDEN_TD}"
  echo ""
fi

# Verify golden files exist
for g in "$GOLDEN_NOSPLIT" "$GOLDEN_NB" "$GOLDEN_TD"; do
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
CONFIGS=(
  # ============================================================
  # Interpreter baseline (should match golden exactly)
  # ============================================================
  "none|none|${GOLDEN_NOSPLIT}"
  "node-based|none|${GOLDEN_NB}"
  "topdown|none|${GOLDEN_TD}"

  # ============================================================
  # Query-jit, compile_mode=llvm (default), skip_hash_cmp=all (default)
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}"
  "node-based|query|${GOLDEN_NB}"
  "topdown|query|${GOLDEN_TD}"

  # ============================================================
  # Query-jit, compile_mode=fastisel
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|fastisel"
  "node-based|query|${GOLDEN_NB}|fastisel"
  "topdown|query|${GOLDEN_TD}|fastisel"

  # ============================================================
  # Query-jit, compile_mode=tpde
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|tpde"
  "node-based|query|${GOLDEN_NB}|tpde"
  "topdown|query|${GOLDEN_TD}|tpde"

  # ============================================================
  # Query-jit, skip_hash_cmp=off
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|llvm|off"
  "node-based|query|${GOLDEN_NB}|llvm|off"
  "topdown|query|${GOLDEN_TD}|llvm|off"
  "node-based|query|${GOLDEN_NB}|fastisel|off"
  "node-based|query|${GOLDEN_NB}|tpde|off"
  "topdown|query|${GOLDEN_TD}|fastisel|off"
  "topdown|query|${GOLDEN_TD}|tpde|off"

  # ============================================================
  # Query-jit, skip_hash_cmp=off + spec-jit=recompile
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|off|off|off|recompile"

  # ============================================================
  # Query-jit, skip_hash_cmp=off + cache tiers
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|off|off|single-run-strict"
  "node-based|query|${GOLDEN_NB}|llvm|off|off|single-run-template"
  "node-based|query|${GOLDEN_NB}|llvm|off|off|full"
  "topdown|query|${GOLDEN_TD}|llvm|off|off|single-run-strict"
  "topdown|query|${GOLDEN_TD}|llvm|off|off|single-run-template"
  "topdown|query|${GOLDEN_TD}|llvm|off|off|full"

  # ============================================================
  # Query-jit, simd=auto
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|llvm|all|auto"
  "node-based|query|${GOLDEN_NB}|llvm|all|auto"
  "topdown|query|${GOLDEN_TD}|llvm|all|auto"

  # ============================================================
  # Query-jit, jit_cache=single-run-strict
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|llvm|all|off|single-run-strict"
  "node-based|query|${GOLDEN_NB}|llvm|all|off|single-run-strict"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|single-run-strict"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|single-run-strict"
  "topdown|query|${GOLDEN_TD}|llvm|all|off|single-run-strict"
  "topdown|query|${GOLDEN_TD}|fastisel|all|off|single-run-strict"
  "topdown|query|${GOLDEN_TD}|tpde|all|off|single-run-strict"

  # ============================================================
  # Query-jit, jit_cache=single-run-template
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|llvm|all|off|single-run-template"
  "node-based|query|${GOLDEN_NB}|llvm|all|off|single-run-template"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|single-run-template"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|single-run-template"
  "topdown|query|${GOLDEN_TD}|llvm|all|off|single-run-template"
  "topdown|query|${GOLDEN_TD}|fastisel|all|off|single-run-template"
  "topdown|query|${GOLDEN_TD}|tpde|all|off|single-run-template"

  # ============================================================
  # Query-jit, jit_cache=full (with --repeat=2 for cold+warm)
  # ============================================================
  "none|query|${GOLDEN_NOSPLIT}|llvm|all|off|full"
  "none|query|${GOLDEN_NOSPLIT}|fastisel|all|off|full"
  "none|query|${GOLDEN_NOSPLIT}|tpde|all|off|full"
  "node-based|query|${GOLDEN_NB}|llvm|all|off|full"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|full"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|full"
  "topdown|query|${GOLDEN_TD}|llvm|all|off|full"
  "topdown|query|${GOLDEN_TD}|fastisel|all|off|full"
  "topdown|query|${GOLDEN_TD}|tpde|all|off|full"

  # ============================================================
  # Speculative JIT, recompile (TPDE on miss)
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|all|off|off|recompile"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|off|recompile"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|off|recompile"
  "topdown|query|${GOLDEN_TD}|llvm|all|off|off|recompile"
  "topdown|query|${GOLDEN_TD}|fastisel|all|off|off|recompile"
  "topdown|query|${GOLDEN_TD}|tpde|all|off|off|recompile"

  # ============================================================
  # Speculative JIT, recompile + jit_cache=single-run-strict
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|all|off|single-run-strict|recompile"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|single-run-strict|recompile"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|single-run-strict|recompile"
  "topdown|query|${GOLDEN_TD}|llvm|all|off|single-run-strict|recompile"
  "topdown|query|${GOLDEN_TD}|fastisel|all|off|single-run-strict|recompile"
  "topdown|query|${GOLDEN_TD}|tpde|all|off|single-run-strict|recompile"

  # ============================================================
  # Speculative JIT, recompile + jit_cache=single-run-template
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|all|off|single-run-template|recompile"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|single-run-template|recompile"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|single-run-template|recompile"
  "topdown|query|${GOLDEN_TD}|llvm|all|off|single-run-template|recompile"
  "topdown|query|${GOLDEN_TD}|fastisel|all|off|single-run-template|recompile"
  "topdown|query|${GOLDEN_TD}|tpde|all|off|single-run-template|recompile"

  # ============================================================
  # Speculative JIT, recompile + jit_cache=full
  # ============================================================
  "node-based|query|${GOLDEN_NB}|llvm|all|off|full|recompile"
  "node-based|query|${GOLDEN_NB}|fastisel|all|off|full|recompile"
  "node-based|query|${GOLDEN_NB}|tpde|all|off|full|recompile"
  "topdown|query|${GOLDEN_TD}|llvm|all|off|full|recompile"
  "topdown|query|${GOLDEN_TD}|fastisel|all|off|full|recompile"
  "topdown|query|${GOLDEN_TD}|tpde|all|off|full|recompile"
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

  # Map simd value: off -> off for jit_simd arg, auto/sse2/etc -> that value
  jit_simd="off"
  if [[ "$jit_level" != "none" ]]; then
    if [[ "$simd" == "off" ]]; then
      jit_simd="none"
    else
      jit_simd="${simd}"
    fi
  fi

  # Run via shared run_job.sh
  bash ./run_job.sh postgresql "${split}" "${jit_level}" "${jit_simd}" \
    on on on "${skip_hash_cmp}" "${jit_cache}" "${spec_jit}" "${compile_mode}"

  # Determine output filename (matches run_job.sh naming)
  flag_suffix=""
  [[ "$skip_hash_cmp"  == "off" ]] && flag_suffix+="_noskiphashcmp"
  if [[ "$jit_cache" == "on" ]]; then
    flag_suffix+="_jitcache"
  elif [[ "$jit_cache" != "off" ]]; then
    flag_suffix+="_jitcache_${jit_cache//-/_}"
  fi
  [[ "$spec_jit"       != "off" ]] && flag_suffix+="_spec${spec_jit}"
  [[ "$compile_mode" != "off" && "$compile_mode" != "llvm" ]] && flag_suffix+="_fc${compile_mode}"
  output="job_result/aqp_middleware_postgresql_${split}_${jit_level}_${jit_simd}${flag_suffix}_job.txt"

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
# Step 3: Per-subquery tune-config correctness
# ============================================================
TUNE_NB="job_result/tuned_per_subquery_node-based.json"
TUNE_TD="job_result/tuned_per_subquery_topdown.json"

for tune_entry in \
  "node-based|${GOLDEN_NB}|${TUNE_NB}" \
  "topdown|${GOLDEN_TD}|${TUNE_TD}"; do
  IFS='|' read -r tsplit tgolden tjson <<< "$tune_entry"
  if [[ ! -f "$tjson" ]]; then
    echo "(skipping ${tsplit} tune-config tests: $tjson not found)"
    continue
  fi
  for cache_spec in "off|off" "single-run-strict|off" "single-run-template|off" "full|off" \
                    "off|recompile" "single-run-strict|recompile" "single-run-template|recompile" "full|recompile"; do
    IFS='|' read -r tcache tspec <<< "$cache_spec"
    ((total++))
    config_label="tune split=${tsplit} cache=${tcache} spec=${tspec}"
    echo "=== Testing: ${config_label} ==="

    bash ./run_job.sh postgresql "${tsplit}" query none \
      on on on all "${tcache}" "${tspec}" llvm "${tjson}"

    # Build output filename to match run_job.sh naming
    tune_flag=""
    if [[ "$tcache" == "on" ]]; then
      tune_flag+="_jitcache"
    elif [[ "$tcache" != "off" ]]; then
      tune_flag+="_jitcache_${tcache//-/_}"
    fi
    [[ "$tspec" != "off" ]] && tune_flag+="_spec${tspec}"
    tune_flag+="_tuned"
    output="job_result/aqp_middleware_postgresql_${tsplit}_query_none${tune_flag}_job.txt"

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

    if [[ "$tcache" == "full" ]]; then
      d0=$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$tgolden") || true)
      d1=$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$tgolden") || true)
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
      d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$tgolden") || true)
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
