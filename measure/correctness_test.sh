#!/usr/bin/env bash
#
# Correctness check: run configs across splits, JIT levels, and kernel paths,
# then diff against golden files.
# Usage: bash correctness_test.sh
#
set -uo pipefail

FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^duckdb runs:|^lingodb runs:|\(base\)|^\[AQP|^\[LingoDB|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average"'

# --- JIT-level configs (no kernel path):
#     engine | split | jit_level | jit_simd | golden_file [| spec_jit]
#     spec_jit (optional): recompile/interpret -> --spec-jit=<v> via run_job.sh arg 10 ---
JIT_CONFIGS=(
  "duckdb|none|none|none|duckdb_job_no-split_golden.txt"
  "duckdb|node-based|none|none|duckdb_job_node-based_golden.txt"
  "duckdb|relationship-center|none|none|duckdb_job_relationship-center_golden.txt"
  "duckdb|none|expr|none|duckdb_job_no-split_golden.txt"
  "duckdb|none|expr|auto|duckdb_job_no-split_golden.txt"
  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt"
  "duckdb|node-based|expr|auto|duckdb_job_node-based_golden.txt"
  "duckdb|relationship-center|expr|none|duckdb_job_relationship-center_golden.txt"
  "duckdb|relationship-center|expr|auto|duckdb_job_relationship-center_golden.txt"
  "duckdb|none|operator|none|duckdb_job_no-split_golden.txt"
  "duckdb|none|operator|auto|duckdb_job_no-split_golden.txt"
  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt"
  "duckdb|node-based|operator|auto|duckdb_job_node-based_golden.txt"
  "duckdb|relationship-center|operator|none|duckdb_job_relationship-center_golden.txt"
  "duckdb|relationship-center|operator|auto|duckdb_job_relationship-center_golden.txt"
  "duckdb|none|pipeline|none|duckdb_job_no-split_golden.txt"
  "duckdb|none|pipeline|auto|duckdb_job_no-split_golden.txt"
  "duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt"
  "duckdb|node-based|pipeline|auto|duckdb_job_node-based_golden.txt"
  "duckdb|relationship-center|pipeline|none|duckdb_job_relationship-center_golden.txt"
  "duckdb|relationship-center|pipeline|auto|duckdb_job_relationship-center_golden.txt"
  "duckdb|none|query|none|duckdb_job_no-split_golden.txt"
  "duckdb|none|query|auto|duckdb_job_no-split_golden.txt"
  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt"
  "duckdb|node-based|query|auto|duckdb_job_node-based_golden.txt"
  "duckdb|relationship-center|query|none|duckdb_job_relationship-center_golden.txt"
  "duckdb|relationship-center|query|auto|duckdb_job_relationship-center_golden.txt"
  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile"
  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|interpret"
  "duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt|recompile"
  "duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt|interpret"
  "duckdb|relationship-center|query|none|duckdb_job_relationship-center_golden.txt|recompile"
  "duckdb|none|query|none|duckdb_job_no-split_golden.txt|recompile"
  "lingodb|none|llvm|none|lingodb_job_no-split_golden.txt"
  "lingodb|node-based|llvm|none|duckdb_job_node-based_golden.txt"
  "lingodb|relationship-center|llvm|none|lingodb_job_no-split_golden.txt"
  "lingodb|none|tpde|none|lingodb_job_no-split_golden.txt"
  "lingodb|node-based|tpde|none|duckdb_job_node-based_golden.txt"
  "lingodb|relationship-center|tpde|none|lingodb_job_no-split_golden.txt"
)

# --- Kernel-path configs: engine | split | kernel_path | jit_simd | golden_file ---
KERNEL_CONFIGS=(
  "duckdb|none|pipeline|none|duckdb_job_no-split_golden.txt"
  "duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt"
  "duckdb|relationship-center|pipeline|none|duckdb_job_relationship-center_golden.txt"
)

passed=0
failed=0
total=$(( ${#JIT_CONFIGS[@]} + ${#KERNEL_CONFIGS[@]} ))
declare -a FAILED_CONFIGS=()
FAIL_LOG="job_result/correctness_failures.log"
: > "$FAIL_LOG"

# --- Run JIT-level configs via run_job.sh ---
for entry in "${JIT_CONFIGS[@]}"; do
  IFS='|' read -r engine split jit_level jit_simd golden spec_jit_mode <<< "$entry"
  spec_jit_mode=${spec_jit_mode:-off}
  echo "=== Testing: engine=${engine} split=${split} jit=${jit_level} spec=${spec_jit_mode} ==="

  bash run_job.sh "${engine}" "${split}" "${jit_level}" "${jit_simd}" \
       on on on on off "${spec_jit_mode}"

  spec_suffix=""
  [[ "$spec_jit_mode" != "off" ]] && spec_suffix="_spec${spec_jit_mode}"
  if [[ "$engine" == "lingodb" ]]; then
    output="job_result/aqp_middleware_${engine}_${jit_level}_${split}_job.txt"
  else
    output="job_result/aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${spec_suffix}_job.txt"
  fi

  config_label="engine=${engine} split=${split} jit=${jit_level} simd=${jit_simd} spec=${spec_jit_mode}"

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

  d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden"))
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
  echo ""
done

# --- Run kernel-path configs via run_job_kernel.sh ---
for entry in "${KERNEL_CONFIGS[@]}"; do
  IFS='|' read -r engine split kernel_path jit_simd golden <<< "$entry"
  echo "=== Testing: engine=${engine} split=${split} kernel-path=${kernel_path} ==="

  bash run_job_kernel.sh "${engine}" "${split}" "${kernel_path}" "${jit_simd}" \
       on on on on on off

  output="job_result/aqp_middleware_${engine}_${split}_kernel-${kernel_path}_${jit_simd}_job.txt"
  config_label="engine=${engine} split=${split} kernel-path=${kernel_path} simd=${jit_simd}"

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

  d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden"))
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
  echo ""
done

# --- Per-subquery tune-config correctness (if JSON exists) ---
TUNE_JSON="job_result/tuned_per_subquery_node-based.json"
if [[ -f "$TUNE_JSON" ]]; then
  echo "=== Testing: per-subquery tune-config (node-based, spec-jit off) ==="
  ((total++))
  golden="duckdb_job_node-based_golden.txt"

  bash run_job.sh duckdb node-based query none \
       on on on on off off llvm "$TUNE_JSON"

  config_label="tune-config node-based spec=off"
  output="job_result/aqp_middleware_duckdb_node-based_query_none_tuned_job.txt"
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
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden"))
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

  # Tune + spec-jit=recompile
  echo "=== Testing: per-subquery tune-config (node-based, spec-jit=recompile) ==="
  ((total++))
  config_label="tune-config node-based spec=recompile"

  bash run_job.sh duckdb node-based query none \
       on on on on off recompile llvm "$TUNE_JSON"

  output="job_result/aqp_middleware_duckdb_node-based_query_none_specrecompile_tuned_job.txt"
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
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden"))
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
else
  echo "(skipping tune-config test: $TUNE_JSON not found)"
fi

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
