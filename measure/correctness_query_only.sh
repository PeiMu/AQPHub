#!/usr/bin/env bash
#
# Correctness check: run configs across splits, JIT levels, and kernel paths,
# then diff against golden files.
# Usage: bash correctness_test.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^duckdb runs:|^lingodb runs:|\(base\)|^\[AQP|^\[LingoDB|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average"'

# --- JIT-level configs (no kernel path): engine | split | jit_level | jit_simd | golden_file ---
JIT_CONFIGS=(
  "duckdb|none|query|none|duckdb_job_no-split_golden.txt"
  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt"
  "duckdb|relationship-center|query|none|duckdb_job_relationship-center_golden.txt"
)

# --- Kernel-path configs: engine | split | kernel_path | jit_simd | golden_file ---
KERNEL_CONFIGS=(
)
OLD_KERNEL_CONFIGS=(
  "duckdb|none|pipeline|none|duckdb_job_no-split_golden.txt"
  "duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt"
  "duckdb|relationship-center|pipeline|none|duckdb_job_relationship-center_golden.txt"
)

passed=0
failed=0
total=$(( ${#JIT_CONFIGS[@]} + ${#KERNEL_CONFIGS[@]} ))

# --- Run JIT-level configs via run_job.sh ---
for entry in "${JIT_CONFIGS[@]}"; do
  IFS='|' read -r engine split jit_level jit_simd golden <<< "$entry"
  echo "=== Testing: engine=${engine} split=${split} jit=${jit_level} ==="

  bash run_job.sh "${engine}" "${split}" "${jit_level}" "${jit_simd}"

  output="job_result/aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}_job.txt"

  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    ((failed++))
    continue
  fi
  if [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
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

  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    ((failed++))
    continue
  fi
  if [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
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
    ((failed++))
  fi
  echo ""
done

echo "==============================="
echo "Results: ${passed}/${total} passed, ${failed} failed"
echo "==============================="
exit $failed
