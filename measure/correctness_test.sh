#!/usr/bin/env bash
#
# Correctness check: run configs across splits, JIT levels, and kernel paths,
# then diff against golden files.
# Usage: bash correctness_test.sh
#
set -uo pipefail

FILTER='grep -v "^Running\|^==\|^Execution\|^$\|^waiting\|^server\|^ANALYZ\|^duckdb runs:\|^(base)\|^\[AQP\|^\[Storage\|^\[CSR\|^\[Dim\|^  [a-z_]*: [0-9]* rows$"'

# --- JIT-level configs (no kernel path): split | jit_level | jit_opt | jit_simd | golden_file ---
JIT_CONFIGS=(
  "none|none|o1|none|duckdb_job_no-split_golden.txt"
  "node-based|none|o1|none|duckdb_job_node-based_golden.txt"
  "relationship-center|none|o1|none|duckdb_job_relationship-center_golden.txt"
)

# --- Kernel-path configs: split | kernel_path | jit_opt | jit_simd | golden_file ---
KERNEL_CONFIGS=(
  "none|pipeline|o1|none|duckdb_job_no-split_golden.txt"
  "node-based|pipeline|o1|none|duckdb_job_node-based_golden.txt"
  "relationship-center|pipeline|o1|none|duckdb_job_relationship-center_golden.txt"
)

passed=0
failed=0
total=$(( ${#JIT_CONFIGS[@]} + ${#KERNEL_CONFIGS[@]} ))

# --- Run JIT-level configs via run_job.sh ---
for entry in "${JIT_CONFIGS[@]}"; do
  IFS='|' read -r split jit_level jit_opt jit_simd golden <<< "$entry"
  echo "=== Testing: split=${split} jit=${jit_level} ==="

  bash run_job.sh duckdb "${split}" "${jit_level}" "${jit_opt}" "${jit_simd}" \
       on on on on on on off

  output="job_result/aqp_middleware_duckdb_${split}_${jit_level}_${jit_opt}_${jit_simd}_job.txt"

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
  IFS='|' read -r split kernel_path jit_opt jit_simd golden <<< "$entry"
  echo "=== Testing: split=${split} kernel-path=${kernel_path} ==="

  bash run_job_kernel.sh duckdb "${split}" "${kernel_path}" "${jit_opt}" "${jit_simd}" \
       on on on on on off

  output="job_result/aqp_middleware_duckdb_${split}_kernel-${kernel_path}_${jit_opt}_${jit_simd}_job.txt"

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
