#!/usr/bin/env bash
#
# Correctness check: run 6 configs (3 splits × 2 JIT levels) and diff against golden.
# JIT configs use --storage-plan --csr-support=inner to verify CSR kernel correctness.
# Usage: bash correctness_test.sh
#
set -uo pipefail

FILTER='grep -v "^Running\|^==\|^Execution\|^$\|^waiting\|^server\|^ANALYZ\|^duckdb runs:\|^(base)\|^\[AQP\|^\[Storage\|^\[CSR"'

# split | jit_level | jit_opt | jit_simd | csr_support | golden_file
CONFIGS=(
  "none|none|o1|none|none|duckdb_job_no-split_golden.txt"
  "none|pipeline|o1|none|inner|duckdb_job_no-split_golden.txt"
  "node-based|none|o1|none|none|duckdb_job_node-based_golden.txt"
  "node-based|pipeline|o1|none|inner|duckdb_job_node-based_golden.txt"
  "relationship-center|none|o1|none|none|duckdb_job_relationship-center_golden.txt"
  "relationship-center|pipeline|o1|none|inner|duckdb_job_relationship-center_golden.txt"
)

passed=0
failed=0
total=${#CONFIGS[@]}

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r split jit_level jit_opt jit_simd csr_support golden <<< "$entry"
  echo "=== Testing: split=${split} jit=${jit_level} csr=${csr_support} ==="

  bash run_job.sh duckdb "${split}" "${jit_level}" "${jit_opt}" "${jit_simd}" \
       on on on on on on off "${csr_support}"

  # Build output filename matching run_job.sh's naming convention
  flag_suffix=""
  [[ "$csr_support" != "none" ]] && flag_suffix="_csr${csr_support}"
  output="job_result/aqp_middleware_duckdb_${split}_${jit_level}_${jit_opt}_${jit_simd}${flag_suffix}_job.txt"

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
