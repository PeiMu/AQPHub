#!/usr/bin/env bash
#
# Correctness check: LinGo-DB with different DBMS optimizers for JOB benchmark.
# Tests split=none with LingoDB (own), DuckDB, and PostgreSQL optimizers.
#
# Usage: bash correctness_test_job_lingodb_diff_opt.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^duckdb runs:|^lingodb runs:|\(base\)|^\[AQP|^\[LingoDB|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|falling back to|^same engine|^error:|^PartialQuery|Doesn.t support type"'

LDB_GOLDEN="lingodb_job_no-split_golden.txt"

# Generate golden file if missing (lingo-db default = own optimizer)
if [[ ! -f "$LDB_GOLDEN" ]]; then
    echo "=== Generating golden file: ${LDB_GOLDEN} ==="
    bash run_aqp.sh job lingodb none llvm none
    vanilla_output="job_result/aqp_middleware_lingodb_llvm_none_job.txt"
    if [[ ! -f "$vanilla_output" || ! -s "$vanilla_output" ]]; then
        echo "FATAL: golden generation produced no output at ${vanilla_output}"
        exit 1
    fi
    eval $FILTER "$vanilla_output" > "$LDB_GOLDEN"
    echo "  Golden file created: ${LDB_GOLDEN} ($(wc -l < "$LDB_GOLDEN") lines)"
fi

# 27 JOB queries that timeout (>30s) with PG optimizer due to suboptimal join ordering.
PG_SKIP="12a|14c|15d|16b|16c|16d|17a|17e|17f|18c|19d|20a|22c|22d|26a|26c|29c|2a|2b|2d|33b|4a|4c|6d|6f|9c|9d"

mkdir -p job_result
passed=0; failed=0; total=3
declare -a FAILED_CONFIGS=()
FAIL_LOG="job_result/correctness_failures_lingodb_diff_opt.log"
: > "$FAIL_LOG"

run_and_check() {
  local plan_opt="$1" description="$2" skip_queries="${3:-}"

  if [[ -z "$plan_opt" ]]; then
    label="default (own)"
    echo "=== Testing: ${description} ==="
  else
    label="--lingodb-plan-optimizer=${plan_opt}"
    echo "=== Testing: ${description} (${label}) ==="
  fi

  AQP_LINGODB_PLAN_OPTIMIZER="${plan_opt}" \
  AQP_SKIP_QUERIES="${skip_queries}" \
  bash run_aqp.sh job lingodb none llvm none

  if [[ -z "$plan_opt" ]]; then
    output="job_result/aqp_middleware_lingodb_llvm_none_job.txt"
  else
    output="job_result/aqp_middleware_lingodb_llvm_none_planopt_${plan_opt}_job.txt"
  fi

  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output not found: $output"
    FAILED_CONFIGS+=("$label  [output missing: $output]")
    echo "--- $label ---" >> "$FAIL_LOG"
    echo "output not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"; ((failed++)); return
  fi

  local golden_filtered
  if [[ -n "$skip_queries" ]]; then
    golden_filtered=$(eval $FILTER "$LDB_GOLDEN" | grep -v -E "^(${skip_queries})" || true)
  else
    golden_filtered=$(eval $FILTER "$LDB_GOLDEN")
  fi

  d=$(diff <(eval $FILTER "$output") <(echo "$golden_filtered") || true)
  if [[ -z "$d" ]]; then
    echo "  PASS"; ((passed++))
  else
    ndiffs=$(echo "$d" | grep -c "^[<>]" || true)
    echo "  FAIL: ${ndiffs} differing lines"
    echo "$d" | head -20
    FAILED_CONFIGS+=("$label")
    echo "--- $label ---" >> "$FAIL_LOG"
    echo "$d" >> "$FAIL_LOG"; echo "" >> "$FAIL_LOG"; ((failed++))
  fi
  echo ""
}

# Step 1: LingoDB own optimizer (default, all queries)
run_and_check "" "LingoDB own optimizer (default)"

# Step 2: DuckDB optimizer (all queries)
run_and_check "duckdb" "DuckDB optimizer"

# Step 3: PostgreSQL optimizer (skip 27 timeout queries)
run_and_check "postgres" "PostgreSQL optimizer" "$PG_SKIP"

echo "==============================="
echo "Results: ${passed}/${total} passed, ${failed} failed"
echo "==============================="
if (( ${#FAILED_CONFIGS[@]} > 0 )); then
  echo ""; echo "Failed configs:"
  for i in "${!FAILED_CONFIGS[@]}"; do echo "  $((i+1)). ${FAILED_CONFIGS[$i]}"; done
  echo ""; echo "Full diffs: $FAIL_LOG"
fi
exit $failed
