#!/usr/bin/env bash
#
# Correctness check: LinGo-DB with different DBMS optimizers for DSB benchmark.
# Tests split=none with LingoDB (own), DuckDB, and PostgreSQL optimizers.
#
# Usage: bash correctness_test_dsb_lingodb_diff_opt.sh [scale_factor]
#
set -uo pipefail

DSB_SF="${1:-50}"
if [[ "$DSB_SF" == "10" ]]; then result_dir="dsb_result"
else result_dir="dsb_result_sf${DSB_SF}"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^duckdb runs:|^lingodb runs:|\(base\)|^\[AQP|^\[DuckDB\]|^\[LingoDB|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|^Test FAILED|^Error:|^warning:|^Do not support yet|^same engine|falling back to|^error:|^PartialQuery|Doesn.t support type"'

LDB_GOLDEN="lingodb_dsb_no-split_golden.txt"
if [[ "$DSB_SF" != "10" ]]; then
    LDB_GOLDEN="lingodb_dsb_no-split_golden_sf${DSB_SF}.txt"
fi

# Own optimizer: 10 timeout queries
OWN_SKIP="query040|query050|query072|query101|query102"

# DuckDB optimizer: crash/abort/timeout (from 1_instance_out_aqp)
DUCKDB_SKIP="query018$|query019$|query025$|query038$|query040$|query050$|query072$|query084$|query085|query087$|query091$|query099$|query100$|query101$|query102"

# PG optimizer: crash/timeout (from 1_instance_out_aqp)
PG_SKIP="query018$|query038$|query040|query050|query072|query084$|query087$|query100$|query101|query102"

# Auto-generate golden file if missing (lingo-db default = own optimizer)
if [[ ! -f "$LDB_GOLDEN" ]]; then
    echo "=== Generating golden file: ${LDB_GOLDEN} ==="
    mkdir -p "${result_dir}"
    bash run_aqp.sh "dsb_${DSB_SF}" lingodb none llvm none
    vanilla_output="${result_dir}/aqp_middleware_lingodb_llvm_none_dsb.txt"
    if [[ ! -f "$vanilla_output" || ! -s "$vanilla_output" ]]; then
        echo "FATAL: golden generation produced no output at ${vanilla_output}"
        exit 1
    fi
    eval $FILTER "$vanilla_output" > "$LDB_GOLDEN"
    echo "  Golden file created: ${LDB_GOLDEN} ($(wc -l < "$LDB_GOLDEN") lines)"
fi

mkdir -p "${result_dir}"
passed=0; failed=0; total=3
declare -a FAILED_CONFIGS=()
FAIL_LOG="${result_dir}/correctness_failures_lingodb_diff_opt.log"
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
  bash run_aqp.sh "dsb_${DSB_SF}" lingodb none llvm none

  if [[ -z "$plan_opt" ]]; then
    output="${result_dir}/aqp_middleware_lingodb_llvm_none_dsb.txt"
  else
    output="${result_dir}/aqp_middleware_lingodb_llvm_none_planopt_${plan_opt}_dsb.txt"
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

# Step 1: LingoDB own optimizer (default, skip timeout queries)
run_and_check "" "LingoDB own optimizer (default)" "$OWN_SKIP"

# Step 2: DuckDB optimizer (skip crash/abort/timeout queries)
run_and_check "duckdb" "DuckDB optimizer" "$DUCKDB_SKIP"

# Step 3: PostgreSQL optimizer (skip crash/timeout queries)
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
