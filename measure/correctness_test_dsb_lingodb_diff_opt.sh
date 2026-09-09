#!/usr/bin/env bash
#
# Correctness check: LinGo-DB with different DBMS optimizers for DSB benchmark.
# Tests split={none,topdown} x optimizer={own,duckdb,postgres}.
# Uses tpde mode (faster compilation).
#
# The plan optimizer flag only affects split=none execution. For split paths,
# sub-queries always use lingo-db's own MLIR optimizer, so the flag is a no-op.
# We still test all combinations to confirm no regressions.
#
# Usage: bash correctness_test_dsb_lingodb_diff_opt.sh [scale_factor]
#
set -uo pipefail

DSB_SF="${1:-50}"
if [[ "$DSB_SF" == "10" ]]; then result_dir="dsb_result"
else result_dir="dsb_result_sf${DSB_SF}"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

NOISE='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^duckdb runs:|^lingodb runs:|\(base\)|^\[AQP|^\[DuckDB\]|^\[LingoDB|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|^Test FAILED|^Error:|^warning:|^Do not support yet|^same engine|falling back to|^error:|^PartialQuery|Doesn.t support type"'

apply_filter() {
  local skip_pattern="${1:-}"
  eval $NOISE | sed -E 's|^Run .*/([^/]+)\.sql$|RUN \1|' | awk -v pat="$skip_pattern" '
    BEGIN { skip=0; if (pat != "") { n=split(pat,arr,"|"); for(i=1;i<=n;i++) skipset[arr[i]]=1 } }
    /^RUN / {
      qname = $2
      skip = (qname in skipset) ? 1 : 0
    }
    !skip { print }
  '
}

LDB_GOLDEN="lingodb_dsb_no-split_golden.txt"
if [[ "$DSB_SF" != "10" ]]; then
    LDB_GOLDEN="lingodb_dsb_no-split_golden_sf${DSB_SF}.txt"
fi

# Own optimizer: timeout queries
OWN_SKIP="query040|query050|query072|query101|query102"

# DuckDB optimizer: crash/abort/timeout
DUCKDB_SKIP="query018$|query019$|query025$|query038$|query040$|query050$|query072$|query084$|query085|query087$|query091$|query099$|query100$|query101$|query102"

# PG optimizer: crash/timeout
PG_SKIP="query018$|query038$|query040|query050|query072|query084$|query087$|query100$|query101|query102"

# Auto-generate golden file if missing (lingo-db default = own optimizer)
if [[ ! -f "$LDB_GOLDEN" ]]; then
    echo "=== Generating golden file: ${LDB_GOLDEN} ==="
    mkdir -p "${result_dir}"
    AQP_SKIP_QUERIES="${OWN_SKIP}" \
    bash run_aqp.sh "dsb_${DSB_SF}" lingodb none tpde none
    vanilla_output="${result_dir}/aqp_middleware_lingodb_tpde_none_dsb.txt"
    if [[ ! -f "$vanilla_output" || ! -s "$vanilla_output" ]]; then
        echo "FATAL: golden generation produced no output at ${vanilla_output}"
        exit 1
    fi
    apply_filter < "$vanilla_output" > "$LDB_GOLDEN"
    echo "  Golden file created: ${LDB_GOLDEN} ($(wc -l < "$LDB_GOLDEN") lines)"
fi

mkdir -p "${result_dir}"
passed=0; failed=0; total=0
declare -a FAILED_CONFIGS=()
FAIL_LOG="${result_dir}/correctness_failures_lingodb_diff_opt.log"
: > "$FAIL_LOG"

run_and_check() {
  local split="$1" plan_opt="$2" description="$3" skip_queries="${4:-}" golden="${5:-${LDB_GOLDEN}}"
  ((total++))

  if [[ -z "$plan_opt" ]]; then
    label="split=${split}, optimizer=own"
  else
    label="split=${split}, optimizer=${plan_opt}"
  fi
  echo "=== Testing: ${description} (${label}) ==="

  AQP_LINGODB_PLAN_OPTIMIZER="${plan_opt}" \
  AQP_SKIP_QUERIES="${skip_queries}" \
  bash run_aqp.sh "dsb_${DSB_SF}" lingodb "${split}" tpde none

  # Construct expected output filename
  local opt_suffix=""
  [[ -n "$plan_opt" ]] && opt_suffix="_planopt_${plan_opt}"
  output="${result_dir}/aqp_middleware_lingodb_tpde_${split}${opt_suffix}_dsb.txt"

  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output not found: $output"
    FAILED_CONFIGS+=("${label}  [output missing: $output]")
    echo "--- ${label} ---" >> "$FAIL_LOG"
    echo "output not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"; ((failed++)); return
  fi

  local golden_filtered
  golden_filtered=$(apply_filter "$skip_queries" < "$golden")

  d=$(diff <(apply_filter < "$output") <(echo "$golden_filtered") || true)
  if [[ -z "$d" ]]; then
    echo "  PASS"; ((passed++))
  else
    ndiffs=$(echo "$d" | grep -c "^[<>]" || true)
    echo "  FAIL: ${ndiffs} differing lines"
    echo "$d" | head -20
    FAILED_CONFIGS+=("${label}")
    echo "--- ${label} ---" >> "$FAIL_LOG"
    echo "$d" >> "$FAIL_LOG"; echo "" >> "$FAIL_LOG"; ((failed++))
  fi
  echo ""
}

# ============================================================
# split=none: optimizer changes the query plan
# ============================================================
run_and_check "none" ""        "split=none, LingoDB own optimizer" "$OWN_SKIP"
run_and_check "none" "duckdb"  "split=none, DuckDB optimizer" "$DUCKDB_SKIP"
run_and_check "none" "postgres" "split=none, PostgreSQL optimizer" "$PG_SKIP"

# ============================================================
# split=topdown: optimizer flag is a no-op for split execution
# (sub-queries always use lingo-db's own MLIR optimizer)
# ============================================================
run_and_check "topdown" ""        "split=topdown, LingoDB own optimizer" "$OWN_SKIP"
run_and_check "topdown" "duckdb"  "split=topdown, DuckDB optimizer" "$DUCKDB_SKIP"
run_and_check "topdown" "postgres" "split=topdown, PostgreSQL optimizer" "$PG_SKIP"

echo "==============================="
echo "Results: ${passed}/${total} passed, ${failed} failed"
echo "==============================="
if (( ${#FAILED_CONFIGS[@]} > 0 )); then
  echo ""; echo "Failed configs:"
  for i in "${!FAILED_CONFIGS[@]}"; do echo "  $((i+1)). ${FAILED_CONFIGS[$i]}"; done
  echo ""; echo "Full diffs: $FAIL_LOG"
fi
exit $failed
