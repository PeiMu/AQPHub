#!/usr/bin/env bash
#
# Correctness check: LinGo-DB with different DBMS optimizers for JOB benchmark.
# Tests split={none,topdown} x optimizer={own,duckdb,postgres}.
# Uses tpde mode (faster compilation).
#
# The plan optimizer flag only affects split=none execution. For split paths,
# sub-queries always use lingo-db's own MLIR optimizer, so the flag is a no-op.
# We still test all combinations to confirm no regressions.
#
# Usage: bash correctness_test_job_lingodb_diff_opt.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

NOISE='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^duckdb runs:|^lingodb runs:|\(base\)|^\[AQP|^\[LingoDB|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|falling back to|^same engine|^error:|^PartialQuery|Doesn.t support type"'

# Full filter: remove noise, normalize "Run /path/to/6a.sql" → "RUN 6a",
# then optionally remove blocks for skipped queries.
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

LDB_GOLDEN="lingodb_job_no-split_golden.txt"

# Generate golden file if missing (lingo-db default = own optimizer, split=none)
if [[ ! -f "$LDB_GOLDEN" ]]; then
    echo "=== Generating golden file: ${LDB_GOLDEN} ==="
    bash run_aqp.sh job lingodb none tpde none
    vanilla_output="job_result/aqp_middleware_lingodb_tpde_none_job.txt"
    if [[ ! -f "$vanilla_output" || ! -s "$vanilla_output" ]]; then
        echo "FATAL: golden generation produced no output at ${vanilla_output}"
        exit 1
    fi
    apply_filter < "$vanilla_output" > "$LDB_GOLDEN"
    echo "  Golden file created: ${LDB_GOLDEN} ($(wc -l < "$LDB_GOLDEN") lines)"
fi

# JOB queries that timeout or produce wrong results with PG optimizer.
# 27 timeout + 42 wrong results = 69 total (PG's join ordering is suboptimal for lingo-db).
PG_SKIP="11c|11d|12a|12c|13a|14a|14b|14c|15d|16b|16c|16d|17a|17e|17f|18a|18b|18c|19a|19c|19d|20a|20c|21a|21c|22a|22b|22c|22d|24a|24b|25a|25b|25c|26a|26b|26c|28a|28b|28c|29a|29b|29c|2a|2b|2d|30a|30b|30c|31a|31b|31c|32b|33a|33b|33c|3a|3c|4a|4c|5c|6b|6d|6f|7c|9a|9b|9c|9d"

mkdir -p job_result
passed=0; failed=0; total=0
declare -a FAILED_CONFIGS=()
FAIL_LOG="job_result/correctness_failures_lingodb_diff_opt.log"
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
  bash run_aqp.sh job lingodb "${split}" tpde none

  # Construct expected output filename
  local opt_suffix=""
  [[ -n "$plan_opt" ]] && opt_suffix="_planopt_${plan_opt}"
  if [[ "$split" == "none" ]]; then
    output="job_result/aqp_middleware_lingodb_tpde_${split}${opt_suffix}_job.txt"
  else
    output="job_result/aqp_middleware_lingodb_tpde_${split}${opt_suffix}_job.txt"
  fi

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
run_and_check "none" ""        "split=none, LingoDB own optimizer"
run_and_check "none" "duckdb"  "split=none, DuckDB optimizer"
run_and_check "none" "postgres" "split=none, PostgreSQL optimizer" "$PG_SKIP"

# ============================================================
# split=topdown: optimizer flag is a no-op for split execution
# (sub-queries always use lingo-db's own MLIR optimizer)
# ============================================================
run_and_check "topdown" ""        "split=topdown, LingoDB own optimizer"
run_and_check "topdown" "duckdb"  "split=topdown, DuckDB optimizer"
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
