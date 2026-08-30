#!/usr/bin/env bash
#
# Correctness check: LinGo-DB for JOB benchmark.
# Tests all split strategies x compile modes x cache modes.
#
# Usage: bash correctness_test_job_lingodb.sh
#
# Config format:
#   engine|split|jit_level|jit_simd|golden[|spec_jit[|jit_cache[|compile_mode]]]
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^duckdb runs:|^lingodb runs:|\(base\)|^\[AQP|^\[LingoDB|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|falling back to|^same engine"'

LDB_GOLDEN="lingodb_job_no-split_golden.txt"
DUCK_NB_GOLDEN="duckdb_job_node-based_golden.txt"

JIT_CONFIGS=(
  # ============================================================
  # No-JIT: 4 splits x 2 lingodb-modes = 8 configs
  # ============================================================
  "lingodb|none|llvm|none|${LDB_GOLDEN}"
  "lingodb|none|tpde|none|${LDB_GOLDEN}"
  "lingodb|node-based|llvm|none|${DUCK_NB_GOLDEN}"
  "lingodb|node-based|tpde|none|${DUCK_NB_GOLDEN}"
  "lingodb|topdown|llvm|none|${LDB_GOLDEN}"
  "lingodb|topdown|tpde|none|${LDB_GOLDEN}"
  "lingodb|relationship-center|llvm|none|${LDB_GOLDEN}"
  "lingodb|relationship-center|tpde|none|${LDB_GOLDEN}"

  # ============================================================
  # Query-JIT, compile_mode=llvm: 4 splits = 4 configs
  # ============================================================
  "lingodb|none|query|none|${LDB_GOLDEN}"
  "lingodb|node-based|query|none|${DUCK_NB_GOLDEN}"
  "lingodb|topdown|query|none|${LDB_GOLDEN}"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}"

  # ============================================================
  # Query-JIT, compile_mode=fastisel: 4 splits = 4 configs
  # ============================================================
  "lingodb|none|query|none|${LDB_GOLDEN}|||fastisel"
  "lingodb|node-based|query|none|${DUCK_NB_GOLDEN}|||fastisel"
  "lingodb|topdown|query|none|${LDB_GOLDEN}|||fastisel"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}|||fastisel"

  # ============================================================
  # Query-JIT, compile_mode=tpde: 4 splits = 4 configs
  # ============================================================
  "lingodb|none|query|none|${LDB_GOLDEN}|||tpde"
  "lingodb|node-based|query|none|${DUCK_NB_GOLDEN}|||tpde"
  "lingodb|topdown|query|none|${LDB_GOLDEN}|||tpde"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}|||tpde"

  # ============================================================
  # Query-JIT + jit-cache=single-run (strict): 4 splits = 4 configs
  # ============================================================
  "lingodb|none|query|none|${LDB_GOLDEN}||single-run"
  "lingodb|node-based|query|none|${DUCK_NB_GOLDEN}||single-run"
  "lingodb|topdown|query|none|${LDB_GOLDEN}||single-run"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}||single-run"

  # ============================================================
  # Query-JIT + jit-cache=single-run-template: 4 splits = 4 configs
  # ============================================================
  "lingodb|none|query|none|${LDB_GOLDEN}||single-run-template"
  "lingodb|node-based|query|none|${DUCK_NB_GOLDEN}||single-run-template"
  "lingodb|topdown|query|none|${LDB_GOLDEN}||single-run-template"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}||single-run-template"

  # ============================================================
  # Query-JIT + jit-cache=structural: 4 splits = 4 configs
  # ============================================================
  "lingodb|none|query|none|${LDB_GOLDEN}||structural"
  "lingodb|node-based|query|none|${DUCK_NB_GOLDEN}||structural"
  "lingodb|topdown|query|none|${LDB_GOLDEN}||structural"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}||structural"

  # ============================================================
  # Query-JIT + jit-cache=full: 4 splits = 4 configs
  # ============================================================
  "lingodb|none|query|none|${LDB_GOLDEN}||full"
  "lingodb|node-based|query|none|${DUCK_NB_GOLDEN}||full"
  "lingodb|topdown|query|none|${LDB_GOLDEN}||full"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}||full"
)

KNOWN_DIFFS_NB="known_diffs_node-based.txt"

filter_known_diffs() {
  local diff_text="$1" golden="$2"
  if [[ "$golden" != *"node-based"* ]] || [[ ! -f "$KNOWN_DIFFS_NB" ]]; then
    echo "$diff_text"; return
  fi
  local known_lines
  known_lines=$(grep -v '^#' "$KNOWN_DIFFS_NB" | grep -v '^$' || true)
  if [[ -z "$known_lines" ]]; then
    echo "$diff_text"; return
  fi
  echo "$diff_text" | awk -v known="$known_lines" '
    BEGIN { split(known, ka, "\n"); for (i in ka) kset[ka[i]] = 1 }
    /^[0-9]/ { hdr=$0; left=""; right=""; next }
    /^< / { left=substr($0,3); next }
    /^---$/ { next }
    /^> / {
      right=substr($0,3);
      if (left in kset || right in kset) { left=""; right=""; next }
      print hdr; print "< " left; print "---"; print "> " right;
      left=""; right=""
    }
  '
}

rm -rf /dev/shm/aqp_jit_cache/
mkdir -p job_result

passed=0
failed=0
total=${#JIT_CONFIGS[@]}
declare -a FAILED_CONFIGS=()
FAIL_LOG="job_result/correctness_failures_lingodb.log"
: > "$FAIL_LOG"

for entry in "${JIT_CONFIGS[@]}"; do
  IFS='|' read -r engine split jit_level jit_simd golden spec_jit_mode jit_cache_mode compile_mode <<< "$entry"
  spec_jit_mode=${spec_jit_mode:-off}
  jit_cache_mode=${jit_cache_mode:-off}
  compile_mode=${compile_mode:-llvm}
  echo "=== Testing: split=${split} jit=${jit_level} compile=${compile_mode} cache=${jit_cache_mode} ==="

  bash run_aqp.sh job "${engine}" "${split}" "${jit_level}" "${jit_simd}" \
       on on on on "${jit_cache_mode}" off "${compile_mode}"

  cache_suffix=""
  if [[ "$jit_cache_mode" != "off" ]]; then
    cache_suffix="_jitcache_${jit_cache_mode//-/_}"
  fi
  fc_suffix=""
  [[ "$compile_mode" != "llvm" ]] && fc_suffix="_${compile_mode}"

  if [[ "$jit_level" == "query" ]]; then
    output="job_result/aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${cache_suffix}${fc_suffix}_job.txt"
  else
    output="job_result/aqp_middleware_${engine}_${jit_level}_${split}_job.txt"
  fi

  config_label="split=${split} jit=${jit_level} compile=${compile_mode} cache=${jit_cache_mode}"

  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++)); continue
  fi
  if [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++)); continue
  fi

  if [[ "$jit_cache_mode" == "full" ]]; then
    d0_raw=$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    d1_raw=$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    d0=$(filter_known_diffs "$d0_raw" "$golden")
    d1=$(filter_known_diffs "$d1_raw" "$golden")
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0 + iter1)"; ((passed++))
    else
      echo "  FAIL: differences found"
      [[ -n "$d0" ]] && echo "  iter0 diff:" && echo "$d0" | head -10
      [[ -n "$d1" ]] && echo "  iter1 diff:" && echo "$d1" | head -10
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "iter0:" >> "$FAIL_LOG" && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "iter1:" >> "$FAIL_LOG" && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"; ((failed++))
    fi
  else
    d_raw=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    d=$(filter_known_diffs "$d_raw" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"; ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"; ((failed++))
    fi
  fi
  echo ""
done

echo "==============================="
echo "Results: ${passed}/${total} passed, ${failed} failed"
echo "==============================="
if (( ${#FAILED_CONFIGS[@]} > 0 )); then
  echo ""; echo "Failed configs:"
  for i in "${!FAILED_CONFIGS[@]}"; do
    echo "  $((i+1)). ${FAILED_CONFIGS[$i]}"
  done
  echo ""; echo "Full diffs saved to: $FAIL_LOG"
fi
exit $failed
