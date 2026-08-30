#!/usr/bin/env bash
#
# Correctness check: LinGo-DB for DSB benchmark.
# Tests all split strategies x compile modes x cache modes.
#
# Usage: bash correctness_test_dsb_lingodb.sh [scale_factor]
#
set -uo pipefail

DSB_SF="${1:-50}"
if [[ "$DSB_SF" == "10" ]]; then result_dir="dsb_result"
else result_dir="dsb_result_sf${DSB_SF}"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^duckdb runs:|^lingodb runs:|\(base\)|^\[AQP|^\[DuckDB\]|^\[LingoDB|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|^Test FAILED|^Error:|^warning:|^Do not support yet|^same engine|falling back to"'

LDB_GOLDEN="lingodb_dsb_no-split_golden.txt"
if [[ "$DSB_SF" != "10" ]]; then
    LDB_GOLDEN="lingodb_dsb_no-split_golden_sf${DSB_SF}.txt"
fi

# Auto-generate golden file if missing
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

JIT_CONFIGS=(
  # No-JIT: 4 splits x 2 modes = 8
  "lingodb|none|llvm|none|${LDB_GOLDEN}"
  "lingodb|none|tpde|none|${LDB_GOLDEN}"
  "lingodb|node-based|llvm|none|${LDB_GOLDEN}"
  "lingodb|node-based|tpde|none|${LDB_GOLDEN}"
  "lingodb|topdown|llvm|none|${LDB_GOLDEN}"
  "lingodb|topdown|tpde|none|${LDB_GOLDEN}"
  "lingodb|relationship-center|llvm|none|${LDB_GOLDEN}"
  "lingodb|relationship-center|tpde|none|${LDB_GOLDEN}"

  # Query-JIT llvm: 4 splits
  "lingodb|none|query|none|${LDB_GOLDEN}"
  "lingodb|node-based|query|none|${LDB_GOLDEN}"
  "lingodb|topdown|query|none|${LDB_GOLDEN}"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}"

  # Query-JIT fastisel: 4 splits
  "lingodb|none|query|none|${LDB_GOLDEN}|||fastisel"
  "lingodb|node-based|query|none|${LDB_GOLDEN}|||fastisel"
  "lingodb|topdown|query|none|${LDB_GOLDEN}|||fastisel"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}|||fastisel"

  # Query-JIT tpde: 4 splits
  "lingodb|none|query|none|${LDB_GOLDEN}|||tpde"
  "lingodb|node-based|query|none|${LDB_GOLDEN}|||tpde"
  "lingodb|topdown|query|none|${LDB_GOLDEN}|||tpde"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}|||tpde"

  # Query-JIT + cache=strict: 4 splits
  "lingodb|none|query|none|${LDB_GOLDEN}||single-run"
  "lingodb|node-based|query|none|${LDB_GOLDEN}||single-run"
  "lingodb|topdown|query|none|${LDB_GOLDEN}||single-run"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}||single-run"

  # Query-JIT + cache=template: 4 splits
  "lingodb|none|query|none|${LDB_GOLDEN}||single-run-template"
  "lingodb|node-based|query|none|${LDB_GOLDEN}||single-run-template"
  "lingodb|topdown|query|none|${LDB_GOLDEN}||single-run-template"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}||single-run-template"

  # Query-JIT + cache=structural: 4 splits
  "lingodb|none|query|none|${LDB_GOLDEN}||structural"
  "lingodb|node-based|query|none|${LDB_GOLDEN}||structural"
  "lingodb|topdown|query|none|${LDB_GOLDEN}||structural"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}||structural"

  # Query-JIT + cache=full: 4 splits
  "lingodb|none|query|none|${LDB_GOLDEN}||full"
  "lingodb|node-based|query|none|${LDB_GOLDEN}||full"
  "lingodb|topdown|query|none|${LDB_GOLDEN}||full"
  "lingodb|relationship-center|query|none|${LDB_GOLDEN}||full"
)

passed=0; failed=0; total=${#JIT_CONFIGS[@]}
declare -a FAILED_CONFIGS=()
FAIL_LOG="${result_dir}/correctness_failures_lingodb.log"
mkdir -p "${result_dir}"; : > "$FAIL_LOG"
rm -rf /dev/shm/aqp_jit_cache/

for entry in "${JIT_CONFIGS[@]}"; do
  IFS='|' read -r engine split jit_level jit_simd golden spec_jit_mode jit_cache_mode compile_mode <<< "$entry"
  spec_jit_mode=${spec_jit_mode:-off}
  jit_cache_mode=${jit_cache_mode:-off}
  compile_mode=${compile_mode:-llvm}
  echo "=== Testing: split=${split} jit=${jit_level} compile=${compile_mode} cache=${jit_cache_mode} ==="

  bash run_aqp.sh "dsb_${DSB_SF}" "${engine}" "${split}" "${jit_level}" "${jit_simd}" \
       on on on on "${jit_cache_mode}" off "${compile_mode}"

  cache_suffix=""
  [[ "$jit_cache_mode" != "off" ]] && cache_suffix="_jitcache_${jit_cache_mode//-/_}"
  fc_suffix=""
  [[ "$compile_mode" != "llvm" ]] && fc_suffix="_${compile_mode}"

  if [[ "$jit_level" == "query" ]]; then
    output="${result_dir}/aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${cache_suffix}${fc_suffix}_dsb.txt"
  else
    output="${result_dir}/aqp_middleware_${engine}_${jit_level}_${split}_dsb.txt"
  fi

  config_label="split=${split} jit=${jit_level} compile=${compile_mode} cache=${jit_cache_mode}"

  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"; ((failed++)); continue
  fi
  if [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"; ((failed++)); continue
  fi

  if [[ "$jit_cache_mode" == "full" ]]; then
    d0=$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    d1=$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0+iter1)"; ((passed++))
    else
      echo "  FAIL"; FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"; ((failed++))
    fi
  else
    d=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    if [[ -z "$d" ]]; then
      echo "  PASS"; ((passed++))
    else
      echo "  FAIL"; echo "$d" | head -10
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"; echo "" >> "$FAIL_LOG"; ((failed++))
    fi
  fi
  echo ""
done

echo "==============================="
echo "Results: ${passed}/${total} passed, ${failed} failed"
echo "==============================="
if (( ${#FAILED_CONFIGS[@]} > 0 )); then
  echo ""; echo "Failed configs:"
  for i in "${!FAILED_CONFIGS[@]}"; do echo "  $((i+1)). ${FAILED_CONFIGS[$i]}"; done
  echo ""; echo "Full diffs: $FAIL_LOG"
fi
exit $failed
