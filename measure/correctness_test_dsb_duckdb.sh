#!/usr/bin/env bash

# DSB scale factor: pass as first argument (default 10).
DSB_SF="${1:-10}"
if [[ "$DSB_SF" == "10" ]]; then
    result_dir="dsb_result"
else
    result_dir="dsb_result_sf${DSB_SF}"
fi

# Per-SF golden file: regenerate automatically if missing.
if [[ "$DSB_SF" == "10" ]]; then
    GOLDEN_FILE="duckdb_dsb_no-split_golden.txt"
else
    GOLDEN_FILE="duckdb_dsb_no-split_golden_sf${DSB_SF}.txt"
fi
#
# Correctness check for DSB benchmark: run configs across splits, JIT levels,
# compile modes, and cache modes, then diff against golden files.
# Usage: bash correctness_test_dsb_duckdb.sh [scale_factor]
#
# Mirrors correctness_test_job_duckdb.sh for DSB queries.
#
# Config format:
#   engine|split|jit_level|jit_simd|golden[|spec_jit[|jit_cache[|compile_mode[|skip_hash_cmp]]]]
#   compile_mode defaults to llvm, skip_hash_cmp defaults to on(=all) when omitted.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^duckdb runs:|^lingodb runs:|\(base\)|^\[AQP|^\[DuckDB\]|^\[LingoDB|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|^Test FAILED|^Error:|^warning:|^Do not support yet|^same engine|falling back to"'

# --- JIT-level configs (no kernel path) ---
JIT_CONFIGS=(
  # ============================================================
  # Interpreter baseline (no JIT)
  # ============================================================
  "duckdb|none|none|none|${GOLDEN_FILE}"
  "duckdb|node-based|none|none|${GOLDEN_FILE}"

  # ============================================================
  # expr-jit, compile_mode=llvm (default)
  # ============================================================
  "duckdb|none|expr|none|${GOLDEN_FILE}"
  "duckdb|node-based|expr|none|${GOLDEN_FILE}"

  # expr-jit, compile_mode=fastisel
#  "duckdb|node-based|expr|none|${GOLDEN_FILE}|||fastisel"

  # expr-jit, compile_mode=tpde
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|||tpde"
  # expr-jit, none-split, fastisel/tpde
#  "duckdb|none|expr|none|${GOLDEN_FILE}|||fastisel"
  "duckdb|none|expr|none|${GOLDEN_FILE}|||tpde"

  # expr-jit spec-jit (node-based, llvm)
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile"
  # expr-jit spec-jit, fastisel/tpde
#  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile||fastisel"
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile||tpde"

  # ============================================================
  # operator-jit, compile_mode=llvm
  # ============================================================
  "duckdb|none|operator|none|${GOLDEN_FILE}"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}"

  # operator-jit, compile_mode=fastisel
#  "duckdb|node-based|operator|none|${GOLDEN_FILE}|||fastisel"

  # operator-jit, compile_mode=tpde
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|||tpde"
  # operator-jit, none-split, fastisel/tpde
#  "duckdb|none|operator|none|${GOLDEN_FILE}|||fastisel"
  "duckdb|none|operator|none|${GOLDEN_FILE}|||tpde"

  # operator-jit spec-jit (node-based, llvm)
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile"
  # operator-jit spec-jit, fastisel/tpde
#  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile||fastisel"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile||tpde"

  # ============================================================
  # pipeline-jit (deprioritized — commented out)
  # ============================================================
  #"duckdb|none|pipeline|none|${GOLDEN_FILE}"
  #"duckdb|node-based|pipeline|none|${GOLDEN_FILE}"
  #"duckdb|node-based|pipeline|none|${GOLDEN_FILE}|recompile"
  #"duckdb|node-based|pipeline|none|${GOLDEN_FILE}|||fastisel"
  #"duckdb|node-based|pipeline|none|${GOLDEN_FILE}|||tpde"

  # ============================================================
  # query-jit, compile_mode=llvm
  # ============================================================
  "duckdb|none|query|none|${GOLDEN_FILE}"
  "duckdb|node-based|query|none|${GOLDEN_FILE}"

  # spec-jit (query, llvm)
  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile"

  # query-jit, compile_mode=fastisel
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|||fastisel"

  # query-jit, compile_mode=tpde
  "duckdb|node-based|query|none|${GOLDEN_FILE}|||tpde"
  # query-jit, none-split, fastisel/tpde
#  "duckdb|none|query|none|${GOLDEN_FILE}|||fastisel"
  "duckdb|none|query|none|${GOLDEN_FILE}|||tpde"
  # query-jit spec-jit, fastisel/tpde
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile||fastisel"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile||tpde"

  # ============================================================
  # query-jit, skip_hash_cmp=off
  # ============================================================
  "duckdb|node-based|query|none|${GOLDEN_FILE}||||off"
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|||fastisel|off"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|||tpde|off"
  "duckdb|none|query|none|${GOLDEN_FILE}||||off"
  # skip_hash_cmp=off + spec-jit / cache tiers
  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile|||off"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|single-run-strict||off"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|single-run-template||off"
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|full||off"

  # ============================================================
  # jit-cache=single-run-strict
  # ============================================================
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|off|single-run-strict"
#  "duckdb|node-based|expr|none|${GOLDEN_FILE}|off|single-run-strict|fastisel"
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|off|single-run-strict|tpde"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|off|single-run-strict"
#  "duckdb|node-based|operator|none|${GOLDEN_FILE}|off|single-run-strict|fastisel"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|off|single-run-strict|tpde"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|single-run-strict"
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|single-run-strict|fastisel"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|single-run-strict|tpde"
  # strict + spec-jit=recompile (llvm)
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile|single-run-strict"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile|single-run-strict"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile|single-run-strict"
  # strict + spec-jit=recompile (fastisel/tpde)
#  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile|single-run-strict|fastisel"
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile|single-run-strict|tpde"
#  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile|single-run-strict|fastisel"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile|single-run-strict|tpde"
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile|single-run-strict|fastisel"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile|single-run-strict|tpde"

  # ============================================================
  # jit-cache=single-run-template
  # ============================================================
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|off|single-run-template"
#  "duckdb|node-based|expr|none|${GOLDEN_FILE}|off|single-run-template|fastisel"
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|off|single-run-template|tpde"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|off|single-run-template"
#  "duckdb|node-based|operator|none|${GOLDEN_FILE}|off|single-run-template|fastisel"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|off|single-run-template|tpde"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|single-run-template"
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|single-run-template|fastisel"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|single-run-template|tpde"
  # template + spec-jit=recompile (llvm)
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile|single-run-template"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile|single-run-template"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile|single-run-template"
  # template + spec-jit=recompile (fastisel/tpde)
#  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile|single-run-template|fastisel"
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile|single-run-template|tpde"
#  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile|single-run-template|fastisel"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile|single-run-template|tpde"
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile|single-run-template|fastisel"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile|single-run-template|tpde"

  # ============================================================
  # jit-cache=full
  # ============================================================
  "duckdb|none|expr|none|${GOLDEN_FILE}|off|full"
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|off|full"
#  "duckdb|none|expr|none|${GOLDEN_FILE}|off|full|fastisel"
#  "duckdb|none|expr|none|${GOLDEN_FILE}|off|full|tpde"
#  "duckdb|node-based|expr|none|${GOLDEN_FILE}|off|full|fastisel"
#  "duckdb|node-based|expr|none|${GOLDEN_FILE}|off|full|tpde"
  "duckdb|none|operator|none|${GOLDEN_FILE}|off|full"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|off|full"
#  "duckdb|none|operator|none|${GOLDEN_FILE}|off|full|fastisel"
#  "duckdb|none|operator|none|${GOLDEN_FILE}|off|full|tpde"
#  "duckdb|node-based|operator|none|${GOLDEN_FILE}|off|full|fastisel"
#  "duckdb|node-based|operator|none|${GOLDEN_FILE}|off|full|tpde"
  "duckdb|none|query|none|${GOLDEN_FILE}|off|full"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|full"
#  "duckdb|none|query|none|${GOLDEN_FILE}|off|full|fastisel"
#  "duckdb|none|query|none|${GOLDEN_FILE}|off|full|tpde"
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|full|fastisel"
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|off|full|tpde"
  # full + spec-jit=recompile (llvm)
  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile|full"
  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile|full"
  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile|full"
  # full + spec-jit=recompile (fastisel/tpde)
#  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile|full|fastisel"
#  "duckdb|node-based|expr|none|${GOLDEN_FILE}|recompile|full|tpde"
#  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile|full|fastisel"
#  "duckdb|node-based|operator|none|${GOLDEN_FILE}|recompile|full|tpde"
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile|full|fastisel"
#  "duckdb|node-based|query|none|${GOLDEN_FILE}|recompile|full|tpde"

  # ============================================================
  # lingodb / lingo-db-runtime
  # ============================================================
  #"lingodb|none|llvm|none|lingodb_dsb_no-split_golden.txt"
  #"lingodb|node-based|llvm|none|lingodb_dsb_no-split_golden.txt"
  #"lingodb|none|tpde|none|lingodb_dsb_no-split_golden.txt"
  #"lingodb|node-based|tpde|none|lingodb_dsb_no-split_golden.txt"
  # lingo-db-runtime disabled: pre-existing hangs/crashes on several DSB queries
  #"lingo-db-runtime|node-based|llvm|none|lingodb_dsb_no-split_golden.txt"
  #"lingo-db-runtime|node-based|tpde|none|lingodb_dsb_no-split_golden.txt"
  #"lingo-db-runtime|none|llvm|none|lingodb_dsb_no-split_golden.txt"
  #"lingo-db-runtime|none|tpde|none|lingodb_dsb_no-split_golden.txt"

  # ============================================================
  # topdown (SDS) — uses the same no-split golden (ground truth)
  # ============================================================
  # no-jit
  "duckdb|topdown|none|none|${GOLDEN_FILE}"
  # expr-jit (llvm / fastisel / tpde)
  "duckdb|topdown|expr|none|${GOLDEN_FILE}"
#  "duckdb|topdown|expr|none|${GOLDEN_FILE}|||fastisel"
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|||tpde"
  # operator-jit (llvm / fastisel / tpde)
  "duckdb|topdown|operator|none|${GOLDEN_FILE}"
#  "duckdb|topdown|operator|none|${GOLDEN_FILE}|||fastisel"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|||tpde"
  # query-jit (llvm / fastisel / tpde)
  "duckdb|topdown|query|none|${GOLDEN_FILE}"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|||fastisel"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|||tpde"
  # query-jit, skip_hash_cmp=off
  "duckdb|topdown|query|none|${GOLDEN_FILE}||||off"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|||fastisel|off"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|||tpde|off"
  # jit-cache=single-run-strict (expr / operator / query x llvm / fastisel / tpde)
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|off|single-run-strict"
#  "duckdb|topdown|expr|none|${GOLDEN_FILE}|off|single-run-strict|fastisel"
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|off|single-run-strict|tpde"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|off|single-run-strict"
#  "duckdb|topdown|operator|none|${GOLDEN_FILE}|off|single-run-strict|fastisel"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|off|single-run-strict|tpde"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|single-run-strict"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|single-run-strict|fastisel"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|single-run-strict|tpde"
  # skip_hash_cmp=off + cache tiers
  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|single-run-strict||off"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|single-run-template||off"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|full||off"
  # jit-cache=single-run-template (expr / operator / query x llvm / fastisel / tpde)
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|off|single-run-template"
#  "duckdb|topdown|expr|none|${GOLDEN_FILE}|off|single-run-template|fastisel"
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|off|single-run-template|tpde"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|off|single-run-template"
#  "duckdb|topdown|operator|none|${GOLDEN_FILE}|off|single-run-template|fastisel"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|off|single-run-template|tpde"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|single-run-template"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|single-run-template|fastisel"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|single-run-template|tpde"
  # jit-cache=full (expr / operator / query x llvm / fastisel / tpde)
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|off|full"
#  "duckdb|topdown|expr|none|${GOLDEN_FILE}|off|full|fastisel"
#  "duckdb|topdown|expr|none|${GOLDEN_FILE}|off|full|tpde"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|off|full"
#  "duckdb|topdown|operator|none|${GOLDEN_FILE}|off|full|fastisel"
#  "duckdb|topdown|operator|none|${GOLDEN_FILE}|off|full|tpde"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|full"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|full|fastisel"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|off|full|tpde"

  # ============================================================
  # topdown + spec-jit=recompile
  # ============================================================
  # cache=off
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile"
#  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile||fastisel"
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile||tpde"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile"
#  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile||fastisel"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile||tpde"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile||fastisel"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile||tpde"
  # cache=single-run-strict + spec=recompile
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile|single-run-strict"
#  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile|single-run-strict|fastisel"
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile|single-run-strict|tpde"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile|single-run-strict"
#  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile|single-run-strict|fastisel"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile|single-run-strict|tpde"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile|single-run-strict"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile|single-run-strict|fastisel"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile|single-run-strict|tpde"
  # cache=single-run-template + spec=recompile
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile|single-run-template"
#  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile|single-run-template|fastisel"
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile|single-run-template|tpde"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile|single-run-template"
#  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile|single-run-template|fastisel"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile|single-run-template|tpde"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile|single-run-template"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile|single-run-template|fastisel"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile|single-run-template|tpde"
  # cache=full + spec=recompile
  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile|full"
#  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile|full|fastisel"
#  "duckdb|topdown|expr|none|${GOLDEN_FILE}|recompile|full|tpde"
  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile|full"
#  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile|full|fastisel"
#  "duckdb|topdown|operator|none|${GOLDEN_FILE}|recompile|full|tpde"
  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile|full"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile|full|fastisel"
#  "duckdb|topdown|query|none|${GOLDEN_FILE}|recompile|full|tpde"
)

passed=0
failed=0
total=${#JIT_CONFIGS[@]}

# Known diffs for DSB (create if needed)
KNOWN_DIFFS_NB="known_diffs_dsb_node-based.txt"

# filter_known_diffs <diff_text> <golden_file>
filter_known_diffs() {
  local diff_text="$1" golden="$2"
  if [[ "$golden" != *"node-based"* ]] || [[ ! -f "$KNOWN_DIFFS_NB" ]]; then
    echo "$diff_text"
    return
  fi
  local known_lines
  known_lines=$(grep -v '^#' "$KNOWN_DIFFS_NB" | grep -v '^$' || true)
  if [[ -z "$known_lines" ]]; then
    echo "$diff_text"
    return
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

# Generate golden file for this SF if it doesn't exist yet (vanilla: no-split + no-jit).
if [[ ! -f "$GOLDEN_FILE" ]]; then
    echo "=== Generating golden file for SF=${DSB_SF}: ${GOLDEN_FILE} ==="
    mkdir -p "${result_dir}"
    bash run_aqp.sh "dsb_${DSB_SF}" duckdb none none none on on on on off off llvm
    vanilla_output="${result_dir}/aqp_middleware_duckdb_none_none_none_dsb.txt"
    if [[ ! -f "$vanilla_output" ]]; then
        echo "FATAL: vanilla run produced no output at ${vanilla_output}"
        exit 1
    fi
    eval $FILTER "$vanilla_output" > "$GOLDEN_FILE"
    echo "  Golden file created: ${GOLDEN_FILE} ($(wc -l < "$GOLDEN_FILE") lines)"
fi

# Clear disk JIT cache to ensure clean slate
rm -rf /dev/shm/aqp_jit_cache/
declare -a FAILED_CONFIGS=()
FAIL_LOG="${result_dir}/correctness_failures.log"
mkdir -p "${result_dir}"
: > "$FAIL_LOG"

# --- Run JIT-level configs via run_aqp.sh dsb_${DSB_SF} ---
for entry in "${JIT_CONFIGS[@]}"; do
  IFS='|' read -r engine split jit_level jit_simd golden spec_jit_mode jit_cache_mode compile_mode skip_hash_cmp <<< "$entry"
  spec_jit_mode=${spec_jit_mode:-off}
  jit_cache_mode=${jit_cache_mode:-off}
  compile_mode=${compile_mode:-llvm}
  skip_hash_cmp=${skip_hash_cmp:-on}
  echo "=== Testing: engine=${engine} split=${split} jit=${jit_level} simd=${jit_simd} compile=${compile_mode} spec=${spec_jit_mode} cache=${jit_cache_mode} skip_hash_cmp=${skip_hash_cmp} ==="

  bash run_aqp.sh "dsb_${DSB_SF}" "${engine}" "${split}" "${jit_level}" "${jit_simd}" \
       on on on "${skip_hash_cmp}" "${jit_cache_mode}" "${spec_jit_mode}" "${compile_mode}"

  shc_suffix=""
  [[ "$skip_hash_cmp" == "off" ]] && shc_suffix="_noskiphashcmp"
  spec_suffix=""
  [[ "$spec_jit_mode" != "off" ]] && spec_suffix="_spec${spec_jit_mode}"
  cache_suffix=""
  if [[ "$jit_cache_mode" == "on" ]]; then
    cache_suffix="_jitcache"
  elif [[ "$jit_cache_mode" != "off" ]]; then
    cache_suffix="_jitcache_${jit_cache_mode//-/_}"
  fi
  fc_suffix=""
  [[ "$compile_mode" != "llvm" ]] && fc_suffix="_${compile_mode}"
  if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    output="${result_dir}/aqp_middleware_${engine}_${jit_level}_${split}_dsb.txt"
  else
    output="${result_dir}/aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${shc_suffix}${cache_suffix}${spec_suffix}${fc_suffix}_dsb.txt"
  fi

  config_label="engine=${engine} split=${split} jit=${jit_level} simd=${jit_simd} compile=${compile_mode} spec=${spec_jit_mode} cache=${jit_cache_mode} skip_hash_cmp=${skip_hash_cmp}"

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

  # For --jit-cache=full (--repeat=2): compare BOTH iterations against golden.
  if [[ "$jit_cache_mode" == "full" ]]; then
    d0_raw=$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    d1_raw=$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)
    d0=$(filter_known_diffs "$d0_raw" "$golden")
    d1=$(filter_known_diffs "$d1_raw" "$golden")
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0 + iter1)"
      ((passed++))
    else
      echo "  FAIL: differences found"
      [[ -n "$d0" ]] && echo "  iter0 diff:" && echo "$d0" | head -10
      [[ -n "$d1" ]] && echo "  iter1 diff:" && echo "$d1" | head -10
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "iter0 diff:" >> "$FAIL_LOG" && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "iter1 diff:" >> "$FAIL_LOG" && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  else
    d_raw=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
    d=$(filter_known_diffs "$d_raw" "$golden")
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
done

# --- Per-subquery tune-config correctness for node-based (if JSON exists) ---
TUNE_JSON="${result_dir}/tuned_cross_split_duckdb.json"
if [[ -f "$TUNE_JSON" ]]; then
  echo "=== Testing: per-subquery tune-config (auto, spec-jit off) ==="
  ((total++))
  golden="${GOLDEN_FILE}"

  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on off off llvm "$TUNE_JSON"

  config_label="tune-config auto spec=off"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on single-run-strict off llvm "$TUNE_JSON"
  config_label="tune-config auto cache=strict spec=off"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_strict_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on single-run-template off llvm "$TUNE_JSON"
  config_label="tune-config auto cache=template spec=off"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_template_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
#       on on on on full off llvm "$TUNE_JSON"
  config_label="tune-config auto cache=full spec=off"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_full_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d0=$(filter_known_diffs "$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)" "$golden")
    d1=$(filter_known_diffs "$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0 + iter1)"
      ((passed++))
    else
      echo "  FAIL: differences found"
      [[ -n "$d0" ]] && echo "  iter0 diff:" && echo "$d0" | head -10
      [[ -n "$d1" ]] && echo "  iter1 diff:" && echo "$d1" | head -10
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "iter0 diff:" >> "$FAIL_LOG" && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "iter1 diff:" >> "$FAIL_LOG" && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + spec-jit=recompile
  echo "=== Testing: per-subquery tune-config (auto, spec-jit=recompile) ==="
  ((total++))
  config_label="tune-config auto spec=recompile"

  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on off recompile llvm "$TUNE_JSON"

  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_specrecompile_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on single-run-strict recompile llvm "$TUNE_JSON"
  config_label="tune-config auto cache=strict spec=recompile"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_strict_specrecompile_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on single-run-template recompile llvm "$TUNE_JSON"
  config_label="tune-config auto cache=template spec=recompile"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_template_specrecompile_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
#       on on on on full recompile llvm "$TUNE_JSON"
  config_label="tune-config auto cache=full spec=recompile"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_full_specrecompile_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d0=$(filter_known_diffs "$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)" "$golden")
    d1=$(filter_known_diffs "$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0 + iter1)"
      ((passed++))
    else
      echo "  FAIL: differences found"
      [[ -n "$d0" ]] && echo "  iter0 diff:" && echo "$d0" | head -10
      [[ -n "$d1" ]] && echo "  iter1 diff:" && echo "$d1" | head -10
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "iter0 diff:" >> "$FAIL_LOG" && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "iter1 diff:" >> "$FAIL_LOG" && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""
else
  echo "(skipping tune-config test: $TUNE_JSON not found)"
fi

# --- Per-subquery tune-config correctness for topdown (if JSON exists) ---
TUNE_JSON_TD="${result_dir}/tuned_cross_split_duckdb.json"
if [[ -f "$TUNE_JSON_TD" ]]; then
  golden="${GOLDEN_FILE}"

  # Tune + spec=off, cache=off
  echo "=== Testing: per-subquery tune-config (auto, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on off off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto spec=off"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on single-run-strict off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=strict spec=off"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_strict_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on single-run-template off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=template spec=off"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_template_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
#       on on on on full off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=full spec=off"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_full_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d0=$(filter_known_diffs "$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)" "$golden")
    d1=$(filter_known_diffs "$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0 + iter1)"
      ((passed++))
    else
      echo "  FAIL: differences found"
      [[ -n "$d0" ]] && echo "  iter0 diff:" && echo "$d0" | head -10
      [[ -n "$d1" ]] && echo "  iter1 diff:" && echo "$d1" | head -10
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "iter0 diff:" >> "$FAIL_LOG" && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "iter1 diff:" >> "$FAIL_LOG" && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + spec-jit=recompile, cache=off
  echo "=== Testing: per-subquery tune-config (auto, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on off recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto spec=recompile"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_specrecompile_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on single-run-strict recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=strict spec=recompile"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_strict_specrecompile_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
       on on on on single-run-template recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=template spec=recompile"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_template_specrecompile_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d=$(filter_known_diffs "$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d" ]]; then
      echo "  PASS"
      ((passed++))
    else
      echo "  FAIL: differences found"
      echo "$d" | head -20
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      echo "$d" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh "dsb_${DSB_SF}" duckdb auto query none \
#       on on on on full recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=full spec=recompile"
  output="${result_dir}/aqp_middleware_duckdb_auto_query_none_jitcache_full_specrecompile_tuned_dsb.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    ((failed++))
  else
    d0=$(filter_known_diffs "$(diff <(sed -n '/^--- Iteration 0 ---$/,/^--- Iteration 1 ---$/{ /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)" "$golden")
    d1=$(filter_known_diffs "$(diff <(sed -n '/^--- Iteration 1 ---$/,$ { /^--- Iteration/d; p; }' "$output" | eval $FILTER) <(eval $FILTER "$golden") || true)" "$golden")
    if [[ -z "$d0" && -z "$d1" ]]; then
      echo "  PASS (iter0 + iter1)"
      ((passed++))
    else
      echo "  FAIL: differences found"
      [[ -n "$d0" ]] && echo "  iter0 diff:" && echo "$d0" | head -10
      [[ -n "$d1" ]] && echo "  iter1 diff:" && echo "$d1" | head -10
      FAILED_CONFIGS+=("$config_label")
      echo "--- $config_label ---" >> "$FAIL_LOG"
      [[ -n "$d0" ]] && echo "iter0 diff:" >> "$FAIL_LOG" && echo "$d0" >> "$FAIL_LOG"
      [[ -n "$d1" ]] && echo "iter1 diff:" >> "$FAIL_LOG" && echo "$d1" >> "$FAIL_LOG"
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""
else
  echo "(skipping topdown tune-config test: $TUNE_JSON_TD not found)"
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
