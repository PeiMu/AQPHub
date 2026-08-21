#!/usr/bin/env bash
#
# Correctness check: run configs across splits, JIT levels, compile modes,
# and cache modes, then diff against golden files.
# Usage: bash correctness_test.sh
#
# Grouped by: jit_level → compile_mode → split × simd.
# Cache tiers grouped after base configs.
#
# Config format:
#   engine|split|jit_level|jit_simd|golden[|spec_jit[|jit_cache[|compile_mode[|skip_hash_cmp[|disable_runtime_opts]]]]]
#   compile_mode defaults to llvm, skip_hash_cmp defaults to on(=all) when omitted.
#   disable_runtime_opts: comma-separated list of runtime opts to disable
#   (range-pred,bloom-filter,range-guard,block-skip,membership,early-term). Empty = all enabled.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^duckdb runs:|^lingodb runs:|\(base\)|^\[AQP|^\[LingoDB|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration"'

# --- JIT-level configs (no kernel path) ---
JIT_CONFIGS=(
  # ============================================================
  # Interpreter baseline (no JIT)
  # ============================================================
  "duckdb|none|none|none|duckdb_job_no-split_golden.txt"
  #  "duckdb|node-based|none|none|duckdb_job_node-based_golden.txt"
  #"duckdb|relationship-center|none|none|duckdb_job_relationship-center_golden.txt"

  # ============================================================
  # expr-jit, compile_mode=llvm (default)
  # ============================================================
  "duckdb|none|expr|none|duckdb_job_no-split_golden.txt"
  #"duckdb|none|expr|auto|duckdb_job_no-split_golden.txt"
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt"
  #"duckdb|node-based|expr|auto|duckdb_job_node-based_golden.txt"
  #"duckdb|relationship-center|expr|none|duckdb_job_relationship-center_golden.txt"
  #"duckdb|relationship-center|expr|auto|duckdb_job_relationship-center_golden.txt"

  # expr-jit, compile_mode=fastisel
#  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|||fastisel"
  #"duckdb|node-based|expr|auto|duckdb_job_node-based_golden.txt|||fastisel"

  # expr-jit, compile_mode=tpde (no simd=auto — TPDE falls back to scalar)
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|||tpde"
  # expr-jit, none-split, fastisel/tpde
#  "duckdb|none|expr|none|duckdb_job_no-split_golden.txt|||fastisel"
  "duckdb|none|expr|none|duckdb_job_no-split_golden.txt|||tpde"

  # expr-jit spec-jit (node-based, llvm)
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile"
  #"duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|interpret"
  # expr-jit spec-jit, fastisel/tpde
#  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile||fastisel"
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile||tpde"

  # ============================================================
  # operator-jit, compile_mode=llvm
  # ============================================================
  "duckdb|none|operator|none|duckdb_job_no-split_golden.txt"
  #"duckdb|none|operator|auto|duckdb_job_no-split_golden.txt"
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt"
  #"duckdb|node-based|operator|auto|duckdb_job_node-based_golden.txt"
  #"duckdb|relationship-center|operator|none|duckdb_job_relationship-center_golden.txt"
  #"duckdb|relationship-center|operator|auto|duckdb_job_relationship-center_golden.txt"

  # operator-jit, compile_mode=fastisel
#  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|||fastisel"
  #"duckdb|node-based|operator|auto|duckdb_job_node-based_golden.txt|||fastisel"

  # operator-jit, compile_mode=tpde (no simd=auto)
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|||tpde"
  # operator-jit, none-split, fastisel/tpde
#  "duckdb|none|operator|none|duckdb_job_no-split_golden.txt|||fastisel"
  "duckdb|none|operator|none|duckdb_job_no-split_golden.txt|||tpde"

  # operator-jit spec-jit (node-based, llvm)
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile"
  #"duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|interpret"
  # operator-jit spec-jit, fastisel/tpde
#  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile||fastisel"
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile||tpde"

  # ============================================================
  # pipeline-jit (deprioritized — commented out)
  # ============================================================
  #"duckdb|none|pipeline|none|duckdb_job_no-split_golden.txt"
  #"duckdb|none|pipeline|auto|duckdb_job_no-split_golden.txt"
  #"duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt"
  #"duckdb|node-based|pipeline|auto|duckdb_job_node-based_golden.txt"
  #"duckdb|relationship-center|pipeline|none|duckdb_job_relationship-center_golden.txt"
  #"duckdb|relationship-center|pipeline|auto|duckdb_job_relationship-center_golden.txt"
  #"duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt|recompile"
  #"duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt|interpret"
  #"duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt|||fastisel"
  #"duckdb|node-based|pipeline|auto|duckdb_job_node-based_golden.txt|||fastisel"
  #"duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt|||tpde"

  # ============================================================
  # query-jit, compile_mode=llvm
  # ============================================================
  "duckdb|none|query|none|duckdb_job_no-split_golden.txt"
  #"duckdb|none|query|auto|duckdb_job_no-split_golden.txt"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt"
  #"duckdb|node-based|query|auto|duckdb_job_node-based_golden.txt"
  #"duckdb|relationship-center|query|none|duckdb_job_relationship-center_golden.txt"
  #"duckdb|relationship-center|query|auto|duckdb_job_relationship-center_golden.txt"
  # spec-jit (query, llvm)
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile"
  #"duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|interpret"
  #"duckdb|relationship-center|query|none|duckdb_job_relationship-center_golden.txt|recompile"
  #"duckdb|none|query|none|duckdb_job_no-split_golden.txt|recompile"

  # query-jit, compile_mode=fastisel
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|||fastisel"
  #"duckdb|node-based|query|auto|duckdb_job_node-based_golden.txt|||fastisel"

  # query-jit, compile_mode=tpde (no simd=auto)
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|||tpde"
  # query-jit, none-split, fastisel/tpde
#  "duckdb|none|query|none|duckdb_job_no-split_golden.txt|||fastisel"
  "duckdb|none|query|none|duckdb_job_no-split_golden.txt|||tpde"
  # query-jit spec-jit, fastisel/tpde
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile||fastisel"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile||tpde"

  # ============================================================
  # query-jit, skip_hash_cmp=off (per-subquery tuning alternative,
  # see breakdown_measurement_script_shc_off.sh)
  # ============================================================
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt||||off"
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|||fastisel|off"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|||tpde|off"
  "duckdb|none|query|none|duckdb_job_no-split_golden.txt||||off"
  # skip_hash_cmp=off + spec-jit / cache tiers (cache keys embed the mode)
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile|||off"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|single-run-strict||off"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|single-run-template||off"
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|full||off"

  # ============================================================
  # jit-cache=single-run-strict
  # ============================================================
  # expr-jit (llvm)
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|off|single-run-strict"
  #"duckdb|node-based|expr|auto|duckdb_job_node-based_golden.txt|off|single-run-strict"
  # expr-jit (fastisel/tpde)
#  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|off|single-run-strict|fastisel"
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|off|single-run-strict|tpde"
  # operator-jit (llvm)
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|off|single-run-strict"
  #"duckdb|node-based|operator|auto|duckdb_job_node-based_golden.txt|off|single-run-strict"
  # operator-jit (fastisel/tpde)
#  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|off|single-run-strict|fastisel"
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|off|single-run-strict|tpde"
  # pipeline-jit (deprioritized)
  #"duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt|off|single-run-strict"
  #"duckdb|node-based|pipeline|auto|duckdb_job_node-based_golden.txt|off|single-run-strict"
  # query-jit (llvm)
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|single-run-strict"
  #"duckdb|node-based|query|auto|duckdb_job_node-based_golden.txt|off|single-run-strict"
  # query-jit (fastisel/tpde)
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|single-run-strict|fastisel"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|single-run-strict|tpde"
  # strict + spec-jit=recompile (llvm)
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile|single-run-strict"
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile|single-run-strict"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile|single-run-strict"
  #"duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|interpret|single-run-strict"
  # strict + spec-jit=recompile (fastisel/tpde)
#  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile|single-run-strict|fastisel"
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile|single-run-strict|tpde"
#  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile|single-run-strict|fastisel"
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile|single-run-strict|tpde"
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile|single-run-strict|fastisel"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile|single-run-strict|tpde"
  #"duckdb|node-based|query|auto|duckdb_job_node-based_golden.txt|recompile|single-run-strict"
  #"duckdb|node-based|query|auto|duckdb_job_node-based_golden.txt|interpret|single-run-strict"

  # ============================================================
  # jit-cache=single-run-template
  # ============================================================
  # expr-jit (llvm)
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|off|single-run-template"
  #"duckdb|node-based|expr|auto|duckdb_job_node-based_golden.txt|off|single-run-template"
  # expr-jit (fastisel/tpde)
#  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|off|single-run-template|fastisel"
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|off|single-run-template|tpde"
  # operator-jit (llvm)
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|off|single-run-template"
  #"duckdb|node-based|operator|auto|duckdb_job_node-based_golden.txt|off|single-run-template"
  # operator-jit (fastisel/tpde)
#  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|off|single-run-template|fastisel"
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|off|single-run-template|tpde"
  # pipeline-jit (deprioritized)
  #"duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt|off|single-run-template"
  #"duckdb|node-based|pipeline|auto|duckdb_job_node-based_golden.txt|off|single-run-template"
  # query-jit (llvm)
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|single-run-template"
  #"duckdb|node-based|query|auto|duckdb_job_node-based_golden.txt|off|single-run-template"
  # query-jit (fastisel/tpde)
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|single-run-template|fastisel"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|single-run-template|tpde"
  # template + spec-jit=recompile (llvm)
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile|single-run-template"
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile|single-run-template"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile|single-run-template"
  #"duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|interpret|single-run-template"
  # template + spec-jit=recompile (fastisel/tpde)
#  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile|single-run-template|fastisel"
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile|single-run-template|tpde"
#  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile|single-run-template|fastisel"
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile|single-run-template|tpde"
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile|single-run-template|fastisel"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile|single-run-template|tpde"
  #"duckdb|node-based|query|auto|duckdb_job_node-based_golden.txt|recompile|single-run-template"
  #"duckdb|node-based|query|auto|duckdb_job_node-based_golden.txt|interpret|single-run-template"

  # ============================================================
  # jit-cache=full
  # ============================================================
  # expr-jit (llvm)
  "duckdb|none|expr|none|duckdb_job_no-split_golden.txt|off|full"
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|off|full"
  #"duckdb|node-based|expr|auto|duckdb_job_node-based_golden.txt|off|full"
  # expr-jit (fastisel/tpde)
#  "duckdb|none|expr|none|duckdb_job_no-split_golden.txt|off|full|fastisel"
#  "duckdb|none|expr|none|duckdb_job_no-split_golden.txt|off|full|tpde"
#  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|off|full|fastisel"
#  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|off|full|tpde"
  # operator-jit (llvm)
  "duckdb|none|operator|none|duckdb_job_no-split_golden.txt|off|full"
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|off|full"
  #"duckdb|node-based|operator|auto|duckdb_job_node-based_golden.txt|off|full"
  # operator-jit (fastisel/tpde)
#  "duckdb|none|operator|none|duckdb_job_no-split_golden.txt|off|full|fastisel"
#  "duckdb|none|operator|none|duckdb_job_no-split_golden.txt|off|full|tpde"
#  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|off|full|fastisel"
#  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|off|full|tpde"
  # pipeline-jit (llvm)
  #"duckdb|none|pipeline|none|duckdb_job_no-split_golden.txt|off|full"
  #"duckdb|none|pipeline|auto|duckdb_job_no-split_golden.txt|off|full"
  #"duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt|off|full"
  #"duckdb|node-based|pipeline|auto|duckdb_job_node-based_golden.txt|off|full"
  # query-jit (llvm)
  "duckdb|none|query|none|duckdb_job_no-split_golden.txt|off|full"
  #"duckdb|none|query|auto|duckdb_job_no-split_golden.txt|off|full"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|full"
  #"duckdb|node-based|query|auto|duckdb_job_node-based_golden.txt|off|full"
  # query-jit (fastisel/tpde)
#  "duckdb|none|query|none|duckdb_job_no-split_golden.txt|off|full|fastisel"
#  "duckdb|none|query|none|duckdb_job_no-split_golden.txt|off|full|tpde"
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|full|fastisel"
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|off|full|tpde"
  # full + spec-jit=recompile (llvm)
  #  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile|full"
  #  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile|full"
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile|full"
  #"duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|interpret|full"
  # full + spec-jit=recompile (fastisel/tpde)
#  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile|full|fastisel"
#  "duckdb|node-based|expr|none|duckdb_job_node-based_golden.txt|recompile|full|tpde"
#  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile|full|fastisel"
#  "duckdb|node-based|operator|none|duckdb_job_node-based_golden.txt|recompile|full|tpde"
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile|full|fastisel"
#  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|recompile|full|tpde"

  # ============================================================
  # topdown (SDS) — uses the same no-split golden (ground truth)
  # ============================================================
  # no-jit
  "duckdb|topdown|none|none|duckdb_job_no-split_golden.txt"
  # expr-jit (llvm / fastisel / tpde)
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt"
#  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|||fastisel"
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|||tpde"
  # operator-jit (llvm / fastisel / tpde)
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt"
#  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|||fastisel"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|||tpde"
  # query-jit (llvm / fastisel / tpde)
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|||fastisel"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|||tpde"
  # query-jit, skip_hash_cmp=off
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt||||off"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|||fastisel|off"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|||tpde|off"
  # jit-cache=single-run-strict (expr / operator / query x llvm / fastisel / tpde)
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|off|single-run-strict"
#  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|off|single-run-strict|fastisel"
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|off|single-run-strict|tpde"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|off|single-run-strict"
#  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|off|single-run-strict|fastisel"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|off|single-run-strict|tpde"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|single-run-strict"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|single-run-strict|fastisel"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|single-run-strict|tpde"
  # skip_hash_cmp=off + cache tiers
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|single-run-strict||off"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|single-run-template||off"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|full||off"
  # jit-cache=single-run-template (expr / operator / query x llvm / fastisel / tpde)
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|off|single-run-template"
#  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|off|single-run-template|fastisel"
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|off|single-run-template|tpde"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|off|single-run-template"
#  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|off|single-run-template|fastisel"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|off|single-run-template|tpde"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|single-run-template"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|single-run-template|fastisel"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|single-run-template|tpde"
  # jit-cache=full (expr / operator / query x llvm / fastisel / tpde)
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|off|full"
#  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|off|full|fastisel"
#  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|off|full|tpde"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|off|full"
#  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|off|full|fastisel"
#  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|off|full|tpde"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|full"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|full|fastisel"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|off|full|tpde"

  # ============================================================
  # topdown + spec-jit=recompile
  # ============================================================
  # cache=off
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile"
#  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile||fastisel"
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile||tpde"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile"
#  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile||fastisel"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile||tpde"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile||fastisel"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile||tpde"
  # cache=single-run-strict + spec=recompile
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile|single-run-strict"
#  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile|single-run-strict|fastisel"
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile|single-run-strict|tpde"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile|single-run-strict"
#  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile|single-run-strict|fastisel"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile|single-run-strict|tpde"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile|single-run-strict"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile|single-run-strict|fastisel"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile|single-run-strict|tpde"
  # cache=single-run-template + spec=recompile
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile|single-run-template"
#  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile|single-run-template|fastisel"
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile|single-run-template|tpde"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile|single-run-template"
#  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile|single-run-template|fastisel"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile|single-run-template|tpde"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile|single-run-template"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile|single-run-template|fastisel"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile|single-run-template|tpde"
  # cache=full + spec=recompile
  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile|full"
#  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile|full|fastisel"
#  "duckdb|topdown|expr|none|duckdb_job_no-split_golden.txt|recompile|full|tpde"
  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile|full"
#  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile|full|fastisel"
#  "duckdb|topdown|operator|none|duckdb_job_no-split_golden.txt|recompile|full|tpde"
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile|full"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile|full|fastisel"
#  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|recompile|full|tpde"

  # ============================================================
  # Runtime-statistics-guided optimization flags (topdown, query-jit, tpde)
  # 7-step waterfall: enable one opt at a time in order
  # ============================================================
  # baseline: all 6 disabled
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|||tpde||range-pred,early-term,range-guard,block-skip,membership,bloom-filter"
  # +range-pred
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|||tpde||early-term,range-guard,block-skip,membership,bloom-filter"
  # +range-pred,early-term
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|||tpde||range-guard,block-skip,membership,bloom-filter"
  # +range-pred,early-term,range-guard
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|||tpde||block-skip,membership,bloom-filter"
  # +range-pred,early-term,range-guard,block-skip
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|||tpde||membership,bloom-filter"
  # +range-pred,early-term,range-guard,block-skip,membership
  "duckdb|topdown|query|none|duckdb_job_no-split_golden.txt|||tpde||bloom-filter"
  # node-based: baseline (all 6 disabled)
  #  "duckdb|node-based|query|none|duckdb_job_node-based_golden.txt|||tpde||range-pred,early-term,range-guard,block-skip,membership,bloom-filter"

  # ============================================================
  # lingodb / lingo-db-runtime
  # ============================================================
  "lingodb|none|llvm|none|lingodb_job_no-split_golden.txt"
  #  "lingodb|node-based|llvm|none|duckdb_job_node-based_golden.txt"
  "lingodb|none|tpde|none|lingodb_job_no-split_golden.txt"
  #  "lingodb|node-based|tpde|none|duckdb_job_node-based_golden.txt"
  #  "lingo-db-runtime|node-based|llvm|none|duckdb_job_node-based_golden.txt"
  #  "lingo-db-runtime|node-based|tpde|none|duckdb_job_node-based_golden.txt"
  "lingo-db-runtime|none|llvm|none|duckdb_job_no-split_golden.txt"
  "lingo-db-runtime|none|tpde|none|duckdb_job_no-split_golden.txt"
)

# --- Kernel-path configs: engine | split | kernel_path | jit_simd | golden_file ---
KERNEL_CONFIGS=(
  #"duckdb|none|pipeline|none|duckdb_job_no-split_golden.txt"
  #"duckdb|node-based|pipeline|none|duckdb_job_node-based_golden.txt"
  #"duckdb|relationship-center|pipeline|none|duckdb_job_relationship-center_golden.txt"
)

passed=0
failed=0
total=$(( ${#JIT_CONFIGS[@]} + ${#KERNEL_CONFIGS[@]} ))

# Known diffs: lines that non-query-JIT NB configs may produce instead of golden.
# These are pre-existing split-level issues (different MIN() due to join ordering).
KNOWN_DIFFS_NB="known_diffs_node-based.txt"

# filter_known_diffs <diff_text> <golden_file>
# Removes diff hunks where the output line is a known acceptable alternative.
filter_known_diffs() {
  local diff_text="$1" golden="$2"
  if [[ "$golden" != *"node-based"* ]] || [[ ! -f "$KNOWN_DIFFS_NB" ]]; then
    echo "$diff_text"
    return
  fi
  # Build grep pattern from known_diffs file (skip comments and empty lines)
  local known_lines
  known_lines=$(grep -v '^#' "$KNOWN_DIFFS_NB" | grep -v '^$' || true)
  if [[ -z "$known_lines" ]]; then
    echo "$diff_text"
    return
  fi
  # Remove diff hunks where the "<" (output) line matches a known diff
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

# Clear disk JIT cache to ensure clean slate
rm -rf /dev/shm/aqp_jit_cache/
declare -a FAILED_CONFIGS=()
FAIL_LOG="job_result/correctness_failures.log"
: > "$FAIL_LOG"

# --- Run JIT-level configs via run_aqp.sh job ---
for entry in "${JIT_CONFIGS[@]}"; do
  IFS='|' read -r engine split jit_level jit_simd golden spec_jit_mode jit_cache_mode compile_mode skip_hash_cmp disable_runtime_opts <<< "$entry"
  spec_jit_mode=${spec_jit_mode:-off}
  jit_cache_mode=${jit_cache_mode:-off}
  compile_mode=${compile_mode:-llvm}
  skip_hash_cmp=${skip_hash_cmp:-on}
  disable_runtime_opts=${disable_runtime_opts:-}
  echo "=== Testing: engine=${engine} split=${split} jit=${jit_level} simd=${jit_simd} compile=${compile_mode} spec=${spec_jit_mode} cache=${jit_cache_mode} skip_hash_cmp=${skip_hash_cmp} disable_rt=${disable_runtime_opts:-all_on} ==="

  bash run_aqp.sh job "${engine}" "${split}" "${jit_level}" "${jit_simd}" \
       on on on "${skip_hash_cmp}" "${jit_cache_mode}" "${spec_jit_mode}" "${compile_mode}" \
       "" "${disable_runtime_opts}"

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
  rt_suffix=""
  [[ "$disable_runtime_opts" == *"range-pred"* ]]   && rt_suffix+="_norangepred"
  [[ "$disable_runtime_opts" == *"bloom-filter"* ]]  && rt_suffix+="_nobloomfilt"
  [[ "$disable_runtime_opts" == *"range-guard"* ]]   && rt_suffix+="_norangeguard"
  [[ "$disable_runtime_opts" == *"block-skip"* ]]    && rt_suffix+="_noblockskip"
  [[ "$disable_runtime_opts" == *"membership"* ]]    && rt_suffix+="_nomembership"
  [[ "$disable_runtime_opts" == *"early-term"* ]]    && rt_suffix+="_noearlyterm"
  if [[ "$engine" == "lingodb" || "$engine" == "lingo-db-runtime" ]]; then
    output="job_result/aqp_middleware_${engine}_${jit_level}_${split}_job.txt"
  else
    output="job_result/aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${shc_suffix}${cache_suffix}${spec_suffix}${fc_suffix}${rt_suffix}_job.txt"
  fi

  config_label="engine=${engine} split=${split} jit=${jit_level} simd=${jit_simd} compile=${compile_mode} spec=${spec_jit_mode} cache=${jit_cache_mode} skip_hash_cmp=${skip_hash_cmp} disable_rt=${disable_runtime_opts:-all_on}"

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
  # Iter 0 = normal path, iter 1 = replay path. Both must match.
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

# --- Run kernel-path configs via run_job_kernel.sh (no unified equivalent yet) ---
for entry in "${KERNEL_CONFIGS[@]}"; do
  IFS='|' read -r engine split kernel_path jit_simd golden <<< "$entry"
  echo "=== Testing: engine=${engine} split=${split} kernel-path=${kernel_path} ==="

  bash run_job_kernel.sh "${engine}" "${split}" "${kernel_path}" "${jit_simd}" \
       on on on on on off

  output="job_result/aqp_middleware_${engine}_${split}_kernel-${kernel_path}_${jit_simd}_job.txt"
  config_label="engine=${engine} split=${split} kernel-path=${kernel_path} simd=${jit_simd}"

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
    echo "" >> "$FAIL_LOG"
    ((failed++))
  fi
  echo ""
done

# --- Per-subquery tune-config correctness (if JSON exists) ---
TUNE_JSON="job_result/tuned_cross_split_duckdb.json"
if [[ -f "$TUNE_JSON" ]]; then
  echo "=== Testing: per-subquery tune-config (auto/cross-split, spec-jit off) ==="
  ((total++))
  golden="duckdb_job_node-based_golden.txt"

  bash run_aqp.sh job duckdb auto query none \
       on on on on off off llvm "$TUNE_JSON"

  config_label="tune-config auto spec=off"
  output="job_result/aqp_middleware_duckdb_auto_query_none_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
       on on on on single-run-strict off llvm "$TUNE_JSON"
  config_label="tune-config auto cache=strict spec=off"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_strict_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
       on on on on single-run-template off llvm "$TUNE_JSON"
  config_label="tune-config auto cache=template spec=off"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_template_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
#       on on on on full off llvm "$TUNE_JSON"
  config_label="tune-config auto cache=full spec=off"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_full_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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

  bash run_aqp.sh job duckdb auto query none \
       on on on on off recompile llvm "$TUNE_JSON"

  output="job_result/aqp_middleware_duckdb_auto_query_none_specrecompile_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
       on on on on single-run-strict recompile llvm "$TUNE_JSON"
  config_label="tune-config auto cache=strict spec=recompile"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_strict_specrecompile_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
       on on on on single-run-template recompile llvm "$TUNE_JSON"
  config_label="tune-config auto cache=template spec=recompile"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_template_specrecompile_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
#       on on on on full recompile llvm "$TUNE_JSON"
  config_label="tune-config auto cache=full spec=recompile"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_full_specrecompile_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
TUNE_JSON_TD="job_result/tuned_cross_split_duckdb.json"
if [[ -f "$TUNE_JSON_TD" ]]; then
  golden="duckdb_job_no-split_golden.txt"

  # Tune + spec=off, cache=off
  echo "=== Testing: per-subquery tune-config (auto, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
       on on on on off off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto spec=off"
  output="job_result/aqp_middleware_duckdb_auto_query_none_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
       on on on on single-run-strict off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=strict spec=off"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_strict_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
       on on on on single-run-template off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=template spec=off"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_template_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=off
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit off) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
#       on on on on full off llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=full spec=off"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_full_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
  bash run_aqp.sh job duckdb auto query none \
       on on on on off recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto spec=recompile"
  output="job_result/aqp_middleware_duckdb_auto_query_none_specrecompile_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-strict, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=strict, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
       on on on on single-run-strict recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=strict spec=recompile"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_strict_specrecompile_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=single-run-template, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=template, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
       on on on on single-run-template recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=template spec=recompile"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_single_run_template_specrecompile_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
      echo "" >> "$FAIL_LOG"
      ((failed++))
    fi
  fi
  echo ""

  # Tune + cache=full, spec=recompile
  echo "=== Testing: per-subquery tune-config (auto, cache=full, spec-jit=recompile) ==="
  ((total++))
  bash run_aqp.sh job duckdb auto query none \
#       on on on on full recompile llvm "$TUNE_JSON_TD"
  config_label="tune-config auto cache=full spec=recompile"
  output="job_result/aqp_middleware_duckdb_auto_query_none_jitcache_full_specrecompile_tuned_job.txt"
  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    FAILED_CONFIGS+=("$config_label  [output missing: $output]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "output file not found: $output" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
    ((failed++))
  elif [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    FAILED_CONFIGS+=("$config_label  [golden missing: $golden]")
    echo "--- $config_label ---" >> "$FAIL_LOG"
    echo "golden file not found: $golden" >> "$FAIL_LOG"
    echo "" >> "$FAIL_LOG"
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
