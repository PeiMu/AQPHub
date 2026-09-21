#!/usr/bin/env bash
#
# Full correctness sweep: verify all configs from main_breakdowns.sh.
# Run from measure/ directory.
#
# Covers (only configs active in main_breakdowns.sh, excluding Umbra):
#   - RQ1/RQ2: DuckDB JOB, PG JOB, DuckDB DSB-50, PG DSB-50
#   - RQ3.1: Stats collection overhead  (job duckdb topdown)
#   - RQ3.2: Runtime-guided optimizations waterfall (job duckdb topdown)
#   - RQ4:   Compiler backend / cache / latency-hiding (job duckdb topdown + node-based)
#   - RQ5:   Bi-directional storage (job duckdb topdown)
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

total_passed=0
total_failed=0
declare -a SUITE_RESULTS=()


# =============================================================
# Shared correctness infrastructure for inline configs
# =============================================================

FILTER='grep -v -E "^Running|^==|^Execution|^$|^waiting|^server|^ANALYZ|^duckdb runs:|^lingodb runs:|\(base\)|^\[AQP|^\[DuckDB\]|^\[LingoDB|^\[Storage|^\[CSR|^\[Dim|^\[RelationshipCenter|^\[IRQuerySplitter|^  [a-z_]*: [0-9]* rows$|^Found [0-9]|^Run |^Passed:|^Failed:|^Total |^Benchmark|^Average|^--- Iteration|^Test FAILED|^Error:|^warning:|^Do not support yet|^same engine|falling back to"'

KNOWN_DIFFS_NB="known_diffs_node-based.txt"
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

inline_passed=0
inline_failed=0
inline_total=0
declare -a INLINE_FAILED_CONFIGS=()
INLINE_FAIL_LOG="inline_correctness_failures.log"
: > "$INLINE_FAIL_LOG"

rm -rf /dev/shm/aqp_jit_cache/

# run_inline_config <label> <golden> <result_dir> <log_suffix>
#   <bench> <engine> <split> <jit_level> <jit_simd>
#   <skip_hash_cmp> <jit_cache> <spec_jit> <compile_mode>
#   <tune_config> <disable_runtime_opts> <disable_compile_opts> <collect_stats>
#
# run_aqp.sh arg order (after fix):
#   1=bench 2=engine 3=split 4=jit_level 5=jit_simd
#   6=payload_prune 7=prefetch 8=batch_probe 9=skip_hash_cmp
#   10=jit_cache 11=spec_jit 12=compile_mode
#   13=tune_config 14=disable_runtime_opts 15=disable_compile_opts 16=collect_stats
run_inline_config() {
  local label="$1" golden="$2" result_dir="$3" log_suffix="$4"
  shift 4
  local bench="$1" engine="$2" split="$3" jit_level="$4" jit_simd="$5"
  local skip_hash_cmp="$6" jit_cache="$7" spec_jit="$8" compile_mode="$9"
  local tune_config="${10}" disable_runtime_opts="${11}"
  local disable_compile_opts="${12}" collect_stats="${13}"

  ((inline_total++))
  echo "=== ${label} ==="

  bash run_aqp.sh "${bench}" "${engine}" "${split}" "${jit_level}" "${jit_simd}" \
       on on on "${skip_hash_cmp}" "${jit_cache}" "${spec_jit}" "${compile_mode}" \
       "${tune_config}" "${disable_runtime_opts}" "${disable_compile_opts}" "${collect_stats}"

  # Build output filename (mirrors run_aqp.sh suffix logic)
  local shc_suffix="" spec_suffix="" cache_suffix="" fc_suffix=""
  local rt_suffix="" cc_suffix="" cs_suffix=""
  [[ "$skip_hash_cmp" == "off" ]] && shc_suffix="_noskiphashcmp"
  [[ "$spec_jit" != "off" ]] && spec_suffix="_spec${spec_jit}"
  if [[ "$jit_cache" == "on" ]]; then
    cache_suffix="_jitcache"
  elif [[ "$jit_cache" != "off" ]]; then
    cache_suffix="_jitcache_${jit_cache//-/_}"
  fi
  [[ "$compile_mode" != "llvm" ]] && fc_suffix="_${compile_mode}"
  [[ "$disable_runtime_opts" == *"range-pred"* ]]   && rt_suffix+="_norangepred"
  [[ "$disable_runtime_opts" == *"bloom-filter"* ]]  && rt_suffix+="_nobloomfilt"
  [[ "$disable_runtime_opts" == *"range-guard"* ]]   && rt_suffix+="_norangeguard"
  [[ "$disable_runtime_opts" == *"block-skip"* ]]    && rt_suffix+="_noblockskip"
  [[ "$disable_runtime_opts" == *"membership"* ]]    && rt_suffix+="_nomembership"
  [[ "$disable_runtime_opts" == *"early-term"* ]]    && rt_suffix+="_noearlyterm"
  [[ "$disable_runtime_opts" == *"disable-bi-directional-storage"* ]] && rt_suffix+="_nobidirstorage"
  [[ "$disable_compile_opts" == *"cross-query-prep"* ]] && cc_suffix+="_nocrossqprep"
  [[ "$collect_stats" == "on" ]]  && cs_suffix="_collectstats"
  [[ "$collect_stats" == "off" ]] && cs_suffix="_nocollectstats"

  local output="${result_dir}/aqp_middleware_${engine}_${split}_${jit_level}_${jit_simd}${shc_suffix}${cache_suffix}${spec_suffix}${fc_suffix}${rt_suffix}${cc_suffix}${cs_suffix}${log_suffix}"

  if [[ ! -f "$output" ]]; then
    echo "  FAIL: output file not found: $output"
    INLINE_FAILED_CONFIGS+=("${label}  [output missing: $output]")
    echo "--- ${label} ---" >> "$INLINE_FAIL_LOG"
    echo "output file not found: $output" >> "$INLINE_FAIL_LOG"
    echo "" >> "$INLINE_FAIL_LOG"
    ((inline_failed++))
    return
  fi
  if [[ ! -f "$golden" ]]; then
    echo "  FAIL: golden file not found: $golden"
    INLINE_FAILED_CONFIGS+=("${label}  [golden missing: $golden]")
    echo "--- ${label} ---" >> "$INLINE_FAIL_LOG"
    echo "golden file not found: $golden" >> "$INLINE_FAIL_LOG"
    echo "" >> "$INLINE_FAIL_LOG"
    ((inline_failed++))
    return
  fi

  local d_raw d
  d_raw=$(diff <(eval $FILTER "$output") <(eval $FILTER "$golden") || true)
  d=$(filter_known_diffs "$d_raw" "$golden")
  if [[ -z "$d" ]]; then
    echo "  PASS"
    ((inline_passed++))
  else
    echo "  FAIL: differences found"
    echo "$d" | head -20
    INLINE_FAILED_CONFIGS+=("${label}")
    echo "--- ${label} ---" >> "$INLINE_FAIL_LOG"
    echo "$d" >> "$INLINE_FAIL_LOG"
    echo "" >> "$INLINE_FAIL_LOG"
    ((inline_failed++))
  fi
  echo ""
}

# =============================================================
# RQ1/RQ2: All benchmark configs from main_breakdowns.sh
# =============================================================

echo ""
echo "############################################################"
echo "# RQ1/RQ2: All benchmark configs from main_breakdowns.sh"
echo "############################################################"
echo ""

# --- Golden files (interpreter no-split baselines, generate if missing) ---

DUCKDB_JOB_NOSPLIT_GOLDEN="duckdb_job_no-split_golden.txt"
if [[ ! -f "$DUCKDB_JOB_NOSPLIT_GOLDEN" ]]; then
  echo "=== Generating golden: DuckDB JOB no-split ==="
  bash run_aqp.sh job duckdb none none none
  cp job_result/aqp_middleware_duckdb_none_none_none_job.txt "${DUCKDB_JOB_NOSPLIT_GOLDEN}"
fi

PG_JOB_NOSPLIT_GOLDEN="pg_job_no-split_golden.txt"
if [[ ! -f "$PG_JOB_NOSPLIT_GOLDEN" ]]; then
  echo "=== Generating golden: PG JOB no-split ==="
  bash run_aqp.sh job postgresql none none none
  cp job_result/aqp_middleware_postgresql_none_none_none_job.txt "${PG_JOB_NOSPLIT_GOLDEN}"
fi

DUCKDB_DSB50_NOSPLIT_GOLDEN="duckdb_dsb_no-split_golden_sf50.txt"
if [[ ! -f "$DUCKDB_DSB50_NOSPLIT_GOLDEN" ]]; then
  echo "=== Generating golden: DuckDB DSB-50 no-split ==="
  bash run_aqp.sh dsb_50 duckdb none none none
  cp dsb_result_sf50/aqp_middleware_duckdb_none_none_none_dsb.txt "${DUCKDB_DSB50_NOSPLIT_GOLDEN}"
fi

PG_DSB50_NOSPLIT_GOLDEN="pg_dsb_no-split_golden_sf50.txt"
if [[ ! -f "$PG_DSB50_NOSPLIT_GOLDEN" ]]; then
  echo "=== Generating golden: PG DSB-50 no-split ==="
  bash run_aqp.sh dsb_50 postgresql none none none
  cp dsb_result_sf50/aqp_middleware_postgresql_none_none_none_dsb.txt "${PG_DSB50_NOSPLIT_GOLDEN}"
fi

# --- Golden files (interpreter node-based baselines, generate if missing) ---

DUCKDB_JOB_NB_GOLDEN="duckdb_job_node-based_golden.txt"

PG_JOB_NB_GOLDEN="pg_job_node-based_golden.txt"
if [[ ! -f "$PG_JOB_NB_GOLDEN" ]]; then
  echo "=== Generating golden: PG JOB node-based ==="
  bash run_aqp.sh job postgresql node-based none none
  cp job_result/aqp_middleware_postgresql_node-based_none_none_job.txt "${PG_JOB_NB_GOLDEN}"
fi

DUCKDB_DSB50_NB_GOLDEN="duckdb_dsb_node-based_golden_sf50.txt"
if [[ ! -f "$DUCKDB_DSB50_NB_GOLDEN" ]]; then
  echo "=== Generating golden: DuckDB DSB-50 node-based ==="
  bash run_aqp.sh dsb_50 duckdb node-based none none
  cp dsb_result_sf50/aqp_middleware_duckdb_node-based_none_none_dsb.txt "${DUCKDB_DSB50_NB_GOLDEN}"
fi

PG_DSB50_NB_GOLDEN="pg_dsb_node-based_golden.txt"
if [[ ! -f "$PG_DSB50_NB_GOLDEN" ]]; then
  echo "=== Generating golden: PG DSB-50 node-based ==="
  bash run_aqp.sh dsb_50 postgresql node-based none none
  cp dsb_result_sf50/aqp_middleware_postgresql_node-based_none_none_dsb.txt "${PG_DSB50_NB_GOLDEN}"
fi

# --- DuckDB JOB (main_breakdowns.sh lines 27-29) ---

run_inline_config "DuckDB JOB none query tpde" "$DUCKDB_JOB_NOSPLIT_GOLDEN" job_result "_job.txt" \
  job duckdb none query none on off off tpde "" "" "" auto

run_inline_config "DuckDB JOB topdown none" "$DUCKDB_JOB_NOSPLIT_GOLDEN" job_result "_job.txt" \
  job duckdb topdown none none on off off llvm "" "" "" auto

run_inline_config "DuckDB JOB topdown query tpde cache=template" "$DUCKDB_JOB_NOSPLIT_GOLDEN" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off tpde "" "" "" auto

# --- PostgreSQL JOB (main_breakdowns.sh lines 39-41) ---

run_inline_config "PG JOB none query tpde" "$PG_JOB_NOSPLIT_GOLDEN" job_result "_job.txt" \
  job postgresql none query none on off off tpde "" "" "" auto

run_inline_config "PG JOB node-based none" "$PG_JOB_NB_GOLDEN" job_result "_job.txt" \
  job postgresql node-based none none on off off llvm "" "" "" auto

run_inline_config "PG JOB node-based query fastisel cache=strict spec=recompile" "$PG_JOB_NB_GOLDEN" job_result "_job.txt" \
  job postgresql node-based query none on single-run-strict recompile fastisel "" "" "" auto

# --- DuckDB DSB-50 (main_breakdowns.sh lines 54-56) ---

run_inline_config "DuckDB DSB-50 none query tpde" "$DUCKDB_DSB50_NOSPLIT_GOLDEN" dsb_result_sf50 "_dsb.txt" \
  dsb_50 duckdb none query none on off off tpde "" "" "" auto

run_inline_config "DuckDB DSB-50 node-based none" "$DUCKDB_DSB50_NB_GOLDEN" dsb_result_sf50 "_dsb.txt" \
  dsb_50 duckdb node-based none none on off off llvm "" "" "" auto

run_inline_config "DuckDB DSB-50 node-based query tpde cache=template" "$DUCKDB_DSB50_NB_GOLDEN" dsb_result_sf50 "_dsb.txt" \
  dsb_50 duckdb node-based query none on single-run-template off tpde "" "" "" auto

# --- PostgreSQL DSB-50 (main_breakdowns.sh lines 60-62) ---

run_inline_config "PG DSB-50 none query tpde" "$PG_DSB50_NOSPLIT_GOLDEN" dsb_result_sf50 "_dsb.txt" \
  dsb_50 postgresql none query none on off off tpde "" "" "" auto

run_inline_config "PG DSB-50 node-based none" "$PG_DSB50_NB_GOLDEN" dsb_result_sf50 "_dsb.txt" \
  dsb_50 postgresql node-based none none on off off llvm "" "" "" auto

run_inline_config "PG DSB-50 node-based query tpde cache=param spec=recompile" "$PG_DSB50_NB_GOLDEN" dsb_result_sf50 "_dsb.txt" \
  dsb_50 postgresql node-based query none on single-run-parameterized recompile tpde "" "" "" auto

# =============================================================
# RQ3.1: Stats collection overhead
# =============================================================

echo ""
echo "############################################################"
echo "# RQ3.1: Stats collection overhead"
echo "############################################################"
echo ""

GOLDEN_NOSPLIT="$DUCKDB_JOB_NOSPLIT_GOLDEN"
GOLDEN_NB="$DUCKDB_JOB_NB_GOLDEN"

# no-jit + collect_stats=on
run_inline_config "RQ3.1 no-jit stats=on" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown none none on off off llvm "" "" "" on

# no-jit + collect_stats=off
run_inline_config "RQ3.1 no-jit stats=off" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown none none on off off llvm "" "" "" off

# query-jit tpde + collect_stats=off
run_inline_config "RQ3.1 query-jit tpde stats=off" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off tpde "" "" "" off

# query-jit tpde + collect_stats=on
run_inline_config "RQ3.1 query-jit tpde stats=on" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off tpde "" "" "" on

# =============================================================
# RQ3.2: Runtime-guided optimization waterfall
# =============================================================

echo ""
echo "############################################################"
echo "# RQ3.2: Runtime-guided optimization waterfall"
echo "############################################################"
echo ""

# Step 1: all disabled
run_inline_config "RQ3.2 step1 all-disabled" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off tpde "" \
  "range-pred,early-term,range-guard,block-skip,membership,bloom-filter" "" auto

# Step 2: +range-pred
run_inline_config "RQ3.2 step2 +range-pred" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off tpde "" \
  "early-term,range-guard,block-skip,membership,bloom-filter" "" auto

# Step 3: +range-pred,range-guard
run_inline_config "RQ3.2 step3 +range-guard" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off tpde "" \
  "early-term,block-skip,membership,bloom-filter" "" auto

# Step 4: +range-pred,range-guard,block-skip
run_inline_config "RQ3.2 step4 +block-skip" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off tpde "" \
  "early-term,membership,bloom-filter" "" auto

# Step 5: all enabled
run_inline_config "RQ3.2 step5 all-enabled" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off tpde "" "" "" auto

# =============================================================
# RQ4: Compiler backend / cache / latency-hiding
# =============================================================

echo ""
echo "############################################################"
echo "# RQ4: Compiler backend / cache / latency-hiding"
echo "############################################################"
echo ""

# --- Figure A: Compiler backends (cache=single-run-template) ---

run_inline_config "RQ4 A1 LLVM" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off llvm "" "" "" auto

run_inline_config "RQ4 A2 FastISel" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off fastisel "" "" "" auto

run_inline_config "RQ4 A3 TPDE" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off tpde "" "" "" auto

# --- Figure B: JIT cache modes ---

# TPDE
run_inline_config "RQ4 B1 cache=off tpde" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on off off tpde "" "" "" auto

run_inline_config "RQ4 B2 cache=strict tpde" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-strict off tpde "" "" "" auto

run_inline_config "RQ4 B3 cache=parameterized tpde" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-parameterized off tpde "" "" "" auto

# B4 = A3 (cache=template tpde — already tested)

# FastISel
run_inline_config "RQ4 B5 cache=off fastisel" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on off off fastisel "" "" "" auto

run_inline_config "RQ4 B6 cache=strict fastisel" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-strict off fastisel "" "" "" auto

run_inline_config "RQ4 B7 cache=parameterized fastisel" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-parameterized off fastisel "" "" "" auto

run_inline_config "RQ4 B8 cache=template fastisel" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off fastisel "" "" "" auto

# LLVM
run_inline_config "RQ4 B9 cache=off llvm" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on off off llvm "" "" "" auto

run_inline_config "RQ4 B10 cache=strict llvm" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-strict off llvm "" "" "" auto

run_inline_config "RQ4 B11 cache=parameterized llvm" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-parameterized off llvm "" "" "" auto

# B12 = A1 (cache=template llvm — already tested)

# --- Figure C: Latency hiding ---

# C1: no hiding tpde (topdown) — cross-query-prep disabled
run_inline_config "RQ4 C1 no-hiding tpde" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off tpde "" "" "cross-query-prep" auto

# C2 = A3 (cross enabled, spec=off — already tested)

# C3: spec+cross tpde (topdown)
run_inline_config "RQ4 C3 spec+cross tpde" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template recompile tpde "" "" "" auto

# C4: no hiding llvm (node-based) — cross-query-prep disabled
run_inline_config "RQ4 C4 no-hiding llvm (node-based)" "$GOLDEN_NB" job_result "_job.txt" \
  job duckdb node-based query none on single-run-template off llvm "" "" "cross-query-prep" auto

# C5: cross-only llvm (node-based)
run_inline_config "RQ4 C5 cross-only llvm (node-based)" "$GOLDEN_NB" job_result "_job.txt" \
  job duckdb node-based query none on single-run-template off llvm "" "" "" auto

# C6: spec+cross llvm (node-based)
run_inline_config "RQ4 C6 spec+cross llvm (node-based)" "$GOLDEN_NB" job_result "_job.txt" \
  job duckdb node-based query none on single-run-template recompile llvm "" "" "" auto

# =============================================================
# RQ5: Bi-directional storage
# =============================================================

echo ""
echo "############################################################"
echo "# RQ5: Bi-directional storage"
echo "############################################################"
echo ""

# Step 1: bi-directional enabled (default) — same as A3/B4, already tested

# Step 2: bi-directional disabled
run_inline_config "RQ5 bidir-disabled" "$GOLDEN_NOSPLIT" job_result "_job.txt" \
  job duckdb topdown query none on single-run-template off tpde "" \
  "disable-bi-directional-storage" "" auto

# =============================================================
# Inline configs summary
# =============================================================
echo ""
echo "############################################################"
echo "# Inline Correctness: ${inline_passed}/${inline_total} passed, ${inline_failed} failed"
echo "############################################################"
if (( ${#INLINE_FAILED_CONFIGS[@]} > 0 )); then
  echo ""
  echo "Failed configs:"
  for i in "${!INLINE_FAILED_CONFIGS[@]}"; do
    echo "  $((i+1)). ${INLINE_FAILED_CONFIGS[$i]}"
  done
  echo ""
  echo "Full diffs saved to: $INLINE_FAIL_LOG"
fi

if (( inline_failed > 0 )); then
  SUITE_RESULTS+=("FAIL  RQ1-RQ5 Inline Configs  (${inline_failed} failures)")
  ((total_failed += inline_failed))
else
  SUITE_RESULTS+=("PASS  RQ1-RQ5 Inline Configs")
fi

# =============================================================
# Overall Summary
# =============================================================
echo ""
echo "############################################################"
echo "# Overall Correctness Summary"
echo "############################################################"
for r in "${SUITE_RESULTS[@]}"; do
  echo "  ${r}"
done
echo ""

if (( total_failed > 0 )); then
  echo "TOTAL FAILURES: ${total_failed}"
  exit 1
else
  echo "ALL SUITES PASSED"
  exit 0
fi
