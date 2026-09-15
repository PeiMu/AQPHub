#!/usr/bin/env bash
#
# Compilation-time reduction breakdown: orthogonal evaluation.
# Target: duckdb topdown query-jit, node-based split.
#
# ============================================================
# Figure A: Compiler backend (cache=off, spec=off, cross=off)
# ============================================================
#
# A1: LLVM O2 (baseline)
# A2: FastISel (LLVM O0 + FastISel)
# A3: TPDE (fastest backend)
#
# ============================================================
# Figure B: JIT cache mode (spec=off, cross=off)
# ============================================================
#
# B1-B4: TPDE backend
#   B1: off (no caching)
#   B2: single-run-strict (exact plan match)
#   B3: single-run-parameterized (parameterized: constants from params array)
#   B4: single-run-template
# B5-B8: FastISel backend (same cache modes as B1-B4)
# B9-B12: LLVM backend (same cache modes as B1-B4)
#
# ============================================================
# Figure C: Latency hiding (cache=template)
# ============================================================
#
# C1: baseline (spec=off, cross=off) tpde
# C2: +cross-query-prep (spec=off, cross=on) tpde
# C3: +spec-jit +cross-query-prep (spec=recompile, cross=on) tpde
# C4: baseline (spec=off, cross=off) llvm
# C5: +cross-query-prep (spec=off, cross=on) llvm
# C6: +spec-jit +cross-query-prep (spec=recompile, cross=on) llvm
#
# ============================================================
# Total configs to measure: 19 (A3=B1 shared)
# ============================================================
#
# Run from measure/ directory.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

DEST_DIR="${SCRIPT_DIR}/job_result"
mkdir -p "${DEST_DIR}"

# Shared args: job duckdb topdown query none on on on all
#   positional: bench engine split jit_level jit_simd payload_prune prefetch batch_probe skip_hash_cmp
BASE="job duckdb topdown query none on on on all"

echo "============================================"
echo "Figure A: Compiler backend comparison"
echo "============================================"

# A1: LLVM O2 
echo "--- A1: LLVM O2 ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off llvm "" "" ""
A1="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_llvm_breakdown_time_log.csv"
cp "$A1" "${DEST_DIR}/figA1_llvm.csv"

# A2: FastISel
echo "--- A2: FastISel ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off fastisel "" "" ""
A2="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_fastisel_breakdown_time_log.csv"
cp "$A2" "${DEST_DIR}/figA2_fastisel.csv"

# A3: TPDE
echo "--- A3: TPDE ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off tpde "" "" ""
A3="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_tpde_breakdown_time_log.csv"
cp "$A3" "${DEST_DIR}/figA3_tpde.csv"

echo ""
echo "============================================"
echo "Figure B: JIT cache mode (TPDE, spec=off)"
echo "============================================"

# B1: off
echo "--- B1: cache=off tpde ---"
bash ./measure_breakdown_time_aqp.sh $BASE off off tpde "" "" ""
B1="${DEST_DIR}/duckdb_topdown_query_none_tpde_breakdown_time_log.csv"
cp "$B1" "${DEST_DIR}/figB1_cache_off_tpde.csv"

# B2: single-run-strict
echo "--- B2: cache=strict tpde ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-strict off tpde "" "" ""
B2="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_strict_tpde_breakdown_time_log.csv"
cp "$B2" "${DEST_DIR}/figB2_cache_strict_tpde.csv"

# B3: single-run-parameterized
echo "--- B3: cache=parameterized tpde ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-parameterized off tpde "" "" ""
B3="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_parameterized_tpde_breakdown_time_log.csv"
cp "$B3" "${DEST_DIR}/figB3_cache_parameterized_tpde.csv"

# B4: single-run-template - Reuse A3
echo "--- B4: cache=template tpde ---"
#bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off tpde "" "" ""
B4="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_tpde_breakdown_time_log.csv"
cp "$B4" "${DEST_DIR}/figB4_cache_template_tpde.csv"

# --- FastISel variants (B5-B8) ---

# B5: off
echo "--- B5: cache=off fastisel ---"
bash ./measure_breakdown_time_aqp.sh $BASE off off fastisel "" "" ""
B5="${DEST_DIR}/duckdb_topdown_query_none_fastisel_breakdown_time_log.csv"
cp "$B5" "${DEST_DIR}/figB5_cache_off_fastisel.csv"

# B6: single-run-strict
echo "--- B6: cache=strict fastisel ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-strict off fastisel "" "" ""
B6="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_strict_fastisel_breakdown_time_log.csv"
cp "$B6" "${DEST_DIR}/figB6_cache_strict_fastisel.csv"

# B7: single-run-parameterized
echo "--- B7: cache=parameterized fastisel ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-parameterized off fastisel "" "" ""
B7="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_parameterized_fastisel_breakdown_time_log.csv"
cp "$B7" "${DEST_DIR}/figB7_cache_parameterized_fastisel.csv"

# B8: single-run-template
echo "--- B8: cache=template fastisel ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off fastisel "" "" ""
B8="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_fastisel_breakdown_time_log.csv"
mv "$B8" "${DEST_DIR}/figB8_cache_template_fastisel.csv"

# --- LLVM variants (B9-B12) ---

# B9: off
echo "--- B9: cache=off llvm ---"
bash ./measure_breakdown_time_aqp.sh $BASE off off llvm "" "" ""
B9="${DEST_DIR}/duckdb_topdown_query_none_llvm_breakdown_time_log.csv"
cp "$B9" "${DEST_DIR}/figB9_cache_off_llvm.csv"

# B10: single-run-strict
echo "--- B10: cache=strict llvm ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-strict off llvm "" "" ""
B10="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_strict_llvm_breakdown_time_log.csv"
cp "$B10" "${DEST_DIR}/figB10_cache_strict_llvm.csv"

# B11: single-run-parameterized
echo "--- B11: cache=parameterized llvm ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-parameterized off llvm "" "" ""
B11="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_parameterized_llvm_breakdown_time_log.csv"
cp "$B11" "${DEST_DIR}/figB11_cache_parameterized_llvm.csv"

# B12: single-run-template
echo "--- B12: cache=template llvm ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off llvm "" "" ""
B12="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_llvm_breakdown_time_log.csv"
mv "$B12" "${DEST_DIR}/figB12_cache_template_llvm.csv"


echo ""
echo "============================================"
echo "Figure C: Latency hiding (cache=template)"
echo "  TPDE: topdown, LLVM: node-based"
echo "============================================"

### TPDE (topdown)
# C1: no hiding (spec=off, cross=off)
echo "--- C1: no hiding tpde (topdown) ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off tpde "" "" "cross-query-prep"
mv "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_tpde_nocrossqprep_breakdown_time_log.csv" \
   "${DEST_DIR}/figC1_no_hiding_tpde.csv"

# C2: cross only (spec=off, cross=on)
echo "--- C2: cross only tpde (topdown) ---"
#bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off tpde "" "" ""
cp "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_tpde_breakdown_time_log.csv" \
   "${DEST_DIR}/figC2_cross_only_tpde.csv"

# C3: spec+cross (spec=recompile, cross=on)
echo "--- C3: spec+cross tpde (topdown) ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template recompile tpde "" "" ""
mv "${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_specrecompile_tpde_breakdown_time_log.csv" \
   "${DEST_DIR}/figC3_spec_and_cross_tpde.csv"

node_based_BASE="job duckdb node-based query none on on on all"

### LLVM (node-based)
# C4: no hiding (spec=off, cross=off)
echo "--- C4: no hiding llvm (node-based) ---"
bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template off llvm "" "" "cross-query-prep"
mv "${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_llvm_nocrossqprep_breakdown_time_log.csv" \
   "${DEST_DIR}/figC4_no_hiding_llvm.csv"

# C5: cross only (spec=off, cross=on)
echo "--- C5: cross only llvm (node-based) ---"
bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template off llvm "" "" ""
mv "${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_llvm_breakdown_time_log.csv" \
   "${DEST_DIR}/figC5_cross_only_llvm.csv"

# C6: spec+cross (spec=recompile, cross=on)
echo "--- C6: spec+cross llvm (node-based) ---"
bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template recompile llvm "" "" ""
mv "${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_specrecompile_llvm_breakdown_time_log.csv" \
   "${DEST_DIR}/figC6_spec_and_cross_llvm.csv"

echo ""
echo "=== Compilation-time reduction breakdown complete ==="
echo "Output files:"
echo "  Figure A: ${DEST_DIR}/figA[1-3]_*.csv"
echo "  Figure B: ${DEST_DIR}/figB[1-12]_*.csv"
echo "  Figure C: ${DEST_DIR}/figC[1-6]_*.csv"
