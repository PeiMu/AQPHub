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
# Figure B: JIT cache mode (compile-mode=tpde, spec=off, cross=off)
# ============================================================
#
# B1: off (no caching) — same CSV as A3
# B2: single-run-strict (exact plan match)
# B3: single-run-template (parameterized: constants from params array)
# B4: full
#
# ============================================================
# Figure C: Latency hiding (compile-mode=tpde, cache=template)
# ============================================================
#
# C1: baseline (spec=off, cross=off) tpde
# C2: +spec-jit (spec=recompile, cross=off) tpde
# C3: +spec-jit +cross-query-prep (spec=recompile, cross=on) tpde
# C4: baseline (spec=off, cross=off) llvm
# C5: +spec-jit (spec=recompile, cross=off) llvm
# C6: +spec-jit +cross-query-prep (spec=recompile, cross=on) llvm
#
# ============================================================
# Total configs to measure: 11 (A3=B1 shared)
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

# A1: LLVM O2 (baseline)
echo "--- A1: LLVM O2 ---"
bash ./measure_breakdown_time_aqp.sh $BASE off off llvm "" "" ""
A1="${DEST_DIR}/duckdb_topdown_query_none_llvm_breakdown_time_log.csv"
cp "$A1" "${DEST_DIR}/figA1_llvm.csv"

# A2: FastISel
echo "--- A2: FastISel ---"
#bash ./measure_breakdown_time_aqp.sh $BASE off off fastisel "" "" ""
#A2="${DEST_DIR}/duckdb_topdown_query_none_fastisel_breakdown_time_log.csv"
#cp "$A2" "${DEST_DIR}/figA2_fastisel.csv"

# A3: TPDE
echo "--- A3: TPDE ---"
#bash ./measure_breakdown_time_aqp.sh $BASE off off tpde "" "" ""
A3="${DEST_DIR}/duckdb_topdown_query_none_tpde_breakdown_time_log.csv"
cp "$A3" "${DEST_DIR}/figA3_tpde.csv"

echo ""
echo "============================================"
echo "Figure B: JIT cache mode (TPDE, spec=off)"
echo "============================================"

# B1: off — reuse A3
echo "--- B1: cache=off (reuse A3) tpde ---"
cp "$A3" "${DEST_DIR}/figB1_cache_off_tpde.csv"

# B2: single-run-strict (no cross-query-prep)
echo "--- B2: cache=strict tpde ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-strict off tpde "" "" "cross-query-prep"
B2="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_strict_tpde_nocrossqprep_breakdown_time_log.csv"
mv "$B2" "${DEST_DIR}/figB2_cache_strict_tpde.csv"

# B3: single-run-template (no cross-query-prep)
echo "--- B3: cache=template tpde ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off tpde "" "" "cross-query-prep"
B3="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_tpde_nocrossqprep_breakdown_time_log.csv"
mv "$B3" "${DEST_DIR}/figB3_cache_template_tpde.csv"

# B4: full
#echo "--- B4: cache=full tpde ---"
#bash ./measure_breakdown_time_aqp.sh $BASE full off tpde "" "" "cross-query-prep"
#B4="${DEST_DIR}/duckdb_topdown_query_none_jitcache_full_tpde_nocrossqprep_breakdown_time_log.csv"
#mv "$B4" "${DEST_DIR}/figB4_cache_full_tpde.csv"


# B5: off — reuse A3
echo "--- B5: cache=off (reuse A3) llvm ---"
cp "$A1" "${DEST_DIR}/figB5_cache_off_llvm.csv"

# B6: single-run-strict (no cross-query-prep)
echo "--- B6: cache=strict llvm ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-strict off llvm "" "" "cross-query-prep"
B6="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_strict_llvm_nocrossqprep_breakdown_time_log.csv"
mv "$B6" "${DEST_DIR}/figB6_cache_strict_llvm.csv"

# B7: single-run-template (no cross-query-prep)
echo "--- B7: cache=template llvm ---"
bash ./measure_breakdown_time_aqp.sh $BASE single-run-template off llvm "" "" "cross-query-prep"
B7="${DEST_DIR}/duckdb_topdown_query_none_jitcache_single_run_template_llvm_nocrossqprep_breakdown_time_log.csv"
mv "$B7" "${DEST_DIR}/figB7_cache_template_llvm.csv"

# B8: full
#echo "--- B8: cache=full ---"
#bash ./measure_breakdown_time_aqp.sh $BASE full off llvm "" "" "cross-query-prep"
#B8="${DEST_DIR}/duckdb_topdown_query_none_jitcache_full_llvm_nocrossqprep_breakdown_time_log.csv"
#mv "$B8" "${DEST_DIR}/figB8_cache_full_llvm.csv"


echo ""
echo "============================================"
echo "Figure C: Latency hiding (TPDE, cache=template)"
echo "============================================"

#node_based_BASE="job duckdb node-based query none on on on all"
#
## C1: baseline (spec=off, cross=off)
#echo "--- C1: no latency hiding tpde ---"
#bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template off tpde "" "" "cross-query-prep"
#C1="${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_tpde_nocrossqprep_breakdown_time_log.csv"
#mv "$C1" "${DEST_DIR}/figC1_no_hiding_tpde.csv"
#
## C2: +spec-jit (cross=off)
#echo "--- C2: +spec-jit tpde ---"
#bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template recompile tpde "" "" "cross-query-prep"
#C2="${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_specrecompile_tpde_nocrossqprep_breakdown_time_log.csv"
#mv "$C2" "${DEST_DIR}/figC2_spec_only_tpde.csv"
#
## C3: +spec-jit +cross-query-prep (both on)
#echo "--- C3: +spec-jit +cross-query-prep tpde ---"
##bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template recompile tpde "" "" ""
#C3="${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_specrecompile_tpde_breakdown_time_log.csv"
#cp "$C3" "${DEST_DIR}/figC3_spec_and_cross_tpde.csv"
#
#### LLVM
## C4: baseline (spec=off, cross=off)
#echo "--- C4: no latency hiding llvm ---"
#bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template off llvm "" "" "cross-query-prep"
#C4="${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_llvm_nocrossqprep_breakdown_time_log.csv"
#mv "$C4" "${DEST_DIR}/figC4_no_hiding_llvm.csv"
#
## C5: +spec-jit (cross=off)
#echo "--- C5: +spec-jit llvm ---"
#bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template recompile llvm "" "" "cross-query-prep"
#C5="${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_specrecompile_llvm_nocrossqprep_breakdown_time_log.csv"
#mv "$C5" "${DEST_DIR}/figC5_spec_only_llvm.csv"
#
## C6: +spec-jit +cross-query-prep (both on)
#echo "--- C6: +spec-jit +cross-query-prep llvm ---"
##bash ./measure_breakdown_time_aqp.sh $node_based_BASE single-run-template recompile llvm "" "" ""
#C6="${DEST_DIR}/duckdb_node-based_query_none_jitcache_single_run_template_specrecompile_llvm_breakdown_time_log.csv"
#cp "$C6" "${DEST_DIR}/figC6_spec_and_cross_llvm.csv"

echo ""
echo "=== Compilation-time reduction breakdown complete ==="
echo "Output files:"
echo "  Figure A: ${DEST_DIR}/figA[1-3]_*.csv"
echo "  Figure B: ${DEST_DIR}/figB[1-4]_*.csv"
echo "  Figure C: ${DEST_DIR}/figC[1-6]_*.csv"
