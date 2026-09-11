#!/usr/bin/env bash
#
# Evaluate the effect of runtime statistics collection (--collect-stats)
# on query execution time.
#
# Compares four configurations (job duckdb topdown):
#   1. no-jit + no-collect-stats  (default for no-jit)
#   2. no-jit + collect-stats     (explicit on)
#   3. query-jit + no-collect-stats
#   4. query-jit + collect-stats  (default for query-jit)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

DEST_DIR="${SCRIPT_DIR}/job_result"

# Args to measure_breakdown_time_aqp.sh:
#   bench engine split jit_level jit_simd payload_prune prefetch
#   batch_probe skip_hash_cmp jit_cache spec_jit compile_mode
#   tune_config disable_runtime_opts disable_compile_opts collect_stats

# Step 1: no-jit + no-collect-stats (default for no-jit — auto resolves to off)
echo "=== Step 1: no-jit + no-collect-stats ==="
bash ./measure_breakdown_time_aqp.sh job duckdb topdown none none on on on all off off llvm "" "" "" off && \
mv "${DEST_DIR}/duckdb_topdown_none_none_nocollectstats_breakdown_time_log.csv" \
   "${DEST_DIR}/eval_stats_nojit_nostats.csv" && \

# Step 2: no-jit + collect-stats (explicit on)
echo "=== Step 2: no-jit + collect-stats ==="
bash ./measure_breakdown_time_aqp.sh job duckdb topdown none none on on on all off off llvm "" "" "" on && \
mv "${DEST_DIR}/duckdb_topdown_none_none_collectstats_breakdown_time_log.csv" \
   "${DEST_DIR}/eval_stats_nojit_stats.csv" && \

# Step 3: query-jit + no-collect-stats
echo "=== Step 3: query-jit + no-collect-stats ==="
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all off off tpde "" "" "" off && \
mv "${DEST_DIR}/duckdb_topdown_query_none_tpde_nocollectstats_breakdown_time_log.csv" \
   "${DEST_DIR}/eval_stats_queryjit_nostats.csv" && \

# Step 4: query-jit + collect-stats (default for query-jit — auto resolves to on)
echo "=== Step 4: query-jit + collect-stats ==="
bash ./measure_breakdown_time_aqp.sh job duckdb topdown query none on on on all off off tpde "" "" "" on && \
mv "${DEST_DIR}/duckdb_topdown_query_none_tpde_collectstats_breakdown_time_log.csv" \
   "${DEST_DIR}/eval_stats_queryjit_stats.csv" && \

echo ""
echo "=== Runtime statistics evaluation complete ==="
echo "Output:"
echo "  ${DEST_DIR}/eval_stats_nojit_nostats.csv"
echo "  ${DEST_DIR}/eval_stats_nojit_stats.csv"
echo "  ${DEST_DIR}/eval_stats_queryjit_nostats.csv"
echo "  ${DEST_DIR}/eval_stats_queryjit_stats.csv"
