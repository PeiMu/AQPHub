#!/usr/bin/env bash
#
# Full performance sweep for LinGo-DB on JOB benchmark.
# 4 splits x (2 no-jit modes + 3 compile-modes + 5 cache-modes) = 40 configs
# Run from measure/ directory.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

SPLITS=(none node-based topdown relationship-center)

# ============================================================
# No-JIT: LinGo-DB's own LLVM JIT
# ============================================================
for s in "${SPLITS[@]}"; do
  bash ./measure_breakdown_time_aqp.sh job lingodb "$s" llvm
done

# ============================================================
# No-JIT: LinGo-DB's own TPDE backend
# ============================================================
for s in "${SPLITS[@]}"; do
  bash ./measure_breakdown_time_aqp.sh job lingodb "$s" tpde
done

# ============================================================
# Query-JIT, compile_mode=llvm
# ============================================================
for s in "${SPLITS[@]}"; do
  bash ./measure_breakdown_time_aqp.sh job lingodb "$s" query none
done

# ============================================================
# Query-JIT, compile_mode=fastisel
# ============================================================
for s in "${SPLITS[@]}"; do
  bash ./measure_breakdown_time_aqp.sh job lingodb "$s" query none on on on on off off fastisel
done

# ============================================================
# Query-JIT, compile_mode=tpde
# ============================================================
for s in "${SPLITS[@]}"; do
  bash ./measure_breakdown_time_aqp.sh job lingodb "$s" query none on on on on off off tpde
done

# ============================================================
# Query-JIT + jit-cache modes (all splits)
# ============================================================
for s in "${SPLITS[@]}"; do
  bash ./measure_breakdown_time_aqp.sh job lingodb "$s" query none on on on on single-run off
done

for s in "${SPLITS[@]}"; do
  bash ./measure_breakdown_time_aqp.sh job lingodb "$s" query none on on on on single-run-template off
done

for s in "${SPLITS[@]}"; do
  bash ./measure_breakdown_time_aqp.sh job lingodb "$s" query none on on on on structural off
done

for s in "${SPLITS[@]}"; do
  bash ./measure_breakdown_time_aqp.sh job lingodb "$s" query none on on on on full off
done

echo "All LinGo-DB JOB configs completed."
