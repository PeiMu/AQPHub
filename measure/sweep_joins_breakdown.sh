#!/usr/bin/env bash
#
# Phase 6 prefetch-distance sweep on JOB join-heavy queries.
# Reports execute-only mean time (column 4 of --timing CSV), skipping JIT
# compile so the comparison is not dominated by codegen latency.
#
# Usage:
#   bash sweep_joins_breakdown.sh                     # default 12 join-heavy queries
#   QUERIES="29a 29b 29c"  bash sweep_joins_breakdown.sh
#   ITERS=20               bash sweep_joins_breakdown.sh
#   OUT=/tmp/my.csv        bash sweep_joins_breakdown.sh
#
# Output CSV columns: query,combo,exec_mean_ms,exec_min_ms,exec_max_ms,jit_compile_ms
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-${SCRIPT_DIR}/../build_release/aqp_middleware}"
DB="${DB:-/home/pei/Project/duckdb/measure/imdb.db}"
SCHEMA="${SCHEMA:-/home/pei/Project/benchmarks/imdb_job-postgres/schema.sql}"
FKEYS="${FKEYS:-/home/pei/Project/benchmarks/imdb_job-postgres/fkeys.sql}"
QDIR="${QDIR:-/home/pei/Project/benchmarks/imdb_job-postgres/queries}"

OUT="${OUT:-${SCRIPT_DIR}/sweep_joins_breakdown.csv}"
ITERS="${ITERS:-15}"   # 5 warmup + 10 measured (last 10 used for stats)

# Default: 12 most join-heavy queries
QUERIES="${QUERIES:-27a 27b 27c 28a 28b 28c 29a 29b 29c 30a 30b 30c}"

# (D_entry, D_row) pairs to sweep, plus the no-Phase-6 baselines.
COMBOS=(
  "noPF_noBP|--no-jit-prefetch --no-jit-batch-probe"   # scalar single-pass (Phase-6 mechanism off)
  "BP_noPF|--no-jit-prefetch"                           # batch probe ON, no prefetch
  "pf_0_0|--jit-prefetch-entry-dist=0 --jit-prefetch-row-dist=0"
  "pf_16_8|--jit-prefetch-entry-dist=16 --jit-prefetch-row-dist=8"
  "pf_24_12|--jit-prefetch-entry-dist=24 --jit-prefetch-row-dist=12"  # current default
  "pf_32_16|--jit-prefetch-entry-dist=32 --jit-prefetch-row-dist=16"
  "pf_48_24|--jit-prefetch-entry-dist=48 --jit-prefetch-row-dist=24"
  "pf_64_24|--jit-prefetch-entry-dist=64 --jit-prefetch-row-dist=24"
)

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found or not executable: $BIN" >&2
  exit 1
fi
if [[ ! -f "$DB" ]]; then
  echo "ERROR: DB not found: $DB" >&2
  exit 1
fi

WORK=$(mktemp -d)
echo "query,combo,exec_mean_ms,exec_min_ms,exec_max_ms,jit_compile_ms" > "$OUT"

cd "$WORK"
for q in $QUERIES; do
  for entry in "${COMBOS[@]}"; do
    name="${entry%%|*}"
    flags="${entry##*|}"
    echo "==> $q / $name" >&2
    rm -f time_log.csv
    "$BIN" --engine=duckdb --db="$DB" \
      --schema="$SCHEMA" --fkeys="$FKEYS" \
      --split=none --no-analyze --timing \
      --repeat="$ITERS" \
      --jit-level=pipeline --jit-opt=o1 --jit-simd=auto $flags \
      "$QDIR/${q}.sql" > /dev/null 2>&1
    # Last 10 rows = measured (first 5 = warmup). Col 3 = JIT compile ms,
    # col 4 = execute ms.
    stats=$(tail -n 10 time_log.csv | awk -F',' '
      { e=$4+0; c=$3+0
        if (NR==1) {emin=e; emax=e}
        if (e<emin) emin=e
        if (e>emax) emax=e
        esum+=e; csum+=c; n++ }
      END { printf "%.3f,%.3f,%.3f,%.3f", esum/n, emin, emax, csum/n }')
    echo "$q,$name,$stats" | tee -a "$OUT"
  done
done

echo ""
echo "===== Summary (per query x combo) ====="
column -t -s, "$OUT"

echo ""
echo "===== Total exec time per combo (sum across queries) ====="
awk -F',' 'NR>1 {sum[$2]+=$3} END {for (k in sum) printf "%-12s  %.1f ms\n", k, sum[k]}' "$OUT" | sort -k2 -n

echo ""
echo "===== Best combo per query (lowest exec_mean_ms) ====="
awk -F',' 'NR>1 {
  if (!(($1) in best) || ($3+0) < best[$1]) { best[$1]=$3+0; who[$1]=$2 }
} END {
  for (q in best) printf "%-6s  %-12s  %.3f ms\n", q, who[q], best[q]
}' "$OUT" | sort

rm -rf "$WORK"
