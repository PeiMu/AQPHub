#!/usr/bin/env bash
#
# count_cache_rate.sh — Measure JIT cache hit rates across cache modes and configurations.
#
# Configurations tested:
#   1. duckdb   topdown    JOB
#   2. duckdb   none       JOB
#   3. duckdb   node-based DSB_50
#   4. duckdb   none       DSB_50
#   5. postgres node-based JOB
#   6. postgres node-based DSB_50
#
# Cache modes: single-run-strict, single-run-parameterized, single-run-template
#
# The script patches ir_to_llvm.cpp to enable AQP_DUMP_CACHE_KEYS in release
# builds, rebuilds, runs all configs, parses results, then reverts the patch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_FILE="${PROJECT}/src/jit/ir_to_llvm.cpp"
BUILD_DIR="${PROJECT}/build_release"
DUMP_DIR="/tmp/aqp_cache_rate_dumps"
RESULTS_FILE="${SCRIPT_DIR}/cache_rate_results.txt"

CACHE_MODES=("single-run-strict" "single-run-parameterized" "single-run-template")

CONFIGS=(
    "job       duckdb   topdown"
    "job       duckdb   none"
    "dsb_50    duckdb   node-based"
    "dsb_50    duckdb   none"
    "job       postgres node-based"
    "dsb_50    postgres node-based"
)

########################################
# Step 1: Patch ir_to_llvm.cpp
########################################
patch_source() {
    echo "=== Patching ${SRC_FILE} to enable AQP_DUMP_CACHE_KEYS in release builds ==="

    # Add #include <fcntl.h> after #include <unistd.h> (line 56) if not present
    if ! grep -q '#include <fcntl.h>' "$SRC_FILE"; then
        sed -i '/#include <unistd.h>/a #include <fcntl.h>' "$SRC_FILE"
        echo "  Added #include <fcntl.h>"
    fi

    # Remove the two #ifndef NDEBUG / #endif guard pairs around AQP_DUMP_CACHE_KEYS blocks.
    # We use a Python script for precise multi-line editing.
    python3 - "$SRC_FILE" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()

out = []
i = 0
removed = 0
while i < len(lines):
    line = lines[i].rstrip('\n')
    # Match: #ifndef NDEBUG  followed (within 2 lines) by AQP_DUMP_CACHE_KEYS
    if line.strip() == '#ifndef NDEBUG':
        # Look ahead for AQP_DUMP_CACHE_KEYS within the next 5 lines
        lookahead = ''.join(lines[i:i+6])
        if 'AQP_DUMP_CACHE_KEYS' in lookahead:
            # Skip this #ifndef NDEBUG line
            i += 1
            removed += 1
            # Now find and skip the matching #endif
            depth = 1
            block_lines = []
            while i < len(lines):
                l = lines[i].rstrip('\n')
                if l.strip().startswith('#ifndef') or l.strip().startswith('#ifdef'):
                    depth += 1
                    block_lines.append(lines[i])
                elif l.strip() == '#endif':
                    depth -= 1
                    if depth == 0:
                        # Skip this #endif line
                        i += 1
                        removed += 1
                        break
                    else:
                        block_lines.append(lines[i])
                else:
                    block_lines.append(lines[i])
                i += 1
            out.extend(block_lines)
            continue
    out.append(lines[i])
    i += 1

with open(path, 'w') as f:
    f.writelines(out)
print(f"  Removed {removed} #ifndef NDEBUG / #endif guard lines")
PYEOF
}

########################################
# Step 2: Rebuild
########################################
rebuild() {
    echo "=== Rebuilding aqp_middleware ==="
    cmake --build "$BUILD_DIR" --target aqp_middleware -j "$(nproc)" 2>&1 | tail -5
    echo "  Build complete."
}

########################################
# Step 3: Revert patch
########################################
revert_source() {
    echo "=== Reverting patch ==="
    (cd "$PROJECT" && git checkout -- src/jit/ir_to_llvm.cpp)
    echo "  Reverted."
}

########################################
# Step 4: Run one configuration
########################################
run_one() {
    local bench="$1" engine="$2" split="$3" cache_mode="$4" dump_file="$5"

    rm -f "$dump_file"

    echo "  Running: bench=${bench} engine=${engine} split=${split} cache=${cache_mode}"

    # run_aqp.sh positional args:
    #   $1=bench $2=engine $3=split $4=jit_level $5=jit_simd
    #   $6=payload_prune $7=prefetch $8=batch_probe $9=skip_hash_cmp
    #   $10=jit_cache $11=spec_jit $12=compile_mode
    AQP_DUMP_CACHE_KEYS="$dump_file" \
        bash "${SCRIPT_DIR}/run_aqp.sh" \
        "$bench" "$engine" "$split" \
        query avx2 \
        on on on all \
        "$cache_mode" \
        off llvm \
        > /dev/null 2>&1 || {
            echo "    WARNING: run_aqp.sh exited with non-zero status"
        }

    if [[ -f "$dump_file" ]]; then
        local total hit miss rate
        total=$(wc -l < "$dump_file")
        hit=$(grep -c 'HIT=1' "$dump_file" || true)
        miss=$(grep -c 'HIT=0' "$dump_file" || true)
        if (( total > 0 )); then
            rate=$(python3 -c "print(f'{100*${hit}/${total}:.1f}')")
        else
            rate="N/A"
        fi
        echo "    => total=${total} hit=${hit} miss=${miss} rate=${rate}%"
        echo -e "${bench}\t${engine}\t${split}\t${cache_mode}\t${total}\t${hit}\t${miss}\t${rate}%" >> "$RESULTS_FILE"
    else
        echo "    => No dump file produced (no JIT compilations?)"
        echo -e "${bench}\t${engine}\t${split}\t${cache_mode}\t0\t0\t0\tN/A" >> "$RESULTS_FILE"
    fi
}

########################################
# Main
########################################
main() {
    mkdir -p "$DUMP_DIR"
    rm -f "$RESULTS_FILE"
    echo -e "BENCH\tENGINE\tSPLIT\tCACHE_MODE\tTOTAL\tHIT\tMISS\tRATE" > "$RESULTS_FILE"

    # Patch and rebuild
    patch_source
    rebuild

    trap 'echo "Caught signal, reverting..."; revert_source; rebuild; exit 1' INT TERM

    # Run all configurations
    for config in "${CONFIGS[@]}"; do
        read -r bench engine split <<< "$config"
        for cache_mode in "${CACHE_MODES[@]}"; do
            safe_name="${bench}_${engine}_${split}_${cache_mode}"
            safe_name="${safe_name//\//_}"
            dump_file="${DUMP_DIR}/${safe_name}.tsv"
            run_one "$bench" "$engine" "$split" "$cache_mode" "$dump_file"
        done
    done

    # Revert patch and rebuild
    revert_source
    rebuild

    # Print summary table
    echo ""
    echo "=================================================================="
    echo "                   JIT Cache Hit Rate Summary"
    echo "=================================================================="
    printf "%-10s %-10s %-12s %-28s %6s %5s %5s %7s\n" \
        "BENCH" "ENGINE" "SPLIT" "CACHE_MODE" "TOTAL" "HIT" "MISS" "RATE"
    echo "------------------------------------------------------------------"
    tail -n +2 "$RESULTS_FILE" | while IFS=$'\t' read -r bench engine split cmode total hit miss rate; do
        printf "%-10s %-10s %-12s %-28s %6s %5s %5s %7s\n" \
            "$bench" "$engine" "$split" "$cmode" "$total" "$hit" "$miss" "$rate"
    done
    echo "=================================================================="
    echo ""
    echo "Raw dump files: ${DUMP_DIR}/"
    echo "Results TSV:    ${RESULTS_FILE}"
}

main "$@"
