#!/bin/bash
# run_llamacpp.sh -- Measure llama.cpp baseline latency, throughput, RAM, and
#                    perplexity for Llama3-8B (F32 + quantized).
#
# Prerequisites:
#   1. llama.cpp source at $LLAMA_CPP_DIR (default: /home/pei/Project/llama.cpp)
#   2. HuggingFace model access for meta-llama/Meta-Llama-3-8B
#
# Usage:
#   bash measure/run_llamacpp.sh [--lengths "25 50 100 200"] [--quants "f32"] [--output-dir DIR]
#
# Options:
#   --lengths   Prompt lengths to benchmark (default: "25 50 100 200")
#   --quants    Quantizations to test (default: "f32" for paper reproduction)
#               Use "f32 q4_k_m q8_0" for all variants
#   --output-dir  Results directory (default: measure/results)
#
# Quantization levels:
#   f32     -- unquantized full-precision (matches TranSQL+ / paper)
#   q4_k_m  -- 4-bit quantization (best quality/size ratio)
#   q8_0    -- 8-bit quantization (middle ground)

set -euo pipefail

# Parse arguments
LENGTHS="25 50 100 200"
OUTPUT_DIR="measure/results"
QUANTS="f32"  # Default: F32 only (matches paper: "unquantized full-precision")

while [[ $# -gt 0 ]]; do
    case $1 in
        --lengths)
            LENGTHS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quants)
            QUANTS="$2"  # e.g., "f32 q4_k_m q8_0" for all
            shift 2
            ;;
        *)
            # Legacy: first positional arg is output dir
            OUTPUT_DIR="$1"
            shift
            ;;
    esac
done

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/pei/Project/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-/home/pei/Project/llama3_gguf}"
HF_MODEL_PATH="${HF_MODEL_PATH:-/home/pei/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
NTHREADS=$(nproc)

mkdir -p "$OUTPUT_DIR" "$MODEL_DIR"

GGUF_F32="$MODEL_DIR/llama3-8b-f32.gguf"
GGUF_Q4KM="$MODEL_DIR/llama3-8b-q4_k_m.gguf"
GGUF_Q8="$MODEL_DIR/llama3-8b-q8_0.gguf"

# ---------------------------------------------------------------------------
# Step 0: Build llama.cpp if needed
# ---------------------------------------------------------------------------
LLAMA_BENCH="$LLAMA_CPP_DIR/build/bin/llama-bench"
LLAMA_QUANT="$LLAMA_CPP_DIR/build/bin/llama-quantize"
LLAMA_PERPLEXITY="$LLAMA_CPP_DIR/build/bin/llama-perplexity"
CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"

if [ ! -f "$LLAMA_BENCH" ]; then
    echo "=== Building llama.cpp ==="
    pushd "$LLAMA_CPP_DIR"
    cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j"$NTHREADS"
    popd
fi

# ---------------------------------------------------------------------------
# Step 1: Convert / quantize models
# ---------------------------------------------------------------------------
if [ ! -f "$GGUF_F32" ]; then
    echo ""
    echo "=== Converting Llama3-8B to GGUF F32 ==="
    python "$CONVERT_SCRIPT" \
        "$HF_MODEL_PATH" \
        --outfile "$GGUF_F32" \
        --outtype f32
fi

# Only quantize if requested
if [[ "$QUANTS" == *"q4_k_m"* ]] && [ ! -f "$GGUF_Q4KM" ]; then
    echo ""
    echo "=== Quantizing to Q4_K_M ==="
    "$LLAMA_QUANT" "$GGUF_F32" "$GGUF_Q4KM" Q4_K_M
fi

if [[ "$QUANTS" == *"q8_0"* ]] && [ ! -f "$GGUF_Q8" ]; then
    echo ""
    echo "=== Quantizing to Q8_0 ==="
    "$LLAMA_QUANT" "$GGUF_F32" "$GGUF_Q8" Q8_0
fi

echo ""
echo "=== Benchmark config ==="
echo "  Prompt lengths: $LENGTHS"
echo "  Quantizations: $QUANTS"
echo ""

# ---------------------------------------------------------------------------
# Helper: run llama-bench and capture output + peak RSS
# ---------------------------------------------------------------------------
run_bench() {
    local model_path="$1"
    local label="$2"
    local pp="$3"
    local tg="$4"
    local outfile="$5"

    echo "--- $label: pp=$pp tg=$tg ---"

    # Use /usr/bin/time for peak RSS measurement
    # New llama-bench CLI: -p for prompt tokens, -n for generation tokens
    /usr/bin/time -v "$LLAMA_BENCH" \
        -m "$model_path" \
        -t "$NTHREADS" \
        -ngl 0 \
        -p "$pp" \
        -n "$tg" \
        -r 3 \
        2>&1 | tee "$outfile"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 2: Prefill + Decoding benchmarks
# ---------------------------------------------------------------------------
for QUANT in $QUANTS; do
    case "$QUANT" in
        f32)    MODEL="$GGUF_F32";   LABEL="F32" ;;
        q4_k_m) MODEL="$GGUF_Q4KM"; LABEL="Q4_K_M" ;;
        q8_0)   MODEL="$GGUF_Q8";   LABEL="Q8_0" ;;
    esac

    echo ""
    echo "=========================================="
    echo "  llama.cpp $LABEL"
    echo "=========================================="

    # Prefill-only
    for PP in $LENGTHS; do
        run_bench "$MODEL" "$LABEL" "$PP" 0 \
            "$OUTPUT_DIR/llamacpp_${QUANT}_pp${PP}.txt"
    done

    # Prefill + decode (50 tokens)
    for PP in $LENGTHS; do
        run_bench "$MODEL" "$LABEL" "$PP" 50 \
            "$OUTPUT_DIR/llamacpp_${QUANT}_pp${PP}_tg50.txt"
    done
done

# ---------------------------------------------------------------------------
# Step 3: Perplexity (accuracy) on WikiText-2
# ---------------------------------------------------------------------------
WIKITEXT="$MODEL_DIR/wikitext-2-raw-v1.txt"
if [ ! -f "$WIKITEXT" ]; then
    echo ""
    echo "=== Downloading WikiText-2 ==="
    python -c "
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
with open('$WIKITEXT', 'w') as f:
    for row in ds:
        f.write(row['text'] + '\n')
print('Saved to $WIKITEXT')
"
fi

for QUANT in $QUANTS; do
    case "$QUANT" in
        f32)    MODEL="$GGUF_F32";   LABEL="F32" ;;
        q4_k_m) MODEL="$GGUF_Q4KM"; LABEL="Q4_K_M" ;;
        q8_0)   MODEL="$GGUF_Q8";   LABEL="Q8_0" ;;
    esac

    echo ""
    echo "=== Perplexity: $LABEL ==="
    /usr/bin/time -v "$LLAMA_PERPLEXITY" \
        -m "$MODEL" \
        -f "$WIKITEXT" \
        -t "$NTHREADS" \
        --chunks 64 \
        2>&1 | tee "$OUTPUT_DIR/llamacpp_${QUANT}_ppl.txt"
done

echo ""
echo "=========================================="
echo "  All results saved to $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Next: python measure/collect_results.py --results-dir $OUTPUT_DIR"
