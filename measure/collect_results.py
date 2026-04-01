"""
collect_results.py -- Aggregate TranSQL+ and llama.cpp evaluation results.

Usage:
    python measure/collect_results.py --results-dir measure/results

Parses JSON results from run_prefill.py / run_decode.py and text output
from run_llamacpp.sh to produce a comparison table.
"""

import argparse
import json
import os
import re


def parse_llama_bench(filepath):
    """Parse llama-bench output for tokens/sec and timing."""
    if not os.path.exists(filepath):
        return None

    with open(filepath) as f:
        text = f.read()

    result = {}

    # Parse pp (prompt processing) throughput: "pp XXX: ... t/s"
    pp_match = re.search(r"pp\s+(\d+).*?(\d+\.\d+)\s+t/s", text)
    if pp_match:
        result["pp_tokens"] = int(pp_match.group(1))
        result["pp_tok_per_s"] = float(pp_match.group(2))

    # Parse tg (token generation) throughput: "tg XXX: ... t/s"
    tg_match = re.search(r"tg\s+(\d+).*?(\d+\.\d+)\s+t/s", text)
    if tg_match:
        result["tg_tokens"] = int(tg_match.group(1))
        result["tg_tok_per_s"] = float(tg_match.group(2))

    # Parse peak RSS from /usr/bin/time -v output
    rss_match = re.search(r"Maximum resident set size.*?:\s*(\d+)", text)
    if rss_match:
        result["peak_rss_kb"] = int(rss_match.group(1))
        result["peak_rss_mb"] = result["peak_rss_kb"] / 1024.0

    return result if result else None


def parse_llama_perplexity(filepath):
    """Parse llama-perplexity output for PPL value."""
    if not os.path.exists(filepath):
        return None

    with open(filepath) as f:
        text = f.read()

    # Look for "Final estimate: PPL = X.XXXX"
    ppl_match = re.search(r"[Ff]inal.*?PPL\s*=\s*(\d+\.\d+)", text)
    if ppl_match:
        return float(ppl_match.group(1))

    # Alternative format: "perplexity: X.XXXX"
    ppl_match = re.search(r"perplexity\s*[:=]\s*(\d+\.\d+)", text)
    if ppl_match:
        return float(ppl_match.group(1))

    return None


def load_transql_results(results_dir):
    """Load TranSQL+ prefill, decode, and perplexity results."""
    results = {}

    prefill_path = os.path.join(results_dir, "prefill.json")
    if os.path.exists(prefill_path):
        with open(prefill_path) as f:
            results["prefill"] = json.load(f)

    decode_path = os.path.join(results_dir, "decode.json")
    if os.path.exists(decode_path):
        with open(decode_path) as f:
            results["decode"] = json.load(f)

    ppl_path = os.path.join(results_dir, "transql_ppl.json")
    if os.path.exists(ppl_path):
        with open(ppl_path) as f:
            results["perplexity"] = json.load(f)

    return results


def load_llamacpp_results(results_dir):
    """Load llama.cpp benchmark results for all quantizations."""
    quants = ["f32", "q4_k_m", "q8_0"]
    lengths = [25, 50, 100, 200]
    results = {}

    for q in quants:
        qr = {"prefill": {}, "decode": {}, "perplexity": None}

        for pp in lengths:
            # Prefill-only
            bench = parse_llama_bench(
                os.path.join(results_dir, f"llamacpp_{q}_pp{pp}.txt"))
            if bench:
                qr["prefill"][pp] = bench

            # Prefill + decode
            bench = parse_llama_bench(
                os.path.join(results_dir, f"llamacpp_{q}_pp{pp}_tg50.txt"))
            if bench:
                qr["decode"][pp] = bench

        # Perplexity
        ppl = parse_llama_perplexity(
            os.path.join(results_dir, f"llamacpp_{q}_ppl.txt"))
        qr["perplexity"] = ppl

        results[q] = qr

    return results


def print_comparison_table(transql, llamacpp):
    """Print a formatted comparison table."""
    lengths = [25, 50, 100, 200]

    # Model file sizes (approximate)
    model_sizes = {
        "f32": "~30 GB",
        "q4_k_m": "~4.9 GB",
        "q8_0": "~8.5 GB",
    }

    print("\n" + "=" * 90)
    print("  EVALUATION RESULTS: TranSQL+ (DuckDB) vs llama.cpp")
    print("=" * 90)

    # --- Prefill Throughput ---
    print("\n--- Prefill Throughput (tokens/sec) ---")
    header = f"{'Prompt Len':>12}"
    header += f"{'TranSQL+':>14}"
    for q in ["f32", "q4_k_m", "q8_0"]:
        header += f"{'llama.' + q:>14}"
    print(header)
    print("-" * len(header))

    for pp in lengths:
        row = f"{pp:>12}"
        # TranSQL+
        if transql.get("prefill"):
            entry = next((e for e in transql["prefill"]
                          if e["prompt_length"] == pp), None)
            if entry:
                row += f"{entry['mean_throughput_tok_per_s']:>14.2f}"
            else:
                row += f"{'--':>14}"
        else:
            row += f"{'--':>14}"

        # llama.cpp variants
        for q in ["f32", "q4_k_m", "q8_0"]:
            bench = llamacpp.get(q, {}).get("prefill", {}).get(pp)
            if bench and "pp_tok_per_s" in bench:
                row += f"{bench['pp_tok_per_s']:>14.2f}"
            else:
                row += f"{'--':>14}"
        print(row)

    # --- Decoding Throughput ---
    print("\n--- Decoding Throughput (tokens/sec) ---")
    print(header)
    print("-" * len(header))

    for pp in lengths:
        row = f"{pp:>12}"
        # TranSQL+
        if transql.get("decode"):
            entry = next((e for e in transql["decode"]
                          if e["prompt_length"] == pp), None)
            if entry:
                row += f"{entry['decode_throughput_tok_per_s']:>14.2f}"
            else:
                row += f"{'--':>14}"
        else:
            row += f"{'--':>14}"

        # llama.cpp variants
        for q in ["f32", "q4_k_m", "q8_0"]:
            bench = llamacpp.get(q, {}).get("decode", {}).get(pp)
            if bench and "tg_tok_per_s" in bench:
                row += f"{bench['tg_tok_per_s']:>14.2f}"
            else:
                row += f"{'--':>14}"
        print(row)

    # --- RAM Usage ---
    print("\n--- Peak RAM (MB) ---")
    row_header = f"{'Config':>14}"
    for label in ["TranSQL+", "llama.f32", "llama.q4_k_m", "llama.q8_0"]:
        row_header += f"{label:>14}"
    print(row_header)
    print("-" * len(row_header))

    row = f"{'Peak RSS':>14}"
    # TranSQL+ peak RSS (take max across all runs)
    max_rss = 0
    for key in ["prefill", "decode"]:
        if transql.get(key):
            for entry in transql[key]:
                max_rss = max(max_rss, entry.get("peak_rss_mb", 0))
    row += f"{max_rss:>14.0f}" if max_rss else f"{'--':>14}"

    for q in ["f32", "q4_k_m", "q8_0"]:
        max_rss_q = 0
        for pp_data in llamacpp.get(q, {}).get("prefill", {}).values():
            max_rss_q = max(max_rss_q, pp_data.get("peak_rss_mb", 0))
        for pp_data in llamacpp.get(q, {}).get("decode", {}).values():
            max_rss_q = max(max_rss_q, pp_data.get("peak_rss_mb", 0))
        row += f"{max_rss_q:>14.0f}" if max_rss_q else f"{'--':>14}"
    print(row)

    # DB size
    db_size = 0
    if transql.get("prefill"):
        for entry in transql["prefill"]:
            db_size = max(db_size, entry.get("db_size_gb", 0))
    if db_size:
        print(f"\n  TranSQL+ DuckDB file size: {db_size:.2f} GB")

    # --- Perplexity ---
    print("\n--- Perplexity (WikiText-2, lower = better) ---")
    row = f"{'PPL':>14}"
    if transql.get("perplexity"):
        row += f"{transql['perplexity']['perplexity']:>14.4f}"
    else:
        row += f"{'--':>14}"
    for q in ["f32", "q4_k_m", "q8_0"]:
        ppl = llamacpp.get(q, {}).get("perplexity")
        row += f"{ppl:>14.4f}" if ppl else f"{'--':>14}"
    print(f"{'':>14}{'TranSQL+':>14}{'llama.f32':>14}"
          f"{'llama.q4_k_m':>14}{'llama.q8_0':>14}")
    print("-" * 70)
    print(row)

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="measure/results")
    args = parser.parse_args()

    transql = load_transql_results(args.results_dir)
    llamacpp = load_llamacpp_results(args.results_dir)

    print_comparison_table(transql, llamacpp)

    # Save combined JSON
    combined = {
        "transql": transql,
        "llamacpp": llamacpp,
    }
    out_path = os.path.join(args.results_dir, "combined_results.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nCombined results saved to {out_path}")


if __name__ == "__main__":
    main()
