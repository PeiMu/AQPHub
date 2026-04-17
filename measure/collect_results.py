"""
collect_results.py -- Aggregate TranSQL+ and llama.cpp evaluation results.

Usage:
    python measure/collect_results.py --results-dir measure/results

Parses JSON results from run_prefill.py / run_decode.py and text output
from run_llamacpp.sh. Produces a unified comparison table and JSON where
every entry (regardless of system) has the same fields:

  prefill_latency_s          -- time to first token (paper definition)
  decode_latency_s           -- avg time per token after the first (paper definition)
  prefill_throughput_tok_per_s
  decode_throughput_tok_per_s
  peak_rss_mb
  db_size_gb                 -- TranSQL+ only (null for llama.cpp)
"""

import argparse
import json
import os
import re


# ---------------------------------------------------------------------------
# llama.cpp result parsers
# ---------------------------------------------------------------------------

def parse_llama_bench(filepath):
    """Parse llama-bench output for pp/tg throughput and peak RSS."""
    if not os.path.exists(filepath):
        return None

    with open(filepath) as f:
        text = f.read()

    result = {}

    # New llama-bench format (markdown table):
    # | llama 8B all F32 | ... | pp25 | 19.40 ± 0.05 |
    pp_match = re.search(r"pp\s*(\d+)\s*\|\s*(\d+\.?\d*)\s*±", text)
    if pp_match:
        result["pp_tokens"] = int(pp_match.group(1))
        result["pp_tok_per_s"] = float(pp_match.group(2))
    else:
        pp_match = re.search(r"pp\s+(\d+).*?(\d+\.\d+)\s+t/s", text)
        if pp_match:
            result["pp_tokens"] = int(pp_match.group(1))
            result["pp_tok_per_s"] = float(pp_match.group(2))

    tg_match = re.search(r"tg\s*(\d+)\s*\|\s*(\d+\.?\d*)\s*±", text)
    if tg_match:
        result["tg_tokens"] = int(tg_match.group(1))
        result["tg_tok_per_s"] = float(tg_match.group(2))
    else:
        tg_match = re.search(r"tg\s+(\d+).*?(\d+\.\d+)\s+t/s", text)
        if tg_match:
            result["tg_tokens"] = int(tg_match.group(1))
            result["tg_tok_per_s"] = float(tg_match.group(2))

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

    ppl_match = re.search(r"[Ff]inal.*?PPL\s*=\s*(\d+\.\d+)", text)
    if ppl_match:
        return float(ppl_match.group(1))

    ppl_match = re.search(r"perplexity\s*[:=]\s*(\d+\.\d+)", text)
    if ppl_match:
        return float(ppl_match.group(1))

    return None


# ---------------------------------------------------------------------------
# Load and normalise results into a common schema
# ---------------------------------------------------------------------------

COMMON_FIELDS = [
    "system", "prompt_length",
    "prefill_latency_s", "prefill_latency_std_s", "prefill_throughput_tok_per_s",
    "decode_latency_s",  "decode_latency_std_s",  "decode_throughput_tok_per_s",
    "peak_rss_mb", "db_size_gb",
]


def _entry(system, prompt_length, *,
           prefill_latency_s, prefill_throughput_tok_per_s,
           decode_latency_s, decode_throughput_tok_per_s,
           peak_rss_mb,
           prefill_latency_std_s=None,
           decode_latency_std_s=None,
           db_size_gb=None):
    return {
        "system": system,
        "prompt_length": prompt_length,
        "prefill_latency_s": prefill_latency_s,
        "prefill_latency_std_s": prefill_latency_std_s,
        "prefill_throughput_tok_per_s": prefill_throughput_tok_per_s,
        "decode_latency_s": decode_latency_s,
        "decode_latency_std_s": decode_latency_std_s,
        "decode_throughput_tok_per_s": decode_throughput_tok_per_s,
        "peak_rss_mb": peak_rss_mb,
        "db_size_gb": db_size_gb,
    }


def load_transql_entries(results_dir):
    """Return list of normalised entries for TranSQL+."""
    entries = []

    prefill_path = os.path.join(results_dir, "prefill.json")
    decode_path  = os.path.join(results_dir, "decode.json")

    prefill_by_len = {}
    if os.path.exists(prefill_path):
        with open(prefill_path) as f:
            for r in json.load(f):
                prefill_by_len[r["prompt_length"]] = r

    decode_by_len = {}
    if os.path.exists(decode_path):
        with open(decode_path) as f:
            for r in json.load(f):
                decode_by_len[r["prompt_length"]] = r

    lengths = sorted(set(prefill_by_len) | set(decode_by_len))
    for length in lengths:
        p = prefill_by_len.get(length, {})
        d = decode_by_len.get(length, {})

        # Support both new field names and legacy names from older runs.
        def pget(key, *fallbacks):
            for k in (key, *fallbacks):
                if k in p: return p[k]
            return None

        def dget(key, *fallbacks):
            for k in (key, *fallbacks):
                if k in d: return d[k]
            return None

        entries.append(_entry(
            system="transql",
            prompt_length=length,
            prefill_latency_s=pget("prefill_latency_s", "mean_latency_s"),
            prefill_latency_std_s=pget("prefill_latency_std_s", "std_latency_s"),
            prefill_throughput_tok_per_s=pget("prefill_throughput_tok_per_s",
                                              "mean_throughput_tok_per_s"),
            decode_latency_s=dget("decode_latency_s", "mean_decode_latency_s"),
            decode_latency_std_s=dget("decode_latency_std_s"),
            decode_throughput_tok_per_s=dget("decode_throughput_tok_per_s"),
            peak_rss_mb=max(p.get("peak_rss_mb") or 0,
                            d.get("peak_rss_mb") or 0) or None,
            db_size_gb=p.get("db_size_gb"),
        ))

    return entries


def load_llamacpp_entries(results_dir):
    """Return list of normalised entries for llama.cpp variants.

    llama-bench reports throughput (tok/s); we derive latency as:
      prefill_latency_s  = prompt_length / pp_tok_per_s
      decode_latency_s   = 1 / tg_tok_per_s
    """
    entries = []
    quants  = ["f32", "q4_k_m", "q8_0"]
    lengths = [25, 50, 100, 200]

    for q in quants:
        ppl = parse_llama_perplexity(
            os.path.join(results_dir, f"llamacpp_{q}_ppl.txt"))

        for pp in lengths:
            # prefill-only bench gives pp throughput
            pp_bench = parse_llama_bench(
                os.path.join(results_dir, f"llamacpp_{q}_pp{pp}.txt"))
            # prefill+decode bench gives both pp and tg throughput
            tg_bench = parse_llama_bench(
                os.path.join(results_dir, f"llamacpp_{q}_pp{pp}_tg50.txt"))

            # Use tg_bench for pp throughput if pp_bench missing
            src_pp = pp_bench or tg_bench
            src_tg = tg_bench

            if not src_pp and not src_tg:
                continue

            pp_tok_per_s = src_pp.get("pp_tok_per_s") if src_pp else None
            tg_tok_per_s = src_tg.get("tg_tok_per_s") if src_tg else None
            peak_rss_mb  = (src_tg or src_pp or {}).get("peak_rss_mb")

            entries.append(_entry(
                system=f"llamacpp_{q}",
                prompt_length=pp,
                prefill_latency_s=pp / pp_tok_per_s if pp_tok_per_s else None,
                prefill_throughput_tok_per_s=pp_tok_per_s,
                decode_latency_s=1.0 / tg_tok_per_s if tg_tok_per_s else None,
                decode_throughput_tok_per_s=tg_tok_per_s,
                peak_rss_mb=peak_rss_mb,
            ))

    return entries


def load_perplexity(results_dir):
    """Load perplexity results for both systems."""
    ppl = {}

    # TranSQL+ — prefer the most recent result file
    for fname in ["transql_ppl_double_fix.json", "transql_ppl.json"]:
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                ppl["transql"] = json.load(f)
            break

    # llama.cpp
    for q in ["f32", "q4_k_m", "q8_0"]:
        v = parse_llama_perplexity(
            os.path.join(results_dir, f"llamacpp_{q}_ppl.txt"))
        if v is not None:
            ppl[f"llamacpp_{q}"] = {"perplexity": v}

    return ppl


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".3f"):
    return f"{val:{fmt}}" if val is not None else "--"


def print_comparison_table(entries, perplexity):
    systems = sorted({e["system"] for e in entries})
    lengths = sorted({e["prompt_length"] for e in entries})

    def get(system, length, field):
        for e in entries:
            if e["system"] == system and e["prompt_length"] == length:
                return e.get(field)
        return None

    col_w = 16
    sys_labels = {s: s.replace("llamacpp_", "llama.") for s in systems}

    print("\n" + "=" * (12 + col_w * len(systems)))
    print("  EVALUATION RESULTS: TranSQL+ vs llama.cpp")
    print("=" * (12 + col_w * len(systems)))

    header = f"{'Prompt':>12}" + "".join(f"{sys_labels[s]:>{col_w}}" for s in systems)

    for label, field in [
        ("Prefill latency (s/prompt) — time to first token", "prefill_latency_s"),
        ("Prefill throughput (tok/s)", "prefill_throughput_tok_per_s"),
        ("Decode latency (s/tok) — avg after first token",   "decode_latency_s"),
        ("Decode throughput (tok/s)", "decode_throughput_tok_per_s"),
    ]:
        print(f"\n--- {label} ---")
        print(header)
        print("-" * len(header))
        for length in lengths:
            row = f"{length:>12}"
            for s in systems:
                row += f"{_fmt(get(s, length, field)):>{col_w}}"
            print(row)

    print(f"\n--- Peak RAM (MB) ---")
    print(f"{'':>12}" + "".join(f"{sys_labels[s]:>{col_w}}" for s in systems))
    print("-" * (12 + col_w * len(systems)))
    row = f"{'Peak RSS':>12}"
    for s in systems:
        vals = [e["peak_rss_mb"] for e in entries
                if e["system"] == s and e["peak_rss_mb"] is not None]
        row += f"{_fmt(max(vals), '.0f') if vals else '--':>{col_w}}"
    print(row)

    db_sizes = {e["db_size_gb"] for e in entries
                if e.get("db_size_gb") is not None}
    if db_sizes:
        print(f"\n  TranSQL+ DuckDB file size: {max(db_sizes):.2f} GB")

    if perplexity:
        print(f"\n--- Perplexity (WikiText-2, lower = better) ---")
        print(f"{'':>12}" + "".join(f"{sys_labels.get(s, s):>{col_w}}"
                                     for s in ["transql"] + [f"llamacpp_{q}"
                                     for q in ["f32","q4_k_m","q8_0"]]))
        print("-" * (12 + col_w * 4))
        row = f"{'PPL':>12}"
        for s in ["transql", "llamacpp_f32", "llamacpp_q4_k_m", "llamacpp_q8_0"]:
            v = perplexity.get(s, {}).get("perplexity")
            row += f"{_fmt(v, '.4f'):>{col_w}}"
        print(row)

    print("\n" + "=" * (12 + col_w * len(systems)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="measure/results")
    args = parser.parse_args()

    transql_entries  = load_transql_entries(args.results_dir)
    llamacpp_entries = load_llamacpp_entries(args.results_dir)
    all_entries      = transql_entries + llamacpp_entries
    perplexity       = load_perplexity(args.results_dir)

    print_comparison_table(all_entries, perplexity)

    combined = {
        "schema": COMMON_FIELDS,
        "results": all_entries,
        "perplexity": perplexity,
    }
    out_path = os.path.join(args.results_dir, "combined_results.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nCombined results saved to {out_path}")


if __name__ == "__main__":
    main()
