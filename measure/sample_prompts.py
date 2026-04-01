"""
sample_prompts.py -- Sample prompts from LMSys-Chat-1M at target token lengths.

Usage:
    python measure/sample_prompts.py \
        --output-dir measure/prompts \
        [--lengths 25 50 100 200] \
        [--model-id meta-llama/Meta-Llama-3-8B]

Outputs JSON files: prompt_25.json, prompt_50.json, etc.
Each file contains {"token_ids": [int, ...], "length": int, "text": str}.
"""

import argparse
import json
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_LENGTHS = [25, 50, 100, 200]
MODEL_ID = "meta-llama/Meta-Llama-3-8B"


def sample_prompts(output_dir, target_lengths, model_id):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Loading LMSys-Chat-1M dataset (first shard)...")
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train",
                      streaming=True)

    # Tokenize conversations, collect (token_ids, text, length) tuples
    print("Tokenizing conversations...")
    candidates = []  # (token_ids, text, length)
    max_target = max(target_lengths)
    count = 0
    for row in ds:
        # Use the first human turn of each conversation
        conv = row.get("conversation", [])
        if not conv:
            continue
        human_turns = [t for t in conv if t.get("role") == "user"]
        if not human_turns:
            continue
        text = human_turns[0]["content"]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) >= 10:  # skip very short prompts
            candidates.append((token_ids, text, len(token_ids)))
        count += 1
        if count % 10000 == 0:
            print(f"  Processed {count} conversations, "
                  f"{len(candidates)} valid candidates...")
        # Stop after enough candidates (100k should be plenty)
        if count >= 100000:
            break

    print(f"  Total candidates: {len(candidates)}")

    # For each target length, find the closest prompt
    for target in target_lengths:
        best = min(candidates, key=lambda x: abs(x[2] - target))
        token_ids, text, length = best

        # Truncate or take as-is
        if length > target:
            token_ids = token_ids[:target]
            length = target

        out = {
            "token_ids": token_ids,
            "length": length,
            "text": text[:200] + ("..." if len(text) > 200 else ""),
        }
        out_path = os.path.join(output_dir, f"prompt_{target}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Target {target}: selected {length} tokens -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="measure/prompts")
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=DEFAULT_LENGTHS)
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()
    sample_prompts(args.output_dir, args.lengths, args.model_id)
