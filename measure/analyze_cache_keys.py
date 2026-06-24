#!/usr/bin/env python3
"""§7.3a: Cache key near-miss clustering analysis.

Reads the dump from AQP_DUMP_CACHE_KEYS and determines how many additional
cache hits parameterized compilation would yield (i.e., keys identical except
for filter constant values).
"""

import re
import sys
from collections import defaultdict


def parse_dump(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = {}
            for part in line.split('\t'):
                k, _, v = part.partition('=')
                fields[k] = v
            entries.append(fields)
    return entries


def strip_filter_constants(plan):
    """Replace filter constant values with a placeholder to find near-misses.

    Filter constants in the plan look like:
      - "Integer const value: 2005"  -> "Integer const value: <CONST>"
      - "Float const value: 3.14"    -> "Float const value: <CONST>"
      - "String const value: \"foo\"" -> "String const value: <CONST>"
      - "Bool const value: 1"         -> "Bool const value: <CONST>"
      - Literal values in quotes: ' "something" '
    """
    s = plan
    # Integer const value: <number>
    s = re.sub(r'Integer const value: -?\d+', 'Integer const value: <CONST>', s)
    # Float const value: <number>
    s = re.sub(r'Float const value: -?[\d.]+', 'Float const value: <CONST>', s)
    # String const value: "..."
    s = re.sub(r'String const value: "(?:[^"\\]|\\.)*"', 'String const value: <CONST>', s)
    # Bool const value: 0|1
    s = re.sub(r'Bool const value: [01]', 'Bool const value: <CONST>', s)
    # Standalone quoted literals: " ... "
    s = re.sub(r' "(?:[^"\\]|\\.)*" ', ' <LITERAL> ', s)
    return s


def strip_column_names(plan):
    """Additionally strip column names from the plan, keeping only indices+dtypes.

    Column entries look like: |table_index.column_index.column_name.dtype
    Replace column_name with <NAME>.
    """
    # Column in step cols: |N.N.name.N
    s = re.sub(r'\|(\d+)\.(\d+)\.[^.]+\.(\d+)', r'|\1.\2.<NAME>.\3', plan)
    # Source table name: S{tablename#N
    s = re.sub(r'S\{[^#]+#', 'S{<TABLE>#', s)
    return s


def analyze(entries):
    plans = [e['PLAN'] for e in entries]
    hits = [e for e in entries if e.get('HIT') == '1']
    misses = [e for e in entries if e.get('HIT') == '0']

    print(f"Total sub-queries: {len(entries)}")
    print(f"Cache hits: {len(hits)} ({100*len(hits)/len(entries):.1f}%)")
    print(f"Cache misses: {len(misses)} ({100*len(misses)/len(entries):.1f}%)")
    print()

    # Current: unique exact plans
    unique_exact = len(set(plans))
    print(f"Unique exact plans: {unique_exact}")

    # After stripping filter constants
    stripped_plans = [strip_filter_constants(p) for p in plans]
    unique_stripped = len(set(stripped_plans))
    print(f"Unique plans (constants stripped): {unique_stripped}")
    print(f"Near-miss groups (would merge): {unique_exact - unique_stripped}")
    print()

    # Compute potential additional hits
    # Group all plans by their stripped version
    groups = defaultdict(list)
    for i, sp in enumerate(stripped_plans):
        groups[sp].append(i)

    additional_hits = 0
    near_miss_groups = []
    for sp, indices in groups.items():
        exact_variants = set(plans[i] for i in indices)
        if len(exact_variants) > 1:
            # This group has multiple distinct constant-value variants.
            # With parameterized compilation, only the first would compile;
            # all subsequent would be cache hits.
            group_total = len(indices)
            # Current hits in this group: how many times the same exact plan appeared before
            exact_counts = defaultdict(int)
            current_hits_in_group = 0
            for i in indices:
                p = plans[i]
                if exact_counts[p] > 0:
                    current_hits_in_group += 1
                exact_counts[p] += 1
            # With parameterization: all but the first occurrence are hits
            param_hits = group_total - 1
            new_additional = param_hits - current_hits_in_group
            additional_hits += new_additional
            near_miss_groups.append({
                'stripped': sp[:120] + '...',
                'variants': len(exact_variants),
                'occurrences': group_total,
                'current_hits': current_hits_in_group,
                'param_hits': param_hits,
                'additional': new_additional,
            })

    near_miss_groups.sort(key=lambda g: g['additional'], reverse=True)

    print(f"=== Near-miss analysis (filter constants stripped) ===")
    print(f"Additional hits from parameterization: {additional_hits}")
    print(f"Current hit rate: {len(hits)}/{len(entries)} = {100*len(hits)/len(entries):.1f}%")
    new_total_hits = len(hits) + additional_hits
    print(f"Projected hit rate: {new_total_hits}/{len(entries)} = {100*new_total_hits/len(entries):.1f}%")
    print()

    if near_miss_groups:
        print(f"Top near-miss groups (by additional hits):")
        for i, g in enumerate(near_miss_groups[:20]):
            print(f"  [{i}] {g['variants']} variants, {g['occurrences']} occurrences, "
                  f"+{g['additional']} additional hits (current: {g['current_hits']}, "
                  f"param: {g['param_hits']})")
            print(f"      Plan: {g['stripped']}")
        print()

    # Also check: column name stripping
    double_stripped = [strip_column_names(strip_filter_constants(p)) for p in plans]
    unique_double = len(set(double_stripped))
    print(f"=== Column name stripping (constants + names) ===")
    print(f"Unique plans (constants + column names stripped): {unique_double}")
    print(f"Additional merges from column name stripping: {unique_stripped - unique_double}")

    # Compute additional hits from double-stripping
    groups2 = defaultdict(list)
    for i, ds in enumerate(double_stripped):
        groups2[ds].append(i)
    additional_hits_2 = 0
    for ds, indices in groups2.items():
        exact_variants = set(plans[i] for i in indices)
        if len(exact_variants) > 1:
            group_total = len(indices)
            exact_counts = defaultdict(int)
            current_hits_in_group = 0
            for i in indices:
                p = plans[i]
                if exact_counts[p] > 0:
                    current_hits_in_group += 1
                exact_counts[p] += 1
            param_hits = group_total - 1
            additional_hits_2 += param_hits - current_hits_in_group
    print(f"Additional hits from constants + names stripping: {additional_hits_2}")
    new_total_2 = len(hits) + additional_hits_2
    print(f"Projected hit rate: {new_total_2}/{len(entries)} = {100*new_total_2/len(entries):.1f}%")

    # Decision gate
    print()
    print(f"=== Decision gate ===")
    if additional_hits > 50:
        print(f"RESULT: {additional_hits} near-misses > 50 threshold.")
        print(f"Parameterized compilation IS worth investigating.")
    elif additional_hits > 20:
        print(f"RESULT: {additional_hits} near-misses, borderline (20-50).")
        print(f"Marginal benefit; consider only if implementation cost is low.")
    else:
        print(f"RESULT: {additional_hits} near-misses < 20 threshold.")
        print(f"21.8% ceiling confirmed. Parameterized compilation NOT worth it.")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cache_keys_dump.txt'
    entries = parse_dump(path)
    analyze(entries)
