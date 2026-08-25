#!/usr/bin/env bash
set -e

# DuckDB: vanilla vs. node-based vs. relationship-center vs. topdown
bash measure_breakdown_time_aqp.sh job duckdb none none
bash measure_breakdown_time_aqp.sh job duckdb node-based none
bash measure_breakdown_time_aqp.sh job duckdb relationship-center none
bash measure_breakdown_time_aqp.sh job duckdb topdown none

# PostgreSQL: vanilla vs. node-based vs. relationship-center vs. topdown
bash measure_breakdown_time_aqp.sh job postgresql none none
bash measure_breakdown_time_aqp.sh job postgresql node-based none
bash measure_breakdown_time_aqp.sh job postgresql relationship-center none
bash measure_breakdown_time_aqp.sh job postgresql topdown none

# Umbra: vanilla (direct) vs. node-based vs. relationship-center vs. topdown
bash measure_umbra.sh job
bash measure_breakdown_time_aqp.sh job umbra node-based none
bash measure_breakdown_time_aqp.sh job umbra relationship-center none
bash measure_breakdown_time_aqp.sh job umbra topdown none
