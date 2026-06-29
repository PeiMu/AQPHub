# Storage Plan Design: Auxiliary In-Memory Structures for AQP Middleware

## Motivation

After 24 iterations of JIT optimization within DuckDB's operator framework, we achieved -7.3% execution improvement. The remaining bottlenecks (InsertHashes 5-17%, Gather 3-9%, fsst_decompress 3-15%, FastMemcpy 3-28%) are all inside DuckDB's internal execution paths and cannot be improved without bypassing DuckDB's execution engine.

Analysis of BespokeOLAP and GenDB — systems that are 14-173x faster than DuckDB on JOB queries — shows that ~90% of their speedup comes from **storage plan** optimizations (CSR indexes, flat arrays, sorted indices, bitmaps), not from tighter compiled loops (~10%).

This design introduces auxiliary in-memory data structures that the AQP middleware builds and owns, enabling it to execute sub-queries directly instead of delegating everything to DuckDB. **DuckDB (and PostgreSQL) remain completely unchanged** — they serve as data source at startup and SQL parser at query time.

## Architecture Overview

```
Startup (one-time, ~1-3 seconds):
  1. Middleware opens DuckDB/PostgreSQL connection (existing code)
  2. Parse --schema and --fkeys to learn table structure + FK relationships
  3. SELECT * from all tables → build flat column arrays in middleware memory
  4. Build CSR indexes on all FK columns (from fkeys.sql)
  5. Build sorted indices on frequently aggregated columns
  6. Identify dimension constants (tiny tables with |D| < threshold)

Query execution (per query):
  1. DuckDB/PostgreSQL parses SQL → logical plan (engine is just a parser)
  2. Node-based split → sequence of sub-query plans
  3. For each sub-query:
     a. Kernel executor: scan flat arrays + filter + CSR join with temps
     b. Build CSR on temp result (runtime info for next sub-query)
  4. Final: aggregation (with sorted index early termination if MIN/MAX)
  5. Return results

Fallback: if a sub-query uses unsupported operations → execute via DuckDB
```

**Key principle**: Native DuckDB/PostgreSQL is NOT changed and does NOT benefit from these structures. All auxiliary structures live in middleware memory. The engine is used only for data loading (startup) and SQL parsing (query time). When the middleware is not used, the engine behaves exactly as before.

## Component 1: Flat Column Arrays

### What
Decompress all base table data from DuckDB's compressed storage (FSST, dictionary, constant segments) into plain C arrays in middleware memory. Each column becomes a contiguous array indexed by row position.

### Why
DuckDB access: `segment_manager → decompress(segment) → FSST_decode → value` (multiple indirections, decompression CPU cost).
Flat array access: `column_array[row_id]` (single pointer dereference).

`fsst_decompress` and `StringScanPartial` account for 3-15% CPU in perf profiles of heavy queries (19d, 6f, 25c). Flat arrays eliminate this entirely.

### Data model
```cpp
struct FlatColumn {
    enum Type { INT32, INT64, VARCHAR };
    Type type;
    idx_t count;                      // number of rows
    // For INT32/INT64:
    void* data;                       // int32_t[] or int64_t[], indexed by row position
    // For VARCHAR:
    std::vector<std::string> strings; // indexed by row position
    // Null bitmap:
    uint8_t* nulls;                   // 1 bit per row, 1 = null
};

struct FlatTable {
    std::string name;
    idx_t row_count;
    std::vector<std::string> column_names;
    std::vector<FlatColumn> columns;
    // PK column index (always "id" in IMDB schema)
    int pk_col_idx;
    // ID-indexed lookup: if PK is dense integer starting near 0,
    // we can use id_to_row[pk_value] = row_position for O(1) PK lookup
    std::vector<idx_t> id_to_row;     // maps PK value → row position
    idx_t max_id;                     // max PK value (for array sizing)
};
```

### IMDB memory budget
| Table | Rows | Est. uncompressed | Notes |
|-------|------|-------------------|-------|
| cast_info | 36.2M | ~1.0 GB | 7 columns, strings in note |
| movie_info | 14.8M | ~0.8 GB | info + note are text |
| name | 4.2M | ~0.3 GB | name is text |
| movie_keyword | 4.5M | ~54 MB | 3 int columns |
| char_name | 3.1M | ~0.2 GB | name is text |
| person_info | 3.0M | ~0.2 GB | info + note |
| movie_companies | 2.6M | ~0.15 GB | note is text |
| title | 2.5M | ~0.2 GB | title is text, 12 columns |
| movie_info_idx | 1.4M | ~0.08 GB | info is text |
| Small tables (12) | <1M total | ~0.05 GB | |
| **Total** | | **~3.1 GB** | On 63 GB machine |

### Loading
```cpp
// Pseudocode — actual implementation in src/storage/
void StoragePlan::LoadFlatTables(DatabaseAdapter& adapter) {
    for (auto& table_def : schema_.tables) {
        // Issue SELECT * through the adapter (works for DuckDB, PostgreSQL, etc.)
        auto result = adapter.ExecuteQuery("SELECT * FROM " + table_def.name);
        FlatTable ft;
        ft.name = table_def.name;
        // ... populate ft.columns from result rows ...
        // Build id_to_row mapping if PK is integer
        if (table_def.has_integer_pk) {
            ft.max_id = max_pk_value;
            ft.id_to_row.resize(ft.max_id + 1, INVALID_IDX);
            for (idx_t row = 0; row < ft.row_count; row++) {
                int32_t pk = ft.columns[ft.pk_col_idx].GetInt32(row);
                ft.id_to_row[pk] = row;
            }
        }
        tables_[table_def.name] = std::move(ft);
    }
}
```

### Engine independence
The loading code uses the adapter abstraction (`DatabaseAdapter::ExecuteQuery`). It works identically for DuckDB, PostgreSQL, MySQL, etc. The engine's internal storage format is irrelevant — we just read result sets.

## Component 2: CSR Indexes (Compressed Sparse Row)

### What
For every foreign key relationship declared in `--fkeys`, build a CSR index that maps FK value → list of row positions in the child table. This replaces hash table build + probe with O(1) array lookup.

### How CSR works
Given `movie_keyword(keyword_id)` → `keyword(id)`:
```
CSR index "movie_keyword.keyword_id":
  offsets[keyword_id]     = start position in row_ids[] for this keyword
  offsets[keyword_id + 1] = end position (exclusive)
  row_ids[]               = list of movie_keyword row positions, grouped by keyword_id

Lookup: "which movie_keyword rows have keyword_id = 42?"
  begin = offsets[42]      // e.g., 500
  end   = offsets[43]      // e.g., 503
  answer = row_ids[500..502]  // 3 rows
```

### Why this replaces hash join
DuckDB hash join:
1. Build: scan one side, hash every key, insert into hash table → random writes, cache misses (`InsertHashes` 5-17%)
2. Probe: scan other side, hash each key, follow hash chain → random reads, cache misses (`AdvancePointers` 6-8%)
3. Cost: O(n + m) but with poor cache behavior

CSR join:
1. Pre-built at load time (one-time cost)
2. Lookup: `offsets[key]` → direct array range, no hashing, no chains, no collisions
3. Cost: O(1) per lookup, sequential scan of row_ids range

### Data model
```cpp
struct CSRIndex {
    std::string child_table;   // e.g., "movie_keyword"
    std::string fk_column;     // e.g., "keyword_id"
    std::string parent_table;  // e.g., "keyword"

    idx_t max_parent_id;       // max value of referenced PK
    std::vector<idx_t> offsets; // size = max_parent_id + 2 (one extra for sentinel)
    std::vector<idx_t> row_ids; // size = child_table.row_count

    // Lookup: rows in child_table where fk_column == parent_id
    // begin = offsets[parent_id], end = offsets[parent_id + 1]
    // row_ids[begin..end) are the matching row positions
};
```

### IMDB FK relationships (from fkeys.sql)
| Child table → Parent | FK column | Max parent ID | Est. CSR size |
|----------------------|-----------|---------------|---------------|
| aka_name → name | person_id | 4,167,491 | 16 MB offsets + 7 MB row_ids |
| aka_title → kind_type | kind_id | 7 | negligible |
| aka_title → title | movie_id | 2,528,312 | 10 MB + 3 MB |
| cast_info → name | person_id | 4,167,491 | 16 MB + 145 MB |
| cast_info → title | movie_id | 2,528,312 | 10 MB + 145 MB |
| cast_info → char_name | person_role_id | 3,140,339 | 12 MB + 145 MB |
| cast_info → role_type | role_id | 12 | negligible |
| complete_cast → title | movie_id | 2,528,312 | 10 MB + 1 MB |
| complete_cast → comp_cast_type | subject_id/status_id | 4 | negligible |
| movie_companies → title | movie_id | 2,528,312 | 10 MB + 10 MB |
| movie_companies → company_name | company_id | 234,997 | 1 MB + 10 MB |
| movie_companies → company_type | company_type_id | 4 | negligible |
| movie_keyword → title | movie_id | 2,528,312 | 10 MB + 18 MB |
| movie_keyword → keyword | keyword_id | 134,170 | 0.5 MB + 18 MB |
| movie_info → title | movie_id | 2,528,312 | 10 MB + 59 MB |
| movie_info → info_type | info_type_id | 113 | negligible |
| movie_info_idx → title | movie_id | 2,528,312 | 10 MB + 6 MB |
| movie_info_idx → info_type | info_type_id | 113 | negligible |
| movie_link → title | movie_id / linked_movie_id | 2,528,312 | 10 MB + 0.2 MB |
| movie_link → link_type | link_type_id | 18 | negligible |
| person_info → name | person_id | 4,167,491 | 16 MB + 12 MB |
| person_info → info_type | info_type_id | 39 | negligible |
| title → kind_type | kind_id | 7 | negligible |
| **Total** | | | **~150 MB offsets + ~580 MB row_ids ≈ 730 MB** |

### Building CSR (O(n) counting sort)
```cpp
void StoragePlan::BuildCSR(const FlatTable& child, int fk_col_idx,
                            idx_t max_parent_id, CSRIndex& csr) {
    csr.max_parent_id = max_parent_id;
    csr.offsets.resize(max_parent_id + 2, 0);
    csr.row_ids.resize(child.row_count);

    // Pass 1: count occurrences of each FK value
    auto fk_data = (int32_t*)child.columns[fk_col_idx].data;
    for (idx_t i = 0; i < child.row_count; i++) {
        csr.offsets[fk_data[i] + 1]++;
    }
    // Pass 2: prefix sum
    for (idx_t i = 1; i <= max_parent_id + 1; i++) {
        csr.offsets[i] += csr.offsets[i - 1];
    }
    // Pass 3: scatter row_ids (uses a copy of offsets as write cursors)
    std::vector<idx_t> cursors(csr.offsets.begin(), csr.offsets.end());
    for (idx_t i = 0; i < child.row_count; i++) {
        idx_t fk = fk_data[i];
        csr.row_ids[cursors[fk]++] = i;
    }
}
```

Build time: O(n) per FK relationship. For cast_info (36M rows), ~0.3s. Total for all FKs: ~1-2s.

### Runtime CSR on temp tables
After each sub-query execution, the middleware builds a CSR on the temp table's join key column. This is the same algorithm but on much smaller data (thousands to millions of rows → microseconds to milliseconds).

```cpp
// After sub-query 1 produces temp_result with join key column
CSRIndex temp_csr;
BuildCSR(temp_result, join_key_col_idx, max_key_value, temp_csr);
// Sub-query 2 uses temp_csr for O(1) lookup instead of hash join
```

This is **runtime information** — the CSR is built from actual sub-query results, adapting to the query's actual selectivity and intermediate cardinalities.

## Component 3: Sorted Indices

### What
For columns used in MIN/MAX aggregations, maintain a sorted permutation array. This enables early termination: scan in sorted order, stop at the first row that satisfies all join/filter predicates.

### Data model
```cpp
struct SortedIndex {
    std::string table_name;
    std::string column_name;
    // sorted_perm[0] = row with smallest value
    // sorted_perm[1] = row with next smallest value, etc.
    std::vector<idx_t> sorted_perm;
};
```

### Example: MIN(title.title)
Without sorted index: scan all matching titles, compare strings → O(n) string comparisons.
With sorted index: scan `title_sorted_by_title[0], [1], [2], ...`, check if row matches join predicates, stop at first match → O(k) where k is the rank of the minimum matching title. For typical JOB queries, k < 100.

### Which columns to index
JOB queries use MIN on: `title.title`, `name.name`, `char_name.name`, `movie_companies.note`, `company_name.name`, `movie_info.info`, `movie_info_idx.info`, `aka_name.name`. We can build sorted indices on all of these (one-time sort per column at load time).

### Build cost
Sorting 36M strings (cast_info): ~2-3s. But we only need sorted indices on columns that appear in MIN/MAX in JOB queries. Sorting 2.5M titles: ~0.3s. Total for all needed columns: ~1-2s.

### When to use
Only for the final aggregation step (last sub-query with MIN/MAX). The kernel executor checks if the aggregation is a MIN/MAX on a column with a sorted index, and switches to the early-termination scan.

## Component 4: Dimension Constants

### What
Tiny dimension tables (|D| < threshold, e.g., < 200 rows) can have their contents cached in the middleware. Join predicates like `WHERE kt.kind = 'movie'` can be resolved to `kind_type_id = 1` at parse time, eliminating the join entirely.

### IMDB dimension tables
| Table | Rows | Used for |
|-------|------|----------|
| kind_type | 7 | `kind = 'movie'` → `kind_id = 1` |
| company_type | 4 | `kind = 'production companies'` → `company_type_id = 2` |
| role_type | 12 | `role = 'actor'` → `role_id = 1` |
| comp_cast_type | 4 | `kind = 'cast'` → `id = 2` |
| info_type | 113 | `info = 'genres'` → `info_type_id = 3` |
| link_type | 18 | `link = 'follows'` → `link_type_id = 1` |

### Implementation
```cpp
struct DimensionCache {
    std::string table_name;
    std::unordered_map<std::string, std::unordered_map<std::string, int32_t>> value_to_id;
    // e.g., value_to_id["kind"]["movie"] = 1
};
```

At query parse time, the middleware detects joins with dimension tables and replaces them with constant equality predicates. This is a query rewrite, not a storage optimization, but it depends on having the dimension data cached.

### Engine independence
Dimension constants are read from the engine at startup via `SELECT * FROM kind_type` etc. Works identically for DuckDB, PostgreSQL, any SQL engine.

## Component 5: Runtime Temp Table Structures

### What
After each sub-query execution, the result is a "temp table" — a flat array of columns produced by the kernel executor. The middleware builds auxiliary structures on these temp tables for use by subsequent sub-queries.

### What we build at runtime
1. **CSR index on the join key**: for the next sub-query to join with this temp table via O(1) lookup
2. **Cardinality info**: for the split strategy to decide execution order (Direction B integration)
3. **Value range info**: min/max of join keys, for bitmap sizing

### This IS runtime information
These structures are built from actual sub-query execution results. They adapt to:
- The query's actual filter selectivity (not estimated cardinality)
- The actual intermediate result sizes
- The actual join key distributions

This is the key advantage over BespokeOLAP (which pre-builds everything for worst case) — we build what we need, when we need it, sized to actual data.

## Execution Model: SubQueryPlan + Generic Executor

### Sub-query pattern analysis

Analysis of all 113 JOB queries with node-based split reveals ~640 total sub-queries falling into highly regular patterns:

**Structural patterns (ignoring filter type variations)**:

| Rank | Pattern | Count | Cumulative | Description |
|------|---------|-------|------------|-------------|
| 1 | `1base + 1temp` | 170 | 25.8% | Join base table with one temp result |
| 2 | `1base + 1dim` | 79 | 37.7% | Base table filtered via dimension table |
| 3 | `2base` | 63 | 47.3% | Two base tables joined (first sub-query, no temps yet) |
| 4 | `1base + 1temp + FINAL` | 57 | 55.9% | Final aggregation: base + temp → MIN |
| 5 | `1base + 1dim + 1temp` | 48 | 63.2% | Base + dimension + temp joined |
| 6 | `2temp` | 47 | 70.3% | Intersect two temp results |
| 7 | `2base + 1temp` | 38 | 76.1% | Two base tables + one temp |
| 8 | `1dim + 1temp` | 30 | 80.6% | Dimension filter on temp result |
| 9 | `3base` | 25 | 84.4% | Three base tables joined |
| 10 | `2base + 1temp + FINAL` | 22 | 87.7% | Final: two base + temp → MIN |
| 11 | `4base` | 15 | 90.0% | Four base tables (large first sub-query) |
| 12 | `2base + 1dim` | 13 | 92.0% | Two base + dimension |
| 13 | `1base + 1dim + 1temp + FINAL` | 12 | 93.8% | Final with dim + temp |
| 14 | `1dim + 2temp` | 12 | 95.6% | Dimension filter on two temps |
| 15 | `2temp + FINAL` | 9 | 97.0% | Final: intersect two temps + MIN |

**Key observations**:
- 6 patterns cover 70% of all sub-queries
- 15 patterns cover 97%
- Single-base + temps (patterns 1, 4, 5, 8) = 48% — fully CSR-joinable
- Multi-base (patterns 3, 7, 9, 11) = 16% — require hash join or DuckDB fallback
- Temp-only (patterns 6, 15) = 9% — intersections of previous results

**Filter type distribution** (within patterns):
- Equality (`=`): most common, used in 41 pattern variants
- Range (`>=`, `<=`): 36 occurrences
- LIKE: 33 occurrences (including NOT LIKE)
- IN: 18 occurrences
- Combined (e.g., eq+range+like): 12 occurrences

**Sub-queries per query**:
- 2-4 sub-queries: 29 queries (simple queries)
- 5-7 sub-queries: 55 queries (typical)
- 8-12 sub-queries: 29 queries (complex queries like 33a)

### Design: SubQueryPlan struct + single executor

Rather than separate primitive functions (which would require intermediate materialization between calls) or fixed templates (which explode combinatorially), we use a **plan struct** that describes each sub-query's operations, and a **single generic executor function** that runs the plan in one scan loop.

```cpp
// src/storage/sub_query_plan.h

struct Predicate {
    int col_idx;                  // column index in scan table
    enum Op { EQ, NE, LT, LE, GT, GE, LIKE, NOT_LIKE, IN, IS_NULL, IS_NOT_NULL };
    Op op;
    // Value (one of):
    int32_t int_val;
    std::string str_val;
    std::vector<int32_t> in_vals; // for IN predicates
};

struct JoinStep {
    CSRIndex* csr;                // CSR index to use for lookup
    int scan_fk_col_idx;          // which column of the scanned table is the FK
    FlatTable* joined_table;      // table being joined (for inner join column access)
    bool is_semi;                 // semi-join (filter) vs inner join (expand)
    // For inner join: which columns from joined_table to include in output
    std::vector<int> joined_col_indices;
};

struct ProjectColumn {
    enum Source { SCAN_TABLE, JOINED_TABLE };
    Source source;
    int join_idx;                 // which JoinStep (if source == JOINED_TABLE)
    int col_idx;                  // column index in source table
    std::string output_name;
};

struct SubQueryPlan {
    // What to scan (exactly one base table or temp table)
    FlatTable* scan_table;
    std::vector<Predicate> scan_filters;

    // Joins: applied as filters/expansions during scan (0-3 steps)
    std::vector<JoinStep> joins;

    // Output projection
    std::vector<ProjectColumn> output_cols;

    // Aggregation (final sub-query only)
    enum AggType { NONE, MIN, COUNT };
    AggType agg_type = NONE;
    SortedIndex* sorted_idx = nullptr; // for MIN early termination

    // Fallback: if true, execute via DuckDB instead of kernel
    bool use_duckdb_fallback = false;
};
```

### Generic executor (single-loop, no intermediate materialization)

```cpp
// src/storage/sub_query_plan.cpp

void ExecuteSubQuery(const SubQueryPlan& plan, FlatTable& output) {
    if (plan.use_duckdb_fallback) {
        // Generate SQL and execute via DuckDB adapter (existing path)
        return;
    }

    for (idx_t i = 0; i < plan.scan_table->row_count; i++) {
        // 1. Apply scan filters (on the base/temp table being scanned)
        if (!EvalFilters(*plan.scan_table, i, plan.scan_filters))
            continue;

        // 2. Apply join steps as filters (CSR lookup)
        bool all_match = true;
        for (const auto& join : plan.joins) {
            int32_t fk = plan.scan_table->GetInt32(join.scan_fk_col_idx, i);
            auto [begin, end] = join.csr->Lookup(fk);
            if (begin == end) { all_match = false; break; }
            // For inner join: would iterate [begin, end) and expand rows
            // For semi-join: just check non-empty
        }
        if (!all_match) continue;

        // 3. Project + output (or update aggregate)
        if (plan.agg_type == SubQueryPlan::NONE) {
            output.AppendRow(plan, i);
        } else if (plan.agg_type == SubQueryPlan::MIN) {
            UpdateMin(plan, i);
        }
    }
}
```

**For MIN with sorted index** (final sub-query only):
```cpp
void ExecuteSubQuerySortedMin(const SubQueryPlan& plan, FlatTable& output) {
    // Scan in sorted order, stop at first match
    for (idx_t rank = 0; rank < plan.sorted_idx->sorted_perm.size(); rank++) {
        idx_t row = plan.sorted_idx->sorted_perm[rank];
        if (!EvalFilters(*plan.scan_table, row, plan.scan_filters))
            continue;
        // Check all joins
        bool all_match = true;
        for (const auto& join : plan.joins) {
            int32_t fk = plan.scan_table->GetInt32(join.scan_fk_col_idx, row);
            auto [begin, end] = join.csr->Lookup(fk);
            if (begin == end) { all_match = false; break; }
        }
        if (all_match) {
            // This is the MIN — stop
            output.SetResult(plan, row);
            return;
        }
    }
}
```

### Sub-Query Composer (template selector)

Maps a sub-IR (from the node-based splitter) to a SubQueryPlan:

```cpp
// src/storage/sub_query_composer.cpp

SubQueryPlan ComposeSubQuery(const SubIR& sub_ir, const StoragePlan& storage) {
    SubQueryPlan plan;

    // 1. Identify the primary scan table
    plan.scan_table = storage.GetFlatTable(sub_ir.primary_table);

    // 2. Convert filters to Predicate structs
    for (const auto& filter : sub_ir.filters) {
        plan.scan_filters.push_back(ConvertFilter(filter, plan.scan_table));
    }

    // 3. For each join in sub_ir, look up the CSR index
    for (const auto& join : sub_ir.joins) {
        JoinStep step;
        step.csr = storage.GetCSR(join.child_table, join.fk_column);
        step.scan_fk_col_idx = plan.scan_table->GetColumnIndex(join.fk_column);
        step.joined_table = storage.GetFlatTable(join.parent_table);
        step.is_semi = join.is_semi;
        plan.joins.push_back(step);
    }

    // 4. Multi-base sub-queries: fallback to DuckDB
    if (sub_ir.base_table_count > 1 && !CanConvertToCSRJoins(sub_ir, storage)) {
        plan.use_duckdb_fallback = true;
    }

    // 5. Aggregation (final sub-query)
    if (sub_ir.has_aggregation) {
        plan.agg_type = SubQueryPlan::MIN; // or COUNT
        plan.sorted_idx = storage.GetSortedIndex(sub_ir.agg_column);
    }

    return plan;
}
```

This maps to the existing sub-plan selector concept: first select the sub-plan (node-based splitter), then select the execution strategy (SubQueryPlan vs DuckDB fallback).

### Why not separate primitives?

The original design proposed separate kernel functions (ScanFilter, CSRSemiJoin, CSRInnerJoin, Project, etc.) called in sequence. The problem: each primitive materializes intermediate results. For a sub-query with scan + 2 filters + 1 CSR join + project, that's 4 separate passes over the data with intermediate ID vectors.

The SubQueryPlan executor does everything in **one scan loop**: filter, CSR lookup, and project all happen per-row without intermediate materialization. This is the same insight that makes BespokeOLAP fast — everything in one tight loop per pipeline stage.

### When would we need a DSL or multi-stage programming?

The generic `ExecuteSubQuery()` handles all 15 structural patterns via the plan struct. It has interpretation overhead: checking `plan.joins.size()`, branching on `agg_type`, iterating the filter vector, etc. For v1, this overhead is estimated at <5% vs. the CSR + flat array gains.

If measurement shows this interpretation overhead is significant (e.g., >10% of execution time on small-cardinality sub-queries), consider:
1. **JIT-compiling SubQueryPlan** into specialized code (Step 7 kernel fusion): generate LLVM IR from the plan struct, unroll the join loop for the exact number of joins, specialize filter evaluation for the exact predicate types. This is Neumann-style query compilation, but at the sub-query granularity.
2. **Lightweight DSL with multi-stage programming** (Shaikhha-style): define sub-query logic in a staged IR, where Stage 1 specializes for query structure (number of joins, filter types), Stage 2 specializes for data statistics (cardinalities, selectivities). The output is specialized C++ or LLVM IR.

Both approaches are deferred until we have measurement data showing the generic executor's interpretation overhead is a real bottleneck.

### Performance estimate vs. BespokeOLAP

**This design achieves 5-20x over DuckDB, not the full 50-170x of BespokeOLAP.** The gap:

| Aspect | Our design | BespokeOLAP | Gap |
|--------|-----------|-------------|-----|
| Join method | CSR O(1) lookup | CSR O(1) lookup | Same |
| Column access | Flat array `col[row]` | Flat array `col[row]` | Same |
| Scan filters | Generic predicate eval | Compiled inline code | Small (~5%) |
| Temp materialization | Yes (between sub-queries) | None (single loop) | Significant |
| Query-specific bitmaps | No | Yes (e.g., `us_movie_bitmap`) | Moderate |
| Loop fusion | No (per-sub-query loops) | Yes (whole-query loop) | Significant |
| MIN optimization | Sorted index early termination | Sorted index early termination | Same |

To close the gap: kernel fusion (Step 7, JIT) + Direction B (fewer sub-queries → fewer materializations) + runtime bitmaps for high-selectivity semi-joins.

## Code Organization

### Non-JIT code (auxiliary storage infrastructure)
These components are general middleware features, not JIT-specific:

```
src/storage/                              (NEW directory)
    storage_plan.h / storage_plan.cpp         — top-level: owns FlatTables, CSRIndexes, SortedIndices, DimensionCaches
    flat_table.h / flat_table.cpp             — FlatTable + FlatColumn data structures and loading
    csr_index.h / csr_index.cpp               — CSR build (base tables) + runtime CSR build (temp tables)
    sorted_index.h / sorted_index.cpp         — sorted permutation arrays
    dimension_cache.h / dimension_cache.cpp   — tiny table caching + query rewrite helpers
    sub_query_plan.h / sub_query_plan.cpp     — SubQueryPlan struct + ExecuteSubQuery() generic executor
    sub_query_composer.h / sub_query_composer.cpp — maps sub-IR to SubQueryPlan (template selector)

include/storage/
    storage_plan.h
    flat_table.h
    csr_index.h
    sorted_index.h
    dimension_cache.h
    sub_query_plan.h
    sub_query_composer.h
```

### JIT code (kernel fusion, if pursued later)
If we later want to JIT-compile SubQueryPlan into specialized loops:

```
src/jit/
    kernel_codegen.cpp                   — LLVM IR generation from SubQueryPlan (Step 7)
    (existing: ir_to_llvm.cpp)           — current expr/operator/pipeline JIT
```

### Integration points
```
src/adapters/duckdb_adapter.cpp          — add StoragePlan loading at connection init
src/split/ir_query_splitter.cpp          — sub-query plans reference flat tables + CSR indexes
src/aqp_middleware.cpp                   — invoke StoragePlan::Load() before query execution
```

## Implementation Plan

Implementation proceeds step by step, with **breakdown measurements after each component** to verify the effect independently.

### Step 1: Flat Column Arrays + Basic Scan
1. Create `src/storage/` directory and `flat_table.h/.cpp`
2. Implement `FlatTable` loading from DuckDB adapter (SELECT *)
3. Implement `id_to_row` mapping for PK lookup
4. Implement basic `EvalFilters` for integer equality/range and string equality/LIKE predicates
5. Wire into middleware startup: load tables when `--storage-plan` flag is set
6. **Measure**: compare scan speed on a single-table query (e.g., keyword filter) vs DuckDB scan. Expect: similar or slightly faster (no decompression).

### Step 2: CSR Indexes on Base Tables
1. Implement `csr_index.h/.cpp` with `BuildCSR()`
2. Parse `fkeys.sql` to discover all FK relationships
3. Build CSR for all FK columns at startup
4. Implement CSR lookup in `ExecuteSubQuery()` JoinStep
5. **Measure**: single FK join (e.g., movie_keyword.keyword_id → keyword) via CSR vs DuckDB hash join. Expect: 5-20x faster per join.

### Step 3: SubQueryPlan Executor + Runtime CSR on Temp Tables
1. Implement `sub_query_plan.h/.cpp` with `ExecuteSubQuery()` generic executor
2. Implement `sub_query_composer.h/.cpp` to map sub-IR → SubQueryPlan
3. After each sub-query execution, build runtime CSR on temp result's join key
4. Wire into `ir_query_splitter.cpp`: use SubQueryPlan executor instead of generating SQL for DuckDB
5. Implement DuckDB fallback for multi-base sub-queries that can't be CSR-joined
6. **Measure**: full node-based split execution on 16b, 8c (highest regression queries). Expect: significant reduction in per-sub-query execution time.

### Step 4: Dimension Constants
1. Implement `dimension_cache.h/.cpp`
2. At startup, cache all tables with < 200 rows
3. In the query parser/splitter, detect joins with dimension tables and rewrite to constant predicates
4. **Measure**: queries with dimension joins (most JOB queries use kind_type, info_type). Expect: one fewer join per eliminated dimension table.

### Step 5: Sorted Indices
1. Implement `sorted_index.h/.cpp`
2. At startup, sort columns used in MIN/MAX in JOB queries
3. Implement `ExecuteSubQuerySortedMin()` with early termination
4. **Measure**: queries with MIN(title.title) aggregation. Expect: aggregation phase drops from O(n) to O(k).

### Step 6: Full Integration + End-to-End Measurement
1. Wire all components together
2. Run full JOB benchmark with storage plan enabled
3. Compare against baseline (none-split/none-jit = original DuckDB)
4. Identify remaining bottlenecks for next iteration

### Step 7 (Optional, JIT): Kernel Fusion
1. JIT-compile SubQueryPlan into specialized loops (unroll joins, specialize filters)
2. This eliminates interpretation overhead in the generic executor
3. Code: `src/jit/kernel_codegen.cpp`
4. **Measure**: fused vs non-fused kernel execution. Expect: 10-30% additional speedup.

## Relationship to Other Directions

### Direction A-JIT (old, JIT within DuckDB operators)
**Status**: Nearly exhausted (-7.3% after 24 iterations). All JIT-gated changes remain in place. AMAC is the only remaining feasible optimization (~20-40ms potential, high complexity). Deprioritized.

### Direction B (Split strategy optimization)
**Status**: Independent and complementary. Better split order produces smaller temp tables, which makes runtime CSR builds faster. A better split strategy would shift sub-query patterns toward more `1base+N_temp` (pure CSR-joinable) and fewer multi-base patterns (`2base`/`3base`/`4base` = currently 16% of sub-queries). In particular, if we improve the split strategy to always start with a single filtered dimension→base join, every sub-query becomes `1base+N_temp` or `1dim+N_temp`, which are fully CSR-joinable with no DuckDB fallback needed.

Direction B is NOT a prerequisite for this design. Implement storage plan + executor first, then improve split strategy. After Direction B, verify if new sub-query patterns need additional executor support (likely not — `SubQueryPlan` already handles `1base+N_temp` with arbitrary N).

### Direction C (Whole-pipeline JIT / sql-jit)
**Status**: Superseded by this design. The "whole-pipeline JIT" concept is now decomposed into:
- Storage plan + kernel executor (this document) — the 90% that comes from better data structures and single-loop execution
- Kernel fusion / Step 7 (JIT) — the 10% that comes from specializing the executor for each sub-query

The storage plan is a prerequisite for effective kernel fusion. Without flat arrays and CSR indexes, there's nothing worth fusing — DuckDB's internal hash table operations can't be inlined.

## Expected Performance Impact

| Component | Bottleneck addressed | Est. improvement |
|-----------|---------------------|-----------------|
| Flat column arrays | fsst_decompress (3-15%), StringScanPartial (2-5%) | Eliminates decompression overhead |
| CSR on base tables | InsertHashes (5-17%), AdvancePointers (6-8%) | Eliminates hash table build + probe |
| CSR on temp tables | Hash join on temp tables (node-based split overhead) | Eliminates +6.6% node-based regression |
| SubQueryPlan executor | Operator boundary overhead, DataChunk copies | Single-loop execution per sub-query |
| Dimension constants | Unnecessary joins with tiny tables | Fewer joins per query |
| Sorted indices | Full scan for MIN/MAX aggregation | O(k) instead of O(n) for MIN |
| **Combined estimate** | | **5-20x over current DuckDB** |

Conservative: 5-8x (generic executor interpretation overhead, multi-base fallbacks). Optimistic: 15-20x (if CSR joins dominate and sorted MIN provides large savings on heavy queries). To reach BespokeOLAP levels (50-170x): need kernel fusion (Step 7) + Direction B.

## Open Questions

1. **String storage**: Should VARCHAR columns use arena allocation (like BespokeOLAP) or std::vector<std::string>? Arena is more cache-friendly but harder to manage. Start with std::vector, optimize later if strings are a bottleneck.
2. **Null handling**: IMDB has nullable columns. Flat arrays need null bitmaps. DuckDB's ValidityMask is 1-bit-per-row; we should use the same.
3. **Thread safety**: Kernel executor is single-threaded per query. Shared structures (flat tables, base CSR) are read-only. Temp table CSR builds are per-query, no contention.
4. **Memory pressure**: ~3.8 GB total (flat arrays + CSR). On 63 GB machine, acceptable. For smaller machines, could load only tables referenced by the query.
5. **PostgreSQL string encoding**: PostgreSQL uses UTF-8 natively; DuckDB uses UTF-8 internally. No encoding conversion needed.
6. **Multi-base sub-queries**: 16% of current sub-queries involve 2-4 base tables without temps. These need either: (a) DuckDB fallback, (b) hash join within the executor, or (c) Direction B to eliminate them. Start with DuckDB fallback; Direction B should reduce or eliminate them.
7. **Inner join expansion**: CSR inner joins (not semi-joins) can expand rows (1:N). Need to handle row multiplication in the executor output. Semi-joins (filter only) are simpler and cover most patterns.
8. **SubQueryComposer input format**: `ComposeSubQuery(const SubIR& sub_ir, ...)` assumes structured sub-IR. Currently the node-based splitter produces SQL strings, not structured IR. Either (a) parse the sub-SQL back into a plan, or (b) modify the splitter to output structured IR directly (preferred — avoids SQL round-trip).
9. **`2temp` pattern scan direction**: The `2temp` pattern (47 occurrences, 7%) intersects two temp tables. The executor scans one and CSR-joins the other. Need a heuristic for which to scan (scan the smaller temp, CSR-join the larger) to minimize iteration count.
10. **CLI flag interaction**: How does a new `--storage-plan` flag interact with existing `--split` and `--jit-level` flags? Define the CLI interface — e.g., `--storage-plan` enables flat table loading + kernel execution, `--split` still controls decomposition strategy, `--jit-level` controls whether Step 7 kernel fusion is used on top.
