# Query Kernel Optimization Plan

Optimizations for the query-kernel path (`--jit-level=query`), which uses CSR-based joins on flat column arrays. These are SUPERSEDED by the pipeline kernel (Step A in `kernel_path_opt.md`), which replaces CSR joins with hash-table joins. Kept here for reference if the query kernel path is revisited.

## Background: Runtime CSR Build Cost

CSR build = 3,291ms (56.8% of MW overhead in step_7_2). After each sub-query, CSR is built on ALL INT32 columns of the temp result for future iterations to probe into.

**CSR build cost model**: `BuildCSR(flat, col_idx, max_val)` does:
1. Allocate `row_ptr[max_val + 2]` as `uint64_t[]` (19-31MB for title/name domains)
2. `memset(row_ptr, 0, ...)` — zeroing 19-31MB
3. Pass 1: count FK values (O(rows))
4. Pass 2: prefix sum (O(domain))
5. Pass 3: scatter row IDs (O(rows), random writes)

**Two cost profiles**:
- **Small temps (90%+ of iterations)**: e.g., 414 rows with domain 2.5M. Cost dominated by `memset(19MB)` ~3-5ms. The data passes over 414 rows are negligible. Query 29c builds 20+ CSRs across 12 iterations, mostly on tiny temps.
- **Large temps (<10% of iterations)**: e.g., 2.83M rows with domain 4M. Cost dominated by 3 data passes over millions of rows ~30-40ms per CSR.

**How CSR is used**: Kernel scans one table, does `csr->Lookup(fk_value)` to find matching rows in the lookup table. Returns `{begin, end}` pointer pair into `col_idx[]` array. The kernel iterates `col_idx[begin..end]` to access lookup table row indices.

**When CSR is NOT needed**: When the join is a pure semi-join (only existence check, all output columns from scan table). In this case, a byte-map (`uint8_t[domain]`) suffices. But many sub-queries output columns FROM the lookup table (FROM_JOIN), requiring actual row indices.

Example (16b, 6 iterations):
```
iter 1: keyword→mk: output mk.movie_id (FROM_JOIN) → CSR on temp1
iter 2: cn→mc: output mc.movie_id (FROM_JOIN) → CSR on temp2
iter 3: temp1→temp2: semi-join on movie_id → byte-map could work
iter 4: temp3→cast_info: output ci.person_id,ci.movie_id (FROM_JOIN, base CSR)
iter 5: aka_name→temp4: output an.*,temp4.movie_id (FROM_JOIN) → CSR on temp5
```

## Step B: Sparse CSR for Small Temps

**Status**: SUPERSEDED by pipeline kernel (Step A).

**Original problem**: 90%+ of CSR builds are on small temps (< 50K rows) with large domains (2.5M+). The cost is dominated by `memset(19-31MB)` to zero the `row_ptr` array, not by the actual data passes.

**Proposed solution**: Use hash-based CSR for temps < 50K rows:
- Replace dense `row_ptr[max_val+2]` with a hash map `unordered_map<int32_t, {begin, end}>`
- Or use a compact sorted array of (key, offset) pairs with binary search
- Eliminates the 19-31MB memset for small temps

**Why superseded**: The pipeline kernel doesn't build CSR on temps at all — it uses hash tables at execution time. The 3,291ms CSR build cost is eliminated entirely.

**Estimated savings** (if query kernel path): -1,500 to -2,000ms.

## Step C: Byte-Map for Semi-Join Patterns

**Status**: SUPERSEDED by pipeline kernel (Step A).

**Original problem**: When runtime CSR is only used for existence checks (1-column temp, semi-join pattern), building a full CSR is wasteful.

**Proposed solution**: Replace CSR with a byte-map (`uint8_t[domain]`) for semi-join-only temps:
- Detect semi-join pattern: temp has 1 INT32 column, future iterations only check existence (no FROM_JOIN columns)
- Allocate `uint8_t[max_val + 1]` and set `byte_map[val] = 1` for each row
- Lookup: `byte_map[val]` — O(1), no indirection

**Why superseded**: The pipeline kernel's `HashJoinTable::Contains()` serves the same purpose. For small dimension tables, the DIRECT probe method (`array[key]`) is even faster than a byte-map.

**Estimated savings** (if query kernel path): ~-500ms.

## Step D: Skip Last-Iteration CSR Build

**Status**: SUPERSEDED by pipeline kernel (Step A).

**Original problem**: After the last sub-query iteration, CSR is still built on the output temp, even though no future iteration will probe into it.

**Proposed solution**: Detect the last iteration and skip CSR build on its output.
- In the split loop, check if the remaining IR has no more sub-queries referencing this temp
- Or simply mark the last iteration and skip CSR build

**Why superseded**: The pipeline kernel never builds CSR on output at all (`RegisterKernelResult(result, name, false)`), including the last iteration.

**Estimated savings** (if query kernel path): ~-500ms.
