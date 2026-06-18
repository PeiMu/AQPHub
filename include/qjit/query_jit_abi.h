/**
 * query_jit_abi.h — Stable C ABI between the AQP middleware LLVM compiler
 * (query-jit, --jit-level=query) and the qjit runtime library.
 *
 * Same discipline as include/jit/aqp_jit_abi.h: FIELD ORDER IS ABI.
 * The compiled query function (one module / one entry per sub-query) sees
 * only these structs and the extern "C" qjit_* entry points; everything
 * behind the opaque pointers is C++ (query_jit_runtime.h / scheduler.h).
 *
 * Conventions (match the pipeline-jit ABI):
 *   - validity: uint64_t*, 1 bit per row, 64 rows per word, 1 = valid,
 *     nullptr = all rows valid.
 *   - strings: 16-byte DuckDB string_t layout (QjitString below);
 *     length <= 12 means bytes are inlined, otherwise prefix + pointer.
 *   - dtypes reuse the AQP_DTYPE_* constants from aqp_jit_abi.h.
 */
#pragma once
#include <stdint.h>

#include "jit/aqp_jit_abi.h" /* AQP_DTYPE_* constants */

#ifdef __cplusplus
extern "C" {
#endif

/* 16-byte string value — bit-compatible with duckdb::string_t.
 * len <= 12: bytes live in prefix[0..3] + extra[0..7] (inlined).
 * len  > 12: prefix holds the first 4 bytes, ptr points at all len bytes. */
#define QJIT_STRING_INLINE_LEN 12u
typedef union {
  struct {
    uint32_t length;
    char prefix[4];
    const char *ptr;
  } pointer;
  struct {
    uint32_t length;
    char inlined[12];
  } inlined;
} QjitString;

/* One column of a source table, flattened for compiled code. */
typedef struct {
  void *data;         /* flat element array; cast per dtype (QjitString for
                         VARCHAR)                                          */
  uint64_t *validity; /* nullptr = all valid; else 1 bit per row, 1=valid  */
  int32_t dtype;      /* AQP_DTYPE_* constant (aqp_jit_abi.h)              */
  int32_t reserved;   /* padding — keep struct 8-byte aligned              */
} QjitColView;

/* A whole table (base FlatTable view or qjit temp table). */
typedef struct {
  QjitColView *cols;
  uint64_t nrows;
  uint64_t ncols;
} QjitTableView;

/* Per-query context passed to the compiled entry function and threaded
 * through every morsel body. All pointers are owned by the executor and
 * outlive the entry call. */
typedef struct {
  void *pool; /* qjit::QjitWorkerPool*                                     */

  const QjitTableView *sources; /* step sources, index fixed at compile    */
  uint64_t num_sources;

  void **hash_tables; /* qjit::QjitHashTable*, index = ht_id               */
  uint64_t num_hash_tables;

  /* Per-worker state blocks: worker_states[worker_id] points at a block
   * laid out by the compiler (agg cells, result-table partition handles,
   * scratch). Allocation/layout is the executor's job (Phase 2/3). */
  void **worker_states;
  uint64_t num_workers;

  uint64_t morsel_size; /* rows per morsel (--query-jit-morsel)            */

  void *result; /* qjit::QjitTable* — final/temp sink                      */
  void *user;   /* spare slot for the executor                             */
} QjitQueryContext;

/* Morsel body emitted by the compiler: process source rows [begin, end). */
typedef void (*QjitMorselFn)(QjitQueryContext *ctx, uint64_t begin,
                             uint64_t end, uint32_t worker_id);

/* Entry function of one compiled sub-query module.
 * Returns produced row count, or negative on error. */
typedef int64_t (*QjitQueryFn)(QjitQueryContext *ctx);

/* ---- runtime entry points (registered as absoluteSymbols) ------------- */

/* Run fn over [0, total) in chunks of `morsel` rows on ctx->pool workers.
 * Blocks until all morsels are done; workers are quiescent on return. */
void qjit_parallel_for(QjitQueryContext *ctx, uint64_t total, uint64_t morsel,
                       QjitMorselFn fn);

/* Reserve `bytes` contiguous bytes in a qjit::QjitBuffer; returns the write
 * pointer. The pointer stays valid for the buffer's lifetime (chunked
 * growth — earlier chunks are never reallocated). */
void *qjit_buffer_grow(void *buffer, uint64_t bytes);

/* Append one row to hash table `ht` from worker `worker_id`; `hash` is the
 * precomputed key hash. Returns the row-payload write pointer (tuple_size
 * bytes). Caller must NOT call this for NULL keys (inner-join semantics:
 * NULL keys are skipped at build). */
void *qjit_ht_append(void *ht, uint32_t worker_id, uint64_t hash);

/* §6.12 fast-path HT append: the compiled morsel body stack-allocates a
 * QjitHtAppendHandle and calls qjit_ht_begin once at morsel entry.  The
 * hot loop then does an inline bump-pointer check (cursor + stride <=
 * limit); on the fast path no function call is needed.  When the chunk is
 * exhausted the slow path calls qjit_ht_append_slow, which allocates a
 * new chunk and refreshes the handle.  qjit_ht_end writes the cursor
 * back to the underlying QjitBuffer at morsel exit.
 *
 * FIELD ORDER IS ABI — compiled code GEPs into this struct. */
typedef struct {
  uint8_t *cursor;   /* current write position in last chunk              */
  uint8_t *limit;    /* end of last chunk capacity                        */
  uint64_t stride;   /* entry_stride_ (constant for this HT)             */
  uint64_t *count;   /* pointer to per-worker fragment count              */
} QjitHtAppendHandle;

void qjit_ht_begin(void *ht, uint32_t worker_id, QjitHtAppendHandle *handle);
void *qjit_ht_append_slow(void *ht, uint32_t worker_id, uint64_t hash,
                           QjitHtAppendHandle *handle);
void qjit_ht_end(void *ht, uint32_t worker_id, QjitHtAppendHandle *handle);

/* Size the pow-2 directory and link all worker fragments into chains,
 * using ctx->pool (ParallelFor + CAS bucket pushes) when the build is big
 * enough, serially otherwise. Must be called on the main thread after the
 * build parallel_for returns (pool quiescent) and before any probe. */
void qjit_ht_finalize(void *ctx /* QjitQueryContext* */, void *ht);

/* Probe-side accessors, valid only after qjit_ht_finalize. The compiled
 * probe loop loads the chain head itself:
 *   Entry *e = ((Entry**)qjit_ht_dir(ht))[hash & qjit_ht_mask(ht)];
 * Entry layout (ABI): { Entry *next @0; uint64_t hash @8; row @16 }. */
void *qjit_ht_dir(void *ht);      /* Entry** directory base                 */
uint64_t qjit_ht_mask(void *ht);  /* directory index mask (dir_size - 1)    */

/* §5.5 A+ join-filter pushdown: build-key statistics, valid only after
 * qjit_ht_finalize on an HT constructed with a key0 offset. Empty HT (or no
 * key0 offset) reports min = INT64_MAX, max = INT64_MIN, so the compiled
 * range guard `lo <= key && key <= hi` correctly drops every probe row. */
int64_t qjit_ht_key0_min(void *ht);
int64_t qjit_ht_key0_max(void *ht);
uint64_t qjit_ht_entries(void *ht);

/* Per-block (QJIT_BLOCK_ROWS source rows) min/max statistics used for
 * morsel-level block skipping: stats[2*b] = min, stats[2*b+1] = max of the
 * non-NULL values in block b (all-NULL block: {INT32_MAX, INT32_MIN}).
 * Delivered to compiled code via QjitQueryContext::user, an array of
 * `const int32_t *` indexed by step (NULL = no stats for that step). */
#define QJIT_BLOCK_ROWS 2048u
#define QJIT_BLOCK_SHIFT 11u

/* ---- ungrouped-aggregate state (qjit::QjitAggState) -------------------- */
/* `state` is the calling worker's own QjitAggState (per-worker, no
 * locking); `cell` indexes the agg cell. NULL inputs must be filtered by
 * the caller — update_i64/update_str assume a valid value. update_count
 * covers COUNT(x) (after the caller's validity check) and COUNT(*). */
void qjit_agg_update_i64(void *state, uint64_t cell, int64_t v);
void qjit_agg_update_str(void *state, uint64_t cell, const QjitString *v);
void qjit_agg_update_count(void *state, uint64_t cell);

/* Deep-copy src (QjitString) into the arena owned by `arena` and write the
 * (possibly re-pointed) 16-byte result to dst. Inline strings are copied
 * by value; long strings get their bytes copied into the arena.
 * Thread-safety: arena must be a per-worker arena OR the call must be
 * single-threaded (merge epilogue). */
void qjit_str_arena_copy(void *arena, QjitString *dst, const QjitString *src);

/* ---- result-table sink (qjit::QjitTable) ------------------------------ */
/* One output row = one append per column (in any order, exactly once per
 * column) + one finish_row. Appends are worker-local: concurrent calls are
 * safe across distinct worker_ids. append_str deep-copies the bytes into
 * the table's per-worker arena (source lifetime ends with the morsel). */
void qjit_table_append_i32(void *table, uint32_t worker_id, uint64_t col,
                           int32_t v);
void qjit_table_append_i64(void *table, uint32_t worker_id, uint64_t col,
                           int64_t v);
void qjit_table_append_str(void *table, uint32_t worker_id, uint64_t col,
                           const QjitString *v);
void qjit_table_append_null(void *table, uint32_t worker_id, uint64_t col);
void qjit_table_finish_row(void *table, uint32_t worker_id);

#ifdef __cplusplus
} // extern "C"
#endif
