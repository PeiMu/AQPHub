/**
 * aqp_jit_abi.h — Stable C ABI shared between the AQP middleware LLVM
 * compiler and the DuckDB JIT receiver.
 *
 * MUST stay in sync with:
 *   duckdb_132/src/include/duckdb/execution/aqp_jit.hpp
 *
 * Key invariants:
 *   - sel_t in DuckDB is uint32_t  (duckdb/common/typedefs.hpp)
 *   - validity_t is uint64_t        (1 bit per row, 64 rows per word)
 *   - nullptr validity  → all rows valid
 *   - AQPChunkView.cols points to a thread-local scratch buffer in DuckDB;
 *     the compiled function must not retain the pointer past the call.
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One column of a DataChunk, flattened for the compiled expression. */
typedef struct {
    void     *data;      /* flat element array; cast per dtype              */
    uint64_t *validity;  /* nullptr = all valid; else 1 bit per row         */
    int32_t   vtype;     /* 0=FLAT, 1=CONSTANT, 2=DICTIONARY               */
    int32_t   dtype;     /* AQP_DTYPE_* constant                            */
} AQPColView;

/* A batch of rows — mirrors DuckDB's DataChunk at the boundary. */
typedef struct {
    AQPColView *cols;
    uint64_t    nrows;  /* ≤ STANDARD_VECTOR_SIZE (2048)                   */
    uint64_t    ncols;
} AQPChunkView;

/* Output selection vector — indices of rows that pass the filter. */
typedef struct {
    uint32_t *indices;  /* sel_t = uint32_t in DuckDB                      */
    uint32_t  count;    /* number of entries written by the compiled expr   */
} AQPSelView;

/* dtype constants — must match aqp_jit.hpp in DuckDB */
#define AQP_DTYPE_BOOL    0
#define AQP_DTYPE_INT8    1
#define AQP_DTYPE_INT16   2
#define AQP_DTYPE_INT32   3
#define AQP_DTYPE_INT64   4
#define AQP_DTYPE_FLOAT   5
#define AQP_DTYPE_DOUBLE  6
#define AQP_DTYPE_VARCHAR 7
#define AQP_DTYPE_DATE    8
#define AQP_DTYPE_OTHER   99

/* JIT compilation flags — bitmask, composable.
 * Prefixed AQP_JIT_ to avoid collision with DuckDB's enum AQPJITFlags. */
#define AQP_JIT_NONE      0u
#define AQP_JIT_EXPR      (1u << 0)   /* Level 1: individual expression compilation   */
#define AQP_JIT_OPERATOR  (1u << 1)   /* Level 2: full operator compilation            */
#define AQP_JIT_PIPELINE  (1u << 2)   /* Level 3: fused pipeline compilation           */
#define AQP_JIT_OPT3      (1u << 3)   /* Use LLVM O3 optimization                     */
#define AQP_JIT_SUBPLAN   (1u << 4)   /* Level 4: multi-pipeline sub-plan compilation  */
#define AQP_JIT_SIMD      (1u << 5)   /* Enable explicit SIMD vectorization            */

/**
 * Compiled expression function type.
 *
 * The LLVM compiler generates one function per filter expression subtree.
 * It fills sel->indices with the row indices of matching rows and stores
 * the count in sel->count.  Returns the number of selected rows.
 *
 * Function name convention: "aqp_expr_<hex_hash>"
 * where hex_hash = FNV-1a hash of the serialised IR subtree.
 */
typedef uint64_t (*AQPExprFn)(AQPChunkView *chunk, AQPSelView *sel);

/* Operator-level: transforms input chunk to output chunk.
 * Returns OperatorResultType cast to int32_t. */
typedef int32_t (*AQPOperatorFn)(AQPChunkView *in, AQPChunkView *out);

/* Pipeline-level: processes one chunk from source through fused operators to sink.
 * Returns count of output rows, or negative on error. */
typedef int64_t (*AQPPipelineFn)(AQPChunkView *source_chunk,
                                 AQPChunkView *sink_chunk,
                                 void *pipeline_state);

/* Sub-plan context: holds all state for a compiled sub-plan.
 * Managed by the middleware; passed to the coordinator function. */
typedef struct {
    void        **hash_tables;     /* array of AQPHashTable pointers              */
    uint32_t      num_hash_tables;
    AQPPipelineFn *pipeline_fns;   /* array of compiled pipeline functions         */
    uint32_t      num_pipelines;
    void         *scratch;         /* scratch buffer for intermediate chunks       */
    uint64_t      scratch_size;
} AQPSubPlanCtx;

/* Sub-plan level: orchestrates multiple pipelines.
 * The coordinator runs build pipelines, then probe pipelines, manages hash tables.
 * Returns 0 on success, negative on error. */
typedef int32_t (*AQPSubPlanFn)(AQPSubPlanCtx *ctx);

#ifdef __cplusplus
} // extern "C"
#endif
