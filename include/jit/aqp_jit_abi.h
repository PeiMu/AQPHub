/**
 * aqp_jit_abi.h — Stable C ABI shared between the AQP middleware LLVM
 * compiler and the DuckDB JIT receiver.
 *
 * MUST stay in sync with:
 *   duckdb/src/include/duckdb/execution/aqp_jit.hpp
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
  void *data;         /* flat element array; cast per dtype              */
  uint64_t *validity; /* nullptr = all valid; else 1 bit per row         */
  int32_t vtype;      /* 0=FLAT, 1=CONSTANT, 2=DICTIONARY               */
  int32_t dtype;      /* AQP_DTYPE_* constant                            */
} AQPColView;

/* A batch of rows — mirrors DuckDB's DataChunk at the boundary. */
typedef struct {
  AQPColView *cols;
  uint64_t nrows; /* ≤ STANDARD_VECTOR_SIZE (2048)                   */
  uint64_t ncols;
} AQPChunkView;

/* Output selection vector — indices of rows that pass the filter. */
typedef struct {
  uint32_t *indices; /* sel_t = uint32_t in DuckDB                      */
  uint32_t count;    /* number of entries written by the compiled expr   */
} AQPSelView;

/* dtype constants — must match aqp_jit.hpp in DuckDB. */
#ifdef __cplusplus
static constexpr int32_t AQP_DTYPE_BOOL    = 0;
static constexpr int32_t AQP_DTYPE_INT8    = 1;
static constexpr int32_t AQP_DTYPE_INT16   = 2;
static constexpr int32_t AQP_DTYPE_INT32   = 3;
static constexpr int32_t AQP_DTYPE_INT64   = 4;
static constexpr int32_t AQP_DTYPE_FLOAT   = 5;
static constexpr int32_t AQP_DTYPE_DOUBLE  = 6;
static constexpr int32_t AQP_DTYPE_VARCHAR = 7;
static constexpr int32_t AQP_DTYPE_DATE    = 8;
static constexpr int32_t AQP_DTYPE_OTHER   = 99;
#else
#define AQP_DTYPE_BOOL 0
#define AQP_DTYPE_INT8 1
#define AQP_DTYPE_INT16 2
#define AQP_DTYPE_INT32 3
#define AQP_DTYPE_INT64 4
#define AQP_DTYPE_FLOAT 5
#define AQP_DTYPE_DOUBLE 6
#define AQP_DTYPE_VARCHAR 7
#define AQP_DTYPE_DATE 8
#define AQP_DTYPE_OTHER 99
#endif

/* JIT compilation flags — bitmask, composable.
 * Prefixed AQP_JIT_ to avoid collision with DuckDB's enum AQPJITFlags. */
#define AQP_JIT_NONE 0u
#define AQP_JIT_EXPR                                                           \
  (1u << 0) /* Level 1: individual expression compilation   */
#define AQP_JIT_OPERATOR                                                       \
  (1u << 1) /* Level 2: full operator compilation            */
/* Legacy define — kept for source compat; use ParamConfig::kernel_path instead. */
#define AQP_JIT_PIPELINE                                                       \
  (1u << 2) /* (legacy) pipeline kernel — use kernel_path    */

/* Pipeline-JIT: compile entire probe pipeline as a single function.
 * Unlike kernel_path=PIPELINE (which routes through PipelineKernel),
 * this compiles the probe chain directly in the DuckDB execution path. */
#define AQP_JIT_PIPELINE_JIT (1u << 6)

/* Mask covering only the JIT-level bits (EXPR / OPERATOR / PIPELINE_JIT).
 * Legacy PIPELINE and QUERY kernel paths are controlled by ParamConfig::kernel_path.
 * OPT and SIMD bits are orthogonal and must not gate "should JIT run?". */
#define AQP_JIT_LEVEL_MASK (AQP_JIT_EXPR | AQP_JIT_OPERATOR | AQP_JIT_PIPELINE_JIT)

/* Legacy defines — kept for source compat; no longer part of AQP_JIT_LEVEL_MASK.
 * Use ParamConfig::kernel_path instead. */
#define AQP_JIT_QUERY                                                          \
  (1u << 4) /* (legacy) query-level kernel — use kernel_path  */

/* Legacy aliases (backward compat — debug prints only) */
#define AQP_JIT_OPT3                                                           \
  (1u << 3) /* (legacy, unused)                              */
#define AQP_JIT_SIMD                                                           \
  (1u << 5) /* Enable explicit SIMD vectorization            */

/* SIMD ISA level — bits 8-10 */
#define AQP_JIT_SIMD_MASK (0x7u << 8)
#define AQP_JIT_SIMD_OFF                                                       \
  (0x0u << 8) /* No SIMD (scalar only)                   */
#define AQP_JIT_SIMD_SSE2                                                      \
  (0x1u << 8) /* 128-bit SIMD (4 x i32)                  */
#define AQP_JIT_SIMD_AVX                                                       \
  (0x2u << 8) /* 256-bit SIMD, no gather                 */
#define AQP_JIT_SIMD_AVX2                                                      \
  (0x3u << 8) /* 256-bit SIMD with gather                */
#define AQP_JIT_SIMD_AVX512                                                    \
  (0x4u << 8) /* 512-bit SIMD with gather + scatter      */
#define AQP_JIT_SIMD_AUTO                                                      \
  (0x7u << 8) /* Detect from host CPU                     */

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

/* Pipeline-level: processes one chunk from source through fused operators to
 * sink. Returns count of output rows, or negative on error. */
typedef int64_t (*AQPPipelineFn)(AQPChunkView *source_chunk,
                                 AQPChunkView *sink_chunk,
                                 void *pipeline_state);

/* Callback for deep-copying a non-inline string_t into a DuckDB Vector's
 * string heap.  Implemented on the DuckDB side (aqp_jit.cpp).
 * src_string: pointer to 16-byte source string_t
 * dst_string: pointer to 16-byte destination string_t (written by callee)
 * dst_vector: opaque pointer to the output duckdb::Vector
 */
typedef void (*AQPCopyStringFn)(const void *src_string,
                                void *dst_string,
                                void *dst_vector);

/* State passed to pipeline filter functions (not fusion functions).
 * Provides access to output Vector pointers for safe VARCHAR copying. */
typedef struct {
  void **col_vectors;       /* col_vectors[i] = &chunk.data[i] for output col */
  uint64_t num_cols;
  AQPCopyStringFn copy_str; /* callback for deep string copy                  */
} AQPPipelineFilterState;

/**
 * Hash-join view: exposes DuckDB's JoinHashTable internals to JIT'd probe
 * code without going through a C++ vtable. Filled at probe time by
 * physical_hash_join.cpp (DuckDB side) and passed as the
 * `pipeline_state` argument of a probe-side AQPPipelineFn.
 *
 * MUST stay in sync with duckdb::AQPJoinHTView in aqp_jit.hpp.
 */
typedef struct {
  void           *entries;        /* ht_entry_t *               */
  uint64_t        bitmask;        /* capacity - 1               */
  uint64_t        use_salt;       /* 1 if capacity > USE_SALT_THRESHOLD (8192) */
  void           *layout_ptr;     /* opaque shared_ptr<TupleDataLayout>* */
  uint32_t        tuple_size;     /* total row width in bytes   */
  uint32_t        pointer_offset; /* offset of next_pointer inside row */
  const uint64_t *data_offsets;   /* layout->GetOffsets().data() — per-col offsets within row */
  uint64_t        no_chains;      /* 1 if chains_longer_than_one is false (skip chain walk) */
  const uint64_t *bf_data;        /* bloom filter bit array (nullptr = no BF)               */
  uint64_t        bf_bitmask;     /* num_sectors - 1 for BF lookup                          */
  uint64_t        has_row_validity; /* 1 if rows start with a per-column validity bit prefix */
} AQPJoinHTView;

/**
 * Multi-probe state: array of AQPJoinHTView pointers for fused multi-probe
 * functions. views[0] = innermost HT (probed first), views[num_stages-1] =
 * outermost. Passed as pipeline_state arg to compiled multi-probe functions.
 */
typedef struct {
  AQPJoinHTView *views[4];
  uint32_t       num_stages;
} AQPMultiProbeState;

#ifdef __cplusplus
} // extern "C"
#endif
