/**
 * aqp_jit_hashtable.h — Portable open-addressing hash table for JIT runtime.
 *
 * Used by JIT-compiled aggregate and join operators. The hash table is
 * engine-agnostic — it operates on raw byte keys and payloads, decoupling
 * the JIT from DuckDB's internal GroupedAggregateHashTable/JoinHashTable.
 *
 * Keys and payloads are fixed-width byte arrays. The hash table stores
 * copies of both (no pointers to external memory).
 *
 * C linkage so LLVM JIT can call these via symbol lookup.
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle — actual struct defined in aqp_jit_hashtable.cpp */
typedef struct AQPHashTable AQPHashTable;

/* Create a hash table.
 * key_width:     byte width of each key (e.g., 4 for int32, 8 for int64)
 * payload_width: byte width of each payload (e.g., 8 for SUM accumulator)
 * est_rows:      estimated number of entries (for initial sizing)
 * Returns opaque handle, or NULL on failure. */
AQPHashTable *aqp_ht_create(uint32_t key_width, uint32_t payload_width,
                             uint64_t est_rows);

/* Destroy hash table and free all memory. */
void aqp_ht_destroy(AQPHashTable *ht);

/* Insert or find: looks up key; if not found, inserts a new entry with
 * zero-initialized payload. Returns pointer to the payload slot (always
 * valid — never NULL unless ht is NULL).
 * key: pointer to key_width bytes. */
void *aqp_ht_insert(AQPHashTable *ht, const void *key);

/* Probe: looks up key. Returns pointer to payload if found, NULL if not. */
void *aqp_ht_probe(const AQPHashTable *ht, const void *key);

/* Iteration: reset iterator, then call aqp_ht_next repeatedly.
 * Returns 1 and sets *key_out/*payload_out on success, 0 when exhausted. */
void aqp_ht_iter_reset(AQPHashTable *ht);
int  aqp_ht_next(AQPHashTable *ht, void **key_out, void **payload_out);

/* Number of entries currently in the table. */
uint64_t aqp_ht_size(const AQPHashTable *ht);

/* Hash function for arbitrary byte keys (FNV-1a). */
uint64_t aqp_hash(const void *key, uint32_t len);

#ifdef __cplusplus
} /* extern "C" */
#endif
