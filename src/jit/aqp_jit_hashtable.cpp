/**
 * aqp_jit_hashtable.cpp — Portable open-addressing hash table for JIT runtime.
 *
 * Simple linear-probing hash table with power-of-2 sizing and 70% load factor.
 * Keys and payloads are stored inline as fixed-width byte arrays.
 *
 * Layout per slot: [occupied:1][key:key_width][payload:payload_width]
 * Total slot size: 1 + key_width + payload_width
 */

#include "jit/aqp_jit_hashtable.h"
#include <cstdlib>
#include <cstring>

struct AQPHashTable {
    uint8_t  *slots;          // flat array of slots
    uint64_t  capacity;       // number of slots (power of 2)
    uint64_t  count;          // number of occupied slots
    uint32_t  key_width;
    uint32_t  payload_width;
    uint32_t  slot_size;      // 1 + key_width + payload_width
    uint64_t  mask;           // capacity - 1
    uint64_t  iter_pos;       // iteration cursor
};

// Byte offsets within a slot
static inline uint8_t *slot_ptr(AQPHashTable *ht, uint64_t idx) {
    return ht->slots + idx * ht->slot_size;
}
static inline uint8_t  slot_occupied(const uint8_t *s) { return s[0]; }
static inline uint8_t *slot_key(uint8_t *s)            { return s + 1; }
static inline uint8_t *slot_payload(uint8_t *s, uint32_t kw) { return s + 1 + kw; }

// Round up to next power of 2
static uint64_t next_pow2(uint64_t v) {
    v--;
    v |= v >> 1;  v |= v >> 2;  v |= v >> 4;
    v |= v >> 8;  v |= v >> 16; v |= v >> 32;
    return v + 1;
}

static void ht_grow(AQPHashTable *ht);

extern "C" {

uint64_t aqp_hash(const void *key, uint32_t len) {
    const uint8_t *p = (const uint8_t *)key;
    uint64_t h = 14695981039346656037ULL;
    for (uint32_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

AQPHashTable *aqp_ht_create(uint32_t key_width, uint32_t payload_width,
                             uint64_t est_rows) {
    auto *ht = (AQPHashTable *)calloc(1, sizeof(AQPHashTable));
    if (!ht) return nullptr;

    ht->key_width     = key_width;
    ht->payload_width = payload_width;
    ht->slot_size     = 1 + key_width + payload_width;

    // Size for ~70% load factor
    uint64_t min_cap = est_rows < 8 ? 16 : (est_rows * 10 / 7);
    ht->capacity = next_pow2(min_cap);
    ht->mask     = ht->capacity - 1;
    ht->count    = 0;
    ht->iter_pos = 0;

    ht->slots = (uint8_t *)calloc(ht->capacity, ht->slot_size);
    if (!ht->slots) { free(ht); return nullptr; }

    return ht;
}

void aqp_ht_destroy(AQPHashTable *ht) {
    if (!ht) return;
    free(ht->slots);
    free(ht);
}

void *aqp_ht_insert(AQPHashTable *ht, const void *key) {
    if (!ht) return nullptr;

    // Grow if load factor > 70%
    if (ht->count * 10 > ht->capacity * 7)
        ht_grow(ht);

    uint64_t h = aqp_hash(key, ht->key_width);
    uint64_t idx = h & ht->mask;

    while (true) {
        uint8_t *s = slot_ptr(ht, idx);
        if (!slot_occupied(s)) {
            // Empty slot — insert
            s[0] = 1;
            memcpy(slot_key(s), key, ht->key_width);
            memset(slot_payload(s, ht->key_width), 0, ht->payload_width);
            ht->count++;
            return slot_payload(s, ht->key_width);
        }
        if (memcmp(slot_key(s), key, ht->key_width) == 0) {
            // Key match — return existing payload
            return slot_payload(s, ht->key_width);
        }
        idx = (idx + 1) & ht->mask;
    }
}

void *aqp_ht_probe(const AQPHashTable *ht, const void *key) {
    if (!ht) return nullptr;

    uint64_t h = aqp_hash(key, ht->key_width);
    uint64_t idx = h & ht->mask;

    while (true) {
        const uint8_t *s = ((AQPHashTable*)ht)->slots + idx * ht->slot_size;
        if (!slot_occupied(s))
            return nullptr;  // empty slot — not found
        if (memcmp(s + 1, key, ht->key_width) == 0)
            return (void *)(s + 1 + ht->key_width);  // found
        idx = (idx + 1) & ht->mask;
    }
}

void aqp_ht_iter_reset(AQPHashTable *ht) {
    if (ht) ht->iter_pos = 0;
}

int aqp_ht_next(AQPHashTable *ht, void **key_out, void **payload_out) {
    if (!ht) return 0;
    while (ht->iter_pos < ht->capacity) {
        uint8_t *s = slot_ptr(ht, ht->iter_pos);
        ht->iter_pos++;
        if (slot_occupied(s)) {
            if (key_out)     *key_out     = slot_key(s);
            if (payload_out) *payload_out = slot_payload(s, ht->key_width);
            return 1;
        }
    }
    return 0;
}

uint64_t aqp_ht_size(const AQPHashTable *ht) {
    return ht ? ht->count : 0;
}

} // extern "C"

// Internal: double the capacity and rehash all entries
static void ht_grow(AQPHashTable *ht) {
    uint64_t old_cap  = ht->capacity;
    uint8_t *old_slots = ht->slots;

    ht->capacity *= 2;
    ht->mask      = ht->capacity - 1;
    ht->slots     = (uint8_t *)calloc(ht->capacity, ht->slot_size);
    ht->count     = 0;

    // Re-insert all occupied entries
    for (uint64_t i = 0; i < old_cap; i++) {
        uint8_t *s = old_slots + i * ht->slot_size;
        if (slot_occupied(s)) {
            void *payload = aqp_ht_insert(ht, slot_key(s));
            // Copy old payload to new slot
            memcpy(payload, slot_payload(s, ht->key_width), ht->payload_width);
        }
    }

    free(old_slots);
}
