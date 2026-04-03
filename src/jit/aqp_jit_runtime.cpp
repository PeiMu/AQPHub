/**
 * aqp_jit_runtime.cpp — Runtime helpers called from LLVM-compiled expressions.
 *
 * These functions are compiled into aqp_jit_lib and linked into the AQP
 * middleware binary.  LLVM-compiled expressions call them via symbol lookup
 * through ORC JIT's dynamic linker (the process symbol table).
 *
 * All functions have C linkage so LLVM can find them by name without mangling.
 */

#include <cstdint>
#include <cstring>
#include <cctype>
#include <cstdlib>

extern "C" {

// ---------------------------------------------------------------------------
// LIKE pattern matching
// Supports '%' (any sequence) and '_' (single character).
// str: pointer to raw UTF-8 bytes, slen: length in bytes
// pat: pattern bytes, plen: length in bytes
// Returns 1 if matches, 0 otherwise.
// ---------------------------------------------------------------------------
int aqp_like_match(const char *str, int32_t slen,
                   const char *pat, int32_t plen) {
    // Recursive LIKE: simple O(n*m) implementation sufficient for JIT.
    // For production, replace with a DFA-compiled matcher.
    const char *s = str, *se = str + slen;
    const char *p = pat, *pe = pat + plen;

    while (p < pe) {
        if (*p == '%') {
            ++p;
            if (p == pe) return 1; // trailing %: match rest of string
            // Try matching suffix from every position in s
            while (s <= se) {
                if (aqp_like_match(s, (int32_t)(se - s), p, (int32_t)(pe - p)))
                    return 1;
                ++s;
            }
            return 0;
        } else if (*p == '_') {
            if (s >= se) return 0;
            ++s; ++p;
        } else {
            if (s >= se || *s != *p) return 0;
            ++s; ++p;
        }
    }
    return (s == se) ? 1 : 0;
}

// Case-insensitive LIKE (ILIKE)
int aqp_ilike_match(const char *str, int32_t slen,
                    const char *pat, int32_t plen) {
    // Lower-fold both sides into stack buffers for short strings.
    // For longer strings, fall back to heap allocation.
    char sbuf[256], pbuf[256];
    char *sl = (slen < 256) ? sbuf : (char*)malloc(slen + 1);
    char *pl = (plen < 256) ? pbuf : (char*)malloc(plen + 1);
    for (int i = 0; i < slen; i++) sl[i] = (char)tolower((unsigned char)str[i]);
    for (int i = 0; i < plen; i++) pl[i] = (char)tolower((unsigned char)pat[i]);
    int r = aqp_like_match(sl, slen, pl, plen);
    if (sl != sbuf) free(sl);
    if (pl != pbuf) free(pl);
    return r;
}

// ---------------------------------------------------------------------------
// IN-set membership tests
// ---------------------------------------------------------------------------

// col IN (int32 values[0..n-1])
int aqp_in_set_i32(int32_t val, const int32_t *values, int32_t n) {
    for (int32_t i = 0; i < n; i++)
        if (values[i] == val) return 1;
    return 0;
}

// col IN (int64 values[0..n-1])
int aqp_in_set_i64(int64_t val, const int64_t *values, int32_t n) {
    for (int32_t i = 0; i < n; i++)
        if (values[i] == val) return 1;
    return 0;
}

// col IN (string values); each entry is (char* ptr, int32 len)
// strs: array of (ptr, len) pairs stored as { char* ptr; int32_t len; }
// n: number of strings in the set
int aqp_in_set_str(const char *str, int32_t slen,
                   const char **ptrs, const int32_t *lens, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        if (lens[i] == slen && memcmp(ptrs[i], str, (size_t)slen) == 0)
            return 1;
    }
    return 0;
}

// Exact VARCHAR equality: returns 1 iff lengths are equal AND bytes match.
// Used for SQL '=' and '<>' on string columns (as opposed to aqp_like_match
// which treats '%' and '_' as wildcards and is only correct for LIKE).
int aqp_str_eq(const char *a, int32_t alen, const char *b, int32_t blen) {
    if (alen != blen) return 0;
    return (memcmp(a, b, (size_t)alen) == 0) ? 1 : 0;
}

} // extern "C"
