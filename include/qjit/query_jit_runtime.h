/**
 * query_jit_runtime.h — qjit runtime data structures (pure C++, no LLVM,
 * no DuckDB). Compiled query code reaches these only through the C entry
 * points in query_jit_abi.h; the executor (Phase 2) uses them directly.
 *
 * Models follow lingo-db's runtime (Hashtable.h / GrowingBuffer):
 *   - QjitBuffer: chunked growing buffer, 1.2x growth, pointers stable.
 *   - QjitHashTable: chained HT; workers append rows into per-worker
 *     fragments. Finalize sizes a pow-2 directory and links every entry
 *     into its chain — in parallel (ParallelFor over fragment chunks with
 *     CAS bucket pushes) when given a worker pool, serially otherwise.
 *   - QjitAggState: per-worker ungrouped-aggregate cells + Merge.
 *   - QjitTable: columnar result/temp table; per-worker partitions
 *     concatenated by Finalize. 1-bit validity, string arena.
 */
#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "qjit/query_jit_abi.h"

namespace qjit {

class QjitWorkerPool;

// ---------------------------------------------------------------------------
// QjitString helpers
// ---------------------------------------------------------------------------

inline const char *StringData(const QjitString &s) {
  return s.inlined.length <= QJIT_STRING_INLINE_LEN ? s.inlined.inlined
                                                    : s.pointer.ptr;
}
inline uint32_t StringLen(const QjitString &s) { return s.inlined.length; }

/* Build a QjitString over caller-owned bytes (no copy; long strings point
 * at `data`). */
QjitString MakeString(const char *data, uint32_t len);

inline bool StringEq(const QjitString &a, const QjitString &b) {
  if (StringLen(a) != StringLen(b))
    return false;
  return std::memcmp(StringData(a), StringData(b), StringLen(a)) == 0;
}
inline int StringCmp(const QjitString &a, const QjitString &b) {
  uint32_t n = StringLen(a) < StringLen(b) ? StringLen(a) : StringLen(b);
  int c = std::memcmp(StringData(a), StringData(b), n);
  if (c != 0)
    return c;
  return StringLen(a) < StringLen(b) ? -1 : (StringLen(a) > StringLen(b));
}

// ---------------------------------------------------------------------------
// Validity helpers (column bitmask convention: 1 = valid, 64 rows/word)
// ---------------------------------------------------------------------------

inline bool RowValid(const uint64_t *validity, uint64_t row) {
  return !validity || (validity[row >> 6] >> (row & 63)) & 1;
}
inline void SetRowInvalid(uint64_t *validity, uint64_t row) {
  validity[row >> 6] &= ~(uint64_t(1) << (row & 63));
}

// ---------------------------------------------------------------------------
// QjitBuffer — chunked growing byte buffer; allocations never move.
// ---------------------------------------------------------------------------

class QjitBuffer {
public:
  explicit QjitBuffer(uint64_t initial_capacity = 4096)
      : next_capacity_(initial_capacity < 64 ? 64 : initial_capacity) {}

  /* Reserve `bytes` contiguous bytes; returns the write pointer. */
  uint8_t *Allocate(uint64_t bytes);

  uint64_t TotalSize() const { return total_size_; }

  struct Chunk {
    std::unique_ptr<uint8_t[]> data;
    uint64_t capacity = 0;
    uint64_t used = 0;
  };
  const std::vector<Chunk> &Chunks() const { return chunks_; }

  void Reset() {
    chunks_.clear();
    total_size_ = 0;
  }

private:
  std::vector<Chunk> chunks_;
  uint64_t next_capacity_;
  uint64_t total_size_ = 0;
};

// ---------------------------------------------------------------------------
// QjitStringArena — byte arena for VARCHAR payloads owned by a sink/agg.
// ---------------------------------------------------------------------------

class QjitStringArena {
public:
  /* Deep-copy: returns a QjitString whose bytes (if not inline) live in
   * this arena. */
  QjitString Copy(const QjitString &src);
  QjitString Copy(const char *data, uint32_t len);

private:
  QjitBuffer bytes_;
};

// ---------------------------------------------------------------------------
// QjitHashTable — chained hash table, per-worker build fragments.
//
// Entry layout in worker buffers: [Entry header][row payload tuple_size B].
// Row payload layout is the compiler's business (keys then payloads, with
// the per-column validity byte prefix convention if needed).
// ---------------------------------------------------------------------------

class QjitHashTable {
public:
  struct Entry {
    Entry *next;
    uint64_t hash;
    uint8_t *Row() { return reinterpret_cast<uint8_t *>(this + 1); }
    const uint8_t *Row() const {
      return reinterpret_cast<const uint8_t *>(this + 1);
    }
  };

  /* key0_offset >= 0: byte offset of join key 0 (i64) within the row;
   * Finalize then also computes the key-0 min/max for the compiled range
   * guard (§5.5 A+ join-filter pushdown). */
  QjitHashTable(uint32_t tuple_size, uint32_t num_workers,
                int64_t key0_offset = -1);

  /* Worker-local append; safe to call concurrently from distinct
   * worker_ids. Returns the row payload pointer (tuple_size bytes).
   * Callers skip NULL keys (inner-join build semantics). */
  uint8_t *AppendRow(uint32_t worker_id, uint64_t hash);

  /* Sizes directory to pow2 >= 2*n and links all fragment entries into
   * chains. With a pool (and enough entries) the link walk runs as a
   * ParallelFor over fragment chunks using CAS bucket pushes; the pool's
   * join establishes happens-before for every subsequent probe. Chain
   * order within a bucket is then nondeterministic (probes emit ALL
   * matches, so result SETS are unaffected). Idempotent via finalized_.
   * Must be called with the pool quiescent (after the build ParallelFor
   * returns). */
  void Finalize(QjitWorkerPool *pool = nullptr);

  Entry *Lookup(uint64_t hash) const {
    assert(finalized_);
    return directory_[hash & dir_mask_].load(std::memory_order_relaxed);
  }

  uint64_t NumEntries() const;
  uint32_t TupleSize() const { return tuple_size_; }
  uint64_t DirSize() const { return dir_mask_ + 1; }
  bool Finalized() const { return finalized_; }

  /* Raw probe accessors for compiled code (valid after Finalize). */
  Entry *const *DirData() const {
    assert(finalized_);
    static_assert(sizeof(std::atomic<Entry *>) == sizeof(Entry *),
                  "directory must be reinterpretable as a plain Entry* array");
    return reinterpret_cast<Entry *const *>(directory_.get());
  }
  uint64_t DirMask() const { return dir_mask_; }

  /* Key-0 range (valid after Finalize). Empty HT or no key0_offset:
   * min = INT64_MAX, max = INT64_MIN (range guard drops everything). */
  int64_t Key0Min() const { return key0_min_.load(std::memory_order_relaxed); }
  int64_t Key0Max() const { return key0_max_.load(std::memory_order_relaxed); }

  /* Per-worker string arena for varchar key/payload bytes whose source
   * lifetime does not cover the HT (e.g. temp-table rebuilds). */
  QjitStringArena &Arena(uint32_t worker_id) { return arenas_[worker_id]; }

private:
  struct Fragment {
    QjitBuffer buffer;
    uint64_t count = 0;
  };
  uint32_t tuple_size_;
  uint64_t entry_stride_;
  int64_t key0_offset_;
  std::vector<Fragment> fragments_;
  std::vector<QjitStringArena> arenas_;
  std::unique_ptr<std::atomic<Entry *>[]> directory_;
  uint64_t dir_mask_ = 0;
  std::atomic<int64_t> key0_min_{INT64_MAX};
  std::atomic<int64_t> key0_max_{INT64_MIN};
  bool finalized_ = false;
};

// ---------------------------------------------------------------------------
// QjitAggState — ungrouped aggregates: per-worker cells + Merge.
// ---------------------------------------------------------------------------

enum class QjitAggFn : uint8_t { Min, Max, Sum, Count, CountStar };
enum class QjitAggDType : uint8_t { I64, F64, Str };

struct QjitAggCellDesc {
  QjitAggFn fn;
  QjitAggDType dtype; /* ignored for Count/CountStar (always count)  */
};

struct QjitAggCell {
  union {
    int64_t i64;
    double f64;
  };
  QjitString str;  /* valid when dtype == Str (Min/Max only)         */
  uint64_t count;  /* Count/CountStar                                */
  bool seen;       /* false => no non-NULL input => NULL result      */
};

class QjitAggState {
public:
  QjitAggState(std::vector<QjitAggCellDesc> descs, QjitStringArena *arena);

  QjitAggCell &Cell(size_t i) { return cells_[i]; }
  const QjitAggCell &Cell(size_t i) const { return cells_[i]; }
  size_t NumCells() const { return cells_.size(); }
  const QjitAggCellDesc &Desc(size_t i) const { return descs_[i]; }

  /* Update entry points (also the reference semantics for codegen). NULL
   * inputs must be filtered by the caller — these assume a valid value. */
  void UpdateI64(size_t i, int64_t v);
  void UpdateF64(size_t i, double v);
  void UpdateStr(size_t i, const QjitString &v); /* deep-copies winner */
  void UpdateCount(size_t i) { cells_[i].count++; cells_[i].seen = true; }

  /* Fold `other` into this (single-threaded epilogue). */
  void Merge(const QjitAggState &other);

private:
  std::vector<QjitAggCellDesc> descs_;
  std::vector<QjitAggCell> cells_;
  QjitStringArena *arena_; /* owns string winners; not owned          */
};

// ---------------------------------------------------------------------------
// QjitTable — columnar sink (result or temp table).
//
// Build protocol: each worker appends rows into its own partition; a
// single-threaded Finalize concatenates partitions (worker order 0..N-1)
// into flat column arrays + 1-bit validity. VARCHAR bytes are always
// deep-copied into the table's arena at append time (lifetime safety).
// ---------------------------------------------------------------------------

class QjitTable {
public:
  struct ColumnDesc {
    int32_t dtype; /* AQP_DTYPE_INT32 / INT64 / DOUBLE / VARCHAR      */
    std::string name;
  };

  QjitTable(std::vector<ColumnDesc> cols, uint32_t num_workers);

  uint32_t ElemSize(size_t col) const;

  /* Worker-local appends; one row = one AppendRow + per-column value or
   * null. Concurrent across distinct worker_ids. */
  void AppendI32(uint32_t worker, size_t col, int32_t v);
  void AppendI64(uint32_t worker, size_t col, int64_t v);
  void AppendF64(uint32_t worker, size_t col, double v);
  void AppendStr(uint32_t worker, size_t col, const QjitString &v);
  void AppendNull(uint32_t worker, size_t col);
  void FinishRow(uint32_t worker) { partitions_[worker].nrows++; }

  /* Single-threaded: concatenate partitions into flat columns. */
  void Finalize();

  uint64_t NumRows() const { return nrows_; }
  size_t NumCols() const { return cols_.size(); }
  const ColumnDesc &Col(size_t i) const { return cols_[i]; }

  const void *Data(size_t col) const { return flat_[col].data.data(); }
  const uint64_t *Validity(size_t col) const {
    return flat_[col].validity.data();
  }
  bool ValueValid(size_t col, uint64_t row) const {
    return RowValid(flat_[col].validity.data(), row);
  }
  int32_t GetI32(size_t col, uint64_t row) const;
  int64_t GetI64(size_t col, uint64_t row) const;
  double GetF64(size_t col, uint64_t row) const;
  QjitString GetStr(size_t col, uint64_t row) const;

  /* Read-only view for use as a step source (valid after Finalize). */
  void FillView(QjitTableView *view, std::vector<QjitColView> *cols) const;

  /* Build flat columns directly (bypasses partition layer).
     Caller pre-allocates flat_[col].data and flat_[col].validity,
     then calls MarkFinalized(nrows). */
  void ReserveFlat(uint64_t total_rows);
  uint8_t *FlatData(size_t col) { return flat_[col].data.data(); }
  uint64_t *FlatValidity(size_t col) { return flat_[col].validity.data(); }
  QjitStringArena &FlatArena() { return flat_arena_; }
  void MarkFinalized(uint64_t total_rows) { nrows_ = total_rows; finalized_ = true; }

private:
  struct PartCol {
    QjitBuffer values;          /* fixed-size elems or QjitString      */
    std::vector<uint8_t> nulls; /* 1 byte per appended value, 1=NULL   */
  };
  struct Partition {
    std::vector<PartCol> cols;
    QjitStringArena arena; /* per-worker: string appends never race    */
    uint64_t nrows = 0;
  };
  struct FlatCol {
    std::vector<uint8_t> data;
    std::vector<uint64_t> validity;
  };

  void AppendBytes(uint32_t worker, size_t col, const void *src);

  std::vector<ColumnDesc> cols_;
  std::vector<Partition> partitions_;
  std::vector<FlatCol> flat_;
  QjitStringArena flat_arena_;
  uint64_t nrows_ = 0;
  bool finalized_ = false;
};

} // namespace qjit
