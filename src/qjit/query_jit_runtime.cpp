#include "qjit/query_jit_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "qjit/query_jit_scheduler.h"

namespace qjit {

// ---------------------------------------------------------------------------
// QjitString
// ---------------------------------------------------------------------------

QjitString MakeString(const char *data, uint32_t len) {
  QjitString s;
  std::memset(&s, 0, sizeof(s));
  if (len <= QJIT_STRING_INLINE_LEN) {
    s.inlined.length = len;
    if (len)
      std::memcpy(s.inlined.inlined, data, len);
  } else {
    s.pointer.length = len;
    std::memcpy(s.pointer.prefix, data, 4);
    s.pointer.ptr = data;
  }
  return s;
}

// ---------------------------------------------------------------------------
// QjitBuffer
// ---------------------------------------------------------------------------

uint8_t *QjitBuffer::Allocate(uint64_t bytes) {
  if (chunks_.empty() || chunks_.back().used + bytes > chunks_.back().capacity) {
    uint64_t cap = next_capacity_;
    if (cap < bytes)
      cap = bytes;
    Chunk c;
    c.data.reset(new uint8_t[cap]);
    c.capacity = cap;
    chunks_.push_back(std::move(c));
    // 1.2x growth (lingo-db GrowingBuffer model)
    next_capacity_ = cap + cap / 5;
  }
  Chunk &c = chunks_.back();
  uint8_t *p = c.data.get() + c.used;
  c.used += bytes;
  total_size_ += bytes;
  return p;
}

// ---------------------------------------------------------------------------
// QjitStringArena
// ---------------------------------------------------------------------------

QjitString QjitStringArena::Copy(const QjitString &src) {
  return Copy(StringData(src), StringLen(src));
}

QjitString QjitStringArena::Copy(const char *data, uint32_t len) {
  if (len <= QJIT_STRING_INLINE_LEN)
    return MakeString(data, len); // inline: copied by value, no arena bytes
  char *dst = reinterpret_cast<char *>(bytes_.Allocate(len));
  std::memcpy(dst, data, len);
  return MakeString(dst, len);
}

// ---------------------------------------------------------------------------
// QjitHashTable
// ---------------------------------------------------------------------------

QjitHashTable::QjitHashTable(uint32_t tuple_size, uint32_t num_workers,
                             int64_t key0_offset)
    : tuple_size_(tuple_size),
      // Keep entries 8-byte aligned: header is 16 bytes, round the row up.
      entry_stride_(sizeof(Entry) + ((uint64_t(tuple_size) + 7) & ~uint64_t(7))),
      key0_offset_(key0_offset), fragments_(num_workers),
      arenas_(num_workers) {}

uint8_t *QjitHashTable::AppendRow(uint32_t worker_id, uint64_t hash) {
  assert(!finalized_);
  Fragment &f = fragments_[worker_id];
  auto *e = reinterpret_cast<Entry *>(f.buffer.Allocate(entry_stride_));
  e->next = nullptr;
  e->hash = hash;
  f.count++;
  return e->Row();
}

uint64_t QjitHashTable::NumEntries() const {
  uint64_t n = 0;
  for (const auto &f : fragments_)
    n += f.count;
  return n;
}

void QjitHashTable::Finalize(QjitWorkerPool *pool) {
  if (finalized_)
    return;
  static const bool trace = std::getenv("AQP_QJIT_FINALIZE_TRACE") != nullptr;
  auto t0 = trace ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};

  uint64_t n = NumEntries();
  uint64_t dir_size = 64;
  while (dir_size < 2 * n)
    dir_size <<= 1;
  directory_.reset(new std::atomic<Entry *>[dir_size]);
  dir_mask_ = dir_size - 1;

  // Flatten fragment chunks into ranges indexable by a global entry index,
  // so ParallelFor morsels map onto (chunk, offset) spans.
  struct Range {
    uint8_t *base;
    uint64_t first; // global index of this chunk's first entry
    uint64_t count;
  };
  std::vector<Range> ranges;
  uint64_t total = 0;
  for (auto &f : fragments_) {
    for (const auto &chunk : f.buffer.Chunks()) {
      uint64_t cnt = chunk.used / entry_stride_;
      if (cnt) {
        ranges.push_back({chunk.data.get(), total, cnt});
        total += cnt;
      }
    }
  }

  // §5.5 A+: fold a thread-local key-0 min/max into the shared atomics
  // (CAS-min/-max — contention is once per ParallelFor lambda invocation).
  auto merge_key_stats = [&](int64_t lmin, int64_t lmax) {
    if (lmin > lmax)
      return; // no entries seen
    int64_t cur = key0_min_.load(std::memory_order_relaxed);
    while (lmin < cur &&
           !key0_min_.compare_exchange_weak(cur, lmin,
                                            std::memory_order_relaxed)) {
    }
    cur = key0_max_.load(std::memory_order_relaxed);
    while (lmax > cur &&
           !key0_max_.compare_exchange_weak(cur, lmax,
                                            std::memory_order_relaxed)) {
    }
  };

  auto link = [&](uint64_t begin, uint64_t end) {
    int64_t lmin = INT64_MAX, lmax = INT64_MIN;
    size_t r = std::upper_bound(ranges.begin(), ranges.end(), begin,
                                [](uint64_t v, const Range &rg) {
                                  return v < rg.first;
                                }) -
               ranges.begin() - 1;
    for (uint64_t i = begin; i < end; r++) {
      const Range &rg = ranges[r];
      uint64_t off = i - rg.first;
      uint64_t stop = std::min(end - rg.first, rg.count);
      for (; off < stop; off++) {
        auto *e = reinterpret_cast<Entry *>(rg.base + off * entry_stride_);
        if (key0_offset_ >= 0) {
          int64_t k;
          std::memcpy(&k, e->Row() + key0_offset_, sizeof(k));
          if (k < lmin)
            lmin = k;
          if (k > lmax)
            lmax = k;
        }
        std::atomic<Entry *> &head = directory_[e->hash & dir_mask_];
        Entry *old = head.load(std::memory_order_relaxed);
        do {
          e->next = old;
        } while (!head.compare_exchange_weak(old, e, std::memory_order_release,
                                             std::memory_order_relaxed));
      }
      i = rg.first + stop;
    }
    if (key0_offset_ >= 0)
      merge_key_stats(lmin, lmax);
  };

  // memset is valid pre-publication: std::atomic<Entry*> is lock-free and
  // pointer-layout (static_assert in DirData); all-zero bytes == nullptr.
  auto zero_dir = [&](uint64_t b, uint64_t e) {
    std::memset(static_cast<void *>(directory_.get() + b), 0,
                (e - b) * sizeof(std::atomic<Entry *>));
  };

  // Below this, pool dispatch costs more than the link walk itself.
  constexpr uint64_t kParallelMin = 4096;
  bool parallel = pool && pool->NumWorkers() > 1 && total >= kParallelMin;
  if (parallel) {
    pool->ParallelFor(dir_size, /*morsel=*/uint64_t(1) << 18,
                      [&](uint64_t b, uint64_t e, uint32_t) { zero_dir(b, e); });
    uint64_t morsel = total / (uint64_t(pool->NumWorkers()) * 8);
    if (morsel < 1024)
      morsel = 1024;
    pool->ParallelFor(total, morsel,
                      [&](uint64_t b, uint64_t e, uint32_t) { link(b, e); });
    // ParallelFor's join orders all CAS pushes before any subsequent probe.
  } else {
    zero_dir(0, dir_size);
    link(0, total);
  }
  finalized_ = true;

  if (trace) {
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
    std::fprintf(stderr,
                 "[AQP-QJIT] ht_finalize n=%llu dir=%llu mode=%s %.3f ms\n",
                 (unsigned long long)n, (unsigned long long)dir_size,
                 parallel ? "parallel" : "serial", ms);
  }
}

// ---------------------------------------------------------------------------
// QjitAggState
// ---------------------------------------------------------------------------

QjitAggState::QjitAggState(std::vector<QjitAggCellDesc> descs,
                           QjitStringArena *arena)
    : descs_(std::move(descs)), cells_(descs_.size()), arena_(arena) {
  for (auto &c : cells_) {
    std::memset(&c, 0, sizeof(c));
  }
}

void QjitAggState::UpdateI64(size_t i, int64_t v) {
  QjitAggCell &c = cells_[i];
  switch (descs_[i].fn) {
  case QjitAggFn::Min:
    if (!c.seen || v < c.i64)
      c.i64 = v;
    break;
  case QjitAggFn::Max:
    if (!c.seen || v > c.i64)
      c.i64 = v;
    break;
  case QjitAggFn::Sum:
    c.i64 += v;
    break;
  case QjitAggFn::Count:
  case QjitAggFn::CountStar:
    c.count++;
    break;
  }
  c.seen = true;
}

void QjitAggState::UpdateF64(size_t i, double v) {
  QjitAggCell &c = cells_[i];
  switch (descs_[i].fn) {
  case QjitAggFn::Min:
    if (!c.seen || v < c.f64)
      c.f64 = v;
    break;
  case QjitAggFn::Max:
    if (!c.seen || v > c.f64)
      c.f64 = v;
    break;
  case QjitAggFn::Sum:
    c.f64 += v;
    break;
  case QjitAggFn::Count:
  case QjitAggFn::CountStar:
    c.count++;
    break;
  }
  c.seen = true;
}

void QjitAggState::UpdateStr(size_t i, const QjitString &v) {
  QjitAggCell &c = cells_[i];
  switch (descs_[i].fn) {
  case QjitAggFn::Min:
    if (!c.seen || StringCmp(v, c.str) < 0)
      c.str = arena_->Copy(v);
    break;
  case QjitAggFn::Max:
    if (!c.seen || StringCmp(v, c.str) > 0)
      c.str = arena_->Copy(v);
    break;
  case QjitAggFn::Sum:
    assert(false && "sum over strings");
    break;
  case QjitAggFn::Count:
  case QjitAggFn::CountStar:
    c.count++;
    break;
  }
  c.seen = true;
}

void QjitAggState::Merge(const QjitAggState &other) {
  assert(other.cells_.size() == cells_.size());
  for (size_t i = 0; i < cells_.size(); i++) {
    const QjitAggCell &o = other.cells_[i];
    if (!o.seen)
      continue;
    QjitAggCell &c = cells_[i];
    switch (descs_[i].fn) {
    case QjitAggFn::Min:
    case QjitAggFn::Max: {
      bool take;
      switch (descs_[i].dtype) {
      case QjitAggDType::I64:
        take = !c.seen || (descs_[i].fn == QjitAggFn::Min ? o.i64 < c.i64
                                                          : o.i64 > c.i64);
        if (take)
          c.i64 = o.i64;
        break;
      case QjitAggDType::F64:
        take = !c.seen || (descs_[i].fn == QjitAggFn::Min ? o.f64 < c.f64
                                                          : o.f64 > c.f64);
        if (take)
          c.f64 = o.f64;
        break;
      case QjitAggDType::Str:
        take = !c.seen || (descs_[i].fn == QjitAggFn::Min
                               ? StringCmp(o.str, c.str) < 0
                               : StringCmp(o.str, c.str) > 0);
        if (take)
          c.str = arena_->Copy(o.str);
        break;
      }
      break;
    }
    case QjitAggFn::Sum:
      if (descs_[i].dtype == QjitAggDType::F64)
        c.f64 += o.f64;
      else
        c.i64 += o.i64;
      break;
    case QjitAggFn::Count:
    case QjitAggFn::CountStar:
      c.count += o.count;
      break;
    }
    c.seen = true;
  }
}

// ---------------------------------------------------------------------------
// QjitTable
// ---------------------------------------------------------------------------

QjitTable::QjitTable(std::vector<ColumnDesc> cols, uint32_t num_workers)
    : cols_(std::move(cols)), partitions_(num_workers) {
  for (auto &p : partitions_)
    p.cols.resize(cols_.size());
}

uint32_t QjitTable::ElemSize(size_t col) const {
  switch (cols_[col].dtype) {
  case AQP_DTYPE_INT32:
    return 4;
  case AQP_DTYPE_INT64:
    return 8;
  case AQP_DTYPE_DOUBLE:
    return 8;
  case AQP_DTYPE_VARCHAR:
    return sizeof(QjitString);
  default:
    assert(false && "unsupported qjit table dtype");
    return 0;
  }
}

void QjitTable::AppendBytes(uint32_t worker, size_t col, const void *src) {
  PartCol &pc = partitions_[worker].cols[col];
  uint32_t sz = ElemSize(col);
  std::memcpy(pc.values.Allocate(sz), src, sz);
  pc.nulls.push_back(0);
}

void QjitTable::AppendI32(uint32_t worker, size_t col, int32_t v) {
  assert(cols_[col].dtype == AQP_DTYPE_INT32);
  AppendBytes(worker, col, &v);
}
void QjitTable::AppendI64(uint32_t worker, size_t col, int64_t v) {
  assert(cols_[col].dtype == AQP_DTYPE_INT64);
  AppendBytes(worker, col, &v);
}
void QjitTable::AppendF64(uint32_t worker, size_t col, double v) {
  assert(cols_[col].dtype == AQP_DTYPE_DOUBLE);
  AppendBytes(worker, col, &v);
}
void QjitTable::AppendStr(uint32_t worker, size_t col, const QjitString &v) {
  assert(cols_[col].dtype == AQP_DTYPE_VARCHAR);
  // Deep copy into the worker-local arena: source chunk lifetime doesn't
  // matter and concurrent workers never touch the same arena.
  QjitString owned = partitions_[worker].arena.Copy(v);
  AppendBytes(worker, col, &owned);
}
void QjitTable::AppendNull(uint32_t worker, size_t col) {
  PartCol &pc = partitions_[worker].cols[col];
  uint32_t sz = ElemSize(col);
  std::memset(pc.values.Allocate(sz), 0, sz);
  pc.nulls.push_back(1);
}

void QjitTable::Finalize() {
  if (finalized_)
    return;
  nrows_ = 0;
  for (const auto &p : partitions_)
    nrows_ += p.nrows;
  flat_.resize(cols_.size());
  for (size_t c = 0; c < cols_.size(); c++) {
    FlatCol &fc = flat_[c];
    uint32_t sz = ElemSize(c);
    fc.data.resize(nrows_ * sz);
    fc.validity.assign((nrows_ + 63) / 64, ~uint64_t(0));
    uint64_t row = 0;
    for (const auto &p : partitions_) {
      const PartCol &pc = p.cols[c];
      assert(pc.nulls.size() == p.nrows && "column appends != FinishRow count");
      uint64_t idx = 0;
      for (const auto &chunk : pc.values.Chunks()) {
        std::memcpy(fc.data.data() + (row + idx) * sz, chunk.data.get(),
                    chunk.used);
        idx += chunk.used / sz;
      }
      for (uint64_t r = 0; r < p.nrows; r++)
        if (pc.nulls[r])
          SetRowInvalid(fc.validity.data(), row + r);
      row += p.nrows;
    }
  }
  finalized_ = true;
}

int32_t QjitTable::GetI32(size_t col, uint64_t row) const {
  return reinterpret_cast<const int32_t *>(flat_[col].data.data())[row];
}
int64_t QjitTable::GetI64(size_t col, uint64_t row) const {
  return reinterpret_cast<const int64_t *>(flat_[col].data.data())[row];
}
double QjitTable::GetF64(size_t col, uint64_t row) const {
  return reinterpret_cast<const double *>(flat_[col].data.data())[row];
}
QjitString QjitTable::GetStr(size_t col, uint64_t row) const {
  return reinterpret_cast<const QjitString *>(flat_[col].data.data())[row];
}

void QjitTable::FillView(QjitTableView *view,
                         std::vector<QjitColView> *cols) const {
  assert(finalized_);
  cols->resize(cols_.size());
  for (size_t c = 0; c < cols_.size(); c++) {
    (*cols)[c].data = const_cast<uint8_t *>(flat_[c].data.data());
    (*cols)[c].validity = const_cast<uint64_t *>(flat_[c].validity.data());
    (*cols)[c].dtype = cols_[c].dtype;
    (*cols)[c].reserved = 0;
  }
  view->cols = cols->data();
  view->nrows = nrows_;
  view->ncols = cols_.size();
}

} // namespace qjit

// ---------------------------------------------------------------------------
// extern "C" entry points (qjit_parallel_for lives in the scheduler TU)
// ---------------------------------------------------------------------------

extern "C" {

void *qjit_buffer_grow(void *buffer, uint64_t bytes) {
  return static_cast<qjit::QjitBuffer *>(buffer)->Allocate(bytes);
}

void *qjit_ht_append(void *ht, uint32_t worker_id, uint64_t hash) {
  return static_cast<qjit::QjitHashTable *>(ht)->AppendRow(worker_id, hash);
}

void qjit_ht_finalize(void *ctx, void *ht) {
  auto *qctx = static_cast<QjitQueryContext *>(ctx);
  static_cast<qjit::QjitHashTable *>(ht)->Finalize(
      static_cast<qjit::QjitWorkerPool *>(qctx->pool));
}

void *qjit_ht_dir(void *ht) {
  return const_cast<qjit::QjitHashTable::Entry **>(
      static_cast<qjit::QjitHashTable *>(ht)->DirData());
}

uint64_t qjit_ht_mask(void *ht) {
  return static_cast<qjit::QjitHashTable *>(ht)->DirMask();
}

int64_t qjit_ht_key0_min(void *ht) {
  return static_cast<qjit::QjitHashTable *>(ht)->Key0Min();
}

int64_t qjit_ht_key0_max(void *ht) {
  return static_cast<qjit::QjitHashTable *>(ht)->Key0Max();
}

uint64_t qjit_ht_entries(void *ht) {
  return static_cast<qjit::QjitHashTable *>(ht)->NumEntries();
}

void qjit_agg_update_i64(void *state, uint64_t cell, int64_t v) {
  static_cast<qjit::QjitAggState *>(state)->UpdateI64(cell, v);
}

void qjit_agg_update_str(void *state, uint64_t cell, const QjitString *v) {
  static_cast<qjit::QjitAggState *>(state)->UpdateStr(cell, *v);
}

void qjit_agg_update_count(void *state, uint64_t cell) {
  static_cast<qjit::QjitAggState *>(state)->UpdateCount(cell);
}

void qjit_str_arena_copy(void *arena, QjitString *dst, const QjitString *src) {
  *dst = static_cast<qjit::QjitStringArena *>(arena)->Copy(*src);
}

void qjit_table_append_i32(void *table, uint32_t worker_id, uint64_t col,
                           int32_t v) {
  static_cast<qjit::QjitTable *>(table)->AppendI32(worker_id, col, v);
}

void qjit_table_append_i64(void *table, uint32_t worker_id, uint64_t col,
                           int64_t v) {
  static_cast<qjit::QjitTable *>(table)->AppendI64(worker_id, col, v);
}

void qjit_table_append_str(void *table, uint32_t worker_id, uint64_t col,
                           const QjitString *v) {
  static_cast<qjit::QjitTable *>(table)->AppendStr(worker_id, col, *v);
}

void qjit_table_append_null(void *table, uint32_t worker_id, uint64_t col) {
  static_cast<qjit::QjitTable *>(table)->AppendNull(worker_id, col);
}

void qjit_table_finish_row(void *table, uint32_t worker_id) {
  static_cast<qjit::QjitTable *>(table)->FinishRow(worker_id);
}

} // extern "C"
