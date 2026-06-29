// Unit tests for the qjit runtime library + scheduler (Phase 1 gate):
//   - buffer growth across chunks
//   - NULL keys skipped at HT build (inner-join convention)
//   - string keys (incl. forced hash collisions)
//   - parallel build N ∈ {1,2,16} → identical probe results
//   - agg merge identity (split+merge == single state; empty input == NULL)
//   - QjitTable partition concat, validity bits, string deep-copy
//   - ParallelFor covers every index exactly once

#include <algorithm>
#include <atomic>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "qjit/query_jit_abi.h"
#include "qjit/query_jit_runtime.h"
#include "qjit/query_jit_scheduler.h"

using namespace qjit;

namespace {

uint64_t HashI64(int64_t k) { // FNV-1a over 8 bytes — deterministic, mixes
  uint64_t h = 1469598103934665603ull;
  for (int i = 0; i < 8; i++) {
    h ^= (uint64_t(k) >> (i * 8)) & 0xff;
    h *= 1099511628211ull;
  }
  return h;
}

uint64_t HashStr(const QjitString &s) {
  uint64_t h = 1469598103934665603ull;
  const char *d = StringData(s);
  for (uint32_t i = 0; i < StringLen(s); i++) {
    h ^= uint8_t(d[i]);
    h *= 1099511628211ull;
  }
  return h;
}

} // namespace

// ---------------------------------------------------------------------------
// QjitBuffer
// ---------------------------------------------------------------------------

TEST(QjitBuffer, GrowthAcrossChunks) {
  QjitBuffer buf(64); // tiny initial capacity to force many chunks
  std::vector<std::pair<uint8_t *, uint64_t>> allocs;
  uint64_t total = 0;
  std::mt19937 rng(42);
  for (int i = 0; i < 2000; i++) {
    uint64_t sz = 1 + rng() % 100;
    uint8_t *p = buf.Allocate(sz);
    std::memset(p, i & 0xff, sz);
    allocs.emplace_back(p, sz);
    total += sz;
  }
  EXPECT_EQ(buf.TotalSize(), total);
  EXPECT_GT(buf.Chunks().size(), 1u) << "growth must span multiple chunks";
  // Pointers stable, contents intact after all growth.
  for (size_t i = 0; i < allocs.size(); i++) {
    for (uint64_t b = 0; b < allocs[i].second; b++)
      ASSERT_EQ(allocs[i].first[b], uint8_t(i & 0xff));
  }
  // Oversized allocation gets its own chunk.
  uint8_t *big = buf.Allocate(1 << 20);
  std::memset(big, 0x5a, 1 << 20);
  EXPECT_EQ(big[(1 << 20) - 1], 0x5a);
  // C entry point hits the same buffer.
  auto *p = static_cast<uint8_t *>(qjit_buffer_grow(&buf, 16));
  std::memset(p, 0x77, 16);
  EXPECT_EQ(p[15], 0x77);
}

// ---------------------------------------------------------------------------
// QjitString / arena
// ---------------------------------------------------------------------------

TEST(QjitString, InlineAndPointerLayout) {
  ASSERT_EQ(sizeof(QjitString), 16u) << "must stay bit-compatible w/ string_t";
  QjitString a = MakeString("hello", 5); // inline
  EXPECT_EQ(StringLen(a), 5u);
  EXPECT_EQ(std::string(StringData(a), 5), "hello");
  const char *long_src = "this_is_longer_than_twelve_bytes";
  QjitString b = MakeString(long_src, 32); // pointer
  EXPECT_EQ(StringLen(b), 32u);
  EXPECT_EQ(StringData(b), long_src) << "MakeString must not copy";
  EXPECT_TRUE(StringEq(a, MakeString("hello", 5)));
  EXPECT_FALSE(StringEq(a, MakeString("hellp", 5)));
  EXPECT_LT(StringCmp(MakeString("abc", 3), MakeString("abd", 3)), 0);
  EXPECT_LT(StringCmp(MakeString("ab", 2), MakeString("abc", 3)), 0);
  EXPECT_EQ(StringCmp(b, b), 0);
}

TEST(QjitStringArena, DeepCopySurvivesSourceMutation) {
  std::string src = "0123456789ABCDEF0123456789ABCDEF"; // 32 B, not inline
  QjitStringArena arena;
  QjitString copy = arena.Copy(src.data(), (uint32_t)src.size());
  std::string expect = src;
  std::fill(src.begin(), src.end(), 'X'); // clobber the source
  EXPECT_EQ(std::string(StringData(copy), StringLen(copy)), expect);
  // C entry point path
  QjitString s2 = MakeString(src.data(), (uint32_t)src.size());
  QjitString dst;
  qjit_str_arena_copy(&arena, &dst, &s2);
  std::fill(src.begin(), src.end(), 'Y');
  EXPECT_EQ(std::string(StringData(dst), StringLen(dst)),
            std::string(32, 'X'));
}

// ---------------------------------------------------------------------------
// QjitHashTable
// ---------------------------------------------------------------------------

// Row payload for tests: [int64 key][int32 value]
struct TestRow {
  int64_t key;
  int32_t value;
};

TEST(QjitHashTable, NullKeysSkippedAtBuild) {
  // Inner-join convention: the build loop (codegen) skips NULL-key rows, so
  // they never enter the HT. Emulate that builder here.
  const int N = 1000;
  std::vector<int64_t> keys(N);
  std::vector<uint64_t> validity((N + 63) / 64, ~uint64_t(0));
  for (int i = 0; i < N; i++) {
    keys[i] = i % 100;
    if (i % 7 == 0)
      SetRowInvalid(validity.data(), i); // every 7th key is NULL
  }
  QjitHashTable ht(sizeof(TestRow), /*num_workers=*/1);
  uint64_t inserted = 0;
  for (int i = 0; i < N; i++) {
    if (!RowValid(validity.data(), i))
      continue; // NULL key skipped
    auto *row = reinterpret_cast<TestRow *>(ht.AppendRow(0, HashI64(keys[i])));
    row->key = keys[i];
    row->value = i;
    inserted++;
  }
  QjitQueryContext fctx;
  std::memset(&fctx, 0, sizeof(fctx)); // pool=nullptr → serial finalize
  qjit_ht_finalize(&fctx, &ht);
  EXPECT_EQ(ht.NumEntries(), inserted);
  EXPECT_LT(inserted, (uint64_t)N);
  // Every found row must be a non-NULL source row.
  for (int k = 0; k < 100; k++) {
    for (auto *e = ht.Lookup(HashI64(k)); e; e = e->next) {
      if (e->hash != HashI64(k))
        continue;
      auto *row = reinterpret_cast<const TestRow *>(e->Row());
      if (row->key != k)
        continue;
      EXPECT_TRUE(RowValid(validity.data(), row->value));
      EXPECT_NE(row->value % 7, 0);
    }
  }
}

// Probe all keys, return sorted matching values — canonical result for
// thread-invariance comparison.
static std::vector<int32_t> ProbeAll(const QjitHashTable &ht,
                                     const std::vector<int64_t> &probe_keys) {
  std::vector<int32_t> out;
  for (int64_t k : probe_keys) {
    uint64_t h = HashI64(k);
    for (auto *e = ht.Lookup(h); e; e = e->next) {
      if (e->hash != h)
        continue;
      auto *row = reinterpret_cast<const TestRow *>(e->Row());
      if (row->key == k)
        out.push_back(row->value);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

TEST(QjitHashTable, ParallelBuildThreadInvariance) {
  const uint64_t N = 50000;
  std::vector<int64_t> keys(N);
  std::mt19937_64 rng(7);
  for (auto &k : keys)
    k = rng() % 5000; // ~10 rows per key on average
  std::vector<int64_t> probe_keys;
  for (int64_t k = 0; k < 5200; k++) // include misses
    probe_keys.push_back(k);

  std::vector<int32_t> reference;
  for (uint32_t workers : {1u, 2u, 16u}) {
    QjitHashTable ht(sizeof(TestRow), workers);
    QjitWorkerPool pool(workers);
    pool.ParallelFor(N, /*morsel=*/1024,
                     [&](uint64_t begin, uint64_t end, uint32_t wid) {
                       for (uint64_t i = begin; i < end; i++) {
                         auto *row = reinterpret_cast<TestRow *>(
                             ht.AppendRow(wid, HashI64(keys[i])));
                         row->key = keys[i];
                         row->value = (int32_t)i;
                       }
                     });
    ASSERT_TRUE(pool.Idle());
    // workers=1 finalizes serially (reference); workers=2/16 take the
    // parallel CAS path. 16 also exercises the C ABI with a live pool.
    if (workers == 16) {
      QjitQueryContext fctx;
      std::memset(&fctx, 0, sizeof(fctx));
      fctx.pool = &pool;
      qjit_ht_finalize(&fctx, &ht);
    } else {
      ht.Finalize(&pool);
    }
    ASSERT_TRUE(pool.Idle());
    EXPECT_EQ(ht.NumEntries(), N);
    EXPECT_GE(ht.DirSize(), 2 * N);
    EXPECT_EQ(ht.DirSize() & (ht.DirSize() - 1), 0u) << "pow-2 directory";
    auto result = ProbeAll(ht, probe_keys);
    EXPECT_EQ(result.size(), N) << "every build row found exactly once";
    if (workers == 1)
      reference = result;
    else
      EXPECT_EQ(result, reference) << "workers=" << workers;
  }
}

TEST(QjitHashTable, StringKeysWithCollisions) {
  // Row payload: [QjitString key][int32 value]
  struct StrRow {
    QjitString key;
    int32_t value;
  };
  std::vector<std::string> names = {
      "a", "bb", "an_inline_str", "definitely_longer_than_twelve",
      "definitely_longer_than_twelvE", // shares 28-byte prefix w/ previous
      "x"};
  QjitHashTable ht(sizeof(StrRow), 1);
  // Force every entry into ONE bucket (hash = 0) — equality must
  // disambiguate purely via StringEq, exercising chain walks.
  for (size_t i = 0; i < names.size(); i++) {
    auto *row = reinterpret_cast<StrRow *>(ht.AppendRow(0, /*hash=*/0));
    row->key = ht.Arena(0).Copy(names[i].data(), (uint32_t)names[i].size());
    row->value = (int32_t)i;
  }
  ht.Finalize();
  for (size_t i = 0; i < names.size(); i++) {
    QjitString want = MakeString(names[i].data(), (uint32_t)names[i].size());
    int found = -1, matches = 0;
    for (auto *e = ht.Lookup(0); e; e = e->next) {
      auto *row = reinterpret_cast<const StrRow *>(e->Row());
      if (StringEq(row->key, want)) {
        found = row->value;
        matches++;
      }
    }
    EXPECT_EQ(matches, 1) << names[i];
    EXPECT_EQ(found, (int)i) << names[i];
  }
  // Miss probe
  QjitString miss = MakeString("not_present_key_here", 20);
  for (auto *e = ht.Lookup(HashStr(miss)); e; e = e->next) {
    auto *row = reinterpret_cast<const StrRow *>(e->Row());
    EXPECT_FALSE(StringEq(row->key, miss));
  }
}

// ---------------------------------------------------------------------------
// QjitAggState
// ---------------------------------------------------------------------------

TEST(QjitAggState, MergeIdentity) {
  std::vector<QjitAggCellDesc> descs = {
      {QjitAggFn::Min, QjitAggDType::I64},
      {QjitAggFn::Max, QjitAggDType::I64},
      {QjitAggFn::Sum, QjitAggDType::I64},
      {QjitAggFn::Count, QjitAggDType::I64},
      {QjitAggFn::CountStar, QjitAggDType::I64},
      {QjitAggFn::Min, QjitAggDType::Str},
      {QjitAggFn::Max, QjitAggDType::Str},
  };
  std::mt19937_64 rng(11);
  const int N = 10000;
  std::vector<int64_t> vals(N);
  std::vector<std::string> strs(N);
  for (int i = 0; i < N; i++) {
    vals[i] = (int64_t)(rng() % 1000000) - 500000;
    strs[i] = "str_padding_to_force_heap_" + std::to_string(rng() % 100000);
  }
  auto update_all = [&](QjitAggState &st, int i) {
    QjitString s = MakeString(strs[i].data(), (uint32_t)strs[i].size());
    st.UpdateI64(0, vals[i]);
    st.UpdateI64(1, vals[i]);
    st.UpdateI64(2, vals[i]);
    st.UpdateCount(3);
    st.UpdateCount(4);
    st.UpdateStr(5, s);
    st.UpdateStr(6, s);
  };

  QjitStringArena arena_single;
  QjitAggState single(descs, &arena_single);
  for (int i = 0; i < N; i++)
    update_all(single, i);

  for (int K : {2, 16}) {
    QjitStringArena arena_m;
    QjitAggState merged(descs, &arena_m);
    std::vector<QjitStringArena> arenas(K);
    for (int w = 0; w < K; w++) {
      QjitAggState part(descs, &arenas[w]);
      for (int i = w; i < N; i += K)
        update_all(part, i);
      merged.Merge(part);
    }
    for (size_t c = 0; c < descs.size(); c++) {
      ASSERT_TRUE(merged.Cell(c).seen);
      if (descs[c].dtype == QjitAggDType::Str &&
          (descs[c].fn == QjitAggFn::Min || descs[c].fn == QjitAggFn::Max)) {
        EXPECT_TRUE(StringEq(merged.Cell(c).str, single.Cell(c).str))
            << "cell " << c << " K=" << K;
      } else if (descs[c].fn == QjitAggFn::Count ||
                 descs[c].fn == QjitAggFn::CountStar) {
        EXPECT_EQ(merged.Cell(c).count, single.Cell(c).count);
      } else {
        EXPECT_EQ(merged.Cell(c).i64, single.Cell(c).i64) << "cell " << c;
      }
    }
  }
}

TEST(QjitAggState, EmptyInputIsNull) {
  std::vector<QjitAggCellDesc> descs = {{QjitAggFn::Min, QjitAggDType::I64},
                                        {QjitAggFn::Count, QjitAggDType::I64}};
  QjitStringArena arena;
  QjitAggState a(descs, &arena), b(descs, &arena);
  a.Merge(b); // merging never-updated states
  EXPECT_FALSE(a.Cell(0).seen) << "MIN over empty input must be NULL";
  EXPECT_EQ(a.Cell(1).count, 0u) << "COUNT over empty input is 0";
}

// ---------------------------------------------------------------------------
// QjitTable
// ---------------------------------------------------------------------------

TEST(QjitTable, PartitionConcatValidityAndStrings) {
  std::vector<QjitTable::ColumnDesc> cols = {
      {AQP_DTYPE_INT32, "id"}, {AQP_DTYPE_VARCHAR, "name"}};
  const uint32_t W = 3;
  QjitTable t(cols, W);
  // Enough rows per worker to force buffer growth across chunks.
  const int per_worker = 5000;
  for (uint32_t w = 0; w < W; w++) {
    for (int i = 0; i < per_worker; i++) {
      int32_t id = (int32_t)(w * per_worker + i);
      t.AppendI32(w, 0, id);
      if (i % 10 == 0) {
        t.AppendNull(w, 1);
      } else {
        std::string s = "name_with_long_padding_" + std::to_string(id);
        QjitString tmp = MakeString(s.data(), (uint32_t)s.size());
        t.AppendStr(w, 1, tmp); // s dies after this iteration → deep copy
      }
      t.FinishRow(w);
    }
  }
  t.Finalize();
  ASSERT_EQ(t.NumRows(), (uint64_t)W * per_worker);
  for (uint32_t w = 0; w < W; w++) {
    for (int i = 0; i < per_worker; i++) {
      uint64_t row = (uint64_t)w * per_worker + i;
      int32_t id = (int32_t)(w * per_worker + i);
      EXPECT_TRUE(t.ValueValid(0, row));
      ASSERT_EQ(t.GetI32(0, row), id) << "partition order broken @" << row;
      if (i % 10 == 0) {
        EXPECT_FALSE(t.ValueValid(1, row));
      } else {
        ASSERT_TRUE(t.ValueValid(1, row));
        QjitString s = t.GetStr(1, row);
        EXPECT_EQ(std::string(StringData(s), StringLen(s)),
                  "name_with_long_padding_" + std::to_string(id));
      }
    }
  }
  // View round-trip
  QjitTableView view;
  std::vector<QjitColView> view_cols;
  t.FillView(&view, &view_cols);
  EXPECT_EQ(view.nrows, t.NumRows());
  EXPECT_EQ(view.ncols, 2u);
  EXPECT_EQ(view.cols[0].dtype, AQP_DTYPE_INT32);
  EXPECT_EQ(reinterpret_cast<const int32_t *>(view.cols[0].data)[1], 1);
  EXPECT_FALSE(RowValid(view.cols[1].validity, 0)); // first name is NULL
}

// ---------------------------------------------------------------------------
// QjitWorkerPool / qjit_parallel_for
// ---------------------------------------------------------------------------

TEST(QjitWorkerPool, ParallelForCoversEveryIndexOnce) {
  for (uint32_t workers : {1u, 2u, 16u}) {
    QjitWorkerPool pool(workers);
    const uint64_t total = 100003; // prime-ish, not a morsel multiple
    std::vector<std::atomic<uint32_t>> hits(total);
    for (auto &h : hits)
      h.store(0);
    std::atomic<uint32_t> max_wid{0};
    pool.ParallelFor(total, /*morsel=*/97,
                     [&](uint64_t begin, uint64_t end, uint32_t wid) {
                       uint32_t cur = max_wid.load();
                       while (wid > cur && !max_wid.compare_exchange_weak(cur, wid)) {
                       }
                       for (uint64_t i = begin; i < end; i++)
                         hits[i].fetch_add(1, std::memory_order_relaxed);
                     });
    EXPECT_TRUE(pool.Idle());
    EXPECT_LT(max_wid.load(), workers);
    for (uint64_t i = 0; i < total; i++)
      ASSERT_EQ(hits[i].load(), 1u) << "index " << i << " workers=" << workers;
    // Pool reusable: second job on the same pool.
    std::atomic<uint64_t> sum{0};
    pool.ParallelFor(1000, 10, [&](uint64_t b, uint64_t e, uint32_t) {
      for (uint64_t i = b; i < e; i++)
        sum.fetch_add(i, std::memory_order_relaxed);
    });
    EXPECT_EQ(sum.load(), 1000ull * 999 / 2);
  }
}

TEST(QjitWorkerPool, CParallelForEntryPoint) {
  QjitWorkerPool pool(4);
  QjitQueryContext ctx;
  std::memset(&ctx, 0, sizeof(ctx));
  ctx.pool = &pool;
  static std::atomic<uint64_t> rows;
  rows.store(0);
  qjit_parallel_for(&ctx, 12345, 100,
                    [](QjitQueryContext *, uint64_t b, uint64_t e, uint32_t) {
                      rows.fetch_add(e - b, std::memory_order_relaxed);
                    });
  EXPECT_EQ(rows.load(), 12345u);
}
