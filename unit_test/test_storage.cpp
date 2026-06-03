#include "storage/flat_table.h"
#include "storage/csr_index.h"
#include "storage/sorted_index.h"
#include "kernel/sub_query_plan.h"
#include "kernel/pipeline_kernel.h"

#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace middleware::storage;

// ============================================================
// Helpers
// ============================================================

static std::unique_ptr<FlatTable> MakeIntTable(
    const std::string &name,
    const std::vector<std::string> &col_names,
    const std::vector<std::vector<int32_t>> &col_data) {
  auto ft = std::make_unique<FlatTable>();
  ft->table_name = name;
  ft->column_names = col_names;
  ft->row_count = col_data.empty() ? 0 : col_data[0].size();
  ft->columns.resize(col_names.size());
  for (size_t c = 0; c < col_names.size(); c++) {
    auto &col = ft->columns[c];
    col.type = FlatColumnType::INT32;
    col.row_count = ft->row_count;
    col.nullable = false;
    col.data = std::unique_ptr<char[]>(
        new char[ft->row_count * sizeof(int32_t)]);
    if (ft->row_count > 0)
      std::memcpy(col.data.get(), col_data[c].data(),
                  ft->row_count * sizeof(int32_t));
  }
  return ft;
}

static std::unique_ptr<FlatTable> MakeVarcharTable(
    const std::string &name,
    const std::vector<std::string> &col_names,
    const std::vector<int32_t> &id_data,
    const std::vector<std::string> &str_data) {
  auto ft = std::make_unique<FlatTable>();
  ft->table_name = name;
  ft->column_names = col_names;
  ft->row_count = id_data.size();
  ft->columns.resize(2);

  auto &id_col = ft->columns[0];
  id_col.type = FlatColumnType::INT32;
  id_col.row_count = ft->row_count;
  id_col.nullable = false;
  id_col.data = std::unique_ptr<char[]>(
      new char[ft->row_count * sizeof(int32_t)]);
  std::memcpy(id_col.data.get(), id_data.data(),
              ft->row_count * sizeof(int32_t));

  auto &str_col = ft->columns[1];
  str_col.type = FlatColumnType::VARCHAR;
  str_col.row_count = ft->row_count;
  str_col.nullable = false;
  uint64_t total_len = 0;
  for (const auto &s : str_data)
    total_len += s.size();
  str_col.data = std::unique_ptr<char[]>(
      new char[(ft->row_count + 1) * sizeof(uint32_t)]);
  str_col.string_pool = std::unique_ptr<char[]>(new char[std::max(total_len, (uint64_t)1)]);
  str_col.string_pool_size = total_len;
  auto *offsets = reinterpret_cast<uint32_t *>(str_col.data.get());
  uint32_t off = 0;
  for (uint64_t r = 0; r < ft->row_count; r++) {
    offsets[r] = off;
    std::memcpy(str_col.string_pool.get() + off, str_data[r].data(),
                str_data[r].size());
    off += static_cast<uint32_t>(str_data[r].size());
  }
  offsets[ft->row_count] = off;
  return ft;
}

// Build a multi-column FlatTable with both INT32 and VARCHAR columns
static std::unique_ptr<FlatTable> MakeMixedTable(
    const std::string &name,
    const std::vector<std::string> &col_names,
    const std::vector<FlatColumnType> &col_types,
    const std::vector<std::vector<int32_t>> &int_data,
    const std::vector<std::vector<std::string>> &str_data) {
  auto ft = std::make_unique<FlatTable>();
  ft->table_name = name;
  ft->column_names = col_names;
  size_t n_rows = 0;
  for (const auto &v : int_data) { if (!v.empty()) { n_rows = v.size(); break; } }
  if (n_rows == 0)
    for (const auto &v : str_data) { if (!v.empty()) { n_rows = v.size(); break; } }
  ft->row_count = n_rows;
  ft->columns.resize(col_names.size());

  size_t int_idx = 0, str_idx = 0;
  for (size_t c = 0; c < col_names.size(); c++) {
    auto &col = ft->columns[c];
    col.type = col_types[c];
    col.row_count = ft->row_count;
    col.nullable = false;
    if (col_types[c] == FlatColumnType::INT32) {
      col.data = std::unique_ptr<char[]>(new char[ft->row_count * sizeof(int32_t)]);
      std::memcpy(col.data.get(), int_data[int_idx].data(),
                  ft->row_count * sizeof(int32_t));
      int_idx++;
    } else {
      const auto &strs = str_data[str_idx];
      uint64_t total_len = 0;
      for (const auto &s : strs) total_len += s.size();
      col.data = std::unique_ptr<char[]>(new char[(ft->row_count + 1) * sizeof(uint32_t)]);
      col.string_pool = std::unique_ptr<char[]>(new char[std::max(total_len, (uint64_t)1)]);
      col.string_pool_size = total_len;
      auto *offsets = reinterpret_cast<uint32_t *>(col.data.get());
      uint32_t off = 0;
      for (uint64_t r = 0; r < ft->row_count; r++) {
        offsets[r] = off;
        std::memcpy(col.string_pool.get() + off, strs[r].data(), strs[r].size());
        off += static_cast<uint32_t>(strs[r].size());
      }
      offsets[ft->row_count] = off;
      str_idx++;
    }
  }
  return ft;
}

static std::string GetStr(const FlatColumn &col, uint64_t row) {
  uint32_t len;
  const char *ptr = col.GetVarchar(row, len);
  return std::string(ptr, len);
}

// ============================================================
// FlatTable Tests
// ============================================================

TEST(FlatColumn, GetInt32) {
  auto ft = MakeIntTable("t", {"a"}, {{10, 20, 30}});
  EXPECT_EQ(ft->columns[0].GetInt32(0), 10);
  EXPECT_EQ(ft->columns[0].GetInt32(1), 20);
  EXPECT_EQ(ft->columns[0].GetInt32(2), 30);
}

TEST(FlatColumn, GetVarchar) {
  auto ft = MakeVarcharTable("t", {"id", "name"}, {1, 2, 3},
                              {"hello", "world", ""});
  EXPECT_EQ(GetStr(ft->columns[1], 0), "hello");
  EXPECT_EQ(GetStr(ft->columns[1], 1), "world");
  EXPECT_EQ(GetStr(ft->columns[1], 2), "");
}

TEST(FlatColumn, LongStrings) {
  std::string long1(1000, 'A');
  std::string long2(5000, 'B');
  auto ft = MakeVarcharTable("t", {"id", "s"}, {1, 2}, {long1, long2});
  EXPECT_EQ(GetStr(ft->columns[1], 0), long1);
  EXPECT_EQ(GetStr(ft->columns[1], 1), long2);
}

TEST(FlatTable, FindColumn) {
  auto ft = MakeIntTable("t", {"x", "y", "z"}, {{1}, {2}, {3}});
  EXPECT_EQ(ft->FindColumn("x"), 0);
  EXPECT_EQ(ft->FindColumn("y"), 1);
  EXPECT_EQ(ft->FindColumn("z"), 2);
  EXPECT_EQ(ft->FindColumn("w"), -1);
}

// ============================================================
// CSR Index Tests
// ============================================================

TEST(CSRIndex, BuildAndLookup) {
  // Row 0: fk=2, Row 1: fk=0, Row 2: fk=2, Row 3: fk=4, Row 4: fk=1
  auto fk_table = MakeIntTable("fk", {"fk_col"}, {{2, 0, 2, 4, 1}});
  auto csr = BuildCSR(*fk_table, 0, 4, "fk", "fk_col", "pk", "id");

  auto [b0, e0] = csr.Lookup(0);
  ASSERT_EQ(e0 - b0, 1);
  EXPECT_EQ(*b0, 1u);

  auto [b1, e1] = csr.Lookup(1);
  ASSERT_EQ(e1 - b1, 1);
  EXPECT_EQ(*b1, 4u);

  auto [b2, e2] = csr.Lookup(2);
  EXPECT_EQ(e2 - b2, 2);

  auto [b3, e3] = csr.Lookup(3);
  EXPECT_EQ(e3 - b3, 0);

  auto [b4, e4] = csr.Lookup(4);
  ASSERT_EQ(e4 - b4, 1);
  EXPECT_EQ(*b4, 3u);
}

TEST(CSRIndex, OutOfRange) {
  auto fk_table = MakeIntTable("fk", {"fk_col"}, {{0, 1}});
  auto csr = BuildCSR(*fk_table, 0, 1, "fk", "fk_col", "pk", "id");

  auto [bn, en] = csr.Lookup(-1);
  EXPECT_EQ(bn, nullptr);
  EXPECT_EQ(en, nullptr);

  auto [bo, eo] = csr.Lookup(2);
  EXPECT_EQ(bo, nullptr);
  EXPECT_EQ(eo, nullptr);
}

TEST(CSRIndex, EmptyTable) {
  auto fk_table = MakeIntTable("fk", {"fk_col"}, {{}});
  fk_table->row_count = 0;
  fk_table->columns[0].row_count = 0;
  auto csr = BuildCSR(*fk_table, 0, 0, "fk", "fk_col", "pk", "id");
  auto [b, e] = csr.Lookup(0);
  EXPECT_EQ(e - b, 0);
}

TEST(CSRIndex, AllSameFK) {
  auto fk_table = MakeIntTable("fk", {"fk_col"}, {{3, 3, 3, 3, 3}});
  auto csr = BuildCSR(*fk_table, 0, 3, "fk", "fk_col", "pk", "id");
  auto [b3, e3] = csr.Lookup(3);
  EXPECT_EQ(e3 - b3, 5);
  auto [b0, e0] = csr.Lookup(0);
  EXPECT_EQ(e0 - b0, 0);
}

TEST(CSRIndex, NegativeValues) {
  auto fk_table = MakeIntTable("fk", {"fk_col"}, {{-1, 0, 1}});
  auto csr = BuildCSR(*fk_table, 0, 1, "fk", "fk_col", "pk", "id");
  auto [bn, en] = csr.Lookup(-1);
  EXPECT_EQ(bn, nullptr);
  auto [b0, e0] = csr.Lookup(0);
  EXPECT_EQ(e0 - b0, 1);
  auto [b1, e1] = csr.Lookup(1);
  EXPECT_EQ(e1 - b1, 1);
}

TEST(CSRIndex, LargeScale) {
  const int N = 100000;
  const int MAX_PK = 10000;
  std::vector<int32_t> fk_vals(N);
  for (int i = 0; i < N; i++)
    fk_vals[i] = i % MAX_PK;
  auto fk_table = MakeIntTable("fk", {"fk_col"}, {fk_vals});
  auto csr = BuildCSR(*fk_table, 0, MAX_PK - 1, "fk", "fk_col", "pk", "id");
  for (int pk = 0; pk < MAX_PK; pk++) {
    auto [b, e] = csr.Lookup(pk);
    EXPECT_EQ(e - b, N / MAX_PK);
  }
}

// ============================================================
// SubQueryPlan Executor Tests
// ============================================================

TEST(SubQueryPlan, SemiJoinBasic) {
  auto scan = MakeIntTable("ci", {"id", "movie_id"},
                            {{100, 101, 102, 103, 104}, {1, 2, 3, 4, 5}});
  auto temp = MakeIntTable("temp1", {"movie_id"}, {{2, 4}});
  auto csr = BuildCSR(*temp, 0, 5, "temp1", "movie_id", "", "");

  SubQueryPlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "ci";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 1;
  step.joined_table = temp.get();
  step.is_semi = true;
  plan.join_steps.push_back(step);

  KernelOutputCol out0{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
  KernelOutputCol out1{KernelOutputCol::FROM_SCAN, -1, 1, FlatColumnType::INT32, "movie_id"};
  plan.output_cols = {out0, out1};

  auto result = ExecuteSubQueryPlan(plan, "result");
  ASSERT_EQ(result->row_count, 2u);
  EXPECT_EQ(result->columns[0].GetInt32(0), 101);
  EXPECT_EQ(result->columns[1].GetInt32(0), 2);
  EXPECT_EQ(result->columns[0].GetInt32(1), 103);
  EXPECT_EQ(result->columns[1].GetInt32(1), 4);
}

TEST(SubQueryPlan, SemiJoinNoMatches) {
  auto scan = MakeIntTable("t", {"id", "fk"}, {{1, 2, 3}, {10, 20, 30}});
  auto temp = MakeIntTable("t2", {"val"}, {{99}});
  auto csr = BuildCSR(*temp, 0, 99, "t2", "val", "", "");

  SubQueryPlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "t";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 1;
  step.joined_table = temp.get();
  step.is_semi = true;
  plan.join_steps.push_back(step);

  KernelOutputCol out{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
  plan.output_cols = {out};

  auto result = ExecuteSubQueryPlan(plan, "result");
  EXPECT_EQ(result->row_count, 0u);
}

TEST(SubQueryPlan, InnerJoinBasic) {
  auto scan = MakeIntTable("orders", {"order_id", "cust_id"},
                            {{10, 20, 30}, {1, 2, 1}});
  auto lookup = MakeVarcharTable("custs", {"cust_id", "name"},
                                  {1, 1, 2}, {"Alice", "Alicia", "Bob"});
  auto csr = BuildCSR(*lookup, 0, 2, "custs", "cust_id", "", "");

  SubQueryPlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "orders";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 1;
  step.joined_table = lookup.get();
  step.is_semi = false;
  plan.join_steps.push_back(step);

  KernelOutputCol out0{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "order_id"};
  KernelOutputCol out1{KernelOutputCol::FROM_JOIN, 0, 1, FlatColumnType::VARCHAR, "name"};
  plan.output_cols = {out0, out1};

  auto result = ExecuteSubQueryPlan(plan, "result");
  ASSERT_EQ(result->row_count, 5u);
  EXPECT_EQ(result->columns[0].GetInt32(0), 10);
  EXPECT_EQ(result->columns[0].GetInt32(1), 10);
  EXPECT_EQ(result->columns[0].GetInt32(2), 20);
  EXPECT_EQ(result->columns[0].GetInt32(3), 30);
  EXPECT_EQ(result->columns[0].GetInt32(4), 30);
  EXPECT_EQ(GetStr(result->columns[1], 0), "Alice");
  EXPECT_EQ(GetStr(result->columns[1], 1), "Alicia");
  EXPECT_EQ(GetStr(result->columns[1], 2), "Bob");
  EXPECT_EQ(GetStr(result->columns[1], 3), "Alice");
  EXPECT_EQ(GetStr(result->columns[1], 4), "Alicia");
}

TEST(SubQueryPlan, InnerJoinEmptyLookup) {
  auto scan = MakeIntTable("t", {"id", "fk"}, {{1, 2}, {10, 20}});
  auto lookup = MakeIntTable("t2", {"key"}, {{}});
  lookup->row_count = 0;
  lookup->columns[0].row_count = 0;
  auto csr = BuildCSR(*lookup, 0, 0, "t2", "key", "", "");

  SubQueryPlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "t";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 1;
  step.joined_table = lookup.get();
  step.is_semi = false;
  plan.join_steps.push_back(step);

  KernelOutputCol out{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
  plan.output_cols = {out};

  auto result = ExecuteSubQueryPlan(plan, "result");
  EXPECT_EQ(result->row_count, 0u);
}

// Mimics 9b iteration 5: large scan table, inner join with VARCHAR output
// from both sides, small lookup table
TEST(SubQueryPlan, InnerJoinMixedColumnsLargeScan) {
  const int SCAN_ROWS = 50000;
  const int LOOKUP_ROWS = 100;

  // Scan table: id (INT32), name (VARCHAR)
  std::vector<int32_t> scan_ids(SCAN_ROWS);
  std::vector<std::string> scan_names(SCAN_ROWS);
  for (int i = 0; i < SCAN_ROWS; i++) {
    scan_ids[i] = i;
    scan_names[i] = "scan_name_" + std::to_string(i);
  }
  auto scan = MakeMixedTable("char_name", {"id", "name"},
                              {FlatColumnType::INT32, FlatColumnType::VARCHAR},
                              {scan_ids}, {scan_names});

  // Lookup table: person_role_id (INT32), lname (VARCHAR), lid (INT32), lmovie (INT32)
  // person_role_id values will match some scan ids
  std::vector<int32_t> lookup_pr_ids(LOOKUP_ROWS);
  std::vector<std::string> lookup_names(LOOKUP_ROWS);
  std::vector<int32_t> lookup_ids(LOOKUP_ROWS);
  std::vector<int32_t> lookup_movies(LOOKUP_ROWS);
  for (int i = 0; i < LOOKUP_ROWS; i++) {
    lookup_pr_ids[i] = i * 500; // spread across scan range
    lookup_names[i] = "lookup_name_" + std::to_string(i);
    lookup_ids[i] = 1000 + i;
    lookup_movies[i] = 2000 + i;
  }
  auto lookup = MakeMixedTable("temp4", {"person_role_id", "lname", "lid", "lmovie"},
                                {FlatColumnType::INT32, FlatColumnType::VARCHAR,
                                 FlatColumnType::INT32, FlatColumnType::INT32},
                                {lookup_pr_ids, lookup_ids, lookup_movies},
                                {lookup_names});

  // CSR on lookup's person_role_id
  int32_t max_val = *std::max_element(lookup_pr_ids.begin(), lookup_pr_ids.end());
  auto csr = BuildCSR(*lookup, 0, max_val, "temp4", "person_role_id", "", "");

  SubQueryPlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "char_name";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 0; // join on scan.id = lookup.person_role_id
  step.joined_table = lookup.get();
  step.is_semi = false;
  plan.join_steps.push_back(step);

  // Output: scan.name (VARCHAR), lookup.lname (VARCHAR), lookup.lid (INT32), lookup.lmovie (INT32)
  KernelOutputCol out0{KernelOutputCol::FROM_SCAN, -1, 1, FlatColumnType::VARCHAR, "name"};
  KernelOutputCol out1{KernelOutputCol::FROM_JOIN, 0, 1, FlatColumnType::VARCHAR, "lname"};
  KernelOutputCol out2{KernelOutputCol::FROM_JOIN, 0, 2, FlatColumnType::INT32, "lid"};
  KernelOutputCol out3{KernelOutputCol::FROM_JOIN, 0, 3, FlatColumnType::INT32, "lmovie"};
  plan.output_cols = {out0, out1, out2, out3};

  auto result = ExecuteSubQueryPlan(plan, "temp5");

  ASSERT_EQ(result->row_count, (uint64_t)LOOKUP_ROWS);
  ASSERT_EQ(result->columns.size(), 4u);
  EXPECT_EQ(result->column_names[0], "name");
  EXPECT_EQ(result->column_names[1], "lname");

  // Verify data integrity
  for (uint64_t r = 0; r < result->row_count; r++) {
    int scan_id = (int)r * 500;
    std::string expected_scan_name = "scan_name_" + std::to_string(scan_id);
    std::string expected_lookup_name = "lookup_name_" + std::to_string(r);
    EXPECT_EQ(GetStr(result->columns[0], r), expected_scan_name);
    EXPECT_EQ(GetStr(result->columns[1], r), expected_lookup_name);
    EXPECT_EQ(result->columns[2].GetInt32(r), 1000 + (int32_t)r);
    EXPECT_EQ(result->columns[3].GetInt32(r), 2000 + (int32_t)r);
  }
}

// Test chained kernel executions: first kernel produces temp,
// then second kernel uses that temp via runtime CSR
TEST(SubQueryPlan, ChainedKernelExecutions) {
  // Step 1: DuckDB-like temp with FK values
  auto base = MakeIntTable("movie_keyword", {"id", "movie_id", "keyword_id"},
                            {{0, 1, 2, 3, 4}, {100, 200, 300, 400, 500}, {10, 20, 10, 30, 20}});
  auto dim = MakeIntTable("keyword", {"id"}, {{10, 20}});
  auto csr1 = BuildCSR(*dim, 0, 30, "keyword", "id", "", "");

  SubQueryPlan plan1;
  plan1.scan_table = base.get();
  plan1.scan_table_name = "movie_keyword";
  plan1.valid = true;

  KernelJoinStep step1;
  step1.csr = &csr1;
  step1.scan_key_col_idx = 2; // keyword_id
  step1.joined_table = dim.get();
  step1.is_semi = true;
  plan1.join_steps.push_back(step1);

  KernelOutputCol out_id{KernelOutputCol::FROM_SCAN, -1, 1, FlatColumnType::INT32, "movie_id"};
  plan1.output_cols = {out_id};

  auto temp1 = ExecuteSubQueryPlan(plan1, "temp1");
  ASSERT_EQ(temp1->row_count, 4u); // ids 10,20 match rows 0,1,2,4

  // Build runtime CSR on temp1
  int32_t max_val1 = 0;
  for (uint64_t r = 0; r < temp1->row_count; r++) {
    int32_t v = temp1->columns[0].GetInt32(r);
    if (v > max_val1) max_val1 = v;
  }
  auto runtime_csr = BuildCSR(*temp1, 0, max_val1, "temp1", "movie_id", "", "");

  // Step 2: scan cast_info, semi-join with temp1 via runtime CSR
  auto cast_info = MakeIntTable("cast_info", {"id", "movie_id"},
                                 {{50, 51, 52, 53}, {100, 999, 200, 500}});
  SubQueryPlan plan2;
  plan2.scan_table = cast_info.get();
  plan2.scan_table_name = "cast_info";
  plan2.valid = true;

  KernelJoinStep step2;
  step2.csr = &runtime_csr;
  step2.scan_key_col_idx = 1;
  step2.joined_table = temp1.get();
  step2.is_semi = true;
  plan2.join_steps.push_back(step2);

  KernelOutputCol out2{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
  plan2.output_cols = {out2};

  auto temp2 = ExecuteSubQueryPlan(plan2, "temp2");
  // movie_id 100,200,500 are in temp1; 999 is not
  ASSERT_EQ(temp2->row_count, 3u);
  EXPECT_EQ(temp2->columns[0].GetInt32(0), 50);
  EXPECT_EQ(temp2->columns[0].GetInt32(1), 52);
  EXPECT_EQ(temp2->columns[0].GetInt32(2), 53);
}

// Test that inner join result FlatTable has valid string data
// that can be read after the source tables are destroyed
TEST(SubQueryPlan, InnerJoinResultStringsOwned) {
  std::unique_ptr<FlatTable> result;
  {
    auto scan = MakeIntTable("s", {"id"}, {{1, 2}});
    auto lookup = MakeVarcharTable("l", {"key", "val"}, {1, 2},
                                    {"hello_world", "goodbye"});
    auto csr = BuildCSR(*lookup, 0, 2, "l", "key", "", "");

    SubQueryPlan plan;
    plan.scan_table = scan.get();
    plan.scan_table_name = "s";
    plan.valid = true;

    KernelJoinStep step;
    step.csr = &csr;
    step.scan_key_col_idx = 0;
    step.joined_table = lookup.get();
    step.is_semi = false;
    plan.join_steps.push_back(step);

    KernelOutputCol out{KernelOutputCol::FROM_JOIN, 0, 1, FlatColumnType::VARCHAR, "val"};
    plan.output_cols = {out};

    result = ExecuteSubQueryPlan(plan, "result");
    // scan, lookup, csr all destroyed here
  }

  // Result should still have valid strings (FlatTableBuilder copies them)
  ASSERT_EQ(result->row_count, 2u);
  EXPECT_EQ(GetStr(result->columns[0], 0), "hello_world");
  EXPECT_EQ(GetStr(result->columns[0], 1), "goodbye");
}

// ============================================================
// SortedIndex Tests
// ============================================================

TEST(SortedIndex, BuildInt32) {
  auto ft = MakeIntTable("t", {"id", "value"}, {{0, 1, 2, 3, 4}, {50, 10, 40, 20, 30}});
  auto idx = BuildSortedIndex(*ft, "value");
  ASSERT_EQ(idx.sorted_perm.size(), 5u);
  // Sorted order: 10(row1), 20(row3), 30(row4), 40(row2), 50(row0)
  EXPECT_EQ(idx.sorted_perm[0], 1u);
  EXPECT_EQ(idx.sorted_perm[1], 3u);
  EXPECT_EQ(idx.sorted_perm[2], 4u);
  EXPECT_EQ(idx.sorted_perm[3], 2u);
  EXPECT_EQ(idx.sorted_perm[4], 0u);
}

TEST(SortedIndex, BuildVarchar) {
  auto ft = MakeVarcharTable("t", {"id", "name"}, {0, 1, 2, 3},
                              {"cherry", "apple", "banana", "date"});
  auto idx = BuildSortedIndex(*ft, "name");
  ASSERT_EQ(idx.sorted_perm.size(), 4u);
  // Sorted: apple(1), banana(2), cherry(0), date(3)
  EXPECT_EQ(idx.sorted_perm[0], 1u);
  EXPECT_EQ(idx.sorted_perm[1], 2u);
  EXPECT_EQ(idx.sorted_perm[2], 0u);
  EXPECT_EQ(idx.sorted_perm[3], 3u);
}

TEST(SortedIndex, ColumnNotFound) {
  auto ft = MakeIntTable("t", {"a"}, {{1, 2, 3}});
  auto idx = BuildSortedIndex(*ft, "nonexistent");
  EXPECT_TRUE(idx.sorted_perm.empty());
}

TEST(SortedIndex, EmptyTable) {
  auto ft = MakeIntTable("t", {"a"}, {{}});
  ft->row_count = 0;
  ft->columns[0].row_count = 0;
  auto idx = BuildSortedIndex(*ft, "a");
  EXPECT_TRUE(idx.sorted_perm.empty());
}

TEST(SortedIndex, DuplicateValues) {
  auto ft = MakeIntTable("t", {"val"}, {{3, 1, 3, 1, 2}});
  auto idx = BuildSortedIndex(*ft, "val");
  ASSERT_EQ(idx.sorted_perm.size(), 5u);
  // Values in sorted order: 1, 1, 2, 3, 3
  EXPECT_EQ(ft->columns[0].GetInt32(idx.sorted_perm[0]), 1);
  EXPECT_EQ(ft->columns[0].GetInt32(idx.sorted_perm[1]), 1);
  EXPECT_EQ(ft->columns[0].GetInt32(idx.sorted_perm[2]), 2);
  EXPECT_EQ(ft->columns[0].GetInt32(idx.sorted_perm[3]), 3);
  EXPECT_EQ(ft->columns[0].GetInt32(idx.sorted_perm[4]), 3);
}

// ============================================================
// FinalAggregatePlan / ExecuteFinalAggregate Tests
// ============================================================

// 2-table: scan=temp (INT), lookup=base with CSR, running MIN on both sides
TEST(FinalAggregate, TwoTableRunningMin) {
  // temp table (scan): movie_id column
  auto temp = MakeIntTable("temp1", {"movie_id"}, {{3, 1, 5, 2}});
  // base table (lookup): id, title (VARCHAR), year (INT)
  auto base = MakeMixedTable("title", {"id", "title", "production_year"},
                              {FlatColumnType::INT32, FlatColumnType::VARCHAR, FlatColumnType::INT32},
                              {{1, 2, 3, 4, 5}, {2000, 1990, 2010, 1985, 2005}},
                              {{"Movie A", "Movie B", "Movie C", "Movie D", "Movie E"}});
  // CSR on temp.movie_id → indexes into temp rows
  auto csr = BuildCSR(*temp, 0, 5, "temp1", "movie_id", "title", "id");

  FinalAggregatePlan plan;
  plan.scan_table = temp.get();
  plan.scan_table_name = "temp1";
  plan.valid = true;

  // Join step: temp.movie_id → base via CSR
  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 0; // movie_id
  step.joined_table = base.get();
  step.is_semi = true;
  plan.join_steps.push_back(step);

  // Actually this CSR goes the wrong way for this test. Let me build a bitset instead.
  // PK bitset for base table's id column (all IDs valid)
  plan.join_steps.clear();
  KernelJoinStep bstep;
  bstep.csr = nullptr;
  bstep.scan_key_col_idx = 0; // temp.movie_id
  bstep.joined_table = base.get();
  bstep.is_semi = true;
  bstep.use_bitset = true;
  bstep.pk_bitset.resize(6, false);
  bstep.pk_to_row = {0, 0, 1, 2, 3, 4}; // pk_to_row[id] = row in base
  for (int i = 1; i <= 5; i++) bstep.pk_bitset[i] = true;
  plan.join_steps.push_back(bstep);

  // MIN columns: MIN(title.title), MIN(title.production_year)
  MinColumnInfo mc0;
  mc0.output_idx = 0;
  mc0.table = base.get();
  mc0.on_scan_table = false;
  mc0.flat_col_idx = 1; // title
  mc0.type = FlatColumnType::VARCHAR;
  mc0.sorted = nullptr;
  mc0.name = "title";
  plan.min_cols.push_back(mc0);

  MinColumnInfo mc1;
  mc1.output_idx = 1;
  mc1.table = base.get();
  mc1.on_scan_table = false;
  mc1.flat_col_idx = 2; // production_year
  mc1.type = FlatColumnType::INT32;
  mc1.sorted = nullptr;
  mc1.name = "production_year";
  plan.min_cols.push_back(mc1);

  plan.output_names = {"min_title", "min_year"};

  auto result = ExecuteFinalAggregate(plan);
  ASSERT_EQ(result.num_rows, 1);
  ASSERT_EQ(result.num_columns, 2);
  // temp has movie_ids 3,1,5,2 → all exist in base
  // base titles: A(row0/id1), B(row1/id2), C(row2/id3), D(row3/id4), E(row4/id5)
  // Joined: temp movie_id 3→row2(C), 1→row0(A), 5→row4(E), 2→row1(B)
  // MIN(title) = "Movie A", MIN(year) = min(2010,2000,2005,1990) = 1990
  EXPECT_EQ(result.rows[0][0], "Movie A");
  EXPECT_EQ(result.rows[0][1], "1990");
}

// 2-table with sorted index on scan table — early termination
TEST(FinalAggregate, SortedScanEarlyTermination) {
  // base table (scan): id, name (VARCHAR)
  auto base = MakeVarcharTable("name", {"id", "name"}, {0, 1, 2, 3, 4},
                                {"Zebra", "Apple", "Mango", "Banana", "Cherry"});
  // temp table (lookup): person_id
  auto temp = MakeIntTable("temp1", {"person_id"}, {{1, 3}}); // match id=1,3
  auto csr = BuildCSR(*temp, 0, 4, "temp1", "person_id", "name", "id");

  // Build sorted index on base.name
  auto sorted = BuildSortedIndex(*base, "name");
  // Sorted order: Apple(1), Banana(3), Cherry(4), Mango(2), Zebra(0)

  FinalAggregatePlan plan;
  plan.scan_table = base.get();
  plan.scan_table_name = "name";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 0; // id
  step.joined_table = temp.get();
  step.is_semi = true;
  plan.join_steps.push_back(step);

  MinColumnInfo mc;
  mc.output_idx = 0;
  mc.table = base.get();
  mc.on_scan_table = true;
  mc.flat_col_idx = 1; // name
  mc.type = FlatColumnType::VARCHAR;
  mc.sorted = &sorted;
  mc.name = "name";
  plan.min_cols.push_back(mc);

  plan.output_names = {"min_name"};

  auto result = ExecuteFinalAggregate(plan);
  ASSERT_EQ(result.num_rows, 1);
  // temp has person_id=1,3. base ids: 0-4.
  // Qualifying: id=1 (Apple), id=3 (Banana). Others don't match CSR.
  // Sorted scan: first Apple(id=1) → CSR lookup 1 → found → MIN = "Apple"
  EXPECT_EQ(result.rows[0][0], "Apple");
}

// No qualifying rows → "NULL"
TEST(FinalAggregate, NoMatchesReturnNull) {
  auto scan = MakeIntTable("t", {"id"}, {{10, 20, 30}});
  auto lookup = MakeIntTable("t2", {"key"}, {{99}});
  auto csr = BuildCSR(*lookup, 0, 99, "t2", "key", "t", "id");

  FinalAggregatePlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "t";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 0; // id
  step.joined_table = lookup.get();
  step.is_semi = true;
  plan.join_steps.push_back(step);

  MinColumnInfo mc;
  mc.output_idx = 0;
  mc.table = scan.get();
  mc.on_scan_table = true;
  mc.flat_col_idx = 0;
  mc.type = FlatColumnType::INT32;
  mc.sorted = nullptr;
  mc.name = "id";
  plan.min_cols.push_back(mc);

  plan.output_names = {"min_id"};

  auto result = ExecuteFinalAggregate(plan);
  ASSERT_EQ(result.num_rows, 1);
  // scan ids 10,20,30 — CSR only has key 99. No matches.
  EXPECT_EQ(result.rows[0][0], "NULL");
}

// Multi-table star join: temp (center) + 2 base tables (arms)
TEST(FinalAggregate, StarJoinThreeTable) {
  // Center = temp (scan): movie_id, person_id
  auto temp = MakeIntTable("temp1", {"movie_id", "person_id"},
                            {{1, 2, 3, 4}, {10, 20, 30, 40}});

  // Arm 1 = title table: id, title
  auto title = MakeVarcharTable("title", {"id", "title"}, {1, 2, 3, 4, 5},
                                 {"TitleA", "TitleB", "TitleC", "TitleD", "TitleE"});
  // Arm 2 = name table: id, name
  auto name_tbl = MakeVarcharTable("name", {"id", "name"}, {10, 20, 30, 40, 50},
                                    {"NameA", "NameB", "NameC", "NameD", "NameE"});

  // PK bitset for title (all IDs 1-5 valid)
  KernelJoinStep step_title;
  step_title.csr = nullptr;
  step_title.scan_key_col_idx = 0; // temp.movie_id
  step_title.joined_table = title.get();
  step_title.is_semi = true;
  step_title.use_bitset = true;
  step_title.pk_bitset.resize(6, false);
  step_title.pk_to_row = {0, 0, 1, 2, 3, 4};
  for (int i = 1; i <= 5; i++) step_title.pk_bitset[i] = true;

  // PK bitset for name (IDs 10-50 valid)
  KernelJoinStep step_name;
  step_name.csr = nullptr;
  step_name.scan_key_col_idx = 1; // temp.person_id
  step_name.joined_table = name_tbl.get();
  step_name.is_semi = true;
  step_name.use_bitset = true;
  step_name.pk_bitset.resize(51, false);
  step_name.pk_to_row.resize(51, 0);
  step_name.pk_to_row[10] = 0;
  step_name.pk_to_row[20] = 1;
  step_name.pk_to_row[30] = 2;
  step_name.pk_to_row[40] = 3;
  step_name.pk_to_row[50] = 4;
  for (int id : {10, 20, 30, 40, 50}) step_name.pk_bitset[id] = true;

  FinalAggregatePlan plan;
  plan.scan_table = temp.get();
  plan.scan_table_name = "temp1";
  plan.valid = true;
  plan.join_steps.push_back(std::move(step_title));
  plan.join_steps.push_back(std::move(step_name));

  // MIN(title.title), MIN(name.name)
  MinColumnInfo mc0;
  mc0.output_idx = 0;
  mc0.table = title.get();
  mc0.on_scan_table = false;
  mc0.flat_col_idx = 1; // title
  mc0.type = FlatColumnType::VARCHAR;
  mc0.sorted = nullptr;
  mc0.name = "title";
  plan.min_cols.push_back(mc0);

  MinColumnInfo mc1;
  mc1.output_idx = 1;
  mc1.table = name_tbl.get();
  mc1.on_scan_table = false;
  mc1.flat_col_idx = 1; // name
  mc1.type = FlatColumnType::VARCHAR;
  mc1.sorted = nullptr;
  mc1.name = "name";
  plan.min_cols.push_back(mc1);

  plan.output_names = {"min_title", "min_name"};

  auto result = ExecuteFinalAggregate(plan);
  ASSERT_EQ(result.num_rows, 1);
  ASSERT_EQ(result.num_columns, 2);
  // All 4 temp rows qualify (all movie_ids and person_ids exist in bitsets)
  // title lookup: id1→TitleA, id2→TitleB, id3→TitleC, id4→TitleD
  // name lookup: id10→NameA, id20→NameB, id30→NameC, id40→NameD
  // MIN(title) = "TitleA", MIN(name) = "NameA"
  EXPECT_EQ(result.rows[0][0], "TitleA");
  EXPECT_EQ(result.rows[0][1], "NameA");
}

// Star join with scan filters
TEST(FinalAggregate, ScanFiltersApplied) {
  auto scan = MakeIntTable("t", {"id", "fk"}, {{1, 2, 3, 4, 5}, {10, 20, 30, 40, 50}});
  auto lookup = MakeIntTable("l", {"key"}, {{10, 20, 30, 40, 50}});
  auto csr = BuildCSR(*lookup, 0, 50, "l", "key", "t", "fk");

  FinalAggregatePlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "t";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 1; // fk
  step.joined_table = lookup.get();
  step.is_semi = true;
  plan.join_steps.push_back(step);

  // Filter: id > 2 (only rows 3,4,5 qualify)
  plan.scan_filters.push_back([](const FlatTable &t, uint64_t row) {
    return t.columns[0].GetInt32(row) > 2;
  });

  MinColumnInfo mc;
  mc.output_idx = 0;
  mc.table = scan.get();
  mc.on_scan_table = true;
  mc.flat_col_idx = 0; // id
  mc.type = FlatColumnType::INT32;
  mc.sorted = nullptr;
  mc.name = "id";
  plan.min_cols.push_back(mc);

  plan.output_names = {"min_id"};

  auto result = ExecuteFinalAggregate(plan);
  ASSERT_EQ(result.num_rows, 1);
  EXPECT_EQ(result.rows[0][0], "3");
}

// Mixed: some MIN cols with sorted index, others running-min
TEST(FinalAggregate, MixedSortedAndRunningMin) {
  // base (scan): id, name(VARCHAR), year(INT)
  auto base = MakeMixedTable("title", {"id", "name", "year"},
                              {FlatColumnType::INT32, FlatColumnType::VARCHAR, FlatColumnType::INT32},
                              {{0, 1, 2, 3, 4}, {2010, 2005, 2020, 1999, 2015}},
                              {{"Zebra", "Alpha", "Mega", "Beta", "Cherry"}});
  // temp (lookup): tid
  auto temp = MakeIntTable("temp", {"tid"}, {{0, 2, 4}}); // match id=0,2,4
  auto csr = BuildCSR(*temp, 0, 4, "temp", "tid", "title", "id");

  // Sorted index on name
  auto sorted_name = BuildSortedIndex(*base, "name");

  FinalAggregatePlan plan;
  plan.scan_table = base.get();
  plan.scan_table_name = "title";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 0; // id
  step.joined_table = temp.get();
  step.is_semi = true;
  plan.join_steps.push_back(step);

  // MIN(name) with sorted index, MIN(year) without
  MinColumnInfo mc0;
  mc0.output_idx = 0;
  mc0.table = base.get();
  mc0.on_scan_table = true;
  mc0.flat_col_idx = 1; // name
  mc0.type = FlatColumnType::VARCHAR;
  mc0.sorted = &sorted_name;
  mc0.name = "name";
  plan.min_cols.push_back(mc0);

  MinColumnInfo mc1;
  mc1.output_idx = 1;
  mc1.table = base.get();
  mc1.on_scan_table = true;
  mc1.flat_col_idx = 2; // year
  mc1.type = FlatColumnType::INT32;
  mc1.sorted = nullptr;
  mc1.name = "year";
  plan.min_cols.push_back(mc1);

  plan.output_names = {"min_name", "min_year"};

  auto result = ExecuteFinalAggregate(plan);
  ASSERT_EQ(result.num_rows, 1);
  // Qualifying rows: id=0(Zebra,2010), id=2(Mega,2020), id=4(Cherry,2015)
  // Sorted name order: Alpha(1), Beta(3), Cherry(4), Mega(2), Zebra(0)
  // First qualifying in sorted order: Cherry(id=4) → CSR has 4 → yes → MIN(name) = "Cherry"
  // Running MIN(year): min(2010, 2020, 2015) = 2010
  EXPECT_EQ(result.rows[0][0], "Cherry");
  EXPECT_EQ(result.rows[0][1], "2010");
}

// ============================================================
// Scan filters on SubQueryPlan
// ============================================================

TEST(SubQueryPlan, ScanFilterApplied) {
  auto scan = MakeIntTable("t", {"id", "fk", "value"},
                            {{0, 1, 2, 3, 4}, {10, 20, 30, 40, 50}, {100, 200, 300, 400, 500}});
  auto lookup = MakeIntTable("l", {"key"}, {{10, 20, 30, 40, 50}});
  auto csr = BuildCSR(*lookup, 0, 50, "l", "key", "", "");

  SubQueryPlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "t";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 1;
  step.joined_table = lookup.get();
  step.is_semi = true;
  plan.join_steps.push_back(step);

  // Filter: value >= 300 → rows 2,3,4
  plan.scan_filters.push_back([](const FlatTable &t, uint64_t row) {
    return t.columns[2].GetInt32(row) >= 300;
  });

  KernelOutputCol out{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
  plan.output_cols = {out};

  auto result = ExecuteSubQueryPlan(plan, "result");
  ASSERT_EQ(result->row_count, 3u);
  EXPECT_EQ(result->columns[0].GetInt32(0), 2);
  EXPECT_EQ(result->columns[0].GetInt32(1), 3);
  EXPECT_EQ(result->columns[0].GetInt32(2), 4);
}

// ============================================================
// Join filters on KernelJoinStep (inner join)
// ============================================================

TEST(SubQueryPlan, JoinFilterApplied) {
  // scan: id, fk
  auto scan = MakeIntTable("s", {"id", "fk"}, {{0, 1, 2}, {10, 20, 30}});
  // lookup: key, type_id — join filter will check type_id == 5
  auto lookup = MakeIntTable("l", {"key", "type_id"},
                              {{10, 10, 20, 30}, {5, 99, 5, 5}});
  // CSR on lookup.key
  auto csr = BuildCSR(*lookup, 0, 30, "l", "key", "", "");

  SubQueryPlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "s";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 1; // fk
  step.joined_table = lookup.get();
  step.is_semi = false;
  // Join filter: type_id == 5
  step.join_filters.push_back([](const FlatTable &t, uint64_t row) {
    return t.columns[1].GetInt32(row) == 5;
  });
  plan.join_steps.push_back(step);

  KernelOutputCol out0{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
  KernelOutputCol out1{KernelOutputCol::FROM_JOIN, 0, 1, FlatColumnType::INT32, "type_id"};
  plan.output_cols = {out0, out1};

  auto result = ExecuteSubQueryPlan(plan, "result");
  // fk=10: lookup rows 0(type=5 pass), 1(type=99 fail) → 1 match
  // fk=20: lookup row 2(type=5 pass) → 1 match
  // fk=30: lookup row 3(type=5 pass) → 1 match
  ASSERT_EQ(result->row_count, 3u);
  EXPECT_EQ(result->columns[0].GetInt32(0), 0);
  EXPECT_EQ(result->columns[1].GetInt32(0), 5);
  EXPECT_EQ(result->columns[0].GetInt32(1), 1);
  EXPECT_EQ(result->columns[0].GetInt32(2), 2);
}

// ============================================================
// Inner join via bitset (pk_to_row) with output from lookup
// ============================================================

TEST(SubQueryPlan, BitsetInnerJoin) {
  auto scan = MakeIntTable("s", {"id"}, {{2, 5, 8, 1}});

  auto lookup = MakeVarcharTable("l", {"id", "name"}, {1, 2, 5, 8, 10},
                                  {"Alice", "Bob", "Charlie", "Diana", "Eve"});

  SubQueryPlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "s";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = nullptr;
  step.scan_key_col_idx = 0; // id
  step.joined_table = lookup.get();
  step.is_semi = false;
  step.use_bitset = true;
  step.pk_bitset.resize(11, false);
  step.pk_to_row.resize(11, 0);
  // Set up: id→row mapping
  step.pk_bitset[1] = true;  step.pk_to_row[1] = 0;
  step.pk_bitset[2] = true;  step.pk_to_row[2] = 1;
  step.pk_bitset[5] = true;  step.pk_to_row[5] = 2;
  step.pk_bitset[8] = true;  step.pk_to_row[8] = 3;
  step.pk_bitset[10] = true; step.pk_to_row[10] = 4;
  plan.join_steps.push_back(step);

  KernelOutputCol out0{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
  KernelOutputCol out1{KernelOutputCol::FROM_JOIN, 0, 1, FlatColumnType::VARCHAR, "name"};
  plan.output_cols = {out0, out1};

  auto result = ExecuteSubQueryPlan(plan, "result");
  // scan ids: 2,5,8,1 → all exist in bitset
  ASSERT_EQ(result->row_count, 4u);
  EXPECT_EQ(result->columns[0].GetInt32(0), 2);
  EXPECT_EQ(GetStr(result->columns[1], 0), "Bob");
  EXPECT_EQ(result->columns[0].GetInt32(1), 5);
  EXPECT_EQ(GetStr(result->columns[1], 1), "Charlie");
  EXPECT_EQ(result->columns[0].GetInt32(2), 8);
  EXPECT_EQ(GetStr(result->columns[1], 2), "Diana");
  EXPECT_EQ(result->columns[0].GetInt32(3), 1);
  EXPECT_EQ(GetStr(result->columns[1], 3), "Alice");
}

// ============================================================
// FinalAggregate with CSR-based lookup (running min on lookup via CSR)
// ============================================================

TEST(FinalAggregate, RunningMinViaCSRLookup) {
  // scan=temp, lookup=base joined via CSR
  auto temp = MakeIntTable("temp", {"movie_id"}, {{1, 3, 5}});

  // base: id, note (VARCHAR) — CSR built on base.id
  auto base = MakeVarcharTable("base", {"id", "note"}, {1, 2, 3, 4, 5},
                                {"Z-note", "B-note", "A-note", "C-note", "M-note"});
  // CSR: key=PK value → rows in base (for existence check on base.id)
  // We need lookup(base) by temp.movie_id, so CSR on base.id
  auto csr = BuildCSR(*base, 0, 5, "base", "id", "", "");

  FinalAggregatePlan plan;
  plan.scan_table = temp.get();
  plan.scan_table_name = "temp";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 0; // temp.movie_id
  step.joined_table = base.get();
  step.is_semi = true;
  plan.join_steps.push_back(step);

  // MIN(base.note) — on lookup table, via CSR
  MinColumnInfo mc;
  mc.output_idx = 0;
  mc.table = base.get();
  mc.on_scan_table = false;
  mc.flat_col_idx = 1; // note
  mc.type = FlatColumnType::VARCHAR;
  mc.sorted = nullptr;
  mc.name = "note";
  plan.min_cols.push_back(mc);

  plan.output_names = {"min_note"};

  auto result = ExecuteFinalAggregate(plan);
  ASSERT_EQ(result.num_rows, 1);
  // temp movie_ids: 1,3,5. CSR lookup: 1→row0(Z-note), 3→row2(A-note), 5→row4(M-note)
  // MIN = "A-note"
  EXPECT_EQ(result.rows[0][0], "A-note");
}

// FinalAggregate with join_filters on lookup step
TEST(FinalAggregate, JoinFiltersOnLookup) {
  auto temp = MakeIntTable("temp", {"fk"}, {{1, 2, 3, 4}});

  // base: key, type_id, value — join filter on type_id
  auto base = MakeIntTable("base", {"key", "type_id", "value"},
                            {{1, 1, 2, 2, 3, 3, 4, 4},
                             {5, 99, 5, 5, 99, 99, 5, 99},
                             {100, 200, 50, 60, 30, 40, 80, 90}});
  auto csr = BuildCSR(*base, 0, 4, "base", "key", "", "");

  FinalAggregatePlan plan;
  plan.scan_table = temp.get();
  plan.scan_table_name = "temp";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 0; // temp.fk
  step.joined_table = base.get();
  step.is_semi = true;
  // Join filter: type_id == 5
  step.join_filters.push_back([](const FlatTable &t, uint64_t row) {
    return t.columns[1].GetInt32(row) == 5;
  });
  plan.join_steps.push_back(step);

  // MIN(base.value) — on lookup side, filtered by join_filter
  MinColumnInfo mc;
  mc.output_idx = 0;
  mc.table = base.get();
  mc.on_scan_table = false;
  mc.flat_col_idx = 2; // value
  mc.type = FlatColumnType::INT32;
  mc.sorted = nullptr;
  mc.name = "value";
  plan.min_cols.push_back(mc);

  plan.output_names = {"min_value"};

  auto result = ExecuteFinalAggregate(plan);
  ASSERT_EQ(result.num_rows, 1);
  // fk=1: rows {0(type=5,val=100), 1(type=99,fail)} → val=100
  // fk=2: rows {2(type=5,val=50), 3(type=5,val=60)} → val=50
  // fk=3: rows {4(type=99,fail), 5(type=99,fail)} → no match, but row_qualifies
  //        checks if ANY pass join_filters → none pass → row doesn't qualify
  // fk=4: rows {6(type=5,val=80), 7(type=99,fail)} → val=80
  // MIN of matching values: min(100, 50, 60, 80) = 50
  EXPECT_EQ(result.rows[0][0], "50");
}

// ============================================================
// FlatColumn NULL bitmap
// ============================================================

TEST(FlatColumn, NullBitmap) {
  auto ft = MakeIntTable("t", {"a"}, {{10, 20, 30, 40, 50}});
  auto &col = ft->columns[0];

  // Make column nullable
  col.nullable = true;
  col.null_bitmap.reset(new uint64_t[(5 + 63) / 64]());
  // Set all valid first
  for (uint64_t r = 0; r < 5; r++) col.SetValid(r);

  EXPECT_FALSE(col.IsNull(0));
  EXPECT_FALSE(col.IsNull(4));

  // Set row 2 to NULL
  col.SetNull(2);
  EXPECT_FALSE(col.IsNull(0));
  EXPECT_FALSE(col.IsNull(1));
  EXPECT_TRUE(col.IsNull(2));
  EXPECT_FALSE(col.IsNull(3));
  EXPECT_FALSE(col.IsNull(4));

  // Non-nullable column always returns false
  auto ft2 = MakeIntTable("t2", {"b"}, {{1}});
  EXPECT_FALSE(ft2->columns[0].IsNull(0));
}

// ============================================================
// Memory Ownership Tests
// ============================================================

TEST(Memory, CSRIndependentOfFlatTable) {
  CSRIndex csr;
  {
    auto temp = MakeIntTable("temp", {"id"}, {{1, 3, 1, 5}});
    csr = BuildCSR(*temp, 0, 5, "temp", "id", "", "");
  }
  // FlatTable destroyed; CSR should still work
  auto [b1, e1] = csr.Lookup(1);
  EXPECT_EQ(e1 - b1, 2);
  auto [b3, e3] = csr.Lookup(3);
  EXPECT_EQ(e3 - b3, 1);
  auto [b5, e5] = csr.Lookup(5);
  EXPECT_EQ(e5 - b5, 1);
}

// Simulate the repeat loop in aqp_middleware.cpp:
// kernel_temps_ cleared, then rebuilt; verify no use-after-free
TEST(Memory, RepeatLoopSimulation) {
  for (int repeat = 0; repeat < 10; repeat++) {
    std::unordered_map<std::string, std::unique_ptr<FlatTable>> kernel_temps;
    std::unordered_map<std::string, const FlatTable *> kernel_temp_ptrs;
    std::unordered_map<std::string, const FlatTable *> duckdb_kernel_map;
    std::unordered_map<std::string, CSRIndex> runtime_csrs;

    // Iteration 1: create temp from kernel
    auto scan = MakeIntTable("base", {"id", "fk"}, {{10, 20, 30}, {1, 2, 1}});
    auto prev = MakeIntTable("prev", {"val"}, {{1, 2}});
    auto csr = BuildCSR(*prev, 0, 2, "prev", "val", "", "");

    SubQueryPlan plan;
    plan.scan_table = scan.get();
    plan.scan_table_name = "base";
    plan.valid = true;
    KernelJoinStep step;
    step.csr = &csr;
    step.scan_key_col_idx = 1;
    step.joined_table = prev.get();
    step.is_semi = true;
    plan.join_steps.push_back(step);
    KernelOutputCol out{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
    plan.output_cols = {out};

    auto result = ExecuteSubQueryPlan(plan, "temp1");
    ASSERT_EQ(result->row_count, 3u);

    // Register raw pointers (simulating DuckDB adapter + splitter)
    duckdb_kernel_map["temp1"] = result.get();
    kernel_temp_ptrs["temp1"] = result.get();

    // Build runtime CSR
    int32_t max_v = 0;
    for (uint64_t r = 0; r < result->row_count; r++) {
      int32_t v = result->columns[0].GetInt32(r);
      if (v > max_v) max_v = v;
    }
    runtime_csrs["temp1.id"] = BuildCSR(*result, 0, max_v, "temp1", "id", "", "");

    kernel_temps["temp1"] = std::move(result);

    // End of repeat: clear everything (the fix)
    kernel_temps.clear();
    kernel_temp_ptrs.clear();
    duckdb_kernel_map.clear(); // THIS is the fix: must clear DuckDB's map too
    runtime_csrs.clear();
    // After clear, all raw pointers are invalid — but we've cleared the maps
    // so nobody will dereference them
  }
}

// Test that clearing kernel_temps before duckdb_kernel_map
// leaves dangling pointers (the bug pattern)
TEST(Memory, DanglingPointerBugPattern) {
  std::unordered_map<std::string, std::unique_ptr<FlatTable>> kernel_temps;
  std::unordered_map<std::string, const FlatTable *> duckdb_kernel_map;

  auto ft = MakeIntTable("t", {"a"}, {{42}});
  duckdb_kernel_map["t"] = ft.get();
  kernel_temps["t"] = std::move(ft);

  // Bug: clear kernel_temps but NOT duckdb_kernel_map
  kernel_temps.clear();

  // duckdb_kernel_map["t"] now points to freed memory!
  // This is the bug pattern that was in ResetQueryState.
  // Just verify the map still has the entry (the pointer is dangling)
  EXPECT_TRUE(duckdb_kernel_map.find("t") != duckdb_kernel_map.end());

  // Fix: clear duckdb_kernel_map too
  duckdb_kernel_map.clear();
}

// ============================================================
// Stress / Edge Case Tests
// ============================================================

// Inner join with high fan-out: each scan row matches many lookup rows
TEST(SubQueryPlan, InnerJoinHighFanout) {
  const int SCAN_ROWS = 100;
  const int FANOUT = 50; // each scan key matches 50 lookup rows

  std::vector<int32_t> scan_ids(SCAN_ROWS);
  for (int i = 0; i < SCAN_ROWS; i++) scan_ids[i] = i;
  auto scan = MakeIntTable("s", {"id"}, {scan_ids});

  // Lookup: FANOUT rows per key, all keys = 0..SCAN_ROWS-1
  std::vector<int32_t> lookup_keys(SCAN_ROWS * FANOUT);
  for (int i = 0; i < SCAN_ROWS * FANOUT; i++)
    lookup_keys[i] = i / FANOUT;
  auto lookup = MakeIntTable("l", {"key"}, {lookup_keys});

  auto csr = BuildCSR(*lookup, 0, SCAN_ROWS - 1, "l", "key", "", "");

  SubQueryPlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "s";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = &csr;
  step.scan_key_col_idx = 0;
  step.joined_table = lookup.get();
  step.is_semi = false;
  plan.join_steps.push_back(step);

  KernelOutputCol out{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
  plan.output_cols = {out};

  auto result = ExecuteSubQueryPlan(plan, "result");
  EXPECT_EQ(result->row_count, (uint64_t)(SCAN_ROWS * FANOUT));
}

// Inner join with VARCHAR from both sides, repeated to stress memory
TEST(SubQueryPlan, InnerJoinVarcharStress) {
  for (int iter = 0; iter < 20; iter++) {
    auto scan = MakeVarcharTable("s", {"id", "sname"}, {0, 1, 2, 3, 4},
                                  {"alpha", "beta", "gamma", "delta", "epsilon"});
    auto lookup = MakeVarcharTable("l", {"key", "lname"}, {0, 0, 1, 2, 2, 3, 4, 4, 4},
                                    {"A1", "A2", "B1", "C1", "C2", "D1", "E1", "E2", "E3"});
    auto csr = BuildCSR(*lookup, 0, 4, "l", "key", "", "");

    SubQueryPlan plan;
    plan.scan_table = scan.get();
    plan.scan_table_name = "s";
    plan.valid = true;

    KernelJoinStep step;
    step.csr = &csr;
    step.scan_key_col_idx = 0;
    step.joined_table = lookup.get();
    step.is_semi = false;
    plan.join_steps.push_back(step);

    KernelOutputCol out0{KernelOutputCol::FROM_SCAN, -1, 1, FlatColumnType::VARCHAR, "sname"};
    KernelOutputCol out1{KernelOutputCol::FROM_JOIN, 0, 1, FlatColumnType::VARCHAR, "lname"};
    plan.output_cols = {out0, out1};

    auto result = ExecuteSubQueryPlan(plan, "result");
    // 0:2 + 1:1 + 2:2 + 3:1 + 4:3 = 9 matches
    ASSERT_EQ(result->row_count, 9u);
    EXPECT_EQ(GetStr(result->columns[0], 0), "alpha");
    EXPECT_EQ(GetStr(result->columns[1], 0), "A1");
    EXPECT_EQ(GetStr(result->columns[0], 8), "epsilon");
    EXPECT_EQ(GetStr(result->columns[1], 8), "E3");
  }
}

// Multiple join steps (semi), simulating 2+ CSR joins
TEST(SubQueryPlan, MultipleSemiJoinSteps) {
  auto scan = MakeIntTable("ci", {"id", "movie_id", "person_id"},
                            {{0, 1, 2, 3, 4},
                             {10, 20, 30, 40, 50},
                             {100, 200, 300, 400, 500}});
  auto temp1 = MakeIntTable("t1", {"mid"}, {{10, 30, 50}});
  auto temp2 = MakeIntTable("t2", {"pid"}, {{100, 300}});
  auto csr1 = BuildCSR(*temp1, 0, 50, "t1", "mid", "", "");
  auto csr2 = BuildCSR(*temp2, 0, 500, "t2", "pid", "", "");

  SubQueryPlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "ci";
  plan.valid = true;

  KernelJoinStep step1;
  step1.csr = &csr1;
  step1.scan_key_col_idx = 1; // movie_id
  step1.joined_table = temp1.get();
  step1.is_semi = true;
  plan.join_steps.push_back(step1);

  KernelJoinStep step2;
  step2.csr = &csr2;
  step2.scan_key_col_idx = 2; // person_id
  step2.joined_table = temp2.get();
  step2.is_semi = true;
  plan.join_steps.push_back(step2);

  KernelOutputCol out{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
  plan.output_cols = {out};

  auto result = ExecuteSubQueryPlan(plan, "result");
  // row 0: movie=10(yes), person=100(yes) -> pass
  // row 1: movie=20(no) -> fail
  // row 2: movie=30(yes), person=300(yes) -> pass
  // row 3: movie=40(no) -> fail
  // row 4: movie=50(yes), person=500(no) -> fail
  ASSERT_EQ(result->row_count, 2u);
  EXPECT_EQ(result->columns[0].GetInt32(0), 0);
  EXPECT_EQ(result->columns[0].GetInt32(1), 2);
}

// Bitset-based semi join
TEST(SubQueryPlan, BitsetSemiJoin) {
  auto scan = MakeIntTable("mk", {"id", "keyword_id"},
                            {{0, 1, 2, 3}, {5, 10, 15, 5}});
  SubQueryPlan plan;
  plan.scan_table = scan.get();
  plan.scan_table_name = "mk";
  plan.valid = true;

  KernelJoinStep step;
  step.csr = nullptr;
  step.scan_key_col_idx = 1;
  step.joined_table = nullptr;
  step.is_semi = true;
  step.use_bitset = true;
  step.pk_bitset.resize(16, false);
  step.pk_bitset[5] = true;
  step.pk_bitset[15] = true;
  plan.join_steps.push_back(step);

  KernelOutputCol out{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
  plan.output_cols = {out};

  auto result = ExecuteSubQueryPlan(plan, "result");
  ASSERT_EQ(result->row_count, 3u);
  EXPECT_EQ(result->columns[0].GetInt32(0), 0);  // keyword_id=5
  EXPECT_EQ(result->columns[0].GetInt32(1), 2);  // keyword_id=15
  EXPECT_EQ(result->columns[0].GetInt32(2), 3);  // keyword_id=5
}

// ============================================================
// Full repeat simulation (mimics benchmark measurement loop)
// ============================================================

TEST(Integration, FullRepeatCycle) {
  // Simulates what happens in the benchmark: multiple repeats of the same
  // query, with kernel temps created and destroyed each time.
  for (int repeat = 0; repeat < 5; repeat++) {
    std::unordered_map<std::string, std::unique_ptr<FlatTable>> kernel_temps;
    std::unordered_map<std::string, const FlatTable *> kernel_temp_ptrs;
    std::unordered_map<std::string, CSRIndex> runtime_csrs;

    // Simulate 3 kernel iterations
    auto base1 = MakeIntTable("t1", {"id", "fk"}, {{1,2,3,4,5}, {10,20,30,40,50}});
    auto prev1 = MakeIntTable("p1", {"val"}, {{10,30,50}});
    auto csr1 = BuildCSR(*prev1, 0, 50, "p1", "val", "", "");

    SubQueryPlan plan1;
    plan1.scan_table = base1.get();
    plan1.scan_table_name = "t1";
    plan1.valid = true;
    KernelJoinStep s1;
    s1.csr = &csr1; s1.scan_key_col_idx = 1;
    s1.joined_table = prev1.get(); s1.is_semi = true;
    plan1.join_steps.push_back(s1);
    KernelOutputCol o1{KernelOutputCol::FROM_SCAN, -1, 0, FlatColumnType::INT32, "id"};
    plan1.output_cols = {o1};

    auto r1 = ExecuteSubQueryPlan(plan1, "temp1");
    ASSERT_EQ(r1->row_count, 3u);

    int32_t max1 = 0;
    for (uint64_t r = 0; r < r1->row_count; r++) {
      int32_t v = r1->columns[0].GetInt32(r);
      if (v > max1) max1 = v;
    }
    runtime_csrs["temp1.id"] = BuildCSR(*r1, 0, max1, "temp1", "id", "", "");
    kernel_temp_ptrs["temp1"] = r1.get();
    kernel_temps["temp1"] = std::move(r1);

    // Iteration 2: use runtime CSR from temp1
    auto base2 = MakeVarcharTable("names", {"id", "name"}, {1,2,3,4,5},
                                   {"Alice","Bob","Charlie","Diana","Eve"});
    SubQueryPlan plan2;
    plan2.scan_table = base2.get();
    plan2.scan_table_name = "names";
    plan2.valid = true;
    KernelJoinStep s2;
    s2.csr = &runtime_csrs["temp1.id"];
    s2.scan_key_col_idx = 0;
    s2.joined_table = kernel_temp_ptrs["temp1"];
    s2.is_semi = false;
    plan2.join_steps.push_back(s2);
    KernelOutputCol o2_0{KernelOutputCol::FROM_SCAN, -1, 1, FlatColumnType::VARCHAR, "name"};
    plan2.output_cols = {o2_0};

    auto r2 = ExecuteSubQueryPlan(plan2, "temp2");
    ASSERT_EQ(r2->row_count, 3u);
    EXPECT_EQ(GetStr(r2->columns[0], 0), "Alice");
    EXPECT_EQ(GetStr(r2->columns[0], 1), "Charlie");
    EXPECT_EQ(GetStr(r2->columns[0], 2), "Eve");

    kernel_temp_ptrs["temp2"] = r2.get();
    kernel_temps["temp2"] = std::move(r2);

    // End of repeat: clear all state
    kernel_temps.clear();
    kernel_temp_ptrs.clear();
    runtime_csrs.clear();
  }
}

// ============================================================
// Pipeline Kernel Tests
// ============================================================

TEST(PipelineKernel, SemiJoinTwoTables) {
  // scan: orders(id, customer_id, amount) — 6 rows
  // build: customers(id, name) — 3 rows
  // Join: orders.customer_id = customers.id (semi)
  // Output: orders.id, orders.amount
  auto orders = MakeIntTable("orders", {"id", "customer_id", "amount"},
    {{1,2,3,4,5,6}, {10,20,10,30,20,40}, {100,200,300,400,500,600}});
  auto customers = MakeIntTable("customers", {"id", "flag"},
    {{10,20,30}, {1,1,1}});

  PipelineKernelPlan plan;
  plan.scan_table = orders.get();
  plan.scan_table_name = "orders";

  PipelineJoinStep step;
  step.build_table = customers.get();
  step.build_key_col = 0; // customers.id
  step.scan_key_col = 1;  // orders.customer_id
  step.probe_step_idx = -1;
  step.is_semi = true;
  plan.join_steps.push_back(std::move(step));

  KernelOutputCol out0;
  out0.source = KernelOutputCol::FROM_SCAN;
  out0.col_idx = 0; // orders.id
  out0.type = FlatColumnType::INT32;
  out0.name = "id";

  KernelOutputCol out1;
  out1.source = KernelOutputCol::FROM_SCAN;
  out1.col_idx = 2; // orders.amount
  out1.type = FlatColumnType::INT32;
  out1.name = "amount";

  plan.output_cols = {out0, out1};
  plan.valid = true;

  auto result = ExecutePipelineKernel(plan, "result");
  ASSERT_NE(result, nullptr);
  // customer_ids 10,20,30 exist; 40 doesn't
  // rows 1(10),2(20),3(10),4(30),5(20) pass; row 6(40) fails
  EXPECT_EQ(result->row_count, 5u);
  EXPECT_EQ(result->columns.size(), 2u);

  // Collect output ids (order may vary with OpenMP)
  std::set<int32_t> ids;
  for (uint64_t r = 0; r < result->row_count; r++)
    ids.insert(result->columns[0].GetInt32(r));
  EXPECT_EQ(ids, (std::set<int32_t>{1,2,3,4,5}));
}

TEST(PipelineKernel, InnerJoinTwoTables) {
  // scan: fact(id, dim_id) — 4 rows
  // build: dim(id, val) — 3 rows, dim_id=2 has 2 matches
  auto fact = MakeIntTable("fact", {"id", "dim_id"},
    {{1,2,3,4}, {10,20,10,30}});
  auto dim = MakeIntTable("dim", {"id", "val"},
    {{10,20,30}, {100,200,300}});

  PipelineKernelPlan plan;
  plan.scan_table = fact.get();
  plan.scan_table_name = "fact";

  PipelineJoinStep step;
  step.build_table = dim.get();
  step.build_key_col = 0; // dim.id
  step.scan_key_col = 1;  // fact.dim_id
  step.probe_step_idx = -1;
  step.is_semi = false;
  plan.join_steps.push_back(std::move(step));

  KernelOutputCol out0;
  out0.source = KernelOutputCol::FROM_SCAN;
  out0.col_idx = 0; out0.type = FlatColumnType::INT32; out0.name = "id";

  KernelOutputCol out1;
  out1.source = KernelOutputCol::FROM_JOIN;
  out1.step_idx = 0; out1.col_idx = 1;
  out1.type = FlatColumnType::INT32; out1.name = "val";

  plan.output_cols = {out0, out1};
  plan.valid = true;

  auto result = ExecutePipelineKernel(plan, "result");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->row_count, 4u);

  // Collect (fact.id, dim.val) pairs
  std::set<std::pair<int32_t,int32_t>> pairs;
  for (uint64_t r = 0; r < result->row_count; r++)
    pairs.insert({result->columns[0].GetInt32(r), result->columns[1].GetInt32(r)});
  std::set<std::pair<int32_t,int32_t>> expected = {{1,100},{2,200},{3,100},{4,300}};
  EXPECT_EQ(pairs, expected);
}

TEST(PipelineKernel, ScanFilterOnly) {
  auto t = MakeIntTable("t", {"id", "val"}, {{1,2,3,4,5}, {10,20,30,40,50}});

  PipelineKernelPlan plan;
  plan.scan_table = t.get();
  plan.scan_table_name = "t";
  plan.scan_filters.push_back([](const FlatTable &tbl, uint64_t row) {
    return tbl.columns[1].GetInt32(row) >= 30;
  });

  KernelOutputCol out0;
  out0.source = KernelOutputCol::FROM_SCAN;
  out0.col_idx = 0; out0.type = FlatColumnType::INT32; out0.name = "id";
  plan.output_cols = {out0};
  plan.valid = true;

  auto result = ExecutePipelineKernel(plan, "result");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->row_count, 3u);
  std::set<int32_t> ids;
  for (uint64_t r = 0; r < result->row_count; r++)
    ids.insert(result->columns[0].GetInt32(r));
  EXPECT_EQ(ids, (std::set<int32_t>{3,4,5}));
}

TEST(PipelineKernel, EmptyBuildTable) {
  auto fact = MakeIntTable("fact", {"id", "fk"}, {{1,2,3}, {10,20,30}});
  auto empty_dim = MakeIntTable("dim", {"id"}, {{}});

  PipelineKernelPlan plan;
  plan.scan_table = fact.get();
  plan.scan_table_name = "fact";

  PipelineJoinStep step;
  step.build_table = empty_dim.get();
  step.build_key_col = 0;
  step.scan_key_col = 1;
  step.probe_step_idx = -1;
  step.is_semi = true;
  plan.join_steps.push_back(std::move(step));

  KernelOutputCol out0;
  out0.source = KernelOutputCol::FROM_SCAN;
  out0.col_idx = 0; out0.type = FlatColumnType::INT32; out0.name = "id";
  plan.output_cols = {out0};
  plan.valid = true;

  auto result = ExecutePipelineKernel(plan, "result");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->row_count, 0u);
}

TEST(PipelineKernel, ThreeTableStarJoin) {
  // scan: fact(id, d1_id, d2_id) — 5 rows
  // build1: dim1(id) — IDs {1,2}
  // build2: dim2(id) — IDs {10,20}
  // fact.d1_id = dim1.id AND fact.d2_id = dim2.id (both semi)
  auto fact = MakeIntTable("fact", {"id", "d1_id", "d2_id"},
    {{1,2,3,4,5}, {1,2,3,1,2}, {10,20,30,10,20}});
  auto dim1 = MakeIntTable("dim1", {"id"}, {{1,2}});
  auto dim2 = MakeIntTable("dim2", {"id"}, {{10,20}});

  PipelineKernelPlan plan;
  plan.scan_table = fact.get();
  plan.scan_table_name = "fact";

  PipelineJoinStep step1;
  step1.build_table = dim1.get();
  step1.build_key_col = 0;
  step1.scan_key_col = 1; // fact.d1_id
  step1.probe_step_idx = -1;
  step1.is_semi = true;

  PipelineJoinStep step2;
  step2.build_table = dim2.get();
  step2.build_key_col = 0;
  step2.scan_key_col = 2; // fact.d2_id
  step2.probe_step_idx = -1;
  step2.is_semi = true;

  plan.join_steps.push_back(std::move(step1));
  plan.join_steps.push_back(std::move(step2));

  KernelOutputCol out0;
  out0.source = KernelOutputCol::FROM_SCAN;
  out0.col_idx = 0; out0.type = FlatColumnType::INT32; out0.name = "id";
  plan.output_cols = {out0};
  plan.valid = true;

  auto result = ExecutePipelineKernel(plan, "result");
  ASSERT_NE(result, nullptr);
  // rows: 1(d1=1,d2=10)✓, 2(d1=2,d2=20)✓, 3(d1=3,d2=30)✗, 4(d1=1,d2=10)✓, 5(d1=2,d2=20)✓
  EXPECT_EQ(result->row_count, 4u);
  std::set<int32_t> ids;
  for (uint64_t r = 0; r < result->row_count; r++)
    ids.insert(result->columns[0].GetInt32(r));
  EXPECT_EQ(ids, (std::set<int32_t>{1,2,4,5}));
}

TEST(PipelineKernel, ChainJoin) {
  // scan: A(id, b_id) — 4 rows
  // build1: B(id, c_id) — 3 rows, joined to A via A.b_id = B.id
  // build2: C(id) — 2 rows, joined to B via B.c_id = C.id (chain)
  // Output: A.id, B.id (inner on B), semi on C
  auto A = MakeIntTable("A", {"id", "b_id"}, {{1,2,3,4}, {10,20,30,10}});
  auto B = MakeIntTable("B", {"id", "c_id"}, {{10,20,30}, {100,200,300}});
  auto C = MakeIntTable("C", {"id"}, {{100,200}});

  PipelineKernelPlan plan;
  plan.scan_table = A.get();
  plan.scan_table_name = "A";

  // Step 0: A.b_id → B.id (inner, output B columns)
  PipelineJoinStep step0;
  step0.build_table = B.get();
  step0.build_key_col = 0; // B.id
  step0.scan_key_col = 1;  // A.b_id
  step0.probe_step_idx = -1;
  step0.is_semi = false;

  // Step 1: B.c_id → C.id (semi, chain from step 0)
  PipelineJoinStep step1;
  step1.build_table = C.get();
  step1.build_key_col = 0; // C.id
  step1.scan_key_col = -1;
  step1.probe_step_idx = 0;  // probe from step 0's build table (B)
  step1.probe_key_col = 1;   // B.c_id
  step1.is_semi = true;

  plan.join_steps.push_back(std::move(step0));
  plan.join_steps.push_back(std::move(step1));

  KernelOutputCol out0;
  out0.source = KernelOutputCol::FROM_SCAN;
  out0.col_idx = 0; out0.type = FlatColumnType::INT32; out0.name = "a_id";

  KernelOutputCol out1;
  out1.source = KernelOutputCol::FROM_JOIN;
  out1.step_idx = 0; out1.col_idx = 0;
  out1.type = FlatColumnType::INT32; out1.name = "b_id";

  plan.output_cols = {out0, out1};
  plan.valid = true;

  auto result = ExecutePipelineKernel(plan, "result");
  ASSERT_NE(result, nullptr);
  // A row 1: b_id=10 → B(10, c_id=100) → C has 100 ✓
  // A row 2: b_id=20 → B(20, c_id=200) → C has 200 ✓
  // A row 3: b_id=30 → B(30, c_id=300) → C has 300? No ✗
  // A row 4: b_id=10 → B(10, c_id=100) → C has 100 ✓
  EXPECT_EQ(result->row_count, 3u);

  std::set<std::pair<int32_t,int32_t>> pairs;
  for (uint64_t r = 0; r < result->row_count; r++)
    pairs.insert({result->columns[0].GetInt32(r), result->columns[1].GetInt32(r)});
  std::set<std::pair<int32_t,int32_t>> expected = {{1,10},{2,20},{4,10}};
  EXPECT_EQ(pairs, expected);
}

TEST(PipelineKernel, BuildFilterSkipsRows) {
  auto fact = MakeIntTable("fact", {"id", "fk"}, {{1,2,3}, {10,20,30}});
  auto dim = MakeIntTable("dim", {"id", "active"},
    {{10,20,30}, {1,0,1}});

  PipelineKernelPlan plan;
  plan.scan_table = fact.get();
  plan.scan_table_name = "fact";

  PipelineJoinStep step;
  step.build_table = dim.get();
  step.build_key_col = 0;
  step.scan_key_col = 1;
  step.probe_step_idx = -1;
  step.is_semi = true;
  // Build filter: only active=1 rows in dim
  step.build_filters.push_back([](const FlatTable &t, uint64_t row) {
    return t.columns[1].GetInt32(row) == 1;
  });
  plan.join_steps.push_back(std::move(step));

  KernelOutputCol out0;
  out0.source = KernelOutputCol::FROM_SCAN;
  out0.col_idx = 0; out0.type = FlatColumnType::INT32; out0.name = "id";
  plan.output_cols = {out0};
  plan.valid = true;

  auto result = ExecutePipelineKernel(plan, "result");
  ASSERT_NE(result, nullptr);
  // dim with active=1: ids {10, 30}. fact rows with fk in {10,30}: rows 1,3
  EXPECT_EQ(result->row_count, 2u);
  std::set<int32_t> ids;
  for (uint64_t r = 0; r < result->row_count; r++)
    ids.insert(result->columns[0].GetInt32(r));
  EXPECT_EQ(ids, (std::set<int32_t>{1,3}));
}

TEST(PipelineKernel, DuplicateKeysInBuild) {
  // Build table has duplicate keys — inner join should produce multiple rows
  auto fact = MakeIntTable("fact", {"id", "fk"}, {{1,2}, {10,10}});
  auto dim = MakeIntTable("dim", {"fk", "val"},
    {{10,10,10}, {100,200,300}});

  PipelineKernelPlan plan;
  plan.scan_table = fact.get();
  plan.scan_table_name = "fact";

  PipelineJoinStep step;
  step.build_table = dim.get();
  step.build_key_col = 0;
  step.scan_key_col = 1;
  step.probe_step_idx = -1;
  step.is_semi = false;
  plan.join_steps.push_back(std::move(step));

  KernelOutputCol out0;
  out0.source = KernelOutputCol::FROM_SCAN;
  out0.col_idx = 0; out0.type = FlatColumnType::INT32; out0.name = "id";

  KernelOutputCol out1;
  out1.source = KernelOutputCol::FROM_JOIN;
  out1.step_idx = 0; out1.col_idx = 1;
  out1.type = FlatColumnType::INT32; out1.name = "val";

  plan.output_cols = {out0, out1};
  plan.valid = true;

  auto result = ExecutePipelineKernel(plan, "result");
  ASSERT_NE(result, nullptr);
  // 2 fact rows × 3 matching dim rows each = 6 output rows
  EXPECT_EQ(result->row_count, 6u);

  std::multiset<std::pair<int32_t,int32_t>> pairs;
  for (uint64_t r = 0; r < result->row_count; r++)
    pairs.insert({result->columns[0].GetInt32(r), result->columns[1].GetInt32(r)});
  // Each fact row (1 and 2) should match all 3 dim rows
  EXPECT_EQ(pairs.count({1,100}), 1u);
  EXPECT_EQ(pairs.count({1,200}), 1u);
  EXPECT_EQ(pairs.count({1,300}), 1u);
  EXPECT_EQ(pairs.count({2,100}), 1u);
  EXPECT_EQ(pairs.count({2,200}), 1u);
  EXPECT_EQ(pairs.count({2,300}), 1u);
}
