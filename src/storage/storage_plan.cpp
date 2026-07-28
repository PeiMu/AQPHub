#include "storage/storage_plan.h"

#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/materialized_query_result.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>

namespace middleware {
namespace storage {

static FlatColumn LoadColumnINT32(duckdb::MaterializedQueryResult &result,
                                  size_t col_idx, uint64_t total_rows,
                                  bool nullable) {
  FlatColumn col;
  col.type = FlatColumnType::INT32;
  col.row_count = total_rows;
  col.nullable = nullable;
  col.data = std::make_unique<char[]>(total_rows * sizeof(int32_t));

  if (nullable) {
    uint64_t bitmap_words = (total_rows + 63) / 64;
    col.null_bitmap = std::make_unique<uint64_t[]>(bitmap_words);
    std::memset(col.null_bitmap.get(), 0xFF,
                bitmap_words * sizeof(uint64_t));
  }

  auto *dst = reinterpret_cast<int32_t *>(col.data.get());
  uint64_t row_offset = 0;

  for (uint64_t chunk_idx = 0;; chunk_idx++) {
    auto chunk = result.Fetch();
    if (!chunk || chunk->size() == 0)
      break;
    chunk->Flatten();
    auto &vec = chunk->data[col_idx];
    auto &validity = duckdb::FlatVector::Validity(vec);

    auto phys_type = result.types[col_idx].InternalType();

    if (phys_type == duckdb::PhysicalType::INT32) {
      auto *src = duckdb::FlatVector::GetData<int32_t>(vec);
      for (uint64_t r = 0; r < chunk->size(); r++) {
        if (nullable && !validity.RowIsValid(r)) {
          dst[row_offset + r] = 0;
          col.SetNull(row_offset + r);
        } else {
          dst[row_offset + r] = src[r];
        }
      }
    } else if (phys_type == duckdb::PhysicalType::INT16) {
      auto *src = duckdb::FlatVector::GetData<int16_t>(vec);
      for (uint64_t r = 0; r < chunk->size(); r++) {
        if (nullable && !validity.RowIsValid(r)) {
          dst[row_offset + r] = 0;
          col.SetNull(row_offset + r);
        } else {
          dst[row_offset + r] = static_cast<int32_t>(src[r]);
        }
      }
    } else if (phys_type == duckdb::PhysicalType::INT8) {
      auto *src = duckdb::FlatVector::GetData<int8_t>(vec);
      for (uint64_t r = 0; r < chunk->size(); r++) {
        if (nullable && !validity.RowIsValid(r)) {
          dst[row_offset + r] = 0;
          col.SetNull(row_offset + r);
        } else {
          dst[row_offset + r] = static_cast<int32_t>(src[r]);
        }
      }
    } else {
      throw std::runtime_error(
          "StoragePlan unsupported: physical type of " +
          result.types[col_idx].ToString() +
          " cannot be loaded into an INT32 flat column without truncation");
    }

    row_offset += chunk->size();
  }

  return col;
}

static FlatColumn LoadColumnINT64(duckdb::MaterializedQueryResult &result,
                                  size_t col_idx, uint64_t total_rows,
                                  bool nullable) {
  FlatColumn col;
  col.type = FlatColumnType::INT64;
  col.row_count = total_rows;
  col.nullable = nullable;
  col.data = std::make_unique<char[]>(total_rows * sizeof(int64_t));

  if (nullable) {
    uint64_t bitmap_words = (total_rows + 63) / 64;
    col.null_bitmap = std::make_unique<uint64_t[]>(bitmap_words);
    std::memset(col.null_bitmap.get(), 0xFF,
                bitmap_words * sizeof(uint64_t));
  }

  auto *dst = reinterpret_cast<int64_t *>(col.data.get());
  uint64_t row_offset = 0;

  for (;;) {
    auto chunk = result.Fetch();
    if (!chunk || chunk->size() == 0)
      break;
    chunk->Flatten();
    auto &vec = chunk->data[col_idx];
    auto &validity = duckdb::FlatVector::Validity(vec);

    auto phys_type = result.types[col_idx].InternalType();

    if (phys_type == duckdb::PhysicalType::INT64) {
      auto *src = duckdb::FlatVector::GetData<int64_t>(vec);
      for (uint64_t r = 0; r < chunk->size(); r++) {
        if (nullable && !validity.RowIsValid(r)) {
          dst[row_offset + r] = 0;
          col.SetNull(row_offset + r);
        } else {
          dst[row_offset + r] = src[r];
        }
      }
    } else if (phys_type == duckdb::PhysicalType::INT32) {
      auto *src = duckdb::FlatVector::GetData<int32_t>(vec);
      for (uint64_t r = 0; r < chunk->size(); r++) {
        if (nullable && !validity.RowIsValid(r)) {
          dst[row_offset + r] = 0;
          col.SetNull(row_offset + r);
        } else {
          dst[row_offset + r] = static_cast<int64_t>(src[r]);
        }
      }
    } else if (phys_type == duckdb::PhysicalType::INT16) {
      auto *src = duckdb::FlatVector::GetData<int16_t>(vec);
      for (uint64_t r = 0; r < chunk->size(); r++) {
        if (nullable && !validity.RowIsValid(r)) {
          dst[row_offset + r] = 0;
          col.SetNull(row_offset + r);
        } else {
          dst[row_offset + r] = static_cast<int64_t>(src[r]);
        }
      }
    } else if (phys_type == duckdb::PhysicalType::INT8) {
      auto *src = duckdb::FlatVector::GetData<int8_t>(vec);
      for (uint64_t r = 0; r < chunk->size(); r++) {
        if (nullable && !validity.RowIsValid(r)) {
          dst[row_offset + r] = 0;
          col.SetNull(row_offset + r);
        } else {
          dst[row_offset + r] = static_cast<int64_t>(src[r]);
        }
      }
    } else {
      throw std::runtime_error(
          "StoragePlan unsupported: physical type of " +
          result.types[col_idx].ToString() +
          " cannot be loaded into an INT64 flat column");
    }

    row_offset += chunk->size();
  }

  return col;
}

// Two-pass VARCHAR loading: first pass counts total string bytes,
// second pass fills the pool and offsets.
// We need to re-scan the result, so we do it in one pass by collecting
// strings temporarily.
static FlatColumn LoadColumnVARCHAR(duckdb::MaterializedQueryResult &result,
                                    size_t col_idx, uint64_t total_rows,
                                    bool nullable) {
  FlatColumn col;
  col.type = FlatColumnType::VARCHAR;
  col.row_count = total_rows;
  col.nullable = nullable;

  // Offsets array: row_count + 1 entries
  col.data =
      std::make_unique<char[]>((total_rows + 1) * sizeof(uint32_t));
  auto *offsets = reinterpret_cast<uint32_t *>(col.data.get());

  if (nullable) {
    uint64_t bitmap_words = (total_rows + 63) / 64;
    col.null_bitmap = std::make_unique<uint64_t[]>(bitmap_words);
    std::memset(col.null_bitmap.get(), 0xFF,
                bitmap_words * sizeof(uint64_t));
  }

  // Collect all strings in a temporary buffer to compute total size
  // and fill in one pass. For very large tables (cast_info 36M rows),
  // this uses ~1GB of temporary std::string overhead, which is acceptable
  // on 63GB machine.
  std::vector<std::pair<const char *, uint32_t>> str_refs;
  str_refs.reserve(total_rows);

  // We need to keep chunk data alive until we copy strings
  std::vector<duckdb::unique_ptr<duckdb::DataChunk>> chunks;

  uint64_t total_bytes = 0;
  uint64_t row_offset = 0;

  for (;;) {
    auto chunk = result.Fetch();
    if (!chunk || chunk->size() == 0)
      break;
    chunk->Flatten();
    auto &vec = chunk->data[col_idx];
    auto &validity = duckdb::FlatVector::Validity(vec);
    auto *src = duckdb::FlatVector::GetData<duckdb::string_t>(vec);

    for (uint64_t r = 0; r < chunk->size(); r++) {
      if (nullable && !validity.RowIsValid(r)) {
        col.SetNull(row_offset + r);
        str_refs.push_back({nullptr, 0});
      } else {
        auto len = static_cast<uint32_t>(src[r].GetSize());
        str_refs.push_back({src[r].GetData(), len});
        total_bytes += len;
      }
    }

    row_offset += chunk->size();
    chunks.push_back(std::move(chunk));
  }

  // Allocate string pool and fill offsets
  col.string_pool_size = total_bytes;
  col.string_pool = std::make_unique<char[]>(total_bytes);
  char *pool = col.string_pool.get();

  uint32_t offset = 0;
  for (uint64_t i = 0; i < total_rows; i++) {
    offsets[i] = offset;
    const char *ptr = str_refs[i].first;
    uint32_t len = str_refs[i].second;
    if (len > 0 && ptr) {
      std::memcpy(pool + offset, ptr, len);
      offset += len;
    }
  }
  offsets[total_rows] = offset;

  return col;
}

void StoragePlan::LoadFromDuckDB(duckdb::Connection &connection) {
  auto start = std::chrono::high_resolution_clock::now();

  // Get all table names
  auto table_result =
      connection.Query("SELECT table_name FROM information_schema.tables "
                       "WHERE table_schema='main' AND table_type='BASE TABLE' "
                       "ORDER BY table_name");
  if (table_result->HasError()) {
    throw std::runtime_error("Failed to enumerate tables: " +
                             table_result->GetError());
  }

  std::vector<std::string> table_names;
  for (uint64_t i = 0; i < table_result->RowCount(); i++) {
    table_names.push_back(table_result->GetValue(0, i).ToString());
  }

#ifndef NDEBUG
  std::cerr << "[StoragePlan] Loading " << table_names.size()
            << " tables into flat arrays..." << std::endl;
#endif

  for (const auto &tname : table_names) {
    // Get column info
    auto col_info = connection.Query(
        "SELECT column_name, data_type, is_nullable "
        "FROM information_schema.columns "
        "WHERE table_name='" +
        tname + "' ORDER BY ordinal_position");
    if (col_info->HasError())
      throw std::runtime_error("Failed to get columns for " + tname +
                                ": " + col_info->GetError());

    std::vector<std::string> col_names;
    std::vector<FlatColumnType> col_types;
    std::vector<bool> col_nullable;
    for (uint64_t i = 0; i < col_info->RowCount(); i++) {
      col_names.push_back(col_info->GetValue(0, i).ToString());
      std::string dtype = col_info->GetValue(1, i).ToString();
      bool is_nullable = col_info->GetValue(2, i).ToString() == "YES";

      // Map DuckDB types to FlatColumnType by their PHYSICAL representation.
      // DATE is physically INT32 (days since epoch); TIME is INT64 (µs);
      // DECIMAL(p,s) is INT16/INT32/INT64/INT128 depending on precision.
      if (dtype == "INTEGER" || dtype == "SMALLINT" || dtype == "TINYINT" ||
          dtype == "INT" || dtype == "DATE") {
        col_types.push_back(FlatColumnType::INT32);
      } else if (dtype == "BIGINT" || dtype == "TIME") {
        col_types.push_back(FlatColumnType::INT64);
      } else if (dtype.rfind("DECIMAL(", 0) == 0) {
        int precision = std::stoi(dtype.substr(8));
        if (precision <= 9) {
          col_types.push_back(FlatColumnType::INT32);
        } else if (precision <= 18) {
          col_types.push_back(FlatColumnType::INT64);
        } else {
          throw std::runtime_error(
              "StoragePlan unsupported: column " + tname + "." +
              col_names.back() + " has type " + dtype +
              " (precision > 18 is physically INT128)");
        }
      } else if (dtype == "VARCHAR" || dtype == "TEXT" ||
                 dtype.rfind("VARCHAR", 0) == 0 ||
                 dtype.rfind("CHAR", 0) == 0 ||
                 dtype.rfind("CHARACTER", 0) == 0) {
        col_types.push_back(FlatColumnType::VARCHAR);
      } else {
        throw std::runtime_error(
            "StoragePlan unsupported: column " + tname + "." +
            col_names.back() + " has type " + dtype +
            "; no FlatColumnType mapping");
      }
      col_nullable.push_back(is_nullable);
    }

    // Get row count
    auto count_result =
        connection.Query("SELECT count(*) FROM " + tname);
    uint64_t row_count = count_result->GetValue(0, 0).GetValue<uint64_t>();

    FlatTable table;
    table.table_name = tname;
    table.row_count = row_count;
    table.column_names = col_names;
    table.columns.resize(col_names.size());

    // Load each column separately to avoid holding all columns in memory
    // simultaneously during loading
    for (size_t ci = 0; ci < col_names.size(); ci++) {
      std::string order_clause;
      if (std::find(col_names.begin(), col_names.end(), "id") != col_names.end())
        order_clause = " ORDER BY id";
      auto data_result =
          connection.Query("SELECT " + col_names[ci] + " FROM " + tname + order_clause);
      if (data_result->HasError())
        throw std::runtime_error("Failed to load column " + col_names[ci] +
                                  " from " + tname + ": " +
                                  data_result->GetError());

      if (col_types[ci] == FlatColumnType::INT32) {
        table.columns[ci] = LoadColumnINT32(*data_result, 0, row_count,
                                            col_nullable[ci]);
      } else if (col_types[ci] == FlatColumnType::INT64) {
        table.columns[ci] = LoadColumnINT64(*data_result, 0, row_count,
                                            col_nullable[ci]);
      } else {
        table.columns[ci] = LoadColumnVARCHAR(*data_result, 0, row_count,
                                              col_nullable[ci]);
      }
    }

    // Compute max PK for the "id" column
    int id_col = table.FindColumn("id");
    if (id_col >= 0 && col_types[id_col] == FlatColumnType::INT32) {
      int32_t max_id = 0;
      const auto *id_data =
          reinterpret_cast<const int32_t *>(table.columns[id_col].data.get());
      for (uint64_t r = 0; r < row_count; r++) {
        if (id_data[r] > max_id)
          max_id = id_data[r];
      }
      table.max_pk = max_id;
      if (row_count > 0 && max_id == static_cast<int32_t>(row_count)) {
        int32_t min_id = max_id;
        for (uint64_t r = 0; r < row_count; r++) {
          if (id_data[r] < min_id) min_id = id_data[r];
        }
        bool in_order = (min_id == 1);
        if (in_order) {
          for (uint64_t r = 0; r < row_count && in_order; r++) {
            if (id_data[r] != static_cast<int32_t>(r + 1))
              in_order = false;
          }
        }
        if (in_order)
          table.dense_pk = true;
      }
      if (!table.dense_pk) {
        table.pk_to_row.assign(static_cast<size_t>(max_id) + 1, 0);
        for (uint64_t r = 0; r < row_count; r++) {
          int32_t pk = id_data[r];
          if (pk >= 0)
            table.pk_to_row[pk] = static_cast<uint32_t>(r);
        }
      }
    }

#ifndef NDEBUG
    std::cerr << "[StoragePlan]   " << tname << ": " << row_count
              << " rows, " << col_names.size() << " cols, "
              << (table.GetMemoryUsage() / (1024 * 1024)) << " MB"
              << (table.dense_pk ? " [dense_pk]" : "")
              << (!table.pk_to_row.empty() ? " [pk_to_row]" : "")
              << std::endl;
#endif

    tables_[tname] = std::move(table);
  }

  loaded_ = true;
  dim_cache_.Build(tables_);

  auto end = std::chrono::high_resolution_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
#ifndef NDEBUG
  std::cerr << "[StoragePlan] Flat table loading complete in " << ms
            << " ms, total memory: " << (GetMemoryUsage() / (1024 * 1024))
            << " MB" << std::endl;
#endif
}

#ifdef HAVE_POSTGRES
#include <libpq-fe.h>

void StoragePlan::LoadFromPostgreSQL(PGconn *conn) {
  auto start = std::chrono::high_resolution_clock::now();

  PGresult *table_result = PQexec(
      conn,
      "SELECT table_name FROM information_schema.tables "
      "WHERE table_schema='public' AND table_type='BASE TABLE' "
      "ORDER BY table_name");
  if (PQresultStatus(table_result) != PGRES_TUPLES_OK) {
    std::string err = PQerrorMessage(conn);
    PQclear(table_result);
    throw std::runtime_error("Failed to enumerate tables: " + err);
  }

  std::vector<std::string> table_names;
  for (int i = 0; i < PQntuples(table_result); i++)
    table_names.emplace_back(PQgetvalue(table_result, i, 0));
  PQclear(table_result);

#ifndef NDEBUG
  std::cerr << "[StoragePlan] Loading " << table_names.size()
            << " tables from PostgreSQL into flat arrays..." << std::endl;
#endif

  for (const auto &tname : table_names) {
    std::string col_sql =
        "SELECT column_name, data_type, is_nullable "
        "FROM information_schema.columns WHERE table_name='" +
        tname + "' ORDER BY ordinal_position";
    PGresult *col_result = PQexec(conn, col_sql.c_str());
    if (PQresultStatus(col_result) != PGRES_TUPLES_OK) {
      std::string err = PQerrorMessage(conn);
      PQclear(col_result);
      throw std::runtime_error("Failed to get columns for " + tname +
                                ": " + err);
    }

    std::vector<std::string> col_names;
    std::vector<FlatColumnType> col_types;
    std::vector<bool> col_nullable;
    std::vector<bool> col_is_bpchar;
    for (int i = 0; i < PQntuples(col_result); i++) {
      col_names.emplace_back(PQgetvalue(col_result, i, 0));
      std::string dtype = PQgetvalue(col_result, i, 1);
      bool nullable = std::string(PQgetvalue(col_result, i, 2)) == "YES";

      if (dtype == "integer" || dtype == "bigint" || dtype == "smallint" ||
          dtype == "int" || dtype == "date") {
        col_types.push_back(FlatColumnType::INT32);
      } else {
        col_types.push_back(FlatColumnType::VARCHAR);
      }
      col_nullable.push_back(nullable);
      col_is_bpchar.push_back(dtype == "character");
    }
    PQclear(col_result);

    std::string count_sql = "SELECT count(*) FROM " + tname;
    PGresult *count_result = PQexec(conn, count_sql.c_str());
    uint64_t row_count = 0;
    if (PQresultStatus(count_result) == PGRES_TUPLES_OK &&
        PQntuples(count_result) > 0)
      row_count = std::stoull(PQgetvalue(count_result, 0, 0));
    PQclear(count_result);

    FlatTable table;
    table.table_name = tname;
    table.row_count = row_count;
    table.column_names = col_names;
    table.columns.resize(col_names.size());

    std::string order_col;
    if (std::find(col_names.begin(), col_names.end(), "id") !=
        col_names.end()) {
      order_col = "id";
    } else {
      std::string pk_sql =
          "SELECT a.attname FROM pg_index i "
          "JOIN pg_attribute a ON a.attrelid = i.indrelid "
          "AND a.attnum = i.indkey[0] "
          "WHERE i.indrelid = '" + tname + "'::regclass "
          "AND i.indisprimary AND array_length(i.indkey, 1) = 1";
      PGresult *pk_result = PQexec(conn, pk_sql.c_str());
      if (PQresultStatus(pk_result) == PGRES_TUPLES_OK &&
          PQntuples(pk_result) > 0)
        order_col = PQgetvalue(pk_result, 0, 0);
      PQclear(pk_result);
    }
    std::string order_clause;
    if (!order_col.empty())
      order_clause = " ORDER BY " + order_col;

    for (size_t ci = 0; ci < col_names.size(); ci++) {
      std::string data_sql =
          "SELECT " + col_names[ci] + " FROM " + tname + order_clause;
      PGresult *data_result = PQexec(conn, data_sql.c_str());
      if (PQresultStatus(data_result) != PGRES_TUPLES_OK) {
        std::string err = PQerrorMessage(conn);
        PQclear(data_result);
        throw std::runtime_error("Failed to load column " + col_names[ci] +
                                  " from " + tname + ": " + err);
      }

      FlatColumn col;
      col.type = col_types[ci];
      col.row_count = row_count;
      col.nullable = col_nullable[ci];
      int nrows = PQntuples(data_result);

      if (col_types[ci] == FlatColumnType::INT32) {
        col.data = std::make_unique<char[]>(sizeof(int32_t) * row_count);
        auto *dest = reinterpret_cast<int32_t *>(col.data.get());
        if (col.nullable) {
          size_t bitmask_size = (row_count + 63) / 64;
          col.null_bitmap = std::make_unique<uint64_t[]>(bitmask_size);
          std::memset(col.null_bitmap.get(), 0xFF,
                      bitmask_size * sizeof(uint64_t));
          for (int r = 0; r < nrows; r++) {
            if (PQgetisnull(data_result, r, 0)) {
              dest[r] = 0;
              col.SetNull(r);
            } else {
              dest[r] = std::atoi(PQgetvalue(data_result, r, 0));
            }
          }
        } else {
          for (int r = 0; r < nrows; r++)
            dest[r] = std::atoi(PQgetvalue(data_result, r, 0));
        }
      } else {
        // VARCHAR: data stores uint32_t[row_count+1] offsets,
        // string_pool stores the contiguous string bytes.
        const bool trim_spaces = col_is_bpchar[ci];
        uint64_t total_bytes = 0;
        for (int r = 0; r < nrows; r++) {
          if (!PQgetisnull(data_result, r, 0)) {
            size_t len = std::strlen(PQgetvalue(data_result, r, 0));
            if (trim_spaces)
              while (len > 0 && PQgetvalue(data_result, r, 0)[len - 1] == ' ')
                --len;
            total_bytes += len;
          }
        }
        col.data = std::make_unique<char[]>(
            sizeof(uint32_t) * (row_count + 1));
        auto *offsets = reinterpret_cast<uint32_t *>(col.data.get());
        col.string_pool = std::make_unique<char[]>(total_bytes);
        col.string_pool_size = total_bytes;
        if (col.nullable) {
          size_t bitmask_size = (row_count + 63) / 64;
          col.null_bitmap = std::make_unique<uint64_t[]>(bitmask_size);
          std::memset(col.null_bitmap.get(), 0xFF,
                      bitmask_size * sizeof(uint64_t));
        }
        uint32_t offset = 0;
        for (int r = 0; r < nrows; r++) {
          offsets[r] = offset;
          if (PQgetisnull(data_result, r, 0)) {
            if (col.nullable)
              col.SetNull(r);
          } else {
            const char *val = PQgetvalue(data_result, r, 0);
            size_t len = std::strlen(val);
            if (trim_spaces)
              while (len > 0 && val[len - 1] == ' ')
                --len;
            std::memcpy(col.string_pool.get() + offset, val, len);
            offset += len;
          }
        }
        offsets[row_count] = offset;
      }

      PQclear(data_result);
      table.columns[ci] = std::move(col);
    }

    // Compute PK metadata for "id" column
    int id_col = table.FindColumn("id");
    if (id_col >= 0 && col_types[id_col] == FlatColumnType::INT32) {
      int32_t max_id = 0;
      const auto *id_data =
          reinterpret_cast<const int32_t *>(table.columns[id_col].data.get());
      for (uint64_t r = 0; r < row_count; r++) {
        if (id_data[r] > max_id)
          max_id = id_data[r];
      }
      table.max_pk = max_id;
      if (row_count > 0 && max_id == static_cast<int32_t>(row_count)) {
        int32_t min_id = max_id;
        for (uint64_t r = 0; r < row_count; r++) {
          if (id_data[r] < min_id)
            min_id = id_data[r];
        }
        bool in_order = (min_id == 1);
        if (in_order) {
          for (uint64_t r = 0; r < row_count && in_order; r++) {
            if (id_data[r] != static_cast<int32_t>(r + 1))
              in_order = false;
          }
        }
        if (in_order)
          table.dense_pk = true;
      }
      if (!table.dense_pk) {
        table.pk_to_row.assign(static_cast<size_t>(max_id) + 1, 0);
        for (uint64_t r = 0; r < row_count; r++) {
          int32_t pk = id_data[r];
          if (pk >= 0)
            table.pk_to_row[pk] = static_cast<uint32_t>(r);
        }
      }
    }

#ifndef NDEBUG
    std::cerr << "[StoragePlan]   " << tname << ": " << row_count
              << " rows, " << col_names.size() << " cols, "
              << (table.GetMemoryUsage() / (1024 * 1024)) << " MB"
              << (table.dense_pk ? " [dense_pk]" : "")
              << (!table.pk_to_row.empty() ? " [pk_to_row]" : "")
              << std::endl;
#endif

    tables_[tname] = std::move(table);
  }

  loaded_ = true;
  dim_cache_.Build(tables_);

  auto end = std::chrono::high_resolution_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
#ifndef NDEBUG
  std::cerr << "[StoragePlan] PostgreSQL flat table loading complete in " << ms
            << " ms, total memory: " << (GetMemoryUsage() / (1024 * 1024))
            << " MB" << std::endl;
#endif
}
#endif // HAVE_POSTGRES

void StoragePlan::BuildCSRIndexes(const std::string &fkeys_path) {
  if (!loaded_) {
    throw std::runtime_error(
        "StoragePlan::BuildCSRIndexes: tables not loaded yet");
  }

  auto start = std::chrono::high_resolution_clock::now();

  // Parse fkeys.sql to extract FK relationships
  std::ifstream file(fkeys_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open fkeys file: " + fkeys_path);
  }

  std::string current_table;
  std::regex alter_regex(R"(ALTER\s+TABLE\s+(\w+))",
                         std::regex_constants::icase);
  std::regex fk_regex(
      R"(FOREIGN\s+KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)\s*\((\w+)\))",
      std::regex_constants::icase);

  std::string line;
  int csr_count = 0;

  while (std::getline(file, line)) {
    std::smatch alter_match;
    if (std::regex_search(line, alter_match, alter_regex)) {
      current_table = alter_match[1].str();
      std::transform(current_table.begin(), current_table.end(),
                     current_table.begin(), ::tolower);
    }

    std::smatch fk_match;
    if (!current_table.empty() &&
        std::regex_search(line, fk_match, fk_regex)) {
      std::string fk_column = fk_match[1].str();
      std::string pk_table = fk_match[2].str();
      std::string pk_column = fk_match[3].str();

      std::transform(fk_column.begin(), fk_column.end(), fk_column.begin(),
                     ::tolower);
      std::transform(pk_table.begin(), pk_table.end(), pk_table.begin(),
                     ::tolower);
      std::transform(pk_column.begin(), pk_column.end(), pk_column.begin(),
                     ::tolower);

      // Look up tables
      auto fk_it = tables_.find(current_table);
      auto pk_it = tables_.find(pk_table);
      if (fk_it == tables_.end() || pk_it == tables_.end())
        continue;

      auto &fk_tbl = fk_it->second;
      auto &pk_tbl = pk_it->second;

      int fk_col_idx = fk_tbl.FindColumn(fk_column);
      if (fk_col_idx < 0) {
#ifndef NDEBUG
        std::cerr << "[StoragePlan] Warning: FK column " << fk_column
                  << " not found in " << current_table << std::endl;
#endif
        continue;
      }

      if (fk_tbl.columns[fk_col_idx].type != FlatColumnType::INT32) {
#ifndef NDEBUG
        std::cerr << "[StoragePlan] Warning: FK column " << fk_column
                  << " in " << current_table
                  << " is not INT32, skipping CSR" << std::endl;
#endif
        continue;
      }

      if (pk_tbl.max_pk < 0) {
#ifndef NDEBUG
        std::cerr << "[StoragePlan] Warning: PK table " << pk_table
                  << " has no max_pk, skipping CSR" << std::endl;
#endif
        continue;
      }

      auto csr = BuildCSR(fk_tbl, fk_col_idx, pk_tbl.max_pk,
                           current_table, fk_column, pk_table, pk_column);

      std::string key = current_table + "." + fk_column;
#ifndef NDEBUG
      std::cerr << "[StoragePlan]   CSR " << key << " → " << pk_table
                << "." << pk_column << ": row_ptr="
                << (csr.row_ptr_size * 8 / (1024 * 1024)) << " MB, col_idx="
                << (csr.col_idx_size * 4 / (1024 * 1024)) << " MB"
                << std::endl;
#endif

      csr_indexes_[key] = std::move(csr);
      csr_count++;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  uint64_t total_csr_mem = 0;
  for (const auto &kv : csr_indexes_)
    total_csr_mem += kv.second.GetMemoryUsage();

#ifndef NDEBUG
  std::cerr << "[StoragePlan] Built " << csr_count << " CSR indexes in "
            << ms << " ms, total CSR memory: "
            << (total_csr_mem / (1024 * 1024)) << " MB" << std::endl;
#endif
}

void StoragePlan::BuildSortedIndices() {
  if (!loaded_)
    return;

  auto start = std::chrono::high_resolution_clock::now();

  static const std::vector<std::pair<std::string, std::string>> kSortedCols = {
      {"title", "title"},
      {"title", "production_year"},
      {"name", "name"},
      {"char_name", "name"},
      {"company_name", "name"},
      {"movie_info", "info"},
      {"movie_info_idx", "info"},
      {"aka_name", "name"},
      {"movie_companies", "note"},
      {"keyword", "keyword"},
      {"link_type", "link"},
  };

  sorted_indices_.clear();
  for (const auto &[tname, cname] : kSortedCols) {
    auto it = tables_.find(tname);
    if (it == tables_.end())
      continue;
    if (it->second.FindColumn(cname) < 0)
      continue;
    std::string key = tname + "." + cname;
    sorted_indices_[key] = BuildSortedIndex(it->second, cname);
#ifndef NDEBUG
    std::cerr << "[StoragePlan]   Sorted index: " << key << " ("
              << sorted_indices_[key].sorted_perm.size() << " entries)"
              << std::endl;
#endif
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
#ifndef NDEBUG
  std::cerr << "[StoragePlan] Built " << sorted_indices_.size()
            << " sorted indices in " << ms << " ms" << std::endl;
#endif
}

const SortedIndex *
StoragePlan::GetSortedIndex(const std::string &table_name,
                            const std::string &col_name) const {
  auto it = sorted_indices_.find(table_name + "." + col_name);
  if (it == sorted_indices_.end())
    return nullptr;
  return &it->second;
}

void StoragePlan::BuildInvertedIndices() {
  if (!loaded_)
    return;

  auto start = std::chrono::high_resolution_clock::now();

  // Define inverted index specifications:
  // {bridge_table, bridge_fk_col (→ dim PK), target_col, dim_table, target_table}
  // Each spec builds: dim_pk_value → sorted [target_col values] via bridge table.
  struct InvSpec {
    std::string bridge_table;
    std::string bridge_fk_col;
    std::string target_col;
    std::string dim_table;
    std::string target_table;
  };

  static const std::vector<InvSpec> kSpecs = {
      // keyword_id → movie_id via movie_keyword (67 queries)
      {"movie_keyword", "keyword_id", "movie_id", "keyword", "title"},
      // person_id → movie_id via cast_info (10 queries)
      {"cast_info", "person_id", "movie_id", "name", "title"},
      // company_id → movie_id via movie_companies (3 queries)
      {"movie_companies", "company_id", "movie_id", "company_name", "title"},
  };

  inverted_indices_.clear();
  int count = 0;

  for (const auto &spec : kSpecs) {
    // Need the CSR on bridge_table.bridge_fk_col
    std::string csr_key = spec.bridge_table + "." + spec.bridge_fk_col;
    auto csr_it = csr_indexes_.find(csr_key);
    if (csr_it == csr_indexes_.end())
      continue;

    const auto &csr = csr_it->second;

    // Need the bridge FlatTable
    auto bridge_it = tables_.find(spec.bridge_table);
    if (bridge_it == tables_.end())
      continue;

    const auto &bridge_flat = bridge_it->second;
    int target_col_idx = bridge_flat.FindColumn(spec.target_col);
    if (target_col_idx < 0)
      continue;
    if (bridge_flat.columns[target_col_idx].type != FlatColumnType::INT32)
      continue;

    auto inv = BuildInvertedIndex(
        csr.row_ptr.get(), csr.col_idx.get(), csr.row_ptr_size,
        bridge_flat, target_col_idx,
        spec.dim_table, spec.bridge_table, spec.bridge_fk_col,
        spec.target_col, spec.target_table);

    std::string key = spec.dim_table + "->" + spec.target_table +
                      "." + spec.bridge_table + "." + spec.bridge_fk_col;
#ifndef NDEBUG
    std::cerr << "[StoragePlan]   Inverted index: " << key
              << " (" << inv.target_vals_size << " entries, "
              << (inv.GetMemoryUsage() / (1024 * 1024)) << " MB)"
              << std::endl;
#endif

    inverted_indices_[key] = std::move(inv);
    count++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
#ifndef NDEBUG
  std::cerr << "[StoragePlan] Built " << count << " inverted indices in "
            << ms << " ms" << std::endl;
#endif
}

const InvertedIndex *
StoragePlan::GetInvertedIndex(const std::string &dim_table,
                              const std::string &target_table) const {
  // Search for any inverted index matching dim_table→target_table
  for (const auto &kv : inverted_indices_) {
    if (kv.second.dim_table == dim_table &&
        kv.second.target_table == target_table)
      return &kv.second;
  }
  return nullptr;
}

const FlatTable *StoragePlan::GetTable(const std::string &table_name) const {
  auto it = tables_.find(table_name);
  if (it == tables_.end())
    return nullptr;
  return &it->second;
}

const CSRIndex *StoragePlan::GetCSR(const std::string &fk_table,
                                    const std::string &fk_column) const {
  auto it = csr_indexes_.find(fk_table + "." + fk_column);
  if (it == csr_indexes_.end())
    return nullptr;
  return &it->second;
}

bool StoragePlan::IsJoinKeyColumn(const std::string &col_name) const {
  if (!join_key_cols_built_) {
    for (const auto &kv : csr_indexes_) {
      join_key_cols_.insert(kv.second.fk_column);
      join_key_cols_.insert(kv.second.pk_column);
    }
    join_key_cols_built_ = true;
  }
  return join_key_cols_.count(col_name) > 0;
}

uint64_t StoragePlan::GetMemoryUsage() const {
  uint64_t mem = 0;
  for (const auto &kv : tables_)
    mem += kv.second.GetMemoryUsage();
  for (const auto &kv : csr_indexes_)
    mem += kv.second.GetMemoryUsage();
  for (const auto &kv : inverted_indices_)
    mem += kv.second.GetMemoryUsage();
  return mem;
}

void StoragePlan::PrintSummary() const {
#ifndef NDEBUG
  std::cerr << "\n=== StoragePlan Summary ===" << std::endl;
  std::cerr << "Tables: " << tables_.size() << std::endl;
  for (const auto &kv : tables_) {
    const auto &table = kv.second;
    std::cerr << "  " << kv.first << ": " << table.row_count << " rows, "
              << table.columns.size() << " cols, "
              << (table.GetMemoryUsage() / (1024 * 1024)) << " MB"
              << ", max_pk=" << table.max_pk << std::endl;
  }
  std::cerr << "CSR indexes: " << csr_indexes_.size() << std::endl;
  for (const auto &kv : csr_indexes_) {
    const auto &csr = kv.second;
    std::cerr << "  " << kv.first << " -> " << csr.pk_table << "."
              << csr.pk_column << ": "
              << (csr.GetMemoryUsage() / (1024 * 1024)) << " MB"
              << std::endl;
  }
  std::cerr << "Total memory: " << (GetMemoryUsage() / (1024 * 1024))
            << " MB" << std::endl;
  std::cerr << "===========================" << std::endl;
#endif
}

// ─── Binary cache format ─────────────────────────────────────────────────────
// Header: magic(8) version(4) num_tables(4) num_csrs(4)
// Per table: name_len(4) name(name_len) row_count(8) max_pk(4) num_cols(4)
//   Per column: type(1) nullable(1) row_count(8)
//     INT32:   data[row_count * 4]
//     INT64:   data[row_count * 8]        (version >= 4)
//     VARCHAR: pool_size(8) offsets[(row_count+1) * 4] pool[pool_size]
//     if nullable: bitmap[((row_count+63)/64) * 8]
//     name_len(4) name(name_len)
// Per CSR: fk_table_len(4) fk_table fk_col_len(4) fk_col
//          pk_table_len(4) pk_table pk_col_len(4) pk_col
//          row_ptr_size(8) row_ptr[row_ptr_size * 8]
//          col_idx_size(8) col_idx[col_idx_size * 4]

static constexpr uint64_t CACHE_MAGIC = 0x41515053544F5245ULL; // "AQPSTORE"
static constexpr uint32_t CACHE_VERSION = 4;

static void WriteStr(FILE *f, const std::string &s) {
  uint32_t len = static_cast<uint32_t>(s.size());
  fwrite(&len, 4, 1, f);
  fwrite(s.data(), 1, len, f);
}

static std::string ReadStr(FILE *f) {
  uint32_t len;
  if (fread(&len, 4, 1, f) != 1) return "";
  std::string s(len, '\0');
  if (len > 0 && fread(&s[0], 1, len, f) != len) return "";
  return s;
}

static void SkipStr(FILE *f) {
  uint32_t len;
  if (fread(&len, 4, 1, f) != 1) return;
  fseek(f, static_cast<long>(len), SEEK_CUR);
}

void StoragePlan::SaveToFile(const std::string &path) const {
  auto start = std::chrono::high_resolution_clock::now();
  FILE *f = fopen(path.c_str(), "wb");
  if (!f)
    throw std::runtime_error("StoragePlan::SaveToFile: cannot open " + path);

  fwrite(&CACHE_MAGIC, 8, 1, f);
  fwrite(&CACHE_VERSION, 4, 1, f);
  uint32_t num_tables = static_cast<uint32_t>(tables_.size());
  uint32_t num_csrs = static_cast<uint32_t>(csr_indexes_.size());
  fwrite(&num_tables, 4, 1, f);
  fwrite(&num_csrs, 4, 1, f);

  for (const auto &kv : tables_) {
    const auto &tbl = kv.second;
    WriteStr(f, tbl.table_name);
    fwrite(&tbl.row_count, 8, 1, f);
    fwrite(&tbl.max_pk, 4, 1, f);
    uint32_t num_cols = static_cast<uint32_t>(tbl.columns.size());
    fwrite(&num_cols, 4, 1, f);

    for (uint32_t c = 0; c < num_cols; c++) {
      const auto &col = tbl.columns[c];
      uint8_t type_byte = static_cast<uint8_t>(col.type);
      uint8_t nullable_byte = col.nullable ? 1 : 0;
      fwrite(&type_byte, 1, 1, f);
      fwrite(&nullable_byte, 1, 1, f);
      fwrite(&col.row_count, 8, 1, f);

      if (col.type == FlatColumnType::INT32) {
        fwrite(col.data.get(), sizeof(int32_t), col.row_count, f);
      } else if (col.type == FlatColumnType::INT64) {
        fwrite(col.data.get(), sizeof(int64_t), col.row_count, f);
      } else {
        fwrite(&col.string_pool_size, 8, 1, f);
        fwrite(col.data.get(), sizeof(uint32_t), col.row_count + 1, f);
        fwrite(col.string_pool.get(), 1, col.string_pool_size, f);
      }

      if (col.nullable && col.null_bitmap) {
        uint64_t bitmap_words = (col.row_count + 63) / 64;
        fwrite(col.null_bitmap.get(), sizeof(uint64_t), bitmap_words, f);
      }

      WriteStr(f, tbl.column_names[c]);
    }
  }

  for (const auto &kv : csr_indexes_) {
    const auto &csr = kv.second;
    WriteStr(f, csr.fk_table);
    WriteStr(f, csr.fk_column);
    WriteStr(f, csr.pk_table);
    WriteStr(f, csr.pk_column);
    fwrite(&csr.row_ptr_size, 8, 1, f);
    fwrite(csr.row_ptr.get(), sizeof(uint64_t), csr.row_ptr_size, f);
    fwrite(&csr.col_idx_size, 8, 1, f);
    fwrite(csr.col_idx.get(), sizeof(uint32_t), csr.col_idx_size, f);
  }

  // Write sorted indices
  uint32_t num_sorted = static_cast<uint32_t>(sorted_indices_.size());
  fwrite(&num_sorted, 4, 1, f);
  for (const auto &kv : sorted_indices_) {
    const auto &si = kv.second;
    WriteStr(f, si.table_name);
    WriteStr(f, si.column_name);
    uint64_t perm_count = si.sorted_perm.size();
    fwrite(&perm_count, 8, 1, f);
    fwrite(si.sorted_perm.data(), sizeof(uint32_t), perm_count, f);
  }

  // Write inverted indices (version >= 3)
  uint32_t num_inverted = static_cast<uint32_t>(inverted_indices_.size());
  fwrite(&num_inverted, 4, 1, f);
  for (const auto &kv : inverted_indices_) {
    const auto &inv = kv.second;
    WriteStr(f, inv.dim_table);
    WriteStr(f, inv.bridge_table);
    WriteStr(f, inv.bridge_fk_col);
    WriteStr(f, inv.target_col);
    WriteStr(f, inv.target_table);
    fwrite(&inv.row_ptr_size, 8, 1, f);
    fwrite(inv.row_ptr.get(), sizeof(uint64_t), inv.row_ptr_size, f);
    fwrite(&inv.target_vals_size, 8, 1, f);
    fwrite(inv.target_vals.get(), sizeof(int32_t), inv.target_vals_size, f);
  }

  fclose(f);
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
#ifndef NDEBUG
  std::cerr << "[StoragePlan] Saved cache to " << path << " in " << ms << " ms"
            << std::endl;
#endif
}

bool StoragePlan::LoadFromFile(const std::string &path, bool skip_indexes) {
  auto start = std::chrono::high_resolution_clock::now();
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;

  uint64_t magic;
  uint32_t version;
  if (fread(&magic, 8, 1, f) != 1 || magic != CACHE_MAGIC) { fclose(f); return false; }
  if (fread(&version, 4, 1, f) != 1 || version > CACHE_VERSION) { fclose(f); return false; }

  uint32_t num_tables, num_csrs;
  fread(&num_tables, 4, 1, f);
  fread(&num_csrs, 4, 1, f);

  tables_.clear();
  csr_indexes_.clear();

  for (uint32_t t = 0; t < num_tables; t++) {
    FlatTable tbl;
    tbl.table_name = ReadStr(f);
    fread(&tbl.row_count, 8, 1, f);
    fread(&tbl.max_pk, 4, 1, f);
    uint32_t num_cols;
    fread(&num_cols, 4, 1, f);
    tbl.columns.resize(num_cols);
    tbl.column_names.resize(num_cols);

    for (uint32_t c = 0; c < num_cols; c++) {
      auto &col = tbl.columns[c];
      uint8_t type_byte, nullable_byte;
      fread(&type_byte, 1, 1, f);
      fread(&nullable_byte, 1, 1, f);
      col.type = static_cast<FlatColumnType>(type_byte);
      col.nullable = nullable_byte != 0;
      fread(&col.row_count, 8, 1, f);

      if (col.type == FlatColumnType::INT32) {
        col.data = std::make_unique<char[]>(col.row_count * sizeof(int32_t));
        fread(col.data.get(), sizeof(int32_t), col.row_count, f);
      } else if (col.type == FlatColumnType::INT64) {
        col.data = std::make_unique<char[]>(col.row_count * sizeof(int64_t));
        fread(col.data.get(), sizeof(int64_t), col.row_count, f);
      } else {
        fread(&col.string_pool_size, 8, 1, f);
        col.data = std::make_unique<char[]>((col.row_count + 1) * sizeof(uint32_t));
        fread(col.data.get(), sizeof(uint32_t), col.row_count + 1, f);
        col.string_pool = std::make_unique<char[]>(col.string_pool_size);
        fread(col.string_pool.get(), 1, col.string_pool_size, f);
      }

      if (col.nullable) {
        uint64_t bitmap_words = (col.row_count + 63) / 64;
        col.null_bitmap = std::make_unique<uint64_t[]>(bitmap_words);
        fread(col.null_bitmap.get(), sizeof(uint64_t), bitmap_words, f);
      }

      tbl.column_names[c] = ReadStr(f);
    }

    if (tbl.max_pk >= 0) {
      int id_col = tbl.FindColumn("id");
      if (id_col >= 0 && tbl.columns[id_col].type == FlatColumnType::INT32) {
        const auto *id_data =
            reinterpret_cast<const int32_t *>(tbl.columns[id_col].data.get());
        if (tbl.row_count > 0 && tbl.max_pk == static_cast<int32_t>(tbl.row_count)) {
          bool in_order = (id_data[0] == 1);
          for (uint64_t r = 1; r < tbl.row_count && in_order; r++) {
            if (id_data[r] != static_cast<int32_t>(r + 1))
              in_order = false;
          }
          if (in_order)
            tbl.dense_pk = true;
        }
        if (!tbl.dense_pk) {
          tbl.pk_to_row.assign(static_cast<size_t>(tbl.max_pk) + 1, 0);
          for (uint64_t r = 0; r < tbl.row_count; r++) {
            int32_t pk = id_data[r];
            if (pk >= 0)
              tbl.pk_to_row[pk] = static_cast<uint32_t>(r);
          }
        }
      }
    }

    tables_[tbl.table_name] = std::move(tbl);
  }

  for (uint32_t i = 0; i < num_csrs; i++) {
    if (skip_indexes) {
      SkipStr(f); // fk_table
      SkipStr(f); // fk_column
      SkipStr(f); // pk_table
      SkipStr(f); // pk_column
      uint64_t row_ptr_size = 0, col_idx_size = 0;
      fread(&row_ptr_size, 8, 1, f);
      fseek(f, static_cast<long>(row_ptr_size * sizeof(uint64_t)), SEEK_CUR);
      fread(&col_idx_size, 8, 1, f);
      fseek(f, static_cast<long>(col_idx_size * sizeof(uint32_t)), SEEK_CUR);
      continue;
    }
    CSRIndex csr;
    csr.fk_table = ReadStr(f);
    csr.fk_column = ReadStr(f);
    csr.pk_table = ReadStr(f);
    csr.pk_column = ReadStr(f);
    fread(&csr.row_ptr_size, 8, 1, f);
    csr.row_ptr = std::make_unique<uint64_t[]>(csr.row_ptr_size);
    fread(csr.row_ptr.get(), sizeof(uint64_t), csr.row_ptr_size, f);
    fread(&csr.col_idx_size, 8, 1, f);
    csr.col_idx = std::make_unique<uint32_t[]>(csr.col_idx_size);
    fread(csr.col_idx.get(), sizeof(uint32_t), csr.col_idx_size, f);

    std::string key = csr.fk_table + "." + csr.fk_column;
    csr_indexes_[key] = std::move(csr);
  }

  // Read sorted indices (version >= 2)
  sorted_indices_.clear();
  if (version >= 2) {
    uint32_t num_sorted;
    if (fread(&num_sorted, 4, 1, f) == 1) {
      for (uint32_t i = 0; i < num_sorted; i++) {
        if (skip_indexes) {
          SkipStr(f); // table_name
          SkipStr(f); // column_name
          uint64_t perm_count = 0;
          fread(&perm_count, 8, 1, f);
          fseek(f, static_cast<long>(perm_count * sizeof(uint32_t)), SEEK_CUR);
          continue;
        }
        SortedIndex si;
        si.table_name = ReadStr(f);
        si.column_name = ReadStr(f);
        uint64_t perm_count;
        fread(&perm_count, 8, 1, f);
        si.sorted_perm.resize(perm_count);
        fread(si.sorted_perm.data(), sizeof(uint32_t), perm_count, f);
        std::string key = si.table_name + "." + si.column_name;
        sorted_indices_[key] = std::move(si);
      }
    }
  }

  // Read inverted indices (version >= 3)
  inverted_indices_.clear();
  if (version >= 3) {
    uint32_t num_inverted;
    if (fread(&num_inverted, 4, 1, f) == 1) {
      for (uint32_t i = 0; i < num_inverted; i++) {
        if (skip_indexes) {
          SkipStr(f); // dim_table
          SkipStr(f); // bridge_table
          SkipStr(f); // bridge_fk_col
          SkipStr(f); // target_col
          SkipStr(f); // target_table
          uint64_t row_ptr_size = 0, target_vals_size = 0;
          fread(&row_ptr_size, 8, 1, f);
          fseek(f, static_cast<long>(row_ptr_size * sizeof(uint64_t)), SEEK_CUR);
          fread(&target_vals_size, 8, 1, f);
          fseek(f, static_cast<long>(target_vals_size * sizeof(int32_t)),
                SEEK_CUR);
          continue;
        }
        InvertedIndex inv;
        inv.dim_table = ReadStr(f);
        inv.bridge_table = ReadStr(f);
        inv.bridge_fk_col = ReadStr(f);
        inv.target_col = ReadStr(f);
        inv.target_table = ReadStr(f);
        fread(&inv.row_ptr_size, 8, 1, f);
        inv.row_ptr = std::make_unique<uint64_t[]>(inv.row_ptr_size);
        fread(inv.row_ptr.get(), sizeof(uint64_t), inv.row_ptr_size, f);
        fread(&inv.target_vals_size, 8, 1, f);
        inv.target_vals = std::make_unique<int32_t[]>(inv.target_vals_size);
        fread(inv.target_vals.get(), sizeof(int32_t), inv.target_vals_size, f);
        std::string key = inv.dim_table + "->" + inv.target_table +
                          "." + inv.bridge_table + "." + inv.bridge_fk_col;
        inverted_indices_[key] = std::move(inv);
      }
    }
  }

  fclose(f);
  loaded_ = true;
  dim_cache_.Build(tables_);

  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
#ifndef NDEBUG
  std::cerr << "[StoragePlan] Loaded cache from " << path << " in " << ms
            << " ms (" << tables_.size() << " tables, " << csr_indexes_.size()
            << " CSR indexes, " << sorted_indices_.size() << " sorted indices, "
            << inverted_indices_.size() << " inverted indices, "
            << (GetMemoryUsage() / (1024 * 1024)) << " MB"
            << (skip_indexes ? ", index sections skipped" : "") << ")"
            << std::endl;
#endif

  return true;
}

} // namespace storage
} // namespace middleware
