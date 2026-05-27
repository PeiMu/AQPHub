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
    } else if (phys_type == duckdb::PhysicalType::INT64) {
      auto *src = duckdb::FlatVector::GetData<int64_t>(vec);
      for (uint64_t r = 0; r < chunk->size(); r++) {
        if (nullable && !validity.RowIsValid(r)) {
          dst[row_offset + r] = 0;
          col.SetNull(row_offset + r);
        } else {
          dst[row_offset + r] = static_cast<int32_t>(src[r]);
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

  std::cout << "[StoragePlan] Loading " << table_names.size()
            << " tables into flat arrays..." << std::endl;

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

      // Map DuckDB types to FlatColumnType
      if (dtype == "INTEGER" || dtype == "BIGINT" || dtype == "SMALLINT" ||
          dtype == "TINYINT" || dtype == "INT") {
        col_types.push_back(FlatColumnType::INT32);
      } else {
        // VARCHAR, TEXT, CHARACTER VARYING, etc.
        col_types.push_back(FlatColumnType::VARCHAR);
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
      auto data_result =
          connection.Query("SELECT " + col_names[ci] + " FROM " + tname);
      if (data_result->HasError())
        throw std::runtime_error("Failed to load column " + col_names[ci] +
                                  " from " + tname + ": " +
                                  data_result->GetError());

      if (col_types[ci] == FlatColumnType::INT32) {
        table.columns[ci] = LoadColumnINT32(*data_result, 0, row_count,
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
    }

    std::cout << "[StoragePlan]   " << tname << ": " << row_count
              << " rows, " << col_names.size() << " cols, "
              << (table.GetMemoryUsage() / (1024 * 1024)) << " MB"
              << std::endl;

    tables_[tname] = std::move(table);
  }

  loaded_ = true;

  auto end = std::chrono::high_resolution_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "[StoragePlan] Flat table loading complete in " << ms
            << " ms, total memory: " << (GetMemoryUsage() / (1024 * 1024))
            << " MB" << std::endl;
}

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
        std::cerr << "[StoragePlan] Warning: FK column " << fk_column
                  << " not found in " << current_table << std::endl;
        continue;
      }

      if (fk_tbl.columns[fk_col_idx].type != FlatColumnType::INT32) {
        std::cerr << "[StoragePlan] Warning: FK column " << fk_column
                  << " in " << current_table
                  << " is not INT32, skipping CSR" << std::endl;
        continue;
      }

      if (pk_tbl.max_pk < 0) {
        std::cerr << "[StoragePlan] Warning: PK table " << pk_table
                  << " has no max_pk, skipping CSR" << std::endl;
        continue;
      }

      auto csr = BuildCSR(fk_tbl, fk_col_idx, pk_tbl.max_pk,
                           current_table, fk_column, pk_table, pk_column);

      std::string key = current_table + "." + fk_column;
      std::cout << "[StoragePlan]   CSR " << key << " → " << pk_table
                << "." << pk_column << ": row_ptr="
                << (csr.row_ptr_size * 8 / (1024 * 1024)) << " MB, col_idx="
                << (csr.col_idx_size * 4 / (1024 * 1024)) << " MB"
                << std::endl;

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

  std::cout << "[StoragePlan] Built " << csr_count << " CSR indexes in "
            << ms << " ms, total CSR memory: "
            << (total_csr_mem / (1024 * 1024)) << " MB" << std::endl;
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

uint64_t StoragePlan::GetMemoryUsage() const {
  uint64_t mem = 0;
  for (const auto &kv : tables_)
    mem += kv.second.GetMemoryUsage();
  for (const auto &kv : csr_indexes_)
    mem += kv.second.GetMemoryUsage();
  return mem;
}

void StoragePlan::PrintSummary() const {
  std::cout << "\n=== StoragePlan Summary ===" << std::endl;
  std::cout << "Tables: " << tables_.size() << std::endl;
  for (const auto &kv : tables_) {
    const auto &table = kv.second;
    std::cout << "  " << kv.first << ": " << table.row_count << " rows, "
              << table.columns.size() << " cols, "
              << (table.GetMemoryUsage() / (1024 * 1024)) << " MB"
              << ", max_pk=" << table.max_pk << std::endl;
  }
  std::cout << "CSR indexes: " << csr_indexes_.size() << std::endl;
  for (const auto &kv : csr_indexes_) {
    const auto &csr = kv.second;
    std::cout << "  " << kv.first << " -> " << csr.pk_table << "."
              << csr.pk_column << ": "
              << (csr.GetMemoryUsage() / (1024 * 1024)) << " MB"
              << std::endl;
  }
  std::cout << "Total memory: " << (GetMemoryUsage() / (1024 * 1024))
            << " MB" << std::endl;
  std::cout << "===========================" << std::endl;
}

// ─── Binary cache format ─────────────────────────────────────────────────────
// Header: magic(8) version(4) num_tables(4) num_csrs(4)
// Per table: name_len(4) name(name_len) row_count(8) max_pk(4) num_cols(4)
//   Per column: type(1) nullable(1) row_count(8)
//     INT32:   data[row_count * 4]
//     VARCHAR: pool_size(8) offsets[(row_count+1) * 4] pool[pool_size]
//     if nullable: bitmap[((row_count+63)/64) * 8]
//     name_len(4) name(name_len)
// Per CSR: fk_table_len(4) fk_table fk_col_len(4) fk_col
//          pk_table_len(4) pk_table pk_col_len(4) pk_col
//          row_ptr_size(8) row_ptr[row_ptr_size * 8]
//          col_idx_size(8) col_idx[col_idx_size * 4]

static constexpr uint64_t CACHE_MAGIC = 0x41515053544F5245ULL; // "AQPSTORE"
static constexpr uint32_t CACHE_VERSION = 1;

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

  fclose(f);
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "[StoragePlan] Saved cache to " << path << " in " << ms << " ms"
            << std::endl;
}

bool StoragePlan::LoadFromFile(const std::string &path) {
  auto start = std::chrono::high_resolution_clock::now();
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;

  uint64_t magic;
  uint32_t version;
  if (fread(&magic, 8, 1, f) != 1 || magic != CACHE_MAGIC) { fclose(f); return false; }
  if (fread(&version, 4, 1, f) != 1 || version != CACHE_VERSION) { fclose(f); return false; }

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

    tables_[tbl.table_name] = std::move(tbl);
  }

  for (uint32_t i = 0; i < num_csrs; i++) {
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

  fclose(f);
  loaded_ = true;

  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "[StoragePlan] Loaded cache from " << path << " in " << ms
            << " ms (" << tables_.size() << " tables, " << csr_indexes_.size()
            << " CSR indexes, " << (GetMemoryUsage() / (1024 * 1024)) << " MB)"
            << std::endl;

  return true;
}

} // namespace storage
} // namespace middleware
