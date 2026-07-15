#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace middleware {
namespace storage {

enum class FlatColumnType : uint8_t {
  INT32,
  VARCHAR,
  INT64,
};

struct FlatColumn {
  FlatColumnType type;
  uint64_t row_count = 0;
  bool nullable = false;

  // INT32: int32_t[row_count]
  // VARCHAR: uint32_t[row_count+1] offsets into string_pool
  std::unique_ptr<char[]> data;

  // 1-bit-per-row validity bitmap. bit=1 means valid (NOT NULL).
  // Only allocated when nullable=true.
  std::unique_ptr<uint64_t[]> null_bitmap;

  // VARCHAR only: contiguous string storage
  std::unique_ptr<char[]> string_pool;
  uint64_t string_pool_size = 0;

  inline int32_t GetInt32(uint64_t row) const {
    assert(type == FlatColumnType::INT32);
    return reinterpret_cast<const int32_t *>(data.get())[row];
  }

  inline int64_t GetInt64(uint64_t row) const {
    assert(type == FlatColumnType::INT64);
    return reinterpret_cast<const int64_t *>(data.get())[row];
  }

  inline bool IsNull(uint64_t row) const {
    if (!nullable || !null_bitmap)
      return false;
    uint64_t word = row / 64;
    uint64_t bit = row % 64;
    return !(null_bitmap[word] & (uint64_t(1) << bit));
  }

  inline void SetValid(uint64_t row) {
    uint64_t word = row / 64;
    uint64_t bit = row % 64;
    null_bitmap[word] |= (uint64_t(1) << bit);
  }

  inline void SetNull(uint64_t row) {
    uint64_t word = row / 64;
    uint64_t bit = row % 64;
    null_bitmap[word] &= ~(uint64_t(1) << bit);
  }

  // Returns pointer to the string and its length for the given row.
  // Caller must check IsNull() first for nullable VARCHAR columns.
  inline const char *GetVarchar(uint64_t row, uint32_t &out_len) const {
    assert(type == FlatColumnType::VARCHAR);
    const auto *offsets =
        reinterpret_cast<const uint32_t *>(data.get());
    uint32_t start = offsets[row];
    uint32_t end = offsets[row + 1];
    out_len = end - start;
    return string_pool.get() + start;
  }

  // Convenience: get as std::string
  inline std::string GetString(uint64_t row) const {
    uint32_t len;
    const char *ptr = GetVarchar(row, len);
    return std::string(ptr, len);
  }

  uint64_t GetMemoryUsage() const {
    uint64_t mem = 0;
    if (type == FlatColumnType::INT32) {
      mem += row_count * sizeof(int32_t);
    } else if (type == FlatColumnType::INT64) {
      mem += row_count * sizeof(int64_t);
    } else {
      mem += (row_count + 1) * sizeof(uint32_t); // offsets
      mem += string_pool_size;
    }
    if (nullable && null_bitmap) {
      mem += ((row_count + 63) / 64) * sizeof(uint64_t);
    }
    return mem;
  }
};

struct FlatTable {
  std::string table_name;
  uint64_t row_count = 0;
  std::vector<std::string> column_names;
  std::vector<FlatColumn> columns;

  int32_t max_pk = -1; // max value of PK "id" column (-1 = not computed)
  bool dense_pk = false; // true when PK is 1..row_count with no gaps (row = key - 1)
  std::vector<uint32_t> pk_to_row; // only allocated for non-dense PK tables

  // Find column index by name (case-insensitive). Returns -1 if not found.
  int FindColumn(const std::string &name) const;

  uint64_t GetMemoryUsage() const {
    uint64_t mem = 0;
    for (const auto &col : columns)
      mem += col.GetMemoryUsage();
    return mem;
  }
};

} // namespace storage
} // namespace middleware
