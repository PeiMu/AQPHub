/*
 * Utility functions
 * */

#pragma once

#include <chrono>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <vector>

#define DEBUG_MIDDLEWARE true
#define MEASURE_SINGLE_QUERY true

namespace middleware {
bool ends_with_sql(const std::string &filename);
std::vector<std::string> get_sql_files(const std::string &directory);
std::string get_filename(const std::string &path);

struct TestResult {
  std::string query_file;
  bool success;
  std::string error_message;
  int num_rows;
  double parse_time_ms;
  double ir_convert_time_ms;
  double sql_gen_time_ms;
  double execution_time_ms;
};

timespec tic();

void toc(timespec *start_time, const char *prefix);

std::chrono::high_resolution_clock::time_point chrono_tic();

long chrono_toc(std::chrono::high_resolution_clock::time_point *start_time,
                const char *prefix, bool print = true);
} // namespace middleware
