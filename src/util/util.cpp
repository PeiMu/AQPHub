#include "util/util.h"

namespace middleware {

// Helper function to check if a file ends with .sql
bool ends_with_sql(const std::string &filename) {
  if (filename.length() < 4)
    return false;
  return filename.substr(filename.length() - 4) == ".sql";
}

// Helper function to get all .sql files from a directory
std::vector<std::string> get_sql_files(const std::string &directory) {
  std::vector<std::string> sql_files;

  DIR *dir = opendir(directory.c_str());
  if (!dir) {
    throw std::runtime_error("Cannot open directory: " + directory);
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string filename = entry->d_name;

    // Skip . and ..
    if (filename == "." || filename == "..") {
      continue;
    }

    // Check if it's a .sql file
    if (ends_with_sql(filename)) {
      std::string full_path = directory;
      if (full_path.back() != '/') {
        full_path += '/';
      }
      full_path += filename;
      sql_files.push_back(full_path);
    }
  }

  closedir(dir);
  return sql_files;
}

// Helper function to extract filename from path
std::string get_filename(const std::string &path) {
  size_t last_slash = path.find_last_of('/');
  if (last_slash != std::string::npos) {
    return path.substr(last_slash + 1);
  }
  return path;
}

// Timer function
typedef struct timespec timespec;
timespec diff(timespec start, timespec end) {
  timespec temp;
  if ((end.tv_nsec - start.tv_nsec) < 0) {
    temp.tv_sec = end.tv_sec - start.tv_sec - 1;
    temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  }
  return temp;
}

timespec sum(timespec t1, timespec t2) {
  timespec temp;
  if (t1.tv_nsec + t2.tv_nsec >= 1000000000) {
    temp.tv_sec = t1.tv_sec + t2.tv_sec + 1;
    temp.tv_nsec = t1.tv_nsec + t2.tv_nsec - 1000000000;
  } else {
    temp.tv_sec = t1.tv_sec + t2.tv_sec;
    temp.tv_nsec = t1.tv_nsec + t2.tv_nsec;
  }
  return temp;
}

void printTimeSpec(timespec t, const char *prefix) {
  std::string str = prefix + std::to_string(t.tv_sec) + "." +
                    std::to_string(t.tv_nsec) + " s";
  std::cout << str;
  //    printf("%s: %d.%09d\n", prefix, (int)t.tv_sec, (int)t.tv_nsec);
}

timespec tic() {
  timespec start_time;
  if (-1 == clock_gettime(CLOCK_REALTIME, &start_time)) {
    throw std::runtime_error("Could not get clock time!");
  }
  return start_time;
}

void toc(timespec *start_time, const char *prefix) {
  timespec current_time;
  if (-1 == clock_gettime(CLOCK_REALTIME, &current_time)) {
    throw std::runtime_error("Could not get clock time!");
  }
  printTimeSpec(diff(*start_time, current_time), prefix);
  *start_time = current_time;
}

std::chrono::high_resolution_clock::time_point chrono_tic() {
  return std::chrono::high_resolution_clock::now();
}

long chrono_toc(std::chrono::high_resolution_clock::time_point *start_time,
                const char *prefix, bool print) {
  auto current_time = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
                       current_time - *start_time)
                       .count();
  std::string str = prefix + std::to_string(time_diff) + " us";
  if (print)
    std::cout << str << std::endl;
  *start_time = current_time;
  return time_diff;
}

} // namespace middleware
