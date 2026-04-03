cmake_minimum_required(VERSION 3.10)
project(AQP_middleware)

set(CMAKE_CXX_STANDARD 14)

# ---------------------------------------------------------------------------
# LLVM JIT support
# LLVM 14 is the default system installation; 17/19 also available.
# ---------------------------------------------------------------------------
find_package(LLVM 14 CONFIG
        HINTS /usr/lib/llvm-14/lib/cmake/llvm
        QUIET)
if (NOT LLVM_FOUND)
    # Try LLVM 17 or 19 as fallback
    find_package(LLVM CONFIG
            HINTS /usr/lib/llvm-17/lib/cmake/llvm
                  /usr/lib/llvm-19/lib/cmake/llvm
            QUIET)
endif ()

if (LLVM_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
    llvm_map_components_to_libnames(LLVM_LIBS
            core orcjit native passes analysis support)
    set(HAVE_LLVM TRUE)

    # aqp_jit_lib: middleware-side LLVM compiler (IR → machine code)
    # Requires C++17 because LLVM headers use C++17 features.
    add_library(aqp_jit_lib STATIC
            src/jit/ir_to_llvm.cpp
            src/jit/aqp_jit_runtime.cpp
    )
    target_include_directories(aqp_jit_lib PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/IR_SQL_Converter/inc
            ${LLVM_INCLUDE_DIRS}
    )
    target_link_libraries(aqp_jit_lib PUBLIC ${LLVM_LIBS})
    set_property(TARGET aqp_jit_lib PROPERTY CXX_STANDARD 17)
    target_compile_definitions(aqp_jit_lib PUBLIC HAVE_LLVM)
    message(STATUS "AQP JIT (LLVM): ENABLED")
else ()
    set(HAVE_LLVM FALSE)
    message(STATUS "LLVM not found — AQP JIT disabled")
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build split library (common for both DuckDB and PostgreSQL)
add_library(split_lib STATIC
        src/util/param_config.cpp
        src/split/ir_reorder_get.cpp
        src/split/ir_utils.cpp
        src/split/foreign_key_extractor.cpp
        src/split/select_sub_ir.cpp
        src/split/fk_based_splitter.cpp
        src/split/topdown_splitter.cpp
        src/split/node_based_splitter.cpp
        src/split/ir_query_splitter.cpp
)

target_include_directories(split_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# DuckDB
set(DUCKDB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../duckdb_132" CACHE PATH "DuckDB source path")
set(HAVE_DUCKDB FALSE)
if (EXISTS ${DUCKDB_ROOT})
    message(STATUS "Found DuckDB source at: ${DUCKDB_ROOT}")

    include_directories(${DUCKDB_ROOT}/src/include)

    find_library(DUCKDB_LIB
            NAMES duckdb libduckdb
            PATHS ${DUCKDB_ROOT}/build/release/src ${DUCKDB_ROOT}/build/debug/src
            NO_DEFAULT_PATH
    )

    if (DUCKDB_LIB)
        set(HAVE_DUCKDB TRUE)
        add_library(duckdb_adapter STATIC src/adapters/duckdb_adapter.cpp src/util/util.cpp)
        target_include_directories(duckdb_adapter PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${DUCKDB_ROOT}/src/include
        )

        # Define HAVE_DUCKDB preprocessor macro for C++ code
        target_compile_definitions(duckdb_adapter PUBLIC HAVE_DUCKDB)
        target_compile_definitions(split_lib PUBLIC HAVE_DUCKDB)

        target_link_libraries(duckdb_adapter PUBLIC ${DUCKDB_LIB} split_lib)
        if (HAVE_LLVM)
            target_link_libraries(duckdb_adapter PUBLIC aqp_jit_lib)
            target_compile_definitions(duckdb_adapter PUBLIC HAVE_LLVM)
            target_compile_definitions(split_lib PUBLIC HAVE_LLVM)
        endif ()

        add_executable(test_duckdb examples/test_duckdb.cpp)
        target_link_libraries(test_duckdb PRIVATE duckdb_adapter split_lib)

        message(STATUS "DuckDB adapter: ENABLED")
    else ()
        message(WARNING "DUCKDB_LIB not found")
    endif ()
else ()
    message(WARNING "DuckDB source not found")
endif ()

# PostgreSQL
set(POSTGRES_ROOT "/home/bita/Project/project_bins" CACHE PATH "PostgreSQL lib/inc path")
set(HAVE_POSTGRES FALSE)
if (EXISTS ${POSTGRES_ROOT})
    message(STATUS "Found PostgreSQL lib/inc at: ${POSTGRES_ROOT}")

    include_directories(
            ${POSTGRES_ROOT}/include)

    # Find libpq library (check custom build and system locations)
    find_library(LIBPQ_LIB
            NAMES pq libpq
            PATHS ${POSTGRES_ROOT}/lib
            NO_DEFAULT_PATH
    )

    if (LIBPQ_LIB)
        set(HAVE_POSTGRES TRUE)
        add_library(postgres_adapter STATIC src/adapters/postgres_adapter.cpp src/util/util.cpp)
        target_include_directories(postgres_adapter PUBLIC
                ${POSTGRES_ROOT}/include
        )

        # Define HAVE_POSTGRES preprocessor macro for C++ code
        target_compile_definitions(postgres_adapter PUBLIC HAVE_POSTGRES)
        target_compile_definitions(split_lib PUBLIC HAVE_POSTGRES)

        target_link_libraries(postgres_adapter PUBLIC ${LIBPQ_LIB} split_lib)

        add_executable(test_postgres examples/test_postgres.cpp)
        target_link_libraries(test_postgres PRIVATE postgres_adapter split_lib)

        message(STATUS "PostgreSQL adapter: ENABLED")
    else ()
        message(WARNING "LIBPQ_LIB not found")
    endif ()
else ()
    message(WARNING "PostgreSQL source not found")
endif ()

# Umbra (uses PostgreSQL wire protocol, requires HAVE_POSTGRES)
set(HAVE_UMBRA FALSE)
if (HAVE_POSTGRES)
    set(HAVE_UMBRA TRUE)
    add_library(umbra_adapter STATIC src/adapters/umbra_adapter.cpp)
    target_compile_definitions(umbra_adapter PUBLIC HAVE_UMBRA)
    target_compile_definitions(split_lib PUBLIC HAVE_UMBRA)
    target_link_libraries(umbra_adapter PUBLIC postgres_adapter)
    message(STATUS "Umbra adapter: ENABLED")
endif ()

# OpenGauss (uses PostgreSQL wire protocol, requires HAVE_POSTGRES)
set(HAVE_OPENGAUSS FALSE)
if (HAVE_POSTGRES)
    set(HAVE_OPENGAUSS TRUE)
    add_library(opengauss_adapter STATIC src/adapters/opengauss_adapter.cpp)
    target_compile_definitions(opengauss_adapter PUBLIC HAVE_OPENGAUSS)
    target_compile_definitions(split_lib PUBLIC HAVE_OPENGAUSS)
    target_link_libraries(opengauss_adapter PUBLIC postgres_adapter)
    message(STATUS "OpenGauss adapter: ENABLED")
endif ()

# MariaDB
set(MARIADB_ROOT "/usr/local/mysql" CACHE PATH "MariaDB lib/inc path")
set(HAVE_MARIADB FALSE)
if (EXISTS ${MARIADB_ROOT})
    message(STATUS "Found MariaDB at: ${MARIADB_ROOT}")

    find_library(LIBMARIADB_LIB
            NAMES mariadb mysqlclient
            PATHS ${MARIADB_ROOT}/lib
            NO_DEFAULT_PATH
    )

    if (LIBMARIADB_LIB)
        set(HAVE_MARIADB TRUE)
        add_library(mariadb_adapter STATIC src/adapters/mariadb_adapter.cpp src/util/util.cpp)
        target_include_directories(mariadb_adapter PUBLIC
                ${MARIADB_ROOT}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        target_compile_definitions(mariadb_adapter PUBLIC HAVE_MARIADB)
        target_compile_definitions(split_lib PUBLIC HAVE_MARIADB)

        target_link_libraries(mariadb_adapter PUBLIC ${LIBMARIADB_LIB} split_lib)

        message(STATUS "MariaDB adapter: ENABLED")
    else ()
        message(WARNING "LIBMARIADB_LIB not found at ${MARIADB_ROOT}/lib")
    endif ()
else ()
    message(WARNING "MariaDB not found at ${MARIADB_ROOT}")
endif ()

# libpg_query
set(LIBPG_QUERY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libpg_query" CACHE PATH "libpg_query source path")
if (EXISTS ${LIBPG_QUERY_ROOT})
    message(STATUS "Found libpg_query at: ${LIBPG_QUERY_ROOT}")

    include_directories(${LIBPG_QUERY_ROOT})

    find_library(LIBPG_QUERY_LIB
            NAMES pg_query libpg_query
            PATHS ${LIBPG_QUERY_ROOT}
            NO_DEFAULT_PATH
    )

    if (LIBPG_QUERY_LIB AND HAVE_POSTGRES)
        target_link_libraries(postgres_adapter PUBLIC ${LIBPG_QUERY_LIB})
        message(STATUS "libpg_query: ENABLED (PostgreSQL)")
    endif ()

    if (LIBPG_QUERY_LIB AND HAVE_MARIADB)
        target_link_libraries(mariadb_adapter PUBLIC ${LIBPG_QUERY_LIB})
        message(STATUS "libpg_query: ENABLED (MariaDB)")
    endif ()

    if (NOT LIBPG_QUERY_LIB)
        message(WARNING "LIBPG_QUERY_LIB not found")
    endif ()
else ()
    message(WARNING "libpg_query not found at ${LIBPG_QUERY_ROOT}")
endif ()


# IR_SQL_Converter (built from source via submodule for full debug support)
set(IR_CONVERTER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/IR_SQL_Converter")
if (EXISTS ${IR_CONVERTER_ROOT}/CMakeLists.txt)
    message(STATUS "Building IR_SQL_Converter from source at: ${IR_CONVERTER_ROOT}")

    # Configure IR_SQL_Converter options before add_subdirectory
    set(DUCKDB_VERSION_010 OFF CACHE BOOL "Build IR_SQL_Converter against DuckDB 0.10.1" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable IR_SQL_Converter tests" FORCE)

    add_subdirectory(${IR_CONVERTER_ROOT} ${CMAKE_BINARY_DIR}/IR_SQL_Converter)

    # Include IR_SQL_Converter headers (not propagated via target properties)
    include_directories(
            ${IR_CONVERTER_ROOT}/inc
            ${IR_CONVERTER_ROOT}/third_party/libpg_query/include
    )

    # Link IR_CONVERTER to split_lib since it uses SimplestIR types
    target_link_libraries(split_lib PUBLIC IR_SQL_Converter_static)

    if (HAVE_DUCKDB)
        target_link_libraries(test_duckdb PUBLIC IR_SQL_Converter_static)
    endif ()
    if (HAVE_POSTGRES)
        target_link_libraries(test_postgres PUBLIC IR_SQL_Converter_static)
    endif ()

    # Build aqp_middleware executable if any adapter is available
    if (HAVE_DUCKDB OR HAVE_POSTGRES OR HAVE_MARIADB)
        add_executable(aqp_middleware src/aqp_middleware.cpp)
        target_link_libraries(aqp_middleware PRIVATE
                split_lib
                IR_SQL_Converter_static
        )

        if (HAVE_DUCKDB)
            target_link_libraries(aqp_middleware PRIVATE duckdb_adapter)
        endif ()
        if (HAVE_POSTGRES)
            target_link_libraries(aqp_middleware PRIVATE postgres_adapter)
        endif ()
        if (HAVE_UMBRA)
            target_link_libraries(aqp_middleware PRIVATE umbra_adapter)
        endif ()
        if (HAVE_MARIADB)
            target_link_libraries(aqp_middleware PRIVATE mariadb_adapter)
        endif ()
        if (HAVE_OPENGAUSS)
            target_link_libraries(aqp_middleware PRIVATE opengauss_adapter)
        endif ()

        message(STATUS "aqp_middleware: ENABLED")
    else ()
        message(WARNING "aqp_middleware: DISABLED (no database adapter available)")
    endif ()
else ()
    message(WARNING "IR_SQL_Converter source not found at ${IR_CONVERTER_ROOT}")
endif ()
