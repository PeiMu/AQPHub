cmake_minimum_required(VERSION 3.25)
project(AQP_middleware)

set(CMAKE_CXX_STANDARD 14)

include(ProcessorCount)
ProcessorCount(AQP_NPROC)
if (AQP_NPROC EQUAL 0)
    set(AQP_NPROC 4)
endif ()
set(AQP_TPDE_PARALLEL_JOBS ${AQP_NPROC} CACHE STRING "Parallel jobs for TPDE build")

# Exclude anaconda prefix from CMake search paths — its libraries (zstd, tinfo,
# libstdc++) lack version symbols and contaminate RUNPATH.
list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
set(CMAKE_IGNORE_PREFIX_PATH "${CMAKE_IGNORE_PREFIX_PATH}" CACHE STRING "" FORCE)

# Load the full zstd CONFIG package before LLVM: LLVM's Findzstd defines only
# zstd::libzstd_shared/_static, which makes Arrow's later zstdTargets.cmake
# include fail with "some (but not all) targets already defined".
find_package(zstd CONFIG QUIET)

# ---------------------------------------------------------------------------
# LLVM 20 (shared with lingo-db and JIT)
# ---------------------------------------------------------------------------
find_package(LLVM 20.1 CONFIG
        HINTS /usr/lib/llvm-20/lib/cmake/llvm
        REQUIRED)

if (LLVM_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
    # Ensure LLVM's lib directory appears in RUNPATH even when CMake treats it
    # as an implicit link directory (happens with some clang++-20 installations).
    list(APPEND CMAKE_BUILD_RPATH "${LLVM_LIBRARY_DIR}")
    llvm_map_components_to_libnames(LLVM_LIBS
            core orcjit native passes analysis support)
    set(HAVE_LLVM TRUE)

    # aqp_jit_lib: middleware-side LLVM compiler (IR → machine code)
    # Requires C++17 because LLVM headers use C++17 features.
    add_library(aqp_jit_lib STATIC
            src/jit/ir_to_llvm.cpp
            src/jit/aqp_jit_runtime.cpp
            src/jit/aqp_jit_hashtable.cpp
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

    # TPDE fast-compile backend (--fast-compile=tpde).
    # Built as an ExternalProject with clang-20 (tpde needs C++20 <format>,
    # -fno-rtti, -fno-exceptions which conflict with AQPHub's own code).
    # The version script (tpde_llvm.map) exports only tpde_llvm:: symbols —
    # lingo-db's librunner.a embeds its own tpde core copy, so internals
    # must stay hidden.
    option(AQP_ENABLE_TPDE "Build TPDE fast-compile backend" ON)
    set(AQP_TPDE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tpde)

    if (AQP_ENABLE_TPDE AND EXISTS ${AQP_TPDE_SOURCE_DIR}/CMakeLists.txt)
        include(ExternalProject)

        set(AQP_TPDE_PREFIX ${CMAKE_BINARY_DIR}/tpde_ep)
        set(AQP_TPDE_BUILD  ${AQP_TPDE_PREFIX}/build)
        set(AQP_TPDE_BUNDLE ${AQP_TPDE_BUILD}/libtpde_bundle.so)
        set(AQP_TPDE_MAP    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tpde_llvm.map)

        find_program(AQP_CLANG_20   clang-20   REQUIRED)
        find_program(AQP_CLANGXX_20 clang++-20 REQUIRED)

        ExternalProject_Add(tpde_ep
            PREFIX          ${AQP_TPDE_PREFIX}
            SOURCE_DIR      ${AQP_TPDE_SOURCE_DIR}
            BINARY_DIR      ${AQP_TPDE_BUILD}
            INSTALL_COMMAND ""
            # tpde_encodegen only initializes LLVM targets for the host
            # arch (#ifdef __x86_64__ / __aarch64__), so cross-target
            # encodegen fails. Patch the guards out so both are initialized.
            # Skip if already applied (idempotent for rebuilds).
            PATCH_COMMAND sh -c "git -C ${AQP_TPDE_SOURCE_DIR} apply --reverse --check ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tpde-encodegen-init-all-targets.patch 2>/dev/null || git -C ${AQP_TPDE_SOURCE_DIR} apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tpde-encodegen-init-all-targets.patch"
            CMAKE_ARGS
                -DCMAKE_C_COMPILER=${AQP_CLANG_20}
                -DCMAKE_CXX_COMPILER=${AQP_CLANGXX_20}
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                -DTPDE_INCLUDE_TESTS=OFF
                -DTPDE_BUILD_DOCS=OFF
                -DTPDE_LOGGING=OFF
                -DLLVM_DIR=${LLVM_DIR}
            BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target tpde_llvm -- -j${AQP_TPDE_PARALLEL_JOBS}
            BUILD_BYPRODUCTS
                ${AQP_TPDE_BUILD}/tpde-llvm/libtpde_llvm.a
                ${AQP_TPDE_BUILD}/tpde/libtpde.a
                ${AQP_TPDE_BUILD}/tpde/deps/fadec/libfadec.a
                ${AQP_TPDE_BUILD}/tpde/deps/disarm/libdisarm64.a
                ${AQP_TPDE_BUNDLE}
        )

        # Link the static archives into a single shared lib with hidden internals.
        ExternalProject_Add_Step(tpde_ep bundle
            DEPENDEES build
            BYPRODUCTS ${AQP_TPDE_BUNDLE}
            COMMAND ${AQP_CLANGXX_20} -shared
                -o ${AQP_TPDE_BUNDLE}
                -Wl,-soname,libtpde_bundle.so
                -Wl,--whole-archive ${AQP_TPDE_BUILD}/tpde-llvm/libtpde_llvm.a
                -Wl,--no-whole-archive ${AQP_TPDE_BUILD}/tpde/libtpde.a
                    ${AQP_TPDE_BUILD}/tpde/deps/fadec/libfadec.a
                    ${AQP_TPDE_BUILD}/tpde/deps/disarm/libdisarm64.a
                -Wl,--version-script=${AQP_TPDE_MAP}
                -L${LLVM_LIBRARY_DIR} -lLLVM -Wl,-rpath,${LLVM_LIBRARY_DIR}
            COMMENT "Bundling libtpde_bundle.so"
        )

        add_library(tpde_bundle SHARED IMPORTED)
        set_target_properties(tpde_bundle PROPERTIES
            IMPORTED_LOCATION ${AQP_TPDE_BUNDLE}
            IMPORTED_SONAME libtpde_bundle.so)
        add_dependencies(tpde_bundle tpde_ep)

        target_include_directories(aqp_jit_lib PRIVATE
                ${AQP_TPDE_SOURCE_DIR}/tpde-llvm/include)
        find_library(TINFO_LIB tinfo PATHS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu NO_DEFAULT_PATH)
        if (NOT TINFO_LIB)
            find_library(TINFO_LIB tinfo)
        endif ()
        target_link_libraries(aqp_jit_lib PUBLIC tpde_bundle ${TINFO_LIB})
        target_compile_definitions(aqp_jit_lib PRIVATE AQP_HAVE_TPDE)
        message(STATUS "AQP fast-compile TPDE backend: ENABLED (ExternalProject)")
    else ()
        message(STATUS "AQP fast-compile TPDE backend: disabled"
                " (set AQP_ENABLE_TPDE=ON and ensure third_party/tpde exists)")
    endif ()
else ()
    set(HAVE_LLVM FALSE)
    message(STATUS "LLVM not found — AQP JIT disabled")
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build split library (common for both DuckDB and PostgreSQL)
add_library(split_lib STATIC
        src/util/param_config.cpp
        src/util/thread_pool.cpp
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

# Utility library (shared across adapters — compiled once to avoid duplicate symbols)
add_library(util_lib STATIC src/util/util.cpp)
target_include_directories(util_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Query-JIT library (--jit-level=query): IR analysis, runtime, scheduler.
# Pure C++ over the IR — no DuckDB/LLVM dependency.
add_library(qjit_lib STATIC
        src/qjit/query_jit_steps.cpp
        src/qjit/query_jit_runtime.cpp
        src/qjit/query_jit_scheduler.cpp
        src/qjit/query_jit_executor.cpp
)
target_include_directories(qjit_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/IR_SQL_Converter/inc
)
# QjitWorkerPool bumps middleware::g_bg_active_threads (thread_pool.cpp,
# owned by split_lib) so OpenMP paths throttle while qjit workers run.
target_link_libraries(qjit_lib PUBLIC split_lib)

# Storage plan library (flat column arrays, CSR indexes)
# Built conditionally with DuckDB below.
set(STORAGE_LIB_SOURCES
    src/storage/flat_table.cpp
    src/storage/storage_plan.cpp
    src/storage/csr_index.cpp
    src/storage/dimension_cache.cpp
    src/storage/sorted_index.cpp
    src/storage/inverted_index.cpp
)

# Kernel library (sub-query executor using storage structures)
# Built conditionally with DuckDB below.
set(KERNEL_LIB_SOURCES
    src/kernel/sub_query_plan.cpp
    src/kernel/pipeline_kernel.cpp
)

# DuckDB
set(DUCKDB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../duckdb" CACHE PATH "DuckDB source path")
set(HAVE_DUCKDB FALSE)
if (EXISTS ${DUCKDB_ROOT})
    message(STATUS "Found DuckDB source at: ${DUCKDB_ROOT}")

    include_directories(${DUCKDB_ROOT}/src/include)

    find_library(DUCKDB_LIB
            NAMES duckdb libduckdb
            PATHS ${DUCKDB_ROOT}/build/release/src ${DUCKDB_ROOT}/build/reldebug/src ${DUCKDB_ROOT}/build/debug/src
            NO_DEFAULT_PATH
    )

    if (DUCKDB_LIB)
        set(HAVE_DUCKDB TRUE)
        add_library(duckdb_adapter STATIC src/adapters/duckdb_adapter.cpp)
        target_include_directories(duckdb_adapter PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${DUCKDB_ROOT}/src/include
        )

        # Define HAVE_DUCKDB preprocessor macro for C++ code
        set_property(TARGET duckdb_adapter PROPERTY CXX_STANDARD 17)
        target_compile_definitions(duckdb_adapter PUBLIC HAVE_DUCKDB)
        target_compile_definitions(split_lib PUBLIC HAVE_DUCKDB)

        # Storage plan library (needs DuckDB headers)
        add_library(storage_lib STATIC ${STORAGE_LIB_SOURCES})
        target_include_directories(storage_lib PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${DUCKDB_ROOT}/src/include
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/IR_SQL_Converter/inc
        )
        target_compile_definitions(storage_lib PUBLIC HAVE_DUCKDB)
        target_link_libraries(storage_lib PUBLIC ${DUCKDB_LIB})

        # Kernel library (needs storage_lib + IR_SQL_Converter)
        add_library(kernel_lib STATIC ${KERNEL_LIB_SOURCES})
        target_include_directories(kernel_lib PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${DUCKDB_ROOT}/src/include
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/IR_SQL_Converter/inc
        )
        target_compile_definitions(kernel_lib PUBLIC HAVE_DUCKDB)
        target_link_libraries(kernel_lib PUBLIC storage_lib ${DUCKDB_LIB})

        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-mavx2 HAS_AVX2)
        if (HAS_AVX2)
            target_compile_options(kernel_lib PRIVATE -mavx2)
            target_compile_definitions(kernel_lib PRIVATE HAS_AVX2)
            message(STATUS "kernel_lib AVX2: ENABLED")
        endif()

        find_package(OpenMP)
        if (OpenMP_CXX_FOUND)
            target_link_libraries(kernel_lib PUBLIC OpenMP::OpenMP_CXX)
            target_compile_definitions(kernel_lib PUBLIC HAVE_OPENMP)
            message(STATUS "kernel_lib OpenMP: ENABLED")
        endif()

        target_link_libraries(duckdb_adapter PUBLIC ${DUCKDB_LIB} split_lib storage_lib kernel_lib util_lib qjit_lib)
        if (HAVE_LLVM)
            target_link_libraries(duckdb_adapter PUBLIC aqp_jit_lib)
            target_compile_definitions(duckdb_adapter PUBLIC HAVE_LLVM)
            target_compile_definitions(split_lib PUBLIC HAVE_LLVM)
        endif ()

        add_executable(test_duckdb examples/test_duckdb.cpp)
        set_target_properties(test_duckdb PROPERTIES ENABLE_EXPORTS ON)
        target_link_libraries(test_duckdb PRIVATE duckdb_adapter split_lib)

        message(STATUS "DuckDB adapter: ENABLED")
    else ()
        message(WARNING "DUCKDB_LIB not found")
    endif ()
else ()
    message(WARNING "DuckDB source not found")
endif ()

# PostgreSQL
set(POSTGRES_ROOT "/home/pei/Project/project_bins" CACHE PATH "PostgreSQL lib/inc path")
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
        add_library(postgres_adapter STATIC src/adapters/postgres_adapter.cpp)
        target_include_directories(postgres_adapter PUBLIC
                ${POSTGRES_ROOT}/include
        )

        # Define HAVE_POSTGRES preprocessor macro for C++ code
        target_compile_definitions(postgres_adapter PUBLIC HAVE_POSTGRES)
        target_compile_definitions(split_lib PUBLIC HAVE_POSTGRES)

        target_link_libraries(postgres_adapter PUBLIC ${LIBPQ_LIB} split_lib util_lib)

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
        add_library(mariadb_adapter STATIC src/adapters/mariadb_adapter.cpp)
        target_include_directories(mariadb_adapter PUBLIC
                ${MARIADB_ROOT}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        target_compile_definitions(mariadb_adapter PUBLIC HAVE_MARIADB)
        target_compile_definitions(split_lib PUBLIC HAVE_MARIADB)

        target_link_libraries(mariadb_adapter PUBLIC ${LIBMARIADB_LIB} split_lib util_lib)

        message(STATUS "MariaDB adapter: ENABLED")
    else ()
        message(WARNING "LIBMARIADB_LIB not found at ${MARIADB_ROOT}/lib")
    endif ()
else ()
    message(WARNING "MariaDB not found at ${MARIADB_ROOT}")
endif ()

# LingoDB
set(LINGODB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../lingo-db" CACHE PATH "LingoDB source path")
set(LINGODB_BUILD "${LINGODB_ROOT}/build/lingodb-release" CACHE PATH "LingoDB build path")
set(HAVE_LINGODB FALSE)
if (EXISTS ${LINGODB_ROOT} AND EXISTS ${LINGODB_BUILD})
    message(STATUS "Found LingoDB at: ${LINGODB_ROOT}")

    # MLIR (required by lingo-db; LLVM already found at top level)
    set(MLIR_DIR "${LLVM_DIR}/../mlir")
    find_package(MLIR CONFIG HINTS ${MLIR_DIR} QUIET)

    if (HAVE_LLVM AND MLIR_FOUND)
        set(HAVE_LINGODB TRUE)

        # Collect all lingo-db static libraries
        set(LINGODB_LIBS
                ${LINGODB_BUILD}/src/execution/librunner.a
                ${LINGODB_BUILD}/src/compiler/frontend/libMLIRSQLFrontend.a
                ${LINGODB_BUILD}/src/compiler/frontend/libMLIRSQLFrontendGen.a
                ${LINGODB_BUILD}/src/catalog/libcatalog.a
                ${LINGODB_BUILD}/src/catalog/libmlirTypes.a
                ${LINGODB_BUILD}/src/utility/libutility.a
                ${LINGODB_BUILD}/src/scheduler/libscheduler.a
                ${LINGODB_BUILD}/lib/libMLIRCompilerHelper.a
                ${LINGODB_BUILD}/lib/libMLIRRelAlg.a
                ${LINGODB_BUILD}/lib/libMLIRTupleStream.a
                ${LINGODB_BUILD}/lib/libMLIRSubOperator.a
                ${LINGODB_BUILD}/lib/libMLIRUtil.a
                ${LINGODB_BUILD}/lib/libMLIRUtilToLLVM.a
                ${LINGODB_BUILD}/lib/libMLIRDB.a
                ${LINGODB_BUILD}/lib/libMLIRArrow.a
                ${LINGODB_BUILD}/lib/libMLIRArrowToStd.a
                ${LINGODB_BUILD}/lib/libMLIRDBToArrowStd.a
                ${LINGODB_BUILD}/lib/libMLIRelAlgToSubOp.a
                ${LINGODB_BUILD}/lib/libMLIRSubOpToControlFlow.a
                ${LINGODB_BUILD}/libruntime_funcs_ptr.a
                ${LINGODB_BUILD}/vendored/mimalloc/libmimalloc.a
        )
        # TPDE baseline backend (ENABLE_BASELINE_BACKEND=ON in lingo-db)
        if (EXISTS ${LINGODB_BUILD}/vendored/tpde/tpde/libtpde.a)
            list(APPEND LINGODB_LIBS
                    ${LINGODB_BUILD}/vendored/tpde/tpde/libtpde.a
                    ${LINGODB_BUILD}/vendored/tpde/tpde/deps/spdlog/libspdlog.a
                    ${LINGODB_BUILD}/vendored/tpde/tpde/deps/fadec/libfadec.a
                    ${LINGODB_BUILD}/vendored/tpde/tpde/deps/disarm/libdisarm64.a
            )
        endif ()

        # Newer lingo-db builds include the tpde baseline backend in
        # librunner.a; link its vendored archives when present.
        foreach (tpde_lib
                ${LINGODB_BUILD}/vendored/tpde/tpde/libtpde.a
                ${LINGODB_BUILD}/vendored/tpde/tpde/deps/spdlog/libspdlog.a
                ${LINGODB_BUILD}/vendored/tpde/tpde/deps/fadec/libfadec.a
                ${LINGODB_BUILD}/vendored/tpde/tpde/deps/disarm/libdisarm64.a)
            if (EXISTS ${tpde_lib})
                list(APPEND LINGODB_LIBS ${tpde_lib})
            endif ()
        endforeach ()

        # Get MLIR and LLVM libraries
        get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
        get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
        get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
        get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

        add_library(lingodb_adapter STATIC
                src/adapters/lingodb_adapter.cpp
                src/adapters/lingodb_runtime_adapter.cpp)
        target_include_directories(lingodb_adapter PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${LINGODB_ROOT}/include
                ${LINGODB_ROOT}/vendored
                ${LINGODB_BUILD}/include
                ${LLVM_INCLUDE_DIRS}
                ${MLIR_INCLUDE_DIRS}
        )
        target_compile_definitions(lingodb_adapter PUBLIC HAVE_LINGODB)
        target_compile_definitions(split_lib PUBLIC HAVE_LINGODB)

        # LingoDB requires C++20
        set_property(TARGET lingodb_adapter PROPERTY CXX_STANDARD 20)

        # Link lingo-db libraries, MLIR, LLVM, Arrow
        set(zstd_DIR "/usr/lib/x86_64-linux-gnu/cmake/zstd" CACHE PATH "" FORCE)
        # LLVM's Findzstd already defined zstd::libzstd_shared/_static.
        # zstdTargets.cmake (pulled in by Arrow) errors if only part of its
        # export set exists, so pre-define the missing zstd::libzstd.
        if (TARGET zstd::libzstd_shared AND NOT TARGET zstd::libzstd)
            add_library(zstd::libzstd INTERFACE IMPORTED)
            target_link_libraries(zstd::libzstd INTERFACE zstd::libzstd_shared)
        endif ()
        find_package(Arrow REQUIRED)
        set(Boost_USE_STATIC_LIBS ON)
        find_package(Boost 1.83.0 REQUIRED COMPONENTS context)
        # Collect runtime object files (OBJECT library in lingo-db)
        file(GLOB_RECURSE LINGODB_RUNTIME_OBJS
                "${LINGODB_BUILD}/src/runtime/CMakeFiles/runtime.dir/*.o")
        target_link_libraries(lingodb_adapter PUBLIC
                -Wl,--start-group
                ${LINGODB_LIBS}
                ${LINGODB_RUNTIME_OBJS}
                ${MLIR_DIALECT_LIBS}
                ${MLIR_CONVERSION_LIBS}
                ${MLIR_TRANSLATION_LIBS}
                ${MLIR_EXTENSION_LIBS}
                -Wl,--end-group
                Arrow::arrow_shared
                /usr/lib/x86_64-linux-gnu/libarrow_compute.so
                Boost::context
                split_lib
                util_lib
                -lpthread -lrt -latomic -lcurl
        )

        message(STATUS "LingoDB adapter: ENABLED (LLVM ${LLVM_PACKAGE_VERSION})")
    else ()
        if (NOT HAVE_LLVM)
            message(STATUS "LLVM not found — LingoDB adapter disabled")
        else ()
            message(STATUS "MLIR not found — LingoDB adapter disabled")
        endif ()
    endif ()
else ()
    message(STATUS "LingoDB not found at ${LINGODB_ROOT}")
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

    if (LIBPG_QUERY_LIB AND HAVE_LINGODB)
        target_link_libraries(lingodb_adapter PUBLIC ${LIBPG_QUERY_LIB})
        message(STATUS "libpg_query: ENABLED (LingoDB)")
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
    if (HAVE_DUCKDB OR HAVE_POSTGRES OR HAVE_MARIADB OR HAVE_LINGODB)
        add_executable(aqp_middleware src/aqp_middleware.cpp)
        set_target_properties(aqp_middleware PROPERTIES ENABLE_EXPORTS ON)
        target_link_libraries(aqp_middleware PRIVATE
                split_lib
                IR_SQL_Converter_static
        )
        if (HAVE_DUCKDB)
            target_link_libraries(aqp_middleware PRIVATE storage_lib kernel_lib)
        endif ()

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
        if (HAVE_LINGODB)
            target_link_libraries(aqp_middleware PRIVATE lingodb_adapter)
        endif ()

        message(STATUS "aqp_middleware: ENABLED")
    else ()
        message(WARNING "aqp_middleware: DISABLED (no database adapter available)")
    endif ()

    # Unit tests (gtest)
    find_package(GTest QUIET)
    if (GTest_FOUND AND HAVE_DUCKDB)
        enable_testing()
        add_executable(test_storage unit_test/test_storage.cpp)
        target_link_libraries(test_storage PRIVATE
            storage_lib kernel_lib split_lib IR_SQL_Converter_static GTest::gtest_main)
        set_property(TARGET test_storage PROPERTY CXX_STANDARD 17)
        add_test(NAME StorageTests COMMAND test_storage)
        message(STATUS "test_storage: ENABLED (gtest)")
        # qjit runtime/scheduler unit tests (Phase 1 gate). split_lib supplies
        # middleware::g_bg_active_threads (thread_pool.cpp).
        add_executable(test_qjit unit_test/test_qjit.cpp)
        target_link_libraries(test_qjit PRIVATE
            qjit_lib split_lib GTest::gtest_main)
        set_property(TARGET test_qjit PROPERTY CXX_STANDARD 17)
        add_test(NAME QjitTests COMMAND test_qjit)
        message(STATUS "test_qjit: ENABLED (gtest)")
    elseif (NOT GTest_FOUND)
        message(STATUS "gtest not found — unit tests disabled")
    endif ()
else ()
    message(WARNING "IR_SQL_Converter source not found at ${IR_CONVERTER_ROOT}")
endif ()
