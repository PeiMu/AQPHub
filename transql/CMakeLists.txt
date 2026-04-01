if (NOT HAVE_DUCKDB)
    message(WARNING "transql: DISABLED (requires DuckDB)")
    return()
endif ()

add_library(transql_lib STATIC
    src/tensor_dag.cpp
    src/sql_template.cpp
    src/dag_to_tree.cpp
    src/transql_runner.cpp
    src/transql_postopt.cpp
)

target_include_directories(transql_lib PUBLIC
    include
    ${CMAKE_SOURCE_DIR}/include               # db_adapter.h, QueryResult
    ${CMAKE_SOURCE_DIR}/third_party/IR_SQL_Converter/inc  # simplest_ir.h, cpp_interface.h
)

target_link_libraries(transql_lib PUBLIC
    IR_SQL_Converter_static
    duckdb_adapter
)

add_executable(test_transql ${CMAKE_SOURCE_DIR}/examples/test_transql.cpp)
target_link_libraries(test_transql PRIVATE transql_lib duckdb_adapter)

message(STATUS "transql: ENABLED")
