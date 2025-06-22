set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(indexing test_index_build.cpp
        ../include/mrqg/mrqg_query.hpp)