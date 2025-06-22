set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(index_mrqg test_mrqg_build.cpp
        ../include/space/matrix.h)

add_executable(search_mrgq test_mrqg_search.cpp)