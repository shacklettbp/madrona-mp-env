cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
cmake_policy(VERSION 3.20)

option(MADRONA_EXTERNAL_DIR "" "")
if (NOT MADRONA_EXTERNAL_DIR)
    message(FATAL_ERROR "MADRONA_EXTERNAL_DIR option must be set")
endif ()

include("${MADRONA_EXTERNAL_DIR}/cmake/madrona_init.cmake")

project(MadronaMPEnv LANGUAGES C CXX)

include(setup)
include(dependencies)

add_subdirectory(external)
add_subdirectory(src)
