cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(fcopy)
set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
enable_language(Fortran)
file(GLOB fcopy_SRC "*.f90")
add_library(fcopy SHARED  ${fcopy_SRC})
set_target_properties(fcopy PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
