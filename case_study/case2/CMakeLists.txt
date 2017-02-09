cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(case)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (case case.cpp)
target_link_libraries (case ${PCL_LIBRARIES})
