cmake_minimum_required(VERSION 2.6)

project(SVM)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(/usr/local/include/eigen3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(SVM SHARED ${PROJECT_SOURCE_DIR}/src/imageProcess.cpp ${PROJECT_SOURCE_DIR}/src/svmKernel.cpp )
target_link_libraries(SVM ${OpenCV_LIBS})

add_executable(demo demo.cpp)

target_link_libraries(demo SVM)

add_definitions(-O2 -pthread)


