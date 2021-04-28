cmake_minimum_required(VERSION 3.5.1)

project(track_merge)

set(Eigen3_DIR /usr/lib/cmake/eigen3/)
find_package(Eigen3 REQUIRED)
set(OpenCV_DIR ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.1/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include_directories(
    $ENV{JAVA_HOME}/include
    $ENV{JAVA_HOME}/include/linux
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/jsoncpp-1.9.4/include
    ${CMAKE_CURRENT_LIST_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}/../mot/src
)

link_directories(
    /usr/local/lib64
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/jsoncpp-1.9.4/lib
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g")

file(GLOB_RECURSE libsrcs 
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../mot/src/lapjv.cpp
)

add_library(track_merge SHARED ${libsrcs})

target_link_libraries(
    track_merge
    libjsoncpp_static.a
    ${OpenCV_LIBS}
)

install(TARGETS track_merge
    LIBRARY DESTINATION lib
)