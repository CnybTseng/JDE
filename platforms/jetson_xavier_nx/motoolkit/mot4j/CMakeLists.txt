cmake_minimum_required(VERSION 3.5.1)

project(mot4j)

set(Eigen3_DIR /usr/lib/cmake/eigen3/)
find_package(Eigen3 REQUIRED)
set(OpenCV_DIR ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.1/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include_directories(
    $ENV{JAVA_HOME}/include
    $ENV{JAVA_HOME}/include/linux
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/jsoncpp-1.9.4/include
    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/mot/include
    ${CMAKE_CURRENT_LIST_DIR}/src
)

link_directories(
    /usr/local/lib64
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/jsoncpp-1.9.4/lib
    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/mot/lib
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g")

file(GLOB_RECURSE libsrcs 
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(mot4j SHARED ${libsrcs})

target_link_libraries(
    mot4j
    -lmot
    libjsoncpp_static.a
    ${OpenCV_LIBS}
    -lpthread
)

install(TARGETS mot4j
    LIBRARY DESTINATION lib
)