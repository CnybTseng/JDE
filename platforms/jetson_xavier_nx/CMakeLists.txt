cmake_minimum_required(VERSION 3.5.1)

project(jde)

include_directories(
    /usr/local/cuda/include
    /usr/include/aarch64-linux-gnu
    ./src
)

link_directories(
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/aarch64-linux/lib
    /usr/lib/aarch64-linux-gnu
)

add_definitions(-std=c++11 -O3)

FILE(GLOB_RECURSE cpps src/*.cpp)
add_executable(jde ${cpps})
target_link_libraries(
    jde
    -lnvinfer
    -lnvinfer_plugin
    -lnvonnxparser
    -lcudart
    -lcublas
    -lcudnn
    -lpthread
    -ldl
)