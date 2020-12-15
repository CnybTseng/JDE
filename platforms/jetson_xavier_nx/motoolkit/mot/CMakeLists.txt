cmake_minimum_required(VERSION 3.5.1)

project(jde)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_72;code=sm_72)

include_directories(
    /usr/local/cuda/include
    /usr/local/cuda-10.2/samples/common/inc
    /usr/include/aarch64-linux-gnu
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.0/include/opencv4
    ./src
)

link_directories(
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/aarch64-linux/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib64
    /usr/local/lib
    /usr/lib/aarch64-linux-gnu/gstreamer-1.0
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.0/lib
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.0/lib/opencv4/3rdparty
)

add_definitions(-std=c++11 -O3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fopenmp -g")

file(GLOB_RECURSE libsrcs src/*.cpp)

add_subdirectory(src)
add_subdirectory(test)