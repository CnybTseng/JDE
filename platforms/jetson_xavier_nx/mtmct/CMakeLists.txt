cmake_minimum_required(VERSION 3.5.1)
project(mtmct)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fPIC -fopenmp -g")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-g;-G;-gencode;arch=compute_72;code=sm_72)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(Eigen3_DIR /usr/lib/cmake/eigen3/)
find_package(Eigen3 REQUIRED)
set(OpenCV_DIR ${CMAKE_CURRENT_LIST_DIR}/../motoolkit/3rdparty/opencv-4.5.0/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include_directories(
    /usr/local/cuda/include
    /usr/local/cuda-10.2/samples/common/inc
    /usr/include/aarch64-linux-gnu
    ${CMAKE_CURRENT_LIST_DIR}/src
)

link_directories(
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/aarch64-linux/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib64
    /usr/local/lib
    /usr/lib/aarch64-linux-gnu/gstreamer-1.0
)

file(GLOB_RECURSE LIBSRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)