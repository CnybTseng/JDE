cmake_minimum_required(VERSION 3.5.1)

project(jde)

find_package(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30;-G)
set(TENSORRT_DIR /home/royzon/DL_frame/TensorRT-7.0.0.11)

include_directories(
    ${TENSORRT_DIR}/include
    /usr/local/cuda/include
    /usr/include/aarch64-linux-gnu
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/src/
)

link_directories(
    ${TENSORRT_DIR}/lib
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/aarch64-linux/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib64
    ${CMAKE_CURRENT_LIST_DIR}/build
)

add_definitions(-std=c++11 -O3)

file(GLOB_RECURSE libsrcs ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
list(FILTER libsrcs EXCLUDE REGEX "mot-test")
list(FILTER libsrcs EXCLUDE REGEX "main")

file(GLOB_RECURSE libgpusrcs ${CMAKE_CURRENT_LIST_DIR}/src/*.cu)
cuda_add_library(jdecoder STATIC ${libgpusrcs})
target_link_libraries(jdecoder -lnvinfer -lcudart)

add_library(mot SHARED ${libsrcs})
target_link_libraries(
    mot
    -ljdecoder
    -lnvinfer
    -lnvinfer_plugin
    -lnvonnxparser
    -lcudart
    -lcublas
    -lcudnn
    -lpthread
    -ldl
    ${OpenCV_LIBS}
)

add_executable(jde ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp ${libsrcs})
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
    ${OpenCV_LIBS}
)

add_executable(mot-test ${CMAKE_CURRENT_LIST_DIR}/src/mot-test.cpp)
add_dependencies(mot-test mot)
target_link_libraries(
    mot-test
    -lmot
    ${OpenCV_LIBS}
)
