cmake_minimum_required(VERSION 3.5.1)

project(jde)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_72;code=sm_72)

include_directories(
    /usr/local/cuda/include
    /usr/local/cuda-10.2/samples/common/inc
    /usr/include/aarch64-linux-gnu
    /usr/include/opencv4
    /usr/local/include/opencv4
    /usr/local/opencv-4.1.1/include/opencv4
    ./src
)

link_directories(
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/aarch64-linux/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib64
    /usr/local/lib
    /usr/local/opencv-4.1.1/lib
    ${CMAKE_CURRENT_LIST_DIR}/build
)

add_definitions(-std=c++11 -O3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fopenmp -g")

file(GLOB_RECURSE libsrcs src/*.cpp)
list(FILTER libsrcs EXCLUDE REGEX "mot-test")
list(FILTER libsrcs EXCLUDE REGEX "jde-test")
list(FILTER libsrcs EXCLUDE REGEX "jdecoderv2-test")
list(FILTER libsrcs EXCLUDE REGEX "preprocess-test")

file(GLOB_RECURSE libgpusrcs src/*.cu)
cuda_add_library(jdecoder SHARED ${libgpusrcs})
target_link_libraries(jdecoder -lnvinfer -lnvinfer_plugin -lcudart)

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
    -lopencv_core
    -lopencv_video
    -lopencv_imgcodecs
    -lopencv_imgproc
)

add_executable(jde-test src/jde-test.cpp ${libsrcs})
target_link_libraries(
    jde-test
    -ljdecoder
    -lnvinfer
    -lnvinfer_plugin
    -lnvonnxparser
    -lcudart
    -lcublas
    -lcudnn
    -lpthread
    -ldl
    -lopencv_core
    -lopencv_video
    -lopencv_imgcodecs
    -lopencv_imgproc
)

add_executable(mot-test src/mot-test.cpp)
add_dependencies(mot-test mot)
target_link_libraries(
    mot-test
    -lmot
    -lx264
    -lpthread
    -lopencv_core
    -lopencv_video
    -lopencv_videoio
    -lopencv_imgcodecs
    -lopencv_imgproc
)

add_executable(jdecoderv2-test src/jdecoderv2-test.cpp)
target_link_libraries(
    jdecoderv2-test
    -ljdecoder
    -lnvinfer
    -lcudart
    -lopencv_core
    -lopencv_video
    -lopencv_imgcodecs
    -lopencv_imgproc)

add_executable(preprocess-test src/preprocess-test.cpp src/utils.cpp)
target_link_libraries(
    preprocess-test
    -lnvinfer
    -lcudart
    -lopencv_core
    -lopencv_video
    -lopencv_imgcodecs
    -lopencv_imgproc)