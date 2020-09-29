cmake_minimum_required(VERSION 3.5.1)

project(jde)

include_directories(
    /usr/local/cuda/include
    /usr/include/aarch64-linux-gnu
    /usr/include/opencv4
    ./src
)

link_directories(
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/aarch64-linux/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib64
    /home/sihan/Documents/jetson_xavier_nx/build
)

add_definitions(-std=c++11 -O3)

file(GLOB_RECURSE libsrcs src/*.cpp)
list(FILTER libsrcs EXCLUDE REGEX "mot-test")
list(FILTER libsrcs EXCLUDE REGEX "main")

add_library(mot SHARED ${libsrcs})
target_link_libraries(
    mot
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

add_executable(jde src/main.cpp ${libsrcs})
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
    -lopencv_core
    -lopencv_video
    -lopencv_imgcodecs
    -lopencv_imgproc
)