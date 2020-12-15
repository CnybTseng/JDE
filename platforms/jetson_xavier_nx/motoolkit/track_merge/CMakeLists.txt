cmake_minimum_required(VERSION 3.5.1)

project(track_merge)

include_directories(
    /usr/local/jdk-15/include
    /usr/local/jdk-15/include/linux
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/jsoncpp-1.9.4/include
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.0/include/opencv4
    ${CMAKE_CURRENT_LIST_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}/../mot/src
)

link_directories(
    /usr/local/lib64
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/jsoncpp-1.9.4/lib
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.0/lib
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/opencv-4.5.0/lib/opencv4/3rdparty
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g")

file(GLOB_RECURSE libsrcs 
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../mot/src/lapjv.cpp
)

add_library(track_merge SHARED ${libsrcs})

target_link_libraries(
    track_merge
    libjsoncpp.a
    libopencv_video.a
    libopencv_imgproc.a
    libopencv_core.a
    libade.a
    libIlmImf.a
    libittnotify.a
    liblibopenjp2.a
    liblibprotobuf.a
    liblibwebp.a
    libquirc.a
    libtegra_hal.a
)

install(TARGETS track_merge
    LIBRARY DESTINATION lib
)