# Copyright (c) Sihan Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(jde)

# Compile options
add_compile_options(-std=c++11 -DDISABLE_NEON=0 -DONLY_DETECTION=0)

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../outputs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "../../outputs")
set(CMAKE_INSTALL_PREFIX "../../../run")
set(CMAKE_OUTPUT_DIR "out")

set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# Header path
include_directories(
../
$ENV{DDK_PATH}/include/inc/
$ENV{DDK_PATH}/include/inc/dvpp
$ENV{DDK_PATH}/include/inc/hiaiengine
$ENV{DDK_PATH}/include/third_party/protobuf/include
$ENV{DDK_PATH}/include/third_party/cereal/include
$ENV{DDK_PATH}/include/third_party/opencv/include
$ENV{DDK_PATH}/include/libc_sec/include
)

if(type STREQUAL host)
    if(target STREQUAL "RC")
        add_compile_options(-DIS_RC)
    endif()

    # add host library path
    link_directories($ENV{NPU_HOST_LIB})

    # add custom host library
    add_library(JDEHost SHARED
        input_engine.cpp
        jdecoder_engine.cpp
        lapjv.cpp
        trajectory.cpp
        jdetracker.cpp
        associate_engine.cpp
        mot.cpp
        datastore.cpp)
    
    if(target STREQUAL "RC")
        target_link_libraries(JDEHost matrixdaemon pthread c_sec dl rt hiai_common opencv_world Dvpp_vpc Dvpp_api)
    else()
        target_link_libraries(JDEHost matrix pthread c_sec dl rt)
    endif()

    # add application
    add_executable(main main.cpp)

    if(target STREQUAL "RC")
        target_link_libraries(main JDEHost Dvpp_vpc)
    else()
        target_link_libraries(main JDEHost)
    endif()

    install(TARGETS main JDEHost DESTINATION ${CMAKE_OUTPUT_DIR})
else()
    add_compile_options(-DCPU_ONLY)

    # add device library path
    link_directories($ENV{NPU_DEV_LIB})

    # add custom device library
    add_library(JDEngine SHARED jde_engine.cpp)

    if(target STREQUAL "RC")
        target_link_libraries(JDEngine idedaemon hiai_common c_sec)
    else()
        target_link_libraries(JDEngine idedaemon c_sec)
    endif()

    install(TARGETS JDEngine DESTINATION ${CMAKE_OUTPUT_DIR})
endif()

