cmake_minimum_required(VERSION 3.5.1)

project(croslndet)

include_directories(
    $ENV{JAVA_HOME}/include
    $ENV{JAVA_HOME}/include/linux
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/jsoncpp-1.9.4/include
    ${CMAKE_CURRENT_LIST_DIR}/src
)

link_directories(
    /usr/local/lib64
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/jsoncpp-1.9.4/lib
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g")

file(GLOB_RECURSE LIBSRCS 
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(croslndet SHARED ${LIBSRCS})

target_link_libraries(
    croslndet
    libjsoncpp_static.a
)

install(TARGETS croslndet
    LIBRARY DESTINATION lib
)