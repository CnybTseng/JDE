link_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

add_executable(osnet-test osnet-test.cpp)
add_dependencies(osnet-test reid)
target_link_libraries(
    osnet-test
    ${OpenCV_LIBS}
    -lreid
)

install(TARGETS osnet-test
    RUNTIME DESTINATION bin
)