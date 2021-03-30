link_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

add_executable(test-tsque test-tsque.cpp)
target_link_libraries(test-tsque -lpthread)

add_executable(test-mtmct test-mtmct.cpp)
add_dependencies(test-mtmct ${PROJECT_NAME})
target_link_libraries(test-mtmct ${OpenCV_LIBS} -lmtmct -lpthread)

install(TARGETS test-tsque test-mtmct
    RUNTIME DESTINATION bin
)