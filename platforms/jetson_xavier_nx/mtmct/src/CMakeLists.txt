# Build shared library.
add_library(${PROJECT_NAME} SHARED ${LIBSRCS})
target_link_libraries(${PROJECT_NAME} -lpthread)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "mtmct.h")
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Build static library.
add_library(${PROJECT_NAME}-static STATIC ${LIBSRCS})
target_link_libraries(${PROJECT_NAME}-static -lpthread)

set_target_properties(${PROJECT_NAME}-static PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}-static
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)