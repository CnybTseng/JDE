add_library(reid SHARED ${LIBSRCS})
target_link_libraries(
    reid
    ${OpenCV_LIBS}
    -lnvinfer
    -lnvonnxparser
    -lcudart
    -lcublas
    -lcudnn
    -lpthread
    -ldl
    -lz
)

set_target_properties(reid PROPERTIES PUBLIC_HEADER "osnet.h")
install(TARGETS reid
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

add_library(reid-static STATIC ${LIBSRCS})
target_link_libraries(
    reid-static
    ${OpenCV_LIBS}
    -lnvinfer
    -lnvonnxparser
    -lcudart
    -lcublas
    -lcudnn
    -lpthread
    -ldl
    -lz
)

set_target_properties(reid-static PROPERTIES OUTPUT_NAME reid)
install(TARGETS reid-static
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)