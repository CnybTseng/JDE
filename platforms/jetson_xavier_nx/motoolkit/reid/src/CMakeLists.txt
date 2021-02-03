add_library(reid SHARED ${LIBSRCS})
target_link_libraries(
    reid
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
    libopencv_dnn.a
    libopencv_video.a
    libopencv_imgcodecs.a
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