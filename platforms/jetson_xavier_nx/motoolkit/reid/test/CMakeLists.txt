link_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

add_executable(osnet-test osnet-test.cpp)
add_dependencies(osnet-test reid)
target_link_libraries(
    osnet-test
    libopencv_video.a
    libopencv_videoio.a
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
    -lreid
    -lx264
    -lpthread
    -ldl
    -lpng
    -lz
    -lgstreamer-1.0
    -lgobject-2.0
    -lgstsdp-1.0
    -lglib-2.0
    -ldc1394
    -ljpeg
    -lpng
    -lpng16
    -ltiff
    -lgstapp
    -lgstapp-1.0
    -lgstriff-1.0
    -lgstencoding
    -lgstpbutils-1.0
)

install(TARGETS osnet-test
    RUNTIME DESTINATION bin
)