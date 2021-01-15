file(GLOB_RECURSE libgpusrcs ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu)

cuda_add_library(motplugin STATIC ${libgpusrcs})
target_link_libraries(motplugin -lnvinfer -lnvinfer_plugin -lcudart)

add_library(mot SHARED ${libsrcs})
add_dependencies(mot motplugin)
target_link_libraries(
    mot
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
    ${CMAKE_CURRENT_BINARY_DIR}/libmotplugin.a
    -lnvinfer
    -lnvonnxparser
    -lcudart
    -lcublas
    -lcudnn
    -lpthread
    -ldl
    -lz
)

set_target_properties(mot PROPERTIES PUBLIC_HEADER "mot.h")
install(TARGETS mot
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)