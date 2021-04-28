link_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

add_executable(jde-test jde-test.cpp ${libsrcs})
add_dependencies(jde-test motplugin)
target_link_libraries(
    jde-test
    libmotplugin.a
    ${OpenCV_LIBS}
    -lnvinfer
    -lnvonnxparser
    -lcudart
    -lcublas
    -lcudnn
)

add_executable(mot-test mot-test.cpp)
add_dependencies(mot-test mot)
target_link_libraries(
    mot-test
    ${OpenCV_LIBS}
    -lmot
)

add_executable(jdecoderv2-test jdecoderv2-test.cpp)
add_dependencies(jdecoderv2-test motplugin)
target_link_libraries(
    jdecoderv2-test
    libmotplugin.a
    ${OpenCV_LIBS}
    -lnvinfer
    -lnvonnxparser
    -lcudart
    -lcublas
    -lcudnn
)

add_executable(preprocess-test preprocess-test.cpp ../src/utils.cpp)
target_link_libraries(
    preprocess-test
    ${OpenCV_LIBS}
    -lnvinfer
    -lnvonnxparser
    -lcudart
    -lcublas
    -lcudnn
)

install(TARGETS jde-test mot-test jdecoderv2-test preprocess-test
    RUNTIME DESTINATION bin
)