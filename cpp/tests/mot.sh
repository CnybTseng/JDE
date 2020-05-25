#!/bin/sh

/opt/rh/devtoolset-6/root/usr/bin/g++ \
    -o mot \
    mot.cpp \
    jdetracker.cpp \
    trajectory.cpp \
    lapjv.cpp \
    /home/image/tseng/project/oss/ncnn/build/release/lib/libncnn.a \
    -I/usr/local/include/opencv4 \
    -I/usr/local/include/python3.6m \
    -I/home/image/tseng/venv/lib/python3.6/site-packages/numpy/core/include/numpy \
    -I/home/image/tseng/project/cpudetector/thirdparty/1.1.114.0/x86_64/include \
    -I/home/image/tseng/project/oss/ncnn/build/release/include/ncnn \
    `/usr/local/bin/python3-config --cflags` \
    `/usr/local/bin/python3-config --ldflags` \
    -Wall -O3 -g -fopenmp \
    -DNCNN_VULKAN \
    -L/home/image/tseng/project/cpudetector/thirdparty/1.1.114.0/x86_64/lib \
    -L/usr/local/lib64 \
    -lopencv_core -lopencv_video -lopencv_imgcodecs -lopencv_imgproc -lvulkan \
    -Wl,-rpath=/usr/local/lib64