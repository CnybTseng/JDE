#!/bin/sh

/opt/rh/devtoolset-6/root/usr/bin/g++ \
    -o jdetracker \
    jdetracker.cpp \
    trajectory.cpp \
    lapjv.cpp \
    -I/usr/local/include/python3.6m \
    -I/home/image/tseng/venv/lib/python3.6/site-packages/numpy/core/include/numpy \
    `/usr/local/bin/python3-config --cflags` \
    `/usr/local/bin/python3-config --ldflags` \
    -Wall -O3 -g \
    -DTEST_JDETRACKER_MODULE \
    -lopencv_core -lopencv_video