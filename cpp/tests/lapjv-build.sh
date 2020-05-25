#!/bin/sh

/opt/rh/devtoolset-6/root/usr/bin/g++ -o lapjv lapjv.cpp lapjv_test.cpp \
    -I/usr/local/include/python3.6m \
    -I/home/image/tseng/venv/lib/python3.6/site-packages/numpy/core/include/numpy \
    `/usr/local/bin/python3-config --cflags` \
    `/usr/local/bin/python3-config --ldflags` \
    -g