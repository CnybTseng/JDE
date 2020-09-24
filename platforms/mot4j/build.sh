#!/bin/sh

echo step 1: make mot4j.class and mot4j.h ...
javac -h . mot4j.java

echo step 2: make libmot4j.so ...
g++ mot4j.cpp -fPIC -shared -o libmot4j.so \
    -Imot/include \
    -I/usr/local/jdk-15/include/linux \
    -I/usr/local/jdk-15/include \
    -I/usr/include/opencv4 \
    -I/usr/local/jsoncpp/include \
    -Lmot/lib \
    -L/usr/local/lib64 \
    -L/usr/local/jsoncpp/lib \
    -lmot \
    -ljson \
    -lopencv_core -lopencv_video -lopencv_imgcodecs -lopencv_imgproc \
    /usr/local/lib/libSPIRV.a \
    /usr/local/lib/libSPIRV-Tools.a \
    /usr/local/lib/libSPIRV-Tools-link.a \
    /usr/local/lib/libSPIRV-Tools-opt.a \
    /usr/local/lib/libSPIRV-Tools-reduce.a \
    /usr/local/lib/libSPVRemapper.a

echo step 3: make test.class ...
javac test.java

echo step 4: set environment variable ...
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/mot/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/jsoncpp/lib:$LD_LIBRARY_PATH