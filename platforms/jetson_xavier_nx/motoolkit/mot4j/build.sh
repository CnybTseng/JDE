#!/bin/sh

echo step 0: update mot library ...
cp -rf ../mot/build/install/* 3rdparty/mot

echo step 1: make mot4j.class and com_sihan_system_jni_utils_mot4j.h ...
javac -h ./src mot4j.java -d .

echo step 2: make libmot4j.so ...
cd build
cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/install ..
make
make install
cd ..

echo step 3: make test.class ...
javac test.java