#!/bin/sh

echo step 1: make TrackMerge.class and com_sihan_system_jni_utils_TrackMerge.h ...
javac -h ./src TrackMerge.java -d .

echo step 2: make libtrack_merge.so ...
cd build
cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/install ..
make
make install
cd ..

echo step 3: make Test.class ...
javac Test.java