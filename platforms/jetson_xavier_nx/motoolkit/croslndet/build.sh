#!/bin/sh
javac -h ./src CrossLineDetector.java -d .

cd build
cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/install ..
make
make install
cd ..

cp com/sihan/system/jni/utils/CrossLineDetector.class ../mot4j/com/sihan/system/jni/utils