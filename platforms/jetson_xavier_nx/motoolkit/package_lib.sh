#!/bin/bash

tar jcvf libmot.tar.bz2 \
./mot/build/install/include/* \
./mot/build/install/lib/* \
"./doc/MOT算法说明文档(C++).doc" \
./doc/mot.yaml \
./mot/build/install/bin/jde.wts