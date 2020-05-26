# 1.依赖项
本模块依赖opencv、ncnn(需扩展JDE解码模块)及其依赖项、yaml-cpp

## 1.1 opencv
编译opencv参考官网https://opencv.org/

## 1.2 ncnn

将ncnn目录下的jdecoder.h和jdecoder.cpp拷贝到...

## 1.3 yaml-cpp
主机端编译命令可参考如下，根据实际情况修改    
cmake -DCMAKE_INSTALL_PREFIX=/home/image/tseng/project/JDE/cpp/thirdparty/yaml-cpp/build/install \
    -DCMAKE_C_COMPILER=/opt/rh/devtoolset-6/root/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/devtoolset-6/root/usr/bin/g++ \
    -DCMAKE_C_FLAGS="-fPIC" \
    -DCMAKE_CXX_FLAGS="-fPIC" ..