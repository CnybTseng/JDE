# 1.依赖项
    本模块依赖opencv、ncnn(需扩展JDE解码模块)及其依赖项、yaml-cpp

## 1.1 yaml-cpp
    主机端编译命令    
cmake -DCMAKE_INSTALL_PREFIX=/home/image/tseng/project/JDE/cpp/thirdparty/yaml-cpp/build/install \
    -DCMAKE_C_COMPILER=/opt/rh/devtoolset-6/root/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/devtoolset-6/root/usr/bin/g++ \
    -DCMAKE_C_FLAGS="-fPIC" \
    -DCMAKE_CXX_FLAGS="-fPIC" ..