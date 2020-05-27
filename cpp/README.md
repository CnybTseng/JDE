# 1. 依赖项
本模块依赖opencv、ncnn(需扩展JDE解码模块)及其依赖项、yaml-cpp

## 1.1 编译opencv
编译opencv参考官网https://opencv.org/

## 1.2 编译ncnn

从github克隆ncnn到本地
git clone https://github.com/Tencent/ncnn.git

将本模块ncnn目录下的jdecoder.h和jdecoder.cpp拷贝到克隆的ncnn目录(ncnn/src/layer)之下，打开ncnn/src/CMakeLists.txt，紧接着ncnn_add_layer(某某层)添加ncnn_add_layer(JDEcoder)

将本模块ncnn目录下的jde.cpp拷贝到克隆的ncnn目录(ncnn/example)之下，打开ncnn/example/CMakeLists.txt，紧接着ncnn_add_example(某某测试程序)添加ncnn_add_example(jde)

编译方法请参考官网https://github.com/Tencent/ncnn

## 1.3 yaml-cpp
主机端编译命令可参考如下，根据实际情况修改    
cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/install \
    -DCMAKE_C_COMPILER=/opt/rh/devtoolset-6/root/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/devtoolset-6/root/usr/bin/g++ \
    -DCMAKE_C_FLAGS="-fPIC" \
    -DCMAKE_CXX_FLAGS="-fPIC" ..

交叉编译命令可参考如下(以华为海思Hi3559AV100为例)，根据实际情况修改
cmake -G "Unix Makefiles" \
    -DCMAKE_C_COMPILER=aarch64-himix100-linux-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-himix100-linux-g++ \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/install \
    -DCMAKE_SYSTEM_NAME=Generic \
    -DCMAKE_FIND_ROOT_PATH=/opt/hisi-linux \
    -DCMAKE_C_FLAGS="-fPIC" \
    -DCMAKE_CXX_FLAGS="-fPIC" ..

# 2. 编译本模块
根据实际情况修改Makefile，然后make即可

# 3. 测试本模块
根据实际情况修改配置文件mot.yaml，注意测试图像必须为jpg格式(后缀名为.jpg)，如需测试其他格式图像，请修改mot-test.cpp文件第60行
./mot ./mot.yaml /path/to/the/test/images