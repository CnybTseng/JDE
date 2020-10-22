# 1. 依赖项
本模块依赖opencv、ncnn(需扩展JDE解码模块)及其依赖项、yaml-cpp

## 1.1 编译opencv
编译opencv参考官网https://opencv.org/

交叉编译命令可参考如下，根据实际情况修改
cmake -DCMAKE_TOOLCHAIN_FILE=../himix100.toolchain.cmake \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/install ..

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

# 4. 使用本模块
使用本模块只需要mot.h、SH_ImageAlgLogSystem.h和libmot.so，详细步骤参考mot-test.cpp

# 编译GPU版本的NCNN

## 1. 安装Vulkan Header
从https://github.com/KhronosGroup/Vulkan-Headers下载Vulkan-Header
wget https://github.com/KhronosGroup/Vulkan-Headers/archive/v1.2.151.tar.gz -O Vulkan-Header-1.2.151.tar.gz
tar zxvf Vulkan-Header-1.2.151.tar.gz
cd Vulkan-Headers-1.2.151
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=install ..
make install

## 2. 安装Vulkan Loader

安装依赖包
sudo apt-get install git build-essential libx11-xcb-dev \
    libxkbcommon-dev libwayland-dev libxrandr-dev

从https://github.com/KhronosGroup/Vulkan-Loader下载Vulkan-Loader的发行版
wget https://github.com/KhronosGroup/Vulkan-Loader/archive/v1.2.151.tar.gz -O Vulkan-Loader-1.2.151.tar.gz
tar zxvf Vulkan-Loader-1.2.151.tar.gz
cd Vulkan-Loader-1.2.151
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DVULKAN_HEADERS_INSTALL_DIR=/home/sihan/Documents/mot/thirdparty/Vulkan-Headers-1.2.151/build/install \
      -DCMAKE_INSTALL_PREFIX=install \
      -DCMAKE_CXX_FLAGS="-fPIC" ..
make -j2
make install

sudo cp loader/libvulkan.so.1.1.100 /usr/lib/aarch64-linux-gnu/
cd /usr/lib/aarch64-linux-gnu/
sudo rm -rf libvulkan.so.1 libvulkan.so
sudo ln -s libvulkan.so.1.1.100 libvulkan.so
sudo ln -s libvulkan.so.1.1.100 libvulkan.so.1

export VULKAN_SDK=/home/sihan/Documents/mot/thirdparty/Vulkan-Loader-1.2.151/build/install

## 3. 安装glslang

git clone --depth=1 https://github.com/KhronosGroup/glslang.git
cd glslang
git clone https://github.com/google/googletest.git External/googletest
./update_glslang_sources.py
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
    -DCMAKE_CXX_FLAGS="-fPIC" ..

cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fPIC" ..

make -j2 install

## 4. 编译ncnn

git clone https://github.com/Tencent/ncnn.git
cd ncnn && mkdir -p build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/jetson.toolchain.cmake \
    -DNCNN_VULKAN=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-I/home/sihan/Documents/mot/thirdparty/glslang/build/install/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/home/sihan/Documents/mot/thirdparty/glslang/build/install/lib" \
    -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
    -DCMAKE_CXX_FLAGS="-fPIC" ..

cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/jetson.toolchain.cmake \
    -DNCNN_VULKAN=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
    -DCMAKE_CXX_FLAGS="-fPIC" ..


make -j2

# 编译yaml-cpp

git clone https://github.com/jbeder/yaml-cpp.git --depth=1
cd yaml-cpp/
mkdir build
cd build
cmake -DYAML_BUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
    -DCMAKE_CXX_FLAGS="-fPIC" ..
make -j2
make install

# pytorch量化
https://discuss.pytorch.org/t/onnx-export-of-quantized-model/76884/10
https://discuss.pytorch.org/t/converting-quantized-models-from-pytorch-to-onnx/84855
https://discuss.pytorch.org/t/converting-quantized-models-from-pytorch-to-onnx/84855
https://discuss.pytorch.org/t/onnx-export-of-quantized-model/76884/8