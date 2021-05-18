https://blog.csdn.net/zong596568821xp/article/details/80405816
"rtspsrc location=rtsp://192.168.1.103:8554/ch20 latency=2000 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! appsink sync=false"
"rtspsrc location=rtsp://192.168.1.103:8554/ch20 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)25/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
"rtspsrc location=rtsp://192.168.1.103:8554/ch20 latency=200 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)NV12  ! appsink sync=false"

./install/bin/test-mtmct \
    --urls \
    20 \
    "rtspsrc location=rtsp://192.168.1.100:8554/20_channel.mkv latency=2000 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! appsink sync=false" \
    21 \
    "rtspsrc location=rtsp://192.168.1.100:8554/21_channel.mkv latency=2000 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! appsink sync=false"






    
CentOS 7编译opencv-4.5.1
准备源码
opencv-4.5.1
├── build
├── downloads
├── opencv-4.5.1
├── opencv-4.5.1.tar.gz
├── opencv_contrib-4.5.1
└── opencv_contrib-4.5.1.tar.gz

修改opencv-4.5.1/3rdparty/ippicv/ippicv.cmake
"https://raw.githubusercontent.com/opencv/opencv_3rdparty/${IPPICV_COMMIT}/ippicv/"替换为
"file:/home/image/tseng/project/thirdparty/opencv-4.5.1/downloads/"
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/a56b6ac6f030c312b2dce17430eef13aed9af274/ippicv/ippicv_2020_lnx_intel64_20191018_general.tgz

修改opencv-4.5.1/modules/gapi/cmake/DownloadADE.cmake
下载https://github.com/opencv/ade/archive/v0.1.1f.zip

修改opencv_contrib-4.5.1/modules/xfeatures2d/cmake/download_boostdesc.cmake
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_bgm.i
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_bgm_bi.i
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_bgm_hd.i
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_binboost_064.i
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_binboost_128.i
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_binboost_256.i
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/34e4206aef44d50e6bbcd0ab06354b52e7466d26/boostdesc_lbgm.i

修改opencv_contrib-4.5.1/modules/xfeatures2d/cmake/download_vgg.cmake
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/fccf7cd6a4b12079f73bbfb21745f9babcd4eb1d/vgg_generated_48.i
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/fccf7cd6a4b12079f73bbfb21745f9babcd4eb1d/vgg_generated_64.i
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/fccf7cd6a4b12079f73bbfb21745f9babcd4eb1d/vgg_generated_80.i
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/fccf7cd6a4b12079f73bbfb21745f9babcd4eb1d/vgg_generated_120.i

修改opencv_contrib-4.5.1/modules/face/CMakeLists.txt
下载https://raw.githubusercontent.com/opencv/opencv_3rdparty/8afa57abc8229d611c4937165d20e2a2d9fc5a12/face_landmark_model.dat

上面下载的玩意儿都放在downloads目录下吧

# cd build

生成编译工程
# cmake -DOPENCV_EXTRA_MODULES_PATH=$(pwd)/../opencv_contrib-4.5.1/modules -DCMAKE_INSTALL_PREFIX=$(pwd)/install $(pwd)/../opencv-4.5.1

编译含cuda的模块
# cmake -DBUILD_CUDA_STUBS=ON -DWITH_CUDA=ON -DCUDA_ARCH_BIN=6.1 -DOPENCV_EXTRA_MODULES_PATH=$(pwd)/../opencv_contrib-4.5.1/modules -DCMAKE_INSTALL_PREFIX=$(pwd)/install $(pwd)/../opencv-4.5.1

-DCUDA_ARCH_BIN=6.1指定cuda的计算能力, 也可忽略

进一步编译cuda dnn模块, 需要安装cuDNN
# cmake -DBUILD_CUDA_STUBS=ON -DOPENCV_DNN_CUDA=ON -DWITH_CUDA=ON -DOPENCV_EXTRA_MODULES_PATH=$(pwd)/../opencv_contrib-4.5.1/modules -DCMAKE_INSTALL_PREFIX=$(pwd)/install $(pwd)/../opencv-4.5.1









Jetson Xavir NX开发环境搭建
安装pip
$ sudo apt install python-pip

安装jetson-stats
$ sudo -H pip install -U jetson-stats

编译安装opencv-4.5.1
$ cmake -DBUILD_SHARED_LIBS=OFF -DBUILD_CUDA_STUBS=ON -DWITH_CUDA=ON -DOPENCV_DNN_CUDA=ON -DCUDA_ARCH_BIN=7.2 -DOPENCV_EXTRA_MODULES_PATH=$(pwd)/../opencv_contrib-4.5.1/modules -DCMAKE_INSTALL_PREFIX=$(pwd)/install $(pwd)/../opencv-4.5.1
BUILD_STATIC_LIBS根本不起作用, 不能同时编译静态和动态库

编译安装jsoncpp-1.9.4
$ cmake -DCMAKE_INSTALL_PREFIX="$(pwd)/install" -DCMAKE_CXX_FLAGS="-fPIC" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON ../jsoncpp-1.9.4/

把algfw_server_agx-nx_1.0.tar, client.tar, algfw-sdk-agx-nx_1.0_aarch64.tar等文件放置到某个目录${ROOT}

安装curl
sudo apt-get install curl

配置IP地址
$ sudo vi /etc/network/interfaces
根据实际的IP添加
auto eth0

iface eth0 inet static
address 192.168.1.100
netmask 255.255.255.0
gateway 192.168.1.1

up ifconfig eth0:0 172.1.1.254 netmask 255.255.255.0
up route add -net 192.168.0.0 netmask 255.255.0.0 gw 192.168.1.1 eth0
重启网卡
$ sudo service network-manager restart
$ sudo reboot

安装registry镜像
$ sudo docker pull registry
$ sudo docker images
$ sudo docker run --restart=always -d -p 5000:5000 -v /opt/registry_data:/var/lib/registry registry
$ sudo docker ps -a
$ sudo vi /lib/systemd/system/docker.service
在ExecStart这一行追加:
--insecure-registry 172.1.1.254:5000 -H tcp://0.0.0.0:9527
重启docker
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker

安装eclipse-mosquitto
$ sudo docker pull eclipse-mosquitto:1.6.14
把mosquitto.conf放置在/mnt/docker_container_volume/ini目录
$ mv mosquitto.conf /mnt/docker_container_volume/ini
启动
$ sudo docker run --restart=always -itd -p 1883:1883 -p 9001:9001 -v /mnt/docker_container_volume/ini/mosquitto.conf:/mosquitto/config/mosquitto.conf  eclipse-mosquitto:1.6.14

安装ftp服务器
$ sudo apt-get install vsftpd
$ sudo mkdir -p /mnt/ftp/pub
$ sudo chown ftp:ftp /mnt/ftp/pub
用南瑞提供的vsftpd.conf替换系统的vsftpd.conf
$ sudo cp /etc/vsftpd.conf /etc/vsftpd.conf.backup
$ sudo cp vsftpd.conf /etc/vsftpd.conf
重启ftp服务器并加入开机启动
$ sudo systemctl start vsftpd.service
$ sudo systemctl enable vsftpd.service

加载主控镜像
把algfw_server_agx-nx.tar放置在当前目录
$ sudo docker load -i algfw_server_agx-nx.tar

把config.xml放置在/mnt/docker_container_volume/ini/目录下

启动主控服务
$ sudo ./start_algfw_server_container.sh

安装视频预览套件
$ sudo docker load -i client.tar
$ sudo docker tag 542f9e0469c9 algfw_client_agx:1.0
$ sudo ./docker_run_client_agx_nx.sh

加载算法基础镜像
$ sudo docker load -i algfw-sdk-agx-nx_1.0_aarch64.tar
构建算法镜像
$ cd ${ROOT}/mot
$ make -j6
$ make install
$ python ../dependcollect.py -d ./mot -p ./docker/
$ sudo docker build -t algfw-mot-agx_nx_aarch64:1.0 ./docker
$ sudo docker save -o algfw-mot-agx_nx_aarch64_1.0.tar algfw-mot-agx_nx_aarch64:1.0
只用于测试:
// $ sudo docker tag algfw-mot-agx_nx_aarch64:1.0 192.168.1.100:5000/algfw-mot-agx_nx_aarch64:1.0
// $ sudo docker push 192.168.1.100:5000/algfw-mot-agx_nx_aarch64:1.0

curl来测试
curl -d '{"EventType":"app_task_config","DevId":"1234567890","AppName":"mot","TaskId":"znfxkj_debug_task_mot_ip_is_0_0_0_0_port_is_18082","Concurrent":1,"SingleRes":1 ,"RunTime":60,"AlgId":2001,"AlarmInterval":120,"Command":1,"TaskName":"debug_task"}' -H 'Content-Type: application/json' 127.0.0.1:18081/api/lapp/intelligence/analysis/interface

手动拉起容器, 自动运行命令:
docker run --rm -p 18082 --privileged --runtime nvidia --log-opt max-size=10m --log-opt max-file=3 -v /mnt/alg_container_volume/log:/var/log/sys -d algfw-mot-agx_nx_aarch64:1.0

手动拉起容器, 进入容器命令:
docker run --rm -p 18082 --privileged --runtime nvidia --log-opt max-size=10m --log-opt max-file=3 -v /mnt/alg_container_volume/log:/var/log/sys -it algfw-mot-agx_nx_aarch64:1.0 /bin/bash

APP安装时的启动参数:
--rm -p 18082 --privileged --runtime nvidia --log-opt max-size=10m --log-opt max-file=3 -v /mnt/alg_container_volume/log:/var/log/sys -d 

如果要用到/usr/bin下的命令, 到容器内的/outer_bin/目录下找










把192.168.1.100作为控制节点+分析节点, 192.168.1.113作为分析节点
对于192.168.1.100
$ sudo vi /lib/systemd/system/docker.service
将对内对外IP改成一样
  <MAINBOARD>
    <ETH>
      <OuterEth>
        <Ip>192.168.1.100</Ip>
      </OuterEth>
      <InnerEth>
        <Ip>192.168.1.100</Ip>
      </InnerEth>
    </ETH>
    <Arch>1</Arch>
  </MAINBOARD>
修改子板配置, Subboard_1字段为新增的分析节点的配置
  <RESOURCE>
    <SUBBOARD>
      <SubboardNum>2</SubboardNum>
      <Subboard_0>
        <Ip>172.1.1.254</Ip>
        <Arch>1</Arch>
        <TpuCoreNum>1</TpuCoreNum>
        <RemoteDockerPort>9527</RemoteDockerPort>
        <ComputerPowerEachCore>128.000000</ComputerPowerEachCore>
      </Subboard_0>
      <Subboard_1>
        <Ip>192.168.1.113</Ip>
        <Arch>1</Arch>
        <TpuCoreNum>1</TpuCoreNum>
        <RemoteDockerPort>9527</RemoteDockerPort>
        <ComputerPowerEachCore>128.000000</ComputerPowerEachCore>
      </Subboard_1>
    </SUBBOARD>
  </RESOURCE>

$ sudo vi /lib/systemd/system/docker.service
ExecStart=这一行由:
--insecure-registry 172.1.1.254:5000 -H tcp://0.0.0.0:9527
改为
--insecure-registry 192.168.1.100:5000 -H tcp://0.0.0.0:9527    

重启docker
sudo systemctl daemon-reload
sudo systemctl restart docker
重启主控
$ sudo docker stop CONTAINER_ID
$ sudo docker rm CONTAINER_ID
$ sudo ./start_algfw_server_container.sh

对于192.168.1.113
$ sudo vi /lib/systemd/system/docker.service
在ExecStart那行追加:
--insecure-registry 192.168.1.100:5000 -H tcp://0.0.0.0:9527
重启docker
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker     