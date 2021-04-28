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

安装curl
sudo apt-get install curl

$ sudo vi /etc/network/interfaces
Append these:
ifconfig eth0 192.168.1.100 netmask 255.255.255.0 up
ifconfig eth0:0 172.1.1.254 netmask 255.255.255.0 up

or append these:
auto eth0

iface eth0 inet static
address 192.168.1.100
netmask 255.255.255.0
gateway 192.168.1.1

up ifconfig eth0:0 172.1.1.254 netmask 255.255.255.0
up route add -net 192.168.0.0 netmask 255.255.0.0 gw 192.168.1.1 eth0

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

$ sudo systemctl daemon-reload
$ sudo systemctl restart docker

安装eclipse-mosquitto
$ sudo docker pull eclipse-mosquitto
把mosquitto.conf放置在当前目录(南瑞提供)
$ sudo docker run --restart=always -itd -p 1883:1883 -p 9001:9001 -v /mnt/docker_container_volume/ini/mosquitto.conf:$(pwd)/mosquitto.conf  eclipse-mosquitto

加载主控镜像
把algfw_server_agx-nx.tar放置在当前目录(南瑞提供)
$ sudo docker load -i algfw_server_agx-nx.tar

安装ftp服务器
$ sudo apt install vsftpd
$ sudo mkdir -p /mnt/ftp/pub
$ sudo chown ftp:ftp /mnt/ftp/pub
把vsftpd.conf放置在当前目录(南瑞提供)
用南瑞提供的vsftpd.conf替换系统的vsftpd.conf
$ sudo cp /etc/vsftpd.conf /etc/vsftpd.conf.backup
$ sudo cp vsftpd.conf /etc/vsftpd.conf
重启ftp服务器并加入开机启动
$ sudo systemctl start vsftpd.service
$ sudo systemctl enable vsftpd.service

启动主控服务
docker run --restart=always --privileged --net=host \
    -v /mnt/data/:/mnt/data/ \
    -v /mnt/ftp/:/mnt/ftp/ \
    -v /mnt/docker_container_volume/ini/:/usr/ini/ \
    -v /mnt/docker_container_volume/log/:/var/log/sys/ \
    -v /usr/lib/aarch64-linux-gnu/libltdl.so.7:/usr/lib/aarch64-linux-gnu/libltdl.so.7 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -itd algfw_server_agx-nx:1.0 #/bin/bash

下发任务
$ curl -d '{"EventType":"app_task_config","DevId":"1234567890","AppName":"mot","TaskId":"znfxkj_debug_task_mot_ip_is_172_1_1_254_port_is_18082","Concurrent":1,"SingleRes":1 ,"RunTime":60,"AlgId":2001,"AlarmInterval":120,"Command":1,"TaskName":"debug_task"}' -H 'Content-Type: application/json' 127.0.0.1:18081/api/lapp/intelligence/analysis/interface

安装视频预览套件
$ sudo docker load -i client.tar
$ sudo docker tag 542f9e0469c9 algfw_client_agx:1.0
$ sudo ./docker_run_client_agx_nx.sh