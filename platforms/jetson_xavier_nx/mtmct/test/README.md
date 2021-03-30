https://blog.csdn.net/zong596568821xp/article/details/80405816
"rtspsrc location=rtsp://192.168.1.103:8554/ch20 latency=2000 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! appsink sync=false"
"rtspsrc location=rtsp://192.168.1.103:8554/ch20 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)25/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
"rtspsrc location=rtsp://192.168.1.103:8554/ch20 latency=200 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)NV12  ! appsink sync=false"

CentOS 7编译opencv-4.5.1

准备源码
opencv-4.5.1
├── build
├── downloads
├── opencv-4.5.1
├── opencv-4.5.1.tar.gz
├── opencv_contrib-4.5.1
└── opencv_contrib-4.5.1.tar.gz

生成编译工程
cmake -DOPENCV_EXTRA_MODULES_PATH=$(pwd)/../opencv_contrib-4.5.1/modules -DCMAKE_INSTALL_PREFIX=$(pwd)/install $(pwd)/../opencv-4.5.1

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