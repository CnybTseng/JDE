1.进入例子目录
$ cd $(ROOT)/algfw-sdk-agx_nx_202103150911/sample
编译algsdk_sample
2.建立一些库文件的软链接
ln -s ../lib/libnvmpi.so ../lib/libnvmpi.so.1.0.0
ln -s ../lib/libevent.so ../lib/libevent-2.1.so.7
ln -s ../lib/libevent_pthreads.so ../lib/libevent_pthreads-2.1.so.7
ln -s ../lib/libmosquitto.so ../lib/libmosquitto.so.1
ln -s ../lib/libopencv_core.so ../lib/libopencv_core.so.3.4
ln -s ../lib/libopencv_imgcodecs.so ../lib/libopencv_imgcodecs.so.3.4
ln -s ../lib/libopencv_imgproc.so ../lib/libopencv_imgproc.so.3.4
ln -s ../lib/libavcodec.so ../lib/libavcodec.so.58
ln -s ../lib/libavformat.so ../lib/libavformat.so.58
ln -s ../lib/libswresample.so ../lib/libswresample.so.3
ln -s ../lib/libavutil.so ../lib/libavutil.so.56

3.编译
make -j4
4.制作镜像
4.1.加载基础镜像
$ sudo docker load -i algfw-sdk-agx-nx_1.0_aarch64.tar
[sudo] password for sihan:
2e95f1db4570: Loading layer [==================================================>]  58.97MB/58.97MB
ccf35b27430d: Loading layer [==================================================>]  991.2kB/991.2kB
f0a302c91d12: Loading layer [==================================================>]  15.87kB/15.87kB
307fb5586e3d: Loading layer [==================================================>]  3.584kB/3.584kB
28a36a68f690: Loading layer [==================================================>]  479.1MB/479.1MB
d221eee15644: Loading layer [==================================================>]  3.072kB/3.072kB
e75a2293ecd5: Loading layer [==================================================>]  3.584kB/3.584kB
3d0087db25dd: Loading layer [==================================================>]  4.096kB/4.096kB
2c70c0d8b308: Loading layer [==================================================>]  4.096kB/4.096kB
e2b716a92ef6: Loading layer [==================================================>]  3.072kB/3.072kB
d3e00dba8347: Loading layer [==================================================>]  51.21MB/51.21MB
35e59e99ae52: Loading layer [==================================================>]  35.47MB/35.47MB
d92310f80aba: Loading layer [==================================================>]  523.3kB/523.3kB
fb851a4a5a82: Loading layer [==================================================>]  23.09MB/23.09MB
1ba9d058a24c: Loading layer [==================================================>]  1.335MB/1.335MB
01523dfcfcd0: Loading layer [==================================================>]    682kB/682kB
609626da1908: Loading layer [==================================================>]    791kB/791kB
e08c26b9035a: Loading layer [==================================================>]  4.096kB/4.096kB
f22e012bfa73: Loading layer [==================================================>]  76.29kB/76.29kB
3fabfa59e486: Loading layer [==================================================>]  5.614MB/5.614MB
f95d82362f9a: Loading layer [==================================================>]  61.89MB/61.89MB
a5e7332081b1: Loading layer [==================================================>]  100.4kB/100.4kB
Loaded image: algfw-sdk-agx_nx_aarch64:1.0
4.2.查看Docker镜像
$ sudo docker images
[sudo] password for sihan:
REPOSITORY                 TAG                 IMAGE ID            CREATED             SIZE
algfw-sdk-agx_nx_aarch64   1.0                 85b74bf05844        3 weeks ago         698MB
用打印的REPOSITORY和TAG来修改Dockerfile, Dockerfile文件内容为
FROM algfw-sdk-agx_nx_aarch64:1.0

COPY sample/algsdk_sample /usr/local/bin/algsdk_sample

EXPOSE 18082
RUN ldconfig

WORKDIR /workspace

#add startup
CMD ["/usr/local/bin/algsdk_sample"]
4.3.构建Docker镜像
$ export SDK_RLS_DIR=$(dirname ${PWD})
$ export DOCKER_IMAGE=algfw-sdk-agx_nx_aarch64
$ export DOCKER_TAG=1.0
$ chmod +x ${SDK_RLS_DIR}/sample/algsdk_sample
$ sudo docker build -f ${SDK_RLS_DIR}/sample/Dockerfile -t ${DOCKER_IMAGE}:${DOCKER_TAG} ${SDK_RLS_DIR}
Sending build context to Docker daemon    789MB
Step 1/6 : FROM algfw-sdk-agx_nx_aarch64:1.0
 ---> 85b74bf05844
Step 2/6 : COPY sample/algsdk_sample /usr/local/bin/algsdk_sample
 ---> 4ad2ba5f22c3
Step 3/6 : EXPOSE 18082
 ---> Running in 5c08c85c455f
Removing intermediate container 5c08c85c455f
 ---> aff1b04a8ae6
Step 4/6 : RUN ldconfig
 ---> Running in 480cafa7d68c
Removing intermediate container 480cafa7d68c
 ---> 9dfe52d1ee46
Step 5/6 : WORKDIR /workspace
 ---> Running in 7bb74b7009f4
Removing intermediate container 7bb74b7009f4
 ---> 40885d03b017
Step 6/6 : CMD ["/usr/local/bin/algsdk_sample"]
 ---> Running in a708f81886cc
Removing intermediate container a708f81886cc
 ---> d3e034a626d7
Successfully built d3e034a626d7
Successfully tagged algfw-sdk-agx_nx_aarch64:1.0
4.4.保存镜像
$ sudo docker save -o ${SDK_RLS_DIR}/sample/${DOCKER_IMAGE}.tar ${DOCKER_IMAGE}:${DOCKER_TAG}
5.安装主控
# docker load -i algfw_server_agx-nx.tar
7beab76caed6: Loading layer [==================================================>]  1.536kB/1.536kB
2c3097a820aa: Loading layer [==================================================>]  3.072kB/3.072kB
28fb81934f1b: Loading layer [==================================================>]   2.56kB/2.56kB
c7bc8102d458: Loading layer [==================================================>]  2.981MB/2.981MB
2570c332bca4: Loading layer [==================================================>]  148.1MB/148.1MB
98e36190fb05: Loading layer [==================================================>]  240.5MB/240.5MB
Loaded image: algfw_server_agx-nx:1.0
# docker run --restart=always --privileged --net=host \
    -v /mnt/data/:/mnt/data/ \
    -v /mnt/ftp/:/mnt/ftp/ \
    -v /mnt/docker_container_volume/ini/:/usr/ini/ \
    -v /mnt/docker_container_volume/log/:/var/log/sys/ \
    -v /usr/lib/aarch64-linux-gnu/libltdl.so.7:/usr/lib/aarch64-linux-gnu/libltdl.so.7 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -itd algfw_server_agx-nx:1.0 #/bin/bash
命令执行完毕返回:
4ccb04c048428b44604ff681590b229051933e1583e0dc3a69da45e0fa264065
6.浏览器登录
chrome浏览器输入http://192.168.1.100:18080/进入登录界面
用户名admin, 密码123456

安装ftp


安装mosquitto





$ export SDK_RLS_DIR=$(dirname ${PWD})
$ export DOCKER_IMAGE_NEW=algfw-mot-agx_nx_aarch64
$ export DOCKER_TAG=1.0
$ chmod +x ${SDK_RLS_DIR}/mot/mot
$ sudo docker build -f ${SDK_RLS_DIR}/mot/Dockerfile -t ${DOCKER_IMAGE_NEW}:${DOCKER_TAG} ${SDK_RLS_DIR}
$ sudo docker save -o ${SDK_RLS_DIR}/mot/${DOCKER_IMAGE_NEW}.tar ${DOCKER_IMAGE_NEW}:${DOCKER_TAG}


docker tag algfw-mot-agx_nx_aarch64:1.0 172.1.1.254:5000/algfw-mot-agx_nx_aarch64_rn:1.0
docker push 172.1.1.254:5000/algfw-mot-agx_nx_aarch64_rn:1.0
docker rmi 172.1.1.254:5000/algfw-mot-agx_nx_aarch64_rn:1.0
docker pull 172.1.1.254:5000/algfw-mot-agx_nx_aarch64_rn:1.0

docker run --restart=always --privileged --net=host \
    -v /mnt/data/:/mnt/data/ \
    -v /mnt/ftp/:/mnt/ftp/ \
    -v /mnt/docker_container_volume/ini/:/usr/ini/ \
    -v /mnt/docker_container_volume/log/:/var/log/sys/ \
    -v /usr/lib/aarch64-linux-gnu/libltdl.so.7:/usr/lib/aarch64-linux-gnu/libltdl.so.7 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -itd algfw-mot-agx_nx_aarch64:1.0 #/bin/bash
    

curl -d '{"EventType":"app_task_config","DevId":"1234567890","AppName":"mot","TaskId":"znfxkj_debug_task_mot_ip_is_172_1_1_254_port_is_18082","Concurrent":1,"SingleRes":1 ,"RunTime":60,"AlgId":2001,"AlarmInterval":30,"Command":1,"TaskName":"debug_task"}' -H 'Content-Type: application/json' 127.0.0.1:18081/api/lapp/intelligence/analysis/interface

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/../lib