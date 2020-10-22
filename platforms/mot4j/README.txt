编译和测试多目标跟踪模块的JNI接口

1. 编译
1.1 编译环境
公司的NVIDIA Jetson Xavier NX (IP(192.168.1.100), Username:Password(sihan:sihan123))
将mot4j包下载到您指定的目录

1.2 编译步骤
$ cd /path/to/mot4j
$ source ./build.sh

2. 测试
$ java test mot.yaml ims

终端出现如下打印信息:
ims/000001.jpg
null

Empty result
ims/000002.jpg
[
   {
      "category" : "person",
      "identifier" : "1",
      "rects" : [
         {
            "height" : "144",
            "width" : "54",
            "x" : "1498",
            "y" : "33"
         }
      ]
   },
   {
      "category" : "person",
      "identifier" : "2",
      "rects" : [
         {
            "height" : "132",
            "width" : "53",
            "x" : "1416",
            "y" : "27"
         }
      ]
   },
...