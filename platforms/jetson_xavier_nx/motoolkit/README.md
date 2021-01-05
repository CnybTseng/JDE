1.安装方法

1.1 解压安装包

1.2. 将动态库(usr/lib/libmot.so, usr/lib/libmot4j.so, usr/lib/libtrack_merge.so)更新到您指定的目录

1.3. 将跟踪模型(mot/build/install/bin/jde.trt)更新到应用程序可执行文件所在目录

2.模块更新说明

2020.12.22
1> 跟踪网络骨干网升级
2> 跟踪器输出边框的bug修复

2021.01.05
1> mot模块新增输出历史轨迹(对应第一个点的参数全为零的轨迹); mot模块配置参数更新
2> mot4j模块保留轨迹中的全零点
3> track_merge的merge_track()接口修改, 新增cost_thresh参数; 接口返回值由轨迹匹配关系替换为融合之后的轨迹