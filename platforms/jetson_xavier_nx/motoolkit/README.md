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

2021.01.06 mot-0.2.1.tar.bz2
1> track_merge模块merge_track函数对传入的jstring类型轨迹进行null判断, 防止GetStringUTFChars崩溃
2> track_merge模块merge_track函数对返回的轨迹点新增通道字段; 返回的轨迹点不再统一映射到第二通道, 保持在原来通道

2021.01.20 mot-0.3.0.tar.bz2
1> mot4j模块新增is_the_track_queue_full()接口, 用于查询轨迹队列状态和重置轨迹队列
2> mot模块新增对FP16和INT8量化两种推理模式的支持, 目前版本配置的是FP16

2021.01.29 mot-0.3.1.tar.bz2
1> 删除mot4j模块is_the_track_queue_full()接口, 新增get_total_tracks()接口, 用于定时获取所有轨迹