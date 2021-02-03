/******************************************************************************

                  版权所有 (C), 2004-2020, 成都思晗科技股份有限公司

 ******************************************************************************
  文 件 名   : mot4j.java
  版 本 号   : 初稿
  作    者   : Zeng Zhiwei
  生成日期   : 2020年9月24日
  最近修改   :
  功能描述   : 多目标跟踪的JNI接口
  
  修改历史   :
  1.日    期   : 2020年9月24日
    作    者   : Zeng Zhiwei
    修改内容   : 创建文件

******************************************************************************/

package com.sihan.system.jni.utils;

public class mot4j
{
    /**
     * @brief 加载JNI接口多目标跟踪算法库
     */
    static
    {
        try
        {
            System.loadLibrary("mot4j");
        }
        catch (UnsatisfiedLinkError e)
        {
            System.err.println("Load libmot4j.so fail\n");
        }
    }
    
    /**
     * @brief 加载多目标跟踪模型.
     * @param cfg_path 配置文件(.yaml)路径
     * @return  0, 模型加载成功
     *         -1, 模型加载失败
     */
    public native int load_mot_model(String cfg_path);
    /**
     * @brief 卸载多目标跟踪模型.
     * @return  0, 卸载模型成功
     *         -1, 卸载模型失败
     */
    public native int unload_mot_model();
    /**
     * @brief 执行多目标跟踪.
     * @param data BGR888格式图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param stride 图像扫描行字节步长
     * @return 多目标跟踪结果, 为Json转换来的字符串. 轨迹坐标序列中的第一个点如果全零,
     *         表示该条轨迹是历史轨迹.
     */
    public native String forward_mot_model(byte data[], int width, int height, int stride);
    
    /**
     * @brief 定时获取所有轨迹.
     * @warning 如果将reset设置为1, 所有的历史轨迹将被清空!!!
     * @param reset 如果查询结果显示轨迹队列已满, 是否将其重置.
     *        0, 不重置. 轨迹队列中的当前轨迹将以FIFO的方式更新.
     *        1, 重置. 队列中的所有历史轨迹将被清空。
     * @return
     *        如果定时器溢出, 返回所有轨迹; 否则, 返回null.
     */
    public native String get_total_tracks(int reset);
}