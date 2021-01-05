/******************************************************************************

                  版权所有 (C), 2004-2020, 成都思晗科技股份有限公司

 ******************************************************************************
  文 件 名   : TrackMerge.java
  版 本 号   : 初稿
  作    者   : Zeng Zhiwei
  生成日期   : 2020年12月7日
  最近修改   :
  功能描述   : 同步通道轨迹融合
  
  修改历史   :
  1.日    期   : 2020年12月7日
    作    者   : Zeng Zhiwei
    修改内容   : 创建文件

******************************************************************************/

package com.sihan.system.jni.utils;

public class TrackMerge
{
    static
    {
        try
        {
            System.loadLibrary("track_merge");
        }
        catch (UnsatisfiedLinkError e)
        {
            System.err.println("load libtrack_merge.so fail\n");
        }
    }
    /**
     * @brief 查询已经配准的通道.
     * @return 返回已经配准的通道号的JSON字符串. 例如: [ "20-21", "53-52" ].
     */
    public native String get_registered_channels();
    /**
     * @brief 融合轨迹.
     * @warning <1>只支持融合get_registered_channels返回的已经配准的通道的轨迹.
     *          <2>channel1和channel2的顺序必须与get_registered_channels返回的顺序一致.
     *          <3>tracks1和tracks2必须为通道同步前提下跟踪到的轨迹.
     *          <4>融合之后的轨迹统一映射到第二个通道的图像坐标系下.
     *          <5>从第一个通道映射到第二个通道的点的width和height为0, x和y表示脚的坐标.
     * @param tracks1 第一个通道的轨迹. 轨迹应来自mot4j的forward_mot_model接口函数.
     * @param tracks2 第二个通道的轨迹. 轨迹应来自mot4j的forward_mot_model接口函数.
     * @param channel1 第一个通道的ID.
     * @param channel2 第二个通道的ID.
     * @param cost_thresh 代价矩阵值域阈值(单位: 像素). 值越小, 匹配越严格. 建议值: 100.
     * @return 如果channel1和channel2是已经配准的通道, 返回融合轨迹的JSON字符串. 如果通道尚未配准,
     *         或者其中之一的轨迹为空, 或者输入的轨迹格式错误, 返回null.
     */
    public native String merge_track(String tracks1, String tracks2, int channel1, int channel2, int cost_thresh);
}