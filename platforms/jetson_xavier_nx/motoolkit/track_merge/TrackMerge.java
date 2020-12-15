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
     * @brief 融合轨迹. 只支持融合get_registered_channels返回的已经配准的通道的轨迹.
     *        注意, channel1和channel2的顺序必须与get_registered_channels返回的顺序一致.
     *        tracks1和tracks2必须为通道同步前提下跟踪到的轨迹.
     * @param tracks1 第一个通道的轨迹. 轨迹应来自mot4j的forward_mot_model接口函数.
     * @param tracks2 第二个通道的轨迹. 轨迹应来自mot4j的forward_mot_model接口函数.
     * @param channel1 第一个通道的ID.
     * @param channel2 第二个通道的ID.
     * @return 如果channel1和channel2是已经配准的通道, 返回融合轨迹ID的JSON字符串. 否则, 返回null.
     *         例如: [{track1: 0, track2: 0}, {track1: 1, track2: 2}], 表示第一个通道的轨迹0和
     *         第二个通道的轨迹0匹配. 第一个通道的轨1迹和第二个通道的轨迹2匹配. 如果至少一个通道
     *         的轨迹为空, 返回值也为null.
     */
    public native String merge_track(String tracks1, String tracks2, int channel1, int channel2);
}