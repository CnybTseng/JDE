/******************************************************************************

                  版权所有 (C), 2004-2020, 成都思晗科技股份有限公司

 ******************************************************************************
  文 件 名   : CrossLineDetector.java
  版 本 号   : 初稿
  作    者   : Zeng Zhiwei
  生成日期   : 2021年4月12日
  最近修改   :
  功能描述   : 同步通道轨迹融合
  
  修改历史   :
  1.日    期   : 2021年4月12日
    作    者   : Zeng Zhiwei
    修改内容   : 创建文件

******************************************************************************/

package com.sihan.system.jni.utils;

/**
 * @brief class CrossLineDetector.
 * @note 跨线检测算法支持多路视频, 但是每路视频只支持设置一条直线.
 *       算法建立在一路视频只覆盖管控区域的一个出入口, 且出入口相同的假设基础上.
 */
public class CrossLineDetector
{
    static
    {
        try
        {
            System.loadLibrary("croslndet");
        }
        catch (UnsatisfiedLinkError e)
        {
            System.err.println("load libcroslndet.so fail\n");
        }
    }
    /**
     * @brief 设置跨线检测的直线.
     * @note 注意, (x1,y1), (x2,y2), 和(x3,y3)必须为不共线的三个点.
     * @param channel 摄像机通道号.
     * @param x1 直线上第一个点的x坐标.
     * @param y1 直线上第一个点的y坐标.
     * @param x2 直线上第二个点的x坐标.
     * @param y2 直线上第二个点的y坐标.
     * @param x3 管控区域内侧任意一点的x坐标.
     * @param y3 管控区域内侧任意一点的y坐标.
     * @return 如果设置成功, 返回true; 否则, 返回false.
     *         如果(x1,y1), (x2,y2), 和(x3,y3)共线, 或者同一通道调用该接口一次以上, 将设置失败.
     */
    public native boolean set_line(int channel, int x1, int y1, int x2, int y2, int x3, int y3);
    // 该接口暂时用不着吧.
    // public native String get_lines();
    /**
     * @brief 检测跨线行为.
     * @note 只统计本通道出入的目标个数.
     * @param tracks 目标轨迹. 轨迹应来自mot4j的forward_mot_model接口函数.
     * @param recall 轨迹回溯长度. 目标状态变化将依据当前时刻t和t-recall时刻的信息得出.
     *        recall∈[2,+∞), 建议值: 2.
     * @return 返回当前时刻管控区域内的目标个数, 进入区域的目标个数, 和离开区域的目标个数.
     *         返回的是Json字符串, 格式为
     *         {
     *             "total_targets": 4,
     *             "entry": 1,
     *             "exit": 0
     *         }
     */
    public native String detect_cross_event(String tracks, int recall);
}