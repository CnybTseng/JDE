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
     * @return 多目标跟踪结果, 为Json转换来的字符串
     */
    public native String forward_mot_model(byte data[], int width, int height, int stride);
}