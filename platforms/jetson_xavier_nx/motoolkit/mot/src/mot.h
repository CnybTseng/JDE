/******************************************************************************

                  版权所有 (C), 2004-2020, 成都思晗科技股份有限公司

 ******************************************************************************
  文 件 名   : mot.h
  版 本 号   : 初稿
  作    者   : Zeng Zhiwei
  生成日期   : 2020年9月24日
  最近修改   :
  功能描述   : 多目标跟踪
  
  修改历史   :
  1.日    期   : 2020年9月24日
    作    者   : Zeng Zhiwei
    修改内容   : 创建文件
  
  2.日    期   : 2020年10月15日
    作    者   : Zeng Zhiwei
    修改内容   : <1> forward_mot_model()输入图像的格式由RGB888修改为BGR888.
                 <2> 算法核心由ncnn框架切换为TensorRT框架, 更好地利用GPU的算力.
                 <3> 算法核心的多线程支持.
                 <4> 算法核心只返回当前帧的跟踪结果.

******************************************************************************/

#ifndef MOT_H
#define MOT_H

#include <deque>
#include <vector>
#include <string>

#if defined _WIN32 || defined __CYGWIN__
  #define MOT_HELPER_DLL_IMPORT __declspec(dllimport)
  #define MOT_HELPER_DLL_EXPORT __declspec(dllexport)
  #define MOT_HELPER_DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define MOT_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
    #define MOT_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
    #define MOT_HELPER_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define MOT_HELPER_DLL_IMPORT
    #define MOT_HELPER_DLL_EXPORT
    #define MOT_HELPER_DLL_LOCAL
  #endif
#endif

#ifdef MOT_DLL
  #ifdef MOT_DLL_EXPORTS
    #define MOT_API MOT_HELPER_DLL_EXPORT
  #else
    #define MOT_API MOT_HELPER_DLL_IMPORT
  #endif // MOT_DLL_EXPORTS
  #define MOT_LOCAL MOT_HELPER_DLL_LOCAL
#else
  #define MOT_API
  #define MOT_LOCAL
#endif // MOT_DLL

namespace mot {

// 目标边框
struct MOT_Rect
{
    float top;
    float left;
    float bottom;
    float right;
};

// 目标跟踪的轨迹
struct MOT_Track
{
    int identifier;                 // 轨迹ID
    std::string category;           // 类别, 目前总是'person'
    std::deque<MOT_Rect> rects;     // 目标边框
};

typedef std::vector<MOT_Track> MOT_Result;

/**
 * @brief 加载多目标跟踪模型.
 * @param cfg_path 配置文件(.yaml)路径
 * @return  0, 模型加载成功
 *         -1, 模型加载失败
 */
extern "C" MOT_API int load_mot_model(const char *cfg_path);

/**
 * @brief 卸载多目标跟踪模型.
 * @return  0, 卸载模型成功
 *         -1, 卸载模型失败
 */
extern "C" MOT_API int unload_mot_model();

/**
 * @brief 执行多目标跟踪.
 * @param data   BGR888格式图像数据
 * @param width  图像宽度
 * @param height 图像高度
 * @param stride 图像扫描行字节步长
 * @param result 多目标跟踪结果
 * @return    0, 执行多目标跟踪成功
 *          非0, 执行多目标跟踪失败
 */
extern "C" MOT_API int forward_mot_model(const unsigned char *data, int width, int height, int stride, MOT_Result &result);

}   // namespace mot

#endif  // MOT_H