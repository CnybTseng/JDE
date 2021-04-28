#ifndef __ALG_CODEC_H_
#define __ALG_CODEC_H_

//基础模块
#include "algsdk.h"

//第三方库
#include "opencv2/opencv.hpp"
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @name alg_codec模块初始化
 */
int alg_codec_init(void);

/**
* @name alg_codec模块退出
*/
int alg_codec_exit(void);

/**
 * @name    获取分析源图片
 * @param   source_url    源url,支持rtsp, http
 * @param   image_type    输出图片格式类型EN_IMAGE_TYPE
 * @retval  !0:成功, 返回分析源图片数据;   nullptr:失败
 */
ALGSDK_IMAGE_ST *alg_codec_get_source_image(char* source_url,  ALGSDK_IMAGE_TYPE_EN image_type);

/**
 * @name    释放图片空间
 * @param   image 待释放的图片地址
 * @retval  无
 */
void alg_codec_release_source_image(ALGSDK_IMAGE_ST *image);


/* 功能: 用于测试 codec api
@ 参数(入参): 空
@ 0
*/
int alg_codec_test_api(void);

#ifdef __cplusplus
}
#endif

#endif