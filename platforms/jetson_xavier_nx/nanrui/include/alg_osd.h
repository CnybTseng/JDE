#ifndef _ALG_OSD_H_
#define _ALG_OSD_H_

#include "algsdk.h"
#include "alg_data_hosting.h"

/* 算法数据托管接口 */
typedef struct alg_data_hosting_t ALG_DATA_HOSTING_T;

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @name    在指定格式图像上绘制元数据信息
 * @param   name osd图片名称
 * @param   image 待绘制的图片地址
 * @param   alg_meta 待叠加的元数据
 * @retval   !0:成功, 返回叠加元数据+jpg编码的告警图片;   nullptr:失败
 */
ALG_DATA_HOSTING_T* alg_osd_get_osd_image(char *name, ALGSDK_IMAGE_ST* image, ALGSDK_OBJ_LIST_ST* alg_meta);

/**
 * @name    释放OSD图像
 * @param   image 待绘制的图片地址
 * @retval   无
 */
void alg_osd_release_osd_image(ALG_DATA_HOSTING_T* alarm_image);

//对获取到的图片进行osd叠加 并保存成文件
void test_alg_osd_get_osd_image(ALGSDK_IMAGE_ST* image);

#ifdef __cplusplus
}
#endif

#endif