
#ifndef _ALG_CAMERA_CONTROL_H
#define _ALG_CAMERA_CONTROL_H

#include <stdio.h>
#include <stdlib.h>

#define  CAM_CONTROL_TERMID_LEN   64
#define  CAM_CONTROL_SRCURL_LEN   128
#define  CAM_CONTROL_TASKID_LEN   128

#ifdef __cplusplus
extern "C"
{
#endif

//通道控制_消息结构体
typedef struct __CAMERA_CONTROL_MSG_ST
{
    char *msg_payload;
    int  msg_payloadlen;
} CAMERA_CONTROL_MSG_ST;

//通道控制_执行消息结构体
typedef struct __CAMERA_CONTROL_STATUS_ST
{
    char SrcUrl[CAM_CONTROL_SRCURL_LEN];            //视频分析源地址
    char ChanTermId[CAM_CONTROL_TERMID_LEN];        //通道号：设备id
    char TaskId[CAM_CONTROL_TASKID_LEN];            //任务号
    int Object;                                     //操作类型：0：云台控制，3：预置位调度
    int Direction;                                  //方向控制。Object为3时，代表预置位序号
    int Action;                                     //Object为0：0->停止，1->开始；Obejct为3：2->预置位调度
    int Speed;                                      //速度控制：取值[1-9]
    int IsChanAvailable;                            //通道控制是否可用。0->空闲；1->占用
    int CamStatus;                                  //设备调度状态。-1->调度失败；0->准备调度，请等待；1->调度成功
} CAMERA_CONTROL_STATUS_ST;

/**
 *  @name   获取通道控制权
 *  @param  pSrcUrl         入参：通道对应资源url
 *  @return                 非空：成功，返回控制句柄（通道设备号）; NULL：失败          
 */
char *alg_camera_control_get_help(const char *pSrcUrl);

/**
 *  @name   进行通道设备控制
 *  @param  pHandle         入参：控制句柄（通道设备号）,由alg_camera_control_get_help接口返回的非空字符串
 *  @param  Object          入参：控制类别  
 *           0:云台控制, 
 *           3：预置位调度
 *  @param  Direction       入参：转动方向  
 * Obejct为0时：
 *           0：上
 *           1：下
 *           2：左
 *           3：右
 *           6：左上
 *           7：右上
 *           8：左下
 *           9：右下
 *  Obejct为3时：
 *  取值范围[0-127]
 *  @param  Action          入参：控制动作
 *  Obejct为0时：
 *           0：停止
 *           1：开始
 *  Obejct为3时：
 *           2：预置位调度
 *  @param  Speed           入参：控制转动速度，取值[1-9]
 *  @return                 -1：调度执行失败，0：准备调度请等待；1->调度执行成功
 */
int alg_camera_control_handler(const char *pHandle, int Object, int Direction, int Action, int Speed);

/**
 *  @name   释放通道控制权（正式对外）
 *  @param  pHandle         入参：控制句柄（通道设备号）
 *  @return                 0：成功; -1：失败          
 */
int alg_camera_control_release_handle(const char *pHandle);

/**
 *  @name   释放通道控制权
 *  @param  srcurl          入参：控制句柄（通道设备号）
 *  @return                 0：成功; -1：失败          
 */
int alg_camera_control_release_help(const char *srcurl);

/**
 *  @name   通道控制服务初始化
 *  @return                 0：成功; -1：失败          
 */
int alg_camera_control_init(const char * ServerIp);

/**
 *  @name   通道控制服务退出           
 */
void alg_camera_control_exit(void);

#ifdef __cplusplus
}
#endif

#endif  //头文件定义