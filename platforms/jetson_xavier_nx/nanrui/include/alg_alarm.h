#ifndef ALG_ALARM_H_INCLUDED
#define ALG_ALARM_H_INCLUDED

#include "algsdk.h"

#ifdef __cplusplus
extern "C"
{
#endif

////////////////////////////////////////////////////////////////////////////////
// 宏定义

#define ALG_ALARM_SYS_TIME_LEN 		20
////////////////////////////////////////////////////////////////////////////////

typedef struct _ALGSDK_ALARM_EVENT_TO_QUEUE_ST
{
    ALG_ALARM_EVENT_PUSH_ST push_data;
    char *pTaskId;
    int pAlgId;
	char alarm_time[ALG_ALARM_SYS_TIME_LEN];       							//告警时间戳，格式"yyyy-mm-dd hh:mm:ss"
} ALGSDK_ALARM_EVENT_TO_QUEUE_ST;

/**
 * @name    告警事件推送
 * @param   alarm 告警推送信息
 * @return  0:成功; -1:失败; 1:未超出告警过滤间隔
 * @note    需要对alarm_time格式做严格校验
 * @note    阻塞执行直到收到应答返回结果
 */
int algsdk_alarm_event_push(ALG_ALARM_EVENT_PUSH_ST *alarm);

/**
 * @name    告警事件推送初始化
 * @param   alarm 告警推送信息
 * @return  0:成功; <0:失败
 */
int algsdk_alarm_init(void);

/**
 * @name    告警事件推送线程退出
 */
void algsdk_alarm_exit(void);

#ifdef __cplusplus
}
#endif

#endif
