#ifndef _ALG_TASK_MNG_H_
#define _ALG_TASK_MNG_H_

/* dependent include files */
#include "alg_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

/* macro declarations */
#define ALG_TASK_MNG_METADATA_NAME      "metadata"
#define ALG_TASK_MNG_METADATA_URL_SIZE      (256)

#define ALG_TASK_MNG_ALARM_NAME         "alarm"
#define ALG_TASK_MNG_ALARM_LAST_TIME_NAME   "alarm_last_time"
#define ALG_TASK_MNG_ALARM_URL_SIZE      (256)

#define ALG_TASK_MNG_SOURCE_NAME         "source_and_rule"

#define ALG_TASK_MNG_PROCESS_UNIT_NAME         "process_unit"

#define ALG_TASK_MNG_IMAGE_TYPE_NAME         "image_type"


/* type declarations */
typedef struct _ALG_TASK_MNG_ALARM_ST
{
    unsigned long alarminterval;                    //告警间隔
    // unsigned long last_alarm_time;                  //上次告警时间
    char alarmurl[ALG_TASK_MNG_ALARM_URL_SIZE];     //告警推送URL
} ALG_TASK_MNG_ALARM_ST;

typedef enum _ALG_TASK_MNG_MUTEX_EN
{
    TASK_MUTEX_SOURCE = 0,                          //source_and_rule资源锁
    TASK_MUTEX_MAX                                  //资源锁末尾
} ALG_TASK_MNG_MUTEX_EN;

typedef ALG_PROTOCOL_SOURCE_LIST_ST ALG_TASK_MNG_SOURCE_ST;

typedef ALG_PROTOCOL_PROCESS_UNIT_LIST_ST ALG_TASK_MNG_PROCESS_UNIT_ST;

/* variable declarations */

/* function declarations */

/**
 * @name    任务状态查询
 * @param   task_id   任务ID
 * @param   err_msg   出参, 错误信息
 * @return  >=0:成功, 返回任务状态;   <0:失败
 * @note    err_msg若有错误信息, 用完后需要手动free释放.
 */
int alg_task_mng_get_state(char *task_id, char **err_msg);

/**
 * @name    任务控制
 * @param   task_id   任务ID
 * @param   cmd       控制命令
 * @param   err_msg   出参, 错误信息
 * @return  0:成功;   <0:失败
 * @note    err_msg若有错误信息, 用完后需要手动free释放.
 */
int alg_task_mng_ctrl(char *task_id, int cmd, char **err_msg);


/**
 * @name    任务配置(创建任务)
 * @param   task_id   任务ID
 * @param   alg_id    算法ID
 * @param   err_msg   出参, 错误信息
 * @return  0:成功, 返回任务状态;   <0:失败
 * @note    err_msg若有错误信息, 用完后需要手动free释放.
 */
int alg_task_mng_set_create_task(char *task_id, unsigned int alg_id, char **err_msg);

/**
 * @name    任务配置(元数据)
 * @param   task_id   任务ID
 * @param   alg_id    算法ID
 * @param   metadataurl 元数据推送URL
 * @param   err_msg   出参, 错误信息
 * @return  0:成功, 返回任务状态;   <0:失败
 * @note    err_msg若有错误信息, 用完后需要手动free释放.
 */
int alg_task_mng_set_metadata(char *task_id, unsigned int alg_id, char *metadataurl, char **err_msg);

/**
 * @name    任务配置(告警)
 * @param   task_id   任务ID
 * @param   alg_id    算法ID
 * @param   alarmurl  告警推送URL
 * @param   alarminterval   告警间隔
 * @param   err_msg   出参, 错误信息
 * @return  0:成功, 返回任务状态;   <0:失败
 * @note    err_msg若有错误信息, 用完后需要手动free释放.
 */
int alg_task_mng_set_alarm(char *task_id, unsigned int alg_id, char *alarmurl, int alarminterval, char **err_msg);

/**
 * @name    任务配置(数据源)
 * @param   task_id   任务ID
 * @param   alg_id    算法ID
 * @param   source_and_rule 数据源
 * @param   err_msg   出参, 错误信息
 * @return  0:成功, 返回任务状态;   <0:失败
 * @note    err_msg若有错误信息, 用完后需要手动free释放.
 */
int alg_task_mng_set_source_and_rule(char *task_id, unsigned int alg_id, ALG_PROTOCOL_SOURCE_LIST_ST *source_and_rule, char **err_msg);

/**
 * @name    任务配置(处理单元)
 * @param   task_id   任务ID
 * @param   alg_id    算法ID
 * @param   process_unit_list 处理单元列表
 * @param   err_msg   出参, 错误信息
 * @return  0:成功, 返回任务状态;   <0:失败
 * @note    err_msg若有错误信息, 用完后需要手动free释放.
 */
int alg_task_mng_set_process_unit(char *task_id, unsigned int alg_id, ALG_PROTOCOL_PROCESS_UNIT_LIST_ST *process_unit_list, char **err_msg);

/**
 * @name    通过流句柄获取任务ID
 * @param   stream_handle   任务流句柄
 * @return  !0:成功, 返回任务ID;   NULL:失败
 * @note    用完后需要手动free释放.
 */
char *alg_task_mng_get_task_id_by_stream_handle(int stream_handle);

/**
 * @name    通过流句柄给资源上锁
 * @param   stream_handle   任务流句柄
 * @param   mutex_idx       资源锁下标
 */
void alg_task_mng_lock_by_stream_handle(int stream_handle, int mutex_idx);

/**
 * @name    通过流句柄给资源解锁
 * @param   stream_handle   任务流句柄
 * @param   mutex_idx       资源锁下标
 */
void alg_task_mng_unlock_by_stream_handle(int stream_handle, int mutex_idx);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _ALG_TASK_MNG_H_ */
