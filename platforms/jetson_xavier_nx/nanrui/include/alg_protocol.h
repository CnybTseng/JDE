#ifndef ALG_PROTOCOL_H_INCLUDED
#define ALG_PROTOCOL_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

////////////////////////////////////////////////////////////////////////////////
// 宏定义

#define ALG_PROTOCOL_ALARM_TIME_LEN 20

////////////////////////////////////////////////////////////////////////////////
// 接口 - 类型

typedef struct _ALGSDK_RULE_POINT_ST        ALG_PROTOCOL_POINT_ST;
typedef struct _ALGSDK_RULE_BOX_ST          ALG_PROTOCOL_RULE_BOX_ST;
typedef struct _ALGSDK_RULE_LIST_ST         ALG_PROTOCOL_RULE_LIST_ST;
typedef struct _ALGSDK_OBJECT_BOX_ST        ALG_PROTOCOL_OBJECT_BOX_ST;
typedef struct _ALGSDK_OBJ_LIST_ST          ALG_PROTOCOL_LIST_ST;
typedef struct _ALGSDK_PROCESS_UNIT_ST      ALG_PROTOCOL_PROCESS_UNIT_ST;
typedef struct _ALGSDK_PROCESS_UNIT_LIST_ST ALG_PROTOCOL_PROCESS_UNIT_LIST_ST;

/* 数据源结构体 */
typedef struct _ALG_PROTOCOL_SOURCE_ST
{
	int source_type;                                    //数据源类型
	char *source_url;                                   //数据源地址
	ALG_PROTOCOL_RULE_LIST_ST *source_rule;             //规则框列表
} ALG_PROTOCOL_SOURCE_ST;

/* 数据源详情结构体 */
typedef struct _ALG_PROTOCOL_SOURCE_LIST_ST
{
	unsigned int source_num;                            //数据源数目
	ALG_PROTOCOL_SOURCE_ST *source_list;                //数据源列表
} ALG_PROTOCOL_SOURCE_LIST_ST;

typedef struct _ALG_PROTOCOL_ALARM_IMAGE_ST
{
	char *name;                                         //告警图像名称(空间由使用方管理)
	int type;                                           //数据类型
	int width;                                          //图像宽度
	int height;                                         //高度
	unsigned long size;                                 //图片大小
	void *data;                                         //图片数据(空间由使用方管理)
} ALG_PROTOCOL_ALARM_IMAGE_ST;

/* 告警推送入参结构体 */
typedef struct _ALG_PROTOCOL_ALARM_PUSH_ST
{
	char *task_id;                                      //算法任务号(空间由使用方管理)
	int source_type;                                    //数据源类型, 0:rtsp,1:jpg
	char *source_url;                                   //数据源地址(空间由使用方管理)
	unsigned int alg_id;                                //告警依据的算法编码
	char *desc;                                         //结果描述(空间由使用方管理)
	char alarm_time[ALG_PROTOCOL_ALARM_TIME_LEN];       //告警时间戳，格式"yyyy-mm-dd hh:mm:ss"
	unsigned int image_num;                             //告警图片数量
	ALG_PROTOCOL_ALARM_IMAGE_ST *image_list;            //告警图片列表
} ALG_PROTOCOL_ALARM_PUSH_ST;

/* 元数据推送入参结构体 */
typedef struct _ALG_PROTOCOL_METADATA_PUSH_ST
{
	char *task_id;                                      //算法任务号(空间由使用方管理)
	int source_type;                                    //数据源类型, 0:rtsp,1:jpg
	char *source_url;                                   //数据源地址(空间由使用方管理)
	unsigned int alg_id;                                //告警依据的算法编码
	char *desc;                                         //结果描述(空间由使用方管理)
	long long pts;                                      //时间戳
	ALG_PROTOCOL_LIST_ST *object;                       //目标框列表
} ALG_PROTOCOL_METADATA_PUSH_ST;

/* 异常日志推送入参结构体 */
typedef struct _ALG_PROTOCOL_ALG_LOG_PUSH_ST
{
	char *task_id;                                      //算法任务号(空间由使用方管理)
	char *file;                                         //异常文件名(空间由使用方管理)
	unsigned long line;                                 //异常行数
	char *func;                                         //异常函数名(空间由使用方管理)
	char *desc;                                         //异常日志信息(空间由使用方管理)
} ALG_PROTOCOL_ALG_LOG_PUSH_ST;

////////////////////////////////////////////////////////////////////////////////
// 接口 - 函数

/**
 * @name    算法协议初始化
 * @return  0:成功; <0:失败
 * @note    主要用于挂载处理回调
 */
int alg_protocol_init(void);

/**
 * @name    算法协议反初始化
 * @return  0:成功; <0:失败
 */
int alg_protocol_deinit(void);

/**
 * @name    算法能力获取处理回调
 * @param   json_in 请求json报文
 * @return  !0:成功, 返回应答json报文; NULL:失败
 * @note    返回值是malloc的空间, 用完后需要手动free.
 */
char *alg_protocol_alg_ability_request(const char *json_in);

/**
 * @name    算法任务状态获取处理回调
 * @param   json_in 请求json报文
 * @return  !0:成功, 返回应答json报文; NULL:失败
 * @note    返回值是malloc的空间, 用完后需要手动free.
 */
char *alg_protocol_alg_status_request(const char *json_in);

/**
 * @name    算法任务配置处理回调
 * @param   json_in 请求json报文
 * @return  !0:成功, 返回应答json报文; NULL:失败
 * @note    返回值是malloc的空间, 用完后需要手动free.
 */
char *alg_protocol_alg_task_config(const char *json_in);

/**
 * @name    算法任务控制处理回调
 * @param   json_in 请求json报文
 * @return  !0:成功, 返回应答json报文; NULL:失败
 * @note    返回值是malloc的空间, 用完后需要手动free.
 */
char *alg_protocol_alg_task_control(const char *json_in);

/**
 * @name    告警推送
 * @param   url 推送URL地址
 * @param   alarm 告警推送信息
 * @return  0:成功; <0:失败
 * @note    需要对alarm_time格式做严格校验
 * @note    阻塞执行直到收到应答返回结果
 */
int alg_protocol_alg_alarm_push(const char *url, ALG_PROTOCOL_ALARM_PUSH_ST *alarm);

/**
 * @name    元数据推送
 * @param   url 推送URL地址
 * @param   metadata 元数据推送信息
 * @return  0:成功; <0:失败
 * @note    阻塞执行直到收到应答返回结果
 */
int alg_protocol_metadata_push(const char *url, ALG_PROTOCOL_METADATA_PUSH_ST *metadata);

/**
 * @name    异常日志推送
 * @param   url 推送URL地址
 * @param   log 异常日志推送信息
 * @return  0:成功; <0:失败
 * @note    阻塞执行直到收到应答返回结果
 */
int alg_protocol_alg_log_push(const char *url, ALG_PROTOCOL_ALG_LOG_PUSH_ST *log);

/**
 * @name    算法任务数据源和规则配置, 空间手动free释放
 * @param   list 待释放内存空间
 */
void alg_protocol_free_sourceandrule(ALG_PROTOCOL_SOURCE_LIST_ST *list);

/**
 * @name    算法任务处理单元配置, 空间手动free释放
 * @param   process_unit_list 待释放内存空间
 */
void alg_protocol_free_processunit(ALG_PROTOCOL_PROCESS_UNIT_LIST_ST *list);

/**
 * @name    算法任务规则配置复制
 * @param   list 待复制的规则配置
 * @return  !NULL:成功; NULL:失败
 */
ALG_PROTOCOL_RULE_LIST_ST *alg_protocol_copy_rule(ALG_PROTOCOL_RULE_LIST_ST *list);

/**
 * @name    算法任务规则配置释放
 * @param   list 待释放内存空间
 */
void alg_protocol_free_rule(ALG_PROTOCOL_RULE_LIST_ST *list);

#ifdef __cplusplus
}
#endif

#endif
