#ifndef _ALGSDK_H_
#define _ALGSDK_H_

/* dependent include files */

#ifdef __cplusplus
extern "C" {
#endif

/* macro declarations */
#define ALG_ALARM_NAME_MAX_LEN 		128

/* type declarations */

/* 公共参数空间共享域枚举 */
typedef enum _ALGSDK_PUBLIC_PARAM_SHARE_EN
{
    SHARE_IN_ATOM = 0,      //算子内共享
    SHARE_IN_TASK,          //任务流内共享
    SHARE_IN_ALG_ABILITY,   //算法内共享
    SHARE_IN_GLOBAL         //全局共享
} ALGSDK_PUBLIC_PARAM_SHARE_EN;

/* 处理单元结构体 */
typedef struct _ALGSDK_PROCESS_UNIT_ST
{
    int id;                                                 //处理单元ID
    int percent;                                            //处理单元百分比
} ALGSDK_PROCESS_UNIT_ST;

/* 处理单元列表结构体 */
typedef struct _ALGSDK_PROCESS_UNIT_LIST_ST
{
    int unit_num;                                           //处理单元数目
    ALGSDK_PROCESS_UNIT_ST *unit_list;                      //处理单元列表
} ALGSDK_PROCESS_UNIT_LIST_ST;                              

/* 分析源句柄结构体 */
typedef struct _ALGSDK_SOURCE_HANDLE_ST
{
    int handle_num;                                         //句柄数量
    int *handle_group;                                      //句柄组
} ALGSDK_SOURCE_HANDLE_ST;

/* 图片格式枚举 */
typedef enum _ALGSDK_IMAGE_TYPE_EN{
	IMAGE_NV12 = 0,     /* SEMI-PLANAR Y4-U1V1 */
	IMAGE_NV21,         /* SEMI-PLANAR Y4-V1U1 */
	IMAGE_I420,         /* PLANAR Y4-U1-V1 */
	IMAGE_YV12,         /* PLANAR Y4-V1-U1 */
	IMAGE_YUYV,         /* 8 BIT PACKED Y2U1Y2V1 */
	IMAGE_UYVY,         /* 8 BIT PACKED U1Y2V1Y2 */
	IMAGE_YVYU,         /* 8 BIT PACKED Y2V1Y2U1 */
	IMAGE_VYUY,         /* 8 BIT PACKED V1Y2U1Y2 */
	IMAGE_BGR,
	IMAGE_RGB,
	IMAGE_BGRA,
	IMAGE_RGBA,
	IMAGE_ABGR,
	IMAGE_ARGB,
	IMAGE_MAX,
}ALGSDK_IMAGE_TYPE_EN;

#define IMAGE_PLANE_NUM (8U)
#define IMAGE_RESERVED_NUM (8U)

/* 图像数据结构体 */
typedef struct _ALGSDK_IMAGE_DATA_ST
{
	int format;								//取值为ALGSDK_IMAGE_TYPE_EN
	unsigned char* data[IMAGE_PLANE_NUM];	//图像数据指针
	int step[IMAGE_PLANE_NUM];				//各个平面的数据行步长
	void* reserved[IMAGE_RESERVED_NUM];		//保留字段
}ALGSDK_IMAGE_DATA_ST;

/* 图片结构体 */
typedef struct _ALGSDK_IMAGE_ST
{
	char *source_url;					//图像源URL
	int width;							//图像宽
	int height;							//图像高
	long long pts;						//时间戳
	ALGSDK_IMAGE_TYPE_EN image_type;	//图像输出类型    
	ALGSDK_IMAGE_DATA_ST image_data;    //图像数据
}ALGSDK_IMAGE_ST;

/* 点信息结构体 */
typedef struct _ALGSDK_POINT_ST
{
	int id;                                            //点的id
	int x;                                             //点的x坐标(绝对坐标)
	int y;                                             //点的y坐标(绝对坐标)
	char *desc;                                        //点的预留字段，多个字符串用","分开
} ALGSDK_POINT_ST;

/* 规则点信息结构体 */      //2020-9-29 更新
typedef struct _ALGSDK_RULE_POINT_ST
{
    char *id;                                          //点的id
    int x;                                             //点的x相对坐标 (X相对坐标 = X实际坐标/实际图像宽度 * 10000)
    int y;                                             //点的y相对坐标 (Y相对坐标 = Y实际坐标/实际图像高度 * 10000)
    char *desc;                                        //点的预留字段，多个字符串用","分开
} ALGSDK_RULE_POINT_ST;

/* 规则框信息结构体 */      //2020-9-29 更新
typedef struct _ALGSDK_RULE_BOX_ST
{
    char *id;                                           //规则id
    char *type;                                         //绘图类型
    char *desc;                                         //规则的预留字段，多个字符串用","分开
    unsigned int point_num;                             //规则中的点的数目
    ALGSDK_RULE_POINT_ST *point_list;                   //规则中的点信息
} ALGSDK_RULE_BOX_ST;

/* 目标框颜色枚举 */
typedef enum _ALGSDK_OBJ_COLOR_EN{
    OBJ_COLOR_RED = 0,                              //红色, 告警目标
    OBJ_COLOR_GREEN,                                //绿色, 非告警目标
    OBJ_COLOR_YELLOW,                               //黄色
    OBJ_COLOR_BLUE,                                 //蓝色
}ALGSDK_OBJ_COLOR_EN;

/* 目标框信息结构体 */
typedef struct _ALGSDK_OBJECT_BOX_ST
{
	char *desc;                                         //目标描述，比如标签，目标置信度等
	ALGSDK_OBJ_COLOR_EN color;                          //目标与标签的绘制颜色
	unsigned int point_num;                             //目标位置的点数目
	ALGSDK_POINT_ST *point_list;                  //目标的位置信息
} ALGSDK_OBJECT_BOX_ST;

/* 规则列表 */
typedef struct _ALGSDK_RULE_LIST_ST
{
	unsigned int rule_num;                              //规则框数目
	ALGSDK_RULE_BOX_ST *rule_list;                //规则框列表
} ALGSDK_RULE_LIST_ST;

/* 目标列表 */
typedef struct _ALGSDK_OBJ_LIST_ST
{
	unsigned int obj_num;                               //目标框数量
	ALGSDK_OBJECT_BOX_ST *obj_list;               //目标框列表(空间由使用方管理)
} ALGSDK_OBJ_LIST_ST;

//告警数据结构体定义
typedef struct _ALG_ALARM_IMAGE_DESC_ST
{
	char name[ALG_ALARM_NAME_MAX_LEN];                  //告警图像名称
	int width;                                          //图像宽度
	int height;                                         //高度
	unsigned long size;                                 //图片大小
} ALG_ALARM_IMAGE_DESC_ST;

/* 告警推送入参结构体 */
typedef struct _ALG_ALARM_EVENT_PUSH_ST
{
	char desc[ALG_ALARM_NAME_MAX_LEN];                  //结果描述
	unsigned int image_num;                             //告警图片数量
	ALG_ALARM_IMAGE_DESC_ST *image_list;                //告警图片列表。为图片列表连续内存空间起始地址，大小为image_num * sizeof(ALG_ALARM_IMAGE_DESC_ST)
} ALG_ALARM_EVENT_PUSH_ST;

/* variable declarations */

/* function declarations */

/**
 * @name    SDK总初始化
 * @brief   用户需要在main最开始时调用
 * @retval  0:成功;   <0:失败
 */
int algsdk_init(void);

/**
 * @name    注册用户自定义退出回调
 * @param   exit_cb     用户自定义的退出回调
 * @retval  0:成功;   <0:失败
 */
int algsdk_user_exit_regist(void (*exit_cb)(void));

/**
 * @name    算法能力注册
 * @param   alg_id      算法ID
 * @param   desc        算法描述
 * @param   init_cb     算法任务初始化回调
 * @param   exit_cb     算法任务退出回调
 * @param   atom_num    算子数量
 * @param   atom_group  算子回调组(按顺序罗列)
 * @retval  0:成功;   <0:失败
 */
int algsdk_alg_ability_regist(
    unsigned int alg_id, 
    const char *desc, 
    int (*init_cb)(void),
    void (*exit_cb)(void),
    unsigned int atom_num,
    void *(**atom_group)(void *));

/**
 * @name    内存空间申请
 * @param   size    空间大小
 * @retval  !0:成功, 返回空间所在指针;   NULL:失败
 * @note    无须用户释放, 在一帧数据跑完所有运行回调后自动释放, 无法长期保存, 建议主要用于算子间结果传递
 */
void *algsdk_malloc(unsigned int size);

/**
 * @name    公共参数空间获取
 * @param   name    公共参数名
 * @param   scope   公共参数共享域(算子内共享, 任务流内共享, 算法内共享, 全局共享)
 * @param   size    空间大小
 * @retval  !0:成功, 返回空间所在指针;   NULL:失败
 * @note    首次获取会分配新空间地址, 之后会根据name和scope返回之前所申请的空间地址.
 * @note    只能在算子运行回调和初始化回调中调用; 当共享域为SHARE_IN_ATOM时, 不能在初始化回调中调用; 
 * @note    共享域解释:
 *          1. 算子内共享(每个算子申请自己的参数空间, 空间内的数据会持续保存直至任务被销毁, 算子间同名参数不共享空间, 类似static类型的局部变量)
 *          2. 任务流内共享(同一个任务流内的不同算子间如果获取的参数name相同, 则共用一个参数空间, 数据保存直至任务被销毁)
 *          3. 算法内共享(同一个算法能力的所有任务如果获取的参数name相同, 则共用一个空间, 数据保存直到程序退出)
 *          4. 全局共享(容器内所有算法能力的所有任务如果获取的参数name相同, 则共用一个空间, 数据保存直到程序退出)
 */
void *algsdk_get_public_param(char *name, ALGSDK_PUBLIC_PARAM_SHARE_EN scope, unsigned int size);

/**
 * @name    设置图片类型
 * @param   image_type      图片类型(枚举量待定)
 * @retval  0:成功;   <0:失败
 */
int algsdk_set_image_type(int image_type);

/**
 * @name    设置默认频率
 * @param   frequency_hz      默认频率(Hz), 1 ~ 25
 * @retval  0:成功;   <0:失败
 */
int algsdk_set_default_frequency(unsigned long frequency_hz);

/**
 * @name    获取算法处理单元
 * @retval  !0:成功, 返回处理单元列表结构体;   NULL:失败
 * @note    空间无须用户释放
 */
ALGSDK_PROCESS_UNIT_LIST_ST *algsdk_get_process_unit(void);

/**
 * @name    获取分析源句柄
 * @retval  !0:成功, 返回句柄结构体;   NULL:失败
 * @note    空间无须用户释放
 */
ALGSDK_SOURCE_HANDLE_ST *algsdk_get_source_handle(void);

/**
 * @name    获取分析源图片
 * @param   source_handle   分析源句柄
 * @retval  !0:成功, 返回分析源图片数据;   NULL:失败
 * @note    空间由用户释放
 */
ALGSDK_IMAGE_ST *algsdk_get_source_image(int source_handle);

/**
 * @name    释放图片空间
 * @param   image 待释放的图片地址
 */
void algsdk_release_source_image(ALGSDK_IMAGE_ST *image);

/**
 * @name    获取分析源Rule
 * @param   source_handle   分析源句柄
 * @retval  !0:成功, 返回分析源Rule;   NULL:失败 或 分析源没有Rule
 * @note    空间无须用户释放, 但内容只读
 */
ALGSDK_RULE_LIST_ST *algsdk_get_source_rule(int source_handle);

/**
 * @name    释放分析源Rule
 * @param   rule 待释放的析源Rule
 */
void algsdk_release_source_rule(ALGSDK_RULE_LIST_ST *rule);

/**
 * @name    获取分析源URL
 * @param   source_handle   分析源句柄
 * @retval  !0:成功, 返回分析源URL;   NULL:失败
 * @note    空间由用户释放, 用free释放
 */
char *algsdk_get_source_url(int source_handle);

/**
 * @name    元数据结果上报
 * @param   desc    结果描述
 * @param   push_obj    推送目标列表
 * @param   source_url  数据源URL
 * @param   pts         这帧图像的时间戳
 * @retval  0:成功;   <0:失败
 * @note    push_obj空间由用户管理, 推送完释放.
 */
int algsdk_push_metadata(char *desc, ALGSDK_OBJ_LIST_ST *push_obj, char *source_url, long long pts);

/**
 * @name    告警间隔检测
 * @return  0:成功; -1:失败;1:未超出过滤时间间隔
 */
int algsdk_alarm_time_check(void);

/**
 * @name    告警事件推送
 * @param   alarm 告警推送信息
 * @return  0:成功; -1:失败; 1:未超出告警过滤间隔
 * @note    需要对alarm_time格式做严格校验
 * @note    阻塞执行直到收到应答返回结果
 */
int algsdk_alarm_event_push(ALG_ALARM_EVENT_PUSH_ST *alarm);

/**
//  * @name    告警结果上报
//  * @param   desc    结果描述
//  * @param   source_url    数据源URL
//  * @param   multi_alarm_image   告警图片组(含目标框 和 图片)
//  * @retval  0:成功;   <0:失败
//  */
// int algsdk_push_alarm(char *desc, char *source_url, ALGSDK_ALARM_MULTI_PUSH_ST *multi_alarm_image);

// /**
//  * @name    申请ALGSDK_ALARM_MULTI_PUSH_ST结构体的空间
//  * @param   alarm_image_num    告警推送图片数量
//  * @retval  !0:成功, 返回空间;   NULL:失败
//  * @note    申请后的空间里ALGSDK_IMAGE_ST *和char *name的指针都是NULL, 需要用户自行填充.
//  * @note    用户在告警推送完毕后, 无需释放空间, 包括在内的图片空间和name空间都无需释放
//  * @note    约定char *name用strdup或malloc分配空间, 需要free释放
//  */
// ALGSDK_ALARM_MULTI_PUSH_ST *algsdk_malloc_alarm_multi_push(unsigned int alarm_image_num);

/**
 * @name    设置点信息
 * @param   point    待设置点
 * @param   id       点ID
 * @param   x        点坐标X
 * @param   y        点坐标Y
 * @param   desc     点描述(可填NULL)
 * @retval  0:成功;   <0:失败
 */
int algsdk_set_point(ALGSDK_POINT_ST *point, int id, int x, int y, const char *desc);

/**
 * @name    释放点资源
 * @param   point    待释放点
 */
void algsdk_free_point(ALGSDK_POINT_ST *point);

/**
 * @name    将点添加进目标框中
 * @param   obj_box   目标框
 * @param   point     待添加点
 * @retval  0:成功;   <0:失败
 * @note    被添加的点资源会转移到目标框内, 添加后的点资源无须释放.
 */
int algsdk_add_point_to_obj_box(ALGSDK_OBJECT_BOX_ST *obj_box, ALGSDK_POINT_ST *point);

/**
 * @name    设置目标框信息
 * @param   obj_box   目标框
 * @param   desc      目标框描述(可填NULL)
 * @param   color     目标框颜色
 * @retval  0:成功;   <0:失败
 */
int algsdk_set_obj_box(ALGSDK_OBJECT_BOX_ST *obj_box, const char *desc, ALGSDK_OBJ_COLOR_EN color);

/**
 * @name    释放目标框资源
 * @param   obj_box    待释放目标框
 * @note    会一同释放内部所含的点资源
 */
void algsdk_free_obj_box(ALGSDK_OBJECT_BOX_ST *obj_box);

/**
 * @name    将目标框添加进目标列表中
 * @param   obj_list  目标列表
 * @param   obj_box   待添加目标框
 * @retval  0:成功;   <0:失败
 * @note    被添加的目标框资源会转移到目标列表内, 添加后的目标框资源无须释放.
 */
int algsdk_add_obj_box_to_list(ALGSDK_OBJ_LIST_ST *obj_list, ALGSDK_OBJECT_BOX_ST *obj_box);

/**
 * @name    释放目标列表资源
 * @param   obj_list    待释放目标列表
 * @note    会一同释放内部所含的目标框资源和点资源
 */
void algsdk_free_obj_list(ALGSDK_OBJ_LIST_ST *obj_list);

/**
 * @name    拷贝目标框资源
 * @param   src_obj_box    源目标框
 * @param   dst_obj_box    终目标框
 * @retval  0:成功;   <0:失败
 */
int algsdk_copy_obj_box(ALGSDK_OBJECT_BOX_ST *src_obj_box, ALGSDK_OBJECT_BOX_ST *dst_obj_box);

/**
 * @name    拷贝目标列表资源
 * @param   src_obj_list    源目标列表
 * @param   dst_obj_list    终目标列表
 * @retval  0:成功;   <0:失败
 */
int algsdk_copy_obj_list(ALGSDK_OBJ_LIST_ST *src_obj_list, ALGSDK_OBJ_LIST_ST *dst_obj_list);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _ALGSDK_H_ */
