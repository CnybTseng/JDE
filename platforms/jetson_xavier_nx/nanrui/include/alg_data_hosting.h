#ifndef ALG_DATA_HOSTING_H_INCLUDED
#define ALG_DATA_HOSTING_H_INCLUDED

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

////////////////////////////////////////////////////////////////////////////////
// 接口 - 函数

/*  算法数据托管接口
 *
 *  托管数据有两种: 结构化数据, 非结构化数据
 *    结构化数据: JSON字符串. 由键值对(Key:Value)组成
 *      示例:
 *        [{"Key":"Num","Value":1},{"Key":"Str","Value":"hello world"}]
 *  非结构化数据: 二进制文件. 可以是本地文件或者已读入内存数据
 */
struct alg_data_hosting_t
{
	int      type;    //数据类型. 结构化数据: 0; 非结构化数据: !0 (>0: 内存数据, <0: 本地数据)
	char    *title;   //待托管数据标题. 结构化数据: NULL; 非结构化数据: !NULL (文件名称)
	void    *content; //待托管数据内容. 结构化数据: JSON字符串; 非结构化数据: 内存地址或文件路径
	size_t   length;  //待托管数据内容长度.
};

/**
 * @name    算法数据PUT
 * @param   data: 待发布数据
 * @return  0:成功; <0:失败
 */
int alg_data_hosting_put(struct alg_data_hosting_t *data);

/**
 * @name    算法数据GET - 结构化数据获取
 * @param   label: 待获取数据Key字段
 * @param   content: 保存Value字段数据的空间地址
 * @param   length: 保存Value字段数据的空间大小
 * @return  0:成功; <0:失败
 */
int alg_data_hosting_get_value(const char *label, void *content, size_t length);

/**
 * @name    算法数据GET - 非结构化数据获取
 * @param   label: 待获取数据(文件名称)
 * @param   localpath: 保存获取数据的地址(文件路径)
 * @return  0:成功; <0:失败
 */
int alg_data_hosting_get_file(const char *label, const char *localpath);

/**
 * @name    数据托管初始化
 * @param   server_ip 数据托管代理地址
 * @return  0:成功; <0:失败
 */
int alg_data_hosting_init(const char *server_ip);

#ifdef __cplusplus
}
#endif

#endif
