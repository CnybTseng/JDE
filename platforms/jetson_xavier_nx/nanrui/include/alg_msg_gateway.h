#ifndef ALG_MSG_GATEWAY_H_INCLUDED
#define ALG_MSG_GATEWAY_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

////////////////////////////////////////////////////////////////////////////////
// 接口 - 函数

/**
 * @name    算法消息网关消息回调设置
 * @param   on_message 消息回调函数
 * @note    void (*on_message)(const char *topic, const char *from, const char *to, void *payload, int payloadlen)
 *          on_message函数参数类型与主题发布一致, 在客户端收到消息后, 会回调该函数;
 *          同时, 会将消息的内容作为参数传递给回调函数.
 */
void alg_msg_gateway_callback_set(void (*on_message)(const char *topic, const char *from, const char *to, void *payload, int payloadlen));

/**
 * @name    算法消息网关主题发布
 * @param   topic 主题
 * @param   from 发送方
 * @param   to 接收方
 * @param   payload 待发布数据地址
 * @param   payloadlen 待发布数据大小
 * @return  0:成功; <0:失败
 * @note    topic为发布的主题, 主题可以分层级使用/分割, 如: topic/a/b/c.
 *          from代表发布层级, 如: 发布者的任务号 - taskid1, from为NULL时, 该层级用""表示
 *          to代表接收方层级, 如: 接收者的任务号 - taskid2, to为NULL时, 该层级用""表示
 */
int alg_msg_gateway_publish(const char *topic, const char *from, const char *to, void *payload, int payloadlen);

/**
 * @name    算法消息网关主题订阅
 * @param   topic 主题
 * @param   from 发送方
 * @param   to 接收方
 * @return  0:成功; <0:失败
 * @note    topic为订阅的主题, 主题可以分层级使用/分割, 如: topic/a/b/c.
 *          from代表发布层级, 如: 发布者的任务号 - taskid1, from为NULL时, 表示通配单个层级
 *          to代表接收方层级, 如: 接收者的任务号 - taskid2, to为NULL时, 表示通配单个层级
 */
int alg_msg_gateway_subscribe(const char *topic, const char *from, const char *to);

/**
 * @name    算法消息网关主题取消订阅
 * @param   topic 主题
 * @param   from 发送方
 * @param   to 接收方
 * @return  0:成功; <0:失败
 * @note    取消订阅功能用于取消已订阅的主题, 参数与订阅一致
 */
 int alg_msg_gateway_unsubscribe(const char *topic, const char *from, const char *to);

/**
 * @name    算法消息网关初始化
 * @param   server_ip 网关代理地址
 * @return  0:成功; <0:失败
 */
int alg_msg_gateway_init(const char *server_ip);

/**
 * @name    算法消息网关反初始化
 * @return  0:成功; <0:失败
 */
int alg_msg_gateway_deinit(void);

#ifdef __cplusplus
}
#endif

#endif
