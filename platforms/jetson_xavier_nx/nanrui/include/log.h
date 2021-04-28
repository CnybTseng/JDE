#ifndef PROMPT_H
#define PROMPT_H

/*
 * Interface to Log.
 *   int pub_log_file(const char *file);
 *   void pub_log(pub_log_level level, cosnt char *fmt, ...);
 *   void pub_error(const char *fmt, ...);
 *   void pub_warning(const char *fmt, ...);
 *   void pub_info(const char *fmt, ...);
 *   void pub_debug(const char *fmt, ...);
 *   int pub_debug_enable(void);
 *   int pub_debug_disable(void);
 */

/*
 * 实现各种调试及告警信息的输出
 */
typedef enum {
	PUB_LOG_LEVEL_ERROR		= 700,
	PUB_LOG_LEVEL_WARNING	= 600,
	PUB_LOG_LEVEL_INFO		= 500,
	PUB_LOG_LEVEL_DEBUG		= 400
} pub_log_level;

/* Interface to Log is just MACRO ! */
#define pub_module(x) do{_pub_module((x));}while(0)
//输出INFO信息
#define pub_info(...) do{                       \
        _pub_log(__FILE__,                      \
                 __LINE__,                      \
                 __FUNCTION__,                  \
                 PUB_LOG_LEVEL_INFO,            \
                 __VA_ARGS__);                  \
    }while(0)
//输出警告信息
#define pub_warning(...) do{                    \
        _pub_log(__FILE__,                      \
                 __LINE__,                      \
                 __FUNCTION__,                  \
                 PUB_LOG_LEVEL_WARNING,         \
                 __VA_ARGS__);                  \
    }while(0)

//输出出错信息
#define pub_error(...) do{                      \
        _pub_log(__FILE__,                      \
                 __LINE__,                      \
                 __FUNCTION__,                  \
                 PUB_LOG_LEVEL_ERROR,           \
                 __VA_ARGS__);                  \
    }while(0)

//输出调试信息
#define pub_debug(...) do{                      \
        _pub_log(__FILE__,                      \
                 __LINE__,                      \
                 __FUNCTION__,                  \
                 PUB_LOG_LEVEL_DEBUG,           \
                 __VA_ARGS__);                  \
    }while(0)

//输出日志信息
#define pub_log(level, ...) do{                 \
        _pub_log(__FILE__,                      \
                 __LINE__,                      \
                 __FUNCTION__,                  \
                 (level),                       \
                 ##__VA_ARGS__);                \
    }while(0)

#define pub_assert(expr) do{                    \
        _pub_assert(__FILE__,                   \
                    __LINE__,                   \
                    __FUNCTION__,               \
                    expr,                       \
                    #expr);                     \
    }while(0)

/* Don't directly use the following functions in your program ! */
#ifdef __cplusplus
extern "C" {
#endif
    int _pub_log(const char *file, int line, const char *function, pub_log_level level, const char *fmt, ...);
    void _pub_assert(const char *file, int line, const char *function, int expr, const char *expr_str);
    void _pub_module(const char *LogModule);
#ifdef __cplusplus
}
#endif

#endif /* PROMPT_H */

