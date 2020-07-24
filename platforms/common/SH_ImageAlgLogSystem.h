/******************************************************************************

                  版权所有 (C), 2004-2017, 成都思晗科技股份有限公司

 ******************************************************************************
  文 件 名   : SH_ImageAlgLogSystem.h
  版 本 号   : 初稿
  作    者   : 苏永生
  生成日期   : 2017年9月12日
  最近修改   :
  功能描述   : 日志回调接口
  函数列表   :
*
*

  修改历史   :
  1.日    期   : 2017年9月12日
    作    者   : 苏永生
    修改内容   : 创建文件
  2.日    期   : 2020年3月23
    作    者   : Zeng Zhiwei
    修改内容   : 删除函数 ImgAlgExcuteLogFun 的默认实参

******************************************************************************/
#pragma once
#include "string.h"
#include "mot.h"
 
#ifndef _ENUM_COMMON_LOG_LEVEL_
#define _ENUM_COMMON_LOG_LEVEL_
typedef enum ENUM_CommonLogLevel
{
    E_CommonLog_Trace,
    E_CommonLog_Debug,
    E_CommonLog_Info,
    E_CommonLog_Warn,
    E_CommonLog_Error,
    E_CommonLog_Fatal,
}E_CommonLogLevel;

typedef void (*PFUN_LogSystemCallBack)(E_CommonLogLevel level, const char * p_log_info, const char * p_file, int line);

#endif




/*****************************************************************************
 函 数 名  : ImgAlgExcuteLogFun
 功能描述  : 执行日志操作
 输入参数  : E_CommonLogLevel level      --- 日志级别
             const char * p_log_info     --- 日志信息
             const char * p_file = NULL  --- 文件名
             int line = 0                --- 行号
 输出参数  : 无
 返 回 值  : void
 调用函数  : 
 被调函数  : 
 
 修改历史      :
  1.日    期   : 2017年9月13日
    作    者   : suyongsheng
    修改内容   : 新生成函数

*****************************************************************************/
void ImgAlgExcuteLogFun(E_CommonLogLevel level, const char * p_log_info, const char * p_file, int line);


/*****************************************************************************
 函 数 名  : ImgAlgRegisterLogSystemCallBack
 功能描述  : 注册回调函数
 输入参数  : PFUN_LogSystemCallBack pLogFun  
 输出参数  : 无
 返 回 值  : void
 调用函数  : 
 被调函数  : 
 
 修改历史      :
  1.日    期   : 2017年9月13日
    作    者   : suyongsheng
    修改内容   : 新生成函数

*****************************************************************************/
MOT_API void ImgAlgRegisterLogSystemCallBack(PFUN_LogSystemCallBack pLogFun);



#ifndef _LOG_SYSTEM_INTERFACE_MARCO_
#define _LOG_SYSTEM_INTERFACE_MARCO_

/* 日志调用接口的宏定义 */
#define LogFatal(str)  ImgAlgExcuteLogFun(E_CommonLog_Fatal, str, __FILE__, __LINE__)
#define LogError(str)  ImgAlgExcuteLogFun(E_CommonLog_Error, str, __FILE__, __LINE__)
#define LogWarn(str)   ImgAlgExcuteLogFun(E_CommonLog_Warn,  str, __FILE__, __LINE__)
#define LogInfo(str)   ImgAlgExcuteLogFun(E_CommonLog_Info,  str, __FILE__, __LINE__)
#define LogDebug(str)  ImgAlgExcuteLogFun(E_CommonLog_Debug, str, __FILE__, __LINE__)
#define LogTrace(str)  ImgAlgExcuteLogFun(E_CommonLog_Trace, str, __FILE__, __LINE__)

#endif


