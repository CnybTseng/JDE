
#include "SH_ImageAlgLogSystem.h"


static PFUN_LogSystemCallBack pLogSystemCallBack  = NULL;


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
void ImgAlgRegisterLogSystemCallBack(PFUN_LogSystemCallBack pLogFun)
{
    pLogSystemCallBack = pLogFun;
}

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
void ImgAlgExcuteLogFun(E_CommonLogLevel level, const char * p_log_info, const char * p_file, int line)
{
    if(NULL != pLogSystemCallBack)
    {
        pLogSystemCallBack(level, p_log_info, p_file, line);
    }
}
