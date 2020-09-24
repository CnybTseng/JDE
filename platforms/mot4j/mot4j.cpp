/******************************************************************************

                  版权所有 (C), 2004-2020, 成都思晗科技股份有限公司

 ******************************************************************************
  文 件 名   : mot4j.cpp
  版 本 号   : 初稿
  作    者   : Zeng Zhiwei
  生成日期   : 2020年9月24日
  最近修改   :
  功能描述   : 多目标跟踪的JNI接口实现
  
  修改历史   :
  1.日    期   : 2020年9月24日
    作    者   : Zeng Zhiwei
    修改内容   : 创建文件

******************************************************************************/

#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <json/json.h>

#include "mot.h"
#include "mot4j.h"

static mot::MOT_Result motres;

// This function is coming from kinson
jstring charTojstring( JNIEnv* env,const char* str )
{
	jclass strClass = env->FindClass( "Ljava/lang/String;"); 
	jmethodID ctorID = env->GetMethodID( strClass, "<init>", 
		"([BLjava/lang/String;)V"); 

	if (env->ExceptionCheck() == JNI_TRUE || str == NULL)
	{
		env->ExceptionDescribe();
		env->ExceptionClear();
		printf("nativeTojstring函数转换时,str为空/n");
		return NULL;
	} 

	jbyteArray bytes = env->NewByteArray( strlen(str)); 
	//如果str为空则抛出异常给jvm

	env->SetByteArrayRegion( bytes, 0,  strlen(str), (jbyte*)str); 
	//jstring encoding = env->NewStringUTF( "GBK"); 
	jstring encoding = env->NewStringUTF( "UTF8"); 
	jstring strRtn = (jstring)env->NewObject( strClass, ctorID, bytes, 
		encoding);
	//释放str内存
	// free(str);
	return strRtn;
}

//*********************************************************************
// 加载多目标跟踪模型
//*********************************************************************
JNIEXPORT jint JNICALL Java_mot4j_load_1mot_1model
  (JNIEnv *env, jobject obj, jstring cfg_path)
{
    std::string path = env->GetStringUTFChars(cfg_path, 0);
    return mot::load_mot_model(path.c_str());
}

//*********************************************************************
// 卸载多目标跟踪模型
//*********************************************************************
JNIEXPORT jint JNICALL Java_mot4j_unload_1mot_1model
  (JNIEnv *env, jobject obj)
{
    return mot::unload_mot_model();
}

//*********************************************************************
// 执行多目标跟踪
//*********************************************************************
JNIEXPORT jstring JNICALL Java_mot4j_forward_1mot_1model
  (JNIEnv *env, jobject obj, jbyteArray rgb, jint width, jint height, jint stride)
{
    Json::Value result;
    jbyte *jrgb = env->GetByteArrayElements(rgb, NULL);
    if (NULL == jrgb)
    {
        std::cout << "invalid rgb data pointer" << std::endl;
        return charTojstring(env, "");
    }
   
    mot::forward_mot_model((unsigned char *)jrgb, (int)width, (int)height, (int)stride, motres);    
    env->ReleaseByteArrayElements(rgb, jrgb, 0);
    
    int i = 0;
    std::vector<mot::MOT_Track>::iterator riter;
    for (riter = motres.begin(); riter != motres.end(); riter++, ++i)
    {
        Json::Value rects;
        result[i]["identifier"] = std::to_string(riter->identifier);
        result[i]["category"] = "person";
        result[i]["rects"] = rects;

        int j = 0;
        std::deque<mot::MOT_Rect>::iterator iter;
        for (iter = riter->rects.begin(); iter != riter->rects.end(); iter++, ++j)
        {
            result[i]["rects"][j]["top"] = std::to_string(iter->top);
            result[i]["rects"][j]["left"] = std::to_string(iter->left);
            result[i]["rects"][j]["bottom"] = std::to_string(iter->bottom);
            result[i]["rects"][j]["right"] = std::to_string(iter->right);
        }
    }
    
    return charTojstring(env, result.toStyledString().c_str());
}