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
  
  2.日    期   : 2020年10月15日
    作    者   : Zeng Zhiwei
    修改内容   : <1> forward_mot_model()输入图像的格式由RGB888修改为BGR888.
                 <2> forward_mot_model()返回的跟踪结果种的边框参数类型由double修改为int.

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
  (JNIEnv *env, jobject obj, jbyteArray data, jint width, jint height, jint stride)
{
    Json::Value result;
    jbyte *jbgr = env->GetByteArrayElements(data, NULL);
    if (NULL == jbgr)
    {
        std::cout << "invalid image data pointer" << std::endl;
        return charTojstring(env, "");
    }
   
    mot::forward_mot_model((unsigned char *)jbgr, (int)width, (int)height, (int)stride, motres);    
    env->ReleaseByteArrayElements(data, jbgr, 0);
    
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
            result[i]["rects"][j]["x"] = std::to_string(static_cast<int>(iter->left));
            result[i]["rects"][j]["y"] = std::to_string(static_cast<int>(iter->top));
            result[i]["rects"][j]["width"] = std::to_string(static_cast<int>(iter->right - iter->left));
            result[i]["rects"][j]["height"] = std::to_string(static_cast<int>(iter->bottom - iter->top));
        }
    }
    
    return charTojstring(env, result.toStyledString().c_str());
}