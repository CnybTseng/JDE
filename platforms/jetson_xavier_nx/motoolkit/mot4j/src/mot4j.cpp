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

#include <map>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <json/json.h>

#include "mot.h"
#include "config.h"
#include "com_sihan_system_jni_utils_mot4j.h"

#define MAX_TRACK_LEN 1000

static std::mutex mtx;
static std::map<std::thread::id, mot::MOT_Result> motres;
static std::map<std::thread::id, unsigned long long> frame_counter;

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
JNIEXPORT jint JNICALL Java_com_sihan_system_jni_utils_mot4j_load_1mot_1model
  (JNIEnv *env, jobject obj, jstring cfg_path)
{
    std::thread::id tid = std::this_thread::get_id();
    frame_counter[tid] = 0;
    std::string path = env->GetStringUTFChars(cfg_path, 0);
    return mot::load_mot_model(path.c_str());
}

//*********************************************************************
// 卸载多目标跟踪模型
//*********************************************************************
JNIEXPORT jint JNICALL Java_com_sihan_system_jni_utils_mot4j_unload_1mot_1model
  (JNIEnv *env, jobject obj)
{
    return mot::unload_mot_model();
}

//*********************************************************************
// 执行多目标跟踪
//*********************************************************************
JNIEXPORT jstring JNICALL Java_com_sihan_system_jni_utils_mot4j_forward_1mot_1model
  (JNIEnv *env, jobject obj, jbyteArray data, jint width, jint height, jint stride)
{
    Json::Value result;
    jbyte *jbgr = env->GetByteArrayElements(data, NULL);
    if (NULL == jbgr)
    {
        std::cout << "invalid image data pointer" << std::endl;
        return charTojstring(env, "");
    }
    
    std::thread::id tid = std::this_thread::get_id();
    mot::forward_mot_model((unsigned char *)jbgr, (int)width, (int)height, (int)stride, motres[tid]);  
    env->ReleaseByteArrayElements(data, jbgr, 0);
    
    int i = -1;
    std::vector<mot::MOT_Track>::iterator riter;
    for (riter = motres[tid].begin(); riter != motres[tid].end(); riter++)
    {    
        int j = 0;
        std::deque<mot::MOT_Rect>::iterator iter;
        for (iter = riter->rects.begin(); iter != riter->rects.end(); iter++)
        {
            int l = static_cast<int>(round(iter->left));
            int t = static_cast<int>(round(iter->top));
            int w = static_cast<int>(round(iter->right - iter->left));
            int h = static_cast<int>(round(iter->bottom - iter->top));
            if (0 == j)
            {
                if (0 == l && 0 == t && 0 == w && 0 == h) // reject history track
                    break;
                else
                {
                    ++i;
                    Json::Value rects;
                    result[i]["identifier"] = std::to_string(riter->identifier);
                    result[i]["category"] = "person";
                    result[i]["rects"] = rects;
                }
            }
            
            result[i]["rects"][j]["x"] = std::to_string(l);
            result[i]["rects"][j]["y"] = std::to_string(t);
            result[i]["rects"][j]["width"] = std::to_string(w);
            result[i]["rects"][j]["height"] = std::to_string(h);
            ++j;
            if (j > MAX_TRACK_LEN)
                break;
        }
    }
    
    frame_counter[tid]++; // accumulate frame number
    return charTojstring(env, result.toStyledString().c_str());
}

//*********************************************************************
// 定时获取所有轨迹
//*********************************************************************
JNIEXPORT jstring JNICALL Java_com_sihan_system_jni_utils_mot4j_get_1total_1tracks
  (JNIEnv *env, jobject obj, jint reset)
{
    Json::Value result;
    std::thread::id tid = std::this_thread::get_id();
    if (frame_counter[tid] >= mot::trajectory_len)
    {
        // get all tracks
        int i = 0;
        std::vector<mot::MOT_Track>::iterator riter;
        for (riter = motres[tid].begin(); riter != motres[tid].end(); riter++, ++i)
        {
            Json::Value rects;
            result[i]["identifier"] = std::to_string(riter->identifier);
            result[i]["category"] = "person";
            result[i]["rects"] = rects;
        
            int j = 0;
            std::deque<mot::MOT_Rect>::iterator iter;
            for (iter = riter->rects.begin(); iter != riter->rects.end(); iter++)
            {
                int l = static_cast<int>(round(iter->left));
                int t = static_cast<int>(round(iter->top));
                int w = static_cast<int>(round(iter->right - iter->left));
                int h = static_cast<int>(round(iter->bottom - iter->top));                
                result[i]["rects"][j]["x"] = std::to_string(l);
                result[i]["rects"][j]["y"] = std::to_string(t);
                result[i]["rects"][j]["width"] = std::to_string(w);
                result[i]["rects"][j]["height"] = std::to_string(h);
                ++j;
            }
        }
        
        // reset track pool
        if (1 == reset)
        {
            std::vector<mot::MOT_Track>::iterator riter;
            for (riter = motres[tid].begin(); riter != motres[tid].end();)
            {
                bool history = false;
                std::deque<mot::MOT_Rect>::iterator iter;
                for (iter = riter->rects.begin(); iter != riter->rects.end(); iter++)
                {
                    int l = static_cast<int>(round(iter->left));
                    int t = static_cast<int>(round(iter->top));
                    int w = static_cast<int>(round(iter->right - iter->left));
                    int h = static_cast<int>(round(iter->bottom - iter->top));
                    if (iter == riter->rects.begin() &&
                        0 == l && 0 == t && 0 == w && 0 == h)
                    {
                        history = true;
                        break;
                    }
                }
                if (history)
                    riter = motres[tid].erase(riter);
                else
                    ++riter;
            }
            frame_counter[tid] = 0; // reset frame frame_counter
        }
    }
    
    return charTojstring(env, result.toStyledString().c_str());
}