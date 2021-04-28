#include <map>
#include <iostream>
#include <json/json.h>
#include "com_sihan_system_jni_utils_CrossLineDetector.h"

typedef int camera;
typedef int triplet[6]; // triplet: [x1,y1;x2,y2;x3,y3]
static std::map<camera, triplet> triplets; // triplet for each camera
static long long total = 0; // total targets

static jstring charTojstring( JNIEnv* env,const char* str )
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

/**
 * @brief 获取三点的位置关系.
 * @param t 三点坐标, t=(x1,y1,x2,y2,x3,y3). (x1,y1), (x2,y2), 和(x3,y3)分别为三个点.
 * @return 如果三点共线, 返回0.
 *         否则, 返回-1或+1.
 */
static int directriplet(triplet t)
{
    int d = (t[4] - t[0]) * (t[3] - t[1]) - (t[5] - t[1]) * (t[2] - t[0]);
    // Only need sign.
    if (d < 0) {
        d = -1;
    } else if (d > 0) {
        d = 1;
    }
    return d;
}

/**
 * @brief 从当前点构建triplet.
 * @param x 目标坐上角x坐标.
 * @param y 目标坐上角y坐标.
 * @param w 目标宽度.
 * @param h 目标高度.
 * @param c 摄像机通道号.
 * @param t 目标对应的triplet.
 */
static inline void setriplet(int x, int y, int w, int h, int c, triplet t)
{
    t[0] = triplets[c][0];
    t[1] = triplets[c][1];
    t[2] = triplets[c][2];
    t[3] = triplets[c][3];
    t[4] = (int)(x + w / 2.f + 0.5f);
    t[5] = y + h;
}

//*********************************************************************
// 设置跨线检测的直线.
//*********************************************************************
jboolean JNICALL Java_com_sihan_system_jni_utils_CrossLineDetector_set_1line
  (JNIEnv *env, jobject obj, jint channel, jint x1, jint y1, jint x2, jint y2, jint x3, jint y3)
{
    if (triplets.find(channel) != triplets.end()) {
        std::cerr << "set line repetitively for channel " << channel << "\n";
        return false;
    }
    
    triplets[channel][0] = x1;
    triplets[channel][1] = y1;
    triplets[channel][2] = x2;
    triplets[channel][3] = y2;
    triplets[channel][4] = x3;
    triplets[channel][5] = y3;
    if (0 == directriplet(triplets[channel])) {
        std::cerr << "colinear triplets are not accepted\n";
        return false;
    }
    
    return true;
}

//*********************************************************************
// 检测跨线行为.
//*********************************************************************
jstring JNICALL Java_com_sihan_system_jni_utils_CrossLineDetector_detect_1cross_1event
  (JNIEnv *env, jobject obj, jstring tracks, jint recall)
{
    Json::Value event;  // cross line event
    Json::Value jtracks;
    std::string stracks = env->GetStringUTFChars(tracks, 0);
    JSONCPP_STRING err;
    Json::CharReaderBuilder builder;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(stracks.c_str(), stracks.c_str() + stracks.length(), &jtracks, &err)) {
        std::cerr << "parse json string fail\n" << std::endl;
        return charTojstring(env, event.toStyledString().c_str());
    }
    
    const jint minrel = 2;  // minimum recall length
    if (recall < minrel) {
        std::cerr << "reset recall as " << minrel << "\n";
        recall = minrel;
    }

    int entry = 0;  // number of entry targets
    int exit = 0;   // number of exit targets
    for (int i = 0; i < jtracks.size(); ++i) {
        triplet now;
        triplet past;
        int recalli = 0;
        int c = 0;
        for (int j = 0; j < jtracks[i]["rects"].size(); ++j) {
            int x = std::stoi(jtracks[i]["rects"][j]["x"].asString(), nullptr);
            int y = std::stoi(jtracks[i]["rects"][j]["y"].asString(), nullptr);
            int w = std::stoi(jtracks[i]["rects"][j]["width"].asString(), nullptr);
            int h = std::stoi(jtracks[i]["rects"][j]["height"].asString(), nullptr);
            c = std::stoi(jtracks[i]["rects"][j]["channel"].asString(), nullptr);
            if (triplets.find(c) == triplets.end()) {
                std::cerr << "tracks from unknown camera\n";
                return charTojstring(env, event.toStyledString().c_str()); 
            }
            if (x > 0 || y > 0 || w > 0 || h > 0) {
                if (0 == recalli) {
                    setriplet(x, y, w, h, c, now);
                }
                if (recall - 1 == recalli || j == jtracks[i]["rects"].size() - 1) {
                    setriplet(x, y, w, h, c, past);
                    break;
                }
                ++recalli;
            } else {
                if (0 == j) {
                    break;
                }
            }
        }
        if (recalli > 0) {
            int d0 = directriplet(triplets[c]);
            int d1 = directriplet(now);
            int d2 = directriplet(past);
            // std::cout << d0 << ", " << d1 << ", " << d2 << "\n";
            if (d1 * d2 < 0) {
                if (d0 * d1 < 0) {
                    ++exit;
                    --total;
                }
                if (d0 * d1 > 0) {
                    ++entry;
                    ++total;
                }
            }
        }
    }
    
    event["total_targets"] = std::to_string(total);
    event["entry"] = std::to_string(entry);
    event["exit"]  = std::to_string(exit);
    return charTojstring(env, event.toStyledString().c_str());
}