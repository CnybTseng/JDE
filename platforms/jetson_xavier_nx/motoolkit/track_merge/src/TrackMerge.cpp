#include <map>
#include <ostream>
#include <iostream>
#include <json/json.h>
#include <opencv2/opencv.hpp>

#include "com_sihan_system_jni_utils_TrackMerge.h"
#include "lapjv.h"
#include "homography.h"

static std::map<std::string, cv::Mat> homography = {
    {"20-21", cv::Mat(3, 3, CV_32F, (void*)h20_21)},
    {"21-53", cv::Mat(3, 3, CV_32F, (void*)h21_53)},
    {"53-52", cv::Mat(3, 3, CV_32F, (void*)h53_52)}
};

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

static void LinearAssignment(const cv::Mat &cost, float cost_limit,
    std::map<int, int> &matches, std::vector<int> &mismatch_row,
    std::vector<int> &mismatch_col)
{
    matches.clear();
    mismatch_row.clear();    
    mismatch_col.clear();
    
    if (cost.empty())
    {
        for (int i = 0; i < cost.rows; ++i)
            mismatch_row.push_back(i);
        for (int i = 0; i < cost.cols; ++i)
            mismatch_col.push_back(i);
        return;
    }
    
    float opt;
    cv::Mat x(cost.rows, 1, CV_32S);
    cv::Mat y(cost.cols, 1, CV_32S);
    
    bool ret = mot::LAPJV::instance()->solve((float *)cost.data, cost.rows, cost.cols, &opt,
        (int *)x.data, (int *)y.data, true, cost_limit);
    if (!ret)
        return;
    
    for (int i = 0; i < x.rows; ++i)
    {
        int j = *x.ptr<int>(i);
        if (j >= 0)
            matches.insert({i, j});
        else
            mismatch_row.push_back(i);
    }
    
    for (int j = 0; j < y.rows; ++j)
    {
        int i = *y.ptr<int>(j);
        if (i < 0)
            mismatch_col.push_back(j);
    }
}

static bool GetLatestLocationOfTrack(JNIEnv *env, jstring tracks,
    Json::Value &jtracks, std::vector<cv::Mat> &locs)
{
    const int j = 0;
    Json::Reader reader;
    std::string stracks = env->GetStringUTFChars(tracks, 0);
    if (!reader.parse(stracks.c_str(), jtracks)) {
        std::cout << "parse json string fail\n" << std::endl;
        return false;
    }

    for (int i = 0; i < jtracks.size(); ++i) {
        cv::Mat pt = cv::Mat::ones(3, 1, CV_32F);
        int x = std::stoi(jtracks[i]["rects"][j]["x"].asString(), nullptr);
        int y = std::stoi(jtracks[i]["rects"][j]["y"].asString(), nullptr);
        int w = std::stoi(jtracks[i]["rects"][j]["width"].asString(), nullptr);
        int h = std::stoi(jtracks[i]["rects"][j]["height"].asString(), nullptr);
        *pt.ptr<float>(0) = x + w * 0.5f;
        *pt.ptr<float>(1) = y + h;
        locs.push_back(pt);
    }
    
    return true;
}

JNIEXPORT jstring JNICALL Java_com_sihan_system_jni_utils_TrackMerge_get_1registered_1channels
  (JNIEnv *env, jobject obj)
{
    Json::Value keys;
    int i = 0;
    for (auto &iter : homography) {
        keys[i++] = iter.first;
    }
    return charTojstring(env, keys.toStyledString().c_str());
}

JNIEXPORT jstring JNICALL Java_com_sihan_system_jni_utils_TrackMerge_merge_1track
  (JNIEnv *env, jobject obj, jstring tracks1, jstring tracks2, jint channel1, jint channel2)
{
    Json::Value merge_pairs;
    std::ostringstream oss;
    oss << channel1 << "-" << channel2;
    const std::string &key = oss.str();
    
    // Requested channels have not been registered yet.
    if (homography.end() == homography.find(key)) {
        std::cout << "channels have not been registered yet" << std::endl;
        return charTojstring(env, merge_pairs.toStyledString().c_str());
    }
    
    // Look up homography matrix.
    cv::Mat &h = homography[key];

    // Construct latest locations array.
    std::vector<cv::Mat> loc1, loc2;
    Json::Value jtracks1, jtracks2;
    if (!GetLatestLocationOfTrack(env, tracks1, jtracks1, loc1) ||
        !GetLatestLocationOfTrack(env, tracks2, jtracks2, loc2)) {
        return charTojstring(env, merge_pairs.toStyledString().c_str());
    }

    if (0 == loc1.size() || 0 == loc2.size()) {
        std::cout << "empty track!" << std::endl;
        return charTojstring(env, merge_pairs.toStyledString().c_str());
    }
    
    // Calculate cost matrix.
    std::vector<cv::Mat> loc1_warp(loc1.size());
    cv::Mat cost(loc1.size(), loc2.size(), CV_32F);
    for (int k = 0; k < loc1.size(); ++k) {
        // Warp location of the first channel to the second channel.
        cv::Mat &loc = loc1_warp[k];
        loc = h * loc1[k];
        *loc.ptr<float>(0) /= *loc.ptr<float>(2);
        *loc.ptr<float>(1) /= *loc.ptr<float>(2);
        // Norm-L2 distance.
        for (int l = 0; l < loc2.size(); ++l) {
            *cost.ptr<float>(k, l) = static_cast<float>(
                cv::norm(loc(cv::Rect(0, 0, 1, 2)), loc2[l](cv::Rect(0, 0, 1, 2))));
        }
    }

    // Linear assignment.
    const float cost_limit = 50;
    std::map<int, int> matches;
    std::vector<int> mismatch_row;
    std::vector<int> mismatch_col;
    LinearAssignment(cost, cost_limit, matches, mismatch_row, mismatch_col);
    
    // Construct merged pair.
    int cnt = 0;
    std::map<int, int>::iterator iter;
    for (iter = matches.begin(); iter != matches.end(); iter++) {
        merge_pairs[cnt]["track1"] = jtracks1[iter->first]["identifier"];
        merge_pairs[cnt]["track2"] = jtracks2[iter->second]["identifier"];
        cnt++;
    }
#if 0
    // Debug.
    FILE *fp = fopen("warp.txt", "w");
    for (int k = 0; k < loc1_warp.size(); ++k) {
        cv::Mat &loc = loc1_warp[k];
        fprintf(fp, "%f %f\n", *loc.ptr<float>(0), *loc.ptr<float>(1));
    }
    fclose(fp);
    std::cout << "cost=\n" << cost << std::endl;
    for (iter = matches.begin(); iter != matches.end(); iter++) {
        std::cout << "match: " << iter->first << "," << iter->second << std::endl;
    }
    for (int i = 0; i < mismatch_row.size(); ++i) {
        std::cout << "row mismatch: " << mismatch_row[i] << std::endl;
    }    
    for (int i = 0; i < mismatch_col.size(); ++i) {
        std::cout << "col mismatch: " << mismatch_col[i] << std::endl;
    }
#endif
    return charTojstring(env, merge_pairs.toStyledString().c_str());
}